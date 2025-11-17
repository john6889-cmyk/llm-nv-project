#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import base64
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import cv2
import requests
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from PIL import Image as PILImage
import yaml

import tf2_ros
from rclpy.time import Time
import tf2_geometry_msgs  # noqa: F401  (required to register transform for PointStamped)

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------- Canonical labels for the TB4 warehouse world ---------------------
CANON: Dict[str, List[str]] = {
    "person": ["person", "human", "worker", "man", "woman", "people"],
    "chair": ["chair", "stool", "seat", "office chair"],
    "table": ["table", "workbench", "bench table", "desk"],
    "shelf": ["shelf", "shelving", "shelving unit", "rack", "warehouse rack",
              "storage rack", "industrial rack", "racking system", "shelving rack"],
    "stack": ["pallet stack", "stack", "stack of boxes", "stacked boxes",
                     "pile of boxes", "pallet", "palletized load", "palletized boxes"],
    "box": ["box", "cardboard box", "carton", "crate", "package", "parcel"]
    # "trash can": ["trash can", "bin", "garbage can", "waste bin", "rubbish bin"],
    # "column": ["column", "pillar", "post", "support column", "support beam"]
    # "barrier": ["barrier", "concrete barrier", "k-rail", "divider", "block"],
    # "door": ["door", "doorway", "entry", "opening"],
}


def canonize(label: str) -> str:
    lab = label.lower().strip()
    for k, vs in CANON.items():
        if lab == k or any(lab == v for v in vs):
            return k
    return lab


def make_depth_colormap(depth_m: np.ndarray) -> np.ndarray:
    """Convert depth (m) to a JET colormap for sending as a 2nd image to the VLM (optional)."""
    d = depth_m.copy()
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, 0.0, 6.0)
    d8 = (d / 6.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


# ------------------------------- VLM HTTP detector client --------------------------------
class Detection:
    def __init__(self, label: str, score: float, bbox: Tuple[int, int, int, int]):
        self.label = label
        self.score = score
        self.bbox = bbox  # (u1, v1, u2, v2)


class VLMDetector:

    def __init__(
        self,
        url: str = "http://ronaldo.eecs.umich.edu:11400/api/generate",
        model: str = "llava",
        timeout: float = 30.0,
        api_style: str = "ollama",
        log_raw: bool = False,
    ):
        self.url = url
        self.model = model
        self.timeout = timeout
        self.api_style = api_style.lower()
        if self.api_style not in ("ollama", "chat"):
            raise ValueError(f"Unsupported VLM API style '{api_style}'. Expected 'ollama' or 'chat'.")
        self.log_raw = log_raw

    @staticmethod
    def _b64_from_bgr(bgr: np.ndarray, jpeg_quality: int = 85) -> str:
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(enc.tobytes()).decode("utf-8")

    def _build_prompt(self, W: int, H: int, categories: List[str]) -> str:
        # print(categories)
        cats = ", ".join(sorted(set(c.lower() for c in categories)))
        return (
            "You are an object detector. You may receive one or more images of the SAME scene "
            "(RGB first, optional depth colormap after). Detect ONLY these categories: "
            f"[{cats}].\n"
            "Return STRICT JSON exactly as: "
            "{\"detections\":[{\"label\":\"<one of categories>\",\"score\":0..1,"
            "\"bbox\":[u1,v1,u2,v2]}...]}\n"
            f"Coordinates are integer pixels in the ORIGINAL RGB image size (width={W}, height={H}). "
            "Ensure 0<=u1<u2<width and 0<=v1<v2<height. If none, return {\"detections\":[]}."
        )

    def detect(
        self,
        bgr_rgb: np.ndarray,
        categories: List[str],
        extra_images_bgr: List[np.ndarray] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[Detection]:
        H, W = bgr_rgb.shape[:2]
        prompt = self._build_prompt(W, H, categories)

        if self.api_style == "ollama":
            images = [self._b64_from_bgr(bgr_rgb)]
            if extra_images_bgr:
                images.extend(self._b64_from_bgr(img) for img in extra_images_bgr)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": images,
                "stream": False,
                "options": {"temperature": temperature, "top_p": top_p},
            }
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{self._b64_from_bgr(bgr_rgb)}"}},
            ]
            if extra_images_bgr:
                for img in extra_images_bgr:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self._b64_from_bgr(img)}"}
                    })

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "top_p": top_p,
            }
        r = requests.post(self.url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        r.raise_for_status()
        resp_obj = r.json()
        if self.api_style == "ollama":
            reply = str(resp_obj.get("response", "")).strip()
        else:
            choices = resp_obj.get("choices", [])
            if not choices:
                raise RuntimeError(f"Unexpected chat response: {resp_obj}")
            reply = str(choices[0]["message"]["content"]).strip()
        if self.log_raw:
            print("[VLM raw response]", resp_obj)

        # Strip code fences if present
        if reply.startswith("```"):
            lines = reply.splitlines()
            if lines and lines[-1].strip().startswith("```"):
                reply = "\n".join(lines[1:-1]).strip()
            else:
                reply = "\n".join(lines[1:]).strip()

        obj = json.loads(reply)
        outs: List[Detection] = []
        for d in obj.get("detections", []):
            lab = str(d["label"]).lower().strip()
            u1, v1, u2, v2 = [int(round(x)) for x in d["bbox"]]
            u1 = max(0, min(u1, W - 1))
            u2 = max(0, min(u2, W - 1))
            v1 = max(0, min(v1, H - 1))
            v2 = max(0, min(v2, H - 1))
            if u2 <= u1 or v2 <= v1:
                continue
            outs.append(Detection(lab, float(d.get("score", 0.0)), (u1, v1, u2, v2)))
        return outs


# ---------------------------------- Main ROS2 anchoring node ----------------------------------
class ObjectAnchorVLM(Node):
    def __init__(self):
        super().__init__("object_anchor_vlm")

        # --- Topics / frames (OAK-D defaults) ---
        self.declare_parameter("camera_rgb", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("camera_info", "/oakd/rgb/preview/camera_info")
        self.declare_parameter("camera_depth", "/oakd/rgb/preview/depth")
        self.declare_parameter("camera_frame", "oakd_rgb_camera_optical_frame")
        self.declare_parameter("robot_pose_topic", "/map/pose")
        self.declare_parameter("map_yaml_path", "./maps/savedmap.yaml")
        self.declare_parameter("map_image_path", "./maps/savedmap.png")

        # Depth units (OAK-D depth is typically 16UC1 in millimeters)
        self.declare_parameter("depth_scale", 0.001)  # mm -> meters

        # --- VLM detection ---
        self.declare_parameter("detect_rate_hz", 1.0)
        self.declare_parameter("categories", list(CANON.keys()))
        self.declare_parameter("vlm_url", "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions")
        self.declare_parameter("vlm_model", "Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.declare_parameter("vlm_api_style", "chat")  # 'ollama' or 'chat'
        self.declare_parameter("send_depth_to_vlm", True)  # include depth colormap as second image
        self.declare_parameter("display_rgb", False)
        self.declare_parameter("display_depth", False)
        self.declare_parameter("log_raw_vlm", False)
        self.declare_parameter("detection_log_path", "/tmp/object_detections.jsonl")

        # --- Tracking ---
        self.declare_parameter("gate_radius", 0.7)       # m, assoc gate

        # --- Output ---
        self.declare_parameter("save_json", True)
        self.declare_parameter("json_path", str(SCRIPT_DIR / "obj_coordinates.json"))

        # Read params
        gp = self.get_parameter
        self.rgb_topic = gp("camera_rgb").value
        self.info_topic = gp("camera_info").value
        self.depth_topic = gp("camera_depth").value
        self.cam_frame = gp("camera_frame").value
        self.robot_pose_topic = gp("robot_pose_topic").value
        self.map_yaml_path = Path(str(gp("map_yaml_path").value))
        self.map_image_path = Path(str(gp("map_image_path").value))
        self.depth_scale = float(gp("depth_scale").value)

        self.detect_dt = 1.0 / float(gp("detect_rate_hz").value)
        self.categories = list(gp("categories").value)
        self.send_depth = bool(gp("send_depth_to_vlm").value)
        self.display_rgb = bool(gp("display_rgb").value)
        self.display_depth = bool(gp("display_depth").value)
        self.vlm_api_style = str(gp("vlm_api_style").value)
        self.log_raw_vlm = bool(gp("log_raw_vlm").value)
        self.detection_log_path = Path(str(gp("detection_log_path").value))

        self.detector = VLMDetector(
            url=str(gp("vlm_url").value),
            model=str(gp("vlm_model").value),
            timeout=60.0,
            api_style=self.vlm_api_style,
            log_raw=self.log_raw_vlm,
        )

        self.gate_r = float(gp("gate_radius").value)
        self.save_json = bool(gp("save_json").value)
        self.json_path = str(gp("json_path").value)

        # IO
        self.bridge = CvBridge()
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, 10)
        self.sub_pose = self.create_subscription(PoseStamped, self.robot_pose_topic, self.cb_robot_pose, 10)

        self.pub_mark = self.create_publisher(MarkerArray, "/objects/markers", 10)
        self.srv_list = self.create_service(Trigger, "list_anchors", self.handle_list)

        # TF
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf, self)

        # State
        self.K: np.ndarray = None         # camera intrinsics (3x3)
        self.fx = self.fy = self.px = self.py = None
        self.depth: np.ndarray = None     # meters
        self._last_t = 0.0                # detector throttle
        self.store: Dict[str, List[dict]] = {}  # label -> tracks
        self.robot_pose: PoseStamped = None
        self._canvas_window = "object_anchor_vlm/Composite"
        self._display_enabled = False
        self._display_error_logged = False
        self._latest_depth_vis = None
        self._latest_rgb = None
        self.map_resolution: Optional[float] = None
        self.map_width_px: Optional[int] = None
        self.map_height_px: Optional[int] = None
        self.map_half_width_m: Optional[float] = None
        self.map_half_height_m: Optional[float] = None
        self._load_map_metadata()
        self._init_display_windows()

    # ---------------- Callbacks ----------------
    def cb_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.px, self.py = self.K[0, 2], self.K[1, 2]

    def cb_depth(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.dtype == np.uint16:
            self.depth = img.astype(np.float32) * self.depth_scale
        else:
            self.depth = img.astype(np.float32)
        if self.display_depth and self.depth is not None:
            self._latest_depth_vis = make_depth_colormap(self.depth)
            self._render_display_canvas()

    def cb_robot_pose(self, msg: PoseStamped):
        self.robot_pose = msg

    def cb_rgb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._latest_rgb = bgr
        self._render_display_canvas()

        now = time.time()
        if now - self._last_t < self.detect_dt:
            return
        self._last_t = now

        if self.K is None or self.depth is None:
            return
        extra_imgs = [make_depth_colormap(self.depth)] if self.send_depth and self.depth is not None else None

        # Ask VLM
        try:
            dets = self.detector.detect(bgr, self.categories, extra_images_bgr=extra_imgs)
        except Exception as e:
            self.get_logger().warn(f"VLM detect failed: {e}")
            return

        H, W = bgr.shape[:2]
        detections_to_report = []
        for d in dets:
            u1, v1, u2, v2 = d.bbox
            # robust center patch
            u = int(0.5 * (u1 + u2))
            v = int(0.5 * (v1 + v2))
            du = max(2, (u2 - u1) // 4)
            dv = max(2, (v2 - v1) // 4)
            umin, umax = max(0, u - du), min(W, u + du)
            vmin, vmax = max(0, v - dv), min(H, v + dv)

            cam_xyz = self.estimate_camera_xyz(umin, umax, vmin, vmax)
            if cam_xyz is None:
                continue

            pt = PointStamped()
            pt.header.stamp = msg.header.stamp
            pt.header.frame_id = self.cam_frame
            pt.point.x, pt.point.y, pt.point.z = [float(v) for v in cam_xyz]

            try:
                # tf2: transform to map frame
                trans = self.tfbuf.lookup_transform("map", self.cam_frame, Time())  
                tf_pt = tf2_geometry_msgs.do_transform_point(pt, trans)
                xyz = np.array([tf_pt.point.x, tf_pt.point.y, tf_pt.point.z], dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"TF failed: {e}")
                continue

            label = canonize(d.label)
            self.update_tracks(label, xyz)
            detections_to_report.append((label, xyz, d.score))

        self.publish_markers()
        if self.save_json:
            self.write_json()
        total_tracks = sum(len(L) for L in self.store.values())
        self.get_logger().info(f"Tracks: {total_tracks}  |  Labels: {list(self.store.keys())[:6]}")
        pose_tuple = self._current_robot_pose()
        if pose_tuple:
            rx, ry, rz, yaw = pose_tuple
            self.get_logger().info(
                f"Robot pose (map frame): x={rx:.2f} m, y={ry:.2f} m, z={rz:.2f} m, yaw={math.degrees(yaw):.1f} deg"
            )
        if detections_to_report:
            self.append_detection_log(detections_to_report)
        for label, xyz, score in detections_to_report:
            dist = float(np.linalg.norm(xyz))
            self.get_logger().info(
                f"[{label}] map coords x={xyz[0]:.2f} m, y={xyz[1]:.2f} m, z={xyz[2]:.2f} m (range {dist:.2f} m, score {score:.2f})"
            )


    # ---------------- Tracking / Promotion ----------------
    def update_tracks(self, label: str, xyz: np.ndarray):
        tracks = self.store.setdefault(label, [])
        best_i, best_d = -1, 1e9
        for i, t in enumerate(tracks):
            d = np.linalg.norm(xyz - t["mean"])
            if d < best_d:
                best_d, best_i = d, i

        if best_d <= self.gate_r and best_i >= 0:
            t = tracks[best_i]
            t["n"] += 1
            delta = xyz - t["mean"]
            t["mean"] += delta / t["n"]
            t["M2"] += delta * (xyz - t["mean"])
            t["last"] = time.time()
        else:
            tracks.append({
                "mean": xyz.copy(),
                "M2": np.zeros(3, np.float32),
                "n": 1,
                "last": time.time(),
            })

    def _init_display_windows(self):
        self._display_enabled = self.display_rgb or self.display_depth
        if not self._display_enabled:
            return
        try:
            cv2.namedWindow(self._canvas_window, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            self.get_logger().warn(f"Failed to create display window: {e}")
            self.display_rgb = False
            self.display_depth = False
            self._display_enabled = False

    def _display_image(self, image: np.ndarray):
        if not self._display_enabled or image is None:
            return
        try:
            cv2.imshow(self._canvas_window, image)
            cv2.waitKey(1)
        except cv2.error as e:
            if not self._display_error_logged:
                self.get_logger().warn(f"Display failed: {e}")
                self._display_error_logged = True
            self._display_enabled = False
            self.display_rgb = False
            self.display_depth = False

    def _render_display_canvas(self):
        if not self._display_enabled:
            return
        tiles = []
        labels = []
        if self.display_rgb and self._latest_rgb is not None:
            tiles.append(self._latest_rgb)
            labels.append("RGB")
        if self.display_depth and self._latest_depth_vis is not None:
            tiles.append(self._latest_depth_vis)
            labels.append("Depth")

        if not tiles:
            return

        max_h = max(img.shape[0] for img in tiles)
        canvas = []
        for img, label in zip(tiles, labels):
            h, w = img.shape[:2]
            if h == 0:
                continue
            scale = max_h / h
            new_w = max(1, int(w * scale))
            resized = cv2.resize(img, (new_w, max_h))
            cv2.putText(resized, label, (10, max_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            canvas.append(resized)
        if not canvas:
            return
        composite = np.hstack(canvas)
        self._display_image(composite)

    def append_detection_log(self, detections: List[Tuple[str, np.ndarray, float]]):
        if not self.detection_log_path:
            return
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        pose = self._current_robot_pose()
        entries = []
        for label, xyz, score in detections:
            pix = self._map_to_lower_left_pixel(float(xyz[0]), float(xyz[1]))
            entry = {
                "timestamp": timestamp,
                "label": label,
                "score": float(score),
                "map_xyz": self._round_vals([xyz[0], xyz[1], xyz[2]]),
            }
            if pix:
                entry["pixel_lower_left"] = self._round_vals(pix)
            if pose:
                entry["robot_pose"] = {
                    "map_xyz": self._round_vals(pose[:3]),
                    "yaw_rad": round(float(pose[3]), 2),
                }
            entries.append(entry)
        try:
            self.detection_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.detection_log_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            self.get_logger().warn(f"Failed to append detection log: {exc}")

    def _load_map_metadata(self):
        try:
            if self.map_yaml_path.exists():
                with open(self.map_yaml_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                res = float(data.get("resolution", 0.05))
                image_rel = data.get("image")
                if image_rel:
                    candidate = (self.map_yaml_path.parent / image_rel).resolve()
                    if candidate.exists():
                        self.map_image_path = candidate
            else:
                res = 0.05
            img_path = self.map_image_path
            if not img_path.exists():
                raise FileNotFoundError(f"Map image {img_path} not found")
            img = PILImage.open(str(img_path))
            self.map_width_px, self.map_height_px = img.size
            self.map_resolution = res
            width_m = self.map_width_px * self.map_resolution
            height_m = self.map_height_px * self.map_resolution
            self.map_half_width_m = width_m / 2.0
            self.map_half_height_m = height_m / 2.0
            self.get_logger().info(
                f"Loaded map metadata: {self.map_width_px}x{self.map_height_px} px @ {self.map_resolution:.3f} m/px"
            )
        except Exception as exc:
            self.get_logger().warn(f"Failed to load map metadata: {exc}")
            self.map_resolution = None
            self.map_width_px = self.map_height_px = None
            self.map_half_width_m = self.map_half_height_m = None

    def estimate_camera_xyz(self, umin: int, umax: int, vmin: int, vmax: int) -> np.ndarray:
        pts = self._depth_patch_points(umin, umax, vmin, vmax)
        if pts is None or pts.shape[0] < 5:
            return None

        # Median is robust against outliers
        return np.median(pts, axis=0)

    def _depth_patch_points(self, umin: int, umax: int, vmin: int, vmax: int):
        if self.depth is None or self.fx is None:
            return None
        patch = self.depth[vmin:vmax, umin:umax]
        mask = np.isfinite(patch) & (patch > 0.1)
        if not np.any(mask):
            return None
        rows, cols = np.where(mask)
        zs = patch[mask]
        us = umin + cols
        vs = vmin + rows
        xs = (us - self.px) * zs / self.fx
        ys = (vs - self.py) * zs / self.fy
        pts = np.stack([xs, ys, zs], axis=1)
        if pts.shape[0] > 600:
            idx = np.linspace(0, pts.shape[0] - 1, 600).astype(np.int32)
            pts = pts[idx]
        return pts.astype(np.float32)

    def _current_robot_pose(self):
        if self.robot_pose is None:
            return None
        p = self.robot_pose.pose.position
        q = self.robot_pose.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (float(p.x), float(p.y), float(p.z), yaw)

    def _map_to_lower_left_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if None in (self.map_resolution, self.map_half_width_m, self.map_half_height_m):
            return None
        px = (x + self.map_half_width_m) / self.map_resolution
        py = (y + self.map_half_height_m) / self.map_resolution
        return float(px), float(py)

    @staticmethod
    def _round_vals(values):
        return [round(float(v), 2) for v in values]

    # ---------------- Outputs ----------------
    def publish_markers(self):
        arr = MarkerArray()
        mid = 0
        for label, tracks in self.store.items():
            for t in tracks:
                x, y, z = [float(v) for v in t["mean"]]
                cube = Marker()
                cube.header.frame_id = "map"
                cube.header.stamp = self.get_clock().now().to_msg()
                cube.ns = f"obj/{label}"
                cube.id = mid; mid += 1
                cube.type = Marker.CUBE
                cube.action = Marker.ADD
                cube.pose.position.x = x
                cube.pose.position.y = y
                cube.pose.position.z = max(0.05, z)
                cube.scale.x = cube.scale.y = cube.scale.z = 0.25
                cube.color.r, cube.color.g, cube.color.b, cube.color.a = 0.1, 0.8, 0.1, 0.9
                arr.markers.append(cube)

                txt = Marker()
                txt.header = cube.header
                txt.ns = cube.ns + "/text"
                txt.id = mid; mid += 1
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = x
                txt.pose.position.y = y
                txt.pose.position.z = max(0.35, z + 0.25)
                txt.scale.z = 0.25
                txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
                txt.text = f"{label}"
                arr.markers.append(txt)

        self.pub_mark.publish(arr)

    def write_json(self):
        out = []
        pose_snapshot = None
        if self.robot_pose is not None:
            p = self.robot_pose.pose.position
            q = self.robot_pose.pose.orientation
            pose_snapshot = {
                "frame": self.robot_pose.header.frame_id or "map",
                "position": self._round_vals([p.x, p.y, p.z]),
                "orientation": self._round_vals([q.x, q.y, q.z, q.w]),
            }
        for label, tracks in self.store.items():
            for t in tracks:
                map_xyz = self._round_vals(t["mean"])
                entry = {
                    "label": label,
                    "map_xyz": map_xyz,
                    "n": int(t["n"]),
                    "last_seen": float(t["last"]),
                }
                pix = self._map_to_lower_left_pixel(map_xyz[0], map_xyz[1])
                if pix:
                    entry["pixel_lower_left"] = self._round_vals(pix)
                if pose_snapshot:
                    entry["robot_pose"] = pose_snapshot
                out.append(entry)
        with open(self.json_path, "w") as f:
            json.dump(out, f, indent=2)

    def handle_list(self, req, res):
        self.write_json()
        with open(self.json_path, "r") as f:
            res.message = f.read()
        res.success = True
        return res


def main():
    rclpy.init()
    rclpy.spin(ObjectAnchorVLM())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
