#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import base64
from typing import Dict, List, Tuple

import numpy as np
import cv2
import requests
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import tf2_ros
from rclpy.time import Time
from rclpy.duration import Duration
import tf2_geometry_msgs  # noqa: F401  (required to register transform for PointStamped)

# Optional organized cloud helper
try:
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None


# --------------------- Canonical labels for the TB4 warehouse world ---------------------
CANON: Dict[str, List[str]] = {
    "person": ["person", "human", "worker", "man", "woman", "people"],
    "chair": ["chair", "stool", "seat", "office chair"],
    "table": ["table", "workbench", "bench table", "desk"],
    "shelf": ["shelf", "shelving", "shelving unit", "rack", "warehouse rack",
              "storage rack", "industrial rack", "racking system", "shelving rack"],
    "pallet stack": ["pallet stack", "stack", "stack of boxes", "stacked boxes",
                     "pile of boxes", "pallet", "palletized load", "palletized boxes"],
    "box": ["box", "cardboard box", "carton", "crate", "package", "parcel"],
    "trash can": ["trash can", "bin", "garbage can", "waste bin", "rubbish bin"],
    "column": ["column", "pillar", "post", "support column", "support beam"],
    "barrier": ["barrier", "concrete barrier", "k-rail", "divider", "block"],
    "door": ["door", "doorway", "entry", "opening"],
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
    """
    Simple client for your Qwen3-VL server that returns strict JSON with pixel bboxes.
    """

    def __init__(
        self,
        url: str = "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions",
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        timeout: float = 20.0,
    ):
        self.url = url
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _b64_from_bgr(bgr: np.ndarray, jpeg_quality: int = 85) -> str:
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(enc.tobytes()).decode("utf-8")

    def _build_prompt(self, W: int, H: int, categories: List[str]) -> str:
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
        content = [
            {"type": "text", "text": self._build_prompt(W, H, categories)},
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
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()

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
        self.declare_parameter("pointcloud", "/oakd/rgb/preview/depth/points")
        self.declare_parameter("camera_frame", "oakd_rgb_camera_optical_frame")

        # Depth units (OAK-D depth is typically 16UC1 in millimeters)
        self.declare_parameter("depth_scale", 0.001)  # mm -> meters

        # --- VLM detection ---
        self.declare_parameter("detect_rate_hz", 1.0)
        self.declare_parameter("categories", list(CANON.keys()))
        self.declare_parameter("vlm_url", "http://saltyfish.eecs.umich.edu:8000/v1/chat/completions")
        self.declare_parameter("vlm_model", "Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.declare_parameter("send_depth_to_vlm", True)  # include depth colormap as second image

        # --- Tracking / promotion ---
        self.declare_parameter("gate_radius", 0.7)       # m, assoc gate
        self.declare_parameter("promote_min_obs", 4)     # min observations to anchor
        self.declare_parameter("promote_max_std", 0.35)  # m, XY std threshold

        # --- Output ---
        self.declare_parameter("save_json", True)
        self.declare_parameter("json_path", "/tmp/objects.json")

        # Read params
        gp = self.get_parameter
        self.rgb_topic = gp("camera_rgb").value
        self.info_topic = gp("camera_info").value
        self.depth_topic = gp("camera_depth").value
        self.pc_topic = gp("pointcloud").value
        self.cam_frame = gp("camera_frame").value
        self.depth_scale = float(gp("depth_scale").value)

        self.detect_dt = 1.0 / float(gp("detect_rate_hz").value)
        self.categories = list(gp("categories").value)
        self.send_depth = bool(gp("send_depth_to_vlm").value)

        self.detector = VLMDetector(
            url=str(gp("vlm_url").value), model=str(gp("vlm_model").value), timeout=20.0
        )

        self.gate_r = float(gp("gate_radius").value)
        self.promote_n = int(gp("promote_min_obs").value)
        self.promote_std = float(gp("promote_max_std").value)
        self.save_json = bool(gp("save_json").value)
        self.json_path = str(gp("json_path").value)

        # IO
        self.bridge = CvBridge()
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)
        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, 10)
        self.sub_pc = self.create_subscription(PointCloud2, self.pc_topic, self.cb_pc, 10)

        self.pub_mark = self.create_publisher(MarkerArray, "/objects/markers", 10)
        self.srv_list = self.create_service(Trigger, "list_anchors", self.handle_list)

        # TF
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf, self)

        # State
        self.K: np.ndarray = None         # camera intrinsics (3x3)
        self.depth: np.ndarray = None     # meters
        self.pc_msg: PointCloud2 = None   # organized cloud (optional)
        self._last_t = 0.0                # detector throttle
        self.store: Dict[str, List[dict]] = {}  # label -> tracks

    # ---------------- Callbacks ----------------
    def cb_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    def cb_depth(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.dtype == np.uint16:
            self.depth = img.astype(np.float32) * self.depth_scale
        else:
            self.depth = img.astype(np.float32)

    def cb_pc(self, msg: PointCloud2):
        self.pc_msg = msg

    def cb_rgb(self, msg: Image):
        now = time.time()
        if now - self._last_t < self.detect_dt:
            return
        self._last_t = now

        if self.K is None or self.depth is None:
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        extra_imgs = [make_depth_colormap(self.depth)] if self.send_depth and self.depth is not None else None

        # Ask VLM
        try:
            dets = self.detector.detect(bgr, self.categories, extra_images_bgr=extra_imgs)
        except Exception as e:
            self.get_logger().warn(f"VLM detect failed: {e}")
            return

        H, W = bgr.shape[:2]
        for d in dets:
            u1, v1, u2, v2 = d.bbox
            # robust center patch
            u = int(0.5 * (u1 + u2))
            v = int(0.5 * (v1 + v2))
            du = max(2, (u2 - u1) // 4)
            dv = max(2, (v2 - v1) // 4)
            umin, umax = max(0, u - du), min(W, u + du)
            vmin, vmax = max(0, v - dv), min(H, v + dv)

            # 1) aligned depth median
            patch = self.depth[vmin:vmax, umin:umax]
            valid = patch[np.isfinite(patch) & (patch > 0.1)]
            Zs = valid.flatten()

            # 2) optional organized cloud sampling for extra robustness
            if pc2 is not None and isinstance(self.pc_msg, PointCloud2):
                try:
                    if self.pc_msg.width > 0 and self.pc_msg.height > 0:
                        step_u = max(1, (umax - umin) // 8)
                        step_v = max(1, (vmax - vmin) // 8)
                        cloud_Zs = []
                        for vv in range(vmin, vmax, step_v):
                            for uu in range(umin, umax, step_u):
                                for p in pc2.read_points(
                                    self.pc_msg, field_names=("x", "y", "z"),
                                    skip_nans=True, uvs=[(uu, vv)]
                                ):
                                    if np.isfinite(p[2]) and p[2] > 0.1:
                                        cloud_Zs.append(p[2])
                        if len(cloud_Zs) >= 5:
                            Zs = np.concatenate([Zs, np.array(cloud_Zs, dtype=np.float32)])
                except Exception:
                    pass

            if Zs.size < 5:
                continue

            Z = float(np.median(Zs))
            fx, fy, px, py = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            Xc = (u - px) * Z / fx
            Yc = (v - py) * Z / fy

            pt = PointStamped()
            pt.header.stamp = msg.header.stamp
            pt.header.frame_id = self.cam_frame
            pt.point.x, pt.point.y, pt.point.z = float(Xc), float(Yc), float(Z)

            try:
                # tf2: transform to map frame
                trans = self.tfbuf.lookup_transform("map", self.cam_frame, Time())  # you can add timeout=Duration(seconds=0.15) if needed
                tf_pt = tf2_geometry_msgs.do_transform_point(pt, trans)
                xyz = np.array([tf_pt.point.x, tf_pt.point.y, tf_pt.point.z], dtype=np.float32)
            except Exception as e:
                self.get_logger().warn(f"TF failed: {e}")
                continue

            self.update_tracks(canonize(d.label), xyz)

        self.publish_markers()
        if self.save_json:
            self.write_json()
        anch = sum(1 for L in self.store.values() for t in L if t["anchored"])
        pend = sum(1 for L in self.store.values() for t in L if not t["anchored"])
        self.get_logger().info(f"Anchors: {anch}  |  Pending tracks: {pend}  |  Labels: {list(self.store.keys())[:6]}")


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
            std = np.sqrt(np.maximum(t["M2"] / max(1, t["n"] - 1), 1e-9))
            if (not t["anchored"]) and (t["n"] >= self.promote_n) and (np.linalg.norm(std[:2]) <= self.promote_std):
                t["anchored"] = True
        else:
            tracks.append({
                "mean": xyz.copy(),
                "M2": np.zeros(3, np.float32),
                "n": 1,
                "anchored": False,
                "last": time.time(),
            })

    # ---------------- Outputs ----------------
    def publish_markers(self):
        arr = MarkerArray()
        mid = 0
        for label, tracks in self.store.items():
            for t in tracks:
                x, y, z = [float(v) for v in t["mean"]]
                if t["anchored"]:
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
                else:
                    sph = Marker()
                    sph.header.frame_id = "map"
                    sph.header.stamp = self.get_clock().now().to_msg()
                    sph.ns = f"pending/{label}"
                    sph.id = mid; mid += 1
                    sph.type = Marker.SPHERE
                    sph.action = Marker.ADD
                    sph.pose.position.x = x
                    sph.pose.position.y = y
                    sph.pose.position.z = max(0.05, z)
                    sph.scale.x = sph.scale.y = sph.scale.z = 0.22
                    sph.color.r, sph.color.g, sph.color.b, sph.color.a = 0.2, 0.2, 0.8, 0.6
                    arr.markers.append(sph)

        self.pub_mark.publish(arr)

    def write_json(self):
        out = []
        for label, tracks in self.store.items():
            for t in tracks:
                if not t["anchored"]:
                    continue
                out.append({
                    "label": label,
                    "xyz": [float(v) for v in t["mean"]],
                    "n": int(t["n"]),
                    "last_seen": float(t["last"]),
                })
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
