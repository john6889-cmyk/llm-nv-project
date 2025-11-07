#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger
import numpy as np, cv2, json, time, os
from pathlib import Path

def grid_to_map_xy(px, py, h, res, ox, oy):
    # image coords (0,0) top-left -> map frame
    x = ox + px * res
    y = oy + (h - 1 - py) * res
    return float(x), float(y)

class RoomGraphBuilder(Node):
    def __init__(self):
        super().__init__('room_graph_builder')

        # -------- Tunables (meters) --------
        self.declare_parameter('room_seed_radius_m', 1.20)
        self.declare_parameter('corridor_max_width_m', 2.00)
        self.declare_parameter('door_max_width_m', 1.30)
        self.declare_parameter('min_room_area_m2', 4.00)

        # Cost thresholds working for /map (0/100/-1) and costmap (0..255)
        self.declare_parameter('free_threshold', 5)       # <= free on costmaps
        self.declare_parameter('occ_threshold', 250)      # >= obstacle on costmaps
        self.declare_parameter('occ_threshold_map', 50)   # >= obstacle on static maps

        # I/O
        self.declare_parameter('save_graph_json', True)
        self.declare_parameter('graph_json_path', '/tmp/room_graph.json')
        self.declare_parameter('save_viz', True)
        self.declare_parameter('viz_dir', '/tmp/room_graph_viz')

        # Run behavior
        self.declare_parameter('oneshot', False)
        self.declare_parameter('min_update_period', 1.0)  # seconds

        # Read params
        gp = self.get_parameter
        self.room_seed_radius_m  = float(gp('room_seed_radius_m').value)
        self.corridor_max_width_m= float(gp('corridor_max_width_m').value)
        self.door_max_width_m    = float(gp('door_max_width_m').value)
        self.min_room_area_m2    = float(gp('min_room_area_m2').value)
        self.free_threshold      = int(gp('free_threshold').value)
        self.occ_threshold       = int(gp('occ_threshold').value)
        self.occ_threshold_map   = int(gp('occ_threshold_map').value)
        self.save_graph_json     = bool(gp('save_graph_json').value)
        self.graph_json_path     = str(gp('graph_json_path').value)
        self.save_viz            = bool(gp('save_viz').value)
        self.viz_dir             = str(gp('viz_dir').value)
        self.oneshot             = bool(gp('oneshot').value)
        self.min_update          = float(gp('min_update_period').value)

        Path(self.viz_dir).mkdir(parents=True, exist_ok=True)

        # Subscribe (prefer /map for segmentation)
        self.sub = self.create_subscription(OccupancyGrid, '/map', self.cb, 10)

        self.graph = {"stamp": 0.0, "rooms": [], "corridors": [], "doors": [], "edges": [],
                      "resolution": None, "origin": None}
        self.srv = self.create_service(Trigger, 'get_room_graph', self.handle_get_graph)

        self._last_update_t = 0.0
        self._last_counts = None

    def handle_get_graph(self, req, res):
        res.success = True
        res.message = json.dumps(self.graph)
        return res

    def cb(self, msg: OccupancyGrid):
        import time as _t
        now = _t.monotonic()
        if now - self._last_update_t < self.min_update:
            return
        self._last_update_t = now

        h, w = msg.info.height, msg.info.width
        res  = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y

        data = np.asarray(msg.data, np.int16).reshape(h, w)
        maxv = int(data.max())

        # --- Build free/occ masks that work for map or costmap
        if maxv <= 100:
            occ = ((data >= self.occ_threshold_map) | (data < 0)).astype(np.uint8) * 255
            free = (data == 0).astype(np.uint8) * 255
        else:
            occ = ((data >= self.occ_threshold) | (data < 0)).astype(np.uint8) * 255
            free = (data <= self.free_threshold).astype(np.uint8) * 255

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k3, iterations=1)
        free = cv2.bitwise_and(free, cv2.bitwise_not(occ))
        if free.sum() == 0:
            self.get_logger().warn("Free mask empty; check thresholds/topics")
            return

        # --- Distance transform (meters)
        dt_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)  # pixel radii
        dt_m  = dt_px * res

        # --- Seeds for rooms
        seeds = (dt_m >= self.room_seed_radius_m).astype(np.uint8) * 255
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, k3, iterations=1)
        n_seeds, markers = cv2.connectedComponents(seeds)
        if n_seeds <= 1:
            thr = max(0.6 * self.room_seed_radius_m, 0.6)
            seeds = (dt_m >= thr).astype(np.uint8) * 255
            n_seeds, markers = cv2.connectedComponents(seeds)

        # --- Watershed on inverted distance
        elev = cv2.normalize(-dt_px, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elev_rgb = cv2.merge([elev, elev, elev])
        markers = markers.astype(np.int32)
        cv2.watershed(elev_rgb, markers)  # -1 on boundaries
        labels = np.where(free > 0, markers, 0)

        # --- Collect components, classify rooms vs corridors, store pixel centroids too
        rooms, corridors = [], []
        min_room_area_px = int(self.min_room_area_m2 / (res*res))
        for lab in range(1, labels.max()+1):
            mask = (labels == lab)
            area = int(mask.sum())
            if area < 100:
                continue
            radii = dt_m[mask]
            med_r = float(np.median(radii))
            ys, xs = np.where(mask)
            cx_pix = float(xs.mean()); cy_pix = float(ys.mean())
            cx, cy = grid_to_map_xy(cx_pix, cy_pix, h, res, ox, oy)
            entry = {"id": int(lab),
                     "centroid": [cx, cy],
                     "centroid_px": [cx_pix, cy_pix],
                     "median_radius_m": med_r,
                     "area_px": area}
            if med_r * 2.0 <= self.corridor_max_width_m:
                corridors.append(entry)
            else:
                if area >= min_room_area_px:
                    rooms.append(entry)

        # --- Edges + doors
        edges, doors = [], []
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        labs = np.unique(labels[labels > 0])

        corridor_ids = set(e["id"] for e in corridors)

        for i in range(len(labs)):
            li = labs[i]
            mask_i = (labels == li).astype(np.uint8)
            nb_i = cv2.dilate(mask_i, k, 1).astype(bool)
            touching = np.unique(labels[nb_i])
            for lj in touching:
                if lj <= li or lj == 0:
                    continue
                mask_j = (labels == lj).astype(np.uint8)
                band = (cv2.dilate(mask_i, k, 1) & mask_j).astype(bool)
                if not band.any():
                    continue

                band_dt = dt_m[band]
                if band_dt.size == 0:
                    continue
                min_half = float(np.min(band_dt))
                ys, xs = np.where(band)
                # pick narrow door locus
                sel = band_dt <= min(self.door_max_width_m/2.0, np.percentile(band_dt, 25))
                if not np.any(sel):
                    sel = (band_dt == min_half)
                xs_sel, ys_sel = xs[sel], ys[sel]
                if xs_sel.size == 0:
                    xs_sel, ys_sel = xs, ys
                mx = float(xs_sel.mean()); my = float(ys_sel.mean())
                door_x, door_y = grid_to_map_xy(mx, my, h, res, ox, oy)

                if (min_half * 2.0 <= self.door_max_width_m) or \
                   (li in corridor_ids) or (lj in corridor_ids):
                    edges.append({"u": int(li), "v": int(lj), "door": [door_x, door_y]})
                    doors.append({"between": [int(li), int(lj)],
                                  "pose": [door_x, door_y],
                                  "width_m": float(2.0*min_half)})

        self.graph = {
            "stamp": self.get_clock().now().nanoseconds * 1e-9,
            "resolution": res, "origin": [ox, oy],
            "rooms": rooms, "corridors": corridors, "doors": doors, "edges": edges
        }

        self.print_summary(rooms, corridors, doors, edges)

        # Save artifacts
        if self.save_graph_json:
            Path(self.graph_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.graph_json_path, 'w') as f:
                json.dump(self.graph, f, indent=2)

        if self.save_viz:
            self.save_visualization(labels, rooms, corridors, doors, h, w, res, ox, oy, free, occ)

        if self.oneshot:
            self.get_logger().info(f"Saved JSON to {self.graph_json_path} and PNGs to {self.viz_dir}. Exiting (oneshot).")
            self.destroy_subscription(self.sub)
            def _stop():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
            self.create_timer(0.05, _stop)

    # ---------- Helpers ----------
    def print_summary(self, rooms, corridors, doors, edges):
        counts = (len(rooms), len(corridors), len(edges))
        if counts != self._last_counts:
            self.get_logger().info(f"Graph: rooms={counts[0]}, corridors={counts[1]}, edges={counts[2]}")
            self._last_counts = counts

            def fmt_xy(xy): return f"(x={xy[0]:6.2f}, y={xy[1]:6.2f})"
            print("\n[rooms]")
            for r in sorted(rooms, key=lambda e: e["id"]):
                print(f" - {r['id']:>3}  {fmt_xy(r['centroid'])}  area_px={r['area_px']:<6}  median_r={r['median_radius_m']:.2f}")
            print("[corridors]")
            for c in sorted(corridors, key=lambda e: e["id"]):
                print(f" - {c['id']:>3}  {fmt_xy(c['centroid'])}  area_px={c['area_px']:<6}  median_r={c['median_radius_m']:.2f}")
            print("[doors]")
            for d in doors:
                print(f" - ({d['between'][0]},{d['between'][1]}) at {fmt_xy(d['pose'])} width≈{d['width_m']:.2f}m")
            print("")

    def save_visualization(self, labels, rooms, corridors, doors, h, w, res, ox, oy, free, occ):
        # base = white free, black obstacles
        base = np.full((h, w, 3), 255, np.uint8)
        base[occ > 0] = (0, 0, 0)

        # random but stable colors per label
        rng = np.random.default_rng(42)
        colors = {}
        for lab in np.unique(labels):
            if lab <= 0: continue
            colors[int(lab)] = tuple(int(v) for v in rng.integers(60, 230, size=3))

        overlay = base.copy()
        # rooms colored stronger; corridors tinted
        room_ids = set(e["id"] for e in rooms)
        corridor_ids = set(e["id"] for e in corridors)
        for lab in room_ids | corridor_ids:
            mask = (labels == lab)
            col = colors[lab]
            if lab in corridor_ids:
                col = (col[0]//2, col[1]//2, col[2])  # bluish tint
            overlay[mask] = col

        vis = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)

        # draw centroids and ids
        def draw_dot(px, py, color, text):
            cv2.circle(vis, (int(px), int(py)), 4, color, thickness=-1)
            cv2.putText(vis, text, (int(px)+6, int(py)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,20,20), 1, cv2.LINE_AA)

        for r in rooms:
            px, py = r["centroid_px"]
            draw_dot(px, py, (0,180,0), f"R{r['id']}")
        for c in corridors:
            px, py = c["centroid_px"]
            draw_dot(px, py, (200,120,0), f"C{c['id']}")

        # doors
        for d in doors:
            # convert back to pixel for drawing
            dx = (d["pose"][0] - ox) / res
            dy = h - 1 - (d["pose"][1] - oy) / res
            cv2.circle(vis, (int(dx), int(dy)), 3, (0,0,255), -1)

        # write file
        ts = int(time.time())
        out_path = os.path.join(self.viz_dir, f"room_graph_{ts}.png")
        cv2.imwrite(out_path, vis)
        self.get_logger().info(f"Wrote viz: {out_path}")

def main():
    rclpy.init()
    rclpy.spin(RoomGraphBuilder())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
