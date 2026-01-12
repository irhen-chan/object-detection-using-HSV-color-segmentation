"""
Can Calibration Tool (HSV + Shape + Logo/Texture Filters)

- HSV-only "white" will detect everything white.
- This tool lets you tune:
  1) HSV range (candidate proposal)
  2) Shape filters (area, aspect ratio, fill ratio)
  3) Texture filters (dark pixel ratio + edge density) to confirm it's a printed can.

Controls:
  S - Save to config.json
  R - Reset to defaults
  F - Freeze/unfreeze frame 
  Q - Quit
"""

import cv2
import numpy as np
import json
import os


class CanCalibrator:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        self.win_main = "Calibrator - Original"
        self.win_mask = "Calibrator - HSV Mask"
        self.win_best = "Calibrator - Best ROI (debug)"
        self.win_edges = "Calibrator - Best ROI Edges"
        self.win_dark = "Calibrator - Best ROI DarkMask"

        # Load existing or defaults 
        lower = self.config.get("can_hsv_lower", [0, 0, 175])
        upper = self.config.get("can_hsv_upper", [180, 45, 255])

        self.h_min, self.s_min, self.v_min = lower
        self.h_max, self.s_max, self.v_max = upper

        self.min_area = int(self.config.get("min_can_area", 3000))
        self.max_area = int(self.config.get("max_can_area", 120000))

        self.min_ar = float(self.config.get("min_aspect_ratio", 1.2))
        self.max_ar = float(self.config.get("max_aspect_ratio", 4.0))

        # fill ratio + texture filters
        self.min_fill_pct = int(self.config.get("min_fill_pct", 35))      # area / (w*h) * 100
        self.dark_thresh = int(self.config.get("dark_thresh", 95))         # grayscale < dark_thresh
        self.min_dark_pct = int(self.config.get("min_dark_pct", 2))        # dark pixels %
        self.canny1 = int(self.config.get("canny1", 60))
        self.canny2 = int(self.config.get("canny2", 140))
        self.min_edge_pct = int(self.config.get("min_edge_pct", 2))        # edge pixels %

        # morphology
        self.kernel_size = int(self.config.get("kernel_size", 5))          # odd
        self.dilate_iter = int(self.config.get("dilate_iter", 1))

        self.frozen = False
        self.frozen_frame = None

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_config(self):
        self.config["can_hsv_lower"] = [self.h_min, self.s_min, self.v_min]
        self.config["can_hsv_upper"] = [self.h_max, self.s_max, self.v_max]

        self.config["min_can_area"] = int(self.min_area)
        self.config["max_can_area"] = int(self.max_area)
        self.config["min_aspect_ratio"] = float(self.min_ar)
        self.config["max_aspect_ratio"] = float(self.max_ar)
        self.config["min_fill_pct"] = int(self.min_fill_pct)
        self.config["dark_thresh"] = int(self.dark_thresh)
        self.config["min_dark_pct"] = int(self.min_dark_pct)
        self.config["canny1"] = int(self.canny1)
        self.config["canny2"] = int(self.canny2)
        self.config["min_edge_pct"] = int(self.min_edge_pct)

        self.config["kernel_size"] = int(self.kernel_size)
        self.config["dilate_iter"] = int(self.dilate_iter)

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"\n[Saved] {self.config_path}")
        print(f"HSV lower: [{self.h_min}, {self.s_min}, {self.v_min}]")
        print(f"HSV upper: [{self.h_max}, {self.s_max}, {self.v_max}]")
        print(f"Area: {self.min_area} - {self.max_area}")
        print(f"AR: {self.min_ar:.2f} - {self.max_ar:.2f}")
        print(f"Fill% >= {self.min_fill_pct} | Dark% >= {self.min_dark_pct} (th={self.dark_thresh}) | Edge% >= {self.min_edge_pct} (canny {self.canny1},{self.canny2})")
        print(f"Morph: kernel={self.kernel_size} dilate={self.dilate_iter}\n")

    def _reset_defaults(self):
        # Slightly tighter than “anything white”
        self.h_min, self.s_min, self.v_min = 0, 0, 175
        self.h_max, self.s_max, self.v_max = 180, 45, 255

        self.min_area, self.max_area = 3000, 120000
        self.min_ar, self.max_ar = 1.2, 4.0
        self.min_fill_pct = 35
        self.dark_thresh = 95
        self.min_dark_pct = 2
        self.canny1, self.canny2 = 60, 140
        self.min_edge_pct = 2

        self.kernel_size = 5
        self.dilate_iter = 1

        self._sync_trackbars()
        print("[Calibrator] Reset defaults")

    def _nothing(self, x):
        pass

    def _ensure_odd(self, v: int) -> int:
        return v if v % 2 == 1 else v + 1

    def _sync_trackbars(self):
        # HSV
        cv2.setTrackbarPos("H Min", self.win_main, self.h_min)
        cv2.setTrackbarPos("S Min", self.win_main, self.s_min)
        cv2.setTrackbarPos("V Min", self.win_main, self.v_min)
        cv2.setTrackbarPos("H Max", self.win_main, self.h_max)
        cv2.setTrackbarPos("S Max", self.win_main, self.s_max)
        cv2.setTrackbarPos("V Max", self.win_main, self.v_max)

        # Shape
        cv2.setTrackbarPos("Min Area", self.win_main, int(self.min_area))
        cv2.setTrackbarPos("Max Area", self.win_main, int(self.max_area))
        cv2.setTrackbarPos("Min AR x100", self.win_main, int(self.min_ar * 100))
        cv2.setTrackbarPos("Max AR x100", self.win_main, int(self.max_ar * 100))
        cv2.setTrackbarPos("Min Fill %", self.win_main, int(self.min_fill_pct))

        # Texture
        cv2.setTrackbarPos("Dark Thresh", self.win_main, int(self.dark_thresh))
        cv2.setTrackbarPos("Min Dark %", self.win_main, int(self.min_dark_pct))
        cv2.setTrackbarPos("Canny 1", self.win_main, int(self.canny1))
        cv2.setTrackbarPos("Canny 2", self.win_main, int(self.canny2))
        cv2.setTrackbarPos("Min Edge %", self.win_main, int(self.min_edge_pct))

        # Morphology
        cv2.setTrackbarPos("Kernel", self.win_main, int(self.kernel_size))
        cv2.setTrackbarPos("Dilate", self.win_main, int(self.dilate_iter))

    def _metrics_for_bbox(self, frame, bbox, contour_area):
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Dark pixels = “printed logo/text” proxy
        dark_mask = (gray < self.dark_thresh).astype(np.uint8) * 255
        dark_ratio = float(np.mean(dark_mask > 0))  # 0..1

        # Edge density = “graphics” proxy
        edges = cv2.Canny(gray, self.canny1, self.canny2)
        edge_ratio = float(np.mean(edges > 0))      # 0..1

        # Fill ratio = contour area relative to bbox
        fill_ratio = float(contour_area) / float(w * h) if (w * h) > 0 else 0.0

        return {
            "roi": roi,
            "gray": gray,
            "dark_mask": dark_mask,
            "edges": edges,
            "dark_ratio": dark_ratio,
            "edge_ratio": edge_ratio,
            "fill_ratio": fill_ratio
        }

    def run(self, camera_index=0):
        print("\nControls: S=Save | R=Reset | F=Freeze | Q=Quit\n"
              "Goal: mask should capture the can body, BUT final green box should only lock on the can.\n")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return

        cv2.namedWindow(self.win_main)
        cv2.namedWindow(self.win_mask)
        cv2.namedWindow(self.win_best)
        cv2.namedWindow(self.win_edges)
        cv2.namedWindow(self.win_dark)

        # Trackbars (HSV)
        cv2.createTrackbar("H Min", self.win_main, self.h_min, 180, self._nothing)
        cv2.createTrackbar("S Min", self.win_main, self.s_min, 255, self._nothing)
        cv2.createTrackbar("V Min", self.win_main, self.v_min, 255, self._nothing)
        cv2.createTrackbar("H Max", self.win_main, self.h_max, 180, self._nothing)
        cv2.createTrackbar("S Max", self.win_main, self.s_max, 255, self._nothing)
        cv2.createTrackbar("V Max", self.win_main, self.v_max, 255, self._nothing)

        # Shape
        cv2.createTrackbar("Min Area", self.win_main, int(self.min_area), 300000, self._nothing)
        cv2.createTrackbar("Max Area", self.win_main, int(self.max_area), 400000, self._nothing)
        cv2.createTrackbar("Min AR x100", self.win_main, int(self.min_ar * 100), 600, self._nothing)
        cv2.createTrackbar("Max AR x100", self.win_main, int(self.max_ar * 100), 800, self._nothing)
        cv2.createTrackbar("Min Fill %", self.win_main, int(self.min_fill_pct), 100, self._nothing)

        # Texture
        cv2.createTrackbar("Dark Thresh", self.win_main, int(self.dark_thresh), 255, self._nothing)
        cv2.createTrackbar("Min Dark %", self.win_main, int(self.min_dark_pct), 30, self._nothing)
        cv2.createTrackbar("Canny 1", self.win_main, int(self.canny1), 255, self._nothing)
        cv2.createTrackbar("Canny 2", self.win_main, int(self.canny2), 255, self._nothing)
        cv2.createTrackbar("Min Edge %", self.win_main, int(self.min_edge_pct), 30, self._nothing)

        # Morphology
        cv2.createTrackbar("Kernel", self.win_main, int(self.kernel_size), 31, self._nothing)
        cv2.createTrackbar("Dilate", self.win_main, int(self.dilate_iter), 5, self._nothing)

        while True:
            if not self.frozen:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                self.frozen_frame = frame.copy()
            else:
                frame = self.frozen_frame.copy()

            # Read trackbars
            self.h_min = cv2.getTrackbarPos("H Min", self.win_main)
            self.s_min = cv2.getTrackbarPos("S Min", self.win_main)
            self.v_min = cv2.getTrackbarPos("V Min", self.win_main)
            self.h_max = cv2.getTrackbarPos("H Max", self.win_main)
            self.s_max = cv2.getTrackbarPos("S Max", self.win_main)
            self.v_max = cv2.getTrackbarPos("V Max", self.win_main)

            self.min_area = cv2.getTrackbarPos("Min Area", self.win_main)
            self.max_area = cv2.getTrackbarPos("Max Area", self.win_main)
            self.min_ar = cv2.getTrackbarPos("Min AR x100", self.win_main) / 100.0
            self.max_ar = cv2.getTrackbarPos("Max AR x100", self.win_main) / 100.0
            self.min_fill_pct = cv2.getTrackbarPos("Min Fill %", self.win_main)

            self.dark_thresh = cv2.getTrackbarPos("Dark Thresh", self.win_main)
            self.min_dark_pct = cv2.getTrackbarPos("Min Dark %", self.win_main)
            self.canny1 = cv2.getTrackbarPos("Canny 1", self.win_main)
            self.canny2 = cv2.getTrackbarPos("Canny 2", self.win_main)
            self.min_edge_pct = cv2.getTrackbarPos("Min Edge %", self.win_main)

            self.kernel_size = self._ensure_odd(max(1, cv2.getTrackbarPos("Kernel", self.win_main)))
            self.dilate_iter = cv2.getTrackbarPos("Dilate", self.win_main)

            # HSV mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([self.h_min, self.s_min, self.v_min])
            upper = np.array([self.h_max, self.s_max, self.v_max])
            mask = cv2.inRange(hsv, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if self.dilate_iter > 0:
                mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display = frame.copy()
            best = None  # (score, bbox, metrics)
            best_roi_debug = None

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area or area > self.max_area:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                if w <= 0 or h <= 0:
                    continue

                ar = h / w
                if not (self.min_ar <= ar <= self.max_ar):
                    continue

                metrics = self._metrics_for_bbox(frame, (x, y, w, h), area)
                if metrics is None:
                    continue

                fill_pct = metrics["fill_ratio"] * 100.0
                dark_pct = metrics["dark_ratio"] * 100.0
                edge_pct = metrics["edge_ratio"] * 100.0

                pass_fill = fill_pct >= self.min_fill_pct
                pass_dark = dark_pct >= self.min_dark_pct
                pass_edge = edge_pct >= self.min_edge_pct

                passed_all = pass_fill and pass_dark and pass_edge

                # Score to pick the “most can-like” candidate
                score = area * (1.0 + 6.0 * metrics["dark_ratio"] + 4.0 * metrics["edge_ratio"])

                if best is None or score > best[0]:
                    best = (score, (x, y, w, h), metrics)
                    best_roi_debug = (fill_pct, dark_pct, edge_pct, passed_all)

                # Draw candidate boxes
                color = (0, 255, 0) if passed_all else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    display,
                    f"AR:{ar:.2f} Fill:{fill_pct:.0f}% Dark:{dark_pct:.1f}% Edge:{edge_pct:.1f}%",
                    (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

            # UI header
            status = "FROZEN" if self.frozen else "LIVE"
            cv2.putText(display, f"Mode: {status} | S=Save R=Reset F=Freeze Q=Quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.putText(display,
                        f"HSV [{self.h_min},{self.s_min},{self.v_min}] - [{self.h_max},{self.s_max},{self.v_max}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Show best ROI debug windows
            if best is not None:
                _, (x, y, w, h), m = best
                roi = m["roi"]
                edges = m["edges"]
                dark = m["dark_mask"]

                # Resize debug views
                roi_show = cv2.resize(roi, (320, 320))
                edges_show = cv2.resize(edges, (320, 320))
                dark_show = cv2.resize(dark, (320, 320))

                cv2.imshow(self.win_best, roi_show)
                cv2.imshow(self.win_edges, edges_show)
                cv2.imshow(self.win_dark, dark_show)

                # Highlight best bbox
                fill_pct, dark_pct, edge_pct, passed_all = best_roi_debug
                label = "BEST ✅" if passed_all else "BEST (fails texture)"
                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 50, 255), 3)
                cv2.putText(display, f"{label} Fill:{fill_pct:.0f}% Dark:{dark_pct:.1f}% Edge:{edge_pct:.1f}%",
                            (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 255), 2)
            else:
                blank = np.zeros((320, 320, 3), dtype=np.uint8)
                cv2.imshow(self.win_best, blank)
                cv2.imshow(self.win_edges, np.zeros((320, 320), dtype=np.uint8))
                cv2.imshow(self.win_dark, np.zeros((320, 320), dtype=np.uint8))

            cv2.imshow(self.win_main, display)
            cv2.imshow(self.win_mask, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                self._save_config()
            elif key == ord("r"):
                self._reset_defaults()
            elif key == ord("f"):
                self.frozen = not self.frozen
                print("[Calibrator] Freeze:", "ON" if self.frozen else "OFF")

        cap.release()
        cv2.destroyAllWindows()
        print("[Calibrator] Done")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--config", default="config.json")
    args = p.parse_args()

    CanCalibrator(args.config).run(args.camera)


if __name__ == "__main__":
    main()
