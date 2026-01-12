"""
Monster Sip Detector
Detects when you take a sip from a white Monster can and triggers Kevin Levrone video playback.

Upgraded can detection:
1) HSV mask proposes "white/silver" candidates
2) Shape filters (area + aspect ratio)
3) Texture verification to avoid detecting random white objects:
   - fill ratio (% of bbox covered by contour)
   - dark pixel ratio (printed logo/text proxy)
   - edge density (graphics proxy)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Audio playback
import pygame


@dataclass
class Detection:
    """Holds detection results for a single frame."""
    can_detected: bool
    can_bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    can_center: Optional[Tuple[int, int]]
    mouth_position: Optional[Tuple[int, int]]
    mouth_open: bool
    distance_to_mouth: Optional[float]
    is_sipping: bool


class Config:
    """Configuration manager for the detector."""

    DEFAULT_CONFIG = {
        "camera_index": 0,
        "window_width": 1280,
        "window_height": 720,

        # White Monster can HSV range (white/silver colors)
        # (slightly tighter than "everything white")
        "can_hsv_lower": [0, 0, 175],
        "can_hsv_upper": [180, 45, 255],

        # Minimum can area to filter noise
        "min_can_area": 3000,
        "max_can_area": 120000,

        # Can aspect ratio (height/width) - cans are taller than wide
        "min_aspect_ratio": 1.2,
        "max_aspect_ratio": 4.0,

        # fill ratio = contour area / bbox area (%)
        "min_fill_pct": 35,

        # dark pixels (printed logo/text proxy). grayscale < dark_thresh counts as "dark"
        "dark_thresh": 95,
        "min_dark_pct": 2,     # % of ROI pixels that are dark

        # edge density (graphics proxy)
        "canny1": 60,
        "canny2": 140,
        "min_edge_pct": 2,     # % of ROI pixels that are edges

        # Morphology controls for mask cleanup (match your calibrator keys)
        "kernel_size": 5,      # must be odd
        "dilate_iter": 1,

        # Sip detection thresholds
        "sip_distance_threshold": 150,  # pixels from mouth to trigger
        "sip_cooldown": 3.0,            # seconds between triggers
        "mouth_open_threshold": 0.02,   # relative mouth opening

        # Video settings
        "video_path": "assets/flex_video.mp4",
        "audio_path": "assets/your_audio.MP3",  # optional separate audio file
        "video_duration": 3.0,          # max seconds to play (ignored - plays full video now)

        # Debug settings
        "debug_mode": True,
        "show_hsv_mask": False
    }

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.settings = self._load_config()

    def _load_config(self) -> dict:
        """Load config from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    config = self.DEFAULT_CONFIG.copy()
                    config.update(loaded)  # merge
                    return config
            except Exception as e:
                print(f"[Config] Error loading config: {e}, using defaults")

        self._save_config(self.DEFAULT_CONFIG)
        return self.DEFAULT_CONFIG.copy()

    def _save_config(self, config: dict):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def get(self, key: str):
        """Get a config value."""
        return self.settings.get(key)

    def set(self, key: str, value):
        """Set a config value and save."""
        self.settings[key] = value
        self._save_config(self.settings)


class CanDetector:
    """Detects white Monster cans using HSV + shape + texture verification."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _ensure_odd(n: int) -> int:
        return n if (n % 2 == 1) else n + 1

    def _texture_pass(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], contour_area: float):
        """
        Verify ROI looks like a printed can (not blank white object).
        Returns: (passed: bool, score: float)
        """
        x, y, w, h = bbox
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return False, 0.0

        min_fill_pct = float(self.config.get("min_fill_pct") or 35)

        dark_thresh = int(self.config.get("dark_thresh") or 95)
        min_dark_pct = float(self.config.get("min_dark_pct") or 2)

        canny1 = int(self.config.get("canny1") or 60)
        canny2 = int(self.config.get("canny2") or 140)
        min_edge_pct = float(self.config.get("min_edge_pct") or 2)

        # Fill ratio
        bbox_area = float(w * h) if (w * h) > 0 else 1.0
        fill_pct = (float(contour_area) / bbox_area) * 100.0

        # Dark pixel ratio (print proxy)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < dark_thresh)
        dark_pct = float(np.mean(dark_mask)) * 100.0

        # Edge density (graphics proxy)
        edges = cv2.Canny(gray, canny1, canny2)
        edge_pct = float(np.mean(edges > 0)) * 100.0

        passed = (fill_pct >= min_fill_pct) and (dark_pct >= min_dark_pct) and (edge_pct >= min_edge_pct)

        # Score candidates so we pick the most can-like contour.
        # Use ratios (0..1) for stable scaling.
        dark_ratio = dark_pct / 100.0
        edge_ratio = edge_pct / 100.0
        score = float(contour_area) * (1.0 + 6.0 * dark_ratio + 4.0 * edge_ratio)

        return passed, score

    def detect(self, frame: np.ndarray):
        """
        Detect white Monster can in frame.
        Returns: (bounding_box, center, mask)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask for white/silver
        lower = np.array(self.config.get("can_hsv_lower"), dtype=np.uint8)
        upper = np.array(self.config.get("can_hsv_upper"), dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphology cleanup (now configurable)
        k = int(self.config.get("kernel_size") or 5)
        k = self._ensure_odd(max(1, k))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        dilate_iter = int(self.config.get("dilate_iter") or 1)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_can = None
        best_score = 0.0

        min_area = float(self.config.get("min_can_area") or 3000)
        max_area = float(self.config.get("max_can_area") or 120000)
        min_ar = float(self.config.get("min_aspect_ratio") or 1.2)
        max_ar = float(self.config.get("max_aspect_ratio") or 4.0)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = h / w
            if not (min_ar <= aspect_ratio <= max_ar):
                continue

            # Step 2: texture verification 
            passed, score = self._texture_pass(frame, (x, y, w, h), area)
            if not passed:
                continue

            if score > best_score:
                best_score = score
                best_can = (x, y, w, h)

        if best_can:
            x, y, w, h = best_can
            center = (x + w // 2, y + h // 2)
            return best_can, center, mask

        return None, None, mask


class FaceDetector:
    """Handles face mesh detection for mouth tracking."""

    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    def __init__(self, config: Config):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame: np.ndarray):
        """
        Detect mouth position and state.
        Returns: (mouth_center, is_open, landmarks_dict)
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, False, {}

        landmarks = results.multi_face_landmarks[0].landmark

        upper_lip = landmarks[self.UPPER_LIP]
        lower_lip = landmarks[self.LOWER_LIP]
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]

        mouth_cx = int((mouth_left.x + mouth_right.x) / 2 * w)
        mouth_cy = int((upper_lip.y + lower_lip.y) / 2 * h)

        mouth_height = abs(lower_lip.y - upper_lip.y)
        mouth_width = abs(mouth_right.x - mouth_left.x)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.0

        is_open = mouth_ratio > float(self.config.get("mouth_open_threshold") or 0.02)

        landmarks_dict = {
            "upper_lip": (int(upper_lip.x * w), int(upper_lip.y * h)),
            "lower_lip": (int(lower_lip.x * w), int(lower_lip.y * h)),
            "mouth_left": (int(mouth_left.x * w), int(mouth_left.y * h)),
            "mouth_right": (int(mouth_right.x * w), int(mouth_right.y * h)),
        }

        return (mouth_cx, mouth_cy), is_open, landmarks_dict

    def release(self):
        self.face_mesh.close()


class VideoPlayer:
    """Handles overlay video playback with sound when sip is detected."""

    def __init__(self, config: Config):
        self.config = config
        self.video_cap = None
        self.is_playing = False
        self.video_fps = 30.0
        self.frame_count = 0
        self.current_frame_idx = 0
        
        # Initialize pygame mixer for audio
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.audio_loaded = False
        
        self._load_video()

    def _load_video(self):
        video_path = self.config.get("video_path")
        audio_path = self.config.get("audio_path") or ""
        
        if video_path and os.path.exists(video_path):
            self.video_cap = cv2.VideoCapture(video_path)
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[VideoPlayer] Loaded video: {video_path}")
            print(f"[VideoPlayer] FPS: {self.video_fps}, Frames: {self.frame_count}")
            
            # Load audio - try separate audio file first, then video file
            audio_file = audio_path if audio_path and os.path.exists(audio_path) else video_path
            try:
                pygame.mixer.music.load(audio_file)
                self.audio_loaded = True
                print(f"[VideoPlayer] Loaded audio from: {audio_file}")
            except pygame.error as e:
                print(f"[VideoPlayer] Could not load audio: {e}")
                print("[VideoPlayer] Video will play without sound")
                self.audio_loaded = False
        else:
            print(f"[VideoPlayer] Video not found: {video_path}")
            print("[VideoPlayer] Add your video at assets/flex_video.mp4")
            self.video_cap = None

    def trigger(self):
        """Start video and audio playback."""
        if self.video_cap is None:
            print("[VideoPlayer] No video loaded!")
            return
        
        # Don't re-trigger if already playing
        if self.is_playing:
            return
            
        # Reset video to beginning
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        self.is_playing = True
        
        # Start audio playback
        if self.audio_loaded:
            pygame.mixer.music.stop()
            pygame.mixer.music.play()
        
        print("[VideoPlayer] 🎬 TRIGGERED! Playing flex video with sound...")

    def get_frame(self, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get current video frame. Plays until video ends completely."""
        if not self.is_playing or self.video_cap is None:
            return None

        ret, frame = self.video_cap.read()
        
        if not ret:
            # Video finished - stop playback
            self.is_playing = False
            if self.audio_loaded:
                pygame.mixer.music.stop()
            print("[VideoPlayer] Video finished")
            return None
        
        self.current_frame_idx += 1
        return cv2.resize(frame, target_size)

    def release(self):
        """Clean up resources."""
        if self.video_cap:
            self.video_cap.release()
        pygame.mixer.music.stop()
        pygame.mixer.quit()


class MonsterSipDetector:
    """Main application class that orchestrates detection and playback."""

    def __init__(self, config_path: str = "config.json"):
        print("\n" + "=" * 50)
        print("  MONSTER SIP DETECTOR v1.0")
        print("  Take a sip, get the flex 💪")
        print("=" * 50 + "\n")

        self.config = Config(config_path)
        self.can_detector = CanDetector(self.config)
        self.face_detector = FaceDetector(self.config)
        self.video_player = VideoPlayer(self.config)

        self.last_sip_time = 0.0
        self.sip_count = 0
        self.running = False

        self.fps = 0.0
        self.frame_times = []

                # Can latch/smoothing state 
        self._can_bbox_smooth = None   # np.array([x,y,w,h], float)
        self._can_center_smooth = None # np.array([cx,cy], float)
        self._can_seen_streak = 0
        self._can_missed_streak = 0
        self._can_from_latch = False

    def _stabilize_can(self, raw_bbox, raw_center):
        """
        Smooth + latch can detection.
        Returns (stable_bbox, stable_center, from_latch)
        """
        alpha = float(self.config.get("bbox_smoothing_alpha") or 0.25)
        confirm = int(self.config.get("can_confirm_frames") or 2)
        miss_tol = int(self.config.get("can_miss_tolerance") or 10)

        if raw_bbox is not None and raw_center is not None:
            self._can_seen_streak += 1
            self._can_missed_streak = 0

            b = np.array(raw_bbox, dtype=np.float32)
            c = np.array(raw_center, dtype=np.float32)

            if self._can_bbox_smooth is None:
                self._can_bbox_smooth = b
                self._can_center_smooth = c
            else:
                self._can_bbox_smooth = (1 - alpha) * self._can_bbox_smooth + alpha * b
                self._can_center_smooth = (1 - alpha) * self._can_center_smooth + alpha * c

            # Only "lock" after confirm frames
            if self._can_seen_streak >= confirm:
                self._can_from_latch = False
                bb = tuple(self._can_bbox_smooth.round().astype(int))
                cc = tuple(self._can_center_smooth.round().astype(int))
                return bb, cc, False

            # Not confirmed yet → don't output
            return None, None, False

        # No raw detection this frame
        self._can_seen_streak = 0
        self._can_missed_streak += 1

        # If we have a previous smooth bbox and we're within tolerance, keep it
        if self._can_bbox_smooth is not None and self._can_missed_streak <= miss_tol:
            self._can_from_latch = True
            bb = tuple(self._can_bbox_smooth.round().astype(int))
            cc = tuple(self._can_center_smooth.round().astype(int))
            return bb, cc, True

        # Lost too long → reset
        self._can_bbox_smooth = None
        self._can_center_smooth = None
        self._can_from_latch = False
        return None, None, False

    @staticmethod
    def _calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def _check_sip(self, can_center: Tuple[int, int], mouth_pos: Tuple[int, int], mouth_open: bool) -> bool:
        distance = self._calculate_distance(can_center, mouth_pos)
        threshold = float(self.config.get("sip_distance_threshold") or 150)
        cooldown = float(self.config.get("sip_cooldown") or 3.0)

        if (time.time() - self.last_sip_time) < cooldown:
            return False


        return distance < threshold

    def _draw_debug_overlay(self, frame: np.ndarray, detection: Detection, landmarks: dict):
        h, w = frame.shape[:2]

        if detection.can_bbox:
            x, y, bw, bh = detection.can_bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.putText(frame, "MONSTER CAN", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


        if detection.mouth_position:
            mx, my = detection.mouth_position
            color = (0, 255, 255) if detection.mouth_open else (255, 255, 0)
            cv2.circle(frame, (mx, my), 10, color, -1)
            for _, pos in landmarks.items():
                cv2.circle(frame, pos, 3, (255, 0, 255), -1)

        if detection.can_center and detection.mouth_position:
            cv2.line(frame, detection.can_center, detection.mouth_position, (255, 128, 0), 2)
            mid_x = (detection.can_center[0] + detection.mouth_position[0]) // 2
            mid_y = (detection.can_center[1] + detection.mouth_position[1]) // 2
            cv2.putText(frame, f"{detection.distance_to_mouth:.0f}px", (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

        panel_y = 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        panel_y += 30
        can_status = "DETECTED" if detection.can_detected else "NOT FOUND"
        can_color = (0, 255, 0) if detection.can_detected else (0, 0, 255)
        cv2.putText(frame, f"Can: {can_status}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, can_color, 2)

        panel_y += 30
        mouth_status = "OPEN" if detection.mouth_open else "CLOSED"
        cv2.putText(frame, f"Mouth: {mouth_status}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        panel_y += 30
        cv2.putText(frame, f"Sips: {self.sip_count}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show if video is playing
        panel_y += 30
        if self.video_player.is_playing:
            cv2.putText(frame, "VIDEO PLAYING", (10, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detection.mouth_position:
            threshold = int(self.config.get("sip_distance_threshold") or 150)
            cv2.circle(frame, detection.mouth_position, threshold, (255, 128, 0), 1)

        cv2.putText(frame, "Q quit | D debug | M mask | R reset count | T trigger",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_sip_alert(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        text = "SIP DETECTED!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.5
        thickness = 5

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2

        cv2.putText(frame, text, (x + 3, y + 3), font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thickness)

    def process_frame(self, frame: np.ndarray):
        frame = cv2.flip(frame, 1)

        raw_bbox, raw_center, mask = self.can_detector.detect(frame)
        can_bbox, can_center, from_latch = self._stabilize_can(raw_bbox, raw_center)

        mouth_pos, mouth_open, landmarks = self.face_detector.detect(frame)

        distance = None
        if can_center and mouth_pos:
            distance = self._calculate_distance(can_center, mouth_pos)

        is_sipping = False
        if can_center and mouth_pos:
            is_sipping = self._check_sip(can_center, mouth_pos, mouth_open)
            if is_sipping:
                self.last_sip_time = time.time()
                self.sip_count += 1
                self.video_player.trigger()

        detection = Detection(
            can_detected=can_bbox is not None,
            can_bbox=can_bbox,
            can_center=can_center,
            mouth_position=mouth_pos,
            mouth_open=mouth_open,
            distance_to_mouth=distance,
            is_sipping=is_sipping
        )

        if self.config.get("debug_mode"):
            self._draw_debug_overlay(frame, detection, landmarks)

        if is_sipping:
            self._draw_sip_alert(frame)

        video_frame = self.video_player.get_frame((frame.shape[1], frame.shape[0]))
        if video_frame is not None:
            frame = cv2.addWeighted(frame, 0.3, video_frame, 0.7, 0)

        if self.config.get("show_hsv_mask"):
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_small = cv2.resize(mask_colored, (320, 240))
            frame[10:250, frame.shape[1] - 330:frame.shape[1] - 10] = mask_small

        return frame, detection

    def run(self):
        cap = cv2.VideoCapture(int(self.config.get("camera_index") or 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("window_width") or 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("window_height") or 720))

        if not cap.isOpened():
            print("[ERROR] Could not open camera!")
            return

        print("[INFO] Camera opened successfully")
        print("[INFO] Hold up your white Monster can and take a sip!")
        print("[INFO] Press 'Q' to quit\n")

        self.running = True
        window_name = "Monster Sip Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Bigger display window (independent of camera resolution)
        dw = int(self.config.get("display_width") or 1600)
        dh = int(self.config.get("display_height") or 900)
        cv2.resizeWindow(window_name, dw, dh)

        while self.running:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            processed_frame, detection = self.process_frame(frame)

            self.frame_times.append(time.time() - frame_start)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            avg = sum(self.frame_times) / max(1, len(self.frame_times))
            self.fps = 1.0 / avg if avg > 0 else 0.0

            cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                self.running = False
            elif key == ord('d'):
                cur = bool(self.config.get("debug_mode"))
                self.config.set("debug_mode", not cur)
                print(f"[INFO] Debug mode: {'ON' if not cur else 'OFF'}")
            elif key == ord('m'):
                cur = bool(self.config.get("show_hsv_mask"))
                self.config.set("show_hsv_mask", not cur)
                print(f"[INFO] HSV mask: {'ON' if not cur else 'OFF'}")
            elif key == ord('r'):
                self.sip_count = 0
                print("[INFO] Sip count reset")
            elif key == ord('t'):
                self.video_player.trigger()
                self.sip_count += 1

        cap.release()
        cv2.destroyAllWindows()
        self.face_detector.release()
        self.video_player.release()

        print(f"\n[STATS] Total sips detected: {self.sip_count}")
        print("[INFO] ayo")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monster Sip Detector")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")

    args = parser.parse_args()

    detector = MonsterSipDetector(args.config)

    if args.debug:
        detector.config.set("debug_mode", True)
    if args.camera != 0:
        detector.config.set("camera_index", args.camera)

    try:
        detector.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")


if __name__ == "__main__":
    main()
