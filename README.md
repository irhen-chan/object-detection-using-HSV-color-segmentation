# Monster Sip Detector 

**Real-time object detection using HSV color segmentation instead of pre-trained models — because sometimes the old ways still hit different.**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange.svg)

---

## The Idea

<img width="721" height="1292" alt="image" src="https://github.com/user-attachments/assets/ed1b0d95-5c58-4479-82e8-366bcf9c1aa7" />  
<img width="718" height="543" alt="image" src="https://github.com/user-attachments/assets/14ec791b-6882-413a-b18c-de1b2aa051cd" />



I've been getting into color grading while editing videos lately, and spending hours in HSV/HSL color wheels made me realize something: if I can isolate specific colors frame-by-frame for creative edits, why not use the same principles for object detection?

Everyone reaches for YOLO or pre-trained models these days (myself included), but I wanted to see how far traditional computer vision techniques could go. Could I detect a specific object, like the white Monster can I drink while coding, using nothing but color math and clever filtering?

Turns out, **yes**

---

## What It Does

Detects when you take a sip from a Monster Energy can and triggers a video overlay (with sound). The detection pipeline is entirely HSV-based with multi-stage filtering:

```
Webcam Frame
     ↓
┌─────────────────────────────────────────┐
│  Stage 1: HSV Color Segmentation        │
│  - Isolate white/silver color range     │
│  - Morphological cleanup (open/close)   │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Stage 2: Shape Filtering               │
│  - Contour area bounds                  │
│  - Aspect ratio (cans are tall)         │
│  - Fill ratio (contour vs bbox)         │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Stage 3: Texture Verification          │
│  - Dark pixel ratio (logo/text proxy)   │
│  - Edge density via Canny (graphics)    │
│  - Scoring function for best candidate  │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Stage 4: Sip Detection                 │
│  - MediaPipe Face Mesh (mouth tracking) │
│  - Euclidean distance: can → mouth      │
│  - Proximity threshold trigger          │
└─────────────────────────────────────────┘
     ↓
 Video + Audio Playback 🎬
```

---

## Why Not Just Use YOLO?

Good question. YOLO would probably work out of the box. But:

1. **Learning** — I wanted to actually understand what's happening under the hood instead of treating models as black boxes
2. **Lightweight** — No model weights to download, no GPU required, runs smooth on any machine
3. **Customizable** — Every parameter is tunable in real-time via the calibration tool
4. **The Challenge** — Constraints breed creativity. Turns out a white can in a messy room is a hard problem when you can't just throw a neural net at it

The texture verification stage (dark pixels + edge density) was the key insight — it filters out random white objects by checking if the region actually *looks* like a printed can.

---

## Demo

The app tracks the can, tracks your mouth via MediaPipe, calculates the distance between them, and triggers when you take a sip:

```
┌─────────────────────────────────────────┐
│  FPS: 32.5                              │
│  Can: DETECTED                          │
│  Mouth: OPEN                            │
│  Sips: 3                                │
│                                         │
│            ┌──────────┐                 │
│            │ MONSTER  │←─ Can bbox      │
│            │   CAN    │                 │
│            └────┬─────┘                 │
│                 │                       │
│              257px ←─ Distance          │
│                 │                       │
│                 ◯ ←─ Mouth position     │
│                                         │
└─────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/yourusername/monster-sip-detector.git
cd monster-sip-detector

pip install -r requirements.txt
```

### Dependencies
```
opencv-python==4.11.0
mediapipe==0.10.21
numpy==1.26.4
pygame>=2.5.0
```

---

## Usage

### Run the Detector
```bash
python monster_sip_detector.py
```

### Calibrate for Your Environment
The calibration tool is crucial — lighting conditions vary, and you'll need to tune the HSV range and texture filters for your setup:

```bash
python calibrate_hsv.py
```

This opens a multi-window interface with **17 trackbars** for real-time tuning:
- HSV range (H/S/V min & max)
- Area bounds
- Aspect ratio
- Fill ratio
- Dark pixel threshold & percentage
- Canny edge thresholds
- Morphology kernel size

Press `S` to save your calibration to `config.json`.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle debug overlay |
| `M` | Toggle HSV mask view |
| `R` | Reset sip counter |
| `T` | Manual trigger (testing) |
| `F` | Freeze frame (calibrator) |
| `S` | Save config (calibrator) |

---

## Configuration

All parameters live in `config.json`:

```json
{
    "can_hsv_lower": [76, 0, 0],
    "can_hsv_upper": [180, 40, 230],
    "min_can_area": 125,
    "max_can_area": 9988,
    "min_aspect_ratio": 1.25,
    "max_aspect_ratio": 8.0,
    "min_fill_pct": 0,
    "dark_thresh": 255,
    "min_dark_pct": 27,
    "min_edge_pct": 12,
    "sip_distance_threshold": 150,
    "sip_cooldown": 3.0
}
```

---

## Adding Your Own Video

Drop your video in the `assets/` folder and update the config:

```json
{
    "video_path": "assets/your_video.mp4",
    "audio_path": "assets/your_audio.mp3"
}
```

The video plays with synced audio when a sip is detected.

---

## Project Structure

```
monster-sip-detector/
├── monster_sip_detector.py  # Main application
├── calibrate_hsv.py         # Calibration tool (17 trackbars)
├── config.json              # All tunable parameters
├── requirements.txt
├── assets/
│   ├── flex_video.mp4       # Triggered video
│   └── your_audio.mp3       # Audio track
└── README.md
```

---

## What I Learned

- HSV is surprisingly powerful when you layer multiple verification stages
- The hardest part wasn't detection — it was *rejection* of false positives
- Texture analysis (dark pixels + edges) is a simple but effective way to distinguish "printed object" from "random white thing"
- MediaPipe Face Mesh is incredible for the price (free)
- Sometimes skipping the obvious solution (YOLO) forces you to learn more

---

## Tech Stack

- **OpenCV** — Image processing, HSV segmentation, morphology, Canny edges
- **MediaPipe** — Face Mesh for mouth landmark tracking
- **NumPy** — Array operations
- **Pygame** — Audio playback

---

## License

MIT — do whatever you want with it.

---
