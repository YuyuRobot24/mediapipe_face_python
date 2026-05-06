# MediaPipe Face Landmarker Demo

A small Python project for trying MediaPipe's Face Landmarker task with a webcam.

The demo opens your default camera, detects one face, and draws MediaPipe face mesh landmarks on the video feed.

## Files

- `webcam_face_landmarker.py` - webcam demo using MediaPipe Tasks Vision API
- `requirements.txt` - Python dependencies
- `models/face_landmarker.task` - downloaded automatically on first run

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python webcam_face_landmarker.py
```

Press `q` to close the webcam window.

On the first run, the script downloads the official MediaPipe face landmarker model into `models/`.

## Notes

If you see a message like this:

```text
QFontDatabase: Cannot find font directory .../cv2/qt/fonts
```

it comes from OpenCV's Qt window backend. If the webcam window works, you can ignore it. If the window does not open correctly, install system fonts such as:

```bash
sudo apt install fontconfig fonts-dejavu
```

## Git Ignore

The project ignores local/generated files such as:

- `.venv/`
- `.mplconfig/`
- `models/`
- `__pycache__/`

This keeps the GitHub repo small and lets each user install dependencies locally.
