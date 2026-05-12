# MediaPipe Face Landmarker Demo

A small Python project for trying MediaPipe's Face Landmarker task with a webcam or Intel RealSense D435i.

The demos open a camera, detect one face, and draw MediaPipe face mesh landmarks on the video feed.

## Files

- `webcam_face_landmarker.py` - webcam demo using MediaPipe Tasks Vision API
- `realsense_face_landmarker.py` - Intel RealSense D435i color stream demo
- `realsense_face_3d_sparse.py` - sparse RealSense depth reconstruction with colored 3D face feature lines
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

For an Intel RealSense D435i, connect the camera and run:

```bash
python realsense_face_landmarker.py
```

Press `q` to close the RealSense window.

For sparse 3D head reconstruction from the RealSense depth map, run:

```bash
python realsense_face_3d_sparse.py
```

Close the Matplotlib 3D window or press `q` in the RGB detection window to stop the renderer. Eyes are drawn in cyan, eyebrows in yellow, and the mouth in red.

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
