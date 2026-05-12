from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("models/face_landmarker.task")


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading face landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_realsense_pipeline() -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def main() -> int:
    model_path = ensure_model()

    try:
        pipeline = create_realsense_pipeline()
    except RuntimeError as exc:
        print(f"Could not start RealSense D435i color stream: {exc}")
        return 1

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    print("Could not read a color frame from the RealSense camera.")
                    continue

                display_frame = np.asanyarray(color_frame.get_data()).copy()
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                for landmarks in result.face_landmarks:
                    vision.drawing_utils.draw_landmarks(
                        display_frame,
                        landmarks,
                        vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_tesselation_style(),
                        landmark_drawing_spec=None,
                    )
                    vision.drawing_utils.draw_landmarks(
                        display_frame,
                        landmarks,
                        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                        connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_contours_style(),
                        landmark_drawing_spec=None,
                    )

                cv2.imshow("MediaPipe Face Landmarker RealSense - press q to quit", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
