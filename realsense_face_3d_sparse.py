from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import cv2
import matplotlib.pyplot as plt
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
WINDOW_NAME = "RealSense MediaPipe Sparse 3D Head"
RGB_WINDOW_NAME = "RealSense RGB Face Detection - press q to quit"

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
OUTER_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

FEATURES = [
    ("left eye", [LEFT_EYE], (0.0, 0.85, 1.0)),
    ("right eye", [RIGHT_EYE], (0.0, 0.85, 1.0)),
    ("left eyebrow", [LEFT_EYEBROW], (1.0, 0.85, 0.0)),
    ("right eyebrow", [RIGHT_EYEBROW], (1.0, 0.85, 0.0)),
    ("mouth", [OUTER_MOUTH, INNER_MOUTH], (1.0, 0.15, 0.1)),
]


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading face landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_realsense_pipeline() -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline, rs.align(rs.stream.color)


def landmark_pixels(face_landmarks: list, width: int, height: int) -> list[tuple[int, int]]:
    pixels = []
    for landmark in face_landmarks:
        x = int(round(landmark.x * (width - 1)))
        y = int(round(landmark.y * (height - 1)))
        pixels.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1)))
    return pixels


def face_bounds(pixels: list[tuple[int, int]], width: int, height: int) -> tuple[int, int, int, int]:
    xs = np.array([pixel[0] for pixel in pixels])
    ys = np.array([pixel[1] for pixel in pixels])
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    pad_x = int((x_max - x_min) * 0.25)
    pad_y = int((y_max - y_min) * 0.35)
    return (
        max(0, x_min - pad_x),
        min(width - 1, x_max + pad_x),
        max(0, y_min - pad_y),
        min(height - 1, y_max + pad_y),
    )


def deproject_pixel(pixel: tuple[int, int], depth_frame: rs.depth_frame, intrinsics: rs.intrinsics) -> list[float] | None:
    x, y = pixel
    depth = depth_frame.get_distance(x, y)
    if depth <= 0.0:
        depth = nearby_depth(depth_frame, x, y)
    if depth <= 0.0:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return [point[0], -point[1], -point[2]]


def nearby_depth(depth_frame: rs.depth_frame, x: int, y: int, radius: int = 4) -> float:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    depths = []
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            depth = depth_frame.get_distance(xx, yy)
            if depth > 0.0:
                depths.append(depth)
    return float(np.median(depths)) if depths else 0.0


def sparse_head_cloud(
    color_frame: np.ndarray,
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
    bounds: tuple[int, int, int, int],
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = bounds
    points = []
    colors = []
    depth_samples = []

    for y in range(y_min, y_max + 1, stride):
        for x in range(x_min, x_max + 1, stride):
            depth = depth_frame.get_distance(x, y)
            if depth > 0.0:
                depth_samples.append(depth)

    if not depth_samples:
        return np.empty((0, 3)), np.empty((0, 3))

    median_depth = float(np.median(depth_samples))
    min_depth = max(0.15, median_depth - 0.18)
    max_depth = median_depth + 0.18

    for y in range(y_min, y_max + 1, stride):
        for x in range(x_min, x_max + 1, stride):
            depth = depth_frame.get_distance(x, y)
            if depth <= 0.0 or depth < min_depth or depth > max_depth:
                continue

            point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
            points.append([point[0], -point[1], -point[2]])
            colors.append(color_frame[y, x, ::-1] / 255.0)

    return np.asarray(points), np.asarray(colors)


def make_feature_lines(
    pixels: list[tuple[int, int]],
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
) -> list[tuple[str, np.ndarray, tuple[float, float, float]]]:
    feature_lines = []
    for label, polylines, color in FEATURES:
        label_used = False

        for polyline in polylines:
            points = []
            for landmark_index in polyline:
                point = deproject_pixel(pixels[landmark_index], depth_frame, intrinsics)
                if point is None:
                    if len(points) > 1:
                        line_label = label if not label_used else "_nolegend_"
                        feature_lines.append((line_label, np.asarray(points), color))
                        label_used = True
                    points = []
                    continue

                points.append(point)

            if len(points) > 1:
                line_label = label if not label_used else "_nolegend_"
                feature_lines.append((line_label, np.asarray(points), color))
                label_used = True

    return feature_lines


def configure_axis(ax, view_angle: tuple[float, float] | None = None) -> None:
    ax.set_facecolor((0.02, 0.02, 0.025))
    ax.figure.set_facecolor((0.02, 0.02, 0.025))
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.set_xlabel("X meters")
    ax.set_ylabel("Y meters")
    ax.set_zlabel("Z meters")
    if view_angle is not None:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.grid(True, color=(0.25, 0.25, 0.25))


def set_equal_view(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        ax.set_xlim(-0.18, 0.18)
        ax.set_ylim(-0.22, 0.22)
        ax.set_zlim(-1.2, -0.4)
        return

    center = points.mean(axis=0)
    span = float(np.max(np.ptp(points, axis=0)))
    radius = max(span * 0.65, 0.18)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_scene(
    ax,
    points: np.ndarray,
    colors: np.ndarray,
    feature_lines: list[tuple[str, np.ndarray, tuple[float, float, float]]],
) -> None:
    view_angle = (ax.elev, ax.azim)
    ax.cla()
    configure_axis(ax, view_angle)

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5, depthshade=False)

    all_points = [points] if len(points) > 0 else []
    for label, line_points, color in feature_lines:
        ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            line_points[:, 2],
            color=color,
            linewidth=3,
            label=label,
        )
        all_points.append(line_points)

    if feature_lines:
        legend = ax.legend(loc="upper right")
        for text in legend.get_texts():
            text.set_color("white")
        legend.get_frame().set_facecolor((0.02, 0.02, 0.025))
        legend.get_frame().set_edgecolor((0.45, 0.45, 0.45))

    set_equal_view(ax, np.vstack(all_points) if all_points else np.empty((0, 3)))


def draw_2d_detection(display_image: np.ndarray, face_landmarks: list) -> None:
    for landmarks in face_landmarks:
        vision.drawing_utils.draw_landmarks(
            display_image,
            landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_tesselation_style(),
            landmark_drawing_spec=None,
        )
        vision.drawing_utils.draw_landmarks(
            display_image,
            landmarks,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            connection_drawing_spec=vision.drawing_styles.get_default_face_mesh_contours_style(),
            landmark_drawing_spec=None,
        )


def main() -> int:
    model_path = ensure_model()

    try:
        pipeline, align = create_realsense_pipeline()
    except RuntimeError as exc:
        print(f"Could not start RealSense D435i depth/color streams: {exc}")
        return 1

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    plt.ion()
    fig = plt.figure(WINDOW_NAME, figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=5, azim=-90)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(WINDOW_NAME)

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while plt.fignum_exists(fig.number):
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                timestamp_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                display_image = color_image.copy()
                draw_2d_detection(display_image, result.face_landmarks)
                cv2.imshow(RGB_WINDOW_NAME, display_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                points = np.empty((0, 3))
                colors = np.empty((0, 3))
                feature_lines = []

                if result.face_landmarks:
                    height, width = color_image.shape[:2]
                    pixels = landmark_pixels(result.face_landmarks[0], width, height)
                    bounds = face_bounds(pixels, width, height)
                    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

                    points, colors = sparse_head_cloud(color_image, depth_frame, intrinsics, bounds)
                    feature_lines = make_feature_lines(pixels, depth_frame, intrinsics)

                render_scene(ax, points, colors, feature_lines)
                fig.canvas.draw_idle()
                plt.pause(0.001)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
