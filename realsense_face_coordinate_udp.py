from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
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
RGB_WINDOW_NAME = "RealSense RGB Face Detection - press q to quit"
SPARSE_WINDOW_NAME = "RealSense Sparse Face Coordinate Debug"
AXIS_LENGTH_METERS = 0.12
MIN_POSITION_ALPHA = 0.12
MAX_POSITION_ALPHA = 0.65
SLOW_POSITION_METERS = 0.003
FAST_POSITION_METERS = 0.04
MIN_ROTATION_ALPHA = 0.1
MAX_ROTATION_ALPHA = 0.55
SLOW_ROTATION_DEG = 1.0
FAST_ROTATION_DEG = 15.0
MAX_ROTATION_JUMP_DEG = 45.0
MAX_POSITION_STEP_METERS = 0.025
LANDMARK_DEPTH_OUTLIER_METERS = 0.035
MAX_MISSING_DEPTH_FRAMES = 10
MOUTH_OCCLUSION_NEAR_METERS = 0.045
MOUTH_DEPTH_DRIFT_METERS = 0.08

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE_TIP = [1, 4, 5]
OUTER_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
CAMERA_FORWARD = np.asarray([0.0, 0.0, 1.0], dtype=float)


@dataclass
class FaceFrame:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    mouth_center: np.ndarray


class FaceFrameFilter:
    def __init__(self) -> None:
        self.max_rotation_jump_rad = np.deg2rad(MAX_ROTATION_JUMP_DEG)
        self.previous: FaceFrame | None = None

    def reset(self) -> None:
        self.previous = None

    def current(self) -> FaceFrame | None:
        return self.previous

    def update(self, frame: FaceFrame) -> FaceFrame | None:
        if self.previous is None:
            self.previous = frame
            return frame

        translation_delta = frame.origin - self.previous.origin
        translation_jump = np.linalg.norm(translation_delta)
        rotation_jump = rotation_angle(rotation_matrix(frame), rotation_matrix(self.previous))
        if translation_jump > MAX_POSITION_STEP_METERS:
            frame = replace_frame_origin(
                frame,
                self.previous.origin + translation_delta / translation_jump * MAX_POSITION_STEP_METERS,
            )
            translation_jump = MAX_POSITION_STEP_METERS

        position_alpha = adaptive_alpha(
            translation_jump,
            SLOW_POSITION_METERS,
            FAST_POSITION_METERS,
            MIN_POSITION_ALPHA,
            MAX_POSITION_ALPHA,
        )
        rotation_alpha = adaptive_alpha(
            min(rotation_jump, self.max_rotation_jump_rad),
            np.deg2rad(SLOW_ROTATION_DEG),
            np.deg2rad(FAST_ROTATION_DEG),
            MIN_ROTATION_ALPHA,
            MAX_ROTATION_ALPHA,
        )
        origin = smooth_vector(self.previous.origin, frame.origin, position_alpha)
        x_axis = smooth_unit_vector(self.previous.x_axis, frame.x_axis, rotation_alpha)
        y_hint = smooth_unit_vector(self.previous.y_axis, frame.y_axis, rotation_alpha)
        if x_axis is None or y_hint is None:
            return self.previous

        z_axis = normalize(np.cross(x_axis, y_hint))
        if z_axis is None:
            return self.previous

        y_axis = normalize(np.cross(z_axis, x_axis))
        if y_axis is None:
            return self.previous

        filtered = FaceFrame(
            origin=origin,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            left_eye_center=smooth_vector(self.previous.left_eye_center, frame.left_eye_center, position_alpha),
            right_eye_center=smooth_vector(self.previous.right_eye_center, frame.right_eye_center, position_alpha),
            mouth_center=origin,
        )
        self.previous = filtered
        return filtered


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


def deproject_pixel(pixel: tuple[int, int], depth_frame: rs.depth_frame, intrinsics: rs.intrinsics) -> np.ndarray | None:
    x, y = pixel
    depth = depth_frame.get_distance(x, y)
    if depth <= 0.0:
        depth = nearby_depth(depth_frame, x, y)
    if depth <= 0.0:
        return None

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return np.asarray([point[0], point[1], point[2]], dtype=float)


def deproject_pixel_at_depth(pixel: tuple[float, float], depth: float, intrinsics: rs.intrinsics) -> np.ndarray:
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel[0], pixel[1]], depth)
    return np.asarray([point[0], point[1], point[2]], dtype=float)


def centroid_pixel(landmark_indices: list[int], pixels: list[tuple[int, int]]) -> tuple[float, float]:
    selected = np.asarray([pixels[landmark_index] for landmark_index in landmark_indices], dtype=float)
    center = np.median(selected, axis=0)
    return float(center[0]), float(center[1])


def mean_landmark_point(
    landmark_indices: list[int],
    pixels: list[tuple[int, int]],
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
) -> np.ndarray | None:
    points = []
    for landmark_index in landmark_indices:
        point = deproject_pixel(pixels[landmark_index], depth_frame, intrinsics)
        if point is not None:
            points.append(point)

    if len(points) < 3:
        return None

    points_array = np.asarray(points)
    median_z = float(np.median(points_array[:, 2]))
    depth_error = np.abs(points_array[:, 2] - median_z)
    inliers = points_array[depth_error <= LANDMARK_DEPTH_OUTLIER_METERS]
    if len(inliers) < 3:
        inliers = points_array

    return np.median(inliers, axis=0)


def mouth_point_with_face_depth_guard(
    pixels: list[tuple[int, int]],
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
    stable_face_depth: float,
) -> np.ndarray | None:
    raw_mouth_center = mean_landmark_point(OUTER_MOUTH + INNER_MOUTH, pixels, depth_frame, intrinsics)
    mouth_pixel = centroid_pixel(OUTER_MOUTH + INNER_MOUTH, pixels)
    estimated_mouth_center = deproject_pixel_at_depth(mouth_pixel, stable_face_depth, intrinsics)
    if raw_mouth_center is None:
        return estimated_mouth_center

    depth_error = raw_mouth_center[2] - stable_face_depth
    if depth_error < -MOUTH_OCCLUSION_NEAR_METERS or abs(depth_error) > MOUTH_DEPTH_DRIFT_METERS:
        return estimated_mouth_center

    return raw_mouth_center


def replace_frame_origin(frame: FaceFrame, origin: np.ndarray) -> FaceFrame:
    origin_delta = origin - frame.origin
    return FaceFrame(
        origin=origin,
        x_axis=frame.x_axis,
        y_axis=frame.y_axis,
        z_axis=frame.z_axis,
        left_eye_center=frame.left_eye_center + origin_delta,
        right_eye_center=frame.right_eye_center + origin_delta,
        mouth_center=origin,
    )


def sparse_head_cloud(
    color_frame: np.ndarray,
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
    bounds: tuple[int, int, int, int],
    stride: int,
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
            points.append([point[0], point[1], point[2]])
            colors.append(color_frame[y, x, ::-1] / 255.0)

    return np.asarray(points), np.asarray(colors)


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return None
    return vector / norm


def smooth_vector(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    return previous * (1.0 - alpha) + current * alpha


def adaptive_alpha(
    change: float,
    slow_change: float,
    fast_change: float,
    min_alpha: float,
    max_alpha: float,
) -> float:
    blend = (change - slow_change) / (fast_change - slow_change)
    return float(min_alpha + np.clip(blend, 0.0, 1.0) * (max_alpha - min_alpha))


def smooth_unit_vector(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray | None:
    if np.dot(previous, current) < 0.0:
        current = -current
    return normalize(smooth_vector(previous, current, alpha))


def rotation_matrix(frame: FaceFrame) -> np.ndarray:
    return np.column_stack([frame.x_axis, frame.y_axis, frame.z_axis])


def rotation_angle(current: np.ndarray, previous: np.ndarray) -> float:
    delta = current @ previous.T
    cos_angle = (np.trace(delta) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def quaternion_from_matrix(matrix: np.ndarray) -> list[float]:
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def make_face_frame(
    pixels: list[tuple[int, int]],
    depth_frame: rs.depth_frame,
    intrinsics: rs.intrinsics,
) -> FaceFrame | None:
    right_eye_center = mean_landmark_point(RIGHT_EYE, pixels, depth_frame, intrinsics)
    left_eye_center = mean_landmark_point(LEFT_EYE, pixels, depth_frame, intrinsics)
    nose_tip = mean_landmark_point(NOSE_TIP, pixels, depth_frame, intrinsics)
    if right_eye_center is None or left_eye_center is None:
        return None

    stable_face_depth = float(np.median([right_eye_center[2], left_eye_center[2]]))
    mouth_center = mouth_point_with_face_depth_guard(pixels, depth_frame, intrinsics, stable_face_depth)
    if mouth_center is None:
        return None
    mouth_orientation_center = deproject_pixel_at_depth(
        centroid_pixel(OUTER_MOUTH + INNER_MOUTH, pixels),
        stable_face_depth,
        intrinsics,
    )

    # Requested face frame:
    #   +Y mouth -> eyes, +Z inward into the face, +X completes a right-handed frame.
    eye_midpoint = (right_eye_center + left_eye_center) * 0.5
    y_axis = normalize(eye_midpoint - mouth_orientation_center)
    if y_axis is None:
        return None

    if nose_tip is None:
        inward_hint = CAMERA_FORWARD
    else:
        face_plane_center = eye_midpoint * 0.65 + mouth_orientation_center * 0.35
        inward_hint = normalize(face_plane_center - nose_tip)
        if inward_hint is None:
            inward_hint = CAMERA_FORWARD

    z_axis = normalize(inward_hint - y_axis * np.dot(inward_hint, y_axis))
    if z_axis is None:
        z_axis = normalize(CAMERA_FORWARD - y_axis * np.dot(CAMERA_FORWARD, y_axis))
    if z_axis is None:
        return None

    x_axis = normalize(np.cross(y_axis, z_axis))
    if x_axis is None:
        return None

    y_axis = normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    return FaceFrame(
        origin=mouth_center,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        mouth_center=mouth_center,
    )


def frame_to_payload(frame: FaceFrame, sequence: int, timestamp_ms: int) -> dict:
    matrix = rotation_matrix(frame)
    return {
        "type": "mediapipe_face_frame",
        "sequence": sequence,
        "timestamp_ms": timestamp_ms,
        "origin": frame.origin.tolist(),
        "quaternion_xyzw": quaternion_from_matrix(matrix),
    }


def zero_frame_payload(sequence: int, timestamp_ms: int) -> dict:
    return {
        "type": "mediapipe_face_frame",
        "sequence": sequence,
        "timestamp_ms": timestamp_ms,
        "origin": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 0.0],
    }


def coordinate_axes(
    frame: FaceFrame | None,
) -> list[tuple[str, np.ndarray, np.ndarray, tuple[float, float, float]]]:
    if frame is None:
        return []

    origin = frame.origin
    return [
        ("face X", origin, origin + frame.x_axis * AXIS_LENGTH_METERS, (1.0, 0.1, 0.1)),
        ("face Y", origin, origin + frame.y_axis * AXIS_LENGTH_METERS, (0.1, 1.0, 0.1)),
        ("face Z", origin, origin + frame.z_axis * AXIS_LENGTH_METERS, (0.2, 0.45, 1.0)),
    ]


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


class DebugVisualizer:
    def __init__(self) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.plt.ion()
        self.fig = self.plt.figure(SPARSE_WINDOW_NAME, figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=15, azim=-70)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(SPARSE_WINDOW_NAME)

    def update(
        self,
        color_image: np.ndarray,
        face_landmarks: list,
        points: np.ndarray,
        colors: np.ndarray,
        frame: FaceFrame | None,
    ) -> bool:
        display_image = color_image.copy()
        draw_2d_detection(display_image, face_landmarks)
        cv2.imshow(RGB_WINDOW_NAME, display_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        if not self.plt.fignum_exists(self.fig.number):
            return False

        view_angle = (self.ax.elev, self.ax.azim)
        self.ax.cla()
        self.configure_axis(view_angle)

        all_points = []
        if len(points) > 0:
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5, depthshade=False)
            all_points.append(points)

        for label, start, end, color in coordinate_axes(frame):
            axis_points = np.vstack([start, end])
            self.ax.plot(
                axis_points[:, 0],
                axis_points[:, 1],
                axis_points[:, 2],
                color=color,
                linewidth=4,
                label=label,
            )
            self.ax.text(end[0], end[1], end[2], label.replace("face ", ""), color=color)
            all_points.append(axis_points)

        if frame is not None:
            legend = self.ax.legend(loc="upper right")
            for text in legend.get_texts():
                text.set_color("white")
            legend.get_frame().set_facecolor((0.02, 0.02, 0.025))
            legend.get_frame().set_edgecolor((0.45, 0.45, 0.45))

        self.set_equal_view(np.vstack(all_points) if all_points else np.empty((0, 3)))
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return True

    def configure_axis(self, view_angle: tuple[float, float]) -> None:
        self.ax.set_facecolor((0.02, 0.02, 0.025))
        self.fig.set_facecolor((0.02, 0.02, 0.025))
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")
        self.ax.set_xlabel("X meters")
        self.ax.set_ylabel("Y meters")
        self.ax.set_zlabel("Z meters")
        self.ax.view_init(elev=view_angle[0], azim=view_angle[1])
        self.ax.grid(True, color=(0.25, 0.25, 0.25))

    def set_equal_view(self, points: np.ndarray) -> None:
        if len(points) == 0:
            self.ax.set_xlim(-0.18, 0.18)
            self.ax.set_ylim(-0.18, 0.18)
            self.ax.set_zlim(0.4, 1.2)
            return

        center = points.mean(axis=0)
        span = float(np.max(np.ptp(points, axis=0)))
        radius = max(span * 0.65, 0.18)
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)

    def close(self) -> None:
        cv2.destroyAllWindows()
        self.plt.close(self.fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a filtered MediaPipe/RealSense face coordinate frame over UDP."
    )
    parser.add_argument("--udp-host", default="127.0.0.1", help="Destination host or Docker bridge IP.")
    parser.add_argument("--udp-port", type=int, default=5005, help="Destination UDP port.")
    parser.add_argument("--print-every", type=int, default=30, help="Print one status line every N sent frames. Use 0 to disable.")
    parser.add_argument("--visualize", action="store_true", help="Show RGB detection and sparse 3D debug windows.")
    parser.add_argument("--cloud-stride", type=int, default=5, help="Sparse debug cloud sampling stride in pixels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    frame_filter = FaceFrameFilter()
    destination = (args.udp_host, args.udp_port)
    sequence = 0
    missing_depth_frames = 0
    visualizer = DebugVisualizer() if args.visualize else None

    print(f"Sending filtered face coordinate frames to udp://{args.udp_host}:{args.udp_port}")
    print("Press Ctrl+C to stop.")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                while True:
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
                    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    points = np.empty((0, 3))
                    colors = np.empty((0, 3))
                    filtered_frame = None

                    if result.face_landmarks:
                        height, width = color_image.shape[:2]
                        pixels = landmark_pixels(result.face_landmarks[0], width, height)
                        frame = make_face_frame(pixels, depth_frame, intrinsics)
                        if frame is not None:
                            missing_depth_frames = 0
                            filtered_frame = frame_filter.update(frame)
                        else:
                            missing_depth_frames += 1
                            if missing_depth_frames <= MAX_MISSING_DEPTH_FRAMES:
                                filtered_frame = frame_filter.current()

                        if visualizer is not None:
                            bounds = face_bounds(pixels, width, height)
                            points, colors = sparse_head_cloud(
                                color_image,
                                depth_frame,
                                intrinsics,
                                bounds,
                                stride=args.cloud_stride,
                            )
                    else:
                        missing_depth_frames = MAX_MISSING_DEPTH_FRAMES + 1

                    if visualizer is not None:
                        keep_running = visualizer.update(
                            color_image,
                            result.face_landmarks,
                            points,
                            colors,
                            filtered_frame,
                        )
                        if not keep_running:
                            break

                    if filtered_frame is None:
                        frame_filter.reset()
                        payload = zero_frame_payload(sequence, timestamp_ms)
                    else:
                        payload = frame_to_payload(filtered_frame, sequence, timestamp_ms)
                    udp_socket.sendto((json.dumps(payload) + "\n").encode("utf-8"), destination)

                    sequence += 1
                    if args.print_every and sequence % args.print_every == 0:
                        origin = payload["origin"]
                        print(
                            f"sent={sequence} origin=({origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f})"
                        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pipeline.stop()
        if visualizer is not None:
            visualizer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
