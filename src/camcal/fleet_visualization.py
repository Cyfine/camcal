"""Lightweight Plotly and image overlays for fleet calibration review."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .fleet_bundle import BundleAdjustmentResult, project_marker_corners
from .marker_map import MarkerManifest
from .observations import ObservationDataset


def write_fleet_calibration_html(
    path: str | Path,
    result: BundleAdjustmentResult,
    *,
    display_T_world: np.ndarray | None = None,
    display_frame_name: str | None = None,
) -> None:
    """Write one self-contained 3D pose and reprojection residual report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]],
        column_widths=[0.66, 0.34],
        subplot_titles=("Common calibration frame", "Per camera / marker RMSE", "Per edge RMSE"),
        vertical_spacing=0.14,
    )
    frame_transform = (
        np.eye(4, dtype=np.float64)
        if display_T_world is None
        else np.asarray(display_T_world, dtype=np.float64)
    )
    display_cameras = {
        name: frame_transform @ pose
        for name, pose in result.camera_poses_W_T_C.items()
    }
    display_markers = {
        marker_id: frame_transform @ pose
        for marker_id, pose in result.marker_poses_W_T_M.items()
    }
    scale = _scene_scale_from_poses(display_cameras, display_markers)
    _add_world_axes(figure, scale)
    for name, pose in sorted(display_cameras.items()):
        _add_camera(figure, name, pose, scale)
    for marker_id, pose in sorted(display_markers.items()):
        _add_marker(
            figure,
            marker_id,
            pose,
            result.marker_side_lengths_m[marker_id],
            anchor=marker_id == result.anchor_marker_id,
        )

    camera_names = sorted(result.per_camera_residual)
    marker_ids = sorted(result.per_marker_residual)
    figure.add_trace(
        go.Bar(
            x=[f"camera {name}" for name in camera_names],
            y=[result.per_camera_residual[name].rmse_px for name in camera_names],
            name="camera RMSE",
            marker_color="#3b82f6",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=[f"marker {marker_id}" for marker_id in marker_ids],
            y=[result.per_marker_residual[marker_id].rmse_px for marker_id in marker_ids],
            name="marker RMSE",
            marker_color="#10b981",
        ),
        row=1,
        col=2,
    )
    edge_items = sorted(result.per_edge_residual.items())
    figure.add_trace(
        go.Bar(
            x=[f"{camera} / M{marker_id}" for (camera, marker_id), _ in edge_items],
            y=[stats.rmse_px for _, stats in edge_items],
            name="edge RMSE",
            marker_color="#f59e0b",
        ),
        row=2,
        col=2,
    )
    figure.update_yaxes(title_text="pixels", row=1, col=2)
    figure.update_yaxes(title_text="pixels", row=2, col=2)
    figure.update_xaxes(tickangle=-35, row=1, col=2)
    figure.update_xaxes(tickangle=-35, row=2, col=2)
    figure.update_scenes(
        aspectmode="data",
        xaxis_title="world x [m]",
        yaxis_title="world y [m]",
        zaxis_title="world z [m]",
    )
    figure.update_layout(
        title=(
            f"Fleet calibration — {'SOLVED' if result.success else 'FAILED TO SOLVE'} — "
            f"global RMSE {result.global_residual.rmse_px:.3f} px — "
            f"frame = {display_frame_name or f'aruco_marker_{result.anchor_marker_id}'} "
            "— visual review required"
        ),
        height=850,
        showlegend=True,
        margin=dict(l=20, r=20, t=90, b=40),
    )
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)


def write_residual_overlays(
    output_directory: str | Path,
    result: BundleAdjustmentResult,
    dataset: ObservationDataset,
    manifest: MarkerManifest,
) -> list[Path]:
    """Draw observed corners, predictions, and residual vectors on source images."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for observation in dataset.observations:
        if observation.source_image:
            grouped[(observation.camera_name, observation.source_image)].append(observation)
    camera_models = dataset.camera_by_name
    written: list[Path] = []
    for (camera_name, source_image), observations in sorted(grouped.items()):
        frame = cv2.imread(source_image)
        if frame is None:
            continue
        for observation in observations:
            predicted, _ = project_marker_corners(
                camera_models[camera_name],
                manifest,
                observation.marker_id,
                result.camera_poses_W_T_C[camera_name],
                result.marker_poses_W_T_M[observation.marker_id],
            )
            observed_int = np.rint(observation.corners_px).astype(np.int32)
            predicted_int = np.rint(predicted).astype(np.int32)
            cv2.polylines(frame, [observed_int], True, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.polylines(frame, [predicted_int], True, (0, 0, 255), 2, cv2.LINE_AA)
            for observed, estimate in zip(observed_int, predicted_int, strict=True):
                cv2.line(
                    frame,
                    tuple(observed),
                    tuple(estimate),
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            errors = np.linalg.norm(predicted - observation.corners_px, axis=1)
            cv2.putText(
                frame,
                f"M{observation.marker_id} mean {np.mean(errors):.2f}px",
                tuple(observed_int[0] + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        source = Path(source_image)
        output_path = output_directory / f"{camera_name}_{source.stem}_residual.png"
        cv2.imwrite(str(output_path), frame)
        written.append(output_path)
    return written


def _scene_scale_from_poses(
    camera_poses: dict[str, np.ndarray],
    marker_poses: dict[int, np.ndarray],
) -> float:
    points = [pose[:3, 3] for pose in camera_poses.values()]
    points.extend(pose[:3, 3] for pose in marker_poses.values())
    array = np.asarray(points, dtype=np.float64)
    extent = float(np.max(np.ptp(array, axis=0))) if len(array) > 1 else 1.0
    return max(0.08, 0.12 * max(extent, 0.5))


def _add_world_axes(figure, scale: float) -> None:
    for vector, color, label in (
        ((scale, 0, 0), "red", "world +x"),
        ((0, scale, 0), "green", "world +y"),
        ((0, 0, scale), "blue", "world +z"),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=[0, vector[0]], y=[0, vector[1]], z=[0, vector[2]],
                mode="lines+text", text=["", label], line=dict(color=color, width=7),
                showlegend=False,
            ),
            row=1,
            col=1,
        )


def _add_camera(figure, name: str, W_T_C: np.ndarray, scale: float) -> None:
    local = np.asarray(
        [
            [0, 0, 0],
            [-0.75 * scale, -0.45 * scale, scale],
            [0.75 * scale, -0.45 * scale, scale],
            [0.75 * scale, 0.45 * scale, scale],
            [-0.75 * scale, 0.45 * scale, scale],
        ]
    )
    world = (W_T_C[:3, :3] @ local.T).T + W_T_C[:3, 3]
    segments = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for a, b in segments:
        x.extend([world[a, 0], world[b, 0], None])
        y.extend([world[a, 1], world[b, 1], None])
        z.extend([world[a, 2], world[b, 2], None])
    figure.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line=dict(width=4), name=f"camera {name}"
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter3d(
            x=[world[0, 0]], y=[world[0, 1]], z=[world[0, 2]],
            mode="markers+text", text=[name], textposition="top center",
            marker=dict(size=5), showlegend=False,
        ),
        row=1,
        col=1,
    )


def _add_marker(
    figure,
    marker_id: int,
    W_T_M: np.ndarray,
    side_length_m: float,
    *,
    anchor: bool,
) -> None:
    half = side_length_m / 2.0
    local = np.asarray(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0], [-half, half, 0]]
    )
    world = (W_T_M[:3, :3] @ local.T).T + W_T_M[:3, 3]
    figure.add_trace(
        go.Scatter3d(
            x=world[:, 0], y=world[:, 1], z=world[:, 2], mode="lines+markers",
            line=dict(color="#e11d48" if anchor else "#7c3aed", width=7 if anchor else 4),
            marker=dict(size=3), name=f"marker {marker_id}{' (world)' if anchor else ''}",
        ),
        row=1,
        col=1,
    )


__all__ = ["write_fleet_calibration_html", "write_residual_overlays"]
