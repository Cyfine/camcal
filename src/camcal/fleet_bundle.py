"""Joint camera/marker pose estimation from static ArUco observations."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from . import se3
from .aruco import square_object_points
from .fleet import FleetExtrinsics, load_fleet, save_fleet
from .marker_map import MarkerManifest
from .observation_graph import ObservationGraph, require_connected_graph
from .observations import MarkerObservation, ObservationDataset


@dataclass(frozen=True)
class ResidualStats:
    corner_count: int
    mean_px: float
    rmse_px: float
    max_px: float


@dataclass(frozen=True)
class _PoseCandidate:
    pose_C_T_M: np.ndarray
    reprojection_rmse_px: float


@dataclass(frozen=True)
class BundleAdjustmentResult:
    success: bool
    message: str
    anchor_marker_id: int
    camera_poses_W_T_C: dict[str, np.ndarray]
    marker_poses_W_T_M: dict[int, np.ndarray]
    marker_side_lengths_m: dict[int, float]
    global_residual: ResidualStats
    per_camera_residual: dict[str, ResidualStats]
    per_marker_residual: dict[int, ResidualStats]
    per_edge_residual: dict[tuple[str, int], ResidualStats]
    graph: ObservationGraph
    initial_raw_cost: float
    final_raw_cost: float
    optimizer_cost: float
    optimizer_evaluations: int
    loss: str
    min_corner_depth_m: float
    observations: tuple[MarkerObservation, ...]
    room_structure: RoomStructure | None
    structure_diagnostics: dict[str, Any] | None


@dataclass(frozen=True)
class RoomStructure:
    """Parallel wall planes, all perpendicular to one floor-marker plane."""

    floor_marker_id: int
    parallel_wall_marker_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.parallel_wall_marker_groups:
            raise ValueError("parallel_wall_marker_groups must not be empty")
        if any(not group for group in self.parallel_wall_marker_groups):
            raise ValueError("wall marker groups must not be empty")
        wall_ids = [item for group in self.parallel_wall_marker_groups for item in group]
        if self.floor_marker_id in wall_ids:
            raise ValueError("floor marker cannot also be a wall marker")
        if len(wall_ids) != len(set(wall_ids)):
            raise ValueError("each wall marker must appear in exactly one group")


@dataclass(frozen=True)
class TableFrameSolution:
    """The table pose recovered from one marker rigidly attached to it."""

    table_marker_id: int
    table_T_marker: np.ndarray
    room_T_table: np.ndarray
    table_T_room: np.ndarray


def solve_fleet_bundle(
    dataset: ObservationDataset,
    manifest: MarkerManifest,
    *,
    anchor_marker_id: int,
    room_structure: RoomStructure | None = None,
    loss: str = "soft_l1",
    f_scale_px: float = 2.0,
    max_nfev: int = 400,
) -> BundleAdjustmentResult:
    """Estimate every observed camera and marker pose in the anchor-marker world.

    ``W_T_C`` maps camera-frame coordinates into the common world frame.
    ``W_T_M`` maps marker-local coordinates into that same world frame.
    The anchor marker pose is fixed to identity and is not optimized.
    """
    if dataset.dictionary != manifest.dictionary:
        raise ValueError(
            f"observation dictionary {dataset.dictionary!r} does not match "
            f"manifest {manifest.dictionary!r}"
        )
    if loss not in {"linear", "soft_l1", "huber", "cauchy"}:
        raise ValueError(f"unsupported robust loss {loss!r}")
    if not np.isfinite(f_scale_px) or f_scale_px <= 0:
        raise ValueError("f_scale_px must be positive and finite")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if not dataset.observations:
        raise ValueError("no marker observations to solve")

    observed_marker_ids = sorted({obs.marker_id for obs in dataset.observations})
    for marker_id in observed_marker_ids:
        marker = manifest.get(marker_id)
        if marker.measured_size_m is None:
            raise ValueError(
                f"marker {marker_id} has no measured_black_square_side_m; "
                "measure the print and update the manifest before solving"
            )

    graph = require_connected_graph(dataset, anchor_marker_id=anchor_marker_id)
    initial_cameras, initial_markers = initialize_fleet_poses(
        dataset,
        manifest,
        anchor_marker_id=anchor_marker_id,
    )
    camera_models = dataset.camera_by_name
    if room_structure is None:
        _decode, parameter_count = _unconstrained_decoder(
            initial_cameras, initial_markers, anchor_marker_id
        )
    else:
        _decode, parameter_count = _structured_room_decoder(
            initial_cameras,
            initial_markers,
            anchor_marker_id=anchor_marker_id,
            structure=room_structure,
        )

    def _residual_vector(x: np.ndarray) -> np.ndarray:
        cameras, markers = _decode(x)
        residuals: list[np.ndarray] = []
        for observation in dataset.observations:
            predicted, _ = _project_observation(
                observation,
                camera_models[observation.camera_name],
                manifest,
                cameras[observation.camera_name],
                markers[observation.marker_id],
            )
            residuals.append((predicted - observation.corners_px).reshape(-1))
        return np.concatenate(residuals)

    x0 = np.zeros(parameter_count, dtype=np.float64)
    initial_residual = _residual_vector(x0)
    optimized = least_squares(
        _residual_vector,
        x0,
        method="trf",
        loss=loss,
        f_scale=float(f_scale_px),
        max_nfev=int(max_nfev),
    )
    camera_poses, marker_poses = _decode(optimized.x)
    final_residual = _residual_vector(optimized.x)
    residuals_by_observation, depths = _observation_errors_and_depths(
        dataset,
        manifest,
        camera_poses,
        marker_poses,
    )
    diagnostics = _residual_diagnostics(dataset, residuals_by_observation)
    min_depth = float(np.min(np.concatenate(depths)))
    finite = all(
        np.all(np.isfinite(pose))
        for pose in (*camera_poses.values(), *marker_poses.values())
    )
    physically_valid = finite and min_depth > 0.0
    success = bool(optimized.success and physically_valid)
    message = str(optimized.message)
    if optimized.success and not physically_valid:
        message = "optimizer converged but at least one marker corner is behind a camera"

    return BundleAdjustmentResult(
        success=success,
        message=message,
        anchor_marker_id=anchor_marker_id,
        camera_poses_W_T_C=camera_poses,
        marker_poses_W_T_M=marker_poses,
        marker_side_lengths_m={
            marker_id: manifest.get(marker_id).side_length_m
            for marker_id in observed_marker_ids
        },
        global_residual=diagnostics[0],
        per_camera_residual=diagnostics[1],
        per_marker_residual=diagnostics[2],
        per_edge_residual=diagnostics[3],
        graph=graph,
        initial_raw_cost=0.5 * float(initial_residual @ initial_residual),
        final_raw_cost=0.5 * float(final_residual @ final_residual),
        optimizer_cost=float(optimized.cost),
        optimizer_evaluations=int(optimized.nfev),
        loss=loss,
        min_corner_depth_m=min_depth,
        observations=dataset.observations,
        room_structure=room_structure,
        structure_diagnostics=(
            evaluate_room_structure(marker_poses, room_structure)
            if room_structure is not None
            else None
        ),
    )


def initialize_fleet_poses(
    dataset: ObservationDataset,
    manifest: MarkerManifest,
    *,
    anchor_marker_id: int,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Initialize world poses from edge PnP estimates and graph propagation."""
    require_connected_graph(dataset, anchor_marker_id=anchor_marker_id)
    camera_models = dataset.camera_by_name
    observations_by_edge: dict[tuple[str, int], list[MarkerObservation]] = defaultdict(list)
    for observation in dataset.observations:
        manifest.get(observation.marker_id)
        observations_by_edge[(observation.camera_name, observation.marker_id)].append(
            observation
        )

    C_T_M_by_edge: dict[tuple[str, int], np.ndarray] = {}
    for edge, observations in observations_by_edge.items():
        camera_name, marker_id = edge
        camera = camera_models[camera_name]
        object_points = square_object_points(manifest.get(marker_id).side_length_m)
        candidate_sets: list[tuple[_PoseCandidate, ...]] = []
        for observation in observations:
            candidates = _square_pnp_candidates(
                object_points,
                observation.corners_px,
                camera.K,
                camera.d,
            )
            if candidates:
                candidate_sets.append(candidates)
        if not candidate_sets:
            raise ValueError(
                f"PnP failed for every observation on edge "
                f"camera={camera_name!r}, marker={marker_id}"
            )
        C_T_M_by_edge[edge] = _select_consistent_edge_pose(
            candidate_sets,
            marker_side_length_m=manifest.get(marker_id).side_length_m,
        )

    incident_cameras: dict[int, list[str]] = defaultdict(list)
    incident_markers: dict[str, list[int]] = defaultdict(list)
    for camera_name, marker_id in C_T_M_by_edge:
        incident_cameras[marker_id].append(camera_name)
        incident_markers[camera_name].append(marker_id)

    cameras: dict[str, np.ndarray] = {}
    markers: dict[int, np.ndarray] = {anchor_marker_id: se3.identity()}
    queue: deque[tuple[str, str | int]] = deque([('marker', anchor_marker_id)])
    while queue:
        kind, value = queue.popleft()
        if kind == 'marker':
            marker_id = int(value)
            for camera_name in sorted(incident_cameras[marker_id]):
                if camera_name in cameras:
                    continue
                C_T_M = C_T_M_by_edge[(camera_name, marker_id)]
                cameras[camera_name] = se3.compose(
                    markers[marker_id], se3.inverse(C_T_M)
                )
                queue.append(('camera', camera_name))
        else:
            camera_name = str(value)
            for marker_id in sorted(incident_markers[camera_name]):
                if marker_id in markers:
                    continue
                C_T_M = C_T_M_by_edge[(camera_name, marker_id)]
                markers[marker_id] = se3.compose(cameras[camera_name], C_T_M)
                queue.append(('marker', marker_id))

    expected_cameras = {camera.name for camera in dataset.cameras}
    expected_markers = {obs.marker_id for obs in dataset.observations}
    if set(cameras) != expected_cameras or set(markers) != expected_markers:
        raise RuntimeError("graph propagation did not initialize every observed node")
    return cameras, markers


def save_fleet_bundle_result(
    path: str | Path,
    result: BundleAdjustmentResult,
    *,
    observation_paths: Iterable[str | Path] = (),
    marker_manifest_path: str | Path = "",
    table_frame: TableFrameSolution | None = None,
) -> None:
    """Export camera poses, marker map, provenance, and raw-pixel diagnostics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_images = sorted(
        {obs.source_image for obs in result.observations if obs.source_image}
    )
    body: dict[str, Any] = {
        "version": 1,
        "solver_status": "SOLVED" if result.success else "FAILED_TO_SOLVE",
        "message": result.message,
        "accuracy_acceptance": (
            "not evaluated; inspect raw reprojection residuals, source-image "
            "overlays, and the 3D scene before using these extrinsics"
        ),
        "world_frame": f"aruco_marker_{result.anchor_marker_id}",
        "anchor_marker_id": result.anchor_marker_id,
        "transform_convention": (
            "W_T_C maps camera-frame coordinates into marker-anchor world; "
            "W_T_M maps marker-local coordinates into the same world"
        ),
        "cameras": {
            name: {
                "W_T_C": pose.tolist(),
                "translation_m": pose[:3, 3].tolist(),
                "quaternion_xyzw": se3.to_quat_t(pose)[0].tolist(),
                "residual": _stats_to_dict(result.per_camera_residual[name]),
            }
            for name, pose in sorted(result.camera_poses_W_T_C.items())
        },
        "markers": {
            int(marker_id): {
                "W_T_M": pose.tolist(),
                "side_length_m": result.marker_side_lengths_m[marker_id],
                "residual": _stats_to_dict(result.per_marker_residual[marker_id]),
            }
            for marker_id, pose in sorted(result.marker_poses_W_T_M.items())
        },
        "diagnostics": {
            "global_residual": _stats_to_dict(result.global_residual),
            "per_edge": {
                f"{camera}|{marker_id}": _stats_to_dict(stats)
                for (camera, marker_id), stats in sorted(result.per_edge_residual.items())
            },
            "edge_observation_counts": {
                f"{camera}|{marker_id}": count
                for (camera, marker_id), count in sorted(result.graph.edge_counts.items())
            },
            "initial_raw_cost": result.initial_raw_cost,
            "final_raw_cost": result.final_raw_cost,
            "optimizer_cost": result.optimizer_cost,
            "optimizer_evaluations": result.optimizer_evaluations,
            "loss": result.loss,
            "min_corner_depth_m": result.min_corner_depth_m,
        },
        "provenance": {
            "observation_files": [str(item) for item in observation_paths],
            "marker_manifest": str(marker_manifest_path),
            "source_images": source_images,
            "marker_observation_count": len(result.observations),
        },
    }
    if table_frame is not None:
        body["cad_table_frame"] = {
            "frame": "cad_table",
            "room_frame": f"aruco_marker_{result.anchor_marker_id}",
            "table_marker_id": table_frame.table_marker_id,
            "transform_convention": (
                "R_T_T maps CAD-table coordinates into the room frame; "
                "T_T_R maps room coordinates into the CAD table frame; "
                "T_T_C maps camera coordinates into the CAD table frame"
            ),
            "T_T_M": table_frame.table_T_marker.tolist(),
            "R_T_T": table_frame.room_T_table.tolist(),
            "T_T_R": table_frame.table_T_room.tolist(),
            "cameras": {
                name: {
                    "T_T_C": se3.compose(table_frame.table_T_room, pose).tolist(),
                }
                for name, pose in sorted(result.camera_poses_W_T_C.items())
            },
            "markers": {
                int(marker_id): {
                    "T_T_M": se3.compose(table_frame.table_T_room, pose).tolist(),
                }
                for marker_id, pose in sorted(result.marker_poses_W_T_M.items())
            },
        }
    if result.room_structure is not None:
        body["room_structure"] = {
            "model": "parallel_walls_perpendicular_to_floor",
            "floor_marker_id": result.room_structure.floor_marker_id,
            "parallel_wall_marker_groups": [
                list(group)
                for group in result.room_structure.parallel_wall_marker_groups
            ],
            "diagnostics": result.structure_diagnostics,
        }
    path.write_text(yaml.safe_dump(body, sort_keys=False))


def save_calibrated_fleet(
    source_fleet_path: str | Path,
    output_fleet_path: str | Path,
    result: BundleAdjustmentResult,
    *,
    table_frame: TableFrameSolution | None = None,
) -> None:
    """Write a fleet copy whose observed cameras contain the solved ``W_T_C``."""
    if not result.success:
        raise ValueError("cannot export a calibrated fleet from a failed solve")
    fleet = load_fleet(source_fleet_path)
    if table_frame is None:
        fleet.world_frame = f"aruco_marker_{result.anchor_marker_id}"
    else:
        fleet.world_frame = "cad_table"
    for camera in fleet.cameras:
        if camera.name not in result.camera_poses_W_T_C:
            camera.extrinsics = None
    for name, pose in result.camera_poses_W_T_C.items():
        camera = fleet.get_camera(name)
        stats = result.per_camera_residual[name]
        output_pose = (
            pose
            if table_frame is None
            else se3.compose(table_frame.table_T_room, pose)
        )
        camera.extrinsics = FleetExtrinsics(
            detection_mode="aruco_marker_bundle",
            W_T_C=output_pose,
            reprojection_error_px_mean=stats.mean_px,
            reprojection_error_px_max=stats.max_px,
            notes=(
                "joint camera/marker bundle; world="
                + (
                    "cad_table"
                    if table_frame is not None
                    else f"marker {result.anchor_marker_id}"
                )
            ),
        )
    save_fleet(output_fleet_path, fleet)


def derive_table_frame(
    result: BundleAdjustmentResult,
    *,
    table_marker_id: int,
    table_T_marker: np.ndarray,
) -> TableFrameSolution:
    """Recover the table frame from a marker rigidly attached to the table.

    The bundle result lives in the room frame ``R``. ``table_T_marker`` is the
    separately known pose of the attached marker ``M`` in table frame ``T``.
    The identities are ``R_T_M = R_T_T T_T_M`` and therefore
    ``R_T_T = R_T_M inv(T_T_M)``.
    """
    if table_marker_id not in result.marker_poses_W_T_M:
        raise ValueError(
            f"table marker {table_marker_id} is not present in the solved bundle"
        )
    table_T_marker = np.asarray(table_T_marker, dtype=np.float64)
    se3.assert_se3(table_T_marker)
    room_T_marker = result.marker_poses_W_T_M[table_marker_id]
    room_T_table = se3.compose(room_T_marker, se3.inverse(table_T_marker))
    table_T_room = se3.inverse(room_T_table)
    return TableFrameSolution(
        table_marker_id=table_marker_id,
        table_T_marker=table_T_marker.copy(),
        room_T_table=room_T_table,
        table_T_room=table_T_room,
    )


def evaluate_room_structure(
    marker_poses: dict[int, np.ndarray],
    structure: RoomStructure,
) -> dict[str, Any]:
    """Measure plane-normal and coplanarity errors without changing poses."""
    floor_normal = _unit(marker_poses[structure.floor_marker_id][:3, 2])
    wall_ids = [
        marker_id
        for group in structure.parallel_wall_marker_groups
        for marker_id in group
    ]
    wall_floor = {
        str(marker_id): _perpendicular_error_deg(
            marker_poses[marker_id][:3, 2], floor_normal
        )
        for marker_id in wall_ids
    }
    wall_parallel: dict[str, float] = {}
    for index, first in enumerate(wall_ids):
        for second in wall_ids[index + 1 :]:
            wall_parallel[f"{first}|{second}"] = _parallel_error_deg(
                marker_poses[first][:3, 2], marker_poses[second][:3, 2]
            )
    coplanarity: dict[str, float] = {}
    for group in structure.parallel_wall_marker_groups:
        reference = group[0]
        normal = _unit(marker_poses[reference][:3, 2])
        origin = marker_poses[reference][:3, 3]
        for marker_id in group[1:]:
            coplanarity[f"{reference}|{marker_id}"] = abs(
                float(normal @ (marker_poses[marker_id][:3, 3] - origin))
            )
    return {
        "wall_floor_perpendicular_error_deg": wall_floor,
        "wall_parallel_error_deg": wall_parallel,
        "same_wall_center_plane_error_m": coplanarity,
    }


def _unconstrained_decoder(
    initial_cameras: dict[str, np.ndarray],
    initial_markers: dict[int, np.ndarray],
    anchor_marker_id: int,
):
    variable_keys = [
        *(("camera", name) for name in sorted(initial_cameras)),
        *(("marker", marker_id) for marker_id in sorted(initial_markers)
          if marker_id != anchor_marker_id),
    ]
    initial_by_key = {
        **{("camera", name): pose for name, pose in initial_cameras.items()},
        **{("marker", marker_id): pose for marker_id, pose in initial_markers.items()
           if marker_id != anchor_marker_id},
    }

    def decode(x: np.ndarray):
        cameras = {name: pose.copy() for name, pose in initial_cameras.items()}
        markers = {marker_id: pose.copy() for marker_id, pose in initial_markers.items()}
        for index, key in enumerate(variable_keys):
            pose = _left_increment(initial_by_key[key], x[index * 6 : (index + 1) * 6])
            if key[0] == "camera":
                cameras[str(key[1])] = pose
            else:
                markers[int(key[1])] = pose
        markers[anchor_marker_id] = se3.identity()
        return cameras, markers

    return decode, len(variable_keys) * 6


def _structured_room_decoder(
    initial_cameras: dict[str, np.ndarray],
    initial_markers: dict[int, np.ndarray],
    *,
    anchor_marker_id: int,
    structure: RoomStructure,
):
    if structure.floor_marker_id != anchor_marker_id:
        raise ValueError("floor_marker_id must equal the room anchor marker ID")
    structured_ids = {
        marker_id
        for group in structure.parallel_wall_marker_groups
        for marker_id in group
    }
    missing = sorted(structured_ids - set(initial_markers))
    if missing:
        raise ValueError(f"room-structure markers were not observed: {missing}")

    floor_pose = initial_markers[structure.floor_marker_id]
    floor_x = _unit(floor_pose[:3, 0])
    floor_y = _unit(floor_pose[:3, 1])
    floor_z = _unit(floor_pose[:3, 2])
    projected_normals: list[np.ndarray] = []
    for marker_id in sorted(structured_ids):
        normal = initial_markers[marker_id][:3, 2]
        projected = normal - floor_z * float(floor_z @ normal)
        projected_normals.append(_unit(projected))
    reference = projected_normals[0]
    aligned = [normal if float(normal @ reference) >= 0 else -normal
               for normal in projected_normals]
    shared_normal = _unit(np.sum(aligned, axis=0))
    theta0 = float(np.arctan2(shared_normal @ floor_y, shared_normal @ floor_x))

    group_offsets: list[float] = []
    marker_parameters: dict[int, tuple[float, float, float, float]] = {}
    tangent0 = _unit(np.cross(floor_z, shared_normal))
    for group in structure.parallel_wall_marker_groups:
        group_offsets.append(float(np.mean([
            shared_normal @ initial_markers[marker_id][:3, 3]
            for marker_id in group
        ])))
        for marker_id in group:
            pose = initial_markers[marker_id]
            sign = 1.0 if float(pose[:3, 2] @ shared_normal) >= 0 else -1.0
            normal = sign * shared_normal
            base = _wall_marker_base(normal, floor_z)
            relative = base.T @ pose[:3, :3]
            phi = float(np.arctan2(relative[1, 0], relative[0, 0]))
            marker_parameters[marker_id] = (
                float(tangent0 @ pose[:3, 3]),
                float(floor_z @ pose[:3, 3]),
                phi,
                sign,
            )

    camera_names = sorted(initial_cameras)
    free_marker_ids = sorted(
        set(initial_markers) - structured_ids - {anchor_marker_id}
    )
    camera_size = 6 * len(camera_names)
    free_size = 6 * len(free_marker_ids)
    structure_size = 1 + len(structure.parallel_wall_marker_groups) + 3 * len(structured_ids)

    def decode(x: np.ndarray):
        cameras = {name: pose.copy() for name, pose in initial_cameras.items()}
        markers = {marker_id: pose.copy() for marker_id, pose in initial_markers.items()}
        cursor = 0
        for name in camera_names:
            cameras[name] = _left_increment(initial_cameras[name], x[cursor:cursor + 6])
            cursor += 6
        for marker_id in free_marker_ids:
            markers[marker_id] = _left_increment(
                initial_markers[marker_id], x[cursor:cursor + 6]
            )
            cursor += 6
        theta = theta0 + float(x[cursor])
        cursor += 1
        wall_normal = _unit(np.cos(theta) * floor_x + np.sin(theta) * floor_y)
        tangent = _unit(np.cross(floor_z, wall_normal))
        offsets = [
            group_offsets[index] + float(x[cursor + index])
            for index in range(len(group_offsets))
        ]
        cursor += len(group_offsets)
        for group_index, group in enumerate(structure.parallel_wall_marker_groups):
            for marker_id in group:
                a0, b0, phi0, sign = marker_parameters[marker_id]
                a, b, phi = np.array([a0, b0, phi0]) + x[cursor:cursor + 3]
                cursor += 3
                position = offsets[group_index] * wall_normal + a * tangent + b * floor_z
                marker_normal = sign * wall_normal
                base = _wall_marker_base(marker_normal, floor_z)
                rotation = base @ Rotation.from_rotvec([0.0, 0.0, phi]).as_matrix()
                markers[marker_id] = se3.from_Rt(rotation, position)
        markers[anchor_marker_id] = se3.identity()
        return cameras, markers

    return decode, camera_size + free_size + structure_size


def _wall_marker_base(normal: np.ndarray, floor_normal: np.ndarray) -> np.ndarray:
    x_axis = floor_normal
    y_axis = _unit(np.cross(normal, floor_normal))
    return np.column_stack([x_axis, y_axis, normal])


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalize a near-zero geometry vector")
    return vector / norm


def _parallel_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip(abs(float(_unit(first) @ _unit(second))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _perpendicular_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip(abs(float(_unit(first) @ _unit(second))), 0.0, 1.0)
    return float(np.degrees(np.arcsin(cosine)))


def _left_increment(initial: np.ndarray, delta: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.asarray(delta[:3], dtype=np.float64)).as_matrix()
    increment = se3.from_Rt(rotation, np.asarray(delta[3:], dtype=np.float64))
    return se3.compose(increment, initial)


def _square_pnp_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
) -> tuple[_PoseCandidate, ...]:
    """Return every positive-depth IPPE square solution with its raw RMSE."""
    object_points = np.ascontiguousarray(object_points, dtype=np.float32).reshape(-1, 1, 3)
    image_points = np.ascontiguousarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        np.asarray(K, dtype=np.float64),
        np.asarray(d, dtype=np.float64).reshape(-1),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not count:
        return ()
    candidates: list[_PoseCandidate] = []
    object_points_3d = object_points.reshape(-1, 3).astype(np.float64)
    observed = image_points.reshape(-1, 2).astype(np.float64)
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        pose = se3.from_rvec_tvec(rvec, tvec)
        points_camera = (
            (pose[:3, :3] @ object_points_3d.T).T + pose[:3, 3]
        )
        if np.any(points_camera[:, 2] <= 0.0):
            continue
        projected, _ = cv2.projectPoints(
            object_points_3d.reshape(-1, 1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64),
            np.asarray(d, dtype=np.float64).reshape(-1),
        )
        difference = projected.reshape(-1, 2) - observed
        rmse = float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))
        candidates.append(_PoseCandidate(pose, rmse))
    candidates.sort(key=lambda candidate: candidate.reprojection_rmse_px)
    return tuple(candidates)


def _select_consistent_edge_pose(
    candidate_sets: list[tuple[_PoseCandidate, ...]],
    *,
    marker_side_length_m: float,
) -> np.ndarray:
    """Choose one coherent planar-PnP branch per repeated static observation.

    Both IPPE branches are retained until this step.  Each candidate acts once
    as a cluster seed; raw reprojection selects the branch and within-edge pose
    dispersion breaks ties.  This prevents a Karcher mean from blending two
    planar mirror branches.
    """
    if len(candidate_sets) == 1:
        return candidate_sets[0][0].pose_C_T_M
    best_key: tuple[float, float] | None = None
    best_poses: list[np.ndarray] | None = None
    seeds = [candidate for candidates in candidate_sets for candidate in candidates]
    for seed in seeds:
        chosen = [
            min(
                candidates,
                key=lambda candidate: _pose_cluster_distance(
                    candidate.pose_C_T_M,
                    seed.pose_C_T_M,
                    marker_side_length_m,
                ),
            )
            for candidates in candidate_sets
        ]
        mean_pose = se3.mean_se3(candidate.pose_C_T_M for candidate in chosen)
        dispersion = sum(
            _pose_cluster_distance(
                candidate.pose_C_T_M,
                mean_pose,
                marker_side_length_m,
            ) ** 2
            for candidate in chosen
        )
        reprojection = sum(
            candidate.reprojection_rmse_px ** 2 for candidate in chosen
        )
        key = (float(reprojection), float(dispersion))
        if best_key is None or key < best_key:
            best_key = key
            best_poses = [candidate.pose_C_T_M for candidate in chosen]
    assert best_poses is not None
    return se3.mean_se3(best_poses)


def _pose_cluster_distance(
    first: np.ndarray,
    second: np.ndarray,
    marker_side_length_m: float,
) -> float:
    translation = se3.translation_distance(first, second) / marker_side_length_m
    rotation = np.deg2rad(se3.rotation_distance(first, second))
    return float(np.hypot(translation, rotation))


def _project_observation(
    observation,
    camera,
    manifest: MarkerManifest,
    W_T_C: np.ndarray,
    W_T_M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return project_marker_corners(
        camera,
        manifest,
        observation.marker_id,
        W_T_C,
        W_T_M,
    )


def project_marker_corners(
    camera,
    manifest: MarkerManifest,
    marker_id: int,
    W_T_C: np.ndarray,
    W_T_M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one solved marker into one camera; return pixels and depths."""
    C_T_M = se3.compose(se3.inverse(W_T_C), W_T_M)
    object_points = square_object_points(
        manifest.get(marker_id).side_length_m
    )
    points_camera = (
        (C_T_M[:3, :3] @ object_points.T).T + C_T_M[:3, 3]
    )
    rvec, tvec = se3.to_rvec_tvec(C_T_M)
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        camera.K,
        camera.d,
    )
    return projected.reshape(4, 2), points_camera[:, 2]


def _observation_errors_and_depths(
    dataset: ObservationDataset,
    manifest: MarkerManifest,
    camera_poses: dict[str, np.ndarray],
    marker_poses: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    errors: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    camera_models = dataset.camera_by_name
    for observation in dataset.observations:
        predicted, depth = _project_observation(
            observation,
            camera_models[observation.camera_name],
            manifest,
            camera_poses[observation.camera_name],
            marker_poses[observation.marker_id],
        )
        errors.append(np.linalg.norm(predicted - observation.corners_px, axis=1))
        depths.append(depth)
    return errors, depths


def _residual_diagnostics(
    dataset: ObservationDataset,
    errors_by_observation: list[np.ndarray],
) -> tuple[
    ResidualStats,
    dict[str, ResidualStats],
    dict[int, ResidualStats],
    dict[tuple[str, int], ResidualStats],
]:
    per_camera: dict[str, list[float]] = defaultdict(list)
    per_marker: dict[int, list[float]] = defaultdict(list)
    per_edge: dict[tuple[str, int], list[float]] = defaultdict(list)
    all_errors: list[float] = []
    for observation, errors in zip(
        dataset.observations, errors_by_observation, strict=True
    ):
        values = [float(value) for value in errors]
        all_errors.extend(values)
        per_camera[observation.camera_name].extend(values)
        per_marker[observation.marker_id].extend(values)
        per_edge[(observation.camera_name, observation.marker_id)].extend(values)
    return (
        _stats(all_errors),
        {key: _stats(values) for key, values in sorted(per_camera.items())},
        {key: _stats(values) for key, values in sorted(per_marker.items())},
        {key: _stats(values) for key, values in sorted(per_edge.items())},
    )


def _stats(values: Iterable[float]) -> ResidualStats:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return ResidualStats(0, float('nan'), float('nan'), float('nan'))
    return ResidualStats(
        corner_count=int(array.size),
        mean_px=float(np.mean(array)),
        rmse_px=float(np.sqrt(np.mean(array * array))),
        max_px=float(np.max(array)),
    )


def _stats_to_dict(stats: ResidualStats) -> dict[str, int | float]:
    return {
        "corner_count": stats.corner_count,
        "mean_px": stats.mean_px,
        "rmse_px": stats.rmse_px,
        "max_px": stats.max_px,
    }


__all__ = [
    "BundleAdjustmentResult",
    "ResidualStats",
    "RoomStructure",
    "TableFrameSolution",
    "derive_table_frame",
    "evaluate_room_structure",
    "initialize_fleet_poses",
    "project_marker_corners",
    "save_calibrated_fleet",
    "save_fleet_bundle_result",
    "solve_fleet_bundle",
]
