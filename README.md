# camcal

ChArUco-based camera calibration for intrinsics and world-frame extrinsics.

Four scripts, one workflow:

1. **`camcal-list-cameras`** — discover connected cameras and print stable
   `--device` paths to paste into the calibration scripts.
2. **`camcal-inspect-cameras`** — open all discovered cameras in one tiled
   viewport so the operator can verify identity, framing, and stream health.
3. **`camcal-intrinsics`** — Zhang's-method recovery of `K` and lens distortion
   from N views of a ChArUco board.
4. **`camcal-extrinsics`** — pose of a camera in a world frame defined by one
   planar target (computes `W_T_C`). Targets are explicit: a ChArUco board, or
   one standalone square ArUco marker with known dictionary, ID, and side
   length.

`camcal-intrinsics` and `camcal-extrinsics` work either from pre-captured image
files or live from a webcam.

## Install

```bash
cd camcal
pip install -e .            # editable install
# or
pip install -e .[dev]       # + pytest for the synthetic test suite
```

Dependencies: `numpy`, `opencv-contrib-python`, `pyyaml`, `scipy`. Python 3.10+.

## Workflow

### Step 0 — Discover your cameras

Before calibrating, find out which devices `cv2.VideoCapture` will see and
what their stable identifiers are:

```
$ camcal-list-cameras

Found 2 cameras:

[0] HD Pro Webcam C920 (Logitech, serial A1B2C3D4)
    --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0
    (cv2 index 0 → /dev/video0)

[1] HD Pro Webcam C920 (Logitech, serial E5F6G7H8)
    --device /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_E5F6G7H8-video-index0
    (cv2 index 1 → /dev/video2)

Tip: copy the --device line. by-id paths stay fixed across reboots
and USB-port swaps; integer indices do not.
```

For multi-camera rigs, **always copy the `--device <by-id-path>` line** into
the next two steps. Integer indices like `--device 0` will silently swap
between cameras on reboot.

Add `--probe` to open each device and verify it streams (useful when a node
is exposed but locked by another process). Add `--paths` to emit just the
by-id paths, one per line — handy for shell loops. Add `--json` for a
machine-readable list.

To inspect everything live before calibrating, open the tiled viewport:

```bash
camcal-inspect-cameras --width 1920 --height 1080
```

Press `ESC` or `q` to close the inspector.

Linux-only at present; on macOS/Windows the utility prints a one-line hint
and you pass integer indices directly to `--device`.

### Step 1 — Print and measure the board

Generate a printable PNG from a board declaration:

```bash
cd examples
python generate_board.py ../board.yaml --out board.png --dpi 300
```

Print at **100% scale** (uncheck "fit to page" / "scale to fit") onto rigid
white paper or cardstock. Mount it flat — a board that bows by a millimetre
will bias every measurement.

Then **caliper-measure** the printed board. The printer rarely yields exactly
the size declared in the YAML; the discrepancy directly biases every
calibration result. Measure across as many squares as you can with one
caliper reading, divide by the count, and update `square_size_m` and
`marker_size_m` in `board.yaml`.

### Step 2 — Camera intrinsics

Capture 15–30 views of the board from varied angles (tilts of ±25–40° in
both axes, board filling different parts of the frame).

**RealSense factory intrinsics:**
If you want the factory calibration stored on connected Intel RealSense
cameras instead of a ChArUco recalibration, use the RealSense API. This reads
the two attached D415 color streams at `1920x1080@30` and writes
`realsense_intrinsics.yaml`:

```bash
uv pip install pyrealsense2
uv run python - <<'PY'
import pyrealsense2 as rs
import yaml
from datetime import datetime, timezone

ctx = rs.context()
records = []

for dev in ctx.query_devices():
    target = None
    for sensor in dev.query_sensors():
        if sensor.get_info(rs.camera_info.name) != "RGB Camera":
            continue
        for profile in sensor.get_stream_profiles():
            if not profile.is_video_stream_profile():
                continue
            video = profile.as_video_stream_profile()
            if (
                profile.stream_type() == rs.stream.color
                and profile.format() == rs.format.bgr8
                and video.width() == 1920
                and video.height() == 1080
                and profile.fps() == 30
            ):
                target = video
                break
        if target is not None:
            break

    if target is None:
        continue

    intr = target.get_intrinsics()
    records.append({
        "serial": dev.get_info(rs.camera_info.serial_number),
        "model": dev.get_info(rs.camera_info.name),
        "product_id": dev.get_info(rs.camera_info.product_id),
        "firmware_version": dev.get_info(rs.camera_info.firmware_version),
        "distortion_model": str(intr.model).split(".")[-1],
        "K": [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        "d": list(intr.coeffs),
    })

body = {
    "source": "realsense_api",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "profile": {"stream": "color", "format": "bgr8", "fps": 30, "image_size": [1920, 1080]},
    "cameras": records,
}

with open("realsense_intrinsics.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(body, f, sort_keys=False)
PY
```

RealSense intrinsics are resolution-specific; use the same stream, format,
width, height, and frame rate when capturing frames later.

**Live mode:**
```bash
camcal-intrinsics --board board.yaml \
                  --live --device 0 --width 1920 --height 1080 \
                  --out intrinsics.yaml
```
SPACE captures the current frame, ENTER finalises once you have ≥15 captures,
ESC aborts. Live mode already includes an operator viewport with detection
feedback overlaid on the stream.

**From pre-captured files:**
```bash
camcal-intrinsics --board board.yaml \
                  --images "captures/*.png" \
                  --out intrinsics.yaml
```

The script prints the RMS reprojection error at the end. Rough guide:

| RMS px | Quality                                                         |
|--------|-----------------------------------------------------------------|
| < 0.5  | Excellent — the board, focus, and exposure are all healthy.     |
| < 1.0  | Acceptable for most downstream uses.                            |
| > 1.5  | Suspect — recheck board flatness, lighting, and view diversity. |

### Step 3 — World-frame extrinsics

Put one planar target flat at the location you want to call the **world
origin**. The target defines the world frame, its plane is `Z = 0`, and the
solver writes `W_T_C`, the camera pose in that world frame.

For a ChArUco board, the origin is the board's first chessboard corner. For a
standalone ArUco marker, the origin is the marker center. The marker corner
correspondences are solved as a single square marker pose using OpenCV's
square-marker PnP path, not by pretending the marker is part of a ChArUco
board.

For each camera in turn, aim it so the entire target is in view and the
detection is stable. Use the `--device <by-id-path>` line you copied from
Step 0; on a multi-camera rig, this is the difference between calibrating
the right camera and silently calibrating the wrong one after a reboot.

`camcal-extrinsics` supports two target types:

- ChArUco board target via `--board board.yaml`.
- Standalone square ArUco marker target via `--target-descriptor aruco.yaml`
  or explicit `--detector aruco --aruco-marker-id ... --aruco-size-mm ...`.

Then either:

**Single image (one-shot):**
```bash
camcal-extrinsics --board board.yaml \
                  --intrinsics intrinsics.yaml \
                  --image world_view.png \
                  --out cam0_extrinsic.yaml
```

**Multiple images, averaged:**
```bash
camcal-extrinsics --board board.yaml \
                  --intrinsics intrinsics.yaml \
                  --images "world_views/*.png" \
                  --out cam0_extrinsic.yaml
```

**Live capture, averaged across N captures:**
```bash
camcal-extrinsics --board board.yaml \
                  --intrinsics intrinsics.yaml \
                  --live --device 0 --num-frames 10 \
                  --out cam0_extrinsic.yaml
```

**RealSense live capture with factory intrinsics by default:**
If `./realsense_intrinsics.yaml` exists and `--device` resolves to an Intel
RealSense camera, `camcal-extrinsics` will automatically use that factory
bundle and lock the live stream to the recorded profile unless you explicitly
override `--intrinsics` with a different file.

```bash
camcal-extrinsics --board board.yaml \
                  --live --device /dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_840412062035-video-index0 \
                  --num-frames 10 \
                  --out realsense_cam0_extrinsic.yaml
```

**Standalone ArUco marker target:**
Descriptor:
```yaml
type: aruco_marker
dictionary: DICT_5X5_100
marker_id: 0
size_m: 0.1975
origin: center
```

`size_m` is the printed marker side length in meters. If you prefer entering
millimetres directly, skip the descriptor and use `--aruco-size-mm`.

Dictionary names use OpenCV's ArUco dictionaries. `DICT_5X5_100` contains IDs
`0..99`; if your printed marker ID is literally `100`, use a larger dictionary
such as `DICT_5X5_250` and make sure it matches the marker you printed.

```bash
camcal-extrinsics --target-descriptor aruco_marker.yaml \
                  --intrinsics realsense_intrinsics.yaml \
                  --camera-serial 840412062035 \
                  --image world_view_far.png \
                  --out cam0_extrinsic.yaml
```

Equivalent without a descriptor:
```bash
camcal-extrinsics --detector aruco \
                  --aruco-dictionary DICT_5X5_100 \
                  --aruco-marker-id 0 \
                  --aruco-size-mm 197.5 \
                  --intrinsics intrinsics.yaml \
                  --live --device 0 \
                  --out cam0_extrinsic.yaml
```

Repeat per camera. All cameras you measure against the same physically-placed
target end up expressed in a common world frame, with no need for them to see
each other directly.

## Loading results in Python

```python
from camcal.io_yaml import load_intrinsics, load_extrinsics

intr = load_intrinsics("intrinsics.yaml")
extr = load_extrinsics("cam0_extrinsic.yaml")

K, d = intr.K, intr.d          # (3, 3), (5,)
W_T_C = extr.W_T_C             # (4, 4) SE(3) — camera in world
```

The math primitives are also importable for advanced use:

```python
from camcal import se3
from camcal.charuco import CharucoBoard
from camcal.fleet import CameraFleetRecord, FleetCamera, FleetExtrinsics, FleetIntrinsics
from camcal.intrinsics import calibrate_intrinsics
from camcal.pnp import solve_pnp, reprojection_error_px
```

## Fleet Configs

For multi-camera rigs, keep a long-lived fleet file that names each camera and
stores its current intrinsics/extrinsics state in one place. The model lives in
[`camcal.fleet`](/home/ycheng/Developer/camcal/src/camcal/fleet.py), and a
sample file is provided at
[examples/fleet.example.yaml](/home/ycheng/Developer/camcal/examples/fleet.example.yaml).

Typical use:

```python
from camcal.fleet import CameraFleetRecord, FleetCamera, FleetExtrinsics, FleetIntrinsics, load_fleet, save_fleet
from camcal.io_yaml import load_extrinsics, load_intrinsics

fleet = CameraFleetRecord(name="alpha-rig", board_path="board.yaml")

intr = load_intrinsics("intrinsics/front-left.yaml")
extr = load_extrinsics("extrinsics/front-left.yaml")

fleet.upsert_camera(
    FleetCamera(
        name="front-left",
        device={"by_id_path": "/dev/v4l/by-id/usb-front-left"},
        intrinsics=FleetIntrinsics.from_record(intr, path="intrinsics/front-left.yaml"),
        extrinsics=FleetExtrinsics.from_record(extr, path="extrinsics/front-left.yaml"),
    )
)
save_fleet("fleet.yaml", fleet)
```

## Consumer Export

The current downstream consumer described in the referenced guide expects a
2-camera calibration directory with these exact filenames:

- `c0_params_color.yaml`
- `c2_params_color.yaml`
- `c0_to_c2_params_color.yaml`
- `camera0_pose.yaml`

Export that bundle directly from the fleet state with:

```bash
camcal-export-consumer \
  --fleet fleet.yaml \
  --camera0 front-left \
  --camera2 front-right \
  --out-dir consumer_calib
```

Semantics of the exported files:

- `c0_params_color.yaml` and `c2_params_color.yaml` contain OpenCV `K` and `D`
  matrices in `FileStorage` YAML.
- `c0_to_c2_params_color.yaml` contains the transform from camera0 frame to
  camera2 frame, so `X_cam2 = R * X_cam0 + T`.
- `camera0_pose.yaml` contains `W_T_C0`, the world-from-camera0 pose.

The exporter validates two important consumer constraints before writing:

- both cameras must resolve to the same calibration image size
- distortion must be a 5-coefficient OpenCV vector, matching the current
  consumer loader

## File formats

All artifacts are flat YAML for human inspection and easy diffing.

### `board.yaml` — board geometry (operator input)
```yaml
squares_x: 5
squares_y: 7
square_size_m: 0.0297          # caliper-measured side of one chess square
marker_size_m: 0.02225         # caliper-measured side of one ArUco marker
dictionary: DICT_5X5_100
```

### `intrinsics.yaml` — Zhang's-method output
```yaml
K:
  - [fx, 0, cx]
  - [0, fy, cy]
  - [0, 0, 1]
d: [k1, k2, p1, p2, k3]
image_size: [width, height]
reprojection_error_px: 0.42
n_views: 24
source: zhang
created_at: 2026-05-26T12:34:56+00:00
camera:                         # provenance of the camera that produced this
  by_id_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0
  dev_path: /dev/video0
  cv2_index: 0
  model: HD Pro Webcam C920
  vendor: Logitech
  serial: A1B2C3D4
# For files-mode runs (no live device), `camera` instead reads:
#   camera:
#     source: files
#     glob: "captures/*.png"
```

### `extrinsics.yaml` — world-frame pose
```yaml
W_T_C:                          # (4, 4) SE(3); camera in world
  - [...]
translation_m: [tx, ty, tz]     # convenience sidecar
quaternion_xyzw: [qx, qy, qz, qw]
n_frames: 10
reprojection_error_px_mean: 0.31
reprojection_error_px_max: 0.55
target:                         # snapshot of the target defining the world frame
  type: charuco_board
  squares_x: 5
  ...
intrinsics_path: intrinsics.yaml
detection_mode: charuco         # or aruco_marker
created_at: 2026-05-26T12:35:10+00:00
camera:                         # same shape as in intrinsics.yaml
  by_id_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_A1B2C3D4-video-index0
  dev_path: /dev/video0
  cv2_index: 0
  model: HD Pro Webcam C920
  vendor: Logitech
  serial: A1B2C3D4
```

For standalone ArUco marker extrinsics, the `target` block instead looks like:

```yaml
target:
  type: aruco_marker
  dictionary: DICT_5X5_100
  marker_id: 0
  size_m: 0.1975
  origin: center
detection_mode: aruco_marker
```

### `fleet.yaml` — stateful multi-camera rig config
```yaml
version: 1
name: alpha-rig
board_path: board.yaml
world_frame: world
cameras:
  - name: front-left
    enabled: true
    device:
      by_id_path: /dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_840412062035-video-index0
      serial: "840412062035"
    intrinsics:
      path: realsense_intrinsics.yaml
      source: realsense_api
      image_size: [1920, 1080]
      K:
        - [1394.09, 0.0, 972.03]
        - [0.0, 1393.01, 552.24]
        - [0.0, 0.0, 1.0]
    extrinsics:
      path: extrinsics/front-left.yaml
      detection_mode: aruco_marker
      W_T_C:
        - [1.0, 0.0, 0.0, 0.1]
        - [0.0, 1.0, 0.0, 0.0]
        - [0.0, 0.0, 1.0, 0.4]
        - [0.0, 0.0, 0.0, 1.0]
```

## Troubleshooting

**"no board detected" repeatedly during live capture.**
Check lighting (avoid glare across the board), focus, and that the
dictionary in `board.yaml` matches the printed board.

**Using a single ArUco marker as the world target.**
Use `--target-descriptor aruco_marker.yaml`, or pass `--detector aruco` with
`--aruco-marker-id` and `--aruco-size-mm`. This solves pose from that one
marker; it does not infer a ChArUco board from embedded markers. Ensure the
dictionary, marker ID, and measured side length match the printed marker.

**Reprojection error > 1.5 px after intrinsic calibration.**
Most often: the board is not flat, the printed size is off (re-measure with
calipers), or the views are too similar (vary tilt and distance more).

**Extrinsics drift between cameras even though the board is stationary.**
Usually a sign that one of the intrinsic calibrations is poor — rerun
`camcal-intrinsics` for the suspect camera with more varied views.

## Tests

```bash
pip install -e .[dev]
uv run --with pytest pytest -p no:cov tests/
```

The tests are synthetic — they generate planar grids, project them
through known camera parameters, and verify the recovery is within
tolerance. There is no live-camera test.

## License

MIT.
