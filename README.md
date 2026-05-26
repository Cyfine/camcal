# camcal

ChArUco-based camera calibration for intrinsics and world-frame extrinsics.

Three scripts, one workflow:

1. **`camcal-list-cameras`** — discover connected cameras and print stable
   `--device` paths to paste into the calibration scripts.
2. **`camcal-intrinsics`** — Zhang's-method recovery of `K` and lens distortion
   from N views of a ChArUco board.
3. **`camcal-extrinsics`** — pose of a camera in a world frame defined by a
   stuck-on ChArUco board (computes `W_T_C`).

`camcal-intrinsics` and `camcal-extrinsics` work either from pre-captured image
files or live from a webcam.

## Install

```bash
cd packages/camcal
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

Linux-only at present; on macOS/Windows the utility prints a one-line hint
and you pass integer indices directly to `--device`.

### Step 1 — Print and measure the board

Generate a printable PNG from a board declaration:

```bash
cd examples
python generate_board.py board.yaml --out board.png --dpi 300
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

**Live mode:**
```bash
camcal-intrinsics --board examples/board.yaml \
                  --live --device 0 --width 1920 --height 1080 \
                  --out intrinsics.yaml
```
SPACE captures the current frame, ENTER finalises once you have ≥15 captures,
ESC aborts.

**From pre-captured files:**
```bash
camcal-intrinsics --board examples/board.yaml \
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

Stick the ChArUco board flat at the location you want to call the **world
origin**. The board defines the world frame: its corners are coordinates,
its plane is `Z = 0`, and the convention follows OpenCV's `(rvec, tvec)`
(origin at the first chessboard corner, board lying in the `XY` plane,
`+Z` out of the board).

For each camera in turn, aim it so the entire board is in view and the
detection is stable. Use the `--device <by-id-path>` line you copied from
Step 0; on a multi-camera rig, this is the difference between calibrating
the right camera and silently calibrating the wrong one after a reboot.

Then either:

**Single image (one-shot):**
```bash
camcal-extrinsics --board examples/board.yaml \
                  --intrinsics intrinsics.yaml \
                  --image world_view.png \
                  --out cam0_extrinsic.yaml
```

**Multiple images, averaged:**
```bash
camcal-extrinsics --board examples/board.yaml \
                  --intrinsics intrinsics.yaml \
                  --images "world_views/*.png" \
                  --out cam0_extrinsic.yaml
```

**Live capture, averaged across N captures:**
```bash
camcal-extrinsics --board examples/board.yaml \
                  --intrinsics intrinsics.yaml \
                  --live --device 0 --num-frames 10 \
                  --out cam0_extrinsic.yaml
```

Repeat per camera. All cameras you measure against the same physically-placed
board end up expressed in a common world frame, with no need for them to see
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
from camcal.intrinsics import calibrate_intrinsics
from camcal.pnp import solve_pnp, reprojection_error_px
```

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
board:                          # snapshot of the board defining the world frame
  squares_x: 5
  ...
intrinsics_path: intrinsics.yaml
created_at: 2026-05-26T12:35:10+00:00
```

## Troubleshooting

**"no board detected" repeatedly during live capture.**
Check lighting (avoid glare across the board), focus, and that the
dictionary in `board.yaml` matches the printed board.

**Reprojection error > 1.5 px after intrinsic calibration.**
Most often: the board is not flat, the printed size is off (re-measure with
calipers), or the views are too similar (vary tilt and distance more).

**Extrinsics drift between cameras even though the board is stationary.**
Usually a sign that one of the intrinsic calibrations is poor — rerun
`camcal-intrinsics` for the suspect camera with more varied views.

## Tests

```bash
pip install -e .[dev]
pytest -p no:cov tests/
```

The tests are synthetic — they generate planar grids, project them
through known camera parameters, and verify the recovery is within
tolerance. There is no live-camera test.

## License

MIT.
