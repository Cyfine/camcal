# Examples

## Generate a printable board

```bash
python generate_board.py ../board.yaml --out board.png --dpi 300
```

Open `board.png` in any image viewer or print client and print at **100% scale**
on rigid (or backed) white paper. Cardstock or photo paper minimises curl.

## Measure the printed board

Use calipers across as many squares as your board allows. Divide by the count
to get per-square size, then update `square_size_m` and `marker_size_m` in
`../board.yaml`.

## Run the calibration

From the repository root:

```bash
# Optional: inspect all discovered cameras in one tiled viewport first.
camcal-inspect-cameras --width 1920 --height 1080

# 1) Intrinsics — capture 15+ views from varied angles.
camcal-intrinsics --board board.yaml --live --device 0 --out intrinsics.yaml

# 2) Extrinsics — place the board where you want the world origin; per camera:
camcal-extrinsics --board board.yaml --intrinsics intrinsics.yaml \
                  --live --device 0 --out cam0_extrinsic.yaml

# 2b) RealSense live extrinsics can auto-use ./realsense_intrinsics.yaml:
camcal-extrinsics --board board.yaml \
                  --live --device /dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_840412062035-video-index0 \
                  --out realsense_cam0_extrinsic.yaml

# 3) Standalone ArUco marker target:
camcal-extrinsics --target-descriptor examples/aruco_marker.example.yaml \
                  --intrinsics realsense_intrinsics.yaml \
                  --camera-serial 840412062035 --image world_view_far.png \
                  --out cam0_extrinsic.yaml
```

Replace `--live --device 0` with `--images "captures/*.png"` (intrinsics) or
`--image world_view.png` (extrinsics) to work from pre-captured files.

## Manage the rig as state

See [fleet.example.yaml](/home/ycheng/Developer/camcal/examples/fleet.example.yaml)
for a stateful multi-camera rig config that records named cameras, their
stable device identity, intrinsics, extrinsics, and `W_T_C` transforms.

## Export to the consumer bundle

Once the fleet state is populated, export the exact `c0/c2` calibration bundle
the consumer runtime expects:

```bash
camcal-export-consumer \
  --fleet fleet.yaml \
  --camera0 front-left \
  --camera2 front-right \
  --out-dir consumer_calib
```

This writes:

- `consumer_calib/c0_params_color.yaml`
- `consumer_calib/c2_params_color.yaml`
- `consumer_calib/c0_to_c2_params_color.yaml`
- `consumer_calib/camera0_pose.yaml`

## Loading results in Python

```python
from camcal.io_yaml import load_intrinsics, load_extrinsics

intr = load_intrinsics("intrinsics.yaml")
extr = load_extrinsics("cam0_extrinsic.yaml")

K, d = intr.K, intr.d          # (3, 3), (5,)
W_T_C = extr.W_T_C             # (4, 4) SE(3) — camera in world
```
