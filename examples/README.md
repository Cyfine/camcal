# Examples

## Generate a printable board

```bash
python generate_board.py board.yaml --out board.png --dpi 300
```

Open `board.png` in any image viewer or print client and print at **100% scale**
on rigid (or backed) white paper. Cardstock or photo paper minimises curl.

## Measure the printed board

Use calipers across as many squares as your board allows. Divide by the count
to get per-square size, then update `square_size_m` and `marker_size_m` in
`board.yaml`.

## Run the calibration

From this directory:

```bash
# 1) Intrinsics — capture 15+ views from varied angles.
camcal-intrinsics --board board.yaml --live --device 0 --out intrinsics.yaml

# 2) Extrinsics — place the board where you want the world origin; per camera:
camcal-extrinsics --board board.yaml --intrinsics intrinsics.yaml \
                  --live --device 0 --out cam0_extrinsic.yaml
```

Replace `--live --device 0` with `--images "captures/*.png"` (intrinsics) or
`--image world_view.png` (extrinsics) to work from pre-captured files.

## Loading results in Python

```python
from camcal.io_yaml import load_intrinsics, load_extrinsics

intr = load_intrinsics("intrinsics.yaml")
extr = load_extrinsics("cam0_extrinsic.yaml")

K, d = intr.K, intr.d          # (3, 3), (5,)
W_T_C = extr.W_T_C             # (4, 4) SE(3) — camera in world
```
