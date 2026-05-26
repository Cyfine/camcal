"""Discover connected V4L2 cameras and print paste-ready ``--device`` snippets.

Walks ``/sys/class/video4linux/`` as the authoritative source for each
camera's model, manufacturer, vendor/product IDs, and (when the device
exposes one) USB serial. Cross-references ``/dev/v4l/by-id/`` to recover
the stable ``--device`` path the operator should copy. Prints one block
per camera; the line you copy is the headline.

Linux-only at present. On other platforms the utility prints a single
hint line and exits cleanly — pass integer indices (``--device 0``,
``--device 1``, …) directly to the calibration scripts there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Small VID → vendor lookup keyed off the 4-hex prefix that udev embeds
# in by-id filenames. Best-effort; unknown VIDs fall back to the prefix
# as-is.
_VENDOR_BY_VID = {
    "046d": "Logitech",
    "045e": "Microsoft",
    "2bdf": "Microsoft",
    "8086": "Intel",
    "1e4e": "Cubeternet",
    "0c45": "Microdia",
    "05a3": "ARC International",
    "1bcf": "Sunplus Innovation",
    "32e4": "HD Webcam",
    "04f2": "Chicony",
    "0bda": "Realtek",
    "0ac8": "Z-Star",
    "058f": "Alcor Micro",
}

_BY_ID_DIR = Path("/dev/v4l/by-id")
_SYSFS_V4L = Path("/sys/class/video4linux")

# usb-<VID-or-vendor>_<product>_<serial>-video-index<N>
_BY_ID_PATTERN = re.compile(
    r"^usb-(?P<vid>[^_]+)_(?P<product>.+?)_(?P<serial>[^_]+)-video-index(?P<idx>\d+)$"
)


@dataclass
class CameraInfo:
    """One V4L2 capture node, with whatever provenance we could recover."""

    by_id_path: str | None
    dev_path: str
    cv2_index: int
    model: str | None
    vendor: str | None
    serial: str | None


# ----- parsing -----


@dataclass
class ParsedByIdName:
    vid_or_vendor: str
    product: str
    serial: str
    subdevice_index: int

    @property
    def is_capture_interface(self) -> bool:
        return self.subdevice_index == 0


def parse_by_id_name(filename: str) -> ParsedByIdName | None:
    """Parse a ``/dev/v4l/by-id/`` filename into its component fields.

    Returns ``None`` if the name doesn't match the udev V4L2 convention.
    """
    m = _BY_ID_PATTERN.match(filename)
    if m is None:
        return None
    return ParsedByIdName(
        vid_or_vendor=m.group("vid"),
        product=m.group("product").replace("_", " "),
        serial=m.group("serial"),
        subdevice_index=int(m.group("idx")),
    )


def vendor_name(vid_or_vendor: str) -> str:
    """Map a 4-hex VID to a vendor name; fall back to the raw prefix."""
    return _VENDOR_BY_VID.get(vid_or_vendor.lower(), vid_or_vendor)


# ----- discovery -----


def list_cameras() -> list[CameraInfo]:
    """Enumerate V4L2 capture cameras. Returns ``[]`` on non-Linux.

    Multi-interface USB cameras (e.g. RealSense exposes several capture
    nodes under one device) collapse to a single entry per physical
    device. The kept entry prefers the one with a stable by-id path.
    """
    if sys.platform != "linux":
        return []
    if not _SYSFS_V4L.is_dir():
        return []

    by_id_index = _build_by_id_index()

    # Pair each candidate with its USB-device key so dedup can group by it.
    candidates: list[tuple[str, CameraInfo]] = []
    for sysfs_entry in sorted(_SYSFS_V4L.iterdir(), key=lambda p: _video_node_index(p.name)):
        if not sysfs_entry.name.startswith("video"):
            continue
        if not _is_capture_node(sysfs_entry):
            continue
        dev_path = f"/dev/{sysfs_entry.name}"
        usb_info = _read_usb_info_for(sysfs_entry)
        usb_key = _usb_device_key(sysfs_entry) or dev_path  # fall back: per-entry group
        candidates.append((usb_key, CameraInfo(
            by_id_path=by_id_index.get(dev_path),
            dev_path=dev_path,
            cv2_index=-1,  # filled in after dedup
            model=(usb_info.product if usb_info else None) or _read_v4l_model(sysfs_entry),
            vendor=_vendor_label(usb_info),
            serial=usb_info.serial if usb_info else None,
        )))

    cameras = _dedup_by_usb(candidates)

    # cv2 index assignment: VideoCapture(N) selects the N-th /dev/videoN
    # in numeric order.
    for i, cam in enumerate(cameras):
        cam.cv2_index = i
    return cameras


def _dedup_by_usb(candidates: list[tuple[str, CameraInfo]]) -> list[CameraInfo]:
    """Collapse multi-interface USB devices to one entry per physical camera.

    Within each USB-device group, prefer the entry that has a ``by_id_path``;
    ties break on lowest ``/dev/videoN`` index. Order across groups follows
    the lowest ``videoN`` index seen in each group.
    """
    groups: dict[str, list[CameraInfo]] = {}
    first_seen: dict[str, int] = {}
    for usb_key, cam in candidates:
        groups.setdefault(usb_key, []).append(cam)
        if usb_key not in first_seen:
            first_seen[usb_key] = _video_node_index(cam.dev_path)

    chosen: list[tuple[int, CameraInfo]] = []
    for usb_key, group in groups.items():
        winner = min(
            group,
            key=lambda c: (c.by_id_path is None, _video_node_index(c.dev_path)),
        )
        chosen.append((first_seen[usb_key], winner))
    chosen.sort(key=lambda pair: pair[0])
    return [cam for _, cam in chosen]


def _usb_device_key(sysfs_entry: Path) -> str | None:
    """Return the USB device directory the V4L2 node lives under.

    ``<sysfs>/device`` resolves to the USB *interface* (ends in ``:I.A``).
    Its parent is the USB device itself — the same path for every V4L2
    interface a single camera exposes.
    """
    device_link = sysfs_entry / "device"
    if not device_link.exists():
        return None
    try:
        return str(device_link.resolve().parent)
    except OSError:
        return None


def resolve_device(device: int | str) -> CameraInfo | None:
    """Return the ``CameraInfo`` that ``--device <device>`` would open.

    Returns ``None`` if the device isn't found (non-Linux, exotic path,
    or simply a node that isn't a capture interface).
    """
    cameras = list_cameras()
    if isinstance(device, int):
        for cam in cameras:
            if cam.cv2_index == device:
                return cam
        return None
    device = device.strip()
    for cam in cameras:
        if device in (cam.by_id_path, cam.dev_path):
            return cam
    return None


def camera_block_from_device(device: int | str) -> dict[str, object]:
    """Build a YAML-friendly identity block for a live-mode device.

    When discovery resolves ``device`` to a known camera, the block
    carries by-id path, dev path, model, vendor, and serial. Otherwise
    it falls back to ``{"raw_device": "<device>"}`` so the artifact
    still records *something* about what was opened.
    """
    info = resolve_device(device)
    if info is None:
        return {"raw_device": str(device)}
    return {
        "by_id_path": info.by_id_path,
        "dev_path": info.dev_path,
        "cv2_index": info.cv2_index,
        "model": info.model,
        "vendor": info.vendor,
        "serial": info.serial,
    }


# ----- sysfs readers -----


@dataclass
class _UsbDeviceInfo:
    vendor_id: str | None        # 4-hex (lowercased), e.g. "046d"
    product_id: str | None       # 4-hex, e.g. "082d"
    manufacturer: str | None     # e.g. "Logitech, Inc."
    product: str | None          # e.g. "HD Pro Webcam C920"
    serial: str | None           # USB device serial number, if exposed


def _read_v4l_model(sysfs_entry: Path) -> str | None:
    """Read the V4L2 driver's friendly name (may be truncated to ~31 chars)."""
    try:
        return (sysfs_entry / "name").read_text().strip() or None
    except OSError:
        return None


def _is_capture_node(sysfs_entry: Path) -> bool:
    """Capture interfaces report ``index = 0``. Subdevices (metadata, IR alt)
    report ``1+``. Missing ``index`` → accept (better to over-report)."""
    try:
        return (sysfs_entry / "index").read_text().strip() == "0"
    except OSError:
        return True


def _read_usb_info_for(sysfs_entry: Path) -> _UsbDeviceInfo | None:
    """Walk up the device symlink to the USB device level and read its fields.

    ``/sys/class/video4linux/videoN/device`` points at the USB *interface*
    (path ending in ``:I.A``). Its parent is the USB device, which carries
    ``idVendor``, ``idProduct``, ``manufacturer``, ``product``, and
    (when the device populates it) ``serial``.
    """
    device_link = sysfs_entry / "device"
    if not device_link.exists():
        return None
    try:
        usb_device_dir = device_link.resolve().parent
    except OSError:
        return None
    if not (usb_device_dir / "idVendor").is_file():
        return None
    return _UsbDeviceInfo(
        vendor_id=_read_text(usb_device_dir / "idVendor"),
        product_id=_read_text(usb_device_dir / "idProduct"),
        manufacturer=_read_text(usb_device_dir / "manufacturer"),
        product=_read_text(usb_device_dir / "product"),
        serial=_read_text(usb_device_dir / "serial"),
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _vendor_label(usb: _UsbDeviceInfo | None) -> str | None:
    """Prefer the USB-reported manufacturer string; fall back to the VID lookup."""
    if usb is None:
        return None
    if usb.manufacturer:
        # The manufacturer string is sometimes a duplicate of the product
        # name (e.g. RealSense). Filter that case to a cleaner label.
        if usb.product and usb.manufacturer.strip() == usb.product.strip():
            return vendor_name(usb.vendor_id) if usb.vendor_id else None
        return usb.manufacturer
    if usb.vendor_id:
        return vendor_name(usb.vendor_id)
    return None


def _build_by_id_index() -> dict[str, str]:
    """Map ``/dev/videoN → /dev/v4l/by-id/...`` for capture interfaces."""
    mapping: dict[str, str] = {}
    if not _BY_ID_DIR.is_dir():
        return mapping
    for entry in sorted(_BY_ID_DIR.iterdir()):
        parsed = parse_by_id_name(entry.name)
        if parsed is None or not parsed.is_capture_interface:
            continue
        try:
            target = (entry.parent / os.readlink(entry)).resolve()
        except OSError:
            continue
        mapping[str(target)] = str(entry)
    return mapping


def _video_node_index(name: str) -> int:
    """Extract the trailing integer from ``videoN`` for stable numeric sort."""
    m = re.search(r"video(\d+)$", name)
    return int(m.group(1)) if m is not None else 1 << 30


# ----- probing -----


@dataclass
class ProbeResult:
    ok: bool
    width: int | None
    height: int | None
    reason: str | None


def probe(camera: CameraInfo, *, timeout_s: float = 1.5) -> ProbeResult:
    """Try to open ``camera`` and grab one frame to confirm it streams."""
    import time

    import cv2

    target = camera.by_id_path or camera.dev_path
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        return ProbeResult(ok=False, width=None, height=None, reason="could not open")
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                return ProbeResult(ok=True, width=int(w), height=int(h), reason=None)
        return ProbeResult(ok=False, width=None, height=None, reason="no frames")
    finally:
        cap.release()


# ----- formatting -----


def _format_human(cameras: list[CameraInfo], probes: list[ProbeResult] | None) -> str:
    if not cameras:
        return (
            "No cameras found.\n"
            "  If a camera is plugged in, check `lsusb` and `dmesg` "
            "for the kernel side."
        )
    out: list[str] = [f"Found {len(cameras)} camera{'s' if len(cameras) != 1 else ''}:"]
    out.append("")
    for i, cam in enumerate(cameras):
        out.extend(_format_camera_block(i, cam, probes[i] if probes is not None else None))
        out.append("")
    out.append(
        "Tip: copy the --device line. by-id paths stay fixed across reboots\n"
        "and USB-port swaps; integer indices do not."
    )
    return "\n".join(out)


def _format_camera_block(
    bullet: int, cam: CameraInfo, probe: ProbeResult | None,
) -> list[str]:
    headline_bits: list[str] = []
    if cam.model:
        headline_bits.append(cam.model)
    if cam.vendor or cam.serial:
        annotation_bits = []
        if cam.vendor:
            annotation_bits.append(cam.vendor)
        if cam.serial:
            annotation_bits.append(f"serial {cam.serial}")
        headline_bits.append(f"({', '.join(annotation_bits)})")
    headline = " ".join(headline_bits) if headline_bits else f"camera at {cam.dev_path}"

    device_arg = cam.by_id_path or cam.dev_path
    lines = [f"[{bullet}] {headline}", f"    --device {device_arg}"]
    sub = f"(cv2 index {cam.cv2_index} → {cam.dev_path}"
    if cam.by_id_path is None:
        sub += "; no by-id entry — integer index is the only handle"
    sub += ")"
    lines.append(f"    {sub}")
    if probe is not None:
        if probe.ok:
            lines.append(f"    probe: ✓ streams at {probe.width}x{probe.height}")
        else:
            lines.append(f"    probe: ✗ {probe.reason}")
    return lines


def _format_json(
    cameras: list[CameraInfo], probes: list[ProbeResult] | None,
) -> str:
    payload = []
    for i, cam in enumerate(cameras):
        item = asdict(cam)
        if probes is not None:
            item["probe"] = asdict(probes[i])
        payload.append(item)
    return json.dumps(payload, indent=2)


def _format_paths(cameras: list[CameraInfo]) -> str:
    return "\n".join(
        cam.by_id_path or cam.dev_path for cam in cameras
    )


# ----- CLI -----


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camcal-list-cameras",
        description=(
            "List connected V4L2 cameras with stable --device paths "
            "ready to paste into camcal-intrinsics / camcal-extrinsics."
        ),
    )
    p.add_argument(
        "--probe", action="store_true",
        help="Open each camera with cv2 and grab one frame to confirm "
             "it streams. Slower; needed if a node is reported but won't open.",
    )
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument(
        "--json", action="store_true",
        help="Emit a structured JSON list instead of human-readable blocks.",
    )
    fmt.add_argument(
        "--paths", action="store_true",
        help="Emit only the by-id path per camera (one per line). "
             "Convenient for shell loops.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if sys.platform != "linux":
        print(
            "Camera discovery is Linux-only at present; pass integer "
            "indices (0, 1, …) directly to --device on this platform."
        )
        return 0

    cameras = list_cameras()
    probes = [probe(c) for c in cameras] if args.probe else None

    if args.json:
        print(_format_json(cameras, probes))
    elif args.paths:
        print(_format_paths(cameras))
    else:
        print(_format_human(cameras, probes))
    return 0


__all__ = [
    "CameraInfo",
    "ParsedByIdName",
    "ProbeResult",
    "camera_block_from_device",
    "list_cameras",
    "main",
    "parse_by_id_name",
    "probe",
    "resolve_device",
    "vendor_name",
]


if __name__ == "__main__":
    sys.exit(main())
