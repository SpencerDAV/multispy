"""Focal length from EXIF.

Worth being careful here, because a focal length that is quietly wrong is worse
than none at all: lock it and every camera position and point depth inherits the
error silently, with nothing in the residuals to give it away.

So each reading carries a confidence, and the two routes are not equally good:

* **35mm equivalent** (``FocalLengthIn35mmFilm``). The camera has already done
  the sensor-size arithmetic, so this needs no guesswork. Usually rounded to a
  whole millimetre, which on a wide lens is about 2-4%.

* **Focal plane resolution** (``FocalLength`` in mm plus ``FocalPlaneXResolution``).
  Exact when present, but many phones and drones omit the focal-plane tags.

Both routes go through the **diagonal**, not the width::

    f_px = f35 * hypot(W, H) / hypot(36, 24)

35mm equivalence is defined on the diagonal field of view, so dividing by the
36 mm film *width* is only right when the image happens to be 3:2 like the film
frame. A 4:3 phone photo comes out 3.8% low that way and a 16:9 frame 4.8% high.
The diagonal is also invariant under a 90 degree rotation, which matters because
a portrait photo reaches us with its width and height already swapped -- by the
width formula the same file reads 28% low purely for having been held upright.

A reading is only as good as the assumption that the pixels still correspond to
the whole sensor. Cropping after capture breaks that silently, so an aspect
ratio that disagrees with the EXIF dimensions downgrades the confidence --
unless it disagrees by exactly a rotation, which changes no geometry at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

# EXIF tag ids, see the Exif 2.3 specification
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_FOCAL_LENGTH = 0x920A
TAG_FOCAL_35MM = 0xA405
TAG_FOCAL_PLANE_X_RES = 0xA20E
TAG_FOCAL_PLANE_Y_RES = 0xA20F
TAG_FOCAL_PLANE_UNIT = 0xA210
TAG_EXIF_WIDTH = 0xA002
TAG_EXIF_HEIGHT = 0xA003
TAG_EXIF_IFD = 0x8769

MM_PER_UNIT = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_HEIGHT_MM = 24.0
FULL_FRAME_DIAGONAL_MM = math.hypot(FULL_FRAME_WIDTH_MM, FULL_FRAME_HEIGHT_MM)

# how far the current aspect ratio may drift from EXIF's before it is a crop
ASPECT_TOLERANCE = 0.02


@dataclass
class ExifFocal:
    image_id: str = ""
    focal_px: float | None = None
    source: str = "none"
    confidence: str = "none"  # good | approximate | none
    focal_mm: float | None = None
    focal_35mm: float | None = None
    sensor_width_mm: float | None = None
    sensor_diag_mm: float | None = None
    rotated: bool = False
    """The file is stored one way round and handed to us the other. Harmless --
    the diagonal does not care -- but worth saying so it is not read as a crop."""
    make: str = ""
    model: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_tags(path: Path) -> dict[int, Any]:
    """Flatten IFD0 and the Exif sub-IFD into one tag dict."""
    tags: dict[int, Any] = {}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return tags
            tags.update(dict(exif))
            try:
                tags.update(dict(exif.get_ifd(TAG_EXIF_IFD)))
            except Exception:  # noqa: BLE001 - malformed sub-IFD, keep what we have
                pass
    except Exception:  # noqa: BLE001 - unreadable or not an image
        return {}
    return tags


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if f > 0 and f == f and f != float("inf") else None


def sensor_width_mm(tags: dict[int, Any]) -> float | None:
    """Sensor width implied by the focal-plane resolution tags."""
    res = _as_float(tags.get(TAG_FOCAL_PLANE_X_RES))
    width = _as_float(tags.get(TAG_EXIF_WIDTH))
    unit = tags.get(TAG_FOCAL_PLANE_UNIT, 2)
    if res is None or width is None:
        return None
    try:
        mm = MM_PER_UNIT[int(unit)]
    except (KeyError, TypeError, ValueError):
        return None
    sensor = (width / res) * mm
    # anything outside this range is a mis-parsed tag, not a camera
    return sensor if 1.0 < sensor < 100.0 else None


def sensor_diagonal_mm(tags: dict[int, Any]) -> float | None:
    """Sensor diagonal implied by the focal-plane resolution tags.

    The diagonal rather than the width, so the conversion survives a rotated
    file. ``FocalPlaneYResolution`` is almost always equal to the X one; when it
    is missing, assuming square pixels is far safer than giving up.
    """
    xres = _as_float(tags.get(TAG_FOCAL_PLANE_X_RES))
    yres = _as_float(tags.get(TAG_FOCAL_PLANE_Y_RES)) or xres
    ew = _as_float(tags.get(TAG_EXIF_WIDTH))
    eh = _as_float(tags.get(TAG_EXIF_HEIGHT))
    unit = tags.get(TAG_FOCAL_PLANE_UNIT, 2)
    if xres is None or yres is None or ew is None or eh is None:
        return None
    try:
        mm = MM_PER_UNIT[int(unit)]
    except (KeyError, TypeError, ValueError):
        return None
    diag = math.hypot(ew / xres, eh / yres) * mm
    # a 35mm frame is 43.3; medium format reaches about 87. Past that it is a
    # mis-parsed tag rather than a camera.
    return diag if 1.0 < diag < 150.0 else None


def read_focal(path: str | Path, width_px: int, height_px: int, image_id: str = "") -> ExifFocal:
    """Focal length in pixels for an image as it currently is on disk."""
    path = Path(path)
    out = ExifFocal(image_id=image_id)
    if not path.exists():
        out.note = "file not found"
        return out

    tags = _read_tags(path)
    if not tags:
        out.note = "no EXIF data"
        return out

    out.make = str(tags.get(TAG_MAKE, "") or "").strip()
    out.model = str(tags.get(TAG_MODEL, "") or "").strip()
    out.focal_mm = _as_float(tags.get(TAG_FOCAL_LENGTH))
    out.focal_35mm = _as_float(tags.get(TAG_FOCAL_35MM))
    out.sensor_width_mm = sensor_width_mm(tags)
    out.sensor_diag_mm = sensor_diagonal_mm(tags)
    if out.sensor_diag_mm is None and out.sensor_width_mm is not None and width_px and height_px:
        # no EXIF height to work from: assume the sensor has the same shape as
        # the pixels, taking the long side as the recorded width so a rotated
        # file still lands on the same diagonal
        long_side, short_side = max(width_px, height_px), min(width_px, height_px)
        out.sensor_diag_mm = out.sensor_width_mm * math.hypot(1.0, short_side / long_side)

    diag_px = math.hypot(width_px, height_px)

    # a crop invalidates the sensor-to-pixel correspondence. A rotation does not:
    # it swaps width and height, leaving the diagonal -- and so the conversion --
    # untouched. Telling them apart is the difference between a usable focal
    # length and a file refused for a crop that never happened.
    cropped = False
    ew = _as_float(tags.get(TAG_EXIF_WIDTH))
    eh = _as_float(tags.get(TAG_EXIF_HEIGHT))
    if ew and eh and width_px and height_px:
        recorded = ew / eh
        current = width_px / height_px
        if abs(recorded - current) <= ASPECT_TOLERANCE * current:
            pass
        elif abs((1.0 / recorded) - current) <= ASPECT_TOLERANCE * current:
            out.rotated = True
        else:
            cropped = True

    if out.focal_35mm is not None:
        out.focal_px = out.focal_35mm * diag_px / FULL_FRAME_DIAGONAL_MM
        out.source = "35mm equivalent"
        out.confidence = "approximate"
        out.note = (
            f"{out.focal_35mm:.0f} mm equivalent; EXIF usually rounds this to a whole "
            f"millimetre, so treat it as a few percent"
        )
    elif out.focal_mm is not None and out.sensor_diag_mm is not None:
        out.focal_px = out.focal_mm * diag_px / out.sensor_diag_mm
        out.source = "focal plane resolution"
        out.confidence = "good"
        out.note = (
            f"{out.focal_mm:.1f} mm on a sensor {out.sensor_diag_mm:.1f} mm across the diagonal"
        )
    elif out.focal_mm is not None:
        out.note = (
            f"EXIF gives {out.focal_mm:.1f} mm but no sensor size, so it cannot be converted "
            f"to pixels. Enter the sensor width, or a focal length in pixels, by hand."
        )
        return out
    else:
        out.note = "EXIF present but carries no focal length"
        return out

    if cropped:
        out.confidence = "none"
        out.note += (
            f". The image is {width_px}x{height_px} but EXIF records {ew:.0f}x{eh:.0f} -- a "
            f"different aspect ratio means it was cropped, which invalidates this conversion"
        )
    elif out.rotated:
        out.note += (
            f". Stored {ew:.0f}x{eh:.0f} and used {width_px}x{height_px}, i.e. rotated a quarter "
            f"turn; the diagonal is unchanged, so this conversion still holds"
        )
    return out


def focal_px_from_mm(focal_mm: float, sensor_diag_mm_: float, width_px: int, height_px: int) -> float:
    return focal_mm * math.hypot(width_px, height_px) / sensor_diag_mm_


def focal_px_from_35mm(focal_35mm: float, width_px: int, height_px: int) -> float:
    return focal_35mm * math.hypot(width_px, height_px) / FULL_FRAME_DIAGONAL_MM


def focal_px_to_35mm(focal_px: float, width_px: int, height_px: int) -> float:
    """Inverse, for showing a pixel focal length in units people recognise."""
    return focal_px * FULL_FRAME_DIAGONAL_MM / math.hypot(width_px, height_px)


@dataclass
class GroupFocal:
    """What EXIF says about one camera group as a whole."""

    group: str
    focal_px: float | None = None
    confidence: str = "none"
    source: str = "none"
    n_with_focal: int = 0
    n_images: int = 0
    spread_pct: float | None = None
    note: str = ""
    per_image: list[ExifFocal] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_image"] = [e.to_dict() for e in (self.per_image or [])]
        return d


def summarise_group(project, group_id: str) -> GroupFocal:
    """Read every image in a group and agree on one focal length.

    Disagreement is the interesting case: if the readings differ by more than a
    few percent the lens was zoomed between shots, and a single shared focal
    length is the wrong model regardless of what any one file says.
    """
    images = [im for im in project.active_images() if im.group == group_id]
    out = GroupFocal(group=group_id, n_images=len(images), per_image=[])
    for im in images:
        out.per_image.append(
            read_focal(project.resolve_path(im), im.width, im.height, im.id)
        )

    usable = [e for e in out.per_image if e.focal_px is not None and e.confidence != "none"]
    out.n_with_focal = len(usable)
    if not usable:
        blocked = next((e.note for e in out.per_image if e.note), "no EXIF focal length found")
        out.note = blocked
        return out

    values = sorted(e.focal_px for e in usable)
    out.focal_px = values[len(values) // 2]
    out.source = usable[0].source
    out.confidence = min((e.confidence for e in usable), key=lambda c: ["good", "approximate", "none"].index(c))
    if len(values) > 1:
        out.spread_pct = 100.0 * (values[-1] - values[0]) / out.focal_px
        if out.spread_pct > 2.0:
            out.confidence = "none"
            out.note = (
                f"EXIF focal lengths across these {len(values)} images differ by "
                f"{out.spread_pct:.0f}% -- the lens was zoomed between shots, so one shared "
                f"focal length is the wrong model. Split them into separate camera groups."
            )
            return out
    out.note = usable[0].note
    return out
