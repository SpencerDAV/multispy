"""EXIF focal length extraction, against files with known tags.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_exif.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy import exif as X  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="multispy_exif_"))


def make_image(name, width, height, tags=None, exif_dims=None):
    """Write a JPEG carrying the given Exif sub-IFD tags."""
    path = TMP / name
    img = Image.new("RGB", (width, height), (90, 110, 130))
    if tags is None:
        img.save(path, quality=70)
        return path
    data = Image.Exif()
    sub = dict(tags)
    if exif_dims:
        sub[X.TAG_EXIF_WIDTH], sub[X.TAG_EXIF_HEIGHT] = exif_dims
    else:
        sub[X.TAG_EXIF_WIDTH], sub[X.TAG_EXIF_HEIGHT] = width, height
    data[X.TAG_MAKE] = "TestCam"
    data[X.TAG_MODEL] = "Model X"
    data[X.TAG_EXIF_IFD] = sub
    img.save(path, exif=data, quality=70)
    return path


def test_35mm_equivalent():
    p = make_image("f35.jpg", 4000, 3000, {X.TAG_FOCAL_35MM: 24})
    r = X.read_focal(p, 4000, 3000)
    assert r.focal_px is not None, r.note
    assert abs(r.focal_px - 24 * 4000 / 36.0) < 1.0, r.focal_px
    assert r.source == "35mm equivalent"
    assert r.confidence == "approximate"


def test_focal_plane_resolution():
    # 50 mm lens on a 36 mm wide sensor recorded as 6000 px
    res_per_inch = 6000 / (36.0 / 25.4)
    p = make_image("fp.jpg", 6000, 4000, {
        X.TAG_FOCAL_LENGTH: 50.0,
        X.TAG_FOCAL_PLANE_X_RES: res_per_inch,
        X.TAG_FOCAL_PLANE_UNIT: 2,
    })
    r = X.read_focal(p, 6000, 4000)
    assert r.focal_px is not None, r.note
    assert abs(r.focal_px - 50 * 6000 / 36.0) < 5.0, r.focal_px
    assert r.source == "focal plane resolution"
    assert r.confidence == "good"
    assert abs(r.sensor_width_mm - 36.0) < 0.2, r.sensor_width_mm


def test_resized_image_still_converts():
    """Downscaling does not change the field of view, so f_px scales with width."""
    p = make_image("resized.jpg", 2000, 1500, {X.TAG_FOCAL_35MM: 24}, exif_dims=(4000, 3000))
    r = X.read_focal(p, 2000, 1500)
    assert abs(r.focal_px - 24 * 2000 / 36.0) < 1.0, r.focal_px
    assert r.confidence == "approximate", r.note


def test_crop_is_refused():
    """A changed aspect ratio means the pixels no longer span the whole sensor."""
    p = make_image("crop.jpg", 3000, 3000, {X.TAG_FOCAL_35MM: 24}, exif_dims=(4000, 3000))
    r = X.read_focal(p, 3000, 3000)
    assert r.confidence == "none", r.note
    assert "crop" in r.note.lower(), r.note


def test_millimetres_without_sensor_size_is_refused():
    p = make_image("nosensor.jpg", 4000, 3000, {X.TAG_FOCAL_LENGTH: 35.0})
    r = X.read_focal(p, 4000, 3000)
    assert r.focal_px is None
    assert r.focal_mm == 35.0
    assert "sensor size" in r.note, r.note


def test_no_exif_at_all():
    p = make_image("bare.jpg", 1000, 800)
    r = X.read_focal(p, 1000, 800)
    assert r.focal_px is None
    assert r.confidence == "none"


def test_round_trip_helpers():
    px = X.focal_px_from_35mm(24, 4000)
    assert abs(X.focal_px_to_35mm(px, 4000) - 24) < 1e-6


def test_group_flags_a_zoom_change():
    """Different focal lengths across a group means one shared focal is wrong."""
    from multispy.project import CameraGroup, ImageView, Project

    project = Project(name="zoomy", path=TMP / "p.multispy")
    project.groups["cam0"] = CameraGroup(id="cam0")
    for i, f35 in enumerate((24, 24, 50)):
        p = make_image(f"z{i}.jpg", 4000, 3000, {X.TAG_FOCAL_35MM: f35})
        project.images.append(ImageView(id=f"img{i}", path=str(p), width=4000,
                                        height=3000, group="cam0"))
    g = X.summarise_group(project, "cam0")
    assert g.n_with_focal == 3
    assert g.confidence == "none", g.note
    assert "zoom" in g.note.lower(), g.note


def test_group_agrees_when_it_should():
    from multispy.project import CameraGroup, ImageView, Project

    project = Project(name="steady", path=TMP / "p2.multispy")
    project.groups["cam0"] = CameraGroup(id="cam0")
    for i in range(3):
        p = make_image(f"s{i}.jpg", 4000, 3000, {X.TAG_FOCAL_35MM: 28})
        project.images.append(ImageView(id=f"img{i}", path=str(p), width=4000,
                                        height=3000, group="cam0"))
    g = X.summarise_group(project, "cam0")
    assert g.n_with_focal == 3
    assert g.confidence == "approximate", g.note
    assert abs(g.focal_px - 28 * 4000 / 36.0) < 1.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\nscratch files in {TMP}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
