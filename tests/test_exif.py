"""EXIF focal length extraction, against files with known tags.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_exif.py
"""

from __future__ import annotations

import sys
import tempfile
from math import hypot
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


def expected_px(f35, width, height):
    """35mm equivalence is a diagonal field of view, so it converts by diagonal."""
    return f35 * hypot(width, height) / hypot(36.0, 24.0)


def test_35mm_equivalent():
    p = make_image("f35.jpg", 4000, 3000, {X.TAG_FOCAL_35MM: 24})
    r = X.read_focal(p, 4000, 3000)
    assert r.focal_px is not None, r.note
    assert abs(r.focal_px - expected_px(24, 4000, 3000)) < 1.0, r.focal_px
    assert r.source == "35mm equivalent"
    assert r.confidence == "approximate"


def test_three_two_is_the_only_aspect_the_width_rule_got_right():
    """The old f35 * W / 36 is the diagonal rule specialised to a 3:2 frame.

    Anywhere else it is wrong, and in a direction that depends on the shape:
    low for a frame squarer than the film, high for a wider one.
    """
    for w, h in ((6000, 4000), (3000, 2000)):  # 3:2, like the film frame
        assert abs(expected_px(24, w, h) - 24 * w / 36.0) < 0.01, (w, h)

    four_three = expected_px(24, 4000, 3000)
    assert four_three > 24 * 4000 / 36.0
    low_by = 1 - (24 * 4000 / 36.0) / four_three
    assert abs(low_by - 0.0385) < 0.001, low_by

    sixteen_nine = expected_px(24, 1920, 1080)
    assert sixteen_nine < 24 * 1920 / 36.0
    high_by = (24 * 1920 / 36.0) / sixteen_nine - 1
    assert abs(high_by - 0.0478) < 0.001, high_by


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
    # a 3:2 sensor, so the width rule and the diagonal rule agree here
    assert abs(r.focal_px - 50 * 6000 / 36.0) < 5.0, r.focal_px
    assert r.source == "focal plane resolution"
    assert r.confidence == "good"
    assert abs(r.sensor_width_mm - 36.0) < 0.2, r.sensor_width_mm
    assert abs(r.sensor_diag_mm - hypot(36.0, 24.0)) < 0.2, r.sensor_diag_mm


def test_focal_plane_route_survives_rotation():
    """The sensor diagonal is the same whichever way the file is stored."""
    res_per_inch = 6000 / (36.0 / 25.4)
    tags = {
        X.TAG_FOCAL_LENGTH: 50.0,
        X.TAG_FOCAL_PLANE_X_RES: res_per_inch,
        X.TAG_FOCAL_PLANE_UNIT: 2,
    }
    p = make_image("fp_rot.jpg", 6000, 4000, tags)
    land = X.read_focal(p, 6000, 4000)
    port = X.read_focal(p, 4000, 6000)
    assert abs(land.focal_px - port.focal_px) < 1.0, (land.focal_px, port.focal_px)
    assert port.confidence == "good", port.note


def test_resized_image_still_converts():
    """Downscaling does not change the field of view, so f_px scales with size."""
    p = make_image("resized.jpg", 2000, 1500, {X.TAG_FOCAL_35MM: 24}, exif_dims=(4000, 3000))
    r = X.read_focal(p, 2000, 1500)
    assert abs(r.focal_px - expected_px(24, 2000, 1500)) < 1.0, r.focal_px
    assert r.confidence == "approximate", r.note
    assert not r.rotated


def test_crop_is_refused():
    """A changed aspect ratio means the pixels no longer span the whole sensor."""
    p = make_image("crop.jpg", 3000, 3000, {X.TAG_FOCAL_35MM: 24}, exif_dims=(4000, 3000))
    r = X.read_focal(p, 3000, 3000)
    assert r.confidence == "none", r.note
    assert "crop" in r.note.lower(), r.note
    assert not r.rotated


def test_rotation_is_not_a_crop():
    """A portrait photo reaches us with width and height swapped against EXIF.

    Nothing was cut away, so refusing it as a crop throws away a perfectly good
    focal length -- which is what every upright phone photo used to get.
    """
    p = make_image("rot.jpg", 3000, 4000, {X.TAG_FOCAL_35MM: 24}, exif_dims=(4000, 3000))
    r = X.read_focal(p, 3000, 4000)
    assert r.rotated, r.note
    assert r.confidence == "approximate", r.note
    assert "crop" not in r.note.lower(), r.note
    assert abs(r.focal_px - expected_px(24, 3000, 4000)) < 1.0, r.focal_px


def test_focal_is_orientation_invariant():
    """The same lens is the same lens whichever way the phone was held."""
    p = make_image("ori.jpg", 4000, 3000, {X.TAG_FOCAL_35MM: 24})
    land = X.read_focal(p, 4000, 3000)
    port = X.read_focal(p, 3000, 4000)
    assert abs(land.focal_px - port.focal_px) < 1e-6, (land.focal_px, port.focal_px)


def test_iphone_portrait_does_not_read_28_percent_low():
    """Regression: a 4032x3024 iPhone file used upright as 3024x4032.

    By the old width rule this read 2016 px against a true 2796 -- a 28% error
    that every camera position and point depth would have inherited in silence.
    """
    p = make_image("iphone.jpg", 3024, 4032, {X.TAG_FOCAL_35MM: 24}, exif_dims=(4032, 3024))
    r = X.read_focal(p, 3024, 4032)
    assert r.confidence == "approximate", r.note
    assert abs(r.focal_px - 2795.7) < 1.0, r.focal_px
    assert r.focal_px > 2016 * 1.3


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
    for w, h in ((4000, 3000), (3000, 4000), (1920, 1080), (6000, 4000)):
        px = X.focal_px_from_35mm(24, w, h)
        assert abs(X.focal_px_to_35mm(px, w, h) - 24) < 1e-6, (w, h)


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
    assert abs(g.focal_px - expected_px(28, 4000, 3000)) < 1.0


# ------------------------------------------------- captured at a cropped ratio
#
# A crop made after the fact is detectable and refused (test_crop_is_refused):
# EXIF still records the dimensions the camera wrote, and they disagree with
# the file. A crop made *at capture* -- the phone set to 16:9 or to the 19.5:9
# it calls "Full" -- leaves no such trace, because the camera writes the
# cropped dimensions into EXIF too. What it does not do is change
# FocalLengthIn35mmFilm, which describes the lens against the whole sensor. So
# the only signal is the shape of the frame itself: no sensor is built 19.5:9.


def test_a_nineteen_point_five_by_nine_capture_is_reconstructed():
    """Galaxy S24+ ultra-wide, 4000x1848, EXIF says 13 mm equivalent.

    Against the cropped diagonal this reads 1324 px. The photographs' own
    control lines put it near 1625, and locking the 1324 drove reprojection to
    64 px where the reconstructed figure managed 1.4 -- the error is large
    enough to wreck a solve and leaves nothing behind that looks like a cause.
    """
    p = make_image("full_ratio.jpg", 4000, 1848, {X.TAG_FOCAL_35MM: 13},
                   exif_dims=(4000, 1848))
    r = X.read_focal(p, 4000, 1848)
    assert r.native_frame_px == (4000.0, 3000.0), r.native_frame_px
    assert abs(r.focal_px - expected_px(13, 4000, 3000)) < 1.0, r.focal_px
    assert abs(r.focal_px - 1502.3) < 1.0, r.focal_px
    cropped_diagonal = expected_px(13, 4000, 1848)
    assert abs(cropped_diagonal - 1323.9) < 1.0, cropped_diagonal
    assert r.focal_px > cropped_diagonal * 1.12


def test_a_reconstructed_focal_is_still_usable():
    """It must not be marked unusable: the mixed-lens check reads these.

    Downgrading the confidence to "none" would take summarise_group's
    two-cameras-in-one-group warning down with it, which is the check that
    catches a phone's wide and main lenses sharing a camera group.
    """
    p = make_image("usable.jpg", 4000, 1848, {X.TAG_FOCAL_35MM: 13},
                   exif_dims=(4000, 1848))
    r = X.read_focal(p, 4000, 1848)
    assert r.confidence == "approximate", r.confidence
    assert "cropped aspect ratio" in r.note, r.note
    assert "control lines" in r.note, r.note


def test_the_reconstruction_survives_rotation():
    """Held upright, the same capture must give the same focal length."""
    land = X.read_focal(
        make_image("land.jpg", 4000, 1848, {X.TAG_FOCAL_35MM: 13}, exif_dims=(4000, 1848)),
        4000, 1848)
    port = X.read_focal(
        make_image("port.jpg", 4000, 1848, {X.TAG_FOCAL_35MM: 13}, exif_dims=(4000, 1848)),
        1848, 4000)
    assert abs(land.focal_px - port.focal_px) < 1e-6, (land.focal_px, port.focal_px)


def test_a_square_capture_keeps_its_short_side():
    """1:1 throws away columns, so it is the height that survived."""
    assert X.native_frame_px(3000, 3000) == (4000.0, 3000.0)


def test_native_shapes_are_left_alone():
    """The ordinary case must not move: 4:3 and 3:2 are what sensors are."""
    for w, h in ((4000, 3000), (3000, 4000), (6000, 4000), (4000, 6000)):
        assert X.native_frame_px(w, h) is None, (w, h)
        r = X.read_focal(
            make_image(f"n{w}x{h}.jpg", w, h, {X.TAG_FOCAL_35MM: 24}), w, h)
        assert abs(r.focal_px - expected_px(24, w, h)) < 0.01, (w, h, r.focal_px)
        assert r.native_frame_px is None
        assert "cropped aspect ratio" not in r.note


def test_sixteen_nine_is_a_crop_too():
    """No stills sensor is 16:9 either; a phone shooting it is cropping 4:3."""
    assert X.native_frame_px(1920, 1080) == (1920.0, 1440.0)


def test_the_focal_plane_route_is_untouched():
    """Sensor dimensions read off the focal-plane tags already describe the crop.

    Those tags give the sensor extent the output image actually covers, so a
    cropped capture is already accounted for and reconstructing would double
    it. Only the 35mm-equivalent route needs the correction.
    """
    res_per_inch = 4000 / (36.0 / 25.4)
    p = make_image("fp_crop.jpg", 4000, 1848, {
        X.TAG_FOCAL_LENGTH: 50.0,
        X.TAG_FOCAL_PLANE_X_RES: res_per_inch,
        X.TAG_FOCAL_PLANE_Y_RES: res_per_inch,
        X.TAG_FOCAL_PLANE_UNIT: 2,
    }, exif_dims=(4000, 1848))
    r = X.read_focal(p, 4000, 1848)
    assert r.source == "focal plane resolution", r.source
    assert r.native_frame_px is None, r.native_frame_px
    assert abs(r.focal_px - 50.0 * hypot(4000, 1848) / r.sensor_diag_mm) < 0.01


def test_a_cropped_group_still_shows_two_lenses():
    """The whole point of keeping these usable: the mixed-lens check works.

    Seven ultra-wide frames and two off the main camera, all 4000x1848, which
    is how the Alberto set arrived.
    """
    from multispy.project import CameraGroup, ImageView, Project

    project = Project(name="alberto-ish", path=TMP / "mixed.multispy")
    project.groups["cam0"] = CameraGroup(id="cam0")
    for i, f35 in enumerate((13, 13, 13, 13, 13, 13, 13, 23, 23)):
        img = make_image(f"m{i}.jpg", 4000, 1848, {X.TAG_FOCAL_35MM: f35},
                         exif_dims=(4000, 1848))
        project.images.append(ImageView(id=f"img{i}", path=str(img), width=4000,
                                        height=1848, group="cam0"))
    g = X.summarise_group(project, "cam0")
    assert g.n_with_focal == 9, g.n_with_focal
    assert g.confidence == "none", g.note
    assert "separate camera groups" in g.note, g.note



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
