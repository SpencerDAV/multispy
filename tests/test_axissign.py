"""One control line drawn backwards, and the flag that now says so.

The failure these defend against has no residual. A vanishing point is blind to
the direction a segment was dragged in -- reverse ``p0`` and ``p1`` and the line
is the same line -- so the only thing the drag direction does is fix the *sign*
of the axis, by a length-weighted vote across that axis's lines. One line drawn
backwards among several therefore does not fail. It dilutes. The axis comes out
pointing whichever way the longer lines happened to go, and if that is the wrong
way the camera arrives as a rotation 180 degrees out with a clean orthogonality
error, a clean focal length and clean reprojection residuals.

Measured below: it is 179.98 degrees out, which is the whole problem. Nothing
downstream is in a position to notice, so the check has to be in the markup.

What is pinned here:

* **The flag fires, on the right axis, naming the right line.** Including the
  case the existing warning got wrong -- see the last test.

* **It does not fire on a correct markup**, over every image of the synthetic
  scene, nor on the one configuration where the naive version of this test
  cannot help firing: two lines straddling a vanishing point inside the frame
  have image directions nearly 180 degrees apart and are both correct.

* **It does not change the sign the solver picks.** The check is a second
  reading of the same poll ``resolve_direction_sign`` already took; if adding it
  moved any solved number, the reading would be the wrong one.

* **It reaches the top of the solve report**, not the per-image notes column,
  because something that leaves no residual is exactly what nobody scrolls to.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_axissign.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.geometry import (
    calibrate_image,
    check_axis_signs,
    estimate_vanishing_point,
    resolve_direction_sign,
    direction_from_vp,
    analyse_axes,
    Intrinsics,
)
from multispy.solve import solve
from multispy.webapp import observability_payload

from test_synthetic import build_project, rotation_error_deg  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def usable_segments(image) -> dict[int, np.ndarray]:
    return {a: s for a, s in image.axis_segments().items() if len(s) >= 2}


def calibrate(image):
    return calibrate_image(usable_segments(image), image.width, image.height)


def flip_nth_line_on(image, axis: int, n: int = 0) -> int:
    """Reverse one control line in place, as a wrong drag would have left it.

    Returns its index in the photo's own line list, which is what the UI marks
    and is *not* the index within the axis -- the two differ as soon as a photo
    carries lines on more than one axis, and conflating them would point the
    user's eye at an innocent line.
    """
    rows = [i for i, ln in enumerate(image.axis_lines) if ln.axis == axis and not ln.is_free]
    idx = rows[n]
    ln = image.axis_lines[idx]
    ln.p0, ln.p1 = ln.p1, ln.p0
    return idx


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_a_clean_markup_is_never_flagged():
    """Every image of the synthetic scene, every axis, both line budgets.

    A direction check that cries wolf is worse than none: this one is meant to
    be read as "go and look at the photo", and a reader who has dismissed it
    twice will dismiss the one that was real.
    """
    for budget in (2, 3):
        project, _truth = build_project(max_lines_per_axis=budget)
        for im in project.images:
            calib = calibrate(im)
            assert calib.contested_axes == [], (budget, im.id, calib.contested_axes)
            analysis = analyse_axes(usable_segments(im), im.width, im.height)
            assert analysis.contested_axes == [], (budget, im.id, analysis.contested_axes)


def test_two_lines_either_side_of_a_near_vanishing_point_are_not_flagged():
    """The case that rules out the obvious implementation.

    "Flag two arrows almost 180 degrees apart" is the natural reading of this
    check and it is wrong, because a vanishing point inside the frame makes that
    the *correct* picture: the lines on either side of it both point at it, so
    their image directions are opposed while their 3D directions are identical.
    Comparing each line against the vanishing point instead of against the other
    lines is what buys immunity to this, and it is the reason the per-line test
    lives in geometry rather than as a dot product at the call site.
    """
    w = h = 1000
    # a vanishing point near the middle of the frame, with a segment approaching
    # it from each side -- a perfectly ordinary markup of a corridor floor
    vp_true = np.array([520.0, 480.0])
    segs = []
    # Bearings nearly, but not exactly, opposite. Exactly opposite would put
    # both segments on one line through the vanishing point, which determines no
    # vanishing point at all -- the 17 degrees between them is what makes this a
    # legitimate markup as well as a 163-degree trap.
    for ang in (0.3, 0.3 + 2.85):
        u = np.array([math.cos(ang), math.sin(ang)])
        far, near = vp_true + u * 430.0, vp_true + u * 150.0
        segs.append([far[0], far[1], near[0], near[1]])   # drawn TOWARD the vp
    segs = np.array(segs)

    d0 = segs[0, 2:4] - segs[0, 0:2]
    d1 = segs[1, 2:4] - segs[1, 0:2]
    naive = math.degrees(math.acos(
        float(d0 @ d1) / (np.linalg.norm(d0) * np.linalg.norm(d1))
    ))
    assert naive > 160.0, f"fixture is not exercising the trap (only {naive:.0f} deg apart)"

    vp = estimate_vanishing_point(segs, w, h)
    check = check_axis_signs(vp.v, segs, axis=2)
    assert not check.contested, (
        f"two correct lines straddling a vanishing point {naive:.0f} deg apart in the image "
        f"were flagged: oppositions {[round(v.opposition_deg, 1) for v in check.votes]}"
    )


def test_one_reversed_line_is_found_and_named():
    """The flag fires, says which line, and says how far round it is."""
    project, _truth = build_project(max_lines_per_axis=3)
    im = project.images[0]
    flip_nth_line_on(im, axis=2, n=1)   # a middling Z line, so the vote is not close

    calib = calibrate(im)
    assert calib.contested_axes == [2], calib.contested_axes
    check = calib.signs[2]
    assert [v.index for v in check.dissenting] == [1], [v.index for v in check.dissenting]
    assert check.worst_opposition_deg > 175.0, check.worst_opposition_deg
    assert check.majority_px > check.minority_px
    assert not check.too_close_to_call

    text = check.message()
    for expected in ("Z", "#2", "180 deg opposed", "length vote"):
        assert expected in text, (expected, text)
    # the warning has to travel with the calibration, not only in the structure
    assert any(w == text for w in calib.warnings), calib.warnings


def test_the_damage_it_prevents_is_a_clean_180():
    """Why this cannot be left to the residuals: there are none.

    Flipping one line on either of the two axes the rotation is built from turns
    the camera exactly half a turn -- a legal rotation, since flipping one axis
    flips the right-handed completion with it. The focal length, the
    orthogonality error and the vanishing points are all untouched, which is to
    say every number a reader would check is still good.

    The third axis is the quiet exception and worth pinning too: it is derived
    by cross product, so a reversed line there does not move the rotation at all
    -- until the photo is one of the many where the solve picks a different pair
    of axes, and then it does. The flag does not depend on which pair was used,
    which is the point of flagging the markup rather than the outcome.
    """
    rotations = {}
    for axis in (0, 1, 2):
        project, truth = build_project(max_lines_per_axis=2)
        im = project.images[0]
        clean = calibrate(im)
        flip_nth_line_on(im, axis=axis, n=0)   # the longest line, so it wins the vote
        spoiled = calibrate(im)

        assert spoiled.contested_axes == [axis], (axis, spoiled.contested_axes)
        # nothing a reader would check has moved
        assert abs(spoiled.intrinsics.f - clean.intrinsics.f) < 1e-6
        assert abs(spoiled.orthogonality_error_deg - clean.orthogonality_error_deg) < 1e-9

        rotations[axis] = (
            rotation_error_deg(clean.R, truth["R"][0]),
            rotation_error_deg(spoiled.R, truth["R"][0]),
        )

    used = sorted(calibrate(build_project(max_lines_per_axis=2)[0].images[0]).axes_used)[:2]
    for axis in used:
        before, after = rotations[axis]
        assert before < 1.0 and after > 175.0, (axis, before, after)
    third = ({0, 1, 2} - set(used)).pop()
    before, after = rotations[third]
    assert abs(after - before) < 1e-6, (third, before, after)


def test_the_solve_shouts_rather_than_notes_it():
    """A reversed line reaches report.warnings, which the CLI prints with a '!'.

    Deliberately not the per-image notes column where the rest of the
    calibration remarks go. Those are matters of degree about one photograph; a
    reversed arrow is a markup error that produces a confident wrong answer, and
    the notes column is where something with no residual goes unread.
    """
    clean, _truth = build_project(max_lines_per_axis=3)
    base = solve(clean)

    spoiled, _truth = build_project(max_lines_per_axis=3)
    # Y rather than X on purpose: the X lines come first in the photo's list, so
    # an axis-local number and a photo-local one coincide there and the fixture
    # could not tell a mix-up from a match.
    idx = flip_nth_line_on(spoiled.images[1], axis=1, n=1)
    report = solve(spoiled)

    new = [w for w in report.warnings if w not in base.warnings]
    assert len(new) == 1, new
    assert new[0].startswith("img1: Y:"), new[0]
    assert "reversed" in new[0]
    # the number in the message counts within the axis, as the panel does; the
    # photo's own index is a separate thing, carried only for the canvas
    assert "#2" in new[0] and f"#{idx + 1}" not in new[0], (idx, new[0])
    assert idx != 1, "fixture would not catch an axis/photo index mix-up"


def test_the_ui_payload_points_at_the_right_line_in_the_photo():
    """The index the browser gets is the photo's, not the axis's.

    An off-by-one here draws a white halo round an innocent arrow, which is a
    worse outcome than silence: it is a specific, confident accusation. Grouped
    lines are the hazard, since the sign check never sees them and every index
    after one shifts.
    """
    project, _truth = build_project(max_lines_per_axis=3)
    im = project.images[0]
    # a direction-group line in front of the axis lines, to shift every index
    im.axis_lines.insert(0, type(im.axis_lines[0])(
        axis=0, p0=[10.0, 10.0], p1=[200.0, 240.0], direction_group="tray"
    ))
    idx = flip_nth_line_on(im, axis=1, n=2)

    payload = observability_payload({
        "width": im.width,
        "height": im.height,
        "axis_lines": [
            {"axis": ln.axis, "p0": ln.p0, "p1": ln.p1, "direction_group": ln.direction_group}
            for ln in im.axis_lines
        ],
    })
    assert len(payload["reversed"]) == 1, payload["reversed"]
    entry = payload["reversed"][0]
    assert entry["name"] == "Y"
    # two numbers per line: the photo's own row, for the canvas, and the number
    # within the axis, which is the only one the user is ever shown
    assert entry["lines"] == [{"index": idx, "n": 3}], (entry["lines"], idx)
    assert idx not in [l["index"] for l in entry["agreeing"]]
    assert [l["n"] for l in entry["agreeing"]] == [1, 2], entry["agreeing"]
    assert f"#{entry['lines'][0]['n']} of 3" in entry["message"], entry["message"]
    assert entry["worst_opposition_deg"] > 175.0
    # and it is not duplicated into the observability advice beside it
    assert entry["message"] not in payload["problems"]


def test_a_consistently_reversed_axis_is_reported_as_what_it_is():
    """Reverse *every* line on an axis and this check cannot see it -- by design.

    That is a consistent choice, indistinguishable in one image from a camera
    turned around, so there is nothing in the lines to disagree about. It is the
    right-handed completion of three axes that catches it, and the point of this
    test is the blame: flip an X line and the completion check finds the *Z*
    axis disagreeing with it. Its old wording named Z and said "flip them",
    which sent the reader to correct the one axis that was right.
    """
    project, _truth = build_project(max_lines_per_axis=2)
    im = project.images[0]
    for n in range(2):
        flip_nth_line_on(im, axis=0, n=n)

    calib = calibrate(im)
    assert calib.contested_axes == [], "a unanimous reversal has nothing to disagree about"
    mirrored = [w for w in calib.warnings if "right-handed completion" in w]
    assert len(mirrored) == 1, calib.warnings
    assert "Which axis is reversed this cannot say" in mirrored[0], mirrored[0]

    # ...and with one X line reversed instead, the same warning now defers to the
    # axis whose own lines are arguing, rather than blaming Z
    project2, _truth2 = build_project(max_lines_per_axis=2)
    im2 = project2.images[0]
    flip_nth_line_on(im2, axis=0, n=0)
    calib2 = calibrate(im2)
    mirrored2 = [w for w in calib2.warnings if "right-handed completion" in w]
    assert len(mirrored2) == 1, calib2.warnings
    assert "The X control lines in this photo disagree" in mirrored2[0], mirrored2[0]


def test_the_sign_the_solver_picks_is_untouched():
    """The check is a second reading of a poll the solver already took.

    Two ways round. Algebraically, that the per-line votes sum to the sign
    ``resolve_direction_sign`` returns; and end to end, that the synthetic
    project solves to the same numbers it did before any of this existed. If
    either moved, the reading would be the wrong reading.
    """
    project, _truth = build_project()
    for im in project.images:
        for axis, segs in usable_segments(im).items():
            vp = estimate_vanishing_point(segs, im.width, im.height)
            cam = Intrinsics(f=3200.0, cx=im.width / 2.0, cy=im.height / 2.0)
            d = direction_from_vp(vp.v, cam)
            resolved = resolve_direction_sign(d, vp.v, segs)
            check = check_axis_signs(vp.v, segs, axis=axis)
            expected = math.copysign(1.0, sum(v.score for v in check.votes) or 1.0)
            assert check.sign == expected
            canonical = -d if (not vp.at_infinity and d[2] < 0.0) else d
            assert np.allclose(resolved, canonical * check.sign)

    report = solve(project)
    assert report.ok
    # the regression lock the rest of the suite uses for this fixture
    assert report.rms_reprojection_px < 1.2, report.rms_reprojection_px
    assert report.n_images_positioned == len(project.images)


def main() -> int:
    tests = [
        test_a_clean_markup_is_never_flagged,
        test_two_lines_either_side_of_a_near_vanishing_point_are_not_flagged,
        test_one_reversed_line_is_found_and_named,
        test_the_damage_it_prevents_is_a_clean_180,
        test_the_solve_shouts_rather_than_notes_it,
        test_the_ui_payload_points_at_the_right_line_in_the_photo,
        test_a_consistently_reversed_axis_is_reported_as_what_it_is,
        test_the_sign_the_solver_picks_is_untouched,
    ]
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

    print("\n--- what one reversed line costs ---")
    for axis in (0, 1, 2):
        project, truth = build_project(max_lines_per_axis=2)
        im = project.images[0]
        clean = calibrate(im)
        flip_nth_line_on(im, axis=axis, n=0)
        bad = calibrate(im)
        print(
            f"  reverse the longest {'XYZ'[axis]} line: rotation "
            f"{rotation_error_deg(clean.R, truth['R'][0]):5.2f} deg -> "
            f"{rotation_error_deg(bad.R, truth['R'][0]):6.2f} deg, "
            f"focal {clean.intrinsics.f:.0f} -> {bad.intrinsics.f:.0f} px, "
            f"flagged on {[('XYZ')[a] for a in bad.contested_axes]}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
