"""The principal point belongs to a frame, not to a camera group.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_principalpoint.py

A camera group is a promise that these photographs share intrinsics, and for
the focal length that is exactly true: the same lens at the same zoom gives the
same focal length in pixels however the camera was held. The principal point is
not like that. It is a point *in the picture*, and turning the phone on its
side moves it from (2000, 924) to (924, 2000) without the lens changing at all.

Until this was fixed the bundle did three things that compounded into a large,
silent error on any set shot partly in portrait and partly in landscape:

  * it seeded the group's principal point from whichever image happened to come
    first in the group -- a coin toss between the two orientations;
  * it wrote that seed back into the project on every solve, locked or not, so
    a value nobody measured became a value the file asserted;
  * it then handed that one pixel coordinate to every image in the group, which
    for half of them put the principal point a thousand pixels off the sensor.

Nothing anywhere complained, because a principal point in the wrong place does
not produce a residual of its own -- the camera positions and point depths
simply bend around it. It was found by noticing that a group whose photographs
were 1848x4000 carried cx=2000 on a picture 1848 pixels wide.

So the properties worth defending: that an unsolved principal point is never
written to the file, that a stored one is only used in the frame it was
measured in, that an old project carrying a fabricated one is repaired on the
way in, and -- the half most likely to break something -- that a group whose
photographs are all the same size behaves exactly as it did before.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.project import (  # noqa: E402
    CameraGroup,
    ImageView,
    Project,
    repair_principal_points,
)

TMP = Path(tempfile.mkdtemp(prefix="multispy_pp_"))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def mixed_project(cx=None, cy=None, pp_frame=None) -> Project:
    """Four photographs off one lens, two held upright and two on their side."""
    p = Project(name="mixed")
    p.groups["cam0"] = CameraGroup(id="cam0", f=1625.0, cx=cx, cy=cy, pp_frame=pp_frame)
    for i, (w, h) in enumerate([(1848, 4000), (1848, 4000), (4000, 1848), (4000, 1848)]):
        p.images.append(
            ImageView(id=f"img{i}", path=f"photos/{i}.jpg", width=w, height=h, group="cam0")
        )
    return p


# ------------------------------------------------------------------ the point


def test_centre_when_nothing_was_measured() -> None:
    """With no stored principal point, every image gets its own centre."""
    g = CameraGroup(id="cam0")
    check("portrait centre", g.principal_point_for(1848, 4000) == (924.0, 2000.0))
    check("landscape centre", g.principal_point_for(4000, 1848) == (2000.0, 924.0))


def test_a_stored_point_applies_only_to_its_own_frame() -> None:
    """The frame it was measured in gets it; a differently shaped one does not."""
    g = CameraGroup(id="cam0", cx=2010.0, cy=930.0, pp_frame=[4000, 1848])
    check(
        "its own frame keeps the measured point",
        g.principal_point_for(4000, 1848) == (2010.0, 930.0),
        str(g.principal_point_for(4000, 1848)),
    )
    check(
        "the rotated frame falls back to its centre",
        g.principal_point_for(1848, 4000) == (924.0, 2000.0),
        str(g.principal_point_for(1848, 4000)),
    )


def test_a_point_off_the_picture_is_refused() -> None:
    """Even with no frame recorded, a point outside the image cannot be one."""
    g = CameraGroup(id="cam0", cx=2000.0, cy=924.0)
    check(
        "cx beyond the width is ignored",
        g.principal_point_for(1848, 4000) == (924.0, 2000.0),
        str(g.principal_point_for(1848, 4000)),
    )


# ------------------------------------------------------- repairing old files


def test_repair_adopts_the_frame_it_must_have_come_from() -> None:
    """(2000, 924) is the centre of the landscape shots and nothing else."""
    p = mixed_project(cx=2000.0, cy=924.0)
    notes = repair_principal_points(p)
    g = p.groups["cam0"]
    check("the point is kept", (g.cx, g.cy) == (2000.0, 924.0))
    check("the frame is recovered", g.pp_frame == [4000, 1848], str(g.pp_frame))
    check("it says so", len(notes) == 1 and "4000x1848" in notes[0], str(notes))
    check(
        "the portrait shots now use their own centre",
        g.principal_point_for(1848, 4000) == (924.0, 2000.0),
    )


def test_repair_clears_a_point_from_no_frame_at_all() -> None:
    """A point near no member's centre was never measured on any of them."""
    p = mixed_project(cx=120.0, cy=90.0)
    notes = repair_principal_points(p)
    g = p.groups["cam0"]
    check("it is cleared", g.cx is None and g.cy is None, f"{g.cx}, {g.cy}")
    check("it says so", len(notes) == 1 and "never measured" in notes[0], str(notes))


def test_repair_leaves_a_recorded_frame_alone() -> None:
    """Nothing to infer once the frame is written down."""
    p = mixed_project(cx=2010.0, cy=930.0, pp_frame=[4000, 1848])
    notes = repair_principal_points(p)
    check("untouched", (p.groups["cam0"].cx, p.groups["cam0"].cy) == (2010.0, 930.0))
    check("silent", notes == [], str(notes))


def test_repair_is_silent_on_a_single_shape_group() -> None:
    """The ordinary case: one orientation, nothing ambiguous, no noise."""
    p = Project(name="plain")
    p.groups["cam0"] = CameraGroup(id="cam0", f=1600.0, cx=2000.0, cy=924.0)
    for i in range(3):
        p.images.append(
            ImageView(id=f"img{i}", path=f"{i}.jpg", width=4000, height=1848, group="cam0")
        )
    notes = repair_principal_points(p)
    check("point kept", (p.groups["cam0"].cx, p.groups["cam0"].cy) == (2000.0, 924.0))
    check("frame recorded", p.groups["cam0"].pp_frame == [4000, 1848])
    check("nothing said", notes == [], str(notes))


def test_repair_runs_on_load() -> None:
    """A project read off disk arrives repaired, and says what it did."""
    p = mixed_project(cx=2000.0, cy=924.0)
    path = TMP / "mixed.multispy"
    p.save(path)
    again = Project.load(path)
    check("frame recovered through the file", again.groups["cam0"].pp_frame == [4000, 1848])
    check("the note survives for the solve to report", len(again.load_notes) == 1)
    check("load_notes is not saved", "load_notes" not in again.to_dict())


def test_pp_frame_survives_a_round_trip() -> None:
    p = mixed_project(cx=2010.0, cy=930.0, pp_frame=[4000, 1848])
    path = TMP / "frame.multispy"
    p.save(path)
    check("read back", Project.load(path).groups["cam0"].pp_frame == [4000, 1848])


# ------------------------------------------------------------- the whole solve


def test_the_bundle_gives_each_image_its_own_centre() -> None:
    """_Problem resolves the principal point per image, not per group."""
    from multispy.geometry import Intrinsics
    from multispy.solve import _Problem

    p = mixed_project(cx=2000.0, cy=924.0, pp_frame=[4000, 1848])
    for im in p.images:
        im.R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        im.t = [0.0, 0.0, 1.0]
    ids = [im.id for im in p.images]
    intr = {i: Intrinsics(f=1625.0, cx=0.0, cy=0.0) for i in ids}
    prob = _Problem(p, intr, ids, [])
    check("portrait image centred on itself", prob.image_pp["img0"] == (924.0, 2000.0))
    check("landscape image keeps the measured point", prob.image_pp["img2"] == (2000.0, 924.0))


def test_a_locked_principal_point_is_never_written_back() -> None:
    """The fault itself: write_back saved a point it had only ever guessed."""
    from multispy.geometry import Intrinsics
    from multispy.solve import _Problem

    p = mixed_project()  # nothing stored, and lock_principal_point defaults True
    for im in p.images:
        im.R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        im.t = [0.0, 0.0, 1.0]
    ids = [im.id for im in p.images]
    intr = {i: Intrinsics(f=1625.0, cx=0.0, cy=0.0) for i in ids}
    prob = _Problem(p, intr, ids, [])
    prob.write_back(prob.pack())
    g = p.groups["cam0"]
    check("still unset after a solve", g.cx is None and g.cy is None, f"{g.cx}, {g.cy}")
    check("and no frame was invented", g.pp_frame is None, str(g.pp_frame))


def test_a_mixed_group_refuses_to_solve_a_shared_point() -> None:
    """One pair of pixel coordinates cannot describe two frame shapes."""
    from multispy.geometry import Intrinsics
    from multispy.solve import _Problem

    p = mixed_project()
    p.groups["cam0"].lock_principal_point = False
    for im in p.images:
        im.R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        im.t = [0.0, 0.0, 1.0]
    ids = [im.id for im in p.images]
    intr = {i: Intrinsics(f=1625.0, cx=0.0, cy=0.0) for i in ids}
    prob = _Problem(p, intr, ids, [])
    check("no slot allocated", "cx" not in prob.group_slots["cam0"])
    check(
        "and it says why",
        any("more than one size" in n for n in prob.setup_notes),
        str(prob.setup_notes),
    )


def test_one_shape_group_still_solves_its_point() -> None:
    """The guard must not take the ordinary refinement down with it."""
    from multispy.geometry import Intrinsics
    from multispy.solve import _Problem

    p = Project(name="plain")
    p.groups["cam0"] = CameraGroup(id="cam0", f=1600.0, lock_principal_point=False)
    for i in range(3):
        p.images.append(
            ImageView(id=f"img{i}", path=f"{i}.jpg", width=4000, height=1848, group="cam0")
        )
    for im in p.images:
        im.R = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        im.t = [0.0, 0.0, 1.0]
    ids = [im.id for im in p.images]
    intr = {i: Intrinsics(f=1600.0, cx=0.0, cy=0.0) for i in ids}
    prob = _Problem(p, intr, ids, [])
    check("slot allocated", "cx" in prob.group_slots["cam0"])
    check("nothing to warn about", prob.setup_notes == [], str(prob.setup_notes))
    prob.write_back(prob.pack())
    g = p.groups["cam0"]
    check("solved point is written back", (g.cx, g.cy) == (2000.0, 924.0), f"{g.cx}, {g.cy}")
    check("with the frame it was measured in", g.pp_frame == [4000, 1848])


def main() -> int:
    for fn in [
        test_centre_when_nothing_was_measured,
        test_a_stored_point_applies_only_to_its_own_frame,
        test_a_point_off_the_picture_is_refused,
        test_repair_adopts_the_frame_it_must_have_come_from,
        test_repair_clears_a_point_from_no_frame_at_all,
        test_repair_leaves_a_recorded_frame_alone,
        test_repair_is_silent_on_a_single_shape_group,
        test_repair_runs_on_load,
        test_pp_frame_survives_a_round_trip,
        test_the_bundle_gives_each_image_its_own_centre,
        test_a_locked_principal_point_is_never_written_back,
        test_a_mixed_group_refuses_to_solve_a_shared_point,
        test_one_shape_group_still_solves_its_point,
    ]:
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all principal-point checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
