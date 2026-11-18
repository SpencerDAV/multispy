"""Taking a known object's anchors out of the solve without deleting them.

The complaint this answers, in the user's words: the CAD bodies are *still*
breaking the solution after a good fit from tie points only. ``cadguard`` could
already measure that and could put the body-free cameras back -- but the marks
stayed live, so the next press of Solve re-made the damage. The restore was a
snapshot; what was missing was a decision.

So there are now two flags, :attr:`ObjectInstance.anchors_off` per placement and
:data:`Settings.anchors_in_solve` for the lot, and a solve stage,
:func:`place_muted_instances`, that fits a muted body to the finished cameras
instead of letting it move them.

What is being defended here, in order of how much it would cost to get wrong:

* **Nothing is deleted.** This is the entire difference between muting and the
  only way it could be done before. A muted body's marks survive the solve, the
  stale-mark sweep and a round trip through the file. Delete a click and it
  cannot be re-examined, re-clicked, or compared with what it used to be.

* **Muted means muted.** Not down-weighted, not loosened: out. No residual, no
  pose parameter, no vote in connectivity, the initialisation or the scale. The
  test for this compares against a project whose anchor marks were *deleted* and
  insists the two solves agree to floating-point noise, because "mostly out" is
  the failure mode that would be hardest to notice and most misleading.

* **A muted body is still placed.** If muting lost the placement it would be
  indistinguishable from deleting the marks, and nobody would use it. It is
  fitted against the final cameras, which makes its shape error a cleaner
  measurement than the voting version -- nothing had the chance to bend to meet
  it.

* **Muting never loses a placement that voting would have kept.** Found on the
  real rear-module cell: a stricter floor here than in ``fit_instances`` left
  five of twenty-eight bodies unplaced, which reads as "muting broke my bodies".

* **The report says so, every time.** A project quietly ignoring 582 of its 615
  clicks while printing a 2.99 px RMS is the most misleading state this program
  can reach. The number is true and the impression is false, so the state has to
  travel with the number.

* **It survives the next Solve.** That is the whole point, and it is the one
  property the previous answer did not have.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_muting.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.cadguard import mute_bodies
from multispy.project import Project, split_anchor_key
from multispy.solve import solve

from test_rig import build_rig_project, spoil_body

_cache: dict = {}


def rig(**kw) -> Project:
    key = tuple(sorted(kw.items()))
    if key not in _cache:
        project, _truth = build_rig_project(**kw)
        _cache[key] = project
    return copy.deepcopy(_cache[key])


def anchor_marks(project: Project) -> int:
    return sum(len(im.anchor_obs) for im in project.images)


def marks_deleted(project: Project) -> Project:
    """The old way, kept only as a yardstick: the clicks actually gone."""
    out = copy.deepcopy(project)
    for im in out.images:
        im.anchor_obs = {}
    return out


def all_muted(project: Project) -> Project:
    out = copy.deepcopy(project)
    out.settings.anchors_in_solve = False
    return out


# --------------------------------------------------------------------------
# nothing is deleted
# --------------------------------------------------------------------------


def test_muting_a_body_keeps_every_click() -> None:
    """The property the whole feature rests on.

    Before this there was exactly one way to stop a bad body: delete its marks.
    That throws away the evidence of what went wrong along with the damage, and
    it is irreversible -- you cannot re-click a corner against a mark that is no
    longer there to compare with.
    """
    project = rig()
    before = anchor_marks(project)
    assert before > 0, "fixture carries no anchor marks; nothing to test"

    project.instance("obj2").anchors_off = True
    solve(project)

    assert anchor_marks(project) == before, (
        "a solve must not consume a muted body's marks -- muting is not deleting"
    )
    assert project.instance("obj2").anchors_off is True, "and the decision survives the solve"


def test_the_stale_sweep_does_not_mistake_a_muted_body_for_a_missing_one() -> None:
    """The sharpest edge in this feature, and the one that would be silent.

    ``prune_stale_anchor_obs`` deletes marks whose object, model or anchor is
    gone, and it decides that from ``anchor_world_points``. If muting had been
    implemented by removing a body from that method -- the obvious place --
    then every muted body's marks would look stale and the first solve would
    delete them all. The user would have pressed a tickbox labelled "out of the
    solve" and lost an afternoon of clicking, with a message saying the solve
    had tidied up after them.
    """
    project = all_muted(rig())
    before = anchor_marks(project)
    removed = project.prune_stale_anchor_obs()
    assert removed == 0, f"the sweep took {removed} mark(s) from muted bodies"
    assert anchor_marks(project) == before

    report = solve(project)
    assert anchor_marks(project) == before, "and a full solve takes none either"
    assert not any("ignored" in m and "anchor" in m for m in report.messages), (
        f"nor should it claim to have ignored stale marks: {report.messages}"
    )


def test_the_flag_round_trips_through_the_file() -> None:
    """A decision that does not survive a reload is not a decision."""
    project = rig()
    project.instance("obj2").anchors_off = True
    project.settings.anchors_in_solve = True

    reloaded = Project.from_dict(project.to_dict())
    assert reloaded.instance("obj2").anchors_off is True
    assert reloaded.instance("obj1").anchors_off is False
    assert reloaded.settings.anchors_in_solve is True

    project.settings.anchors_in_solve = False
    assert Project.from_dict(project.to_dict()).settings.anchors_in_solve is False


def test_a_project_written_before_this_existed_still_votes() -> None:
    """Default on. An old file must not silently acquire a muted body."""
    raw = rig().to_dict()
    raw["settings"].pop("anchors_in_solve", None)
    for inst in raw["instances"]:
        inst.pop("anchors_off", None)

    old = Project.from_dict(raw)
    assert old.settings.anchors_in_solve is True
    assert not old.muted_instances()
    assert set(old.solving_anchor_points()) == set(old.anchor_world_points())


# --------------------------------------------------------------------------
# muted means out, not down-weighted
# --------------------------------------------------------------------------


def test_muting_every_body_matches_deleting_every_mark() -> None:
    """The test that says "out" rather than "mostly out".

    Deleting the marks is the unambiguous version of the same intent, so the two
    must land on the same cameras, the same points and the same focal. A
    discrepancy here would mean some stage still reads a muted mark -- which is
    the failure that would be hardest to see from the outside, since the solve
    would merely be slightly better than it should be.
    """
    project = rig()
    muted = solve(all_muted(project))
    deleted = solve(marks_deleted(project))

    assert muted.ok and deleted.ok
    assert abs(muted.rms_reprojection_px - deleted.rms_reprojection_px) < 1e-6, (
        f"{muted.rms_reprojection_px} vs {deleted.rms_reprojection_px}: something still "
        f"reads a muted mark"
    )
    assert muted.n_observations == deleted.n_observations, (
        "the two solves must be over the same observations"
    )
    for gid, f in muted.focals.items():
        assert abs(f - deleted.focals[gid]) < 1e-6, gid


def test_no_muted_mark_appears_as_a_residual() -> None:
    """Checked against the report rather than inferred from the RMS.

    The per-mark table is what the UI draws and what ``cadguard`` reads, so a
    muted mark surviving into it would be visible as a residual line in the
    panel for a body the panel also says is not voting.
    """
    project = rig()
    project.instance("obj2").anchors_off = True
    report = solve(project)

    labels = {row.get("point") for row in report.residual_lines}
    leaked = {lab for lab in labels if lab and split_anchor_key(lab)[0] == "obj2"}
    assert not leaked, f"muted anchors reached the residual table: {sorted(leaked)[:5]}"
    assert any(lab and split_anchor_key(lab)[0] == "obj1" for lab in labels), (
        "and the bodies that are voting must still be in it"
    )


def test_a_muted_body_contributes_no_scale() -> None:
    """Muting is not a free lunch, and this is the bill.

    A known object's own geometry is the only thing that makes a photogrammetric
    solve metric without a tape measure. Mute every body and that goes with
    them. Reported, loudly, rather than left for someone to discover from a
    dimension in the wrong units.
    """
    voting = solve(rig())
    assert voting.scale_is_metric, "the fixture's scale comes from its bodies"

    muted = solve(all_muted(rig()))
    assert not muted.scale_is_metric, "a muted body must not set the scale"
    assert muted.scale_source == "arbitrary", muted.scale_source
    assert any("no scale" in w for w in muted.warnings), (
        f"and the report has to say so: {muted.warnings}"
    )


# --------------------------------------------------------------------------
# a muted body is still placed
# --------------------------------------------------------------------------


def test_a_muted_body_is_still_placed_and_still_measured() -> None:
    """Otherwise muting would be indistinguishable from deleting the marks.

    The pose comes from triangulating its anchors against the *finished*
    cameras, which is a better measurement than the voting path gives: the
    cameras had no opportunity to move towards this body, so the shape error is
    the model against the photographs and nothing else.
    """
    project = rig()
    project.instance("obj2").anchors_off = True
    report = solve(project)

    rep = report.instances["obj2"]
    assert rep.anchors_off is True, "the report must carry the state, not just the instance"
    assert rep.placed, f"a muted body should still be placed: {rep.notes}"
    assert rep.n_anchors_marked > 0, "and still report the marks it carries"
    assert project.instance("obj2").R is not None
    assert project.instance("obj2").t is not None
    assert any("out of the solve" in n for n in rep.notes), rep.notes

    # and the placement is a real one: on a correctly marked body it agrees with
    # the model to about the click noise, not to within a body's width
    assert rep.shape_rms < 0.01, f"shape error {rep.shape_rms * 1000:.1f} mm is not a placement"


def test_a_muted_body_reports_a_real_anchor_rms_rather_than_zero() -> None:
    """``_fill_statistics`` builds that figure from the bundle's anchor blocks,
    and a muted body has none -- so it would sit at 0.00 px, which the panel
    prints as an anchor fit and which reads as flawless. The one number in the
    report that would have been a lie rather than an omission.

    It is also the more honest of the two readings: nothing was able to move a
    camera to make it smaller.
    """
    project = rig()
    project.instance("obj2").anchors_off = True
    report = solve(project)

    rep = report.instances["obj2"]
    assert rep.placed
    assert rep.n_observations > 0, "no anchor mark was measured against the placed pose"
    assert rep.rms_px > 0.0, "a muted body must not report a flawless anchor fit"
    assert rep.max_px >= rep.rms_px

    # and it is the same quantity a voting body reports, so the two are
    # comparable: on a well-marked body both land near the click noise
    voting = solve(rig()).instances["obj2"]
    assert abs(rep.rms_px - voting.rms_px) < 1.0, (
        f"muted {rep.rms_px:.3f} px vs voting {voting.rms_px:.3f} px -- these are supposed "
        f"to be the same measurement, differing only in what was allowed to move"
    )


def test_muting_loses_no_placement_that_voting_would_have_kept() -> None:
    """Found on the real rear-module cell, where a stricter floor here than in
    ``fit_instances`` left five of twenty-eight bodies unplaced -- which reads as
    "muting broke my bodies" and would have been reason enough not to use it.
    """
    voting = solve(rig())
    muted = solve(all_muted(rig()))

    kept = {i for i, v in voting.instances.items() if v.placed}
    still = {i for i, v in muted.instances.items() if v.placed}
    lost = sorted(kept - still)
    assert not lost, f"muting lost the placement of {lost}"


def test_locked_and_muted_keeps_the_pose_and_says_the_lock_is_inert() -> None:
    """The one combination that cannot both be meant.

    ``locked`` says this body defines the world frame, so its pose is not a
    thing to measure. Muting says its marks set nothing. Refitting it here would
    destroy the frame the whole scene is expressed in; silently honouring the
    lock would hide that ``fit_instances`` already skipped it and no frame is
    being taken from it at all. So: keep the pose, and say both.
    """
    project = rig()
    solve(project)                       # gives obj2 a pose to lock
    project.instance("obj2").locked = True
    project.instance("obj2").anchors_off = True
    report = solve(project)

    rep = report.instances["obj2"]
    assert rep.shape_rms == 0.0, (
        "nothing may refit a locked body: a shape error here means its pose was re-measured "
        "from the very anchors that are supposed to be setting nothing, and the frame the "
        "rest of the scene is expressed in was rewritten underneath it"
    )
    assert any("locked" in n and "out of the solve" in n for n in rep.notes), rep.notes
    assert any("locked" in w and "out of the solve" in w for w in report.warnings), (
        f"and the inert lock has to be named where it will be read: {report.warnings}"
    )

    # The pose is *carried* with the scene like any other, which is the point:
    # it moves only because the gauge it was supposed to be defining is now
    # coming from somewhere else -- which is exactly what the warning says.
    assert project.instance("obj2").R is not None
    assert project.instance("obj2").t is not None


def test_repeated_solves_do_not_walk_a_muted_body_across_the_scene() -> None:
    """The 2.9-metres-per-solve bug, in the one place it could come back.

    ``pose_in_part_space`` says whether a pose has already been through
    ``_transform_scene``. A muted body's pose is written after that transform
    has run, so the answer is yes -- and writing anything else would have the
    next solve move the body by the whole part-space offset, again, and again,
    with no residual anywhere to say so.
    """
    project = rig()
    project.instance("obj2").anchors_off = True
    solve(project)
    first = project.instance("obj2").t.copy()
    for _ in range(2):
        solve(project)
    drift = float(np.linalg.norm(project.instance("obj2").t - first))
    assert drift < 1e-6, f"the body moved {drift * 1000:.3f} mm across three solves"


# --------------------------------------------------------------------------
# the state is visible, and it lasts
# --------------------------------------------------------------------------


def test_the_report_says_what_it_is_ignoring() -> None:
    """A clean RMS over a third of the evidence is worse than a bad one over all
    of it, unless it says which it is."""
    project = all_muted(rig())
    report = solve(project)

    assert report.n_muted_instances == len(project.placed_object_instances())
    assert report.n_muted_anchor_marks == anchor_marks(project) > 0
    said = " ".join(report.messages)
    assert "out of the solve" in said, f"no message names the state: {report.messages}"
    assert str(report.n_muted_anchor_marks) in said, "and it should name the number of clicks"
    assert "Nothing was deleted" in said or "still there" in said, (
        "and reassure that the clicks are still there, since that is the first worry"
    )


def test_a_voting_project_says_nothing_about_muting() -> None:
    """No noise in the normal case."""
    report = solve(rig())
    assert report.n_muted_instances == 0
    assert report.n_muted_anchor_marks == 0
    assert not any("out of the solve" in m for m in report.messages)


def test_mute_bodies_is_a_decision_the_next_solve_respects() -> None:
    """The difference from ``restore_without_bodies``, which is the difference
    the whole feature exists for: that one puts a camera solution back and
    leaves the marks voting, so Solve undoes it."""
    project = spoil_body(rig())
    said, report = mute_bodies(project)

    assert report is not None and report.ok, said
    assert "out of the solve" in said, said
    assert project.settings.anchors_in_solve is False
    assert anchor_marks(project) > 0, "and it kept the clicks"

    rms = report.rms_reprojection_px
    again = solve(project)
    assert abs(again.rms_reprojection_px - rms) < 1e-6, (
        "a second Solve must not let the bodies back in -- that is exactly what "
        "restoring the cameras did wrong"
    )


def test_mute_bodies_can_take_one_body_rather_than_all_of_them() -> None:
    """The granularity the failure actually comes in: one anchor clicked onto
    the wrong one of two similar corners, while the same model placed four other
    times is marked perfectly."""
    project = rig()
    said, report = mute_bodies(project, instances=["obj2"])

    assert report is not None and report.ok, said
    assert project.instance("obj2").anchors_off is True
    assert all(
        project.instance(f"obj{i}").anchors_off is False for i in (0, 1, 3, 4)
    ), "muting one body must not mute the others"
    assert project.settings.anchors_in_solve is True, "nor reach for the project-wide switch"
    assert report.n_muted_instances == 1, report.n_muted_instances


def test_mute_bodies_says_so_rather_than_pretending_when_there_is_nothing_to_do() -> None:
    project = all_muted(rig())
    said, report = mute_bodies(project)
    assert "already" in said, said
    assert report is None, "and it must not spend a solve saying nothing changed"


def test_unmuting_is_the_same_switch_back() -> None:
    """Reversibility is the claim the tickbox makes; it has to be true."""
    project = rig()
    reference = solve(copy.deepcopy(project)).rms_reprojection_px

    project.settings.anchors_in_solve = False
    solve(project)
    project.settings.anchors_in_solve = True
    back = solve(project)

    assert abs(back.rms_reprojection_px - reference) < 1e-6, (
        f"{back.rms_reprojection_px} vs {reference}: letting the bodies back in did not "
        f"restore the solve they were in"
    )


# --------------------------------------------------------------------------
# what it is worth, measured rather than asserted
# --------------------------------------------------------------------------


def test_muting_undoes_the_damage_a_bad_body_does_to_the_photographs() -> None:
    """The user's actual complaint, on the fixture that can reproduce it.

    The rig's own defences -- per-body reweighting and the outlier cull -- already
    absorb a bad body's effect on the *cameras*, which is why ``cadguard`` returns
    a quiet verdict there. What they cannot do is stop the body's own badly
    fitting marks from being in the solve at all, so the project's reprojection
    RMS carries them. Muting is what takes them out.
    """
    sloppy = spoil_body(rig(), mode="sloppy")
    voting = solve(copy.deepcopy(sloppy))
    muted = solve(all_muted(sloppy))

    assert voting.ok and muted.ok
    assert muted.rms_reprojection_px < voting.rms_reprojection_px / 2, (
        f"{voting.rms_reprojection_px:.3f} px -> {muted.rms_reprojection_px:.3f} px is not "
        f"the improvement muting is supposed to buy"
    )


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    # ---- the readable part: what muting costs and what it buys
    print("\nthe camera-array rig, five known bodies")
    print(f"{'marking':<28}{'rms px':>9}{'culled':>8}{'placed':>8}{'scale':>12}")
    rows = [
        ("as marked", rig(), False),
        ("as marked, all muted", rig(), True),
        ("obj2 clicked carelessly", spoil_body(rig(), mode="sloppy"), False),
        ("careless, obj2 muted", spoil_body(rig(), mode="sloppy"), "obj2"),
        ("careless, all muted", spoil_body(rig(), mode="sloppy"), True),
    ]
    for label, project, mute in rows:
        if mute is True:
            project.settings.anchors_in_solve = False
        elif isinstance(mute, str):
            project.instance(mute).anchors_off = True
        r = solve(project)
        placed = sum(1 for v in r.instances.values() if v.placed)
        scale = "metric" if r.scale_is_metric else "arbitrary"
        print(
            f"{label:<28}{r.rms_reprojection_px:>9.3f}{r.n_culled:>8}{placed:>8}{scale:>12}"
        )

    print("\nMuting takes a body's marks out of the solve and the solve's RMS stops")
    print("carrying them; every body is still placed, against the finished cameras.")
    print("The cost is in the last column: a known object's own geometry is what")
    print("makes this solve metric, and muting them all gives that up.")
    print("\nOn real data (rearmod3: 8 photos, 28 bodies, 582 anchor marks) muting")
    print("every body took the fit from 9.37 px to 2.99 px and the culled marks from")
    print("15 to 2 -- and left the bodies' agreement with CAD where it was, 3.49 mm")
    print("to 3.74 mm on one yardstick for both arms. It buys back the camera")
    print("solution, not accuracy. Both halves of that belong in the same sentence.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
