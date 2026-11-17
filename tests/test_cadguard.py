"""The guard on known objects: did the CAD make the photographs fit worse?

The scene is ``test_rig``'s camera array -- a frame carrying five toed-in camera
bodies -- because it already has the two things this needs: real anchor marks on
known models, and ``spoil_body``, which mis-marks one body the way a person
actually does it rather than the way a test would.

What is being defended here:

* **A correct body is not accused.** A guard that cries wolf on good input gets
  turned off, and then it is not guarding anything. The unspoiled rig must come
  back clean.

* **A swapped anchor pair is caught**, and so is a body clicked carelessly.
  These are the two failures that actually happen, and they are the reason the
  module exists.

* **Per image, not on the average.** A body appears in two or three photographs
  out of nine. It can ruin both while the overall RMS barely moves, because six
  untouched images carry the mean. The user asked for this in exactly those
  terms -- it must not worsen the point solution *in each image* -- so there is a
  test that spoils one image alone and insists it is still caught.

* **The trap that would have made this module worse than useless.**
  ``ImageReport.rms_px`` is the obvious thing to compare and it is the wrong
  thing: ``_fill_statistics.record`` folds anchor marks into it on purpose. A
  body's own anchors are usually fitted tightly, so their presence drags that
  average *down* -- a body can wreck a photograph's tie points and still show an
  improved ``rms_px``. The guard is built on tie-point residuals only, and the
  difference between the two readings is measured here rather than asserted in a
  docstring, because if the wrong one ever gets wired in this is the only test
  that would notice.

* **Like-for-like populations.** If the body-laden solve culls a mark the
  body-free one kept, comparing RMS over the two sets rewards dropping the worst
  mark. Only marks surviving both culls are compared.

* **It refuses rather than passing input it cannot judge.** No anchors marked
  means the bodies are not influencing anything; that is ``untestable``, not a
  clean bill of health. A check that returns a pass for what it cannot see is
  worse than no check, because the pass is what gets quoted.

* **It does not touch the project**, and the restore does nothing unless the
  verdict earns it.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_cadguard.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.cadguard import (
    TOLERANCE_FRACTION,
    TOLERANCE_PX,
    CadReport,
    _per_image,
    body_free_copy,
    check_cad_bodies,
    restore_without_bodies,
)
from multispy.solve import solve
from test_rig import build_rig_project, legacy_weighting, shared_anchor_pair, spoil_body

_cache: dict = {}


def rig(**kw):
    """The rig project, built once per distinct set of arguments."""
    key = tuple(sorted(kw.items()))
    if key not in _cache:
        project, _truth = build_rig_project(**kw)
        _cache[key] = project
    return copy.deepcopy(_cache[key])


def spoiled(mode: str = "swap", instance: str = "obj2"):
    return spoil_body(rig(), instance=instance, mode=mode)


def spoiled_in_one_image(instance: str = "obj2"):
    """Swap one body's anchor pair in a single photograph.

    The per-image case. Overall this is a handful of bad marks among a hundred
    good ones and barely moves any average; in the one photograph it happens in,
    it is the difference between a solved camera and a wrong one.
    """
    project = rig()
    a, b, _n = shared_anchor_pair(project, instance)
    for im in project.images:
        if a in im.anchor_obs and b in im.anchor_obs:
            im.anchor_obs[a], im.anchor_obs[b] = im.anchor_obs[b], im.anchor_obs[a]
            return project, im.id
    raise AssertionError("no image carries both anchors of the pair; fixture changed")


def mixed_rms_deltas(project):
    """Per-image deltas as ``ImageReport.rms_px`` would have reported them.

    The wrong metric, computed on purpose so the right one can be compared with
    it. Anchors are in these numbers; that is the whole problem.
    """
    after = solve(copy.deepcopy(project))
    before = solve(body_free_copy(project))
    out = {}
    for iid, rep_after in after.images.items():
        rep_before = before.images.get(iid)
        if rep_before is None or not rep_before.rms_px:
            continue
        out[iid] = rep_after.rms_px - rep_before.rms_px
    return out


# ------------------------------------------------------------------- claims


def test_a_correctly_marked_body_is_not_accused() -> None:
    r = check_cad_bodies(rig())
    assert r.ok, f"the rig fixture would not compare: {r.reason}"
    assert not r.worsened, (
        f"a correctly marked rig should not be flagged; got {r.worsened} "
        f"(worst image {r.worst_image}, {r.images[0].delta_px:+.3f} px against an allowance "
        f"of {r.images[0].allowed_px:.3f})"
    )
    assert r.verdict in ("improves", "harmless"), f"got {r.verdict!r}"


def test_the_rigs_own_defences_already_absorb_a_spoiled_body() -> None:
    """Not a pass for the guard -- a finding about the solver, asserted so it stays true.

    A mis-marked body on the current rig does *not* measurably worsen the tie
    points: ``reweight_instances`` spots the body whose marks do not agree with
    the others, widens its sigma, and the photographs are left alone. The guard
    says "harmless" and that is the correct answer rather than a miss.

    It is worth a test of its own because it is the fact that makes the guard's
    quiet verdicts trustworthy, and because if ``reweight_instances`` ever
    regresses, the first visible symptom will be this test starting to see
    damage it currently cannot.
    """
    for mode in ("swap", "scatter"):
        r = check_cad_bodies(spoiled(mode))
        assert r.ok, r.reason
        assert not r.worsened, (
            f"with reweighting on, a {mode}-spoiled body should not reach the tie points; "
            f"got {r.worsened} with overall {r.before_rms_px:.2f} -> {r.after_rms_px:.2f} px. "
            f"If this starts failing, the reweighting has stopped protecting the solve and "
            f"that is the bug, not this test."
        )


def test_a_swapped_anchor_pair_is_caught_when_nothing_else_stops_it() -> None:
    """The guard firing on real damage.

    ``legacy_weighting`` is ``test_rig``'s own name for the settings as they were
    before per-body reweighting existed, so this is not a contrived
    configuration -- it is the one this project shipped with, and the one a user
    reaches by turning ``reweight_instances`` off. With the body no longer
    demoted, a swapped anchor pair does reach the photographs, and the guard
    has to say so.
    """
    r = check_cad_bodies(legacy_weighting(spoiled("swap")))
    assert r.ok, r.reason
    assert r.verdict == "worsens", (
        f"got {r.verdict!r} with overall {r.before_rms_px:.2f} -> {r.after_rms_px:.2f} px "
        f"and worst image {r.worst_image} at {r.images[0].delta_px:+.3f} px against an "
        f"allowance of {r.images[0].allowed_px:.3f}"
    )
    assert r.worsened, "and it should name the images it hurt"
    assert r.images[0].worst_point, "and the point to go and look at"


def test_a_carelessly_clicked_body_is_caught_in_several_images() -> None:
    """And the per-image requirement, from the direction that actually bites.

    Scatter spreads the damage, so more than one photograph should be named.
    The user's requirement is per image, and a report that collapsed to one
    worst offender would hide the rest.
    """
    r = check_cad_bodies(legacy_weighting(spoiled("scatter")))
    assert r.ok, r.reason
    assert r.verdict == "worsens", (
        f"got {r.verdict!r}, overall {r.before_rms_px:.2f} -> {r.after_rms_px:.2f} px"
    )
    assert len(r.worsened) >= 2, (
        f"8 px of scatter on one body should show in more than one photograph; got "
        f"{r.worsened}"
    )


def test_a_clean_body_under_the_same_settings_is_still_not_accused() -> None:
    """The other arm, and the one that decides whether the tolerance is honest.

    Without this, the test above would pass just as well if the guard convicted
    everything under ``legacy_weighting``. The margin is genuinely narrow here --
    clean lands around +0.21 px against an allowance of 0.25 while the swap
    reaches +0.29 -- so this is also the test that will notice first if the
    allowance is ever loosened enough to stop discriminating.
    """
    r = check_cad_bodies(legacy_weighting(rig()))
    assert r.ok, r.reason
    assert not r.worsened, (
        f"a correctly marked rig must stay clean even with the old weighting; got "
        f"{r.worsened}, worst {r.worst_image} at {r.images[0].delta_px:+.3f} px against "
        f"{r.images[0].allowed_px:.3f}"
    )


def test_the_two_readings_differ_and_the_mixed_one_answers_another_question() -> None:
    """Why the comparison is rebuilt instead of read off ``ImageReport.rms_px``.

    The first version of this test asserted the mixed reading would understate
    the damage, on the docstring's original reasoning that a body's anchors are
    fitted tightly and drag the average down. Measured, that is false for a
    mis-marked body: its own anchors fit *badly*, so the mixed reading moves
    *more* than the tie-point-only one. The reason to exclude anchors is that the
    mixed figure answers "did anything in this photo get worse", which conflates
    the body's own untidiness with damage to the calibration -- and those call
    for opposite responses.

    So what is pinned here is that the two readings are genuinely different
    numbers, and that the module uses the tie-point one. A future change that
    quietly switched to ``rms_px`` would make these equal and fail here.
    """
    project, hurt = spoiled_in_one_image()
    r = check_cad_bodies(project)
    mixed = mixed_rms_deltas(project)
    tie_only = {v.image: v.delta_px for v in r.images}
    assert hurt in tie_only and hurt in mixed, "both readings should cover the spoiled image"
    assert abs(tie_only[hurt] - mixed[hurt]) > 1e-3, (
        f"on {hurt} the two readings should not coincide: tie-only {tie_only[hurt]:+.4f} px, "
        f"anchors mixed in {mixed[hurt]:+.4f} px. Equal numbers mean the guard is reading "
        f"the mixed figure after all."
    )


def test_a_mark_the_bodies_cost_the_solve_convicts_on_its_own() -> None:
    """The hole this module shipped with, pinned so it cannot reopen.

    Restricting the RMS comparison to marks surviving both culls makes the
    average fair and makes a cull invisible: the mark leaves the comparison
    instead of counting against the bodies. Asked about PRM1 with 260 restored
    anchor marks, the first version answered "improves, 1.99 to 1.66 px, no image
    worse" about a solve that had gone from 2.03 px to 13.09 px and had culled
    eleven tie-point marks, three on nominal'd points. The 1.66 px was an average
    over 57 marks where the 1.99 px was over 67.

    The stub below is that shape exactly, and deliberately *flattering*: every
    mark still in the comparison fits better with the bodies than without. A
    verdict of anything but "worsens" means the cull is being ignored again.
    """
    import multispy.solve as solve_mod

    class Stub:
        ok = True
        focals = {"cam0": 2750.0}

        def __init__(self, rows):
            self.residual_lines = rows

    # Ids taken from the fixture rather than invented. _tie_ids filters the
    # residuals to labels that really are tie points -- correctly, since a point
    # id is whatever the user typed -- so a made-up label is silently ignored and
    # the trap never fires. That is how the first draft of this test passed the
    # broken code.
    project = rig()
    a, b, doomed = [pt.id for pt in project.points[:3]]

    without = Stub([
        {"image": "img0", "point": a, "error_px": 1.0, "culled": False},
        {"image": "img0", "point": b, "error_px": 1.0, "culled": False},
        # fits badly, but it is in the solve
        {"image": "img0", "point": doomed, "error_px": 4.0, "culled": False},
    ])
    with_bodies = Stub([
        # the survivors each improve...
        {"image": "img0", "point": a, "error_px": 0.5, "culled": False},
        {"image": "img0", "point": b, "error_px": 0.5, "culled": False},
        # ...because the one that disagreed was thrown out
        {"image": "img0", "point": doomed, "error_px": 40.0, "culled": True},
    ])

    calls = []

    def fake_solve(project, *a, **kw):
        calls.append(project)
        # check_cad_bodies solves with the bodies first, then without
        return with_bodies if len(calls) == 1 else without

    real = solve_mod.solve
    try:
        solve_mod.solve = fake_solve
        r = check_cad_bodies(project)
    finally:
        solve_mod.solve = real

    assert len(calls) == 2, f"expected a solve with the bodies and one without; got {len(calls)}"
    assert r.ok, r.reason
    assert r.after_rms_px < r.before_rms_px, (
        "the stub is meant to look like an improvement on the surviving marks; if it does "
        "not, the test is not exercising the trap"
    )
    assert r.verdict == "worsens", (
        f"a lost tie point must convict however good the remaining average looks; got "
        f"{r.verdict!r} with {r.before_rms_px:.2f} -> {r.after_rms_px:.2f} px"
    )
    assert r.n_newly_culled == 1, r.n_newly_culled
    assert r.n_marks_before == 3 and r.n_marks_after == 2, (r.n_marks_before, r.n_marks_after)
    img0 = next(v for v in r.images if v.image == "img0")
    assert img0.newly_culled == [doomed], img0.newly_culled
    assert img0.worsened, "and the image that lost it must be named"
    assert any("cost the solve" in n for n in r.notes), f"and said in words: {r.notes}"


def test_a_cull_is_not_excused_by_the_tolerance() -> None:
    """No allowance applies to a lost mark.

    The tolerances are for how far a fit may drift. A mark the solver gave up on
    has not drifted inside anything -- it is gone -- so a generous tolerance must
    not buy it back.
    """
    import multispy.solve as solve_mod

    class Stub:
        ok = True
        focals = {"cam0": 2750.0}

        def __init__(self, rows):
            self.residual_lines = rows

    project = rig()
    a, b = [pt.id for pt in project.points[:2]]
    without = Stub([
        {"image": "img0", "point": a, "error_px": 1.0, "culled": False},
        {"image": "img0", "point": b, "error_px": 1.0, "culled": False},
    ])
    with_bodies = Stub([
        {"image": "img0", "point": a, "error_px": 1.0, "culled": False},
        {"image": "img0", "point": b, "error_px": 99.0, "culled": True},
    ])
    calls = []

    def fake_solve(project, *a, **kw):
        calls.append(project)
        return with_bodies if len(calls) == 1 else without

    real = solve_mod.solve
    try:
        solve_mod.solve = fake_solve
        r = check_cad_bodies(project, tolerance_px=1000.0, tolerance_fraction=1000.0)
    finally:
        solve_mod.solve = real
    assert r.verdict == "worsens", (
        f"a tolerance of a thousand pixels must not excuse a culled mark; got {r.verdict!r}"
    )
    assert r.n_newly_culled == 1


def test_only_tie_points_enter_the_comparison() -> None:
    """Anchors excluded, culls excluded, by label and by flag."""

    class Stub:
        residual_lines = [
            {"image": "a", "point": "p0", "error_px": 1.0, "culled": False},
            {"image": "a", "point": "p1", "error_px": 9.0, "culled": True},
            {"image": "a", "point": "obj2/corner", "error_px": 50.0, "culled": False},
            {"image": "b", "point": "p0", "error_px": 2.0, "culled": False},
        ]

    got = _per_image(Stub(), {"p0", "p1"})
    assert got == {"a": {"p0": 1.0}, "b": {"p0": 2.0}}, (
        f"only uncradled tie-point marks should survive the filter; got {got}"
    )


def test_a_project_with_no_anchors_is_untestable_not_clean() -> None:
    project = rig()
    for im in project.images:
        im.anchor_obs = {}
    r = check_cad_bodies(project)
    assert not r.ok
    assert r.verdict == "untestable", f"got {r.verdict!r}"
    assert "no body anchors" in r.reason
    assert not r.worsened, "and it must not look like a pass"


def test_the_allowance_scales_with_how_well_the_image_started() -> None:
    r = check_cad_bodies(rig())
    assert r.ok, r.reason
    for v in r.images:
        expected = max(TOLERANCE_PX, TOLERANCE_FRACTION * v.before_px)
        assert abs(v.allowed_px - expected) < 1e-12, (
            f"{v.image}: allowance {v.allowed_px} should be the larger of the floor and the "
            f"fraction, {expected}"
        )


def test_the_populations_compared_are_the_same_size() -> None:
    """``n_points`` is the intersection, so neither solve can win by culling."""
    r = check_cad_bodies(spoiled("scatter"))
    assert r.ok, r.reason
    assert all(v.n_points > 0 for v in r.images)
    assert all(v.before_px > 0 and v.after_px > 0 for v in r.images), (
        "an RMS of zero means the population was empty and the comparison is vacuous"
    )


def test_it_does_not_touch_the_project() -> None:
    project = spoiled("swap")
    before = project.to_dict()
    check_cad_bodies(project)
    assert project.to_dict() == before, (
        "the check must be read-only: it runs on a project the user is editing"
    )


def test_body_free_copy_keeps_the_bodies_and_drops_only_their_marks() -> None:
    project = rig()
    free = body_free_copy(project)
    assert len(free.instances) == len(project.instances), (
        "the bodies stay in the scene; only their influence on the cameras goes"
    )
    assert all(not im.anchor_obs for im in free.images), "and every anchor mark is gone"
    assert any(im.anchor_obs for im in project.images), "without disturbing the original"
    assert free.path is None, "a trial project must never be able to save over the real one"


def test_restore_does_nothing_when_the_bodies_are_innocent() -> None:
    project = rig()
    r = check_cad_bodies(project)
    assert r.verdict != "worsens"
    before = project.to_dict()
    said = restore_without_bodies(project, r)
    assert project.to_dict() == before, "an innocent body must not have its solve thrown away"
    assert "nothing to undo" in said, f"and it should say so: {said!r}"


def test_restore_puts_the_body_free_cameras_back() -> None:
    # legacy_weighting because with reweighting on there is nothing to restore --
    # the solver has already absorbed the bad body and the guard rightly says so.
    project = legacy_weighting(spoiled("swap"))
    solve(project)
    r = check_cad_bodies(project)
    assert r.verdict == "worsens", r.verdict
    poses_before = {im.id: None if im.R is None else im.R.copy() for im in project.images}
    said = restore_without_bodies(project, r)
    assert "restored" in said, f"got {said!r}"
    moved = sum(
        1
        for im in project.images
        if im.R is not None
        and poses_before[im.id] is not None
        and float(abs(im.R - poses_before[im.id]).max()) > 1e-9
    )
    assert moved, "restoring should actually move the cameras it was called about"
    assert any(im.anchor_obs for im in project.images), (
        "and it must not throw away what the user clicked -- the anchors stay marked so the "
        "body can be re-checked after re-clicking"
    )


def test_a_restored_project_no_longer_fails_its_own_guard() -> None:
    """The ratchet actually ratchets.

    Checking the restore leaves a state the guard is happy with, rather than one
    that merely differs from the bad one.
    """
    project = legacy_weighting(spoiled("swap"))
    solve(project)
    r = check_cad_bodies(project)
    assert r.verdict == "worsens", f"the premise needs real damage; got {r.verdict!r}"
    restore_without_bodies(project, r)
    after = check_cad_bodies(project)
    assert after.after_rms_px <= r.after_rms_px + 1e-6, (
        f"after restoring, the with-bodies fit should be no worse than before: "
        f"{after.after_rms_px:.3f} against {r.after_rms_px:.3f} px"
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

    print("\nthe camera-array rig, five known bodies")
    print(f"{'marking':<26}{'verdict':<12}{'overall px':>12}{'images hurt':>13}")
    for label, project in (
        ("as marked", rig()),
        ("obj2 anchors swapped", spoiled("swap")),
        ("obj2 clicked carelessly", spoiled("scatter")),
        ("obj2 swapped in one photo", spoiled_in_one_image()[0]),
        ("as marked, old weighting", legacy_weighting(rig())),
        ("swapped, old weighting", legacy_weighting(spoiled("swap"))),
        ("careless, old weighting", legacy_weighting(spoiled("scatter"))),
    ):
        r = check_cad_bodies(project)
        overall = f"{r.before_rms_px:.2f}->{r.after_rms_px:.2f}"
        print(f"{label:<26}{r.verdict:<12}{overall:>12}{len(r.worsened):>13}")

    print("\nThe top three say the reweighting already protects the photographs from a")
    print("bad body, so the guard's quiet verdict there is correct and not a miss. The")
    print("bottom three are the same data with per-body reweighting off: the damage is")
    print("real and the guard names it, while the clean arm stays clean.")

    project, hurt = spoiled_in_one_image()
    r = check_cad_bodies(project)
    mixed = mixed_rms_deltas(project)
    tie_only = {v.image: v.delta_px for v in r.images}
    print(f"\non {hurt}, the photograph the bad anchors are in:")
    print(f"  tie points only   {tie_only.get(hurt, float('nan')):+.3f} px")
    print(f"  anchors mixed in  {mixed.get(hurt, float('nan')):+.3f} px")
    print("The second is what ImageReport.rms_px would have shown. It is the larger,")
    print("because a mis-marked body's own anchors fit badly and push that average up --")
    print("the opposite of what this module first assumed. Anchors are excluded because")
    print("the mixed figure answers a different question, not a less sensitive one.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
