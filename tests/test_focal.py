"""The wall around the focal length, and why a wall and not a weight.

The focal length is the one parameter every image in a camera group shares. That
makes it the cheapest thing in the problem for a badly marked known object to
spend: a rigid body whose anchors were clicked onto the wrong features cannot
flex to absorb its own error, so it looks for somewhere else to put it, and
shrinking the focal makes every projection in every photograph stop having an
opinion at once. The body buys a fit and nine photographs pay for it.

This was found twice on the same cell, a week apart, both times on a nine-photo
solve that had been healthy at 2.37 px. What it looked like:

* the group focal went from **2786 px** -- measured from 61 control lines across
  nine images, to a stated +-2.6% -- down to **1.6 px**;
* the solve reported **converged**, at a reprojection RMS of 924 px, with
  **nothing culled**, because at that focal every mark is equally wrong;
* ``reweight_instances`` had already noticed and widened that body to **569 px**,
  and it made no difference at all.

That last point is what rules out the obvious fix. Down-weighting the body is
not enough and cannot be made enough, because the body's gain from collapsing
the focal is unbounded while the tie points defending it are each capped by the
robust kernel at the Huber knee. Sixty-eight bounded objections lose to one
unbounded demand however quietly it is asked.

What settles the design is the cost. Re-running that same broken project with
the focal simply held still finishes at a cost of **3557**; letting it run free
"converges" at **1.04e5**, thirty times worse. So the collapse is not a cheaper
answer that the solver rationally preferred and that a prior would have to
out-bid -- it is a local minimum it fell into and could not climb out of, and
the honest instrument for a trap is a wall, not a tax. Hence
``Settings.focal_bound_factor``: a generous multiplicative window around the
focal the control lines measured, clamped where the focal is consumed and
penalised where it is parameterised, exactly the bargain ``_project_block``
already strikes with points that drift behind a camera.

What is actually being defended here:

* **The wall is inert on a healthy solve.** A guard that quietly changes good
  answers is worse than the bug. The same scene solved with the bound on and
  off has to agree to a hair, and on the real nine-photo project it agrees
  bit-for-bit.
* **The wall holds at the extreme.** Not just a plausible mis-click but anchors
  scattered at random across the frame, which is the worst a correspondence can
  be, has to leave the focal positive and the right order of magnitude.
* **The wall does not hide the body.** The failure this came from was dangerous
  because it looked like convergence. A guarded solve must still say loudly that
  the body is wrong -- a wide per-body sigma, a bad reprojection, culled marks --
  or the next person loses a day to it the way the last two did.
* **The wall is not doing the solving.** With the bound in place the focal has to
  come back *right*, not merely bounded, or all that has been achieved is a
  prettier wrong answer pinned to a fence.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_focal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.project import ObjectAnchor, ObjectInstance, ObjectTemplate, Project
from multispy.solve import solve

from test_part import build_part_project

# A clamp: a plain box in millimetres, with its eight corners offered as anchors.
# No mesh file, the same way ``test_rig``'s bodies have none -- a template's
# anchor list is all the solve ever reads.
CLAMP_MM = np.array(
    [
        [0, 0, 0],
        [300, 0, 0],
        [300, 200, 0],
        [0, 200, 0],
        [0, 0, 150],
        [300, 0, 150],
        [300, 200, 150],
        [0, 200, 150],
    ],
    dtype=float,
)
ANCHOR_NAMES = [f"c{i}" for i in range(8)]

# Where the eight marks land relative to each other in ``wrong_feature`` mode.
# Roughly a box in projection, so the marks look like a plausible reading of the
# body's outline rather than nonsense -- the error is which corner is which, not
# that the clicking was careless.
OUTLINE_PX = np.array(
    [[0, 0], [400, 0], [400, 260], [0, 260], [60, -190], [460, -190], [460, 70], [60, 70]],
    dtype=float,
)

MARGIN_PX = 900.0
"""Keeps generated marks clear of the frame edge, so nothing is rejected for
being off-image and every arm of every test carries the same 8 marks per photo."""


def add_mismarked_body(
    project: Project, mode: str = "wrong_feature", seed: int = 3
) -> Project:
    """Add a known object whose anchors are marked on the wrong features.

    Two modes, and the distinction matters to what each test is entitled to
    claim:

    ``wrong_feature``
        The eight marks trace a believable outline, but no mark is on the corner
        it names. This is the real failure and it is a mundane one: the cell this
        came from is near-symmetric left to right and carries fifteen sensors of
        three repeated models, so clicking the identical clamp one bay over is a
        mistake anyone makes and nothing on screen contradicts.

    ``scattered``
        Every mark at an independent random point in the frame. Nobody clicks
        like this; it is here because a wall is only worth having if it holds at
        the extreme. Unguarded, this mode takes the focal to hundredths of a
        pixel, and on some draws straight through zero to a negative value -- a
        camera turned inside out, which no amount of down-weighting is going to
        argue its way back from.
    """
    project.templates["tplX"] = ObjectTemplate(
        id="tplX",
        name="clamp",
        units="mm",
        unit_scale=0.001,
        anchors=[
            ObjectAnchor(name=n, xyz=p.tolist()) for n, p in zip(ANCHOR_NAMES, CLAMP_MM)
        ],
    )
    project.instances.append(ObjectInstance(id="objX", template="tplX", label="clamp 1"))

    rng = np.random.default_rng(seed)
    for im in project.images:
        if mode == "scattered":
            marks = np.column_stack(
                [
                    rng.uniform(MARGIN_PX, im.width - MARGIN_PX, len(ANCHOR_NAMES)),
                    rng.uniform(MARGIN_PX, im.height - MARGIN_PX, len(ANCHOR_NAMES)),
                ]
            )
        elif mode == "wrong_feature":
            corner = np.array(
                [
                    rng.uniform(MARGIN_PX, im.width - MARGIN_PX),
                    rng.uniform(MARGIN_PX, im.height - MARGIN_PX),
                ]
            )
            marks = corner + OUTLINE_PX
        else:
            raise ValueError(f"unknown mode {mode!r}")
        for name, uv in zip(ANCHOR_NAMES, marks):
            im.anchor_obs[f"objX/{name}"] = [float(uv[0]), float(uv[1])]
    return project


def solved_focal(bound: float, mode: str | None = None, seed: int = 3):
    """One solve of the plate scene, optionally with a mis-marked body on it.

    ``bound`` goes straight into ``focal_bound_factor``, so 0 reproduces the
    behaviour before the wall existed exactly -- same code path, same scene, one
    setting -- and the two arms of these tests differ in nothing else.
    """
    project, truth = build_part_project()
    if mode is not None:
        add_mismarked_body(project, mode=mode, seed=seed)
    project.settings.focal_bound_factor = bound
    report = solve(project)
    return report.focals.get("cam0", 0.0), truth["focal"], report, project


SEEDS = (1, 2, 3, 4, 5)
SCATTER_SEEDS = (1, 2, 3)
"""Fewer, because the scattered case asserts only survival and each solve is a
few seconds. The realistic case is the one that earns five."""


def true_focal() -> float:
    return float(build_part_project()[1]["focal"])


# --------------------------------------------------------------------------
# the bug
# --------------------------------------------------------------------------


def test_a_mismarked_body_cannot_collapse_the_focal():
    """The headline. One body marked on the wrong features, nothing else wrong
    with the scene, and the shared focal has to survive it.

    Both arms are asserted. The unguarded one has to actually collapse, or this
    test is measuring nothing and would keep passing if the wall were removed.
    """
    unguarded = [solved_focal(0.0, "wrong_feature", seed)[0] for seed in SEEDS]
    guarded = [solved_focal(1.5, "wrong_feature", seed) for seed in SEEDS]
    true = guarded[0][1]

    # the trap is real: without the wall the focal is gone, not merely off
    assert all(f < true / 50 for f in unguarded), unguarded

    for f, _true, _rep, _proj in guarded:
        assert f > 0, f
        assert true / 1.5 - 1e-6 <= f <= true * 1.5 + 1e-6, (f, true)

    # ...and it comes back *right*, not just inside the fence. A single seed may
    # sit against the wall; the body is garbage, after all. Typical behaviour is
    # what is being claimed, so this reads the median.
    median = float(np.median([f for f, _t, _r, _p in guarded]))
    assert abs(median - true) < 0.05 * true, (median, true)


def test_the_wall_holds_against_the_worst_a_correspondence_can_be():
    """Anchors scattered at random across the frame.

    Asserted separately from the realistic case because the two are different
    claims. There the focal has to come back *correct*; here it only has to come
    back *usable*. A body of eight marks with nothing behind them is not
    evidence about a lens, and the win is that the calibration is merely dragged
    rather than destroyed.
    """
    unguarded = [solved_focal(0.0, "scattered", seed)[0] for seed in SCATTER_SEEDS]
    assert all(f < true_focal() / 1000 for f in unguarded), unguarded

    for seed in SCATTER_SEEDS:
        f, true, _rep, _proj = solved_focal(1.5, "scattered", seed)
        assert f > 0, (seed, f)
        assert true / 1.5 - 1e-6 <= f <= true * 1.5 + 1e-6, (seed, f, true)


# --------------------------------------------------------------------------
# the guard must cost nothing and hide nothing
# --------------------------------------------------------------------------


def test_the_wall_is_inert_on_a_clean_solve():
    """A clean scene has to solve to the same focal with the bound on and off.

    This is the assertion that earns the guard the right to be on by default. A
    wall that nudges good answers is a bias, not a guard, and the whole case for
    a bound over a prior is that inside the window there is no force at all.
    """
    free, true, free_rep, _p = solved_focal(0.0)
    walled, _true, walled_rep, _p2 = solved_focal(1.5)

    assert abs(walled - free) < 1e-6 * true, (walled, free)
    assert abs(walled_rep.rms_reprojection_px - free_rep.rms_reprojection_px) < 1e-6, (
        walled_rep.rms_reprojection_px,
        free_rep.rms_reprojection_px,
    )
    # and the clean solve is nowhere near its own wall, so the agreement above is
    # not two runs both pinned to the same fence
    assert abs(free - true) < 0.1 * true, (free, true)


def test_the_wall_does_not_hide_the_bad_body():
    """A guarded solve must still accuse the body.

    The reason this bug cost two days is that it did not look like a bug: the
    report said converged. Protecting the focal while leaving that impression
    intact would be the more dangerous of the two failures, because the user
    would then trust a solve whose body is still in the wrong place.
    """
    _f, _true, report, project = solved_focal(1.5, "wrong_feature", seed=1)

    body = report.instances["objX"]
    assert body.sigma_px > 5 * project.settings.sigma_anchor_px, body.sigma_px
    assert body.rms_px > 50.0, body.rms_px
    assert report.rms_reprojection_px > 20.0, report.rms_reprojection_px
    assert any("disagree" in note for note in body.notes), body.notes


def test_a_focal_outside_the_window_never_reaches_the_project():
    """The clamp lives in ``intrinsics_at``, which is what ``write_back`` reads,
    so a focal outside the window cannot be reported or saved even transiently."""
    _f, true, report, project = solved_focal(1.5, "scattered", seed=2)
    saved = Project.from_dict(project.to_dict(), path=project.path)

    for source in (project.groups["cam0"].f, saved.groups["cam0"].f, report.focals["cam0"]):
        assert source is not None
        assert true / 1.5 - 1e-6 <= float(source) <= true * 1.5 + 1e-6, source


def test_the_bound_is_a_setting_and_says_so():
    """The window is on by default, it round-trips, and it is the thing doing
    the work.

    Someone whose control lines are genuinely poor has to be able to widen it,
    and someone reading the project file afterwards has to be able to see that
    they did -- a guard that cannot be turned off is a guard nobody can diagnose.
    A wider setting is not asserted to produce a worse answer: the window is a
    limit on where the focal may go, not a prediction of where it lands.
    """
    project, _truth = build_part_project()
    assert project.settings.focal_bound_factor == 1.5, project.settings.focal_bound_factor

    project.settings.focal_bound_factor = 4.0
    again = Project.from_dict(project.to_dict(), path=project.path)
    assert again.settings.focal_bound_factor == 4.0, again.settings.focal_bound_factor

    # the window really is where the limit comes from: widen it and the focal is
    # free to sit outside the tighter one it would otherwise have been held to
    tight, true, _rep, _p = solved_focal(1.5, "scattered", seed=2)
    wide, _true, _rep2, _p2 = solved_focal(6.0, "scattered", seed=2)

    # What the window promises, and all it promises: the focal it returns is
    # inside the window it was given. That is deterministic, because it is a
    # clamp and not an outcome.
    assert true / 1.5 - 1e-6 <= tight <= true * 1.5 + 1e-6, (tight, true)
    assert true / 6.0 - 1e-6 <= wide <= true * 6.0 + 1e-6, (wide, true)

    # This used to assert that the 6x run *left* the 1.5x window, on the reasoning
    # that otherwise the wall was not what had been holding it. That contradicted
    # the paragraph above -- "a wider setting is not asserted to produce a worse
    # answer" -- and it was an assertion about where an optimiser lands, which is
    # not a property of this code. It passed on Windows and failed on Linux CI at
    # 1916.9 px against a true 2100, comfortably inside the tighter window: a
    # different scipy found a better answer without needing the wall, which is
    # good news being reported as a regression.
    #
    # That the wall is load-bearing is already established, and established far
    # more sharply, by test_a_mismarked_body_cannot_collapse_the_focal: it sets
    # focal_bound_factor to 0 and watches the focal collapse to single digits. So
    # nothing is lost here beyond a false alarm, and no replacement assertion is
    # put in its place: there is nothing true to say about where the wide run
    # lands, and a tautology dressed as a check is worse than an honest absence.


def main() -> int:
    tests = [
        test_a_mismarked_body_cannot_collapse_the_focal,
        test_the_wall_holds_against_the_worst_a_correspondence_can_be,
        test_the_wall_is_inert_on_a_clean_solve,
        test_the_wall_does_not_hide_the_bad_body,
        test_a_focal_outside_the_window_never_reaches_the_project,
        test_the_bound_is_a_setting_and_says_so,
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

    print("\n--- what one mis-marked body does to the shared focal ---")
    clean, true, _rep, _p = solved_focal(0.0)
    print(f"true focal {true:.1f} px; clean solve {clean:.1f} px\n")
    print(f"{'marking':<15}{'seed':>5}{'no wall':>12}{'wall 1.5':>12}{'reproj px':>12}")
    for mode in ("wrong_feature", "scattered"):
        for seed in SCATTER_SEEDS:
            off, _t, _r, _p = solved_focal(0.0, mode, seed)
            on, _t2, rep, _p2 = solved_focal(1.5, mode, seed)
            print(f"{mode:<15}{seed:>5}{off:>12.2f}{on:>12.2f}{rep.rms_reprojection_px:>12.1f}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
