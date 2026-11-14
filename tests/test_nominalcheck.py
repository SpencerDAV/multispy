"""The shape check that catches a nominal linked to the wrong feature.

The scene is ``test_part``'s machined plate again, for the same reason
``test_uncertainty`` reuses it: the photographs, the marks and the drawing are
known-good, so anything this file provokes was provoked by the one thing it
changed -- which nominal is attached to which clicked point.

What is being defended here:

* **A swap is caught, and caught as a shape problem.** Exchanging the nominals
  of two corners leaves every mark exactly where it was and every coordinate a
  real coordinate off the drawing. Nothing is individually wrong; only the
  pairing is. If this file stops failing on a swap, the check has stopped
  doing the single thing it was written for.

* **The hard-lock trap, which is the real reason the module exists.** Give a
  swapped nominal a tight tolerance and an ordinary solve reports it *met*, to
  two microns, because the solve bent the scene to meet it -- see the deviation
  printed in the report below. The check has to go on saying "contradicted"
  about a project whose own datum table looks perfect, because it re-solves
  without the nominals first. A version that read the solved coordinates as it
  found them would pass this and be worthless.

* **Both readings earn their place.** The pair-ratio spread is fit-free and
  loud on most swaps, but a swap along the plate's 20 mm thickness barely moves
  it (about 6%) while the shape residual goes to forty times tolerance. Either
  one alone would have a blind spot the other covers.

* **It refuses to answer when it cannot.** Two points and a free scale fit each
  other whatever they are, and a collinear set has no shape across the line.
  Both must come back 'untestable' and not 'consistent' -- a check that returns
  a pass for input it cannot judge is worse than no check, because it is the
  pass somebody will quote.

* **It does not touch the project.** The re-solve is the whole method, so a
  copy that leaked back would silently strip the nominals off the caller's
  project.

* **A tolerance finer than the camera does not convict.** Grading against the
  stated figure alone made a correct link on a 0.5 mm feature read as 20 sigma,
  because nine phone photographs cannot place a point that finely. The verdict
  is taken against the larger of the drawing's tolerance and the reconstruction's
  own precision -- and the floor that makes that work must never be loose enough
  to excuse a real swap, which is asserted directly.

* **The precision follows the geometry, not just the pixels.** A point in two
  views is barely placed along the viewing direction however sharp its marks
  are. If that stops being true the floor is back to convicting weak geometry
  for being weak.

* **Drop-one-out names a single wrong link**, and stays silent when it cannot
  mean anything -- at exactly three points a removal leaves two, and two points
  agree with each other whatever they are.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_nominalcheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.nominalcheck import (
    COLLINEAR_LIMIT,
    MIN_POINTS,
    check_nominal_shape,
)
from multispy.solve import solve
from test_part import build_part_project

TIGHT_SIGMA = 0.01
"""A tolerance tight enough that the solve must meet the nominal exactly.

The point of it: at this sigma the nominal stops being evidence and becomes an
instruction, which is what ``Settings.sigma_nominal``'s 0.1 mm default does to
any feature the calibration file left untoleranced.
"""

_cache: dict = {}


def fixture(**kw):
    """The plate project, built once per distinct set of arguments."""
    key = json.dumps(kw, sort_keys=True)
    if key not in _cache:
        project, _ = build_part_project(**kw)
        _cache[key] = project
    import copy

    return copy.deepcopy(_cache[key])


def swapped(a: int, b: int, **kw):
    """The plate with the nominals of points ``a`` and ``b`` exchanged."""
    project = fixture(**kw)
    pa, pb = project.points[a], project.points[b]
    pa.nominal, pb.nominal = pb.nominal, pa.nominal
    return project


def test_correct_links_are_consistent() -> None:
    r = check_nominal_shape(fixture())
    assert r.ok, f"the fixture would not compare: {r.reason}"
    assert r.verdict == "consistent", (
        f"a correctly linked set came back {r.verdict!r} "
        f"(spread {r.spread:.3f}, rms {r.rms:.3f} mm, worst {r.worst_sigmas:.2f} sigma)"
    )
    assert r.spread < 1.1, f"pair ratios spread {r.spread:.3f}x on a correct set"
    assert r.n_compared >= MIN_POINTS


def test_a_swapped_link_is_contradicted() -> None:
    for a, b in ((0, 1), (0, 4), (1, 3), (2, 6)):
        r = check_nominal_shape(swapped(a, b))
        assert r.ok, f"p{a}<->p{b} would not compare: {r.reason}"
        assert r.verdict == "contradicted", (
            f"swapping p{a} and p{b} came back {r.verdict!r} "
            f"(spread {r.spread:.3f}, rms {r.rms:.2f} mm, "
            f"worst {r.worst_sigmas:.1f} sigma) -- a swap is the failure this exists for"
        )


def test_the_residual_catches_the_swap_the_spread_nearly_misses() -> None:
    """Both readings are reported because neither alone would do.

    p0 and p4 are the same corner top and bottom, 20 mm apart on a plate
    120 mm long, so exchanging them leaves the pair ratios almost undisturbed.
    The shape residual is what fires. Asserted so that nobody simplifies the
    report down to whichever number looks sufficient on the loud cases.
    """
    r = check_nominal_shape(swapped(0, 4))
    assert r.ok, r.reason
    assert r.spread < 1.2, (
        f"the thickness swap moved the spread to {r.spread:.3f}x, so this test no "
        f"longer covers the quiet case it was written for"
    )
    assert r.worst_sigmas > 10.0, (
        f"the shape residual let a 20 mm swap through at "
        f"{r.worst_sigmas:.1f} sigma while the spread was quiet"
    )


def test_a_hard_locked_wrong_link_still_contradicts() -> None:
    """The trap: a wrong nominal held tight reports itself satisfied.

    This is the shape of the real failure it was written after -- two
    untoleranced features pinned at the default sigma held their own deviation
    at zero and pushed the honestly toleranced points hundreds of millimetres.
    """
    project = swapped(0, 1, nominal_sigma=TIGHT_SIGMA)
    report = solve(project)
    assert report.ok, "the hard-locked project did not solve at all"

    deviations = []
    for pt in project.points[:2]:
        assert pt.xyz is not None, f"{pt.id} was not placed"
        deviations.append(
            float(np.linalg.norm(np.array(pt.xyz) * 1000.0 - np.array(pt.nominal)))
        )
    assert max(deviations) < 0.1, (
        f"the premise of this test has gone: the solve was supposed to meet the "
        f"swapped nominals exactly and left them {max(deviations):.3f} mm out, so it "
        f"is no longer demonstrating a check that reads solved coordinates would pass"
    )

    r = check_nominal_shape(project)
    assert r.ok, r.reason
    assert r.verdict == "contradicted", (
        f"a swap the solve reports as met to {max(deviations):.4f} mm came back "
        f"{r.verdict!r} -- the re-solve is not isolating the nominals"
    )


def test_two_points_cannot_be_tested() -> None:
    r = check_nominal_shape(fixture(nominals_on=2))
    assert not r.ok
    assert r.verdict == "untestable", (
        f"two nominals came back {r.verdict!r}; one distance against one free scale "
        f"always agrees, so a verdict here would be a word with nothing behind it"
    )
    assert str(MIN_POINTS) in r.reason


def test_a_collinear_set_is_not_graded() -> None:
    """Nominals strung along one line leave the directions across it untested.

    The points made collinear are the ones the nominal-free solve can actually
    place, read off a first pass. Picking by index instead would have tested
    the too-few-points guard by accident -- not every corner of this fixture is
    marked in enough photographs to triangulate.
    """
    placeable = [p.id for p in check_nominal_shape(fixture()).points if p.positioned]
    assert len(placeable) >= MIN_POINTS, "the fixture cannot place enough points to test this"

    project = fixture()
    for pt in project.points:
        if pt.id in placeable:
            pt.nominal = [30.0 * placeable.index(pt.id), 0.0, 0.0]
        else:
            pt.nominal = None

    r = check_nominal_shape(project)
    assert not r.ok
    assert r.verdict == "untestable", (
        f"a collinear nominal set came back {r.verdict!r} instead of declining"
    )
    assert r.collinearity < COLLINEAR_LIMIT
    assert "line" in r.reason


def test_unplaced_points_are_named_not_dropped() -> None:
    """A nominal'd point the photographs cannot place has to be said out loud.

    Silently comparing the rest would report a verdict about a smaller set than
    the reader thinks they are looking at.
    """
    project = fixture()
    target = next(pt for pt in project.points if pt.nominal)
    seen = 0
    for im in project.images:
        if target.id in im.observations:
            seen += 1
            if seen > 1:
                del im.observations[target.id]
    r = check_nominal_shape(project)

    row = next(p for p in r.points if p.id == target.id)
    assert not row.positioned, (
        f"{target.id} was left with one mark and still got a position"
    )
    assert any(target.id in w for w in r.warnings), (
        f"{target.id} could not be placed and no warning names it"
    )
    assert r.n_compared == len([p for p in r.points if p.positioned])


def test_it_does_not_modify_the_project() -> None:
    project = fixture()
    before = json.dumps(project.to_dict(), sort_keys=True, default=str)
    check_nominal_shape(project)
    after = json.dumps(project.to_dict(), sort_keys=True, default=str)
    assert before == after, (
        "check_nominal_shape changed the project it was given -- it strips every "
        "nominal to do its work, so a leak here deletes the caller's nominals"
    )


def test_a_tolerance_finer_than_the_camera_does_not_convict() -> None:
    """The bug this grading was changed to fix.

    The plate's nominals are right. Tightening what the drawing *asks* cannot
    make them wrong, because the photographs place these points to about 0.7 mm
    and nothing stated below that is a claim either way.
    """
    for tol in (0.5, 0.05, 0.001):
        r = check_nominal_shape(fixture(nominal_sigma=tol))
        assert r.ok, r.reason
        assert r.verdict == "consistent", (
            f"a correct set with a {tol} mm stated tolerance came back {r.verdict!r} "
            f"(rms {r.rms:.3f} mm against a precision floor of {r.precision:.3f} mm) -- "
            f"grading is demanding more than the camera can show"
        )
    assert r.n_floored == r.n_compared, "the floor should be governing every point at 0.001 mm"


def test_the_precision_floor_cannot_whitewash_a_swap() -> None:
    """The floor must excuse only what no camera could resolve.

    The other half of the fix above, and the more important half: a floor loose
    enough to pass a swapped link would have turned the whole check off while
    leaving it looking like it worked.
    """
    for tol in (0.5, 0.001):
        for a, b in ((0, 1), (0, 4)):
            r = check_nominal_shape(swapped(a, b, nominal_sigma=tol))
            assert r.ok, r.reason
            assert r.verdict == "contradicted", (
                f"p{a}<->p{b} at stated {tol} mm came back {r.verdict!r} -- the precision "
                f"floor ({r.precision:.3f} mm) is excusing a {r.rms:.1f} mm swap"
            )
            assert r.rms > 10.0 * r.precision, (
                f"a swap left {r.rms:.2f} mm against a {r.precision:.2f} mm floor; if those "
                f"ever come close the floor is too loose to separate them"
            )


def test_precision_tracks_how_many_views_place_the_point() -> None:
    """Two views place a point far worse than seven, at identical pixel error.

    Asserted because the first version of this floor used pixel error and depth
    alone, which is blind to the baseline -- it called a two-view point precise
    and then convicted it for not being.
    """
    r = check_nominal_shape(fixture())
    assert r.ok, r.reason
    project = fixture()
    seen = {}
    for im in project.images:
        for pid in im.observations:
            seen[pid] = seen.get(pid, 0) + 1
    rows = [(seen.get(q.id, 0), q.floor, q.id) for q in r.points if q.positioned]
    fewest = min(rows)
    most = max(rows)
    assert fewest[0] <= 2 and most[0] >= 5, (
        f"the fixture no longer spans few-view and many-view points: {rows}"
    )
    assert fewest[1] > 2.0 * most[1], (
        f"{fewest[2]} is in {fewest[0]} view(s) and {most[2]} in {most[0]}, yet their "
        f"precision floors are {fewest[1]:.3f} and {most[1]:.3f} mm -- the floor is not "
        f"seeing the baseline"
    )


def test_drop_one_out_names_a_single_wrong_link() -> None:
    """One displaced nominal, and the report should say which one.

    A displacement rather than a swap, because a swap makes two links wrong at
    once and the honest answer there is that no single removal explains it --
    which the next assertion covers.
    """
    placeable = [q.id for q in check_nominal_shape(fixture()).points if q.positioned]
    for target in placeable[:3]:
        project = fixture()
        q = next(x for x in project.points if x.id == target)
        q.nominal = [q.nominal[0] + 40.0, q.nominal[1], q.nominal[2]]
        r = check_nominal_shape(project)
        assert r.verdict != "consistent", f"a 40 mm displacement on {target} went unnoticed"
        assert r.suspect == target, (
            f"displaced {target} and the report named {r.suspect or 'nobody'}"
        )
        assert r.suspect_rms < r.rms, (
            f"{target} was named but leaving it out did not improve the fit "
            f"({r.suspect_rms:.2f} vs {r.rms:.2f} mm)"
        )

    # a swap is two wrong links, and should not be pinned on one point
    r = check_nominal_shape(swapped(0, 1))
    assert r.suspect == "", (
        f"a swap makes two links wrong, but the report pinned it on {r.suspect}"
    )


def test_drop_one_out_stays_silent_at_three_points() -> None:
    """At MIN_POINTS a removal leaves two, which agree whatever they are."""
    placeable = [q.id for q in check_nominal_shape(fixture()).points if q.positioned]
    project = fixture()
    keep = set(placeable[:MIN_POINTS])
    for x in project.points:
        if x.nominal and x.id not in keep:
            x.nominal = None
    q = next(x for x in project.points if x.id == placeable[0])
    q.nominal = [q.nominal[0] + 40.0, q.nominal[1], q.nominal[2]]

    r = check_nominal_shape(project)
    assert r.n_compared == MIN_POINTS, f"expected {MIN_POINTS} compared, got {r.n_compared}"
    assert r.verdict != "consistent", "the displacement should still be caught"
    assert r.suspect == "", (
        f"named {r.suspect} from a set of three -- dropping one leaves two, and any two "
        f"points fit each other, so every candidate would have 'cleared' it"
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

    good = check_nominal_shape(fixture())
    print(f"\nmachined plate, {good.n_compared} nominal'd points compared "
          f"(nominal-free re-solve {good.reproj_px:.2f} px)")
    print(f"{'linking':<22}{'verdict':<14}{'spread':>9}{'rms mm':>9}{'worst sigma':>13}")
    print(f"{'as drawn':<22}{good.verdict:<14}{good.spread:9.3f}{good.rms:9.2f}"
          f"{good.worst_sigmas:13.2f}")
    for a, b in ((0, 1), (0, 4), (1, 3), (2, 6)):
        r = check_nominal_shape(swapped(a, b))
        print(f"{f'p{a} <-> p{b} swapped':<22}{r.verdict:<14}{r.spread:9.3f}{r.rms:9.2f}"
              f"{r.worst_sigmas:13.1f}")

    locked = swapped(0, 1, nominal_sigma=TIGHT_SIGMA)
    solve(locked)
    dev = max(
        float(np.linalg.norm(np.array(pt.xyz) * 1000.0 - np.array(pt.nominal)))
        for pt in locked.points[:2]
    )
    r = check_nominal_shape(locked)
    print(f"\nthe same swap held at {TIGHT_SIGMA} mm: the solve reports it met to "
          f"{dev:.4f} mm,")
    print(f"and the shape check still says {r.verdict} at {r.rms:.1f} mm RMS.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
