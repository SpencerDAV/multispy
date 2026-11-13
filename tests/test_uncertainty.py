"""The confidence sphere around the part-space origin.

The scene is ``test_part``'s machined plate, reused unchanged: the same photos,
the same marks, the same drawing. The only thing these tests vary is *where the
origin sits* relative to the features that pin it -- which costs nothing to
change, since moving every nominal by a constant moves the origin and leaves
the geometry, the solve and the residuals exactly as they were.

That is the whole point. What is being defended here:

* **The sphere follows the lever arm, not the residuals.** A plate measured
  from its own datum corner knows where its origin is. The same plate, same
  marks, same reprojection RMS, dimensioned from a corner a metre away does
  not -- and nothing in the solve report says so. If this file ever stops
  failing when the lever grows, the sphere has stopped measuring the thing it
  exists to measure.

* **A perfect datum table is not knowledge of the origin.** The nominals come
  back met to a fraction of their tolerance in both cases, because the fit puts
  them there by construction. The sphere has to be free to disagree.

* **It scales with the noise it is given.** Doubling the assumed click accuracy
  roughly doubles the sphere. A figure that ignored ``sigma_px`` would be
  reporting the scene's shape and calling it an uncertainty.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_uncertainty.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.project import Project
from multispy.solve import solve
from multispy.uncertainty import (
    MIN_SAMPLES,
    OriginUncertainty,
    fingerprint,
    origin_uncertainty,
)
from test_part import build_part_project

SAMPLES = 12
"""Few, because every one is a full re-solve. Enough to separate a near origin
from a far one by a factor of several, which is all these assertions ask of it;
far too few for the radius itself to be quoted to two figures."""

LEVER_MM = 1200.0
"""Ten times the plate's longest side -- the drawing datum is off the part, the
way a vehicle's origin is off any single bracket."""

_cache: dict = {}


def solved(lever_mm: float = 0.0, sigma_px: float | None = None) -> Project:
    """The plate, solved, with its origin ``lever_mm`` away along every axis.

    Shifting every nominal by one constant is a pure change of where zero is:
    the marks, the photos and the part are untouched, so anything that moves in
    the report moved because of the origin's position and nothing else.
    """
    key = (lever_mm, sigma_px)
    if key not in _cache:
        project, _ = build_part_project()
        if sigma_px is not None:
            project.settings.sigma_px = sigma_px
        if lever_mm:
            for pt in project.points:
                if pt.nominal:
                    pt.nominal = [v + lever_mm for v in pt.nominal]
        report = solve(project)
        assert report.ok, f"the fixture failed to solve: {report.stage}"
        _cache[key] = project
    return copy.deepcopy(_cache[key])


def measure(lever_mm: float = 0.0, sigma_px: float | None = None, seed: int = 3):
    key = ("u", lever_mm, sigma_px, seed)
    if key not in _cache:
        _cache[key] = origin_uncertainty(solved(lever_mm, sigma_px), n=SAMPLES, seed=seed)
    return _cache[key]


# ------------------------------------------------------------------ the claim

def test_the_sphere_grows_with_the_lever_arm():
    near, far = measure(0.0), measure(LEVER_MM)
    assert near.n >= MIN_SAMPLES, f"only {near.n} samples converged"
    assert far.n >= MIN_SAMPLES, f"only {far.n} samples converged"
    ratio = far.radius / max(near.radius, 1e-9)
    assert ratio > 2.0, (
        f"an origin {LEVER_MM:.0f} mm off the part came back only {ratio:.1f}x less "
        f"certain than one on it ({near.radius:.3f} -> {far.radius:.3f} mm)"
    )


def test_the_solve_report_does_not_notice_the_lever_at_all():
    """The counterpart: this is *why* the sphere has to exist.

    Same marks, same photos, so the reprojection RMS is identical to the digit.
    Anyone reading solve quality off the report would call these two projects
    equally good, and for the origin they are not.
    """
    a, b = solve(solved(0.0)), solve(solved(LEVER_MM))
    assert abs(a.rms_reprojection_px - b.rms_reprojection_px) < 0.01, (
        "the fixture is not holding the solve constant across the lever, so the "
        "comparison above is not clean"
    )


def test_a_perfect_datum_table_does_not_mean_a_known_origin():
    project = solved(LEVER_MM)
    report = solve(project)
    worst = max((abs(c.error) for c in report.constraints), default=0.0)
    far = measure(LEVER_MM)
    assert worst * 1000.0 < 1.0, f"fixture nominals are not being met ({worst*1000:.2f} mm)"
    assert far.radius > 5.0 * worst * 1000.0, (
        f"the datum table is met to {worst*1000:.3f} mm while the origin is good to "
        f"{far.radius:.3f} mm -- if these track each other the sphere is just "
        f"restating the residuals"
    )


def test_noisier_marks_make_a_bigger_sphere():
    """Monotonic in the assumed click accuracy -- but a long way from linear.

    Eight times the sigma buys well under twice the sphere, and that is the
    solver being sensible rather than the measure being deaf. ``sigma_px`` sets
    how far the marks are jittered *and* how much the bundle trusts them, so
    loosening it hands the frame to the nominals, whose own tolerance has not
    moved. The nominals are a floor the sphere cannot fall through, which is
    worth knowing: past a point, clicking more carefully stops helping and the
    drawing tolerance is what is left.
    """
    base = measure(LEVER_MM, sigma_px=0.5)
    loud = measure(LEVER_MM, sigma_px=4.0)
    assert loud.radius > 1.25 * base.radius, (
        f"eight times the click sigma moved the sphere from {base.radius:.2f} to "
        f"{loud.radius:.2f} mm, which is not a response to the noise at all"
    )
    assert loud.radius < 4.0 * base.radius, (
        f"the sphere tracked the click sigma almost linearly ({base.radius:.2f} -> "
        f"{loud.radius:.2f} mm), which means the nominals have stopped holding the "
        f"frame and something has changed in how they are weighted"
    )


def test_the_scatter_is_reported_as_an_ellipsoid_too():
    far = measure(LEVER_MM)
    assert len(far.sigma) == 3 and len(far.axes) == 3
    assert far.sigma[0] >= far.sigma[1] >= far.sigma[2] >= 0.0, "sigmas are not ordered"
    for ax in far.axes:
        assert abs(np.linalg.norm(ax) - 1.0) < 1e-6, "principal axes are not unit vectors"
    assert far.radius > far.sigma[0], "a 95% ball cannot be smaller than one sigma"


def test_the_bias_is_small_against_the_spread():
    """A Monte Carlo around the solution should scatter, not drift."""
    far = measure(LEVER_MM)
    assert np.linalg.norm(far.bias) < far.radius, (
        f"the samples are centred {np.linalg.norm(far.bias):.2f} mm off against a "
        f"{far.radius:.2f} mm sphere -- that is a pull, not noise"
    )


# ------------------------------------------------------------ the bookkeeping

def test_the_fingerprint_notices_an_edited_mark():
    project = solved(LEVER_MM)
    before = fingerprint(project)
    image = next(im for im in project.images if im.observations)
    pid = sorted(image.observations)[0]
    image.observations[pid] = [image.observations[pid][0] + 3.0, image.observations[pid][1]]
    assert fingerprint(project) != before, "moving a mark left the fingerprint unchanged"


def test_the_fingerprint_notices_an_edited_nominal():
    project = solved(LEVER_MM)
    before = fingerprint(project)
    pt = next(p for p in project.points if p.nominal)
    pt.nominal = [pt.nominal[0] + 1.0, pt.nominal[1], pt.nominal[2]]
    assert fingerprint(project) != before, "editing a nominal left the fingerprint unchanged"


def test_it_round_trips_through_the_project_file():
    project = solved(LEVER_MM)
    far = measure(LEVER_MM)
    project.origin_uncertainty = far.to_dict()
    back = Project.from_dict(project.to_dict())
    again = OriginUncertainty.from_dict(back.origin_uncertainty)
    assert again is not None
    assert abs(again.radius - far.radius) < 1e-9
    assert again.fingerprint == far.fingerprint
    assert again.grade(1e9, 2e9) == "good" and again.grade(0.0, 0.0) == "bad"


def test_a_thin_sample_grades_as_unknown():
    thin = OriginUncertainty(n=MIN_SAMPLES - 1, radius=0.001)
    assert thin.grade(1.0, 2.0) == "unknown", (
        "too few samples must not be allowed to read as a good result"
    )


def test_it_refuses_outside_part_space():
    project, _ = build_part_project(nominals_on=0)
    solve(project)
    try:
        origin_uncertainty(project, n=2)
    except ValueError as exc:
        assert "part space" in str(exc)
    else:
        raise AssertionError("measured an origin for a project that has no origin")


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
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    near, far = measure(0.0), measure(LEVER_MM)
    print(f"\n{SAMPLES} samples each, plate 120x80x20 mm")
    print(f"  origin on the part:        {near.radius:7.3f} mm 95%   "
          f"sigma {[round(v, 3) for v in near.sigma]}")
    print(f"  origin {LEVER_MM:.0f} mm off the part: {far.radius:7.3f} mm 95%   "
          f"sigma {[round(v, 3) for v in far.sigma]}")
    print(f"  ratio {far.radius / max(near.radius, 1e-9):.1f}x for the same photographs")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
