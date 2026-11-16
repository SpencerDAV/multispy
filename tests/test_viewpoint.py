"""Apportioning the information along the origin's worst direction.

This module exists to answer a question the solve cannot ("where should the
next photograph stand?") without answering one it already has ("how much should
each camera count?"). The tests are mostly about keeping those apart, because
the failure mode is not a wrong number -- it is a right number read as a licence
to start weighting cameras by where they point.

What is being defended here:

* **The aiming term is real.** With distance, focal and mark count held equal, a
  camera staring down a direction contributes less information about it than one
  looking across it. If this stops being true the panel is apportioning noise.

* **And the refusal to act on it is also real.** The same camera that is worst
  for the weak direction is *best* for a direction across it, by the same
  geometry and in the same report. That inversion is the whole argument against
  a per-camera scalar weight, so it is asserted rather than left in a comment:
  a scalar cannot tell the two axes apart, and down-weighting a camera for being
  blind to X would suppress the Y and Z it sees better than anything else in the
  set.

* **Distance is not quietly folded into aim.** Information falls off as the
  square of the distance, which is often the larger effect, so a close camera
  aimed badly can beat a far one aimed well. The panel warns about this; the
  warning is worthless if ``mean_distance`` is not actually reported, so a case
  where distance wins outright is constructed and checked.

* **The weak direction is read, not re-derived.** When an origin measurement
  exists the apportionment must use *its* worst direction and say
  ``source == "origin"``. The pooled information here describes where the marks
  are weakest, which is a different question -- the origin also carries the lever
  from the datum triangle out to (0,0,0) -- and silently substituting one for the
  other would make the panel confidently answer something nobody asked.

* **A stale measurement is named as stale.** Edit a mark and the stored
  direction no longer describes the project; the panel has to say so rather than
  point somewhere that was right an hour ago.

* **The verdict tracks anisotropy, not quality.** A nearly spherical error gets
  ``evidence-limited`` even when the origin is terrible, because no viewpoint
  will fix it. A panel that said "viewpoint-limited" whenever the number was bad
  would send someone out to take a photograph that cannot help.

* **The suggestion is perpendicular to the weak direction and genuinely new.**
  Perpendicularity is what makes it worth taking; being furthest from the lines
  of sight already held is what makes it worth travelling for. Both are checked
  against a brute-force sweep rather than trusted.

* **It does not touch the project.** Read-only is the entire safety story for
  something that runs on a solved project the user is in the middle of editing.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_viewpoint.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.project import CameraGroup, ImageView, Project, TiePoint
from multispy.solve import solve
from multispy.uncertainty import fingerprint, origin_uncertainty
from multispy.viewpoint import (
    ANISOTROPY_FLOOR,
    MIN_CAMERAS,
    _information,
    viewpoint_report,
)
from test_part import build_part_project, look_at

FOCAL = 2100.0
WIDTH, HEIGHT = 2400, 1800
SIGMA_PX = 1.0

TARGET = np.array([0.0, 0.0, 0.0])
"""Everything in the hand-built scenes looks at the origin, so a camera's
optical axis and its bearing to the points are the same vector and the two
cannot be confused in a failing assertion."""


# --------------------------------------------------------------- hand-built


def bench(
    places: list[tuple[np.ndarray, float]],
    *,
    points: np.ndarray | None = None,
    weak: np.ndarray | None = None,
    sigma: list[float] | None = None,
) -> Project:
    """A project with poses set by hand and no solve.

    Hand-built rather than solved because every claim below is about geometry
    alone: a solve would introduce its own residuals and make an assertion about
    ``cos_weak`` an assertion about the solver's luck. ``places`` is a list of
    ``(direction, distance)`` -- the camera sits at ``direction * distance`` and
    looks back at the origin, so ``direction`` is the reverse of its optical
    axis.
    """
    p = Project(name="bench", nominal_units="mm", nominal_scale=0.001)
    p.groups["cam0"] = CameraGroup(id="cam0", f=FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)
    p.settings.sigma_px = SIGMA_PX

    xs = points if points is not None else np.array(
        [[0.02, 0.0, 0.0], [-0.02, 0.01, 0.0], [0.0, -0.015, 0.012], [0.01, 0.01, -0.01]]
    )
    for i, xyz in enumerate(xs):
        p.points.append(TiePoint(id=f"q{i}", xyz=np.asarray(xyz, dtype=float)))

    for k, (direction, dist) in enumerate(places):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        R, t = look_at(d * dist, TARGET)
        im = ImageView(id=f"v{k}", path=f"v{k}.jpg", width=WIDTH, height=HEIGHT, group="cam0")
        im.R, im.t = R, t
        im.observations = {pt.id: [WIDTH / 2.0, HEIGHT / 2.0] for pt in p.points}
        p.images.append(im)

    if weak is not None:
        w = np.asarray(weak, dtype=float)
        w = w / np.linalg.norm(w)
        a, b = _orthonormal_pair(w)
        sig = list(sigma) if sigma else [9.0, 3.0, 2.0]
        p.origin_uncertainty = {
            "n": 32,
            "radius": 10.0,
            "sigma": sig,
            "axes": [list(map(float, w)), list(map(float, a)), list(map(float, b))],
            "anisotropy": sig[0] / sig[-1],
            "units": "mm",
            "fingerprint": fingerprint(p),
        }
    return p


def _orthonormal_pair(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(w @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    a = np.cross(w, seed)
    a /= np.linalg.norm(a)
    return a, np.cross(w, a)


def share_along(project: Project, direction: np.ndarray) -> dict[str, float]:
    """Each camera's share of the information along an arbitrary direction.

    The public report only ever apportions along the weak direction, which is the
    right public surface; proving that the ranking *inverts* for a perpendicular
    direction needs the same arithmetic aimed somewhere else, so it is done here
    rather than widening the module for a test.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    pts = {pt.id: np.asarray(pt.xyz, dtype=float) for pt in project.points if pt.xyz is not None}
    posed = [im for im in project.active_images() if im.R is not None and im.t is not None]
    per, pooled, _ = _information(project, posed, pts)
    total = float(d @ pooled @ d)
    return {iid: float(d @ H @ d) / total for iid, H in per.items()}


# ------------------------------------------------------------------- claims


def test_a_camera_aimed_down_a_direction_says_less_about_it() -> None:
    """Everything equal but aim: along-axis loses to across-axis."""
    weak = np.array([1.0, 0.0, 0.0])
    p = bench(
        [
            (np.array([1.0, 0.0, 0.0]), 1.0),   # staring down the weak direction
            (np.array([0.0, 1.0, 0.0]), 1.0),   # looking across it
        ],
        weak=weak,
    )
    r = viewpoint_report(p)
    assert r.ok, r.reason
    along = next(c for c in r.cameras if c.image == "v0")
    across = next(c for c in r.cameras if c.image == "v1")
    assert along.cos_weak > 0.9, f"v0 should be aimed down the weak direction, got {along.cos_weak:.3f}"
    assert across.cos_weak < 0.1, f"v1 should look across it, got {across.cos_weak:.3f}"
    assert across.share > along.share * 5, (
        f"at equal distance the across-axis camera should dominate; got "
        f"{across.share:.3f} against {along.share:.3f}"
    )


def test_the_worst_camera_for_one_axis_is_the_best_for_another() -> None:
    """The reason no weight follows from any of this.

    A scalar weight per camera cannot express "discount this for X but not for
    Y and Z". This is the measurement that says such a weight would destroy
    real information.
    """
    weak = np.array([1.0, 0.0, 0.0])
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 1.0), (np.array([0.0, 1.0, 0.0]), 1.0)],
        weak=weak,
    )
    on_weak = share_along(p, weak)
    across = share_along(p, np.array([0.0, 1.0, 0.0]))
    assert on_weak["v0"] < on_weak["v1"], (
        f"v0 should be the weaker witness for X: {on_weak['v0']:.3f} vs {on_weak['v1']:.3f}"
    )
    assert across["v0"] > across["v1"], (
        f"but the better witness for Y, which is the whole point: "
        f"{across['v0']:.3f} vs {across['v1']:.3f}"
    )
    assert across["v0"] > 5 * on_weak["v0"], (
        f"v0's Y contribution ({across['v0']:.3f}) should dwarf its X contribution "
        f"({on_weak['v0']:.3f}) -- a scalar weight would cut both by the same factor"
    )


def test_standing_closer_can_beat_aiming_better() -> None:
    """Why the panel reports distance next to aim instead of aim alone.

    The badly-aimed camera is 14 degrees off the weak direction rather than
    exactly on it, because exactly on it contributes exactly zero -- ``1 - (u.w)^2``
    vanishes -- and no distance beats zero. 14 degrees is also the honest case:
    PRM1's most X-ward camera is 0.97, not 1.00, and a real photograph is never
    perfectly degenerate.
    """
    weak = np.array([1.0, 0.0, 0.0])
    p = bench(
        [
            (np.array([1.0, 0.25, 0.0]), 0.4),   # aimed badly, but close
            (np.array([0.0, 1.0, 0.0]), 4.0),    # aimed perfectly, ten times further
        ],
        weak=weak,
    )
    r = viewpoint_report(p)
    near = next(c for c in r.cameras if c.image == "v0")
    far = next(c for c in r.cameras if c.image == "v1")
    assert near.cos_weak > 0.9, f"v0 should still be aimed badly, got {near.cos_weak:.3f}"
    assert near.share > far.share, (
        f"the close badly-aimed camera should win here ({near.share:.3f} vs "
        f"{far.share:.3f}); if it does not, the 1/z^2 term has gone missing"
    )
    assert near.mean_distance < far.mean_distance, (
        "and the distances must be reported, or the reader cannot see why"
    )


def test_an_exactly_on_axis_camera_contributes_exactly_nothing() -> None:
    """The limit the test above steps back from, asserted on its own.

    Worth pinning because it is the sharpest form of the argument: a camera
    perfectly aligned with a direction is not merely a poor witness to it, it is
    not a witness at all, and the solve already knows that without being told.
    """
    weak = np.array([1.0, 0.0, 0.0])
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 0.2), (np.array([0.0, 1.0, 0.0]), 3.0)],
        # Both points on the X axis, so the rays from v0 are exactly its optical
        # axis. Off-axis points would spread the rays and leave a small genuine
        # contribution, which is true but is not the limit being pinned here.
        points=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        weak=weak,
    )
    r = viewpoint_report(p)
    on_axis = next(c for c in r.cameras if c.image == "v0")
    assert on_axis.cos_weak > 1.0 - 1e-9
    assert on_axis.share < 1e-9, (
        f"an exactly on-axis camera should supply nothing along that axis however close "
        f"it stands; got {on_axis.share:.3e} from 0.2 m"
    )


def test_the_shares_are_an_apportionment() -> None:
    p = bench(
        [
            (np.array([1.0, 0.2, 0.1]), 1.2),
            (np.array([0.1, 1.0, 0.3]), 0.9),
            (np.array([0.3, -0.4, 1.0]), 1.5),
        ],
        weak=np.array([1.0, 0.0, 0.0]),
    )
    r = viewpoint_report(p)
    total = sum(c.share for c in r.cameras)
    assert abs(total - 1.0) < 1e-9, f"shares should sum to 1, got {total:.6f}"


def test_the_weak_direction_comes_from_the_origin_measurement() -> None:
    """Not re-derived from the cameras, which answers a different question."""
    odd = np.array([0.0, 0.0, 1.0])
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 1.0), (np.array([0.0, 1.0, 0.0]), 1.0)],
        weak=odd,
    )
    r = viewpoint_report(p)
    assert r.source == "origin", f"source should name the measurement, got {r.source!r}"
    assert abs(abs(float(np.asarray(r.weak) @ odd)) - 1.0) < 1e-6, (
        f"the stored direction {odd} should have been used, got {r.weak}"
    )
    assert abs(r.weak_sigma - 9.0) < 1e-9, "and its sigma carried with it"


def test_without_a_measurement_it_says_the_direction_is_geometric() -> None:
    p = bench(
        [
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([0.0, 1.0, 0.0]), 1.0),
            # Not straight down: look_at's up vector is +Z, so a camera directly
            # overhead has no definable right-hand axis and its pose comes out
            # NaN. See test_a_non_finite_pose_is_named_rather_than_crashed_on.
            (np.array([0.15, 0.1, 1.0]), 1.0),
        ]
    )
    r = viewpoint_report(p)
    assert r.ok, r.reason
    assert r.source == "geometry", f"got {r.source!r}"
    assert any("not the one the origin is weakest in" in n for n in r.notes), (
        "the fallback has to say what it is answering instead; notes were "
        f"{r.notes}"
    )
    assert r.weak_sigma == 0.0, "a geometric direction has no measured sigma to quote"


def test_an_edited_mark_makes_the_stored_direction_stale() -> None:
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 1.0), (np.array([0.0, 1.0, 0.0]), 1.0)],
        weak=np.array([1.0, 0.0, 0.0]),
    )
    assert not viewpoint_report(p).stale, "the fixture should start fresh"
    p.images[0].observations["q0"] = [10.0, 20.0]
    r = viewpoint_report(p)
    assert r.stale, "moving a mark must invalidate the stored weak direction"
    assert any("may have moved" in n for n in r.notes), f"and say so: {r.notes}"


def test_a_near_spherical_error_is_evidence_limited() -> None:
    """A near-sphere has nothing for a new vantage to fix.

    The verdict has to come off the shape of the error and not off its size:
    the stored sigmas below are large *and* nearly equal, which is the case
    where the honest answer is "take more marks", not "take another photograph".
    """
    p = bench(
        [
            (np.array([1.0, 0.1, 0.2]), 1.0),
            (np.array([0.1, 1.0, 0.3]), 1.1),
            (np.array([0.2, -0.3, 1.0]), 0.9),
        ],
        weak=np.array([1.0, 0.0, 0.0]),
        sigma=[40.0, 36.0, 32.0],
    )
    r = viewpoint_report(p)
    assert r.ok, r.reason
    assert r.anisotropy < ANISOTROPY_FLOOR, f"anisotropy came out {r.anisotropy:.2f}"
    assert r.verdict == "evidence-limited", f"got {r.verdict!r}"
    assert any("nearly spherical" in n for n in r.notes), f"notes were {r.notes}"


def test_the_plates_own_camera_arc_turns_out_to_be_lopsided() -> None:
    """A real solve, and the answer is not the flattering one.

    The plate fixture's eight photographs sweep a wide arc of azimuth but are
    all *above* the plate looking down, within 125-220 mm of height. That is a
    ring, not a sphere, and the origin pays for it: the measured anisotropy comes
    out well past the floor. Asserted in the direction it actually falls, because
    a test written to the assumption that a spread-looking set is well
    conditioned would have been a test that quietly encoded the opposite of the
    truth -- and the suggested bearing is then checked to be across the weak
    direction, which is the part a person would act on.
    """
    project, _ = build_part_project()
    solve(project)
    measured = origin_uncertainty(project, n=16, seed=3)
    project.origin_uncertainty = measured.to_dict()
    r = viewpoint_report(project)
    assert r.ok, r.reason
    assert r.source == "origin"
    assert r.anisotropy > ANISOTROPY_FLOOR, (
        f"the plate's arc is one-sided in height, so its origin error should be clearly "
        f"lopsided; anisotropy came out {r.anisotropy:.2f}"
    )
    assert r.verdict == "viewpoint-limited", f"got {r.verdict!r}"
    assert r.suggestion is not None, "and there should be somewhere to go"
    b = np.asarray(r.suggestion.bearing, dtype=float)
    assert abs(float(b @ np.asarray(r.weak))) < 1e-6, (
        "the way out of a lopsided set is a line of sight across its weak direction"
    )
    assert r.suggestion.nearest_existing_deg > 10.0, (
        f"and it should be a genuinely new vantage, not one already held; got "
        f"{r.suggestion.nearest_existing_deg:.1f} deg"
    )


def test_a_lopsided_set_is_viewpoint_limited() -> None:
    """And the verdict does flip when there is a real lever."""
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 1.0), (np.array([1.0, 0.05, 0.0]), 1.0)],
        weak=np.array([1.0, 0.0, 0.0]),
    )
    r = viewpoint_report(p)
    assert r.verdict == "viewpoint-limited", (
        f"anisotropy {r.anisotropy:.2f} is above the floor of {ANISOTROPY_FLOOR}, so the "
        f"verdict should be viewpoint-limited; got {r.verdict!r}"
    )


def test_the_suggested_bearing_is_across_the_weak_direction() -> None:
    weak = np.array([0.3, -0.5, 0.81])
    p = bench(
        [
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([0.0, 1.0, 0.0]), 1.1),
            (np.array([0.6, 0.6, 0.1]), 1.2),
        ],
        weak=weak,
    )
    r = viewpoint_report(p)
    assert r.suggestion is not None, "three cameras and a weak direction should yield one"
    b = np.asarray(r.suggestion.bearing, dtype=float)
    w = weak / np.linalg.norm(weak)
    assert abs(float(b @ w)) < 1e-6, (
        f"the bearing must be perpendicular to the weak direction; dot was {float(b @ w):.2e}"
    )
    assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-9, "and a unit vector"


def test_the_suggested_bearing_is_the_furthest_from_what_is_already_held() -> None:
    """Checked against a brute-force sweep, not trusted."""
    weak = np.array([0.0, 0.0, 1.0])
    places = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.9, 0.3, 0.0]), 1.0),
        (np.array([0.8, -0.2, 0.1]), 1.0),
    ]
    p = bench(places, weak=weak)
    r = viewpoint_report(p)
    s = r.suggestion
    assert s is not None
    looks = []
    for direction, dist in places:
        d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        R, _t = look_at(d * dist, TARGET)
        looks.append(R[2, :])

    def nearest(v: np.ndarray) -> float:
        return min(math.degrees(math.acos(min(1.0, abs(float(v @ L))))) for L in looks)

    w = weak / np.linalg.norm(weak)
    a, b2 = _orthonormal_pair(w)
    best = max(
        nearest(math.cos(th) * a + math.sin(th) * b2)
        for th in np.linspace(0.0, 2.0 * math.pi, 4000)
    )
    assert s.nearest_existing_deg > best - 1.0, (
        f"the chosen bearing sits {s.nearest_existing_deg:.1f} deg from the nearest existing "
        f"line of sight, but a sweep of the same circle found {best:.1f} deg available"
    )
    assert abs(nearest(np.asarray(s.bearing)) - s.nearest_existing_deg) < 1e-6, (
        "and the reported separation must be the one its own bearing actually has"
    )


def test_the_two_offered_positions_are_opposite_and_at_the_stated_standoff() -> None:
    p = bench(
        [
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([0.9, 0.3, 0.0]), 1.4),
            (np.array([0.7, -0.4, 0.2]), 2.0),
        ],
        weak=np.array([0.0, 0.0, 1.0]),
    )
    r = viewpoint_report(p)
    s = r.suggestion
    assert s is not None
    near = np.asarray(s.position, dtype=float)
    far = np.asarray(s.position_opposite, dtype=float)
    mid = (near + far) / 2.0
    pts = np.array([pt.xyz for pt in p.points]) / p.nominal_scale
    centroid = pts.mean(axis=0)
    assert np.allclose(mid, centroid, atol=1e-6), (
        f"the two positions should straddle the marked centroid {centroid}, midpoint was {mid}"
    )
    assert abs(float(np.linalg.norm(near - centroid)) - s.standoff) < 1e-6, (
        "and sit at the stated standoff"
    )
    # The closest standoff in the set, measured from the marks' centroid rather
    # than from the point the cameras happen to be aimed at -- those differ here
    # by a couple of millimetres, and the centroid is the right one: it is where
    # the evidence is, not where the lenses were pointed.
    cams = np.array([-(np.asarray(im.R).T @ np.asarray(im.t)) for im in p.images])
    closest = float(np.min(np.linalg.norm(cams - centroid * p.nominal_scale, axis=1)))
    assert abs(s.standoff - closest / p.nominal_scale) < 1e-6, (
        f"the standoff should be the closest camera's distance from the centroid "
        f"({closest / p.nominal_scale:.1f} mm), got {s.standoff:.1f}"
    )


def test_a_gain_is_also_reported_as_a_distance() -> None:
    """So that a large-sounding percentage cannot be read as a large improvement."""
    p = bench(
        [
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([0.9, 0.3, 0.0]), 1.0),
            (np.array([0.8, -0.3, 0.0]), 1.0),
        ],
        weak=np.array([1.0, 0.0, 0.0]),
    )
    r = viewpoint_report(p)
    s = r.suggestion
    assert s is not None
    assert s.gain > 0.0, "a new vantage against a one-sided set should add something"
    expected = 1.0 / math.sqrt(1.0 + s.gain)
    assert abs(s.sigma_factor - expected) < 1e-12, (
        f"sigma goes as the inverse square root of information: expected "
        f"{expected:.6f}, got {s.sigma_factor:.6f}"
    )
    assert s.sigma_factor < 1.0, "more information cannot make a sigma larger"
    # The claim the field's own docstring makes, checked rather than asserted in
    # prose: half as much information again is about 18% off the sigma. The point
    # of carrying this number at all is that a gain which sounds enormous is a
    # modest move in millimetres, so the translation must be the real one.
    assert abs(1.0 / math.sqrt(1.5) - 0.8165) < 1e-3


def test_unmarked_in_frame_counts_only_what_is_left_to_click() -> None:
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 1.0), (np.array([0.0, 1.0, 0.0]), 1.0)],
        weak=np.array([1.0, 0.0, 0.0]),
    )
    r = viewpoint_report(p)
    for c in r.cameras:
        assert c.unmarked_in_frame == 0, (
            f"{c.image} marks every point, so nothing should be left: got "
            f"{c.unmarked_in_frame}"
        )
        assert c.gain_if_marked == 0.0, "and no gain from marking them again"

    del p.images[0].observations["q0"]
    del p.images[0].observations["q1"]
    r2 = viewpoint_report(p)
    v0 = next(c for c in r2.cameras if c.image == "v0")
    assert v0.unmarked_in_frame == 2, (
        f"two marks were removed and both points are still in frame; got "
        f"{v0.unmarked_in_frame}"
    )
    assert v0.gain_if_marked > 0.0, "and marking them again should be worth something"
    assert r2.most_left_to_click == "v0", f"got {r2.most_left_to_click!r}"


def test_a_point_behind_the_camera_is_not_left_to_click() -> None:
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 0.3), (np.array([0.0, 1.0, 0.0]), 1.0)],
        points=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        weak=np.array([0.0, 0.0, 1.0]),
    )
    # v0 sits at +X looking back at the origin, so the point far out at +5 m is
    # behind it. A projection that ignored the sign of z would place it in frame.
    p.images[0].observations.pop("q2", None)
    r = viewpoint_report(p)
    v0 = next(c for c in r.cameras if c.image == "v0")
    assert v0.unmarked_in_frame == 0, (
        f"the only unmarked point is behind v0, so nothing is left to click there; got "
        f"{v0.unmarked_in_frame}"
    )


def test_it_does_not_touch_the_project() -> None:
    project, _ = build_part_project()
    solve(project)
    before = project.to_dict()
    viewpoint_report(project)
    assert project.to_dict() == before, (
        "the report must be read-only: it runs on a project the user is editing"
    )


def test_a_non_finite_pose_is_named_rather_than_crashed_on() -> None:
    """Found by a test that meant to do something else, which is how it got in.

    A camera pointed exactly along the scene's up vector comes out of ``look_at``
    as NaN, and a NaN in one pose poisons the pooled information matrix. Left
    alone, that surfaced three steps later as LAPACK's "eigenvalues did not
    converge" -- true, useless, and naming no camera. The report has to say which
    image is broken.
    """
    p = bench(
        [(np.array([1.0, 0.0, 0.0]), 1.0), (np.array([0.0, 1.0, 0.0]), 1.0)],
    )
    p.images[1].R = np.full((3, 3), np.nan)
    r = viewpoint_report(p)
    assert not r.ok, "a NaN pose cannot be apportioned"
    assert "v1" in r.reason, f"and the broken camera should be named: {r.reason!r}"
    assert "not finite" in r.reason, f"got {r.reason!r}"


def test_one_camera_is_refused_rather_than_guessed_at() -> None:
    p = bench([(np.array([1.0, 0.0, 0.0]), 1.0)], weak=np.array([1.0, 0.0, 0.0]))
    r = viewpoint_report(p)
    assert not r.ok, "one camera cannot be compared with anything"
    assert str(MIN_CAMERAS) in r.reason, f"and the reason should say so: {r.reason!r}"
    assert not r.cameras, "with no table to mislead anyone"


def test_the_nominal_only_shares_are_flagged_while_there_are_too_few() -> None:
    project, _ = build_part_project(nominals_on=2)
    solve(project)
    r = viewpoint_report(project)
    assert r.ok, r.reason
    assert r.n_nominal_points < 3
    assert any("not worth reading yet" in n for n in r.notes), (
        f"two nominals cannot carry a per-camera split; notes were {r.notes}"
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

    weak = np.array([1.0, 0.0, 0.0])
    p = bench(
        [
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([1.0, 0.0, 0.0]), 0.5),
            (np.array([0.0, 1.0, 0.0]), 1.0),
            (np.array([0.0, 1.0, 0.0]), 4.0),
        ],
        weak=weak,
    )
    r = viewpoint_report(p)
    print("\nfour cameras, same marks, against a weak direction of +X")
    print(f"{'camera':<9}{'aim':<14}{'dist m':>8}{'|cos|':>8}{'X share':>10}{'Y share':>10}")
    y = share_along(p, np.array([0.0, 1.0, 0.0]))
    aims = {"v0": "down X", "v1": "down X, close", "v2": "across X", "v3": "across X, far"}
    for c in sorted(r.cameras, key=lambda c: c.image):
        print(f"{c.image:<9}{aims[c.image]:<14}{c.mean_distance / 1000.0:8.2f}"
              f"{c.cos_weak:8.2f}{100 * c.share:9.1f}%{100 * y[c.image]:9.1f}%")
    print("\nThe two columns disagree about which camera matters, which is why no")
    print("per-camera weight follows from either of them.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
