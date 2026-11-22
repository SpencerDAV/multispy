"""The rotation is built from the best-determined pair of axes, not the first two.

``calibrate_image`` used to do ``a0, a1 = axes[0], axes[1]`` -- X and Y whenever
both were marked -- and derive the third axis by cross product. The focal length
in the same function has long been an inverse-variance weighted average over
*every* axis pair, "because a pair whose vanishing points sit near the image is
worth orders of magnitude more than one relying on a near-infinite vanishing
point". Only the rotation went on taking whichever pair came first.

**The failure this defends against leaves no residual**, which is why it needs a
test of its own rather than a tighter threshold somewhere. A pair whose
vanishing points sit far outside the frame fixes its two directions badly, and
the rotation built on them is *confident*: the control lines fit it to the
pixel, the focal absorbs the rest of the error, and the orthogonality figure
stays small because it is measured on those same two directions. Every number a
reader would check is still good. The only hint available was the per-image
``focal_sigma_pct``, and that is reported against the *best* pair -- not against
the pair the rotation actually used.

What is pinned here:

* **The pair is chosen by conditioning**, on a geometry built from first
  principles: a camera turned so that world X runs nearly parallel to the image
  plane, which sends X's vanishing point off toward infinity while Y and Z stay
  well placed. X x Y is then the worst pair available and X is still axis 0.

* **What it is worth, as a distribution rather than a number.** Over 80 draws
  of the pixel noise the median rotation error against known truth falls 1.3x
  to 1.7x and the 90th percentile 1.2x to 1.8x, and both ratios grow as the
  conditioning gap between the pairs widens -- which is the mechanism, so the
  trend is the real claim. On any *one* draw the badly conditioned pair gets
  lucky about a fifth of the time, and that is pinned too.

* **The choice is reported**, as ``Calibration.rotation_axes``. It used to be
  ``axes_used[:2]`` by construction and callers read it off there; a reversed
  line on the derived axis is cosmetic while one on either chosen axis turns the
  camera half a turn, so anything reasoning about that has to be able to ask.

* **A locked focal still rescues a photo that cannot rank its pairs at all.**
  When no pair yields a real focal length there is nothing to rank by, and the
  fallback is the old behaviour -- see ``test_focal.py`` for the lock itself.

* **An unranked photo is not disturbed.** Two axes marked means one pair, and
  the answer has to be bit-identical to what it always was.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_rotationpair.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy.geometry import (  # noqa: E402
    AXIS_NAMES,
    calibrate_image,
    direction_from_vp,
    orthonormalize,
    resolve_direction_sign,
)

WIDTH, HEIGHT, TRUE_FOCAL = 4000, 3000, 2400.0

# How far world X is tipped out of the image plane. 0.1 puts the X x Y pair at
# about 16% focal uncertainty against Y x Z's 3%, which is the same spread the
# real alberto photographs show (img3: 16.4% against 1.2%). Low enough to be
# the ordinary case rather than a contrived one: X x Y still yields a perfectly
# real focal length here, it is just a bad one.
X_TIP = 0.10


# --------------------------------------------------------------------------
# a camera with one axis nearly parallel to the image plane
# --------------------------------------------------------------------------


def look_at(view: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for a camera pointed along ``view``."""
    z = view / np.linalg.norm(view)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def build_view(x_tip: float = X_TIP, seed: int = 3, noise_px: float = 1.5):
    """One camera and its control lines on all three axes.

    The viewing direction is ``(x_tip, 0.6, 0.8)``, and the third row of a
    world-to-camera rotation *is* the camera-space z of each world axis -- the
    depth that divides into a vanishing point's position. So a small first
    entry is precisely the statement "world X is nearly parallel to the image
    plane, and its vanishing point is a long way outside the frame".

    Three segments per axis, each on a different parallel edge, so no axis is
    privileged by line count.
    """
    R = look_at(np.array([x_tip, 0.60, 0.80]), np.array([0.0, 0.0, 1.0]))
    t = -R @ np.array([5.0, -40.0, -30.0])
    rng = np.random.default_rng(seed)

    def project(X: np.ndarray) -> np.ndarray:
        Xc = R @ X + t
        return np.array([
            TRUE_FOCAL * Xc[0] / Xc[2] + WIDTH / 2.0,
            TRUE_FOCAL * Xc[1] / Xc[2] + HEIGHT / 2.0,
        ])

    def lines(axis: int, origins, length: float) -> np.ndarray:
        d = np.zeros(3)
        d[axis] = 1.0
        rows = []
        for origin in origins:
            a = np.asarray(origin, dtype=float)
            p0 = project(a) + rng.normal(0.0, noise_px, 2)
            p1 = project(a + length * d) + rng.normal(0.0, noise_px, 2)
            rows.append([p0[0], p0[1], p1[0], p1[1]])
        return np.array(rows)

    segs = {
        0: lines(0, [(0, 0, 0), (0, 6, 0), (0, 0, 5)], 14.0),
        1: lines(1, [(0, 0, 0), (9, 0, 0), (0, 0, 5)], 14.0),
        2: lines(2, [(0, 0, 0), (9, 0, 0), (0, 6, 0)], 10.0),
    }
    return R, segs


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    cos = (np.trace(R_est @ R_true.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def rotation_from_pair(calib, segs: dict[int, np.ndarray], a0: int, a1: int) -> np.ndarray:
    """Rebuild a rotation from a nominated pair, as the old code did.

    A local copy on purpose rather than a hook into the module: this is the
    behaviour being replaced, so it has to stay written down here even once no
    production path can produce it.
    """
    dirs = {}
    for axis, vp in calib.vps.items():
        d = direction_from_vp(vp.v, calib.intrinsics)
        dirs[axis] = resolve_direction_sign(d, vp.v, segs[axis])
    cols = {a0: dirs[a0], a1: dirs[a1]}
    missing = ({0, 1, 2} - {a0, a1}).pop()
    order = sorted([a0, a1])
    cross = np.cross(cols[order[0]], cols[order[1]])
    cross /= max(np.linalg.norm(cross), 1e-12)
    if order == [0, 2]:
        cross = -cross
    cols[missing] = cross
    return orthonormalize(np.column_stack([cols[0], cols[1], cols[2]]))


def in_frame(segs: dict[int, np.ndarray]) -> bool:
    """Every endpoint on the picture, so no axis is marked off the photograph."""
    return all(
        0.0 <= row[i] <= (WIDTH if i % 2 == 0 else HEIGHT)
        for rows in segs.values()
        for row in rows
        for i in range(4)
    )


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_the_worst_conditioned_pair_is_not_chosen():
    """X is axis 0 and the worst pair available; the rotation declines it."""
    _R, segs = build_view()
    assert in_frame(segs), "fixture drew a control line off the photograph"
    calib = calibrate_image(segs, WIDTH, HEIGHT)

    assert calib.axes_used == (0, 1, 2), calib.axes_used
    sigmas = {
        (p.axis_a, p.axis_b): p.relative_sigma
        for p in calib.observability.pairs
        if p.relative_sigma is not None
    }
    assert sigmas[(0, 1)] > 3.0 * sigmas[(1, 2)], sigmas
    assert calib.rotation_axes == (1, 2), calib.rotation_axes


def test_on_one_photograph_it_more_than_halves_the_rotation_error():
    """Measured against truth, not against a residual -- there is no residual.

    Both rotations come out of the same vanishing points and the same focal
    length. The only difference is which pair of directions was measured and
    which was derived by cross product.

    One photograph and one draw of the pixel noise, which is all this says. The
    distribution is the test below, and it is the one to read: on a single draw
    the badly conditioned pair gets lucky about a fifth of the time.
    """
    R_true, segs = build_view()
    calib = calibrate_image(segs, WIDTH, HEIGHT)

    chosen = rotation_error_deg(calib.R, R_true)
    old = rotation_error_deg(rotation_from_pair(calib, segs, 0, 1), R_true)
    assert chosen < 1.5, chosen
    assert old > 2.0, old
    assert old > 2.0 * chosen, (old, chosen)


def errors_over_seeds(tip: float, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Rotation error from the old pair and from the chosen pair, over n draws."""
    old, chosen = [], []
    for seed in range(n):
        R_true, segs = build_view(x_tip=tip, seed=seed)
        try:
            calib = calibrate_image(segs, WIDTH, HEIGHT)
        except ValueError:
            continue
        if calib.rotation_axes != (1, 2):
            continue
        chosen.append(rotation_error_deg(calib.R, R_true))
        old.append(rotation_error_deg(rotation_from_pair(calib, segs, 0, 1), R_true))
    return np.array(old), np.array(chosen)


def test_the_gain_is_distributional_and_not_true_of_every_draw():
    """The honest shape of the improvement, pinned so nobody tightens it back.

    The first version of this test asserted the chosen pair wins on every
    combination of tip and seed, and it failed -- at tip 0.02, seed 29, the
    badly conditioned pair came out 0.96 deg against the good pair's 1.02. That
    was the premise being wrong, not the threshold. A worse-conditioned pair is
    not reliably wrong on any one draw; it is *higher variance*, and over 80
    draws of the pixel noise it wins about a fifth of the time by luck.

    So what is pinned is the distribution, and the *trend*: over 80 draws the
    median falls by 1.3x to 1.7x and the 90th percentile by 1.2x to 1.8x, and
    both ratios grow as world X is tipped further out of view -- which is the
    whole physical claim. The wider the conditioning gap between the two pairs,
    the more the choice is worth. A fixed threshold at one geometry would not
    say that.

    Deliberately **not** pinned: the worst draw. The tail reads well at the
    shallow end (9.1 deg against 3.8 at tip 0.02) but it is one draw out of
    eighty, and at tip 0.10 the badly conditioned pair's worst, 3.31 deg, is
    *better* than the good pair's 3.79. Both distributions have a tail and an
    extremum over 80 samples is not a statistic. Asserting it passed at the
    seeds first tried and would have failed for the next reader.

    Pinning the "not every draw" half matters as much as the rest. Anyone who
    measures this on one photograph and concludes the change does nothing will
    be right about that photograph and wrong about the method, and this is
    where they should find that out.
    """
    sometimes_worse = False
    ratios = {}
    for tip in (0.02, 0.10, 0.30):
        old, chosen = errors_over_seeds(tip)
        assert len(old) > 50, (tip, len(old))

        median_ratio = float(np.median(old) / np.median(chosen))
        p90_ratio = float(np.percentile(old, 90) / np.percentile(chosen, 90))
        ratios[tip] = median_ratio
        assert median_ratio > 1.2, (tip, median_ratio)
        assert p90_ratio > 1.1, (tip, p90_ratio)

        wins = float(np.mean(old > chosen))
        assert wins > 0.6, (tip, wins)
        if wins < 1.0:
            sometimes_worse = True

    assert sometimes_worse, (
        "the chosen pair won every single draw at every tip, which this test "
        "says should not happen -- either the fixture lost its noise or the "
        "comparison is no longer comparing two different pairs"
    )
    # the physical claim: the worse the conditioning gap, the more it is worth
    assert ratios[0.02] > 1.5, ratios
    assert ratios[0.02] > ratios[0.30], ratios


def test_the_choice_is_reported_and_not_inferable_from_axes_used():
    """``rotation_axes`` is the field to read, and it really does differ.

    Pinned because the old answer was available by construction -- the rotation
    was ``sorted(axes_used)[:2]`` -- and a caller that kept reading it there
    would now be wrong without failing.
    """
    _R, segs = build_view()
    calib = calibrate_image(segs, WIDTH, HEIGHT)
    assert calib.rotation_axes != tuple(sorted(calib.axes_used)[:2])
    assert set(calib.rotation_axes) <= set(calib.axes_used)
    assert len(set(calib.rotation_axes)) == 2


def test_two_marked_axes_are_untouched():
    """One pair available means one pair chosen, bit-identical to before.

    The fallback matters as much as the choice: most projects in the wild carry
    two axes per photograph, and a change that moved those numbers would be a
    change to every existing solve rather than to the badly conditioned ones.
    """
    _R, segs = build_view()
    for pair in ((0, 1), (0, 2), (1, 2)):
        two = {a: segs[a] for a in pair}
        calib = calibrate_image(two, WIDTH, HEIGHT, fixed_focal=TRUE_FOCAL)
        assert calib.rotation_axes == pair, (pair, calib.rotation_axes)
        # bit-identical to the old construction on the same two axes
        expected = rotation_from_pair(calib, two, *pair)
        assert np.array_equal(calib.R, expected), (pair, np.abs(calib.R - expected).max())


def test_a_photo_that_cannot_rank_its_pairs_falls_back_to_the_old_behaviour():
    """No real focal from any pair means nothing to rank by.

    This is the geometry a locked group focal exists to rescue. Ranking is
    unavailable, so the rotation takes the two lowest-numbered axes exactly as
    it always did -- a deliberate fallback, not an accident, because any other
    choice would be equally arbitrary and harder to explain.
    """
    R_true, segs = build_view(x_tip=0.02)
    # X x Y yields no real focal at this tip, so X and Y alone cannot be ranked
    try:
        calibrate_image({0: segs[0], 1: segs[1]}, WIDTH, HEIGHT)
    except ValueError as exc:
        assert "no vanishing point pair" in str(exc), exc
    else:
        raise AssertionError("expected X and Y alone to yield no real focal at this tip")

    calib = calibrate_image({0: segs[0], 1: segs[1]}, WIDTH, HEIGHT, fixed_focal=TRUE_FOCAL)
    assert calib.observability.best is None, calib.observability.best
    assert calib.rotation_axes == (0, 1), calib.rotation_axes
    assert np.all(np.isfinite(calib.R))
    assert rotation_error_deg(calib.R, R_true) < 180.0


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_the_worst_conditioned_pair_is_not_chosen,
        test_on_one_photograph_it_more_than_halves_the_rotation_error,
        test_the_gain_is_distributional_and_not_true_of_every_draw,
        test_the_choice_is_reported_and_not_inferable_from_axes_used,
        test_two_marked_axes_are_untouched,
        test_a_photo_that_cannot_rank_its_pairs_falls_back_to_the_old_behaviour,
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

    print("\n--- what the pair is worth, as world X is tipped out of the image plane ---")
    print("  X tip  XxY sigma  YxZ sigma  chose |   from X x Y   |   from chosen  |  XxY")
    print("                                     | median    p90  | median    p90  | worse")
    for tip in (0.02, 0.06, 0.10, 0.16, 0.22, 0.30):
        _R, segs = build_view(x_tip=tip)
        calib = calibrate_image(segs, WIDTH, HEIGHT)
        sig = {(q.axis_a, q.axis_b): q.relative_sigma for q in calib.observability.pairs}
        xy, yz = sig.get((0, 1)), sig.get((1, 2))
        chose = "x".join(AXIS_NAMES[a] for a in calib.rotation_axes)
        old, new = errors_over_seeds(tip)
        print(
            f"  {tip:5.2f}  "
            f"{'     none' if xy is None else f'{xy * 100:8.1f}%'}  "
            f"{'     none' if yz is None else f'{yz * 100:8.1f}%'}  "
            f"{chose:>5s} | "
            f"{np.median(old):6.2f} {np.percentile(old, 90):6.2f} | "
            f"{np.median(new):6.2f} {np.percentile(new, 90):6.2f} | "
            f"{100 * float(np.mean(old > new)):4.0f}%"
        )
    print("  (rotation error in degrees against truth, over 80 draws of the pixel noise;")
    print("   the last column is how often the badly conditioned pair was the worse of")
    print("   the two -- never all of them, which is the point of the third test)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
