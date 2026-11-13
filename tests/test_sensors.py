"""Can the calibration file be caught lying, and caught lying *by name*?

The fixture is a synthetic measurement cell: eight sensors on a ring around a
working volume, every one aimed at it, in millimetres at the scale of the real
Perceptron cell. Synthetic on purpose -- the ground truth is known exactly, so a
fault can be injected deliberately and the verdict checked against what was
actually done rather than against what the report happens to say.

Each fault the shop floor reports gets its own test, and so does each of the
three things this module *cannot* see. That second half matters more:

* **A flip about the aim axis is invisible**, because a turn about the sight line
  leaves the sight line alone. The test asserts the report says so on every
  sensor, including the ones that passed. A verdict of "consistent" that quietly
  meant "I did not look" would be the worst output this feature could produce.
* **A translation along the sight line is invisible**, for the same reason: the
  ray only moves if the shift has a component across it. So the correction a
  verdict carries is the *smallest* one that works, and the test pins that
  claim rather than letting it be read as the whole error.
* **A fault shared by the majority reads as the truth**, and convicts the honest
  minority. The test injects exactly that and asserts the report warns about it
  instead of presenting the inverted answer with confidence.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_sensors.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.sensors import (
    CONSISTENT,
    EXPLAINED,
    MIN_CONSENSUS_SENSORS,
    SUSPECT,
    UNTESTABLE,
    SensorClaim,
    aim_consensus,
    diagnose,
    euler_matrix,
    fault_catalogue,
    quaternion_matrix,
    quaternion_scalar_position,
)

VOLUME = np.array([1500.0, 0.0, 1000.0])
"""The working volume, placed where the real cell's part sits relative to the
part-space datum rather than at a tidy origin -- a bug that confuses the volume
with the origin would pass at (0,0,0) and fail here."""

STANDOFF = 900.0
TOL = 10.0
WARN = 50.0


def _normalize(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float) / float(np.linalg.norm(v))


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """A sensor-to-world rotation whose local +Z points from ``position`` at ``target``.

    Local +Z as the aim axis is a choice made here and nowhere else; the module
    infers it rather than sharing the assumption, which is what
    ``test_the_aim_axis_is_inferred_not_assumed`` checks.
    """
    z = _normalize(np.asarray(target, dtype=float) - np.asarray(position, dtype=float))
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(z @ up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x = _normalize(np.cross(up, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def build_cell(n: int = 8) -> list[SensorClaim]:
    """``n`` sensors on a ring about the volume, each aimed at it, all honest.

    Spread around the ring rather than banked along one wall, because a bank of
    near-parallel sensors pins the volume across the beam and not along it, and a
    fixture with that weakness in it would be testing the conditioning guard
    instead of the verdicts.
    """
    claims = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        height = 250.0 * math.sin(3.0 * theta)
        position = VOLUME + np.array(
            [STANDOFF * math.cos(theta), STANDOFF * math.sin(theta), height]
        )
        claims.append(
            SensorClaim(
                name=f"S{i:02d}",
                t=position,
                R=look_at(position, VOLUME),
                raw={"id": i, "note": "synthetic"},
            )
        )
    return claims


def verdict_for(report, name: str):
    return next(v for v in report.verdicts if v.name == name)


def aim_of(claim: SensorClaim) -> np.ndarray:
    return claim.aim(np.array([0.0, 0.0, 1.0]))


def most_across(claims: list[SensorClaim], direction: np.ndarray) -> SensorClaim:
    """The sensor whose sight line is most nearly perpendicular to ``direction``.

    Which sensor a translation fault is injected into is not a detail. A shift
    along a sensor's own sight line does not move its ray at all, so injecting an
    X error into a sensor that looks along X would be testing the blind spot, not
    the detector. This picks the sensor where the fault is most observable, and
    ``test_a_shift_along_the_sight_line_is_invisible`` tests the other end
    deliberately.
    """
    d = _normalize(direction)
    return min(claims, key=lambda c: abs(float(aim_of(c) @ d)))


# ------------------------------------------------------- the clean file first


def test_a_clean_file_has_every_sensor_agree():
    report = diagnose(build_cell(), tol=TOL, warn=WARN)
    assert report.ok, report.reason
    assert report.n_suspect == 0 and report.n_explained == 0, (
        f"an honest file produced {report.n_suspect} suspect and {report.n_explained} "
        f"explained verdicts: " + "; ".join(
            v.reason for v in report.verdicts if v.verdict != CONSISTENT
        )
    )
    assert report.n_agreed == 8
    assert report.grade(TOL, WARN) == "good"
    assert np.linalg.norm(np.array(report.volume) - VOLUME) < 1.0, (
        f"the working volume came back at {report.volume}, not {list(VOLUME)}"
    )


def test_the_aim_axis_is_inferred_not_assumed():
    """The fixture aims along local +Z and never says so."""
    report = diagnose(build_cell(), tol=TOL, warn=WARN)
    assert report.axis == "+Z", f"inferred the aim axis as {report.axis}"


# ------------------------------------------------- the faults, one at a time


def test_a_sensor_translated_to_the_wrong_nominal_is_caught_and_called_single_axis():
    """Fault 1: calibrated against the wrong nominal, a metre out in X."""
    claims = build_cell()
    target = most_across(claims, np.array([1.0, 0.0, 0.0]))
    target.t = target.t + np.array([1100.0, 0.0, 0.0])

    report = diagnose(claims, tol=TOL, warn=WARN)
    v = verdict_for(report, target.name)
    assert v.verdict == SUSPECT, f"{target.name} came back {v.verdict}: {v.reason}"
    assert v.distance > 500.0, f"a 1100 mm shift showed up as only {v.distance:.1f}"
    assert v.dominant_axis == "X", f"blamed {v.dominant_axis}, not X"
    assert v.dominant_fraction > 0.9, (
        f"only {v.dominant_fraction:.0%} of the correction lay on one axis, so the report "
        f"will not call this a mistyped nominal"
    )
    assert report.grade(TOL, WARN) == "bad"
    assert sum(1 for x in report.verdicts if x.verdict != CONSISTENT) == 1, (
        "one bad sensor dragged others out with it, which is what leave-one-out exists "
        "to prevent"
    )


def test_the_mirrored_feature_fault_is_explained_by_name():
    """Fault 2: calibrated against the mirror-image feature, so Y has the wrong sign."""
    claims = build_cell()
    target = next(c for c in claims if abs(c.t[1]) > STANDOFF * 0.5)
    target.t = np.array([target.t[0], -target.t[1], target.t[2]])

    report = diagnose(claims, tol=TOL, warn=WARN)
    v = verdict_for(report, target.name)
    assert v.verdict == EXPLAINED, f"{target.name} came back {v.verdict}: {v.reason}"
    assert v.fault in ("mirror_y", "reflect_y"), f"named the fault {v.fault!r}"
    assert v.residual_after <= v.threshold, (
        f"the named correction left {v.residual_after:.1f} against a {v.threshold:.1f} "
        f"threshold, so it did not actually explain anything"
    )
    assert v.residual_after < v.distance / 10.0, (
        f"the correction only improved the miss from {v.distance:.1f} to "
        f"{v.residual_after:.1f} -- a fault has to collapse the residual to count as an "
        f"explanation, not merely dent it"
    )


def test_a_nudged_aim_shows_up_amplified_by_the_standoff():
    """Fault 4: the aim was adjusted slightly. The body barely moved; the volume did.

    This is the lever arm again, at the sensor instead of at the part origin, and
    it is why a rotation tolerance stated in degrees is nearly useless on its own:
    two degrees is a tidy number and 31 mm is not.
    """
    claims = build_cell()
    target = claims[2]
    nudge = math.radians(2.0)
    c, s = math.cos(nudge), math.sin(nudge)
    target.R = target.R @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    report = diagnose(claims, tol=TOL, warn=WARN)
    v = verdict_for(report, target.name)
    expected = STANDOFF * math.tan(nudge)
    assert v.verdict == SUSPECT, f"a 2 degree nudge came back {v.verdict}: {v.reason}"
    assert abs(v.distance - expected) < 0.2 * expected, (
        f"a 2 degree nudge at {STANDOFF:.0f} mm standoff should miss by about "
        f"{expected:.0f} mm; it missed by {v.distance:.0f}"
    )
    # The miss has to be reported back as the angle that caused it, because 31 mm
    # of position error and 2 degrees of aim error are the same event described
    # two ways and only one of them is actionable on a mount.
    assert abs(v.implied_angle - 2.0) < 0.2, (
        f"the miss came back as {v.implied_angle:.2f} degrees of implied aim error, not the "
        f"2.0 that was injected"
    )
    assert abs(v.standoff - STANDOFF) < STANDOFF * 0.05
    assert "degrees of aim error" in v.reason and "nudge" in v.reason, (
        f"a nudge-sized miss was not described as one: {v.reason}"
    )


def test_a_nudge_is_not_described_as_a_mistyped_nominal():
    """The discriminator is magnitude, and direction is no help at all.

    This was wrong in an earlier version and the fixture is what caught it. A two
    degree nudge puts 95-99% of its correction on a single axis -- the standoff
    direction dominates it -- so "nearly all on one axis" does not distinguish a
    nudged aim from a transform calibrated against the wrong nominal feature. Both
    cases are asserted here together so the two descriptions cannot be swapped
    again without a failure.
    """
    nudged = build_cell()
    n = math.radians(2.0)
    c, s = math.cos(n), math.sin(n)
    nudged[2].R = nudged[2].R @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    small = verdict_for(diagnose(nudged, tol=TOL, warn=WARN), nudged[2].name)

    typo = build_cell()
    far = most_across(typo, np.array([1.0, 0.0, 0.0]))
    far.t = far.t + np.array([1100.0, 0.0, 0.0])
    large = verdict_for(diagnose(typo, tol=TOL, warn=WARN), far.name)

    assert small.dominant_fraction > 0.9 and large.dominant_fraction > 0.9, (
        f"the fixture no longer exercises the trap: nudge {small.dominant_fraction:.2f}, "
        f"wrong nominal {large.dominant_fraction:.2f} -- both used to be near 1, which is "
        f"the whole point"
    )
    assert "wrong nominal feature" in large.reason, (
        f"a 1100 mm single-axis error was not called a wrong nominal: {large.reason}"
    )
    assert "wrong nominal feature" not in small.reason, (
        f"a 2 degree nudge was described as a mistyped nominal: {small.reason}"
    )


# ------------------------------------------------- and the three blind spots


def test_a_flip_about_the_aim_axis_is_not_reported_as_verified():
    """Fault 3: wrong sweep face. Invisible here, and it must say so.

    The sensor legitimately comes back consistent -- its sight line really is
    unchanged, and claiming otherwise would be a false positive. What must not
    happen is a bare pass: the verdict has to carry the fact that this fault was
    never tested, or "consistent" reads as a clean bill of health it has not
    earned.
    """
    claims = build_cell()
    target = claims[4]
    before = aim_of(target).copy()
    turn = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    target.R = target.R @ turn
    assert np.linalg.norm(aim_of(target) - before) < 1e-9, (
        "the fixture's flip moved the sight line, so this is not testing the blind spot"
    )

    report = diagnose(claims, tol=TOL, warn=WARN)
    v = verdict_for(report, target.name)
    assert v.verdict == CONSISTENT
    assert v.aim_blind, "a passing verdict carried no record of what was never tested"
    assert any("sweep" in text or "aim axis" in text for text in v.aim_blind)
    assert any("photographs" in m for m in report.messages), (
        "the summary does not point at the only referee that can catch this"
    )
    assert "not the same as correct" in v.reason, (
        "a consistent verdict has to say what it is narrower than"
    )


def test_a_shift_along_the_sight_line_is_invisible_and_the_correction_says_so():
    """The second blind spot, and the honest reading of the correction vector.

    Moving a sensor along its own sight line does not move the ray, so the test
    cannot see it. That is geometry, not a bug -- but it means the correction a
    verdict reports is the *smallest* translation that reconciles the sensor, and
    any amount of additional shift along the sight line is consistent with the
    same verdict. This pins both halves.
    """
    claims = build_cell()
    target = claims[1]
    slide = aim_of(target) * 400.0
    target.t = target.t + slide

    report = diagnose(claims, tol=TOL, warn=WARN)
    v = verdict_for(report, target.name)
    assert v.verdict == CONSISTENT, (
        f"a 400 mm slide along the sight line was reported as {v.verdict}, which would mean "
        f"the ray test had somehow become sensitive to it"
    )
    assert v.distance < TOL

    # And the other direction: a correction is a minimum, never the whole error.
    clean = build_cell()
    moved = most_across(clean, np.array([1.0, 0.0, 0.0]))
    moved.t = moved.t + np.array([200.0, 0.0, 0.0])
    report2 = diagnose(clean, tol=TOL, warn=WARN)
    v2 = verdict_for(report2, moved.name)
    assert abs(np.linalg.norm(v2.correction) - v2.distance) < 1e-6, (
        "the reported correction is not the perpendicular one, so its magnitude no longer "
        "equals the miss it is supposed to close"
    )


def test_a_fault_shared_by_the_majority_inverts_the_answer_and_is_flagged():
    """The third blind spot, and the one that could do real damage.

    Five of eight sensors translated by the same amount still form a consistent
    pencil -- translating a whole configuration translates its intersection -- so
    consensus adopts the error as the truth and the three honest sensors become
    the outliers. There is no way for this module to know which group is right.
    What it must not do is present the inverted answer without a word.
    """
    claims = build_cell()
    shift = np.array([0.0, 0.0, 300.0])
    for c in claims[:5]:
        c.t = c.t + shift

    report = diagnose(claims, tol=TOL, warn=WARN)
    disagreeing = [v.name for v in report.verdicts if v.verdict != CONSISTENT]
    assert len(disagreeing) == 3, (
        f"expected the honest minority of three to be convicted, got {disagreeing}"
    )
    assert set(disagreeing) == {c.name for c in claims[5:]}
    assert report.grade(TOL, WARN) in ("warn", "bad")
    assert any("minority" in m or "shares one error" in m for m in report.messages), (
        "the majority's error was adopted as the truth with no warning attached"
    )


# --------------------------------------------------- refusing to answer at all


def test_too_few_sensors_refuses_a_verdict():
    report = diagnose(build_cell(MIN_CONSENSUS_SENSORS - 1), tol=TOL, warn=WARN)
    assert not report.ok
    assert all(v.verdict == UNTESTABLE for v in report.verdicts)
    assert report.grade(TOL, WARN) == "unknown", (
        "too few sensors must not be allowed to grade as good"
    )
    assert str(MIN_CONSENSUS_SENSORS) in report.reason, (
        "the refusal does not say how many sensors would be enough"
    )


def test_near_parallel_sensors_decline_rather_than_guess():
    """A bank of sensors along one wall pins the volume across the beam, not along it."""
    claims = []
    for i in range(6):
        position = np.array([200.0 * i, -1200.0, 1000.0])
        claims.append(
            SensorClaim(name=f"P{i}", t=position, R=look_at(position, position + np.array(
                [0.0, 1200.0, 0.0])))
        )
    got = aim_consensus(claims, tol=TOL)
    assert not got.ok or got.axis_margin < 1.5, (
        "a parallel bank produced a confident volume, which it cannot have earned"
    )


def test_leave_one_out_keeps_a_bad_sensor_from_widening_its_own_threshold():
    """The whole reason the scoring is leave-one-out.

    A sensor judged against a volume it helped place, with a threshold scaled by
    residuals it contributed to, gets to argue for its own innocence. The
    threshold a bad sensor faces must be the one the clean file produces.
    """
    clean = diagnose(build_cell(), tol=TOL, warn=WARN)
    dirty_claims = build_cell()
    bad = most_across(dirty_claims, np.array([1.0, 0.0, 0.0]))
    bad.t = bad.t + np.array([1500.0, 0.0, 0.0])
    dirty = diagnose(dirty_claims, tol=TOL, warn=WARN)

    clean_threshold = verdict_for(clean, bad.name).threshold
    dirty_threshold = verdict_for(dirty, bad.name).threshold
    assert dirty_threshold < clean_threshold * 1.5, (
        f"a sensor 1500 mm out faced a threshold of {dirty_threshold:.1f} where the clean "
        f"file gives {clean_threshold:.1f} -- it is inflating the bar it has to clear"
    )
    for v in dirty.verdicts:
        if v.name != bad.name:
            assert v.verdict == CONSISTENT, (
                f"{v.name} was convicted alongside the genuinely bad sensor: {v.reason}"
            )


# ------------------------------------------------------------- the units check


def test_the_span_notices_a_file_written_in_metres():
    """The 1000x mistake ``part.py`` exists to catch, in its calibration clothing."""
    mm = diagnose(build_cell(), tol=TOL, warn=WARN)
    assert 1000.0 < mm.span < 5000.0, f"a cell-sized span read as {mm.span:.1f}"

    metres = build_cell()
    for c in metres:
        c.t = c.t / 1000.0
    m = diagnose(metres, tol=TOL, warn=WARN)
    assert m.span < 10.0, (
        f"a file in metres reported a span of {m.span:.3f}, which does not distinguish it "
        f"from a cell measured in millimetres"
    )


# ----------------------------------------------- the conventions, which bite


def test_the_euler_order_changes_the_answer():
    """Why the convention is an argument with no default worth having.

    The same three angles under two orders are not nearly the same rotation, and
    neither call fails. Anything that guessed would produce a plausible wrong
    answer, which is the worst kind.
    """
    angles = [20.0, 35.0, 50.0]
    a = euler_matrix(angles, "XYZ")
    b = euler_matrix(angles, "ZYX")
    assert np.linalg.norm(a - b) > 0.3, (
        "two different Euler orders produced near-identical matrices, so this fixture is "
        "not exercising the trap"
    )
    for M in (a, b):
        assert abs(np.linalg.det(M) - 1.0) < 1e-12
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)


def test_intrinsic_and_extrinsic_are_the_same_product_reversed():
    angles = [11.0, -24.0, 63.0]
    assert np.allclose(
        euler_matrix(angles, "XYZ", intrinsic=True),
        euler_matrix(angles[::-1], "ZYX", intrinsic=False),
    )


def test_a_quaternion_round_trips_and_its_scalar_end_is_detectable():
    axis = _normalize(np.array([0.3, -0.5, 0.81]))
    angle = math.radians(12.0)
    w = math.cos(angle / 2.0)
    v = axis * math.sin(angle / 2.0)
    first = [w, *v]
    last = [*v, w]

    R = quaternion_matrix(first, scalar_first=True)
    assert np.allclose(R, quaternion_matrix(last, scalar_first=False))
    assert abs(np.linalg.det(R) - 1.0) < 1e-12
    assert abs(math.degrees(math.acos((np.trace(R) - 1.0) / 2.0)) - 12.0) < 1e-6

    assert quaternion_scalar_position([first, first]) == "scalar_first"
    assert quaternion_scalar_position([last, last]) == "scalar_last"


def test_the_scalar_end_is_undetectable_for_large_rotations():
    """And the detector has to say so instead of inverting.

    Past about 51 degrees the scalar stops being the largest component, so a
    scalar-first file of large rotations has its maximum in the *last* slot --
    indistinguishable, by that test, from a scalar-last file of small ones. They
    are opposite answers. An earlier version here looked only at which slot held
    the maximum and confidently returned ``scalar_last`` for a scalar-first
    input; this pins the refusal.
    """
    axis = _normalize(np.array([0.3, -0.5, 0.81]))
    half = math.radians(85.0)
    big = [math.cos(half), *(axis * math.sin(half))]
    assert int(np.argmax(np.abs(big))) == 3, (
        "the fixture's rotation is not large enough to move the maximum out of the scalar "
        "slot, so it is not testing the inversion"
    )
    assert quaternion_scalar_position([big, big]) is None, (
        "a scalar-first quaternion with a large rotation was reported as scalar-last, which "
        "is the wrong answer stated confidently"
    )
    # One sample satisfies "the same end always wins" trivially, so it is never
    # evidence however small the rotation.
    small = [math.cos(math.radians(3.0)), *(axis * math.sin(math.radians(3.0)))]
    assert quaternion_scalar_position([small]) is None
    assert quaternion_scalar_position([small, small]) == "scalar_first"


def test_every_catalogued_fault_is_a_proper_rotation_and_reversible():
    """A reflection applied carelessly yields determinant -1 and no valid pose."""
    claims = build_cell()
    c = claims[3]
    R, t = c.pose()
    for fault in fault_catalogue():
        Rc, tc = fault.apply(R, t)
        assert abs(np.linalg.det(Rc) - 1.0) < 1e-9, (
            f"{fault.key} produced a matrix of determinant {np.linalg.det(Rc):.3f}, which is "
            f"not a rotation"
        )
        assert np.allclose(Rc @ Rc.T, np.eye(3), atol=1e-9), f"{fault.key} is not orthonormal"
        assert not np.shares_memory(Rc, R) or np.allclose(Rc, R)
        assert np.allclose(R, c.pose()[0]) and np.allclose(t, c.pose()[1]), (
            f"{fault.key} mutated the claim it was given, so the file's own numbers are gone"
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
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n8 sensors on a {STANDOFF:.0f} mm ring about "
          f"{[round(float(v), 1) for v in VOLUME]}, tol {TOL:.0f} / warn {WARN:.0f} mm")
    claims = build_cell()
    worst = most_across(claims, np.array([1.0, 0.0, 0.0]))
    worst.t = worst.t + np.array([1100.0, 0.0, 0.0])
    mirrored = next(c for c in claims if abs(c.t[1]) > STANDOFF * 0.5)
    mirrored.t = np.array([mirrored.t[0], -mirrored.t[1], mirrored.t[2]])
    claims[4].R = claims[4].R @ np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    report = diagnose(claims, tol=TOL, warn=WARN)
    print(f"  aim axis {report.axis}, volume {[round(v, 1) for v in report.volume]}, "
          f"span {report.span:.0f} mm, grade {report.grade(TOL, WARN)}")
    for v in report.verdicts:
        tag = {CONSISTENT: "ok  ", EXPLAINED: "FIX ", SUSPECT: "BAD ", UNTESTABLE: "?   "}[
            v.verdict
        ]
        extra = f" [{v.fault}]" if v.fault else ""
        print(f"    {tag} {v.name}  miss {v.distance:8.1f}  thr {v.threshold:6.1f}{extra}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
