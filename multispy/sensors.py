"""A sensor calibration file is a set of claims, and this module argues with them.

A measurement cell's calibration file states, per sensor, where that sensor sits
and which way it faces. Importing it is trivial. Believing it is the problem:
these files go wrong in specific, repeatable ways -- a sensor calibrated against
the wrong nominal feature and landing a metre out in X, one calibrated against
the mirror-image feature on the other side of the cell and landing at negative
Y, one whose body is turned 180 degrees about its own line of sight so the laser
sweeps the wrong face, and one whose aim has been nudged since, which moves the
body hardly at all and its measurement volume a long way.

Three of those four are *discrete*. That matters more than it sounds. A generic
robust regression treats a bad sensor as a large residual and reports a
magnitude; but "1043 mm of translation error" is a number someone still has to
interpret, while "calibrated against the mirrored feature, correct it by
negating Y" is a diagnosis with a fix attached. So the faults are enumerated and
tested by name, and only what the catalogue cannot name is left to a residual.

**Where the leverage comes from, since it is not obvious.** The file claims to
already be in part space, so there is no frame to fit and nothing for one
sensor's pose to predict about another's -- the sensors are not rigidly related
by anything this tool knows, there being no CAD of the cell. Consensus has to
come from physics instead, and it does: *every sensor on a measurement cell is
aimed at the part*. That makes the sensors' aim rays a pencil through the working
volume, which is a strong shared constraint that costs nothing to evaluate. A
sensor translated a metre in X keeps its aim direction, so its ray now misses the
volume by about a metre. A sensor mirrored to negative Y misses by twice its
offset. A sensor whose aim was nudged misses by the standoff times the angle,
which is the same lever-arm amplification that makes the part-space origin
uncertain in :mod:`multispy.uncertainty`. Three faults, one test.

**And the honest limits, which are the reason feature two exists.** The aim ray
is a line, so anything that leaves that line alone is invisible here -- not
poorly measured, invisible:

* **A 180 degree turn about the aim axis.** The wrong sweep face is exactly this,
  and no amount of consensus will find it, because nothing in the file
  distinguishes a sensor from itself rotated about its own sight line. Only the
  photographs can see that the body is upside down.
* **A translation along the sight line.** Slide a sensor towards or away from the
  part and its ray is unchanged, so the correction a verdict reports is the
  *smallest* translation that reconciles it, never necessarily the whole error.

This module says so per sensor rather than quietly returning "good" -- a verdict
of "consistent" here means "consistent with everything the file can be asked",
which is a narrower claim than "correct" and is reported in those words.

Two further limits worth stating before anyone trusts a verdict:

* **Below four sensors there is no consensus and no verdict.** Two rays always
  meet somewhere and three generically pass within a hair of a common point, so a
  tight-looking intersection among three sensors is not evidence that any of them
  is right. The report says "cannot tell" and names the minimum rather than
  grading a coincidence.
* **A fault shared by most sensors reads as correct.** If two thirds of the file
  was translated by the same error, consensus will adopt that error as the truth
  and convict the honest third. The independent referees for that are the
  control-line axes, the clicked tie points, and the photographs -- never this
  module on its own, which measures agreement and cannot measure correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .geometry import AXIS_NAMES

MIN_CONSENSUS_SENSORS = 4
"""Fewest sensors that can produce a verdict at all.

Two aim rays always have a closest approach and three generically pass within a
hair of a common point, so with fewer than four a tight intersection is a
property of the geometry rather than evidence about the calibration. Four is the
first count at which one sensor can be the odd one out and be seen to be.
"""

AIM_AXES: tuple[tuple[str, np.ndarray], ...] = tuple(
    (f"{sign}{name}", np.asarray(vec, dtype=float) * mul)
    for name, vec in zip(AXIS_NAMES, np.eye(3))
    for sign, mul in (("+", 1.0), ("-", -1.0))
)
"""The six signed local axes, as candidates for "which way does a sensor look".

Inferred rather than assumed. The file's local convention is not written
anywhere in it, and guessing wrong turns the one real consensus this module has
into noise -- not loudly, which is the danger, just into a volume estimate with
no inliers. So all six are tried and the winner has to beat the runner-up by a
stated margin before it is used.
"""


# --------------------------------------------------------------- rotation input


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_ELEMENTARY = {"X": _rx, "Y": _ry, "Z": _rz}
_TURN = (_rx, _ry, _rz)


def euler_matrix(
    angles: Sequence[float],
    order: str = "XYZ",
    *,
    degrees: bool = True,
    intrinsic: bool = True,
) -> np.ndarray:
    """Euler angles to a rotation matrix, with the convention stated explicitly.

    There is no default worth having here. Euler angles are six orders times two
    frame conventions times two angle units, and every combination produces a
    valid-looking rotation matrix from the same three numbers -- so the wrong
    convention does not fail, it returns a plausible wrong answer and every
    downstream residual quietly absorbs it. That is why this takes the convention
    as an argument with no clever sniffing behind it, and why the importer records
    which one was used alongside the numbers.

    ``intrinsic`` means the axes travel with the body, so ``order="XYZ"`` is
    ``Rx @ Ry @ Rz`` -- read left to right, each rotation about the frame the
    previous one left behind. Extrinsic means the axes stay fixed in the world,
    which is the same product reversed; the two agree only for a palindromic
    order, which no three-distinct-axis order is.
    """
    if len(angles) != 3:
        raise ValueError("Euler input needs exactly three angles")
    order = order.upper()
    if len(order) != 3 or any(c not in _ELEMENTARY for c in order):
        raise ValueError(f"rotation order {order!r} is not three of X, Y and Z")
    radians = [math.radians(float(v)) if degrees else float(v) for v in angles]
    seq = list(zip(order, radians))
    if not intrinsic:
        seq = seq[::-1]
    R = np.eye(3)
    for axis, angle in seq:
        R = R @ _ELEMENTARY[axis](angle)
    return R


def quaternion_matrix(q: Sequence[float], *, scalar_first: bool = True) -> np.ndarray:
    """Unit quaternion to a rotation matrix.

    ``scalar_first`` picks ``(w, x, y, z)`` over ``(x, y, z, w)``. Both orders are
    in wide use and the two differ by a rotation rather than by a sign, so this is
    as consequential as the Euler order and just as unguessable from the numbers
    -- except in the one lucky case of near-identity rotations, where the scalar
    is the component near 1 and the rest near 0. :func:`quaternion_scalar_position`
    checks for that and declines to answer when it cannot tell.
    """
    v = np.asarray(q, dtype=float).reshape(4)
    w, x, y, z = (v[0], v[1], v[2], v[3]) if scalar_first else (v[3], v[0], v[1], v[2])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("quaternion has no magnitude")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


SCALAR_EVIDENCE = 0.9
"""How large the scalar component has to be before its size is evidence at all.

0.9 is a rotation of about 51 degrees. Past that the scalar stops being the
biggest component and some element of the vector part takes over, so "the big one
is the scalar" silently inverts -- which is not a tuning question but the limit of
the heuristic, and the reason it has to decline rather than extrapolate.
"""


def quaternion_scalar_position(quats: Iterable[Sequence[float]]) -> str | None:
    """Which end the scalar sits on, when the numbers can actually say.

    Sensors on a cell are mostly aimed in similar directions and rarely far from
    one axis of the cell frame, so for modest rotations the scalar is the largest
    component and the vector part is small. That is the whole basis of the
    inference, and it has a hard limit: past about 51 degrees the scalar is *not*
    the largest component any more, so unanimity on "the last slot holds the
    maximum" is equally consistent with a scalar-last file of small rotations and
    a scalar-first file of large ones. Those are opposite answers.

    So the test is not just that one end always wins, but that it wins *because
    the rotations are small* -- the winning component has to be consistently at
    least :data:`SCALAR_EVIDENCE`. Two samples minimum, because one quaternion
    trivially satisfies "always", and an answer from a single sample would be the
    coin toss this exists to refuse. Returning ``None`` is the common case and
    the right one: ask whoever wrote the file.
    """
    mags: list[np.ndarray] = [
        np.abs(np.asarray(q, dtype=float).reshape(4)) for q in quats
    ]
    if len(mags) < 2:
        return None
    ends = {0: "scalar_first", 3: "scalar_last"}
    for k, answer in ends.items():
        other = 3 - k
        if all(int(np.argmax(v)) == k for v in mags) and all(
            v[k] >= SCALAR_EVIDENCE for v in mags
        ):
            # Guard the mirror case explicitly: if the opposite end is just as
            # large, the layout is symmetric in the only thing being measured.
            if all(v[other] < v[k] for v in mags):
                return answer
    return None


# ------------------------------------------------------------------- the claims


@dataclass
class SensorClaim:
    """One sensor exactly as the calibration file states it, plus its parsed pose.

    ``raw`` is the file's own record, kept verbatim and never written back to. An
    import has to be reversible: once a verdict says a transform is wrong, the
    next question is always "what did the file actually say", and a parsed pose
    that has been through a convention choice and a unit conversion cannot answer
    that. This is the same bargain ``ImageView.R_unrefined`` makes.
    """

    name: str
    t: np.ndarray
    """Position in the file's own units."""
    R: np.ndarray
    """Sensor-to-world rotation: ``X_world = R @ p_local + t``, the same sense as
    :class:`~multispy.project.ObjectInstance`. A file written world-to-sensor is
    inverted on the way in, by the importer and not here, so the choice is made
    in one place and recorded beside the numbers."""
    raw: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def aim(self, axis: np.ndarray) -> np.ndarray:
        """Unit world direction this sensor looks along, given its local aim axis."""
        d = self.R @ np.asarray(axis, dtype=float)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else d

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.R, dtype=float), np.array(self.t, dtype=float)


# -------------------------------------------------------------- fault catalogue


@dataclass
class Fault:
    """A named, correctable way a calibration transform goes wrong.

    Each one is a concrete operation on a pose, not a residual pattern to match.
    The test is therefore direct: apply the correction and see whether the sensor
    joins the consensus. A sensor that only agrees once its Y is negated was
    calibrated against the mirrored feature, and saying that is worth far more
    than reporting twice its Y offset as an error magnitude.
    """

    key: str
    label: str
    """What to tell the user, in words they would use about their own cell."""
    apply: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    aim_blind: bool = False
    """``True`` when the aim-ray consensus cannot see this fault at all.

    Not a limitation of the implementation: a turn about the sensor's own sight
    line leaves the sight line where it was, so the ray test is blind to it by
    construction. Reporting such a fault as "not found" would be a lie of
    omission, so the report lists it as untested and points at the photographs.
    """


def _mirror(axis: int) -> np.ndarray:
    S = np.eye(3)
    S[axis, axis] = -1.0
    return S


def _flip_local(axis: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    turn = _TURN[axis](math.pi)

    def op(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return R @ turn, np.array(t, dtype=float)

    return op


def _negate_axis(axis: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def op(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.array(t, dtype=float)
        u[axis] = -u[axis]
        return np.array(R, dtype=float), u

    return op


def _reflect_pose(axis: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    S = _mirror(axis)

    def op(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # A reflection is improper, so reflecting a rotation outright would hand
        # back a matrix of determinant -1 and no meaning as a pose. Conjugating by
        # the reflection keeps it proper, and is what "the same sensor, mounted on
        # the other side of the cell" actually means.
        return S @ np.asarray(R, dtype=float) @ S, S @ np.asarray(t, dtype=float)

    return op


def fault_catalogue() -> list[Fault]:
    """The faults seen on this cell, in the order they are worth trying.

    Deliberately short. Every entry was reported from the shop floor rather than
    imagined, because a catalogue of speculative faults costs accuracy: each extra
    correction is another chance for a wrong one to fit a noisy sensor better than
    the truth does, which is the multiple-comparisons trap wearing a different
    coat. The report prints how many corrections were tried, for that reason.
    """
    return [
        Fault(
            "mirror_y",
            "calibrated against the mirrored feature across the cell, so Y has the "
            "wrong sign",
            _negate_axis(1),
        ),
        Fault(
            "reflect_y",
            "mounted as the mirror image of what it was calibrated as -- whole pose "
            "reflected through the cell centreline",
            _reflect_pose(1),
        ),
        Fault(
            "flip_z",
            "body turned 180 degrees about its own aim axis: the wrong sweep face",
            _flip_local(2),
            aim_blind=True,
        ),
        Fault(
            "flip_x",
            "body turned 180 degrees about local X",
            _flip_local(0),
        ),
        Fault(
            "flip_y",
            "body turned 180 degrees about local Y",
            _flip_local(1),
        ),
    ]


# --------------------------------------------------------------- aim consensus


def _closest_point(positions: np.ndarray, dirs: np.ndarray) -> np.ndarray | None:
    """The point minimising summed squared distance to a set of rays.

    Each ray contributes ``(I - d d^T)``, the projector onto the plane across it,
    so the normal equations are a plain 3x3 solve. It is ill conditioned exactly
    when the rays are near parallel -- a bank of sensors all looking the same way
    down the cell pins the volume across the beam and not along it -- so the
    conditioning is checked rather than assumed, and a degenerate pencil returns
    nothing instead of a point somewhere out past the far wall.
    """
    M = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(positions, dirs):
        P = np.eye(3) - np.outer(d, d)
        M += P
        b += P @ p
    w = np.linalg.svd(M, compute_uv=False)
    if w[0] < 1e-12 or w[-1] / w[0] < 1e-6:
        return None
    return np.linalg.solve(M, b)


def _ray_distances(point: np.ndarray, positions: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Perpendicular distance from ``point`` to each ray, ignoring direction of travel.

    Deliberately not clamped to the forward half-ray. A calibration that puts the
    working volume *behind* a sensor is reported by the behind-ness itself, as a
    separate and far more specific complaint than a large distance; clamping here
    would flatten that signal into a generic big residual.
    """
    rel = positions - point
    along = np.einsum("ij,ij->i", rel, dirs)
    perp = rel - along[:, None] * dirs
    return np.linalg.norm(perp, axis=1)


def robust_scale(errs: np.ndarray, floor: float) -> float:
    """A spread read off the data, never below the tolerance that was stated.

    Median absolute deviation rather than RMS, for the reason
    ``solve._robust_scale`` gives: the estimate must not be inflated by the very
    outlier it exists to catch. The floor does the other half of the work -- a set
    of sensors that happen to agree to a millimetre has not earned a millimetre
    threshold, and without a floor one tight cluster would convict every honest
    sensor outside it.
    """
    errs = np.asarray(errs, dtype=float)
    if errs.size == 0:
        return float(floor)
    med = float(np.median(errs))
    mad = float(np.median(np.abs(errs - med)))
    return max(float(floor), 1.4826 * mad)


@dataclass
class AimConsensus:
    """Where the sensors agree they are all looking, and who disagrees."""

    ok: bool = False
    reason: str = ""
    axis: str = ""
    """The local aim axis this was computed with: inferred unless supplied."""
    axis_margin: float = 0.0
    """How much worse the runner-up axis was, as a ratio. Near 1 means the
    inference did not really decide, and the axis should be supplied rather than
    guessed -- every verdict depends on it."""
    point: list[float] = field(default_factory=list)
    """The working volume's centre, in the file's units."""
    scale: float = 0.0
    """Robust spread of inlier ray distances: the cell's own idea of "close"."""
    threshold: float = 0.0
    inliers: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    distances: dict[str, float] = field(default_factory=dict)
    behind: list[str] = field(default_factory=list)
    """Sensors whose working volume lies behind them. Not a distance complaint but
    a direction one: a sensor cannot measure what is at its back, so this is a
    sign error or a flipped aim axis rather than a drifted position."""
    n_front: int = 0
    """How many sensors have the working volume in front of them.

    The only thing that can choose the *sign* of the aim axis. Perpendicular ray
    distance is identical for an axis and its negation -- the ray is the same line
    either way -- so residual cannot separate +Z from -Z, not because the search
    is weak but because the quantity being minimised does not depend on it. Which
    way a sensor faces is decided here instead, by the one test that cares."""

    def distance(self, name: str) -> float:
        return float(self.distances.get(name, float("nan")))


def aim_consensus(
    claims: Sequence[SensorClaim],
    *,
    axis: str | None = None,
    tol: float = 50.0,
    k: float = 3.0,
) -> AimConsensus:
    """Find the working volume the sensors agree on, robustly, and name the strays.

    The minimal set is two, because one ray determines no point and two determine
    their closest approach, so every pair is a hypothesis and the pair count is
    small enough to enumerate outright for any real cell. Each hypothesis is
    scored by how many of the other sensors it puts within tolerance -- which is
    the only thing that separates "this sensor is wrong" from "everything else is
    wrong", and the reason this refuses to answer below
    :data:`MIN_CONSENSUS_SENSORS`.
    """
    live = [c for c in claims if c.enabled]
    if len(live) < MIN_CONSENSUS_SENSORS:
        return AimConsensus(
            ok=False,
            reason=(
                f"{len(live)} sensor(s) is too few to tell a bad transform from a bad "
                f"consensus: {MIN_CONSENSUS_SENSORS} are needed before one can be the odd "
                f"one out and be seen to be"
            ),
        )

    candidates = (
        AIM_AXES if axis is None else tuple((nm, v) for nm, v in AIM_AXES if nm == axis)
    )
    if not candidates:
        raise ValueError(f"{axis!r} is not one of {[nm for nm, _ in AIM_AXES]}")

    scored: list[tuple[int, float, AimConsensus]] = []
    for name, vec in candidates:
        got = _consensus_for_axis(live, name, vec, tol=tol, k=k)
        if got.ok:
            scored.append((-len(got.inliers), got.scale, got))

    if not scored:
        return AimConsensus(
            ok=False,
            axis=axis or "",
            reason=(
                "no local aim axis produced a usable intersection -- the sensors' "
                "directions may be near parallel, which pins the volume across the beam "
                "but not along it"
            ),
        )

    # An axis and its negation describe the same line, so they always score
    # identically and comparing them would report a dead heat on every clean file.
    # Rank the three *families* on residual, then let the front/behind count pick
    # the sign -- which is the only measurement that distinguishes them.
    scored.sort(key=lambda s: (s[0], s[1], -s[2].n_front))
    best = scored[0][2]
    family = best.axis[1]
    same = [s for s in scored if s[2].axis[1] == family]
    if len(same) > 1 and same[1][2].n_front > best.n_front:
        best = same[1][2]

    rival = next((s for s in scored if s[2].axis[1] != family), None)
    if rival is not None:
        best.axis_margin = float(rival[2].scale / max(best.scale, 1e-9))
        if len(best.inliers) == len(rival[2].inliers) and best.axis_margin < 1.5:
            best.reason = (
                f"the aim axis could not be inferred: local {family} and "
                f"{rival[2].axis[1]} fit about equally well ({best.axis_margin:.2f}x apart "
                f"on residual, same inlier count). Supply the axis rather than trusting a "
                f"coin toss -- every verdict below depends on it"
            )
    if best.n_front * 2 < len(live):
        best.reason = (
            (best.reason + " ") if best.reason else ""
        ) + (
            f"only {best.n_front} of {len(live)} sensors have the working volume in front "
            f"of them along {best.axis}, which usually means the aim axis points out of the "
            f"back of the body and its sign should be flipped"
        )
    return best


def _consensus_for_axis(
    live: Sequence[SensorClaim], name: str, vec: np.ndarray, *, tol: float, k: float
) -> AimConsensus:
    positions = np.array([c.t for c in live], dtype=float)
    dirs = np.array([c.aim(vec) for c in live], dtype=float)
    n = len(live)

    best_point: np.ndarray | None = None
    best_count = -1
    best_resid = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            p = _closest_point(positions[[i, j]], dirs[[i, j]])
            if p is None:
                continue
            d = _ray_distances(p, positions, dirs)
            keep = d <= tol
            count = int(keep.sum())
            resid = float(np.mean(d[keep])) if count else math.inf
            if count > best_count or (count == best_count and resid < best_resid):
                best_point, best_count, best_resid = p, count, resid

    if best_point is None or best_count < MIN_CONSENSUS_SENSORS - 1:
        return AimConsensus(
            ok=False, axis=name, reason="no pair of rays gathered a consensus"
        )

    # Refine on the inliers only, then re-measure. Refitting on everything would
    # drag the volume towards the very sensors being judged -- outliers helping to
    # produce the residuals they are then screened by, which is the whole trap.
    point = best_point
    for _ in range(8):
        d = _ray_distances(point, positions, dirs)
        keep = d <= max(k * robust_scale(d, tol), tol)
        if int(keep.sum()) < 2:
            break
        nxt = _closest_point(positions[keep], dirs[keep])
        if nxt is None:
            break
        moved = float(np.linalg.norm(nxt - point))
        point = nxt
        if moved < 1e-9:
            break

    d = _ray_distances(point, positions, dirs)
    # The scale is read off the sensors that are plausibly good, so that a metre
    # of error cannot widen the threshold it is about to be judged against.
    inner = d[d <= max(3.0 * robust_scale(d, tol), tol)]
    scale = robust_scale(inner, tol)
    threshold = max(k * scale, tol)
    keep = d <= threshold
    rel = positions - point
    along = np.einsum("ij,ij->i", rel, dirs)

    return AimConsensus(
        ok=True,
        axis=name,
        point=[float(v) for v in point],
        scale=float(scale),
        threshold=float(threshold),
        inliers=[c.name for c, f in zip(live, keep) if f],
        outliers=[c.name for c, f in zip(live, keep) if not f],
        distances={c.name: float(v) for c, v in zip(live, d)},
        # ``rel`` runs from the volume to the sensor, so a positive projection on
        # the aim direction has the sensor looking away from the volume.
        behind=[c.name for c, a, f in zip(live, along, keep) if a > 0 and not f],
        n_front=int((along < 0).sum()),
    )


# ------------------------------------------------------------------- verdicts


CONSISTENT = "consistent"
"""Agrees with everything the file can be asked. Narrower than "correct": a flip
about the aim axis lands here too, because nothing in the file can see it."""
EXPLAINED = "explained"
"""Disagrees, and one named fault accounts for it. The most useful verdict there
is -- it carries a correction, not just a magnitude."""
SUSPECT = "suspect"
"""Disagrees and no catalogued fault explains it. Either a drift the catalogue
does not cover, or a fault worth adding once it is understood."""
UNTESTABLE = "untestable"
"""No independent consensus could be formed for this sensor, so nothing is
claimed about it. Not a pass."""


@dataclass
class SensorVerdict:
    """What can and cannot be said about one sensor's transform."""

    name: str
    verdict: str = UNTESTABLE
    reason: str = ""
    distance: float = 0.0
    """How far the sensor's aim ray passes from the working volume the *other*
    sensors agree on, in file units. Leave-one-out, always: a sensor that helped
    place the volume would be scored against its own vote."""
    threshold: float = 0.0
    fault: str = ""
    """Catalogue key of the correction that explains it, or empty."""
    fault_label: str = ""
    residual_after: float = 0.0
    """Ray distance once the named correction is applied. The point of the
    catalogue is that this collapses to inlier scale rather than merely
    improving: a correction that halves a metre has not explained anything."""
    correction: list[float] = field(default_factory=list)
    """The free translation that would put this sensor's ray through the volume.
    Its magnitude equals ``distance`` by construction -- it is reported as a
    vector because the *direction* is the diagnosis: dominantly along X is the
    wrong-nominal fault, dominantly along Y the mirrored-feature one."""
    dominant_axis: str = ""
    dominant_fraction: float = 0.0
    """How much of the correction lies along ``dominant_axis``.

    Reported because the axis of a large error is worth knowing -- a metre along
    X is the signature of a transform calibrated against the wrong nominal
    feature. It is *not* on its own evidence of which fault occurred, and an
    earlier version of this module said it was: measured on the test cell, a two
    degree nudge of a sensor's aim also produces a correction 95-99% on one axis,
    because the standoff direction dominates it. Magnitude separates those two;
    direction does not."""
    standoff: float = 0.0
    """Distance from the sensor to the working volume along its own sight line."""
    implied_angle: float = 0.0
    """The aim error, in degrees, that would produce this miss at this standoff.

    The number that makes a miss interpretable. 31 mm sounds like a position
    error and reads very differently as 2.0 degrees of aim at 900 mm -- and a
    fraction of a degree is a plausible mechanical nudge, where 40 degrees is
    not and points at a translation instead."""
    behind: bool = False
    """The working volume is behind this sensor. A sign error or a flipped aim
    axis, not a drift, and worth saying separately."""
    aim_blind: list[str] = field(default_factory=list)
    """Faults that could not be tested for this sensor at all, because the aim
    ray does not move under them. Listed on every verdict including a passing
    one, since that is exactly where it would otherwise be mistaken for a clean
    bill of health."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "reason": self.reason,
            "distance": self.distance,
            "threshold": self.threshold,
            "fault": self.fault,
            "fault_label": self.fault_label,
            "residual_after": self.residual_after,
            "correction": list(self.correction),
            "dominant_axis": self.dominant_axis,
            "dominant_fraction": self.dominant_fraction,
            "standoff": self.standoff,
            "implied_angle": self.implied_angle,
            "behind": self.behind,
            "aim_blind": list(self.aim_blind),
        }


@dataclass
class CalibrationReport:
    """The whole file judged against itself, and what that is and is not worth."""

    ok: bool = False
    reason: str = ""
    axis: str = ""
    axis_margin: float = 0.0
    volume: list[float] = field(default_factory=list)
    scale: float = 0.0
    threshold: float = 0.0
    units: str = "mm"
    n_sensors: int = 0
    n_agreed: int = 0
    n_explained: int = 0
    n_suspect: int = 0
    faults_tried: int = 0
    span: float = 0.0
    """Largest distance between any two sensor positions, in file units. The units
    check: a measurement cell is metres across, so a span of 3 says the file is in
    metres while the project is in millimetres, and a span of 3000000 says
    something worse. ``part.py`` exists to catch the same mistake and for the same
    reason -- a factor of 1000 applied quietly leaves every deviation reading as a
    clean zero."""
    verdicts: list[SensorVerdict] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def grade(self, good: float, warn: float) -> str:
        """Overall traffic light, on the same pattern as ``OriginUncertainty.grade``.

        Graded on *how many sensors disagree and whether anything explains them*
        rather than on a residual, because the residual of a consensus fit is a
        statement about the sensors that agreed and says nothing about the ones
        that did not.
        """
        if not self.ok or self.n_sensors < MIN_CONSENSUS_SENSORS:
            return "unknown"
        if self.n_suspect:
            return "bad"
        if self.n_explained:
            return "warn"
        worst = max((v.distance for v in self.verdicts), default=0.0)
        if worst <= good:
            return "good"
        return "warn" if worst <= warn else "bad"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "axis": self.axis,
            "axis_margin": self.axis_margin,
            "volume": list(self.volume),
            "scale": self.scale,
            "threshold": self.threshold,
            "units": self.units,
            "n_sensors": self.n_sensors,
            "n_agreed": self.n_agreed,
            "n_explained": self.n_explained,
            "n_suspect": self.n_suspect,
            "faults_tried": self.faults_tried,
            "span": self.span,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "messages": list(self.messages),
        }


def _span(claims: Sequence[SensorClaim]) -> float:
    if len(claims) < 2:
        return 0.0
    P = np.array([c.t for c in claims], dtype=float)
    # Diameter of the bounding box rather than every pairwise distance: for a few
    # dozen sensors either is instant, and this one stays instant for a few
    # thousand without changing what it is used for.
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))


def diagnose(
    claims: Sequence[SensorClaim],
    *,
    axis: str | None = None,
    tol: float = 10.0,
    warn: float = 50.0,
    k: float = 3.0,
    units: str = "mm",
) -> CalibrationReport:
    """Judge every sensor's transform against the consensus of the others.

    ``tol`` is the translation error that counts as agreement and ``warn`` the one
    that counts as bad; both are in the file's own units and both belong to
    whoever is inspecting the cell, not to this module.

    The scoring is leave-one-out throughout, which is the point the handoff
    insisted on: a sensor is measured against a working volume computed from the
    *other* sensors, so it never helps produce the residual it is then judged by.
    That costs one 3x3 solve per sensor and removes the whole class of error where
    a bad transform quietly widens its own threshold.
    """
    live = [c for c in claims if c.enabled]
    report = CalibrationReport(
        n_sensors=len(live), units=units, span=_span(live), faults_tried=len(fault_catalogue())
    )

    base = aim_consensus(live, axis=axis, tol=max(warn, tol), k=k)
    report.axis = base.axis
    report.axis_margin = base.axis_margin
    if not base.ok:
        report.reason = base.reason
        report.verdicts = [
            SensorVerdict(name=c.name, verdict=UNTESTABLE, reason=base.reason) for c in live
        ]
        return report

    report.ok = True
    report.volume = list(base.point)
    report.scale = base.scale
    report.threshold = base.threshold
    if base.reason:
        report.messages.append(base.reason)

    vec = dict(AIM_AXES)[base.axis]
    catalogue = fault_catalogue()
    blind = [f.label for f in catalogue if f.aim_blind]
    inlier_names = set(base.inliers)
    positions = {c.name: np.array(c.t, dtype=float) for c in live}
    dirs = {c.name: c.aim(vec) for c in live}

    for c in live:
        others = [o for o in live if o.name != c.name and o.name in inlier_names]
        if len(others) < MIN_CONSENSUS_SENSORS - 1:
            report.verdicts.append(
                SensorVerdict(
                    name=c.name,
                    verdict=UNTESTABLE,
                    aim_blind=list(blind),
                    reason=(
                        f"only {len(others)} other sensor(s) agree on a working volume, which "
                        f"is not enough to judge this one against something it had no part in"
                    ),
                )
            )
            continue

        P = np.array([positions[o.name] for o in others])
        D = np.array([dirs[o.name] for o in others])
        point = _closest_point(P, D)
        if point is None:
            report.verdicts.append(
                SensorVerdict(
                    name=c.name,
                    verdict=UNTESTABLE,
                    aim_blind=list(blind),
                    reason="the other sensors' rays are too near parallel to place a volume",
                )
            )
            continue

        inner = _ray_distances(point, P, D)
        threshold = max(k * robust_scale(inner, tol), tol)
        report.verdicts.append(
            _judge(c, point, threshold, vec, catalogue, blind, warn)
        )

    report.n_agreed = sum(1 for v in report.verdicts if v.verdict == CONSISTENT)
    report.n_explained = sum(1 for v in report.verdicts if v.verdict == EXPLAINED)
    report.n_suspect = sum(1 for v in report.verdicts if v.verdict == SUSPECT)

    if blind:
        report.messages.append(
            "untested on every sensor, including the ones that passed: "
            + "; ".join(blind)
            + ". A turn about the aim axis leaves the aim ray exactly where it was, so no "
            "test in this module can see it -- the photographs are the only referee for it"
        )
    # The trigger is the size of the *disagreeing* group, not of the agreeing one.
    # A shared error inverts the answer rather than splitting it: translate most
    # of the file by one offset and those sensors still form a consistent pencil,
    # just about a displaced volume, so consensus adopts the error as truth and
    # the honest sensors become the outliers. In that state the majority looks
    # healthy and the warning has to come from how many are being convicted.
    dissent = len(report.verdicts) - report.n_agreed
    if dissent >= 2 and dissent >= len(report.verdicts) * 0.25:
        report.messages.append(
            f"{dissent} of {len(report.verdicts)} sensors disagree with the consensus, which "
            f"is too many to be independent mistakes. If a group of sensors shares one error "
            f"this reads that error as the truth and convicts the honest ones instead -- the "
            f"two readings are indistinguishable from the file alone, so check against the "
            f"control-line axes or the photographs before correcting anything"
        )
    return report


def _judge(
    claim: SensorClaim,
    point: np.ndarray,
    threshold: float,
    vec: np.ndarray,
    catalogue: Sequence[Fault],
    blind: Sequence[str],
    warn: float,
) -> SensorVerdict:
    """One sensor against one volume it did not help place."""
    R, t = claim.pose()
    P = t.reshape(1, 3)
    D = claim.aim(vec).reshape(1, 3)
    distance = float(_ray_distances(point, P, D)[0])
    rel = t - point
    along = float(rel @ D[0])
    perp = rel - along * D[0]
    correction = -perp
    mag = float(np.linalg.norm(correction))
    axis_i = int(np.argmax(np.abs(correction))) if mag > 1e-12 else 0

    standoff = abs(along)
    v = SensorVerdict(
        name=claim.name,
        distance=distance,
        threshold=threshold,
        correction=[float(x) for x in correction],
        dominant_axis=AXIS_NAMES[axis_i],
        dominant_fraction=float(abs(correction[axis_i]) / mag) if mag > 1e-12 else 0.0,
        standoff=float(standoff),
        implied_angle=(
            math.degrees(math.atan2(distance, standoff)) if standoff > 1e-9 else 90.0
        ),
        behind=along > 0 and distance > threshold,
        aim_blind=list(blind),
    )

    if distance <= threshold:
        v.verdict = CONSISTENT
        v.reason = (
            f"aims within {distance:.1f} of the working volume the other sensors agree on "
            f"(threshold {threshold:.1f}). Consistent with everything the file can be asked, "
            f"which is not the same as correct"
        )
        return v

    # A correction has to collapse the residual to inlier scale, not merely
    # improve it. Halving a metre still leaves the sensor nowhere near the volume,
    # and accepting that as an explanation would attach a confident diagnosis to
    # a sensor nobody has actually accounted for.
    best: tuple[float, Fault] | None = None
    for fault in catalogue:
        if fault.aim_blind:
            continue
        Rc, tc = fault.apply(R, t)
        d = float(
            _ray_distances(
                point,
                tc.reshape(1, 3),
                (Rc @ vec / max(float(np.linalg.norm(Rc @ vec)), 1e-12)).reshape(1, 3),
            )[0]
        )
        if d <= threshold and (best is None or d < best[0]):
            best = (d, fault)

    if best is not None:
        v.verdict = EXPLAINED
        v.fault = best[1].key
        v.fault_label = best[1].label
        v.residual_after = best[0]
        v.reason = (
            f"misses the working volume by {distance:.1f}, and exactly one catalogued fault "
            f"accounts for it: {best[1].label}. Correcting it brings the miss to "
            f"{best[0]:.1f}, inside the {threshold:.1f} the agreeing sensors hold to"
        )
        return v

    v.verdict = SUSPECT
    # Magnitude, read against the standoff, is what separates a nudged aim from a
    # mistyped nominal. Direction does not: measured on the test cell, a two
    # degree nudge also lands 95-99% on a single axis, because the standoff
    # direction dominates the correction whatever caused it.
    if mag <= warn and v.implied_angle < 5.0:
        shape = (
            f"that is {v.implied_angle:.1f} degrees of aim error at a {standoff:.0f} standoff, "
            f"which is the size of a mechanical nudge -- the body will have barely moved while "
            f"its measurement volume moved this far"
        )
    elif mag > 10.0 * max(warn, 1.0) and v.dominant_fraction > 0.9:
        shape = (
            f"an error this large and this nearly along {v.dominant_axis} alone "
            f"({v.dominant_fraction * 100:.0f}%) is the signature of a transform calibrated "
            f"against the wrong nominal feature rather than of anything mechanical"
        )
    else:
        shape = (
            f"at {v.implied_angle:.1f} degrees of implied aim error over a {standoff:.0f} "
            f"standoff it is too large for a nudge and too small for a wrong nominal, so it "
            f"is worth looking at this sensor in the photographs before correcting anything"
        )
    v.reason = (
        f"misses the working volume by {distance:.1f} against a threshold of {threshold:.1f}"
        + (
            ", and the volume is behind it, so this is a sign error or a flipped aim axis "
            "rather than a drift"
            if v.behind
            else ""
        )
        + f". No catalogued correction brings it in. The smallest translation that would is "
        f"{mag:.1f}, mostly along {v.dominant_axis}: {shape}"
        + (f". Past the {warn:.0f} that counts as badly wrong" if mag > warn else "")
    )
    return v

