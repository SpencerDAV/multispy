"""A sensor calibration file is a set of claims, and this module argues with them.

A measurement cell's calibration file states, per sensor, where that sensor sits,
which way it faces, how far it stands off the part, how big its field of view is,
and which feature it is pointed at. Importing that is trivial. Believing it is
the problem: these files go wrong in specific, repeatable ways -- a sensor
calibrated against the wrong nominal feature and landing a metre out, one
calibrated against the mirror-image feature across the cell so its Y has the
wrong sign, one whose body is turned 180 degrees about its own sight line so the
laser sweeps the wrong face, and one whose aim has been nudged since, which moves
the body hardly at all and its measurement volume a long way.

**The file falsifies itself, and that is the whole design.** It states each
sensor's field of view *and* the nominal of the feature that sensor is supposed
to measure. If the feature lies outside the field of view, the sensor cannot see
the thing it claims to be measuring, and that is a contradiction inside a single
row -- no consensus, no fitting, no reference data of any kind required. Measured
on the real Perceptron export this was built against: 15 sensors, and the four
features that fail this test fail it by 2.3 to 7.5 times their own FOV box, which
no reading of any convention can explain away.

That is worth dwelling on, because an earlier version of this module did
something much weaker and more fragile. It assumed every sensor on a cell aims at
one shared working volume and built a consensus from the pencil of aim rays. That
holds for a cell inspecting a single part; it is simply false for a car body,
where the sensors are spread down the length of the vehicle and each has its own
field of view on its own feature -- on the real file the FOV centres span 2.7
metres. Consensus among them was a confident intersection of a pencil that is not
one. :func:`aim_consensus` survives for the cells where it does apply and now
refuses, loudly, where it does not.

**Faults cluster by calibration session, not by sensor.** The file carries a
calibration date per sensor, and grouping by it is the single most diagnostic
thing that can be done to one of these files. On the real export, every sensor
calibrated on one particular day is wrong by the same ~412 mm in part X and no
sensor from any other session is wrong at all. Three "suspect sensor" verdicts
are a list of things to go and check; one "this session was calibrated against a
part positioned 412 mm out" is a cause, and it is fixed in one action. So
sessions are reported as first-class objects alongside sensors.

**Three faults are discrete**, which matters more than it sounds. A generic
robust regression calls a bad sensor a large residual and reports a magnitude,
but "412 mm of translation error" is a number someone still has to interpret,
where "calibrated against the mirrored feature, negate Y" is a diagnosis with a
fix attached. So the faults are enumerated and tested by name, each a concrete
operation on a pose, and a correction only counts as an explanation if it brings
the sensor's features inside its field of view -- halving a metre explains
nothing.

**The honest limits, which are the reason the photographs are still needed.**

* **A 180 degree turn about the aim axis is invisible to every test here.** It
  moves neither the sight line nor the field of view, so the wrong sweep face
  comes back "consistent". Every verdict, including a passing one, carries the
  fact that this fault was never tested. Measured off the sensor CAD, it is only
  weakly visible in a photograph too: the bodies are 70 x 168 mm in section, so
  the turn maps the cross-section onto itself and preserves the outline, leaving
  only off-centre detail to tell them apart.
* **A translation along the sight line is nearly invisible.** The depth half-range
  of these sensors is 75-120 mm against lateral extents of 55-160 mm, so the FOV
  test has some power in depth, but much less than across it.
* **A feature inside the field of view is not a correct transform.** The FOV box
  is 100-300 mm across and the tolerance that matters is a few millimetres. This
  test convicts gross faults and is silent about everything finer, by design.
  Reading "in FOV" as "calibrated well" would be a serious misreading, so the
  verdict says what it is narrower than in words.
* **A fault shared by a whole session cannot be told from a moved part.** If every
  sensor calibrated on one day shares an offset, this module cannot know whether
  the sensors are wrong or the nominals they were calibrated against are. It
  reports the pattern and says which readings are consistent with it.

The independent referees for all of that are the clicked tie-point reprojection,
the control-line axes, the mounting structure the sensors must physically sit on,
and the photographs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .geometry import AXIS_NAMES

MIN_CONSENSUS_SENSORS = 4
"""Fewest sensors that can produce a *consensus* verdict.

Two aim rays always have a closest approach and three generically pass within a
hair of a common point, so below four a tight intersection is a property of the
geometry rather than evidence about the calibration. Four is the first count at
which one sensor can be the odd one out and be seen to be.

The field-of-view test has no such floor: it compares a sensor against its own
declared optics, so it works on a file with one sensor in it.
"""

EDGE_RATIO = 1.3
"""How far outside its field of view a feature has to be before the sensor is
convicted rather than flagged.

Between 1.0 and this, the feature is at the edge of the field of view and the
honest reading is that something is marginal -- a feature nominal slightly wrong,
a sensor that was always tight on that feature, or the frustum being approximated
by its far plane here. Measured on the real export, the two marginal cases sit at
1.01x and 1.02x while the genuine faults sit at 2.3x to 7.5x, so nothing of
consequence lives near this threshold and it is not a number anyone has to tune.
"""

AIM_AXES: tuple[tuple[str, np.ndarray], ...] = tuple(
    (f"{sign}{name}", np.asarray(vec, dtype=float) * mul)
    for name, vec in zip(AXIS_NAMES, np.eye(3))
    for sign, mul in (("+", 1.0), ("-", -1.0))
)
"""The six signed local axes, as candidates for "which way does a sensor look"."""


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
    as an argument with no clever sniffing behind it.

    Not needed for the Perceptron export, which carries direction cosines and so
    has no convention to get wrong; kept for the files that do.

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
    -- except in the one lucky case of near-identity rotations, where
    :func:`quaternion_scalar_position` can sometimes tell.
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
    trivially satisfies "always". Returning ``None`` is the common case and the
    right one: ask whoever wrote the file.
    """
    mags: list[np.ndarray] = [np.abs(np.asarray(q, dtype=float).reshape(4)) for q in quats]
    if len(mags) < 2:
        return None
    for k, answer in ((0, "scalar_first"), (3, "scalar_last")):
        other = 3 - k
        if (
            all(int(np.argmax(v)) == k for v in mags)
            and all(v[k] >= SCALAR_EVIDENCE for v in mags)
            and all(v[other] < v[k] for v in mags)
        ):
            return answer
    return None


# ------------------------------------------------------------------- the claims


@dataclass
class FeatureClaim:
    """One feature a sensor is pointed at, as the file states it.

    The unit of falsification. A sensor is only wrong *about something*, and this
    is the something: the file says this sensor measures this feature at this
    nominal, which is a claim the sensor's own declared optics can contradict.
    """

    name: str
    nominal: np.ndarray
    """Where the drawing puts this feature, in the file's own units and frame."""
    algorithm: str = ""
    offsets: tuple[float | None, float | None, float | None] = (None, None, None)
    """The deviation the system reported, per axis, where it reported one.

    A second, independent channel on the same question: a sensor reporting 412 mm
    of deviation against a +-4 mm spec limit is not measuring a part that is 412 mm
    out, it is a sensor whose transform is wrong. Independent of the FOV test
    because it comes from measurement rather than from geometry, and weaker
    because a feature can be out of spec for honest reasons."""
    limits: tuple[
        tuple[float | None, float | None],
        tuple[float | None, float | None],
        tuple[float | None, float | None],
    ] = ((None, None), (None, None), (None, None))
    """Upper and lower spec limit per axis, where the file states them."""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorClaim:
    """One sensor exactly as the calibration file states it, plus its features.

    ``raw`` is the file's own record, kept verbatim and never written back to. An
    import has to be reversible: once a verdict says a transform is wrong, the
    next question is always "what did the file actually say", and a parsed pose
    that has been through a convention choice and a unit conversion cannot answer
    that. This is the same bargain ``ImageView.R_unrefined`` makes.
    """

    serial: str
    """The sensor's identity. In a feature-level export the rows are features and
    several share a sensor, so this is what must be grouped on -- an importer that
    treated rows as sensors would report the same transform several times and
    count one fault as many."""
    M: np.ndarray = field(default_factory=lambda: np.eye(3))
    """Rows are the sensor's local axes expressed in the part frame, which is how
    a direction-cosine block is written. So ``R_sensor_to_world`` is ``M.T`` --
    see :meth:`R`, which exists so that transpose is written once."""
    cfov: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Centre of the field of view, in the part frame. This, not the sensor body,
    is what the file actually pins down, and it is what the feature nominals have
    to be compared against."""
    fov_near: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fov_far: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Half-extents of the view frustum at its near and far planes. The component
    on the depth axis is the depth half-range; the other two are lateral."""
    standoff: float = 0.0
    """Declared distance from the sensor to the centre of its field of view. Given
    rather than inferred, and it is what turns a FOV centre into a body position."""
    part_number: str = ""
    """Sensor model. Also the CAD filename, in the export this was built against,
    which is what lets the photographic check project the right body."""
    sweep_face: str = ""
    calibrated: str = ""
    """Calibration timestamp, verbatim. The grouping key for sessions, and the
    most diagnostic column in the file -- see :func:`session_key`."""
    features: list[FeatureClaim] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def R(self) -> np.ndarray:
        """Sensor-to-world rotation: ``X_world = R @ p_local + t``.

        The same sense as :class:`~multispy.project.ObjectInstance`, so a claim
        can become an instance without anyone having to remember which way round
        this file's block was written.
        """
        return np.asarray(self.M, dtype=float).T

    def aim(self, depth_axis: int) -> np.ndarray:
        """Unit world direction the sensor looks along."""
        d = np.asarray(self.M, dtype=float)[depth_axis]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else d

    def position(self, depth_axis: int, *, sign: float = 1.0) -> np.ndarray:
        """Where the body sits: back off the FOV centre along the sight line.

        ``sign`` is the one thing the file does not settle -- whether the depth
        axis points from the sensor at the part or the other way. It is left as an
        argument rather than guessed because the photographs settle it in one
        look, and a wrong sign puts the body an equal distance the wrong side of
        the part, which is obvious once drawn and invisible in a residual.
        """
        return np.asarray(self.cfov, dtype=float) - sign * self.standoff * self.aim(depth_axis)

    def local(self, world: np.ndarray, depth_axis: int = 0) -> np.ndarray:
        """A part-frame point in the sensor's local axes, relative to the FOV centre."""
        return np.asarray(self.M, dtype=float) @ (
            np.asarray(world, dtype=float) - np.asarray(self.cfov, dtype=float)
        )

    def fov_limits(self, depth_axis: int) -> np.ndarray:
        """Per-axis half-extent a feature is allowed to sit at, deliberately generous.

        The lateral limits are taken from the *far* plane, which is the widest the
        frustum ever gets, rather than interpolated to the feature's own depth. So
        a feature this calls outside the field of view is outside it at any depth
        -- the test is built to be conservative, since its output is a conviction.

        The depth axis takes the larger of the two planes' depth figures for the
        same reason.
        """
        far = np.abs(np.asarray(self.fov_far, dtype=float))
        near = np.abs(np.asarray(self.fov_near, dtype=float))
        lim = np.maximum(far, near)
        return np.where(lim > 1e-9, lim, 1e-9)


def session_key(claim: SensorClaim) -> str:
    """The calibration *session* a sensor belongs to: the date, without the time.

    Grouping on this is the most diagnostic thing that can be done to one of these
    files, and it is nearly free. Faults in a calibration file are made by people
    in sittings: the part is set up once, the nominals are loaded once, and every
    sensor done that day inherits whatever was wrong with that setup. On the real
    export, all three bad sensors were calibrated on one day, within three and a
    half hours of each other, and every sensor from that day is bad while no
    sensor from any other day is.

    The date and not the timestamp, because sensors calibrated in one sitting
    carry different clock times; and the date as the file wrote it, because
    parsing it into a real date would invite a locale guess on a string that only
    has to group.
    """
    text = (claim.calibrated or "").strip()
    match = re.match(r"\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})", text)
    return match.group(1) if match else text


# -------------------------------------------------------------- fault catalogue


@dataclass
class Fault:
    """A named, correctable way a calibration transform goes wrong.

    Each one is a concrete operation on a pose, not a residual pattern to match.
    The test is therefore direct: apply the correction and see whether the
    sensor's features come inside its field of view. A sensor that only agrees
    once its Y is negated was calibrated against the mirrored feature, and saying
    that is worth far more than reporting twice its Y offset as a magnitude.
    """

    key: str
    label: str
    """What to tell the user, in words they would use about their own cell."""
    apply: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    """Takes ``(M, cfov)`` and returns the corrected pair."""
    fov_blind: bool = False
    """``True`` when the field-of-view test cannot see this fault at all.

    Not a limitation of the implementation: a turn about the sight line moves
    neither the sight line nor the field of view, so the test is blind to it by
    construction. Reporting such a fault as "not found" would be a lie of
    omission, so the report lists it as untested and points at the photographs.
    """


def _mirror(axis: int) -> np.ndarray:
    S = np.eye(3)
    S[axis, axis] = -1.0
    return S


def _flip_local(axis: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    turn = _TURN[axis](math.pi)

    def op(M: np.ndarray, cfov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # M's rows are local axes in world; turning the body is a rotation applied
        # on the local side, which for a row-wise block is a left multiply.
        return turn.T @ np.asarray(M, dtype=float), np.array(cfov, dtype=float)

    return op


def _negate_axis(axis: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def op(M: np.ndarray, cfov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.array(cfov, dtype=float)
        u[axis] = -u[axis]
        return np.array(M, dtype=float), u

    return op


def _reflect_pose(axis: int) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    S = _mirror(axis)

    def op(M: np.ndarray, cfov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # A reflection is improper, so reflecting the frame outright would hand
        # back a block of determinant -1 and no meaning as a pose. Conjugating by
        # the reflection keeps it proper, and is what "the same sensor, mounted on
        # the other side of the cell" actually means.
        return S @ np.asarray(M, dtype=float) @ S, S @ np.asarray(cfov, dtype=float)

    return op


def _translate(vec: np.ndarray) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def op(M: np.ndarray, cfov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.array(M, dtype=float), np.asarray(cfov, dtype=float) + vec

    return op


def fault_catalogue() -> list[Fault]:
    """The faults seen on this cell, in the order they are worth trying.

    Deliberately short. Every entry was reported from the shop floor rather than
    imagined, because a catalogue of speculative faults costs accuracy: each extra
    correction is another chance for a wrong one to fit a marginal sensor better
    than the truth does, which is the multiple-comparisons trap wearing a
    different coat. The report prints how many corrections were tried, for that
    reason.
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
            "mounted as the mirror image of what it was calibrated as -- the whole "
            "pose reflected through the cell centreline",
            _reflect_pose(1),
        ),
        Fault(
            "flip_depth",
            "body turned 180 degrees about its own aim axis: the wrong sweep face",
            _flip_local(0),
            fov_blind=True,
        ),
        Fault(
            "flip_y",
            "body turned 180 degrees about local Y",
            _flip_local(1),
        ),
        Fault(
            "flip_z",
            "body turned 180 degrees about local Z",
            _flip_local(2),
        ),
    ]


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


# ------------------------------------------------------------ the depth axis


@dataclass
class DepthAxis:
    """Which of a sensor's local axes is its range direction, and how we know."""

    axis: int = 0
    name: str = "X"
    unanimous: bool = False
    """Whether every sensor in the file agreed. Disagreement is not a tie to be
    broken but a sign the file mixes conventions, which is worth stopping for."""
    votes: dict[str, int] = field(default_factory=dict)
    reason: str = ""


def infer_depth_axis(claims: Sequence[SensorClaim]) -> DepthAxis:
    """Read the range axis off the shape of each sensor's own field of view.

    A line sensor's frustum is not square: its depth half-range is smaller than
    its lateral extents, because a triangulation sensor trades depth for
    resolution. So the axis carrying the smallest half-extent is the range axis,
    and each sensor votes on its own optics -- no consensus between sensors, no
    reference data, and nothing about where anything is.

    Measured on the real export this is unanimous and not close: at 600 mm
    standoff the half-extents are 117 / 158 / 161 mm and at 400 mm they are
    40 / 54 / 54 and 81 / 110 / 110. The same answer arrives independently from
    the sensor CAD, where body length along the same axis is 337 mm for the
    standoff-400 parts and 410 for the standoff-600 one -- a longer standoff is a
    longer body along the range direction.

    A sensor whose frustum is square in cross-section abstains rather than voting
    for whichever axis floating point put first.
    """
    votes: dict[int, int] = {}
    for c in claims:
        if not c.enabled:
            continue
        lim = np.maximum(
            np.abs(np.asarray(c.fov_far, dtype=float)),
            np.abs(np.asarray(c.fov_near, dtype=float)),
        )
        if not np.all(np.isfinite(lim)) or float(lim.max()) <= 1e-9:
            continue
        order = np.argsort(lim)
        # Abstain on a tie: two axes within a per cent of each other say nothing
        # about which is the range direction.
        if lim[order[0]] > 0.99 * lim[order[1]]:
            continue
        votes[int(order[0])] = votes.get(int(order[0]), 0) + 1

    if not votes:
        return DepthAxis(
            axis=0,
            name=AXIS_NAMES[0],
            unanimous=False,
            reason=(
                "no sensor's field of view is clearly deeper in one axis than the others, so "
                "the range direction could not be read off the optics; local X is assumed and "
                "every depth figure below depends on that"
            ),
        )

    best = max(votes, key=lambda k: votes[k])
    total = sum(votes.values())
    unanimous = votes[best] == total
    out = DepthAxis(
        axis=best,
        name=AXIS_NAMES[best],
        unanimous=unanimous,
        votes={AXIS_NAMES[k]: v for k, v in sorted(votes.items())},
    )
    if not unanimous:
        out.reason = (
            f"the sensors do not agree on which local axis is the range direction "
            f"({out.votes}); local {out.name} wins {votes[best]} of {total}, but a file that "
            f"mixes conventions should be split rather than averaged"
        )
    return out


# --------------------------------------------------------- field of view test


def fov_ratio(
    M: np.ndarray, cfov: np.ndarray, nominal: np.ndarray, limits: np.ndarray
) -> tuple[float, np.ndarray]:
    """How far outside its field of view a feature sits, as a multiple of the box.

    1.0 is exactly on the boundary. Returned alongside the feature's offset in the
    sensor's own local axes, because the *axis* of the overrun is what makes the
    number interpretable -- an overrun on the range axis is a standoff error and
    one across it is a position error, and they are fixed differently.
    """
    L = np.asarray(M, dtype=float) @ (
        np.asarray(nominal, dtype=float) - np.asarray(cfov, dtype=float)
    )
    return float(np.max(np.abs(L) / limits)), L


# ------------------------------------------------------------------- verdicts


CONSISTENT = "consistent"
"""Every feature this sensor measures is inside the field of view it declares.
Narrower than "correct": the box is 100-300 mm across and says nothing about the
few millimetres that matter, and a flip about the aim axis lands here too."""
EDGE = "edge"
"""A feature sits just outside the declared field of view. Marginal rather than
damning -- a slightly wrong feature nominal, a sensor always tight on that
feature, or the frustum being approximated by its widest plane."""
EXPLAINED = "explained"
"""Wrong, and one named fault accounts for it. The most useful verdict there is --
it carries a correction, not just a magnitude."""
SUSPECT = "suspect"
"""Wrong, and no catalogued fault explains it. Either a fault the catalogue does
not cover, or one worth adding once it is understood."""
UNTESTABLE = "untestable"
"""Nothing in the file could be brought to bear on this sensor. Not a pass."""


@dataclass
class SensorVerdict:
    """What can and cannot be said about one sensor's transform."""

    serial: str
    verdict: str = UNTESTABLE
    reason: str = ""
    session: str = ""
    part_number: str = ""
    worst_feature: str = ""
    ratio: float = 0.0
    """Worst feature's distance outside the field of view, as a multiple of it."""
    local: list[float] = field(default_factory=list)
    """That feature's offset from the FOV centre in the sensor's own axes."""
    limits: list[float] = field(default_factory=list)
    overrun_axis: str = ""
    """Which local axis the field of view is exceeded on. The range axis means a
    standoff or depth error; a lateral axis means the sensor is pointed in the
    wrong place."""
    correction: list[float] = field(default_factory=list)
    """The single part-frame translation that best brings every one of this
    sensor's features into its field of view.

    One translation has to serve all of them, so this is their mean offset rather
    than any one feature's -- a sensor with two features 115 mm apart cannot put
    both at the centre, and it does not need to: it needs both inside the box."""
    ratio_after: float = 0.0
    """Worst ratio once that translation is applied. Under 1.0 means a single
    rigid shift explains the whole sensor, which is the common case and the useful
    one."""
    fault: str = ""
    fault_label: str = ""
    out_of_spec: list[str] = field(default_factory=list)
    """Features whose own reported deviation breaks the file's own spec limits --
    a second channel on the same question, from measurement rather than geometry."""
    fov_blind: list[str] = field(default_factory=list)
    """Faults that could not be tested at all, listed on every verdict including a
    passing one, since that is exactly where it would be mistaken for a clean bill
    of health."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "serial": self.serial,
            "verdict": self.verdict,
            "reason": self.reason,
            "session": self.session,
            "part_number": self.part_number,
            "worst_feature": self.worst_feature,
            "ratio": self.ratio,
            "local": list(self.local),
            "limits": list(self.limits),
            "overrun_axis": self.overrun_axis,
            "correction": list(self.correction),
            "ratio_after": self.ratio_after,
            "fault": self.fault,
            "fault_label": self.fault_label,
            "out_of_spec": list(self.out_of_spec),
            "fov_blind": list(self.fov_blind),
        }


@dataclass
class SessionVerdict:
    """One calibration sitting, judged as a unit.

    The level at which these faults are actually made, and therefore the level at
    which they are actually fixed. A session every one of whose sensors is wrong
    by the same amount is one mistake, not several.
    """

    key: str
    serials: list[str] = field(default_factory=list)
    n_wrong: int = 0
    shared: list[float] = field(default_factory=list)
    """Median correction across the session's wrong sensors, per axis."""
    spread: list[float] = field(default_factory=list)
    """How much those corrections disagree, per axis. Small on an axis means the
    whole session shares one error there, which is the signature of a setup
    mistake rather than of sensors drifting independently."""
    verdict: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "serials": list(self.serials),
            "n_wrong": self.n_wrong,
            "shared": list(self.shared),
            "spread": list(self.spread),
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass
class CalibrationReport:
    """The whole file judged against itself, and what that is and is not worth."""

    ok: bool = False
    reason: str = ""
    units: str = "mm"
    depth_axis: str = ""
    depth_unanimous: bool = False
    n_sensors: int = 0
    n_features: int = 0
    n_consistent: int = 0
    n_edge: int = 0
    n_explained: int = 0
    n_suspect: int = 0
    faults_tried: int = 0
    span: float = 0.0
    """Largest distance between any two FOV centres, in file units.

    The units check: a measurement cell is metres across, so a span of 3 says the
    file is in metres while the project is in millimetres, and 3000000 says
    something worse. ``part.py`` exists to catch the same mistake and for the same
    reason -- a factor of 1000 applied quietly leaves every deviation in the
    report reading as a clean zero."""
    fov_span: float = 0.0
    """Typical field-of-view size, for comparison with ``span``. When the centres
    are spread much wider than one field of view, the sensors are not all looking
    at the same thing and :func:`aim_consensus` does not apply."""
    sessions: list[SessionVerdict] = field(default_factory=list)
    verdicts: list[SensorVerdict] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def grade(self, good: float, warn: float) -> str:
        """Overall traffic light, on the pattern of ``OriginUncertainty.grade``.

        Graded on whether the file contradicts itself, and only then on how far
        out the contradictions are. ``good`` and ``warn`` are translation
        tolerances in the file's units.

        The correction is read only off sensors that actually failed, which an
        earlier version got wrong. For a sensor whose features all sit inside its
        field of view, ``correction`` is just the centroid offset of those
        features -- a sensor watching two features 100 mm apart reports a 50 mm
        correction and is perfectly healthy, since one field of view cannot centre
        on both. Grading on that marked a clean file amber for having more than
        one feature per sensor.
        """
        if not self.ok or not self.verdicts:
            return "unknown"
        # A file nothing could be brought to bear on must not come back green.
        # Every sensor untestable means every sensor unexamined, and "good" would
        # be a verdict on work that was never done.
        if all(v.verdict == UNTESTABLE for v in self.verdicts):
            return "unknown"
        wrong = [v for v in self.verdicts if v.verdict in (EXPLAINED, SUSPECT)]
        if wrong:
            worst = max(float(np.linalg.norm(v.correction or [0.0])) for v in wrong)
            return "bad" if worst > warn else "warn"
        return "warn" if self.n_edge else "good"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "units": self.units,
            "depth_axis": self.depth_axis,
            "depth_unanimous": self.depth_unanimous,
            "n_sensors": self.n_sensors,
            "n_features": self.n_features,
            "n_consistent": self.n_consistent,
            "n_edge": self.n_edge,
            "n_explained": self.n_explained,
            "n_suspect": self.n_suspect,
            "faults_tried": self.faults_tried,
            "span": self.span,
            "fov_span": self.fov_span,
            "sessions": [s.to_dict() for s in self.sessions],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "messages": list(self.messages),
        }


def _span(points: Sequence[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    P = np.array(points, dtype=float)
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))


def diagnose(
    claims: Sequence[SensorClaim],
    *,
    depth_axis: int | None = None,
    tol: float = 10.0,
    warn: float = 50.0,
    units: str = "mm",
) -> CalibrationReport:
    """Judge every sensor against the field of view it declares for itself.

    ``tol`` and ``warn`` are translation tolerances in the file's own units, used
    to grade the corrections; both belong to whoever is inspecting the cell rather
    than to this module. They do not set the conviction threshold -- that comes
    from each sensor's own optics, which is the point.
    """
    live = [c for c in claims if c.enabled]
    report = CalibrationReport(
        n_sensors=len(live),
        n_features=sum(len(c.features) for c in live),
        units=units,
        faults_tried=len(fault_catalogue()),
        span=_span([c.cfov for c in live]),
    )
    if not live:
        report.reason = "no sensors in the file"
        return report

    report.fov_span = float(
        np.median([float(np.max(c.fov_limits(0))) * 2.0 for c in live])
    )

    axis = infer_depth_axis(live)
    if depth_axis is not None:
        axis = DepthAxis(
            axis=int(depth_axis),
            name=AXIS_NAMES[int(depth_axis)],
            unanimous=True,
            reason="range axis supplied rather than inferred",
        )
    report.depth_axis = axis.name
    report.depth_unanimous = axis.unanimous
    if axis.reason:
        report.messages.append(axis.reason)

    catalogue = fault_catalogue()
    blind = [f.label for f in catalogue if f.fov_blind]
    report.ok = True

    for c in live:
        report.verdicts.append(_judge(c, axis.axis, catalogue, blind, tol, warn))

    report.n_consistent = sum(1 for v in report.verdicts if v.verdict == CONSISTENT)
    report.n_edge = sum(1 for v in report.verdicts if v.verdict == EDGE)
    report.n_explained = sum(1 for v in report.verdicts if v.verdict == EXPLAINED)
    report.n_suspect = sum(1 for v in report.verdicts if v.verdict == SUSPECT)
    report.sessions = _judge_sessions(live, report.verdicts, tol)

    if blind:
        report.messages.append(
            "untested on every sensor, including the ones that passed: "
            + "; ".join(blind)
            + ". It moves neither the sight line nor the field of view, so nothing in the "
            "file can see it -- the photographs are the only referee, and only a close one, "
            "since these bodies keep their outline under that turn"
        )
    report.messages.append(
        f"a field of view about {report.fov_span:.0f} {units} across is what every verdict "
        f"here is measured against, so 'consistent' means 'not grossly wrong' and says "
        f"nothing about the few {units} that an inspection actually turns on"
    )
    if report.span > 3.0 * max(report.fov_span, 1e-9):
        report.messages.append(
            f"the sensors' fields of view are spread {report.span:.0f} {units} apart against "
            f"a {report.fov_span:.0f} {units} field of view, so they are not all looking at "
            f"one working volume and the aim-ray consensus does not apply to this file"
        )
    return report


def _judge(
    claim: SensorClaim,
    depth_axis: int,
    catalogue: Sequence[Fault],
    blind: Sequence[str],
    tol: float,
    warn: float,
) -> SensorVerdict:
    """One sensor against its own declared optics."""
    v = SensorVerdict(
        serial=claim.serial,
        session=session_key(claim),
        part_number=claim.part_number,
        fov_blind=list(blind),
    )
    if not claim.features:
        v.reason = (
            "the file lists no feature for this sensor, so there is no claim for its own "
            "field of view to contradict. Nothing is asserted about it either way"
        )
        return v

    limits = claim.fov_limits(depth_axis)
    M = np.asarray(claim.M, dtype=float)
    cfov = np.asarray(claim.cfov, dtype=float)

    scored = [
        (*fov_ratio(M, cfov, f.nominal, limits), f) for f in claim.features
    ]
    ratio, local, worst = max(scored, key=lambda s: s[0])
    v.ratio = ratio
    v.local = [float(x) for x in local]
    v.limits = [float(x) for x in limits]
    v.worst_feature = worst.name
    v.overrun_axis = AXIS_NAMES[int(np.argmax(np.abs(local) / limits))]

    offsets = [np.asarray(f.nominal, dtype=float) - cfov for f in claim.features]
    shift = np.mean(offsets, axis=0)
    v.correction = [float(x) for x in shift]
    v.ratio_after = max(
        fov_ratio(M, cfov + shift, f.nominal, limits)[0] for f in claim.features
    )
    v.out_of_spec = [f.name for f in claim.features if _out_of_spec(f)]

    depth_note = (
        "on the range axis, so it is a standoff or depth error"
        if v.overrun_axis == AXIS_NAMES[depth_axis]
        else "across the sight line, so the sensor is pointed somewhere other than at the "
        "feature"
    )

    if ratio <= 1.0:
        v.verdict = CONSISTENT
        v.reason = (
            f"every feature it measures sits inside the field of view it declares -- the "
            f"worst, {worst.name}, at {ratio:.2f} of the box. That rules out a gross fault "
            f"and nothing finer: the box is about {2 * float(np.max(limits)):.0f} across, so "
            f"this is not a statement about millimetres"
        )
        if v.out_of_spec:
            v.reason += (
                f". Note that {', '.join(v.out_of_spec)} still break the file's own spec "
                f"limits, which geometry cannot see"
            )
        return v

    if ratio <= EDGE_RATIO:
        v.verdict = EDGE
        v.reason = (
            f"{worst.name} sits just outside the declared field of view, at {ratio:.2f} of "
            f"the box {depth_note}. That is marginal rather than damning -- a feature nominal "
            f"slightly out, a sensor that was always tight on this feature, or the frustum "
            f"being approximated here by its widest plane. Worth a look, not a correction"
        )
        return v

    # A correction has to bring the features inside the box, not merely closer. A
    # fault that halves a metre has explained nothing, and attaching a confident
    # diagnosis to a sensor nobody has accounted for is worse than saying so.
    best: tuple[float, Fault] | None = None
    for fault in catalogue:
        if fault.fov_blind:
            continue
        Mc, cc = fault.apply(M, cfov)
        got = max(
            fov_ratio(Mc, cc, f.nominal, claim.fov_limits(depth_axis))[0]
            for f in claim.features
        )
        if got <= 1.0 and (best is None or got < best[0]):
            best = (got, fault)

    mag = float(np.linalg.norm(shift))
    if best is not None:
        v.verdict = EXPLAINED
        v.fault = best[1].key
        v.fault_label = best[1].label
        v.reason = (
            f"{worst.name} lands {ratio:.1f}x outside the field of view, {depth_note}, and one "
            f"catalogued fault accounts for it: {best[1].label}. Correcting it brings every "
            f"feature inside the box ({best[0]:.2f} of it)"
        )
        return v

    v.verdict = SUSPECT
    if v.ratio_after <= 1.0:
        shape = (
            f"A single translation of {mag:.1f} does bring every feature inside the box "
            f"({v.ratio_after:.2f} of it), so the transform is displaced rather than "
            f"misoriented -- which is what a calibration run against the wrong nominal looks "
            f"like"
        )
    else:
        shape = (
            f"No single translation fixes it either: the best one still leaves "
            f"{v.ratio_after:.2f} of the box, so the orientation is wrong too and this sensor "
            f"needs recalibrating rather than correcting"
        )
    v.reason = (
        f"{worst.name} lands {ratio:.1f}x outside the field of view this sensor declares, "
        f"{depth_note}. No catalogued fault accounts for it. {shape}"
        + (
            f". The deviation it reports on {', '.join(v.out_of_spec)} breaks the file's own "
            f"spec limits too, which is the same conclusion from measurement rather than "
            f"geometry"
            if v.out_of_spec
            else ""
        )
        + (f". Past the {warn:.0f} that counts as badly wrong" if mag > warn else "")
    )
    return v


def _out_of_spec(feature: FeatureClaim) -> bool:
    for off, (usl, lsl) in zip(feature.offsets, feature.limits):
        if off is None:
            continue
        if usl is not None and off > usl:
            return True
        if lsl is not None and off < lsl:
            return True
    return False


def _judge_sessions(
    claims: Sequence[SensorClaim], verdicts: Sequence[SensorVerdict], tol: float
) -> list[SessionVerdict]:
    """Group the verdicts by calibration sitting and look for a shared mistake.

    The test that turns a list of bad sensors into a cause: if every sensor from
    one session is wrong, and their corrections agree on some axis, the session's
    setup was wrong on that axis and one fix covers all of them.
    """
    by_key: dict[str, list[SensorVerdict]] = {}
    for v in verdicts:
        by_key.setdefault(v.session, []).append(v)

    out: list[SessionVerdict] = []
    for key, group in sorted(by_key.items()):
        wrong = [v for v in group if v.verdict in (EXPLAINED, SUSPECT)]
        s = SessionVerdict(
            key=key,
            serials=sorted(v.serial for v in group),
            n_wrong=len(wrong),
        )
        if not wrong:
            s.verdict = "clean"
            s.reason = f"all {len(group)} sensor(s) calibrated in this sitting are consistent"
            out.append(s)
            continue

        C = np.array([v.correction for v in wrong], dtype=float)
        s.shared = [float(x) for x in np.median(C, axis=0)]
        # Spread per axis rather than overall: a session mistake is usually along
        # one axis, and an overall figure would hide agreement on that axis behind
        # disagreement on the others.
        s.spread = [
            float(np.max(np.abs(C[:, i] - np.median(C[:, i])))) for i in range(3)
        ]
        agreed = [
            AXIS_NAMES[i]
            for i in range(3)
            if abs(s.shared[i]) > max(tol, 1e-9) and s.spread[i] <= max(0.25 * abs(s.shared[i]), tol)
        ]
        if len(wrong) == len(group) and len(wrong) > 1 and agreed:
            s.verdict = "session"
            axes = ", ".join(f"{AXIS_NAMES[i]} {s.shared[i]:+.0f}" for i in range(3) if AXIS_NAMES[i] in agreed)
            s.reason = (
                f"every one of the {len(group)} sensors calibrated in this sitting is wrong, "
                f"and they are wrong by the same amount ({axes}, agreeing to "
                f"{max(s.spread[i] for i in range(3) if AXIS_NAMES[i] in agreed):.0f}). That "
                f"is one mistake in the setup, not {len(wrong)} faulty sensors -- the part was "
                f"positioned wrongly or the wrong nominals were loaded, and recalibrating the "
                f"session fixes all of them. Which of the two it was cannot be told from the "
                f"file: a shared offset looks identical whether the sensors moved or the "
                f"nominals did"
            )
        elif len(wrong) == len(group) and len(wrong) > 1:
            s.verdict = "session"
            s.reason = (
                f"every one of the {len(group)} sensors from this sitting is wrong, but by "
                f"different amounts (spread {s.spread[0]:.0f}, {s.spread[1]:.0f}, "
                f"{s.spread[2]:.0f}), so a single setup error does not explain it and they "
                f"are worth treating separately"
            )
        else:
            s.verdict = "mixed"
            s.reason = (
                f"{len(wrong)} of {len(group)} sensors from this sitting are wrong, so the "
                f"sitting itself is not the explanation"
            )
        out.append(s)
    return out


# --------------------------------------- the aim-ray consensus, where it applies


@dataclass
class AimConsensus:
    """Where the sensors agree they are all looking, and who disagrees.

    Only meaningful on a cell where every sensor really is pointed at one working
    volume. :func:`diagnose` says when that is not the case, and this refuses
    rather than intersecting a pencil that is not one.
    """

    ok: bool = False
    reason: str = ""
    point: list[float] = field(default_factory=list)
    scale: float = 0.0
    threshold: float = 0.0
    inliers: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    distances: dict[str, float] = field(default_factory=dict)
    n_front: int = 0


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
    """Perpendicular distance from ``point`` to each ray, ignoring direction of travel."""
    rel = positions - point
    along = np.einsum("ij,ij->i", rel, dirs)
    perp = rel - along[:, None] * dirs
    return np.linalg.norm(perp, axis=1)


def aim_consensus(
    claims: Sequence[SensorClaim],
    *,
    depth_axis: int = 0,
    sign: float = 1.0,
    tol: float = 50.0,
    k: float = 3.0,
) -> AimConsensus:
    """Find the working volume the sensors agree on, robustly, and name the strays.

    For the cells where every sensor is pointed at one volume. The minimal set is
    two, because one ray determines no point and two determine their closest
    approach, so every pair is a hypothesis and the pair count is small enough to
    enumerate outright. Each hypothesis is scored by how many of the other sensors
    it puts within tolerance -- the only thing that separates "this sensor is
    wrong" from "everything else is wrong", and the reason this refuses below
    :data:`MIN_CONSENSUS_SENSORS`.

    Note the blind spot that the field-of-view test does not share: this is a test
    on a *line*, so a translation along a sensor's own sight line does not move
    its ray and cannot be seen here at all.
    """
    live = [c for c in claims if c.enabled]
    if len(live) < MIN_CONSENSUS_SENSORS:
        return AimConsensus(
            reason=(
                f"{len(live)} sensor(s) is too few to tell a bad transform from a bad "
                f"consensus: {MIN_CONSENSUS_SENSORS} are needed before one can be the odd one "
                f"out and be seen to be"
            )
        )

    spread = _span([c.cfov for c in live])
    fov = float(np.median([float(np.max(c.fov_limits(depth_axis))) * 2.0 for c in live]))
    if spread > 3.0 * max(fov, 1e-9):
        return AimConsensus(
            reason=(
                f"these sensors are not all looking at one working volume: their fields of "
                f"view are spread {spread:.0f} apart against a {fov:.0f} field of view. A "
                f"consensus among them would be an intersection of a pencil that is not one, "
                f"so the field-of-view test is the one that applies to this file"
            )
        )

    positions = np.array([c.position(depth_axis, sign=sign) for c in live])
    dirs = np.array([c.aim(depth_axis) * sign for c in live])
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
        return AimConsensus(reason="no pair of rays gathered a consensus")

    # Refine on the inliers only. Refitting on everything would drag the volume
    # towards the very sensors being judged -- outliers helping to produce the
    # residuals they are then screened by, which is the whole trap.
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
    inner = d[d <= max(3.0 * robust_scale(d, tol), tol)]
    scale = robust_scale(inner, tol)
    threshold = max(k * scale, tol)
    keep = d <= threshold
    rel = positions - point
    along = np.einsum("ij,ij->i", rel, dirs)

    return AimConsensus(
        ok=True,
        point=[float(x) for x in point],
        scale=float(scale),
        threshold=float(threshold),
        inliers=[c.serial for c, f in zip(live, keep) if f],
        outliers=[c.serial for c, f in zip(live, keep) if not f],
        distances={c.serial: float(x) for c, x in zip(live, d)},
        n_front=int((along < 0).sum()),
    )
