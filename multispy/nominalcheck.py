"""Do the nominal'd points really form the shape their coordinates describe?

A nominal says "this point is at X=5151, Y=-527.1, Z=520". Three of them put
the solve in part space. Nothing in that machinery asks whether the point
someone clicked is *the feature those numbers belong to*, and on hardware that
repeats -- a frame that is near-symmetric left to right, identical clamps down
a tube, three sensor bodies across fifteen stations -- picking the wrong one
from the drawing's own list is the easiest mistake there is to make and the
hardest to see afterwards.

It is hard to see because the usual readouts go quiet. ``align_to_nominals``
will happily fit an origin and a scale to a mislabelled set; the
asked-versus-achieved table then shows each nominal satisfied, because the
solve bent itself to satisfy them; and a point with no stated tolerance falls
back to ``Settings.sigma_nominal``, which is tight enough to act as a rigid
anchor, so the one point that is wrong is also the one reported as perfect
while the correctly toleranced points absorb the error. Measured on a
nine-photograph cell: two mislabelled points pinned at the 0.1 mm default held
their own deviation at 0.0 mm and pushed the two honest nominals 97 mm and
210 mm off, took the reprojection RMS from 2.4 px to 20 px, drove the shared
focal to 2.37x its EXIF value, and stopped the bundle adjustment converging at
all. Every one of those numbers says "something is wrong" and none of them says
what.

So this asks a question the fit cannot answer for itself: forget the frame and
forget the units, and compare *shapes*. Distances between points are unchanged
by any rotation or translation, and dividing one by another cancels the scale,
so a correctly linked set has one scale and every pair agrees on it. A swapped
pair does not -- the two features the drawing puts closest together stop being
the two the photographs put closest together, and no rigid motion repairs it.

**The trap this exists to guard, and the reason it re-solves.** Run the
comparison against a solve that the nominals are driving and it passes
trivially: the nominals *are* what put those points where they are, so they
agree with themselves by construction, the mislabelled ones most tightly of
all. So the project is copied, every nominal is stripped out of the copy, and
it is re-solved from the photographs alone before anything is compared. Scale
goes arbitrary when the nominals go, which costs nothing here -- the whole
comparison was built to be scale-free -- and it means this works on a project
that has never reached part space, which is exactly when it is most wanted.

**Below three points it cannot fire, and says so.** Two points give one
distance and one unknown scale, so the single ratio is always satisfiable and
"consistent" would be a word with no content behind it. Three give three
ratios against one scale, which is the first case with anything left over to
contradict. A set that is nearly collinear is reported as such rather than
graded, because a line has no shape to speak of across it.

What it does not do: it cannot tell which point is wrong, only that the set
disagrees. With one bad link in a set of four the residuals spread over all
four, since the fit splits the error among them. It also cannot see a nominal
that is wrong in a way that preserves shape -- the whole set shifted, or taken
from the solve it is then used to anchor -- which is ``uncertainty.py``'s
warning restated: a quiet verdict here is a floor on your confidence, not a
ceiling.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .geometry import similarity_fit
from .project import Project

MIN_POINTS = 3
"""Fewest nominal'd points that leave anything to contradict.

Two points and a free scale fit each other exactly whatever they are, so a
check run on two would return the same answer for correct and swapped links
alike. Guarded explicitly rather than left to produce a flattering number.
"""

COLLINEAR_LIMIT = 0.05
"""Below this the nominal'd points are treated as a line, not a shape.

The second singular value of the centred set over the first. A set strung out
along one direction pins scale and that direction and says nothing about the
two across it, so a swap between two points near the line is invisible. 0.05
is about where the across-the-line signal drops under the click noise these
solves carry.
"""

DEFAULT_TOL_SIGMAS = 3.0
DEFAULT_WARN_SIGMAS = 5.0
"""How far off a point may sit, in multiples of its own stated tolerance.

In sigmas rather than millimetres because the tolerances in one calibration
file span 0.5 mm to 8 mm, and a fixed millimetre threshold would either excuse
the tight features or condemn the loose ones.
"""


@dataclass
class PointShape:
    """One nominal'd point, and how far it sits from where its shape puts it."""

    id: str = ""
    label: str = ""
    source: str = ""
    """``Point.nominal_source`` -- the feature this nominal was picked from, or
    empty for one typed by hand. A typed nominal that matches no feature in the
    file is worth reading twice; it is the one kind this cannot cross-check."""
    nominal: list[float] = field(default_factory=list)
    sigma: float = 0.0
    """The tolerance in force, the default already substituted, so the report
    shows what was actually used rather than a blank."""
    sigma_defaulted: bool = False
    """True when the file stated no tolerance and ``Settings.sigma_nominal``
    was used. These are the points that act as rigid anchors in a solve, so a
    large residual on one of them matters more than its size suggests."""
    residual: list[float] = field(default_factory=list)
    """Where the shape fit puts it, minus where the nominal says, nominal units."""
    deviation: float = 0.0
    sigmas: float = 0.0
    """``deviation`` over ``sigma``. What the grade is read from."""
    positioned: bool = True
    """False when the nominal-free solve could not place this point at all --
    usually a point marked in only one photograph once its bad marks are gone.
    It is excluded from the comparison and named, never silently dropped."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairRatio:
    """One pair of points: how far apart the photographs put them, against the drawing."""

    a: str = ""
    b: str = ""
    solved: float = 0.0
    """In the arbitrary units of the nominal-free solve. Only its ratios mean
    anything, which is the point."""
    nominal: float = 0.0
    ratio: float = 0.0
    off_median_pct: float = 0.0
    """How far this pair's ratio sits from the set's median, as a percentage.
    This is the fit-free reading: it owes nothing to a rotation or an origin,
    so it cannot be flattered by a fit absorbing the error."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShapeReport:
    """Whether the nominal'd points and their nominals are the same shape."""

    ok: bool = False
    """The comparison ran. False means it could not -- too few points, a
    degenerate set, or a solve that failed -- and ``reason`` says which."""
    reason: str = ""
    verdict: str = "untestable"
    """'consistent' / 'suspect' / 'contradicted' / 'untestable'.

    'untestable' is a real answer and not a failure: it means this project
    cannot support the question yet, which is different from passing it."""
    n_points: int = 0
    n_compared: int = 0
    spread: float = 1.0
    """Largest pair ratio over smallest. 1.0 is a set that agrees with itself.

    The honest headline, because it survives without fitting anything. On the
    cell this module was written for, three correctly marked points whose links
    had been crossed came out at 3.6."""
    scale: float = 0.0
    """Median solved-over-nominal ratio: what one nominal unit is worth in the
    nominal-free solve. Reported so the deviations below can be read in
    millimetres rather than in solver units."""
    rms: float = 0.0
    """Similarity-fit residual over the compared points, in nominal units."""
    worst_sigmas: float = 0.0
    collinearity: float = 0.0
    """Second singular value over the first for the nominal'd set. See
    ``COLLINEAR_LIMIT``."""
    reproj_px: float = 0.0
    """Reprojection RMS of the nominal-free re-solve. Read it first: if the
    photographs do not agree with each other, a disagreement with the drawing
    says nothing yet about the links."""
    line_px: float = 0.0
    units: str = "mm"
    points: list[PointShape] = field(default_factory=list)
    pairs: list[PairRatio] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verdict(worst_sigmas: float, tol_sigmas: float, warn_sigmas: float) -> str:
    if worst_sigmas <= tol_sigmas:
        return "consistent"
    if worst_sigmas <= warn_sigmas:
        return "suspect"
    return "contradicted"


def check_nominal_shape(
    project: Project,
    *,
    tol_sigmas: float = DEFAULT_TOL_SIGMAS,
    warn_sigmas: float = DEFAULT_WARN_SIGMAS,
) -> ShapeReport:
    """Compare the shape of the nominal'd points with the shape of their nominals.

    ``project`` is not modified: the re-solve happens on a copy with its path
    cleared, so it can never be written back over the project it came from.
    """
    from .solve import solve  # local, as in uncertainty: solve reaches back into geometry

    units = project.nominal_units or "mm"
    scale_mm = project.nominal_scale or 0.001

    linked = [p for p in project.points if p.nominal and len(p.nominal) == 3
              and all(v is not None for v in p.nominal)]
    report = ShapeReport(n_points=len(linked), units=units)

    if len(linked) < MIN_POINTS:
        report.reason = (
            f"{len(linked)} point(s) carry a nominal; {MIN_POINTS} are needed before "
            f"the comparison has anything left over to contradict. Two points and a free "
            f"scale agree with each other whatever they are."
        )
        return report

    # Strip the nominals and re-solve, so what comes back owes them nothing.
    trial = copy.deepcopy(project)
    trial.path = None
    for pt in trial.points:
        pt.nominal = None
        pt.nominal_sigma = None
    solved = solve(trial)
    report.reproj_px = float(solved.rms_reprojection_px)
    report.line_px = float(solved.rms_line_px)
    if not solved.ok:
        report.reason = (
            "the photographs alone do not solve, so there is nothing to compare the "
            "drawing against yet"
        )
        return report

    got = {p.id: np.array(p.xyz, dtype=float)
           for p in trial.points if p.xyz is not None}

    default_sigma = float(project.settings.sigma_nominal)
    rows: list[PointShape] = []
    for p in linked:
        defaulted = p.nominal_sigma is None
        rows.append(PointShape(
            id=p.id,
            label=p.label or "",
            source=p.nominal_source or "",
            nominal=[float(v) for v in p.nominal],
            sigma=default_sigma if defaulted else float(p.nominal_sigma),
            sigma_defaulted=defaulted,
            positioned=p.id in got,
        ))

    usable = [r for r in rows if r.positioned]
    lost = [r for r in rows if not r.positioned]
    if lost:
        report.warnings.append(
            f"{len(lost)} nominal'd point(s) could not be placed from the photographs "
            f"alone ({', '.join(r.id for r in lost)}) -- marked in too few photographs "
            f"to triangulate. They are left out of the comparison, so the verdict is "
            f"about the rest."
        )
    if len(usable) < MIN_POINTS:
        report.points = rows
        report.reason = (
            f"only {len(usable)} nominal'd point(s) could be placed from the photographs, "
            f"below the {MIN_POINTS} the comparison needs"
        )
        return report

    A = np.array([got[r.id] for r in usable])
    B = np.array([r.nominal for r in usable])

    centred = B - B.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    report.collinearity = float(sv[1] / sv[0]) if sv[0] > 1e-12 else 0.0
    if report.collinearity < COLLINEAR_LIMIT:
        report.points = rows
        report.n_compared = len(usable)
        report.reason = (
            f"the nominal'd points lie too nearly along one line "
            f"(collinearity {report.collinearity:.3f} < {COLLINEAR_LIMIT}) for their shape "
            f"to be worth testing -- a swap between two of them would not show"
        )
        return report

    pairs: list[PairRatio] = []
    for ra, rb in itertools.combinations(usable, 2):
        ds = float(np.linalg.norm(got[ra.id] - got[rb.id]))
        dn = float(np.linalg.norm(np.array(ra.nominal) - np.array(rb.nominal)))
        if dn <= 1e-9:
            report.warnings.append(
                f"{ra.id} and {rb.id} carry the same nominal coordinate, so one of the "
                f"two links is certainly wrong"
            )
            continue
        pairs.append(PairRatio(a=ra.id, b=rb.id, solved=ds, nominal=dn, ratio=ds / dn))
    if not pairs:
        report.points = rows
        report.reason = "no pair of nominal'd points is separated in the drawing"
        return report

    ratios = np.array([p.ratio for p in pairs])
    median = float(np.median(ratios))
    for pr in pairs:
        pr.off_median_pct = float((pr.ratio / median - 1.0) * 100.0) if median else 0.0
    report.scale = median
    report.spread = float(ratios.max() / ratios.min()) if ratios.min() > 1e-18 else float("inf")

    fit = similarity_fit(A, B, allow_scale=True)
    placed = fit.scale * (A @ fit.R.T) + fit.t
    for r, got_xyz in zip(usable, placed):
        d = got_xyz - np.array(r.nominal)
        r.residual = [float(v) for v in d]
        r.deviation = float(np.linalg.norm(d))
        r.sigmas = r.deviation / r.sigma if r.sigma > 1e-12 else float("inf")

    report.rms = float(np.sqrt(np.mean([r.deviation ** 2 for r in usable])))
    report.worst_sigmas = max(r.sigmas for r in usable)
    report.points = rows
    report.pairs = pairs
    report.n_compared = len(usable)
    report.ok = True
    report.verdict = _verdict(report.worst_sigmas, tol_sigmas, warn_sigmas)

    worst = max(usable, key=lambda r: r.sigmas)
    report.messages.append(
        f"compared {len(usable)} nominal'd point(s) over {len(pairs)} pair(s) against a "
        f"solve with every nominal stripped out, so the agreement is the photographs' "
        f"own and not the nominals agreeing with themselves."
    )
    report.messages.append(
        f"pair ratios spread {report.spread:.2f}x around a scale of {median:.6g} solve "
        f"units per {units}; shape fit leaves {report.rms:.1f} {units} RMS, worst "
        f"{worst.id} at {worst.deviation:.1f} {units} = {worst.sigmas:.1f}x its "
        f"{worst.sigma:g} {units} tolerance."
    )
    if report.verdict != "consistent":
        report.messages.append(
            "a set this far out of shape is not a marking-precision problem: no rotation, "
            "translation or scale takes one of these shapes to the other, so at least one "
            "point is linked to a feature that is not the one under it. Check the pairs "
            "the drawing puts closest together against the pairs the photographs do, and "
            "the left/right sides first."
        )
    if len(usable) == MIN_POINTS:
        report.warnings.append(
            "exactly three points compared, which is the fewest that can disagree at all. "
            "A fourth is what turns a disagreement into a direction to look in."
        )
    defaulted = [r for r in usable if r.sigma_defaulted]
    if defaulted:
        report.warnings.append(
            f"{len(defaulted)} compared point(s) have no stated tolerance "
            f"({', '.join(r.id for r in defaulted)}) and are being judged against the "
            f"{default_sigma:g} {units} default. That default also makes them rigid "
            f"anchors in an ordinary solve, so a wrong link on one of these does more "
            f"damage than a wrong link anywhere else."
        )
    return report
