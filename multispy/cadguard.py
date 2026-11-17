"""Did adding the CAD bodies make the photographs fit worse?

The failure this exists for is specific and it has happened repeatedly on this
project: the solve is good -- a couple of pixels, nothing culled -- then a model
is dropped in, its anchors are marked, Solve is pressed, and the calibration is
worse than it was. Nothing in the report says so, because the report describes
the solve it just did and has nothing to compare it against. The body is not
obviously wrong, the numbers are not obviously bad, and the good state is gone.

A known object is a much stronger claim than a tie point. A tie point says "this
feature is somewhere"; a body says "these N features are in exactly this rigid
arrangement, to the micron, because a CAD file says so". When the arrangement is
right that is worth more than any number of tie points. When it is wrong -- an
anchor on the wrong corner, a model at the wrong scale, a pose fitted to a
mirror-image reading of a symmetric part -- it is a lie told with enormous
confidence, and the bundle will bend the cameras to accommodate it. The focal
length is the usual casualty: a single mis-marked body has dragged a shared focal
from 2796 px to under 2000, which is why :mod:`solve` grew a focal wall. The wall
stops the collapse. It does not notice the damage below it.

So the rule this module enforces, in the user's words: the CAD solution should
only stand if it does not worsen the point solution **in each image**. Per image,
not overall, because an average lets a body ruin the two photographs it appears
in while twelve others carry the mean.

Measuring it honestly
---------------------
``ImageReport.rms_px`` looks like the number to compare and it is the wrong one,
though not for the reason first supposed. It is not a tie-point figure:
``_fill_statistics.record`` folds anchor marks into it deliberately, and says so
-- "Anchor marks go through here on equal terms with tie points: they are clicks
in that photo, so they belong in that photo's RMS." That is right for the panel
it feeds, and it answers a different question from this one. A body-free solve
has no anchor residuals at all, so the two ``rms_px`` figures are not even
measuring the same population.

The first version of this docstring claimed the mixed reading would be *less*
sensitive, on the grounds that a body's anchors are fitted tightly and would drag
the average down. Measured on the rig fixture, that is false for the cases that
matter most: a mis-marked body's own anchors fit *badly*, so they push the mixed
average up and make the damage look larger, not smaller -- on one spoiled
photograph the mixed reading moved +0.186 px where the tie-point-only reading
moved +0.071. The dilution argument holds only for a body whose anchors fit well
while sitting in the wrong place, which is a real failure but not one measured
here, so it is not claimed.

The reason to exclude anchors is therefore about meaning rather than sensitivity,
and it is the stronger reason. The mixed figure conflates "this body's own marks
are bad" with "this body damaged the point solution". Those call for opposite
responses -- re-click the anchor, versus stop letting the body move the cameras
-- and the user asked for the second. A guard that fires because a body's own
anchors are untidy, in a photograph whose tie points are untouched, is reporting
a fact about the body and calling it a fact about the calibration.

So the comparison is rebuilt from ``SolveReport.residual_lines``, filtered to
labels that are tie point ids, and restricted to the marks that survived the cull
in *both* solves. That last restriction matters for the same reason: if the
body-laden solve culls a mark the body-free one kept, the two RMS figures are
taken over different populations, and dropping the worst mark is an easy way to
look better while being worse.

What this does and does not do
------------------------------
It measures, and it can put the body-free cameras back. It does not place the
bodies against locked cameras -- that would need a solve mode that does not exist
yet, since ``fit_instances`` is part of the linear initialisation and works from
triangulated anchor positions rather than from a finished camera solution. So a
rejected body is reported as rejected, and what to do about it is the user's call:
re-click its anchors, check its units, or leave it out.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from .project import Project

TOLERANCE_PX = 0.25
"""How much worse one image's tie-point RMS may get before the bodies are judged
to have harmed it, as an absolute floor.

A judgement, not a measurement. Two solves of the same data do not land on
identical numbers -- the parameter vector is larger with bodies in it, the
starting point differs, and least squares is not deterministic to the last digit
-- so some allowance is needed or the guard fires on arithmetic. A quarter of a
pixel is well below anything a person could re-click and well above solver
jitter, which on this project runs in the thousandths."""

TOLERANCE_FRACTION = 0.10
"""And a relative allowance, for images that start out poorly.

An image already sitting at 8 px has nothing to gain from being held to a quarter
of a pixel; the absolute floor would make every body look harmful there. The
allowance is the larger of the two, so a sharp image is judged tightly and a
soft one proportionally."""

IMPROVEMENT_PX = 0.05
"""Below this, an overall change is called neither better nor worse.

Reported rather than hidden, because "the bodies changed nothing" is a useful
answer: it means the anchors are consistent with the photographs and are also not
contributing much, which is worth knowing before marking twenty more."""


@dataclass
class ImageVerdict:
    """One photograph's tie-point fit, with the bodies and without them."""

    image: str
    n_points: int = 0
    """Tie-point marks compared. Only those that survived the cull in *both*
    solves, so the two figures describe the same population."""
    before_px: float = 0.0
    """Tie-point-only RMS with no body anchors in the solve at all."""
    after_px: float = 0.0
    """Tie-point-only RMS with them."""
    delta_px: float = 0.0
    """``after - before``. Positive is worse."""
    allowed_px: float = 0.0
    """The allowance in force for this image -- see :data:`TOLERANCE_PX` and
    :data:`TOLERANCE_FRACTION`."""
    worsened: bool = False
    worst_point: str = ""
    """The tie point whose own error grew most in this image. Where to look
    first, and usually an anchor clicked on the wrong feature nearby."""
    worst_point_delta_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CadReport:
    """Whether the known objects earned their place in the solve."""

    ok: bool = False
    reason: str = ""
    verdict: str = ""
    """``"improves"``, ``"harmless"``, ``"worsens"`` or ``"untestable"``."""
    n_instances: int = 0
    n_anchor_marks: int = 0
    before_rms_px: float = 0.0
    """Overall tie-point-only RMS without bodies. Read the per-image table
    rather than this: the whole point is that an average hides the damage."""
    after_rms_px: float = 0.0
    focal_before_px: float | None = None
    focal_after_px: float | None = None
    """The shared focal either side. Carried because it is the quantity a bad
    body attacks first, and a focal that moved several percent is a stronger
    signal than any single image's RMS."""
    images: list[ImageVerdict] = field(default_factory=list)
    """Worst first."""
    worsened: list[str] = field(default_factory=list)
    """Images whose tie-point fit the bodies made worse past the allowance."""
    worst_image: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["images"] = [v.to_dict() for v in self.images]
        return d


def _tie_ids(project: Project) -> set[str]:
    """Tie point ids, which is also what tells a point residual from an anchor's.

    Taken from the project rather than by looking for the anchor separator in the
    label, because a point id is whatever the user typed and could contain one.
    """
    return {p.id for p in project.points}


def _per_image(report: Any, tie_ids: set[str]) -> dict[str, dict[str, float]]:
    """``{image: {point: error_px}}`` for uncradled tie-point marks only.

    Anchors are excluded here rather than subtracted later: this is the whole
    correction the module docstring describes.
    """
    out: dict[str, dict[str, float]] = {}
    for row in getattr(report, "residual_lines", None) or []:
        if row.get("culled"):
            continue
        label = row.get("point")
        if label not in tie_ids:
            continue
        err = row.get("error_px")
        if err is None:
            continue
        out.setdefault(row.get("image", ""), {})[label] = float(err)
    return out


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return float((sum(v * v for v in values) / len(values)) ** 0.5)


def _shared_focal(report: Any) -> float | None:
    focals = getattr(report, "focals", None) or {}
    values = [float(f) for f in focals.values() if f]
    return sum(values) / len(values) if values else None


def body_free_copy(project: Project) -> Project:
    """The project as it was before any body anchor was clicked.

    The anchor *marks* are what is removed, not the instances or their models.
    The bodies are still in the scene and still drawn; they simply stop being
    evidence about where the cameras are, which is exactly the state the user
    describes as "the good solution" before the models went in.
    """
    trial = copy.deepcopy(project)
    trial.path = None          # a trial must never be saved over the project
    for im in trial.images:
        im.anchor_obs = {}
    return trial


def check_cad_bodies(
    project: Project,
    *,
    tolerance_px: float = TOLERANCE_PX,
    tolerance_fraction: float = TOLERANCE_FRACTION,
) -> CadReport:
    """Solve with the bodies and without them, and compare per image.

    ``project`` is not modified: both solves run on deep copies.
    """
    from .solve import solve      # local, as in nominalcheck: solve imports widely

    rep = CadReport()
    instances = list(getattr(project, "instances", []) or [])
    rep.n_instances = len(instances)
    rep.n_anchor_marks = sum(len(im.anchor_obs or {}) for im in project.active_images())

    if not rep.n_anchor_marks:
        rep.verdict = "untestable"
        rep.reason = (
            "no body anchors are marked, so the bodies are not influencing the solve and "
            "there is nothing to compare. This check answers 'did the CAD make the "
            "photographs fit worse', and without anchor marks the answer is no by "
            "construction."
        )
        return rep

    tie_ids = _tie_ids(project)
    if not tie_ids:
        rep.verdict = "untestable"
        rep.reason = "no tie points, so there is no point solution for the bodies to worsen"
        return rep

    with_bodies = copy.deepcopy(project)
    with_bodies.path = None
    after_report = solve(with_bodies)
    before_report = solve(body_free_copy(project))

    if not getattr(after_report, "ok", False) or not getattr(before_report, "ok", False):
        rep.verdict = "untestable"
        which = "with the bodies" if not getattr(after_report, "ok", False) else "without them"
        rep.reason = (
            f"the solve {which} did not succeed, so there are not two solutions to compare. "
            f"Fix the solve first; this check cannot say anything about bodies in a project "
            f"that will not solve."
        )
        return rep

    before = _per_image(before_report, tie_ids)
    after = _per_image(after_report, tie_ids)
    rep.focal_before_px = _shared_focal(before_report)
    rep.focal_after_px = _shared_focal(after_report)

    all_before: list[float] = []
    all_after: list[float] = []
    for iid in sorted(set(before) | set(after)):
        b, a = before.get(iid, {}), after.get(iid, {})
        shared = sorted(set(b) & set(a))
        if not shared:
            continue
        v = ImageVerdict(image=iid, n_points=len(shared))
        v.before_px = _rms([b[p] for p in shared])
        v.after_px = _rms([a[p] for p in shared])
        v.delta_px = v.after_px - v.before_px
        v.allowed_px = max(float(tolerance_px), float(tolerance_fraction) * v.before_px)
        v.worsened = v.delta_px > v.allowed_px
        worst = max(shared, key=lambda p: a[p] - b[p])
        v.worst_point, v.worst_point_delta_px = worst, a[worst] - b[worst]
        rep.images.append(v)
        all_before.extend(b[p] for p in shared)
        all_after.extend(a[p] for p in shared)

    if not rep.images:
        rep.verdict = "untestable"
        rep.reason = (
            "no tie-point mark survived the cull in both solves, so the two have no common "
            "ground to be compared on"
        )
        return rep

    rep.before_rms_px = _rms(all_before)
    rep.after_rms_px = _rms(all_after)
    rep.images.sort(key=lambda v: -v.delta_px)
    rep.worsened = [v.image for v in rep.images if v.worsened]
    rep.worst_image = rep.images[0].image
    rep.ok = True

    if rep.worsened:
        rep.verdict = "worsens"
        rep.notes.append(
            f"the bodies make {len(rep.worsened)} of {len(rep.images)} image(s) fit their own "
            f"tie points worse: {', '.join(rep.worsened)}. Worst is {rep.worst_image}, "
            f"{rep.images[0].before_px:.2f} to {rep.images[0].after_px:.2f} px, with "
            f"{rep.images[0].worst_point} alone moving "
            f"{rep.images[0].worst_point_delta_px:+.2f} px. Look at that point's "
            f"neighbourhood in that photo: an anchor clicked on the wrong one of two similar "
            f"corners is the usual cause, and a model at the wrong scale is the next."
        )
    elif rep.after_rms_px < rep.before_rms_px - IMPROVEMENT_PX:
        rep.verdict = "improves"
        rep.notes.append(
            f"the bodies improve the fit, {rep.before_rms_px:.2f} to {rep.after_rms_px:.2f} px "
            f"overall, and no image is worse than its allowance. This is what a correctly "
            f"marked known object looks like: it is a stronger claim than a tie point and it "
            f"is paying for itself."
        )
    else:
        rep.verdict = "harmless"
        rep.notes.append(
            f"the bodies change the tie-point fit very little, {rep.before_rms_px:.2f} to "
            f"{rep.after_rms_px:.2f} px, and no image is worse than its allowance. They are "
            f"not doing damage and they are not doing much work either -- worth knowing "
            f"before marking more anchors on them."
        )

    if rep.focal_before_px and rep.focal_after_px:
        drift = abs(rep.focal_after_px - rep.focal_before_px) / rep.focal_before_px
        if drift > 0.01:
            rep.notes.append(
                f"the shared focal moved {100 * drift:.1f}%, {rep.focal_before_px:.0f} to "
                f"{rep.focal_after_px:.0f} px. A body pulling the focal is the failure the "
                f"focal wall was added for; the wall stops a collapse but does not object to "
                f"a few percent, and a few percent is still the cameras being bent to suit "
                f"the model."
            )
    return rep


def restore_without_bodies(project: Project, report: CadReport) -> str:
    """Put the body-free camera solution back, when the bodies made things worse.

    The honest half of the enforcement. It does not place the bodies against the
    restored cameras -- see the module docstring for why that needs a solve mode
    that does not exist -- so this leaves the project in the state the user calls
    "the good solution", with the bodies present but no longer dictating where the
    cameras are. Their anchors stay marked, so nothing the user clicked is lost
    and the check can be run again after re-clicking.

    Returns what it did, for the caller to show. Does nothing unless the verdict
    is ``"worsens"``, so calling it unconditionally is safe.
    """
    from .solve import solve

    if report.verdict != "worsens":
        return "nothing to undo: the bodies are not making any image worse"

    trial = body_free_copy(project)
    fresh = solve(trial)
    if not getattr(fresh, "ok", False):
        return (
            "the body-free solve did not succeed on its own, so there is no better state to "
            "go back to; the project is left as it is"
        )
    by_id = {im.id: im for im in trial.images}
    moved = 0
    for im in project.images:
        src = by_id.get(im.id)
        if src is None or src.R is None or src.t is None:
            continue
        im.R, im.t = src.R, src.t
        moved += 1
    positions = {p.id: p.xyz for p in trial.points if p.xyz is not None}
    for pt in project.points:
        if pt.id in positions:
            pt.xyz = positions[pt.id]
    return (
        f"restored the body-free solution: {moved} camera pose(s) and "
        f"{len(positions)} point(s) put back, anchors left marked. The bodies are still in "
        f"the scene but are no longer bending the cameras."
    )
