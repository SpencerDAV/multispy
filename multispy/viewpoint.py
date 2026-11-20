"""Which photograph is doing the work, and where should the next one stand?

The origin's uncertainty is anisotropic -- :mod:`uncertainty` measures it and
reports the principal axes -- but knowing *that* X is the worst direction does
not tell you what to do about it. This module answers the next question: which
of the photographs you already have is supplying the information along that
direction, and what viewpoint would supply more.

It exists because of a reasonable instinct that turns out to be a trap. Looking
at a set like PRM1 -- five of nine cameras aimed within 25 degrees of the part's
X axis, and X the weakest direction of the origin -- the obvious move is to
down-weight those cameras, since a camera cannot see depth along its own axis.
Do not. A reprojection residual is measured in pixels, and the derivative of a
pixel with respect to a world displacement *along* the viewing ray is already
nearly zero, so the normal equations already receive almost no along-ray
information from that camera. The Jacobian is a per-observation 3x3 information
matrix and it has been doing this the whole time. Multiplying such a camera's
residual by a scalar would discount it a second time, and worse, a scalar cannot
tell the axes apart: a camera blind to X is looking straight *across* Y and Z and
is your best witness for both. Down-weighting it by aim throws the good away
with the bad.

So nothing here changes a weight. Every number below is a readout of what the
solve already does, turned into the one thing the solve cannot tell you: where
to point the camera next.

What is being measured
----------------------
One view pins a point across its ray and not along it. At distance ``z`` with a
focal of ``f`` pixels and a click precision of ``sigma_px``, the across-ray slack
is ``(sigma_px / f) * z``, and the information that view contributes about a
direction ``w`` is

    (1 - (u . w)^2) / across^2

for a unit ray ``u``. Both terms matter and both are geometry the solve already
has: the numerator is the aiming penalty -- zero for a camera staring straight
down ``w``, one for a camera looking across it -- and the denominator is why
standing closer is worth more than standing anywhere clever.

Summing that over a camera's marks gives its share of the total information
along ``w``. Three honest limits on reading those shares:

*   This is information about the *points*, pooled. The origin is an
    extrapolation from the nominal-linked points through ``align_to_nominals``,
    so the pooled figure describes where the structure is weak along ``w``, not
    the origin's covariance. ``origin_uncertainty`` is the covariance and this is
    not a replacement for it -- which is why the weak direction is read *from*
    that measurement rather than re-derived here. The per-camera breakdown is
    reported twice for the same reason: over every positioned point, and over
    the nominal-linked points alone, which are the ones the origin actually
    hangs on.
*   A share is a fraction of what exists, not a verdict on what is possible.
    When the error ellipsoid is nearly spherical there is no lopsided axis left
    to harvest and no viewpoint will help much; the information has to come from
    more marks and more nominals instead. ``verdict`` says which case you are in,
    rather than leaving a suggestion to be read as a recommendation.
*   Known-object anchors are left out. They refine six pose parameters per body
    rather than a point's position, so they do not belong in a sum over point
    positions, and folding them in would make a densely anchored body look like
    a well-placed camera.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .project import ImageView, Project
from .uncertainty import OriginUncertainty, fingerprint

ANISOTROPY_FLOOR = 2.0
"""Below this ratio of worst to best sigma, call the origin viewpoint-limited no
more.

A sphere has nothing for a new vantage to fix: every direction is equally bad,
so the deficit is not geometry but evidence, and the answer is more nominals
rather than another photograph. 2.0 is where the worst axis is still only twice
the best -- a camera placed perfectly against the worst direction could at most
bring it level with the others, and the containment radius, which is dominated
by the whole ellipsoid and not by one axis, would barely move. Above it the
lever is real enough to aim at."""

CIRCLE_SAMPLES = 720
"""Bearings tried around the circle perpendicular to the weak direction. Half a
degree of resolution, far finer than anyone can stand."""

MIN_CAMERAS = 2
"""Fewer posed cameras than this and there is nothing to compare."""

MIN_POINTS = 2
"""Fewer positioned points than this and there is no information to apportion."""


@dataclass
class CameraContribution:
    """One photograph's share of the information along the weak direction."""

    image: str
    marks: int = 0
    """Marks in this photo on points the solve managed to position."""
    nominal_marks: int = 0
    """How many of those carry a nominal, and so bear on the origin directly."""
    look: list[float] = field(default_factory=list)
    """Optical axis in part space -- the direction the camera faces."""
    centre: list[float] = field(default_factory=list)
    """Camera position in part space, in ``ViewpointReport.units``."""
    cos_weak: float = 0.0
    """``|cos|`` between the optical axis and the weak direction. Near 1 means
    the camera is staring down the very direction that is poorly known and can
    say almost nothing about it; near 0 means it is looking across it."""
    share: float = 0.0
    """Fraction of the information along the weak direction that this camera
    supplies, over every positioned point."""
    share_per_mark: float = 0.0
    """``share`` divided by ``marks``. The figure to compare cameras on: a total
    can be large merely because a photo is densely marked, and what you want to
    know before clicking is what the *next* mark in this photo is worth."""
    nominal_share: float = 0.0
    """``share`` restricted to the nominal-linked points."""
    mean_distance: float = 0.0
    """Mean distance from this camera to the points it marks, in
    ``ViewpointReport.units``. Information falls off as its square, so this is
    often the larger half of why one photo beats another -- worth checking
    before reading a share as being about aim."""
    unmarked_in_frame: int = 0
    """Positioned points that fall inside this photo's frame and are not marked
    in it. What is left to click here."""
    gain_if_marked: float = 0.0
    """Information along the weak direction that marking all of those would add,
    as a fraction of the present total. An upper bound, and a soft one: it
    assumes every one of them is actually findable in this photo, which only
    looking will settle. Occlusion is not modelled -- a point inside the frame
    may be behind the part."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NewViewpoint:
    """A line of sight that is not already covered, for the next photograph."""

    bearing: list[float] = field(default_factory=list)
    """Direction the camera should face, in part space. Perpendicular to the
    weak direction by construction -- that is what makes it worth taking -- and
    chosen, among all such directions, to be the one furthest from the lines of
    sight already in the set."""
    position: list[float] = field(default_factory=list)
    """Where to stand: the marked points' centroid, backed off along
    ``bearing``. In ``ViewpointReport.units``.

    One of two equally informative sides -- a camera and its opposite constrain
    a point identically -- so this is the side with fewer cameras already on it.
    If the cell is in the way, the other side is just as good; see
    ``position_opposite``."""
    position_opposite: list[float] = field(default_factory=list)
    """The same line of sight from the other side, for when the first is a wall."""
    standoff: float = 0.0
    """Distance from the centroid, in ``ViewpointReport.units``. Set to the
    closest standoff already in the set rather than to anything optimal, because
    a distance one photograph was taken from is a distance that can be stood at
    -- and closer is always better, so there is no reason to suggest further."""
    nearest_existing_deg: float = 0.0
    """Angle between this line of sight and the nearest one already in the set.
    Small means the vantage is not new and the gain below is mostly from standing
    closer, not from standing elsewhere."""
    gain: float = 0.0
    """Information along the weak direction a photograph from here would add, as
    a fraction of the present total, assuming it marks as many points as the
    median photo already does."""
    marks_assumed: int = 0
    """The median mark count that ``gain`` assumes."""
    sigma_factor: float = 1.0
    """What ``gain`` is worth as a *distance*: multiply the weak sigma by this.

    Here so that a gain cannot be misread. Information adds, sigma goes as its
    inverse square root, so a gain that sounds enormous is a modest improvement
    in millimetres -- half as much again buys about 18% off the sigma along this
    one direction, and the containment radius, which answers to the whole
    ellipsoid, moves less still. First order and indicative: it assumes the
    origin's sigma along this direction tracks the marks' sigma along it, which
    is the right shape of relationship but not a measurement. The only honest way
    to know what a new photograph did is to take it and re-measure."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ViewpointReport:
    """Where the information along the origin's worst direction comes from."""

    ok: bool = False
    reason: str = ""
    """Why there is nothing to report, when ``ok`` is false."""
    weak: list[float] = field(default_factory=list)
    """The direction the origin is least certain in, as a unit vector in part
    space."""
    weak_sigma: float = 0.0
    """Its sigma, in ``units``. Zero when the direction came from geometry
    rather than from a measurement."""
    anisotropy: float = 1.0
    """Worst sigma over best. 1 is a sphere; see ``ANISOTROPY_FLOOR``."""
    source: str = ""
    """``"origin"`` when the weak direction was read from a stored
    ``origin_uncertainty``, ``"geometry"`` when that was missing and the
    direction was taken from the pooled information instead.

    Worth reading before anything else. The measured one is the real thing: it
    knows about the nominals, the control lines and the lever from the datum
    triangle out to the origin. The geometric fallback knows only where the
    cameras are, so it answers "which direction are the marks weakest in", which
    is related to the origin's worst direction but is not it."""
    stale: bool = False
    """The stored origin measurement was taken against different data. The weak
    direction may have moved; re-measure before standing anywhere on it."""
    units: str = "mm"
    sigma_px: float = 0.0
    n_points: int = 0
    n_nominal_points: int = 0
    cameras: list[CameraContribution] = field(default_factory=list)
    """Largest share first."""
    best_per_mark: str = ""
    """The photograph where one more mark buys the most."""
    most_left_to_click: str = ""
    """The photograph with the largest ``gain_if_marked`` -- usually the cheapest
    real improvement available, since it needs no new photograph at all."""
    suggestion: NewViewpoint | None = None
    verdict: str = ""
    """``"viewpoint-limited"``, ``"evidence-limited"`` or ``"unknown"``."""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["suggestion"] = self.suggestion.to_dict() if self.suggestion else None
        return d


# ----------------------------------------------------------------- geometry


def _posed(project: Project) -> list[ImageView]:
    return [im for im in project.active_images() if im.R is not None and im.t is not None]


def _axis_and_centre(im: ImageView) -> tuple[np.ndarray, np.ndarray]:
    """Optical axis and centre in world coordinates.

    ``R`` is world-to-camera -- ``_Problem._project_block`` computes
    ``Xw @ R.T + t`` -- so the camera's own +Z expressed in the world is the
    third *row* of ``R``, and the centre is ``-R.T @ t``.
    """
    R = np.asarray(im.R, dtype=float)
    t = np.asarray(im.t, dtype=float)
    return np.array(R[2, :], dtype=float), np.asarray(-R.T @ t, dtype=float)


def _focal(project: Project, im: ImageView) -> float | None:
    f = getattr(project.group_of(im), "f", None)
    return float(f) if f else None


def _information(
    project: Project, images: list[ImageView], pts: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, list[float]]]:
    """Per-camera and pooled across-ray information over ``pts``.

    In world units throughout; only ratios of these are ever reported, so the
    scale cancels.
    """
    sig = float(project.settings.sigma_px)
    per: dict[str, np.ndarray] = {}
    dists: dict[str, list[float]] = {}
    pooled = np.zeros((3, 3))
    for im in images:
        f = _focal(project, im)
        H = np.zeros((3, 3))
        zs: list[float] = []
        if f:
            _look, C = _axis_and_centre(im)
            for pid in im.observations:
                X = pts.get(pid)
                if X is None:
                    continue
                d = X - C
                z = float(np.linalg.norm(d))
                if z <= 1e-9:
                    continue
                u = d / z
                across = (sig / f) * z
                if across <= 1e-12:
                    continue
                H += (np.eye(3) - np.outer(u, u)) / (across * across)
                zs.append(z)
        per[im.id] = H
        dists[im.id] = zs
        pooled += H
    return per, pooled, dists


def _in_frame(project: Project, im: ImageView, X: np.ndarray) -> bool:
    """Does this world point land inside this photo?

    Lens distortion is ignored. It moves a point by a few pixels at the frame
    edge, which can only ever change the answer for a point that is on the
    boundary anyway, and the count this feeds is advisory.
    """
    f = _focal(project, im)
    if not f:
        return False
    g = project.group_of(im)
    R = np.asarray(im.R, dtype=float)
    t = np.asarray(im.t, dtype=float)
    Xc = R @ np.asarray(X, dtype=float) + t
    if Xc[2] <= 1e-6:
        return False
    cx, cy = g.principal_point_for(im.width, im.height)
    x = f * Xc[0] / Xc[2] + cx
    y = f * Xc[1] / Xc[2] + cy
    return 0.0 <= x < float(im.width) and 0.0 <= y < float(im.height)


def _weakest(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Direction of largest variance, and the ratio of worst sigma to best."""
    ev, V = np.linalg.eigh(cov)
    ev = np.maximum(ev, 0.0)
    order = np.argsort(-ev)
    worst, best = float(np.sqrt(ev[order[0]])), float(np.sqrt(ev[order[-1]]))
    return np.asarray(V[:, order[0]], dtype=float), (worst / best if best > 0 else float("inf"))


def _perpendicular_basis(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(w @ seed)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(w, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(w, e1)


# ------------------------------------------------------------------- report


def viewpoint_report(project: Project) -> ViewpointReport:
    """Apportion the information along the origin's worst direction.

    Read-only. ``project`` must already be solved; nothing here re-solves, so
    the figures describe the poses as they stand.
    """
    rep = ViewpointReport(
        units=project.nominal_units or "mm",
        sigma_px=float(project.settings.sigma_px),
    )
    scale = float(project.nominal_scale) or 0.001

    images = _posed(project)
    if len(images) < MIN_CAMERAS:
        rep.reason = (
            f"only {len(images)} camera(s) are posed; at least {MIN_CAMERAS} are needed "
            f"before one viewpoint can be compared with another"
        )
        return rep

    pts = {p.id: np.asarray(p.xyz, dtype=float) for p in project.points if p.xyz is not None}
    nominal_ids = {p.id for p in project.points if p.nominal and p.xyz is not None}
    if len(pts) < MIN_POINTS:
        rep.reason = f"only {len(pts)} positioned point(s); there is no information to apportion"
        return rep
    rep.n_points = len(pts)
    rep.n_nominal_points = len(nominal_ids)

    per, pooled, dists = _information(project, images, pts)
    if not np.all(np.isfinite(pooled)):
        # A pose or a point carrying a NaN -- a camera pointed exactly along its
        # own up vector will do it. Caught here rather than left to surface as a
        # LAPACK "eigenvalues did not converge" three steps further on, which
        # says nothing about which camera is broken.
        bad = [im.id for im in images if not np.all(np.isfinite(per[im.id]))]
        rep.reason = (
            f"the pose or the marks of {', '.join(bad) or 'some image'} are not finite, so "
            f"there is no information to apportion. Re-solve before reading this."
        )
        return rep
    nom_pts = {pid: X for pid, X in pts.items() if pid in nominal_ids}
    per_nom, pooled_nom, _ = _information(project, images, nom_pts)

    # ---- the weak direction: measured if there is a measurement, geometric if not
    stored = OriginUncertainty.from_dict(project.origin_uncertainty)
    w: np.ndarray | None = None
    if stored is not None and stored.axes and stored.sigma:
        cand = np.asarray(stored.axes[0], dtype=float)
        n = float(np.linalg.norm(cand))
        if n > 1e-12:
            w = cand / n
            rep.weak_sigma = float(stored.sigma[0])
            rep.anisotropy = float(stored.anisotropy or 1.0)
            rep.source = "origin"
            rep.stale = bool(stored.fingerprint) and stored.fingerprint != fingerprint(project)
    if w is None:
        try:
            cov = np.linalg.inv(pooled)
            w, rep.anisotropy = _weakest(cov)
        except np.linalg.LinAlgError:
            rep.reason = (
                "the marks do not constrain all three directions, so there is no worst one: "
                "every point is seen from too few directions to be placed in space at all"
            )
            return rep
        rep.source = "geometry"
        rep.notes.append(
            "no origin measurement is stored, so the weak direction here is the one the "
            "marks themselves are weakest in, not the one the origin is weakest in. Measure "
            "the origin and read this again -- the two can differ, because the origin also "
            "carries the lever from the datum triangle out to (0,0,0)."
        )

    rep.weak = [float(v) for v in w]
    total = float(w @ pooled @ w)
    total_nom = float(w @ pooled_nom @ w)
    if total <= 0:
        rep.reason = "no camera contributes anything along the weak direction"
        return rep

    # ---- per-camera apportionment
    cams: list[CameraContribution] = []
    for im in images:
        look, C = _axis_and_centre(im)
        marked = [pid for pid in im.observations if pid in pts]
        zs = dists[im.id]
        unmarked = [
            pid for pid, X in pts.items() if pid not in im.observations and _in_frame(project, im, X)
        ]
        c = CameraContribution(
            image=im.id,
            marks=len(marked),
            nominal_marks=sum(1 for pid in marked if pid in nominal_ids),
            look=[float(v) for v in look],
            centre=[float(v) / scale for v in C],
            cos_weak=abs(float(look @ w)),
            share=float(w @ per[im.id] @ w) / total,
            nominal_share=(float(w @ per_nom[im.id] @ w) / total_nom) if total_nom > 0 else 0.0,
            mean_distance=(float(np.mean(zs)) / scale) if zs else 0.0,
            unmarked_in_frame=len(unmarked),
        )
        c.share_per_mark = c.share / c.marks if c.marks else 0.0
        # What the unmarked ones would add, at this camera's own geometry.
        f = _focal(project, im)
        if f and unmarked:
            add = 0.0
            for pid in unmarked:
                d = pts[pid] - C
                z = float(np.linalg.norm(d))
                if z <= 1e-9:
                    continue
                u = d / z
                across = (float(project.settings.sigma_px) / f) * z
                if across <= 1e-12:
                    continue
                add += (1.0 - float(u @ w) ** 2) / (across * across)
            c.gain_if_marked = add / total
        cams.append(c)

    counts = sorted(c.marks for c in cams if c.marks)
    median_marks = counts[len(counts) // 2] if counts else 0
    cams.sort(key=lambda c: -c.share)
    rep.cameras = cams
    rep.best_per_mark = max(cams, key=lambda c: c.share_per_mark).image if cams else ""
    rep.most_left_to_click = max(cams, key=lambda c: c.gain_if_marked).image if cams else ""

    # ---- where an uncovered line of sight would be
    centroid = np.mean(np.array(list(pts.values())), axis=0)
    standoffs = [float(np.linalg.norm(_axis_and_centre(im)[1] - centroid)) for im in images]
    standoff = min(standoffs) if standoffs else 0.0
    looks = [_axis_and_centre(im)[0] for im in images]
    e1, e2 = _perpendicular_basis(w)
    best_dir, best_sep = None, -1.0
    for k in range(CIRCLE_SAMPLES):
        th = 2.0 * np.pi * k / CIRCLE_SAMPLES
        d = np.cos(th) * e1 + np.sin(th) * e2
        # Lines of sight, not arrows: a camera and its opposite see the same
        # across-ray slice, so novelty is measured on |cos|.
        nearest = min(
            float(np.degrees(np.arccos(min(1.0, abs(float(d @ L)))))) for L in looks
        )
        if nearest > best_sep:
            best_dir, best_sep = d, nearest
    focals = [f for f in (_focal(project, im) for im in images) if f]
    if best_dir is not None and focals and standoff > 0 and median_marks:
        f_med = float(np.median(focals))
        across = (float(project.settings.sigma_px) / f_med) * standoff
        # best_dir is perpendicular to w, so the aiming term is 1.
        add = median_marks / (across * across)
        side = sum(1 for im in images if float((_axis_and_centre(im)[1] - centroid) @ best_dir) < 0)
        flip = side > len(images) - side
        near = centroid + (best_dir if flip else -best_dir) * standoff
        far = centroid - (best_dir if flip else -best_dir) * standoff
        rep.suggestion = NewViewpoint(
            bearing=[float(v) for v in (-best_dir if flip else best_dir)],
            position=[float(v) / scale for v in near],
            position_opposite=[float(v) / scale for v in far],
            standoff=standoff / scale,
            nearest_existing_deg=best_sep,
            gain=add / total,
            marks_assumed=median_marks,
            sigma_factor=float(1.0 / np.sqrt(1.0 + add / total)),
        )

    # ---- verdict and notes
    if rep.anisotropy >= ANISOTROPY_FLOOR:
        rep.verdict = "viewpoint-limited"
        rep.notes.append(
            f"the worst direction is {rep.anisotropy:.2f}x the best, so there is a real lever "
            f"here: a photograph across the weak direction can take some of it out."
        )
    else:
        rep.verdict = "evidence-limited"
        rep.notes.append(
            f"the error is nearly spherical ({rep.anisotropy:.2f}x worst to best), so no "
            f"viewpoint is badly missing and none will help much -- a new vantage redistributes "
            f"information, it does not create any. What is short is evidence: more marks, and "
            f"above all more nominals. Read the camera table as where to click, not as where "
            f"to stand."
        )
        # And say the harder thing: at this ratio the direction is not just
        # unhelpful to aim at, it is barely defined. Three near-equal axes means
        # the choice of "worst" turns on small differences, so it swings on what
        # the nominals happen to pin -- on PRM1, changing which axes two of four
        # links constrain moved it 54 degrees while the reprojection RMS held at
        # 2 px. The shares above are geometry and stand regardless; this
        # direction is the part that should not be read as precise.
        rep.notes.append(
            "at that ratio the weak *direction* is itself poorly determined -- three near-equal "
            "axes leave little to choose between them, so it moves with which axes the nominals "
            "pin rather than with the photographs. Trust the per-camera shares, which are "
            "geometry; treat the direction as provisional until the error is clearly lopsided."
        )
    if rep.stale:
        rep.notes.append(
            "the stored origin measurement was taken against different marks, lines, nominals "
            "or sigmas, so the weak direction below may have moved. Re-measure the origin "
            "before acting on it."
        )
    if rep.n_nominal_points < 3:
        rep.notes.append(
            f"only {rep.n_nominal_points} positioned point(s) carry a nominal, so the "
            f"nominal-only shares are not worth reading yet -- the origin hangs on too few "
            f"points for a per-camera split of them to mean anything."
        )
    spread = [c.cos_weak for c in cams]
    if spread and max(spread) - min(spread) > 0.5:
        rep.notes.append(
            "no weight follows from any of this. A camera aimed down the weak direction is "
            "already discounted for it by the Jacobian, and it is your best witness for the "
            "two directions across it -- down-weighting it by aim would suppress both."
        )
    rep.ok = True
    return rep
