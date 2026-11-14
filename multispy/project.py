"""Project data model and JSON persistence.

A project is one ``.multispy`` JSON file plus the photos it points at. Image
paths are stored relative to the project file so a project folder stays
portable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .mesh import FEATURE_ANGLE_DEG

FORMAT_VERSION = 3
"""Version 2 adds object templates and instances. Version 3 adds the part
instance and the per-image record of an edge refinement. Older files load
unchanged -- they simply have none."""

ANCHOR_SEP = "/"
"""Separates instance id from anchor name in an observation key.

Anchor observations live in the same flat ``{key: [u, v]}`` shape as tie points
so that the solver's pruning, connectivity and linear-initialisation stages can
treat both alike, which is what keeps this feature out of three different code
paths. Anchor names are sanitised to keep the key unambiguous.
"""


def anchor_key(instance_id: str, anchor_name: str) -> str:
    return f"{instance_id}{ANCHOR_SEP}{anchor_name}"


def split_anchor_key(key: str) -> tuple[str, str]:
    instance_id, _sep, anchor_name = key.partition(ANCHOR_SEP)
    return instance_id, anchor_name


def clean_anchor_name(name: str) -> str:
    """Anchor names index into observation keys, so the separator cannot appear."""
    cleaned = str(name or "").replace(ANCHOR_SEP, "-").strip()
    return cleaned[:60] or "anchor"


@dataclass
class AxisLine:
    """A control line segment, drawn along a world axis.

    Direction matters: ``p0 -> p1`` is taken to point along the *positive* axis,
    which is how the sign ambiguity of the vanishing point gets resolved.
    """

    axis: int  # 0=X, 1=Y, 2=Z
    p0: list[float]
    p1: list[float]
    uncertain: bool = False
    """Drawn along an edge you are not sure is really parallel to the axis.

    Not the same as deleting it. A doubtful line still says something, and in a
    photo with only two lines on an axis it may be all there is -- so it keeps
    contributing, at ``Settings.uncertain_factor`` times its usual sigma, and it
    is the first thing dropped when the solve decides something has to go."""

    def length(self) -> float:
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))


@dataclass
class CameraGroup:
    """A set of images sharing intrinsics -- i.e. shot on one camera at one zoom.

    Grouping is what makes the whole approach stable: one focal length
    constrained by every image's control lines *and* every tie point beats N
    independent focal lengths each constrained by four clicks.
    """

    id: str
    f: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float = 0.0
    lock_focal: bool = False
    lock_principal_point: bool = True
    lock_k1: bool = True


@dataclass
class ImageView:
    id: str
    path: str
    width: int
    height: int
    group: str = "cam0"
    enabled: bool = True
    axis_lines: list[AxisLine] = field(default_factory=list)
    observations: dict[str, list[float]] = field(default_factory=dict)
    anchor_obs: dict[str, list[float]] = field(default_factory=dict)
    """Known-object anchors marked in this photo, keyed ``"<instance>/<anchor>"``.

    Kept separate from ``observations`` because these are not free points -- their
    world positions are dictated by the instance's pose and the model's geometry.
    """
    # solved extrinsics, world-to-camera
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    edge_refined: bool = False
    """This camera's pose was nudged by the edge refinement rather than by the
    solve alone.

    Recorded on the image, not in a report, because the pose outlives the report
    that produced it: a project reopened a week later would otherwise give no
    sign that one of its cameras carries a number no bundle adjustment ever
    agreed to."""
    R_unrefined: np.ndarray | None = None
    t_unrefined: np.ndarray | None = None
    """The pose as the solve left it, kept so a refinement can be undone exactly.

    Cheap insurance against the failure mode this whole feature has: an
    alignment onto the wrong edges is confident, plausible and wrong, and the
    only honest answer to it is to be able to put the old pose back without
    re-solving the entire project."""

    def axis_segments(self) -> dict[int, np.ndarray]:
        """Control lines regrouped as ``{axis: (N,4) array}`` for the geometry code."""
        out: dict[int, list[list[float]]] = {}
        for ln in self.axis_lines:
            out.setdefault(ln.axis, []).append([ln.p0[0], ln.p0[1], ln.p1[0], ln.p1[1]])
        return {a: np.array(v, dtype=float) for a, v in out.items()}

    def center(self) -> np.ndarray | None:
        """Camera centre in world coordinates."""
        if self.R is None or self.t is None:
            return None
        return -self.R.T @ self.t


@dataclass
class TiePoint:
    id: str
    label: str = ""
    xyz: np.ndarray | None = None
    nominal: list[float] | None = None
    """Where this point is *supposed* to be, in ``Project.nominal_units``.

    This is what puts a reconstruction into part space. A photogrammetric solve
    has no idea where the origin is or which way is up; a drawing does. Give
    three or more points the coordinates the drawing states and the solved frame
    stops being arbitrary and becomes the part's own frame -- at which point a
    CAD model of the part can be dropped in at the identity and the fit is
    meaningful.

    Stored in the project's nominal units rather than in metres, because that is
    what you read off the drawing, and a round 120.0 that silently becomes
    0.12000000000000001 in the file is a round number nobody can search for."""
    nominal_sigma: float | None = None
    """Tolerance on this point's nominal, in nominal units. ``None`` uses
    ``Settings.sigma_nominal``.

    A real tolerance, not a formality: it is what decides whether the solve
    bends to meet the drawing or the drawing bends to meet the solve. Tight on
    the datum features that define the frame, loose on everything else."""
    nominal_source: str = ""
    """Which named feature this point's nominal was taken from, if any.

    Provenance again, and for the same reason ``ObjectInstance.calibrated``
    carries a session: a typed nominal and one adopted from a calibration file
    are different kinds of claim. A typed one is what someone read off a drawing.
    An adopted one inherits everything that was wrong with the sitting that
    produced it, so when a session turns out to have been calibrated against a
    part positioned half a metre out, this is what makes it possible to find
    every nominal that came from it instead of re-checking all of them."""
    uncertain: bool = False
    """Marked where you could not quite see the feature.

    Keeps contributing, at ``Settings.uncertain_factor`` times its usual sigma,
    and is dropped sooner than a confident mark when the outlier pass runs."""


@dataclass
class ObjectAnchor:
    """A point on a known object's surface, in the model's own local coordinates
    and its own units.

    Only features a person can unambiguously identify in a photograph are worth
    marking -- body corners, mount-hole centres, the rim of a lens. The middle of
    a flat face is unmarkable because nothing in the photo locates it.
    """

    name: str
    xyz: list[float]


@dataclass
class OpticalFrame:
    """Where a camera actually looks, relative to the body the CAD model describes.

    Without this, solving a camera array tells you where each plastic shell sits,
    which is not what anyone wants to know. With it, the solve reports the real
    extrinsics of every camera in the rig: entrance pupil and optical axis.

    ``forward`` and ``up`` are directions in model-local coordinates; ``center``
    is a position in model-local units.
    """

    center: list[float]
    forward: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    up: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    fov_deg: float | None = None
    """Nominal horizontal field of view, if known. Only ever used for drawing a
    frustum at the right size -- nothing in the solve depends on it, since the
    photos being marked come from a different camera entirely."""


@dataclass
class ObjectTemplate:
    """A known object: its mesh, its real size, and the points marked on it.

    One template, many instances -- which is the whole point for a camera array.
    Ten identical cameras share one template, so measuring the body once pins all
    ten placements, and every anchor marked in every photo refines the same
    shared geometry.
    """

    id: str
    name: str = ""
    mesh_path: str | None = None
    units: str = "mm"
    unit_scale: float = 0.001
    """World units per model unit. Kept explicitly alongside ``units`` so an
    unusual scale factor survives a round trip."""
    crease_angle: float = FEATURE_ANGLE_DEG
    """Facet-to-facet angle below which an edge is not drawn, in degrees.

    A property of the part rather than of the file: a heavily filleted body, or
    one tessellated finely enough that no two neighbouring facets differ by much,
    loses its curves from the wireframe entirely at the default. Stored per
    template because one project can hold both a machined bracket and a turned
    shaft, which want different answers."""
    anchors: list[ObjectAnchor] = field(default_factory=list)
    optical: OpticalFrame | None = None

    def anchor(self, name: str) -> ObjectAnchor | None:
        return next((a for a in self.anchors if a.name == name), None)

    def anchor_points(self) -> dict[str, np.ndarray]:
        """Anchors in world units, keyed by name."""
        return {
            a.name: np.array(a.xyz, dtype=float) * self.unit_scale for a in self.anchors
        }


@dataclass
class ObjectInstance:
    """One placement of a template in the scene, with a 6-DoF pose to solve for.

    ``R`` and ``t`` map model-local world-unit coordinates into the scene:
    ``X_world = R @ p_local + t``. Note this is object-to-world, the opposite
    sense from an image's world-to-camera ``R``.
    """

    id: str
    template: str
    label: str = ""
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    locked: bool = False
    """Hold this placement fixed during the bundle adjustment. Useful once one
    body is trusted and you want it to define the frame for the rest."""
    is_part: bool = False
    """This is *the part itself*, not an object placed in the scene.

    A placed object has an unknown pose that anchors measure. The part has no
    pose to measure: once the point nominals have put the reconstruction into
    the part's own coordinate frame, the part is at the origin by definition,
    because that frame was built from the part's own drawing. So this instance
    is pinned at ``R = I, t = 0`` and never enters the bundle as a parameter --
    it carries no anchors, costs no unknowns, and its wireframe lands in every
    photograph the moment the cameras are solved.

    The flag lives on the instance rather than the template because it is a
    statement about a *placement*, which is what an instance is: the same CAD
    file could legitimately be the part in one project and a placed object in
    another. It sits beside ``locked`` and ``manual`` for the same reason -- all
    three say where a pose came from.

    At most one instance in a project carries it; :meth:`Project.set_part`
    enforces that."""
    manual: bool = False
    """This pose was set by hand, by dragging the model over a photograph, and
    is not to be thrown away.

    A solve normally clears the pose of any body it cannot place from anchors,
    which is right for a stale fit but wrong for a placement someone has just
    made by eye. ``locked`` would also preserve it, but ``locked`` means
    something else -- that this body defines the world frame -- so a hand-placed
    body needs a flag of its own. The pose still moves with the scene when the
    coordinates are re-expressed, and anchors still override it: three anchors
    in two photos measure the pose far better than an eye can judge it."""
    pose_in_part_space: bool = False
    """This pose is already expressed in part space, so re-expressing the scene
    must leave it alone.

    The bug this exists to stop, which cost a real placement three metres: when a
    solve re-expresses the scene in part space it carries hand-placed bodies
    along, because nothing downstream reconstructs them. Every other pose in the
    scene is rebuilt from scratch each solve and so is correctly in the *old*
    frame when that transform is applied -- but a manual pose is loaded from the
    project file, where the previous solve already left it in part space. Apply
    the transform again and it moves by the whole part-space offset. Solve again
    and it moves again. Measured on this cell: 2.9 metres per solve, cumulative,
    marching a fixture clean off the scene while the solve reported nothing wrong
    because a manual pose has no residual to report.

    So the frame a pose is in has to be recorded rather than assumed. False means
    the pose predates this flag or was set against an unaligned scene, and must
    be carried; True means it is already in part space and must not be touched.
    Set when the transform is applied, and set by the front end when a body is
    dragged in a project that is already in part space."""
    calibrated: str = ""
    """This pose came out of a sensor calibration file, and this says which one.

    The third provenance, beside ``locked`` and ``manual``, and it exists for the
    same reason they do: everything downstream has to know where a pose came
    from. A calibrated pose is not a measurement this project made, it is a claim
    imported from a document that :mod:`multispy.sensors` has already shown can
    be wrong by half a metre -- so it must never be mistaken for a solved pose,
    and a solve must not silently discard it either.

    It holds the calibration session rather than a bare ``True`` because that is
    the unit the faults come in: sensors calibrated in one sitting share whatever
    was wrong with that setup, so being able to see at a glance which bodies in
    the scene came from the same morning is worth the extra string.

    The circularity this guards is the one ``uncertainty.py`` names. Imported
    poses make excellent nominals -- they are spread across the whole cell, which
    is exactly what shortens the lever arm on the origin -- but a fit judged by
    how well it met nominals taken from the thing being judged has proved
    nothing. Keeping the provenance means a report can say which bodies are
    evidence and which are assumptions."""


@dataclass
class AxisLockConstraint:
    """Pins one world coordinate of a point. Three of these pin an origin, and a
    shared value across several points states a plane (``z = 0`` for a floor)."""

    point: str
    axis: int
    value: float = 0.0
    sigma: float = 0.01
    type: str = "axis_lock"
    source: str = "constraint"
    """``"constraint"`` when someone typed it, ``"nominal"`` when it was
    generated from a point's nominal XYZ. The solve treats the two identically;
    this only keeps the UI from showing a drawing's worth of generated locks in
    a panel meant for the handful a person entered by hand."""


@dataclass
class DistanceConstraint:
    """A known real-world distance between two tie points. One of these turns the
    solve from arbitrary-scale into metric."""

    point_a: str
    point_b: str
    value: float
    sigma: float = 0.01
    type: str = "distance"


Constraint = AxisLockConstraint | DistanceConstraint


@dataclass
class Settings:
    sigma_px: float = 1.5
    """Expected click accuracy for tie points, in pixels."""
    sigma_line_px: float = 2.0
    """Expected accuracy of control line endpoints, in pixels."""
    sigma_anchor_px: float = 2.0
    """Floor on a known-object anchor mark's accuracy, in pixels.

    A *floor*, not an estimate: ``reweight_instances`` raises it per body to
    whatever that body's own marks turn out to deserve. Keep it near
    ``sigma_px``, and resist the urge to set it high.

    That is counter-intuitive, because an anchor really is the least reliable
    thing in the project -- a tie point is a feature you can see and click,
    while an anchor is a model vertex whose position you have to *judge* from
    one 2D projection with all three axes guessed at once, and it has no free
    position to absorb the mistake: the model's geometry is rigid, so whatever
    you get wrong is pushed into the camera poses instead of staying local.

    But loosening the whole class is the wrong response, and measurably so. On
    the rig scene with one body deliberately mis-marked, raising this from 1.5
    to 6.0 made the *other* bodies' baselines worse by 2x (0.54 mm -> 1.08 mm),
    because it softens the well-marked bodies just as much as the bad one and
    they are what was holding the solve together. Discrimination is what helps,
    not blanket distrust: let this stay tight and let the per-body sigma and the
    outlier cull find the body that actually deserves to be doubted."""
    anchor_count_ref: float = 6.0
    """Anchor marks per body beyond which extra ones stop buying linear
    influence, so a densely marked object cannot outvote the tie-point network
    that holds the cameras. 0 disables the normalisation."""
    reweight_instances: bool = True
    """After a first pass, give each body a sigma measured from its own
    residuals, so an object that disagrees with the photos declares itself
    unreliable instead of dragging the cameras over to meet it."""
    sigma_nominal: float = 0.1
    """Default tolerance on a point nominal, in the project's nominal units.

    The number that decides whether the drawing bends to the photos or the
    photos bend to the drawing. Loose enough by default that a nominal places
    the part without overriding what the images actually show; tighten it per
    point on the datum features that are supposed to define the frame."""
    origin_good: float = 5.0
    """At or under this, the origin's confidence sphere reads green, in nominal
    units. A default, not a standard: the only right value is the tolerance the
    part is actually inspected to, and anyone reading the colour without
    setting this is reading my guess, not their requirement. The figure itself
    is always drawn beside the sphere for that reason."""
    origin_warn: float = 20.0
    """Over this, the sphere reads red. Between the two, amber."""
    sensor_tol: float = 10.0
    """How far a calibrated sensor may aim from the working volume the other
    sensors agree on and still count as agreeing, in nominal units.

    Loose on purpose, and chosen by the person inspecting the cell rather than
    here. It is the floor on the consensus threshold as well as the pass mark:
    :func:`multispy.sensors.robust_scale` will not let a cluster of sensors that
    happen to agree to a millimetre convict every honest sensor outside it, so
    this number is what the file is actually held to when it is unusually
    self-consistent."""
    sensor_warn: float = 50.0
    """Past this a sensor's transform counts as badly wrong rather than drifted,
    in nominal units. It also picks which description a failing sensor gets: a
    miss inside this, at a small implied aim angle, reads as a mechanical nudge,
    while one far outside it reads as a transform calibrated against the wrong
    nominal feature."""
    uncertain_factor: float = 3.0
    """How much looser a mark flagged uncertain is weighted, and how much sooner
    it is culled. 1.0 makes the flag a label with no effect on the solve."""
    cull_sigma: float = 5.0
    """Clicked marks worse than this many sigma after the first pass are dropped
    from the second. 0 disables culling. Guarded: nothing is dropped that would
    leave a track, an image or a body without enough marks to stay solvable."""
    huber_f_scale: float = 3.0
    """Robust loss knee, in multiples of sigma. Clicks beyond this get downweighted."""
    solve_principal_point: bool = False
    """Try the 3-vanishing-point orthocenter during initialisation. Noisy; off by default."""
    refine_focal: bool = True
    focal_bound_factor: float = 1.5
    """How far the bundle may move the focal length from the value the control
    lines measured, as a multiplying factor either way. Anything at or below 1
    removes the bound entirely -- to hold the focal still, set ``lock_focal`` on
    the camera group, which says so plainly and does not pretend to refine.

    A bound rather than a prior, because the focal does not need steering inside
    this window -- it needs a wall around a trap outside it. The focal is the one
    parameter shared by every image in a group, so a single badly marked known
    object can buy itself a cheap fit by shrinking it, and the cost lands on all
    nine photographs at once. Measured on the cell this was found on: a known
    object whose anchors were clicked onto the wrong features, already
    down-weighted to 569 px by ``reweight_instances``, still dragged a 2786 px
    focal to 1.6 px and reported convergence at a cost of 1.04e5 -- while the
    same solve with the focal held still finished at 3557. The collapse is not a
    cheaper answer the solver preferred, it is a local minimum it fell into and
    could not climb back out of, which is why tapering the body's weight cannot
    stop it and a wall can. A synthetic version of the same body takes the focal
    to 0.016 px, and to *negative* values on another seed: a camera turned
    inside out.

    1.5 is deliberately generous. The legitimate refinement on that cell moved
    the focal by 1% (2786 -> 2757), which is 0.4 of the control lines' own
    stated uncertainty, and EXIF agreed with the lines to 0.35%; a bundle that
    wants to move it by half has not refined a calibration, it has lost one.
    The number is a setting because control lines can be genuinely poor -- see
    the warning ``calibrate_all`` raises past 8% -- and someone in that position
    may need the room. Widen it before suspecting it, and prefer
    ``lock_focal`` if the focal is known."""
    edge_refine: bool = False
    """Allow the image-edge refinement to touch camera poses at all.

    Off, deliberately. Stage one -- projecting the part's wireframe from the
    pose the nominals already imply -- is pure geometry and cannot be wrong in
    any way the solve was not already wrong. Stage two looks at the photograph
    and is a different kind of claim: on a busy background it can find a
    confident alignment onto the wrong edges, and a wrong answer delivered
    confidently is worse than no answer. So it is opt-in, per image, and
    reversible."""
    edge_sigma_px: float = 1.6
    """Gaussian blur, in pixels of the *working* image, before the gradient.

    Sets what counts as an edge: too little and every bit of texture is one, too
    much and a real silhouette softens into nothing."""
    edge_percentile: float = 92.0
    """Gradient-magnitude percentile above which a pixel is kept as an edge.

    A percentile rather than an absolute threshold because exposure varies
    wildly between photographs and an absolute number tuned on one of them is
    wrong on the next."""
    edge_truncate_px: float = 24.0
    """Chamfer distances are clipped here, in working pixels.

    The single most important number for not being dragged onto a background:
    an unclipped chamfer cost lets one wireframe edge that found nothing pull
    the whole pose towards whatever distant clutter is nearest. Clipped, an edge
    with no support simply stops voting."""
    edge_orientation_bins: int = 8
    """Separate distance fields, one per edge direction.

    The other defence against clutter. A vertical window frame behind the part
    is a strong edge, but it is not a *candidate* for a horizontal wireframe
    edge, and binning by orientation is what lets the cost say so."""
    edge_soften_px: float = 3.0
    """Distance at which a chamfer residual stops growing, in working pixels.

    Truncation alone is not enough and the measurement said so. The cost is an
    RMS, so a handful of samples sitting near the clip dominate it -- on a clean
    synthetic render at the *correct* pose, 95% of the wireframe lay within a
    pixel of its edge and the cost still read 7 px, because the remaining 5%
    (edges the detector never found) were at the cap. The optimiser could then
    lower the cost by dragging the pose to rescue those few at the expense of
    the many, which is the opposite of what it is for.

    A saturating residual, ``d / sqrt(1 + (d/soften)^2)``, fixes that properly:
    a sample that found its edge votes in proportion to how far off it is, and a
    sample that found nothing contributes an almost constant term with almost no
    gradient. Set it near the accuracy you expect of the alignment, not near the
    size of the mistakes you want it to fix."""
    edge_max_shift_px: float = 40.0
    """How far a refinement may move the projection of the part's centre, in
    original image pixels. A correction, not a search: anything larger means the
    solve was not close to begin with, and sliding onto distant edges is exactly
    the failure being guarded against."""
    edge_min_improvement: float = 0.08
    """Fractional drop in chamfer cost below which a refinement is not worth
    accepting. Keeps noise-level wobble from being recorded as a change."""
    edge_max_reproj_worsening_px: float = 0.25
    """How much a refinement may raise that image's own mark reprojection RMS
    before it is rejected outright.

    The independent check. Chamfer cost is the thing being minimised, so of
    course it goes down; the clicked marks in that photo were not part of the
    objective and have every right to veto the result."""
    gauge_weight: float = 1e-3
    """Weak prior holding the solution near its initialisation in the 7 gauge
    directions. Too small to bias anything, large enough to keep the normal
    equations full rank when the user has supplied no constraints."""


UNIT_SCALES: dict[str, float] = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
"""World metres per unit, for the units a drawing is likely to be dimensioned in."""


@dataclass
class Project:
    name: str = "untitled"
    images: list[ImageView] = field(default_factory=list)
    groups: dict[str, CameraGroup] = field(default_factory=dict)
    points: list[TiePoint] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    templates: dict[str, ObjectTemplate] = field(default_factory=dict)
    instances: list[ObjectInstance] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
    nominal_units: str = "mm"
    """Units the point nominals are typed in. Drawings are rarely in metres."""
    nominal_scale: float = 0.001
    """World metres per nominal unit. Kept explicitly beside ``nominal_units``,
    the same way a template does, so an odd scale survives a round trip."""
    calibration_nominals: list[dict[str, Any]] | None = None
    """Features a calibration import offered as nominals, as
    ``sensors.NominalCandidate.to_dict()``.

    Stored with the project so the file does not have to be re-imported every
    session, and so the pick-list a point was chosen from is still there to be
    audited later. Withheld candidates are kept in the list rather than filtered
    out, with their reason, because "this feature exists and here is why you may
    not use it" is more useful than its absence."""
    origin_uncertainty: dict[str, Any] | None = None
    """Last measured spread of the part-space origin, as
    ``uncertainty.OriginUncertainty.to_dict()``. Stored rather than recomputed
    because it costs a few dozen re-solves, and carries the fingerprint of the
    data it was measured on so a later edit shows as stale instead of silently
    describing a project that no longer exists."""
    path: Path | None = None

    # ---------------------------------------------------------------- lookups

    def image(self, image_id: str) -> ImageView | None:
        return next((im for im in self.images if im.id == image_id), None)

    def point(self, point_id: str) -> TiePoint | None:
        return next((p for p in self.points if p.id == point_id), None)

    def nominal_constraints(self) -> list["AxisLockConstraint"]:
        """Every point nominal, expanded into the axis locks it already means.

        Deliberately not a new kind of residual. "This point is at X=120" is
        exactly what an axis lock says, so expanding nominals into locks means
        the nominals inherit the whole existing apparatus for free: the datum
        residuals, the asked-versus-achieved report, the sigmas-off warning, the
        scale-is-metric detection. A separate parallel implementation would have
        had to re-earn all of that, and would drift out of step with it.

        Regenerated on demand rather than stored, so editing a nominal cannot
        leave a stale lock behind.
        """
        out: list[AxisLockConstraint] = []
        for pt in self.points:
            if not pt.nominal or len(pt.nominal) != 3:
                continue
            sigma = pt.nominal_sigma
            if sigma is None:
                sigma = self.settings.sigma_nominal
            sigma_world = max(float(sigma) * self.nominal_scale, 1e-9)
            for axis in range(3):
                value = pt.nominal[axis]
                if value is None:
                    continue
                out.append(
                    AxisLockConstraint(
                        point=pt.id,
                        axis=axis,
                        value=float(value) * self.nominal_scale,
                        sigma=sigma_world,
                        source="nominal",
                    )
                )
        return out

    def all_constraints(self) -> list["Constraint"]:
        """What the solver should actually read: the constraints someone typed,
        plus the ones the nominals imply. Explicit ones come first so their
        indices in the report stay stable as nominals come and go."""
        return list(self.constraints) + self.nominal_constraints()

    def in_part_space(self) -> bool:
        """True once the nominals pin all seven gauge freedoms.

        Three non-collinear nominal'd points is the real test, but counting
        them is not enough -- three points in a row leave a rotation about that
        row free. Collinearity is checked against the nominals themselves,
        since those are known before anything is solved."""
        pts = [np.array(p.nominal, dtype=float) for p in self.points if p.nominal]
        if len(pts) < 3:
            return False
        P = np.array(pts) - np.mean(pts, axis=0)
        return bool(np.linalg.matrix_rank(P, tol=1e-6) >= 2)

    def instance(self, instance_id: str) -> ObjectInstance | None:
        return next((o for o in self.instances if o.id == instance_id), None)

    # ------------------------------------------------------------- the part

    def part_instance(self) -> ObjectInstance | None:
        """The instance designated as the part itself, if any."""
        return next((o for o in self.instances if o.is_part), None)

    def part_template(self) -> ObjectTemplate | None:
        inst = self.part_instance()
        return None if inst is None else self.template_of(inst)

    def pin_part(self) -> ObjectInstance | None:
        """Hold the part at the origin.

        Called everywhere a pose could have been disturbed -- after loading,
        before and after a solve -- because the identity is not a starting guess
        that refinement improves on. It is the definition of the coordinate
        frame the nominals built, and anything that moved the part off it would
        be re-expressing the answer in coordinates that are no longer the
        drawing's.
        """
        inst = self.part_instance()
        if inst is None:
            return None
        inst.R = np.eye(3)
        inst.t = np.zeros(3)
        inst.locked = True
        inst.manual = False
        return inst

    def set_part(self, template_id: str, label: str = "") -> ObjectInstance:
        """Designate a template as the part, and pin its placement at the origin.

        Reuses the existing part instance when there is one, so swapping the CAD
        file for a corrected export does not leave two parts behind. Any anchor
        marks that were made against it are deliberately left alone: they are
        harmless (the part takes no pose from them) and deleting a user's marks
        on a flag change would be a nasty surprise.
        """
        if template_id not in self.templates:
            raise KeyError(f"unknown template {template_id!r}")
        for other in self.instances:
            other.is_part = False
        inst = next((o for o in self.instances if o.id == "part"), None)
        if inst is None:
            inst = ObjectInstance(id="part", template=template_id)
            self.instances.append(inst)
        inst.template = template_id
        inst.label = label or inst.label or (self.templates[template_id].name or "part")
        inst.is_part = True
        self.pin_part()
        return inst

    def clear_part(self) -> bool:
        """Stop treating anything as the part. Returns whether something changed.

        The instance goes rather than reverting to an ordinary placed object: a
        pose of exactly identity is not a placement anybody made, and leaving it
        in the objects list would read as a body solved at the origin.
        """
        inst = self.part_instance()
        if inst is None:
            return False
        self.instances = [o for o in self.instances if o is not inst]
        for im in self.images:
            for key in [k for k in im.anchor_obs if split_anchor_key(k)[0] == inst.id]:
                del im.anchor_obs[key]
        return True

    def placed_object_instances(self) -> list[ObjectInstance]:
        """Instances that are placements to be solved -- everything but the part.

        What the solver's object machinery should iterate. The part is not a
        placement: giving it to ``fit_instances`` would have it looking for
        anchors that are not there and then clearing the one pose that is known
        for certain.
        """
        return [o for o in self.instances if not o.is_part]

    def part_unit_ratio(self) -> float | None:
        """How the part model's declared units compare with the nominals'.

        1.0 when they agree. Anything else is reported rather than applied: a
        model declared in metres against a drawing dimensioned in millimetres is
        off by exactly 1000, and silently rescaling it would make every residual
        in the fit report look immaculate while the wireframe landed a thousand
        times too large in every photograph.
        """
        template = self.part_template()
        if template is None:
            return None
        if self.nominal_scale <= 0:
            return None
        return float(template.unit_scale) / float(self.nominal_scale)

    def template_of(self, instance: ObjectInstance) -> ObjectTemplate | None:
        return self.templates.get(instance.template)

    def next_instance_id(self) -> str:
        used = {o.id for o in self.instances}
        i = 0
        while f"obj{i}" in used:
            i += 1
        return f"obj{i}"

    def next_template_id(self) -> str:
        i = 0
        while f"tpl{i}" in self.templates:
            i += 1
        return f"tpl{i}"

    # -------------------------------------------------------- known objects

    def anchor_world_points(self) -> dict[str, np.ndarray]:
        """Every valid anchor observation key mapped to its model-local position
        in world units.

        Keys naming a missing instance, a missing template or a deleted anchor
        are simply absent, which is how stale marks get ignored rather than
        crashing a solve.

        The part is left out. Its pose is not measured from anchors -- it is the
        origin by definition -- so an anchor mark on the part has nothing to
        contribute to a placement and would instead be read as evidence about
        the cameras. The marks are not deleted, only ignored: see
        :meth:`part_anchor_keys`.
        """
        out: dict[str, np.ndarray] = {}
        for inst in self.placed_object_instances():
            template = self.template_of(inst)
            if template is None:
                continue
            for name, xyz in template.anchor_points().items():
                out[anchor_key(inst.id, name)] = xyz
        return out

    def part_anchor_keys(self) -> set[str]:
        """Observation keys that name the part.

        Exists so the stale-mark sweep can tell "this anchor's object is gone"
        from "this anchor belongs to the part and is deliberately inert". The
        two look identical from ``anchor_world_points`` alone, and treating the
        second as the first would quietly delete a user's clicks.
        """
        inst = self.part_instance()
        if inst is None:
            return set()
        return {
            key
            for im in self.images
            for key in im.anchor_obs
            if split_anchor_key(key)[0] == inst.id
        }

    def anchor_observations_of(self, key: str) -> list[tuple[ImageView, np.ndarray]]:
        return [
            (im, np.array(im.anchor_obs[key], dtype=float))
            for im in self.active_images()
            if key in im.anchor_obs
        ]

    def merged_observations(self, image: ImageView, valid: set[str] | None = None) -> dict[str, list[float]]:
        """Tie points and anchor marks as one table.

        The solver's pruning, connectivity and linear-initialisation stages all
        want "every 2D mark in this photo that refers to a 3D position", with no
        interest in which kind it is. Giving them one table is what lets known
        objects reuse those stages untouched instead of duplicating them.
        """
        merged = dict(image.observations)
        for key, uv in image.anchor_obs.items():
            if valid is None or key in valid:
                merged[key] = uv
        return merged

    def instance_center(self, instance: ObjectInstance) -> np.ndarray | None:
        """World position of a placed instance, taken as its anchor centroid.

        The anchor centroid rather than the mesh's: it needs no mesh file to be
        loadable, and it is the part of the object the solve actually saw.
        """
        template = self.template_of(instance)
        if template is None or instance.R is None or instance.t is None:
            return None
        points = list(template.anchor_points().values())
        local = np.mean(points, axis=0) if points else np.zeros(3)
        return instance.R @ local + instance.t

    def instance_optical(self, instance: ObjectInstance) -> tuple[np.ndarray, np.ndarray] | None:
        """World-space entrance pupil and unit optical axis of a placed instance,
        or None if its template declares no optical frame.

        This is the difference between knowing where a camera's shell is and
        knowing where the camera looks, which is the only version of the answer
        a camera array is actually after.
        """
        template = self.template_of(instance)
        if template is None or template.optical is None:
            return None
        if instance.R is None or instance.t is None:
            return None
        center_local = np.array(template.optical.center, dtype=float) * template.unit_scale
        forward = np.array(template.optical.forward, dtype=float)
        norm = float(np.linalg.norm(forward))
        if norm < 1e-12:
            return None
        return instance.R @ center_local + instance.t, instance.R @ (forward / norm)

    def placed_instances(self) -> list[ObjectInstance]:
        return [o for o in self.instances if o.R is not None and o.t is not None]

    def prune_stale_anchor_obs(self) -> int:
        """Drop anchor marks whose instance, template or anchor has gone. Returns
        how many were removed."""
        valid = set(self.anchor_world_points()) | self.part_anchor_keys()
        removed = 0
        for im in self.images:
            stale = [k for k in im.anchor_obs if k not in valid]
            for k in stale:
                del im.anchor_obs[k]
            removed += len(stale)
        return removed

    def group_of(self, image: ImageView) -> CameraGroup:
        if image.group not in self.groups:
            self.groups[image.group] = CameraGroup(id=image.group)
        return self.groups[image.group]

    def active_images(self) -> list[ImageView]:
        return [im for im in self.images if im.enabled]

    def observations_of(self, point_id: str) -> list[tuple[ImageView, np.ndarray]]:
        out = []
        for im in self.active_images():
            if point_id in im.observations:
                out.append((im, np.array(im.observations[point_id], dtype=float)))
        return out

    def next_point_id(self) -> str:
        used = {p.id for p in self.points}
        i = 0
        while f"p{i}" in used:
            i += 1
        return f"p{i}"

    def next_image_id(self) -> str:
        used = {im.id for im in self.images}
        i = 0
        while f"img{i}" in used:
            i += 1
        return f"img{i}"

    def resolve_path(self, image: ImageView) -> Path:
        p = Path(image.path)
        if p.is_absolute() or self.path is None:
            return p
        return (self.path.parent / p).resolve()

    def resolve_mesh_path(self, template: ObjectTemplate) -> Path | None:
        if not template.mesh_path:
            return None
        p = Path(template.mesh_path)
        if p.is_absolute() or self.path is None:
            return p
        return (self.path.parent / p).resolve()

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "name": self.name,
            "origin_uncertainty": self.origin_uncertainty,
            "calibration_nominals": self.calibration_nominals,
            "settings": asdict(self.settings),
            "groups": {gid: asdict(g) for gid, g in self.groups.items()},
            "images": [
                {
                    "id": im.id,
                    "path": im.path,
                    "width": im.width,
                    "height": im.height,
                    "group": im.group,
                    "enabled": im.enabled,
                    "axis_lines": [asdict(ln) for ln in im.axis_lines],
                    "observations": im.observations,
                    "anchor_obs": im.anchor_obs,
                    "R": im.R.tolist() if im.R is not None else None,
                    "t": im.t.tolist() if im.t is not None else None,
                    "notes": im.notes,
                    "edge_refined": im.edge_refined,
                    "R_unrefined": im.R_unrefined.tolist() if im.R_unrefined is not None else None,
                    "t_unrefined": im.t_unrefined.tolist() if im.t_unrefined is not None else None,
                }
                for im in self.images
            ],
            "points": [
                {
                    "id": p.id,
                    "label": p.label,
                    "xyz": p.xyz.tolist() if p.xyz is not None else None,
                    "nominal": p.nominal,
                    "nominal_sigma": p.nominal_sigma,
                    "nominal_source": p.nominal_source,
                    "uncertain": p.uncertain,
                }
                for p in self.points
            ],
            "nominal_units": self.nominal_units,
            "nominal_scale": self.nominal_scale,
            "constraints": [asdict(c) for c in self.constraints],
            "templates": {
                tid: {
                    "id": t.id,
                    "name": t.name,
                    "mesh_path": t.mesh_path,
                    "units": t.units,
                    "unit_scale": t.unit_scale,
                    "crease_angle": t.crease_angle,
                    "anchors": [asdict(a) for a in t.anchors],
                    "optical": asdict(t.optical) if t.optical is not None else None,
                }
                for tid, t in self.templates.items()
            },
            "instances": [
                {
                    "id": o.id,
                    "template": o.template,
                    "label": o.label,
                    "locked": o.locked,
                    "manual": o.manual,
                    "pose_in_part_space": o.pose_in_part_space,
                    "calibrated": o.calibrated,
                    "is_part": o.is_part,
                    "R": o.R.tolist() if o.R is not None else None,
                    "t": o.t.tolist() if o.t is not None else None,
                }
                for o in self.instances
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: Path | None = None) -> "Project":
        version = d.get("version", FORMAT_VERSION)
        if version > FORMAT_VERSION:
            raise ValueError(
                f"project file is format version {version}, this build understands {FORMAT_VERSION}"
            )

        settings = Settings(**{k: v for k, v in d.get("settings", {}).items() if k in Settings.__annotations__})
        groups = {
            gid: CameraGroup(**{k: v for k, v in g.items() if k in CameraGroup.__annotations__})
            for gid, g in d.get("groups", {}).items()
        }

        images = []
        for raw in d.get("images", []):
            im = ImageView(
                id=raw["id"],
                path=raw["path"],
                width=int(raw["width"]),
                height=int(raw["height"]),
                group=raw.get("group", "cam0"),
                enabled=bool(raw.get("enabled", True)),
                axis_lines=[
                    AxisLine(
                        axis=int(l["axis"]),
                        p0=list(l["p0"]),
                        p1=list(l["p1"]),
                        uncertain=bool(l.get("uncertain", False)),
                    )
                    for l in raw.get("axis_lines", [])
                ],
                observations={k: list(v) for k, v in raw.get("observations", {}).items()},
                anchor_obs={k: list(v) for k, v in raw.get("anchor_obs", {}).items()},
                notes=list(raw.get("notes", [])),
                edge_refined=bool(raw.get("edge_refined", False)),
            )
            if raw.get("R") is not None:
                im.R = np.array(raw["R"], dtype=float)
            if raw.get("t") is not None:
                im.t = np.array(raw["t"], dtype=float)
            if raw.get("R_unrefined") is not None:
                im.R_unrefined = np.array(raw["R_unrefined"], dtype=float)
            if raw.get("t_unrefined") is not None:
                im.t_unrefined = np.array(raw["t_unrefined"], dtype=float)
            images.append(im)

        points = []
        for raw in d.get("points", []):
            pt = TiePoint(
                id=raw["id"],
                label=raw.get("label", ""),
                uncertain=bool(raw.get("uncertain", False)),
            )
            if raw.get("xyz") is not None:
                pt.xyz = np.array(raw["xyz"], dtype=float)
            nominal = raw.get("nominal")
            if nominal is not None and len(nominal) == 3:
                pt.nominal = [None if v is None else float(v) for v in nominal]
            if raw.get("nominal_sigma") is not None:
                pt.nominal_sigma = float(raw["nominal_sigma"])
            pt.nominal_source = str(raw.get("nominal_source", "") or "")
            points.append(pt)

        constraints: list[Constraint] = []
        for raw in d.get("constraints", []):
            kind = raw.get("type")
            if kind == "axis_lock":
                constraints.append(
                    AxisLockConstraint(
                        point=raw["point"],
                        axis=int(raw["axis"]),
                        value=float(raw.get("value", 0.0)),
                        sigma=float(raw.get("sigma", 0.01)),
                    )
                )
            elif kind == "distance":
                constraints.append(
                    DistanceConstraint(
                        point_a=raw["point_a"],
                        point_b=raw["point_b"],
                        value=float(raw["value"]),
                        sigma=float(raw.get("sigma", 0.01)),
                    )
                )
            else:
                raise ValueError(f"unknown constraint type {kind!r}")

        templates: dict[str, ObjectTemplate] = {}
        for tid, raw in (d.get("templates") or {}).items():
            optical_raw = raw.get("optical")
            templates[tid] = ObjectTemplate(
                id=raw.get("id", tid),
                name=raw.get("name", ""),
                mesh_path=raw.get("mesh_path"),
                units=raw.get("units", "mm"),
                unit_scale=float(raw.get("unit_scale", 0.001)),
                crease_angle=float(raw.get("crease_angle", FEATURE_ANGLE_DEG)),
                anchors=[
                    ObjectAnchor(name=clean_anchor_name(a["name"]), xyz=[float(v) for v in a["xyz"]])
                    for a in raw.get("anchors", [])
                ],
                optical=None
                if not optical_raw
                else OpticalFrame(
                    center=[float(v) for v in optical_raw["center"]],
                    forward=[float(v) for v in optical_raw.get("forward", [0.0, 0.0, 1.0])],
                    up=[float(v) for v in optical_raw.get("up", [0.0, -1.0, 0.0])],
                    fov_deg=None
                    if optical_raw.get("fov_deg") is None
                    else float(optical_raw["fov_deg"]),
                ),
            )

        instances: list[ObjectInstance] = []
        for raw in d.get("instances", []):
            inst = ObjectInstance(
                id=raw["id"],
                template=raw["template"],
                label=raw.get("label", ""),
                locked=bool(raw.get("locked", False)),
                manual=bool(raw.get("manual", False)),
                pose_in_part_space=bool(raw.get("pose_in_part_space", False)),
                calibrated=str(raw.get("calibrated", "") or ""),
                is_part=bool(raw.get("is_part", False)),
            )
            if raw.get("R") is not None:
                inst.R = np.array(raw["R"], dtype=float)
            if raw.get("t") is not None:
                inst.t = np.array(raw["t"], dtype=float)
            instances.append(inst)

        # At most one part, whatever the file says, and pinned at the origin.
        # A file is user-editable and a second ``is_part`` would otherwise make
        # every later lookup depend on list order.
        seen_part = False
        for inst in instances:
            if not inst.is_part:
                continue
            if seen_part:
                inst.is_part = False
            seen_part = True

        project = cls(
            name=d.get("name", "untitled"),
            images=images,
            groups=groups,
            points=points,
            constraints=constraints,
            templates=templates,
            instances=instances,
            settings=settings,
            nominal_units=d.get("nominal_units", "mm"),
            nominal_scale=float(d.get("nominal_scale", 0.001)),
            origin_uncertainty=d.get("origin_uncertainty") or None,
            calibration_nominals=d.get("calibration_nominals") or None,
            path=path,
        )
        project.pin_part()
        return project

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        # write to a sibling temp file first so an interrupted save can't eat the project
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path=path)
