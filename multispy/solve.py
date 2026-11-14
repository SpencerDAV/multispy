"""The solve pipeline.

Three stages, in order of how much they trust the user's clicking:

1. **Per-image calibration.** Control lines give each image a focal length and a
   *world-to-camera rotation*, independently of every other image. This is the
   unusual property the whole tool is built on: vanishing points hand us global
   rotations with no match graph, no chaining, no drift.

2. **Linear initialisation.** With rotations and intrinsics fixed, solving for
   camera translations and 3D point positions is a *linear* least-squares
   problem -- no initial pair to pick, no incremental mapper to wander off, no
   local minima. It needs roughly two shared tie points per image pair, versus
   the dozens of well-spread matches an SfM pipeline wants.

3. **Joint bundle adjustment.** Everything refines together: rotations,
   translations, shared intrinsics, point positions, under both tie-point
   reprojection error and control-line residuals. This is the stage that fixes
   fSpy's fragility -- a sloppy vanishing point in one image gets outvoted by
   tie points and by the other images sharing its focal length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from .geometry import (
    AXIS_NAMES,
    Intrinsics,
    calibrate_image,
    similarity_fit,
    so3_exp,
    so3_log,
)
from .project import (
    AxisLockConstraint,
    CameraGroup,
    Constraint,
    DistanceConstraint,
    ImageView,
    Project,
    anchor_key,
    split_anchor_key,
)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _huberize(r: np.ndarray, k: float) -> np.ndarray:
    """Huber kernel expressed as ``sign(r) * sqrt(rho(r))``.

    Folding the robust kernel into the residual lets a plain least-squares
    solver carry it, which matters because scipy's own ``loss=`` applies to
    *every* residual -- including hard constraints, which must not be
    downweighted for being badly violated at the start.
    """
    a = np.abs(r)
    return np.where(a <= k, r, np.sign(r) * np.sqrt(np.maximum(k * (2.0 * a - k), 0.0)))


@dataclass
class ImageReport:
    id: str
    positioned: bool = False
    n_observations: int = 0
    n_culled: int = 0
    """Marks in this photo dropped as outliers before the final pass. They are
    still drawn and still listed in ``residual_lines``, flagged, because the
    user has to see the bad arrow to know what to go and re-click."""
    rms_px: float = 0.0
    max_px: float = 0.0
    worst_point: str | None = None
    line_rms_px: float = 0.0
    focal_px: float | None = None
    focal_sigma_pct: float | None = None
    """Predicted 1-sigma uncertainty on the focal length from this image's
    control lines alone, as a percentage. Available before any solve."""
    observability: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class PointReport:
    id: str
    n_observations: int = 0
    rms_px: float = 0.0
    max_px: float = 0.0
    triangulated: bool = False


@dataclass
class ConstraintReport:
    """What a constraint asked for, and what the solve actually delivered.

    A constraint is the one thing in the whole pipeline the user states as fact
    rather than clicks as evidence, so it is the one place where "did the
    reconstruction agree with me?" has a definite answer. Reporting only the
    reprojection RMS hides that: a datum lock the solve misses by a millimetre
    costs almost nothing in pixels and is invisible in every other number here.
    """

    index: int
    type: str
    description: str
    source: str = "constraint"
    """``"nominal"`` for the datum locks a point's nominal XYZ generated."""
    points: list[str] = field(default_factory=list)
    applied: bool = True
    """False when the constraint's points were not solved, so it did nothing."""
    target: float = 0.0
    achieved: float = 0.0
    error: float = 0.0
    """achieved - target, in world units."""
    sigma: float = 0.0
    sigmas_off: float = 0.0
    note: str = ""


@dataclass
class InstanceReport:
    """How one known object's placement came out."""

    id: str
    template: str = ""
    label: str = ""
    placed: bool = False
    """A pose was measured from anchors. Deliberately false for a hand-placed
    body: ``shape_rms`` defaults to 0 and ``implied_scale`` to 1, so claiming a
    manual pose was "placed" would report a flawless fit for something nothing
    ever checked."""
    manual: bool = False
    """This body carries a pose someone set by hand, with no anchors behind it."""
    n_anchors_marked: int = 0
    n_anchors_used: int = 0
    """Anchors seen in 2+ photos. Three is the floor for a determined pose: two
    leave the body free to spin about the line joining them."""
    n_observations: int = 0
    n_culled: int = 0
    """Anchor marks on this body dropped as outliers before the final pass."""
    rms_px: float = 0.0
    max_px: float = 0.0
    sigma_px: float = 0.0
    """The uncertainty this body was actually weighted with in the final pass.

    Starts at ``Settings.sigma_anchor_px`` and is raised to match the body's own
    residuals whenever they are worse. That is the whole mechanism by which a
    mis-marked object stops pulling the cameras around: it is not excluded, it
    simply stops claiming an accuracy it has not demonstrated."""
    shape_rms: float = 0.0
    """How far the triangulated anchors sit from the model's own geometry, in
    world units. This is the diagnostic the free-point workflow cannot give you:
    it compares the marked points against a shape that is known in advance, so a
    mis-clicked corner shows up as a shape error rather than hiding inside an
    otherwise plausible reconstruction."""
    implied_scale: float = 1.0
    """Size this body came out relative to the scene's agreed scale. Should be
    1.0; anything else means this object is not the shape the model claims, or an
    anchor is marked on the wrong feature."""
    center: list[float] | None = None
    optical_center: list[float] | None = None
    optical_forward: list[float] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RigReport:
    """The layout of the placed objects relative to each other.

    When the objects being placed are the cameras of an array, this is the whole
    deliverable: the baselines and relative angles are what you came for, and
    they are more useful than the absolute poses because they do not depend on
    where the arbitrary world origin ended up.
    """

    n_instances: int = 0
    units: str = "m"
    extent: float = 0.0
    pairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class SolveReport:
    ok: bool = False
    stage: str = "not started"
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_images_positioned: int = 0
    n_points_triangulated: int = 0
    n_observations: int = 0
    n_culled: int = 0
    """Clicked marks dropped as outliers between the two bundle passes."""
    rms_reprojection_px: float = 0.0
    rms_line_px: float = 0.0
    init_condition: float = 0.0
    focal_sigma_pct: float | None = None
    """Combined focal uncertainty across every calibrated image in the group."""
    images: dict[str, ImageReport] = field(default_factory=dict)
    points: dict[str, PointReport] = field(default_factory=dict)
    instances: dict[str, InstanceReport] = field(default_factory=dict)
    constraints: list[ConstraintReport] = field(default_factory=list)
    constraint_worst: float = 0.0
    """Largest constraint violation, in world units."""
    datum_scale: float | None = None
    """Scale the stated figures imply, against the scale a known object set,
    measured at initialisation before the refinement trades the two off. 1.0
    means they agree. Set only when an object set the scale, since otherwise the
    constraints *are* the scale and this would be 1.0 by construction."""
    rig: RigReport = field(default_factory=RigReport)
    residual_lines: list[dict[str, Any]] = field(default_factory=list)
    """Per-observation observed->reprojected pairs, for drawing in the UI."""
    focals: dict[str, float] = field(default_factory=dict)
    scale_is_metric: bool = False
    scale_source: str = "arbitrary"
    part_space: bool = False
    """True when point nominals pinned the frame, so the solved coordinates are
    the part's own coordinates and a CAD model of the part belongs at the
    identity. Distinct from ``scale_is_metric``: a distance constraint gives
    real units but leaves the origin and orientation arbitrary."""
    n_nominals: int = 0
    """Points carrying a nominal XYZ."""
    part_fit: dict[str, Any] | None = None
    """How well the part's own CAD model agrees with the reconstruction, when a
    part is designated and the solve reached part space.

    Carried in the solve report rather than fetched separately so the agreement
    is reported at the same moment as the numbers that produced it. ``None``
    means no part was designated or its model could not be read -- which is not
    a failed fit and is not reported as one."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "messages": self.messages,
            "warnings": self.warnings,
            "n_images_positioned": self.n_images_positioned,
            "n_points_triangulated": self.n_points_triangulated,
            "n_observations": self.n_observations,
            "n_culled": self.n_culled,
            "rms_reprojection_px": self.rms_reprojection_px,
            "rms_line_px": self.rms_line_px,
            "init_condition": self.init_condition,
            "focal_sigma_pct": self.focal_sigma_pct,
            "scale_is_metric": self.scale_is_metric,
            "scale_source": self.scale_source,
            "part_space": self.part_space,
            "n_nominals": self.n_nominals,
            "part_fit": self.part_fit,
            "focals": self.focals,
            "images": {k: vars(v) for k, v in self.images.items()},
            "points": {k: vars(v) for k, v in self.points.items()},
            "instances": {k: vars(v) for k, v in self.instances.items()},
            "constraints": [vars(c) for c in self.constraints],
            "constraint_worst": self.constraint_worst,
            "datum_scale": self.datum_scale,
            "rig": self.rig.to_dict(),
            "residual_lines": self.residual_lines,
        }


# --------------------------------------------------------------------------
# stage 1 -- per-image calibration from control lines
# --------------------------------------------------------------------------


def calibrate_all(project: Project, report: SolveReport) -> dict[str, Intrinsics]:
    """Calibrate every image from its control lines, then agree on one focal
    length per camera group and re-derive the rotations using it.

    Re-deriving matters: the group focal (a median over N images) is a far better
    estimate than any single image's, and a rotation built from a bad focal is
    tilted even when the vanishing points are perfect.
    """
    per_image: dict[str, Intrinsics] = {}
    per_group_focals: dict[str, list[float]] = {}

    for im in project.active_images():
        rep = report.images.setdefault(im.id, ImageReport(id=im.id))
        segs = im.axis_segments()
        usable = {a: s for a, s in segs.items() if len(s) >= 2}
        if len(usable) < 2:
            rep.notes.append(
                f"{len(usable)} axis/axes with 2+ control lines; need 2 axes to calibrate"
            )
            continue
        try:
            calib = calibrate_image(
                usable,
                im.width,
                im.height,
                solve_principal_point=project.settings.solve_principal_point,
                sigma_px=project.settings.sigma_line_px,
            )
        except ValueError as exc:
            rep.notes.append(str(exc))
            continue
        rep.notes.extend(calib.warnings)
        if calib.focal_relative_sigma is not None:
            rep.focal_sigma_pct = 100.0 * calib.focal_relative_sigma
            if calib.observability and calib.observability.best:
                rep.observability = calib.observability.best.verdict
        per_image[im.id] = calib.intrinsics
        per_group_focals.setdefault(im.group, []).append(
            (calib.intrinsics.f, calib.focal_relative_sigma)
        )
        im.R = calib.R

    # settle one focal length per group, weighting each image by how well its own
    # control lines actually pin the focal down
    for gid, entries in per_group_focals.items():
        group = project.groups.setdefault(gid, CameraGroup(id=gid))
        focals = np.array([e[0] for e in entries], dtype=float)
        sigmas = np.array([e[1] if e[1] is not None else 1.0 for e in entries], dtype=float)
        weights = 1.0 / np.maximum(sigmas, 1e-6) ** 2
        combined = float(1.0 / math.sqrt(float(weights.sum())))
        if report.focal_sigma_pct is None or combined * 100.0 > report.focal_sigma_pct:
            report.focal_sigma_pct = combined * 100.0

        if not (group.lock_focal and group.f is not None):
            group.f = float(np.sum(weights * focals) / np.sum(weights))

        if combined > 0.08:
            report.warnings.append(
                f"camera group {gid}: the focal length is not well observable from these control "
                f"lines -- combining all {len(entries)} image(s) still leaves it uncertain to "
                f"+-{combined * 100:.0f}%. Anything downstream of the focal length (camera "
                f"positions, point depths) inherits that. Either add longer control lines on the "
                f"limiting axis, or lock the focal to a value you trust."
            )
        elif len(entries) > 1:
            best_single = float(np.min(sigmas))
            if best_single > combined * 1.3:
                report.messages.append(
                    f"camera group {gid}: sharing one focal across {len(entries)} image(s) "
                    f"tightened it from +-{best_single * 100:.1f}% (best single image) to "
                    f"+-{combined * 100:.1f}%."
                )

        if len(focals) > 2:
            spread = (np.percentile(focals, 90) - np.percentile(focals, 10)) / group.f
            if spread > 0.3:
                report.warnings.append(
                    f"camera group {gid}: per-image focal estimates spread {spread * 100:.0f}% "
                    "(10th-90th percentile), though they are weighted by reliability. A wide "
                    "spread usually means some control lines are on non-parallel edges."
                )

    # re-derive rotations with the agreed group focal
    intrinsics: dict[str, Intrinsics] = {}
    for im in project.active_images():
        if im.id not in per_image:
            continue
        group = project.group_of(im)
        cam = Intrinsics(
            f=group.f if group.f is not None else per_image[im.id].f,
            cx=group.cx if group.cx is not None else im.width / 2.0,
            cy=group.cy if group.cy is not None else im.height / 2.0,
            k1=group.k1,
        )
        try:
            calib = calibrate_image(
                {a: s for a, s in im.axis_segments().items() if len(s) >= 2},
                im.width,
                im.height,
                fixed_focal=cam.f,
                sigma_px=project.settings.sigma_line_px,
            )
            im.R = calib.R
        except ValueError:
            pass
        intrinsics[im.id] = cam
        report.images[im.id].focal_px = cam.f

    return intrinsics


# --------------------------------------------------------------------------
# stage 2 -- linear translations and structure, with rotations fixed
# --------------------------------------------------------------------------


Table = dict[str, dict[str, list[float]]]
"""``{image_id: {track_id: [u, v]}}``.

A *track* is anything with one 3D position observed across photos: a free tie
point, or an anchor on a known object (keyed ``"<instance>/<anchor>"``). The
stages below do not care which, and that is deliberate -- marking a CAD model's
corner constrains a camera in exactly the way clicking a free feature does, so
it gets to reuse the same pruning, connectivity and linear initialisation rather
than a parallel implementation of all three.
"""


def _observation_tables(project: Project, image_ids: set[str], valid_anchors: set[str]) -> Table:
    return {
        iid: project.merged_observations(project.image(iid), valid_anchors) for iid in image_ids
    }


def _prune_to_solvable(tables: Table, track_ids: set[str]) -> tuple[list[str], list[str]]:
    """Iteratively drop what cannot be solved, so the user gets told what to click
    rather than a LinAlgError.

    A track needs 2+ images to triangulate; an image needs 2+ tracks to fix its
    3 translation unknowns (each observation contributes 2 equations). Dropping
    either can starve the other, so this iterates to a fixed point.
    """
    keep_images = set(tables)
    keep_tracks = set(track_ids)

    while True:
        seen_in: dict[str, list[str]] = {}
        for iid in keep_images:
            for tid in tables[iid]:
                if tid in keep_tracks:
                    seen_in.setdefault(tid, []).append(iid)

        drop_tracks = {t for t in keep_tracks if len(seen_in.get(t, [])) < 2}
        if drop_tracks:
            keep_tracks -= drop_tracks
            continue

        drop_images = {
            iid
            for iid in keep_images
            if sum(1 for tid in tables[iid] if tid in keep_tracks) < 2
        }
        if drop_images:
            keep_images -= drop_images
            continue
        break

    return sorted(keep_images), sorted(keep_tracks)


def _largest_connected_component(
    tables: Table, image_ids: list[str], track_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Images only share a coordinate frame if tracks connect them. Separate
    components each float with their own translation and scale, so solve the
    biggest one and report the rest."""
    adjacency: dict[str, set[str]] = {iid: set() for iid in image_ids}
    track_set = set(track_ids)
    by_point: dict[str, list[str]] = {}
    for iid in image_ids:
        for tid in tables[iid]:
            if tid in track_set:
                by_point.setdefault(tid, []).append(iid)
    for pid, viewers in by_point.items():
        for a in viewers:
            for b in viewers:
                if a != b:
                    adjacency[a].add(b)

    unvisited = set(image_ids)
    components: list[list[str]] = []
    while unvisited:
        seed = unvisited.pop()
        comp, stack = [seed], [seed]
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if nxt in unvisited:
                    unvisited.remove(nxt)
                    comp.append(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))

    components.sort(key=len, reverse=True)
    main = components[0] if components else []
    orphans = [iid for comp in components[1:] for iid in comp]
    main_set = set(main)
    kept_tracks = [
        tid for tid in track_ids if sum(1 for iid in by_point.get(tid, []) if iid in main_set) >= 2
    ]
    return main, kept_tracks, orphans


def initialize_structure(
    project: Project,
    intrinsics: dict[str, Intrinsics],
    image_ids: list[str],
    track_ids: list[str],
    tables: Table,
    report: SolveReport,
) -> dict[str, np.ndarray] | None:
    """Linear solve for camera translations and point positions, rotations fixed.

    For each observation, the back-projected ray ``xhat`` must be parallel to the
    camera-space point, so ``xhat x (R X + t) = 0``. That is linear in both ``t``
    and ``X``.

    The homogeneous system has a spurious solution where every point and every
    camera centre collapses to one location, which is also exactly the global
    translation gauge. Fixing the first camera at the origin removes both at
    once, leaving a one-dimensional null space: the answer, up to scale.
    """
    img_index = {iid: k for k, iid in enumerate(image_ids)}
    pt_index = {pid: k for k, pid in enumerate(track_ids)}
    n_img, n_pts = len(image_ids), len(track_ids)
    if n_img < 2 or n_pts < 2:
        report.messages.append("need at least 2 images and 2 shared tie points to place cameras")
        return None

    n_t = 3 * (n_img - 1)  # camera 0 is pinned at the origin
    n_cols = n_t + 3 * n_pts

    rows: list[np.ndarray] = []
    for iid in image_ids:
        im = project.image(iid)
        cam = intrinsics[iid]
        Kinv = cam.K_inv()
        i = img_index[iid]
        for pid, uv in tables[iid].items():
            if pid not in pt_index:
                continue
            xhat = Kinv @ np.array([uv[0], uv[1], 1.0])
            xhat = xhat / max(np.linalg.norm(xhat), 1e-12)
            S = np.array(
                [
                    [0.0, -xhat[2], xhat[1]],
                    [xhat[2], 0.0, -xhat[0]],
                    [-xhat[1], xhat[0], 0.0],
                ]
            )
            block = np.zeros((3, n_cols))
            j = pt_index[pid]
            block[:, n_t + 3 * j : n_t + 3 * j + 3] = S @ im.R
            if i > 0:
                block[:, 3 * (i - 1) : 3 * (i - 1) + 3] = S
            rows.append(block)

    if not rows:
        report.messages.append("no usable tie point observations")
        return None

    A = np.vstack(rows)
    AtA = A.T @ A
    evals, evecs = np.linalg.eigh(AtA)
    z = evecs[:, 0]
    report.init_condition = float(evals[1] / evals[0]) if evals[0] > 1e-18 else float("inf")
    if report.init_condition < 10.0:
        report.warnings.append(
            f"the linear initialisation is poorly conditioned (eigenvalue ratio "
            f"{report.init_condition:.1f}). Usually this means some image pair is held "
            "together by too few tie points, or all your tie points are nearly coplanar "
            "as seen from the cameras. Add a couple of well-spread points."
        )

    translations = {image_ids[0]: np.zeros(3)}
    for iid in image_ids[1:]:
        i = img_index[iid]
        translations[iid] = z[3 * (i - 1) : 3 * (i - 1) + 3].copy()
    positions = {pid: z[n_t + 3 * k : n_t + 3 * k + 3].copy() for pid, k in pt_index.items()}

    # the null vector's sign is arbitrary; pick the one putting points in front
    depths = []
    for iid in image_ids:
        im = project.image(iid)
        for pid in tables[iid]:
            if pid in positions:
                depths.append(float((im.R @ positions[pid] + translations[iid])[2]))
    depths = np.array(depths)
    if np.median(depths) < 0:
        for d in (translations, positions):
            for k in d:
                d[k] = -d[k]
        depths = -depths

    behind = int(np.sum(depths <= 0))
    if behind:
        report.warnings.append(
            f"{behind} of {len(depths)} observations initialise behind their camera. "
            "The bundle adjustment will try to pull them forward, but check for a tie point "
            "clicked on the wrong feature."
        )

    # normalise to a sane working scale; a distance constraint sets real units later
    spread = float(np.linalg.norm(np.array(list(positions.values())) - np.mean(list(positions.values()), axis=0), axis=1).mean())
    if spread > 1e-12:
        for d in (translations, positions):
            for k in d:
                d[k] = d[k] / spread

    for iid in image_ids:
        project.image(iid).t = translations[iid]
    return positions


# --------------------------------------------------------------------------
# stage 2b -- place the known objects, and let them set the scale
# --------------------------------------------------------------------------


def _transform_scene(
    project: Project,
    image_ids: list[str],
    positions: dict[str, np.ndarray],
    scale: float,
    Rg: np.ndarray,
    tg: np.ndarray,
) -> None:
    """Apply the similarity ``X -> scale * Rg @ X + tg`` to the whole solution.

    Cameras move with it rather than being re-solved: a camera is rigidly
    attached to the world, so requiring every image to stay identical gives
    ``R' = R @ Rg.T`` and ``t' = scale * t - R' @ tg`` directly. Nothing about
    any projection changes, which is the point -- this only re-expresses the
    answer in better coordinates.
    """
    for iid in image_ids:
        im = project.image(iid)
        if im.R is None or im.t is None:
            continue
        R_new = im.R @ Rg.T
        im.t = scale * im.t - R_new @ tg
        im.R = R_new
    for key in positions:
        positions[key] = scale * (Rg @ positions[key]) + tg

    # Hand-placed bodies are carried along too. Every other pose here is
    # recomputed from `positions` after this runs, so it needs no help; a manual
    # pose is the one thing in the scene that nothing downstream reconstructs,
    # and leaving it behind would silently detach it from the photographs it was
    # aligned against. The rotation takes no scale factor: the body's own size is
    # CAD truth, and this call is re-expressing the scene *into* those units, not
    # resizing the object.
    for inst in project.placed_object_instances():
        if not inst.manual or inst.R is None or inst.t is None:
            continue
        # ...but only once. A pose already in part space came out of a previous
        # solve in this very frame; transforming it again moves it by the whole
        # part-space offset, and again on the next solve, and again -- 2.9 metres
        # a time on the cell this was found on, with nothing in the report to say
        # so, because a manual pose has no residual. See
        # ObjectInstance.pose_in_part_space.
        if inst.pose_in_part_space:
            continue
        inst.R = Rg @ inst.R
        inst.t = scale * (Rg @ inst.t) + tg
        inst.pose_in_part_space = True


def align_to_nominals(
    project: Project,
    positions: dict[str, np.ndarray],
    image_ids: list[str],
    report: SolveReport,
) -> bool:
    """Re-express the initialised scene in the frame the nominals state.

    The linear initialisation deliberately normalises the scene to a working
    scale of about 1, because at that point nothing has said what the real units
    are. Nominals do say -- and on a machined part they say "0.12 metres across",
    which leaves the initialisation twenty times too large.

    Handing that to the bundle adjustment as a residual to fix is a bad idea,
    and measurably so. A nominal with a half-millimetre tolerance evaluated on a
    scene twenty times too big starts at some thousands of sigma, which dwarfs
    every reprojection residual in the problem; the optimiser then satisfies the
    nominals by whatever route is cheapest, and the cheapest route is to collapse
    the focal length so that projections stop having an opinion. That is not a
    hypothetical: on a 120 mm plate it drove a 2100 px focal to 1 px and left a
    362 px reprojection RMS while reporting convergence.

    So the nominals are applied as a *change of coordinates* before the refine,
    the same way a locked known object is, rather than as a force during it.
    Afterwards every nominal residual starts near zero and the bundle spends its
    effort on the photographs, which is what it is for. Nothing about any
    projection changes -- see :func:`_transform_scene`.

    Returns whether the scene was moved.
    """
    nominal = [
        (p.id, np.array([float(v) for v in p.nominal], dtype=float) * project.nominal_scale)
        for p in project.points
        if p.nominal and len(p.nominal) == 3 and all(v is not None for v in p.nominal)
        and p.id in positions
    ]
    if len(nominal) < 2:
        return False

    source = np.array([positions[pid] for pid, _target in nominal])
    target = np.array([t for _pid, t in nominal])

    if project.in_part_space() and len(nominal) >= 3:
        fit = similarity_fit(source, target, allow_scale=True)
        if fit.degenerate or not (fit.scale > 0) or not np.isfinite(fit.scale):
            report.messages.append(
                "the point nominals could not be fitted to the initialisation, so the refinement "
                "starts from arbitrary coordinates instead of the part's."
            )
            return False
        _transform_scene(project, image_ids, positions, fit.scale, fit.R, fit.t)
        report.messages.append(
            f"initialisation re-expressed in part space from {len(nominal)} nominal(s) before "
            f"refining: scale {fit.scale:.4g}, residual {fit.rms:.4g} in world units. The "
            "nominals now start satisfied, so the bundle adjustment spends itself on the "
            "photographs rather than on a change of units."
        )
        return True

    # Two nominals, or three in a line: the frame is not pinned but the distance
    # between them still states the scale, and the scale is the part of the
    # mismatch that actually destabilises the refine.
    best = max(
        (
            (float(np.linalg.norm(target[i] - target[j])), i, j)
            for i in range(len(nominal))
            for j in range(i + 1, len(nominal))
        ),
        default=(0.0, 0, 0),
    )
    span_target, i, j = best
    span_source = float(np.linalg.norm(source[i] - source[j]))
    if span_target <= 1e-12 or span_source <= 1e-12:
        return False
    scale = span_target / span_source
    _transform_scene(project, image_ids, positions, scale, np.eye(3), np.zeros(3))
    report.messages.append(
        f"{len(nominal)} nominal(s) is not enough to fix the frame, but the distance between two "
        f"of them set the scale ({scale:.4g}x) before refining, which is the part of it the bundle "
        "adjustment handles badly."
    )
    return True


def fit_instances(
    project: Project,
    positions: dict[str, np.ndarray],
    image_ids: list[str],
    report: SolveReport,
) -> tuple[bool, str]:
    """Fit every known object's model to its triangulated anchors, and use that
    to put the scene into real units.

    This is where a known object earns its keep. The linear stage leaves the
    reconstruction correct only up to a similarity -- arbitrary scale, arbitrary
    origin and orientation. One object of known size removes the scale outright,
    with no tape measure and no distance constraint; one *locked* object removes
    all seven degrees of freedom and makes the world frame that object's own, so
    re-solving after adding photos does not shuffle the coordinates around.

    Returns ``(scale_was_set, description)``.
    """
    anchor_locals = project.anchor_world_points()
    by_instance: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    marked_count: dict[str, set[str]] = {}
    for im in project.active_images():
        for key in im.anchor_obs:
            if key in anchor_locals:
                inst_id, name = split_anchor_key(key)
                marked_count.setdefault(inst_id, set()).add(name)
    for key, local in anchor_locals.items():
        if key not in positions:
            continue
        inst_id, name = split_anchor_key(key)
        by_instance.setdefault(inst_id, []).append((name, local, positions[key]))

    if not project.placed_object_instances():
        return False, ""

    # ---- first pass: a scaled fit per instance
    scaled: dict[str, Any] = {}
    for inst in project.placed_object_instances():
        rep = report.instances.setdefault(
            inst.id,
            InstanceReport(id=inst.id, template=inst.template, label=inst.label),
        )
        template = project.template_of(inst)
        if template is None:
            rep.notes.append(f"template {inst.template!r} is missing from this project")
            continue
        rep.n_anchors_marked = len(marked_count.get(inst.id, ()))
        rep.manual = bool(inst.manual and inst.R is not None and inst.t is not None)
        entries = sorted(by_instance.get(inst.id, []))
        rep.n_anchors_used = len(entries)
        if len(entries) < 2:
            rep.notes.append(
                f"{rep.n_anchors_used} of {rep.n_anchors_marked} marked anchor(s) reached 2+ "
                "photos; a placement needs 3 (2 leaves the body free to spin about the line "
                "joining them)"
            )
            if rep.manual:
                rep.notes.append(
                    "this pose was set by hand and is kept as it is. Nothing has refined it and "
                    "nothing has checked it: snap its anchors from the model, nudge them onto the "
                    "real features, and it becomes a measurement with an error to read"
                )
                center = project.instance_center(inst)
                rep.center = None if center is None else center.tolist()
            continue
        fit = similarity_fit(
            np.array([e[1] for e in entries]), np.array([e[2] for e in entries])
        )
        scaled[inst.id] = (fit, entries)
        if fit.degenerate:
            rep.notes.append(
                "the used anchors are collinear (or there are only 2), so the rotation about "
                "that line is not determined -- mark an anchor off the line"
            )

    if not scaled:
        report.warnings.append(
            f"{len(project.placed_object_instances())} known object(s) have no placeable anchors "
            "yet. Each needs "
            "3 anchors marked in 2+ photos each before its pose is determined."
        )
        return False, ""

    # ---- choose the gauge: a locked object defines the frame outright, otherwise
    # we only borrow its size and leave origin and orientation where they were
    locked = [
        inst
        for inst in project.placed_object_instances()
        if inst.locked and inst.R is not None and inst.t is not None and inst.id in scaled
    ]
    description = ""
    if locked:
        inst = locked[0]
        _fit, entries = scaled[inst.id]
        target = np.array([inst.R @ e[1] + inst.t for e in entries])
        source = np.array([e[2] for e in entries])
        gauge = similarity_fit(source, target)
        _transform_scene(project, image_ids, positions, gauge.scale, gauge.R, gauge.t)
        description = f"locked object {inst.label or inst.id}"
        if len(locked) > 1:
            report.messages.append(
                f"{len(locked)} objects are locked, but only one can define the frame; used "
                f"{inst.label or inst.id}. The others are held fixed during refinement, so a "
                "disagreement between them will show up in their residuals."
            )
    else:
        usable = [
            (inst_id, fit, len(entries))
            for inst_id, (fit, entries) in sorted(scaled.items())
            if fit.scale > 0
        ]
        if not usable:
            return False, ""
        logs = np.array([math.log(fit.scale) for _i, fit, _n in usable])
        weights = np.array([float(n) for _i, _f, n in usable])

        # Combine the per-object scales robustly rather than by weighted mean.
        # One body marked on the wrong corners would otherwise drag the whole
        # scene's scale, and with it every *other* body's shape error -- turning
        # one local mistake into a global one that points nowhere in particular.
        # The median locates what the objects agree on; a weighted mean over
        # whatever agrees with it keeps the precision of the better-marked ones.
        centre = float(np.median(logs))
        deviation = np.abs(logs - centre)
        mad = float(np.median(deviation))
        tolerance = max(3.0 * mad, math.log(1.02))
        inliers = deviation <= tolerance
        if not inliers.any():
            inliers = np.ones(len(logs), dtype=bool)
            report.warnings.append(
                "the known objects disagree about scale with no majority to trust, so all of "
                "them were used. Check each body's size reading in the objects panel."
            )

        combined = float(
            np.exp(np.sum(weights[inliers] * logs[inliers]) / np.sum(weights[inliers]))
        )
        excluded = [
            (inst_id, fit)
            for (inst_id, fit, _n), keep in zip(usable, inliers)
            if not keep
        ]
        for inst_id, fit in excluded:
            report.instances[inst_id].notes.append(
                f"left out of the scale estimate: this body solved {fit.scale / combined:.3f}x "
                "the size the others agree on"
            )
        if excluded:
            report.warnings.append(
                f"{len(excluded)} object(s) disagreed with the others about scale and were left "
                "out of the scale estimate, so one mis-marked body does not shift the whole "
                f"scene: {', '.join(i for i, _f in excluded)}. Their own placements are still "
                "reported, with their size error."
            )
        _transform_scene(project, image_ids, positions, 1.0 / combined, np.eye(3), np.zeros(3))
        used_units = {
            project.templates[project.instance(iid).template].units
            for iid in scaled
            if project.instance(iid) is not None
            and project.instance(iid).template in project.templates
        }
        unit = used_units.pop() if len(used_units) == 1 else "model units"
        description = f"known object geometry ({unit})"

    # ---- second pass: the rigid pose each instance actually carries
    for inst in project.placed_object_instances():
        rep = report.instances.get(inst.id)
        if inst.id not in scaled:
            # Clearing the pose is right for a fit that no longer has the
            # anchors behind it, and wrong for a placement someone made by eye:
            # that pose IS the user's work, and no later stage recreates it.
            if not inst.locked and not inst.manual:
                inst.R, inst.t = None, None
            continue
        _old_fit, entries = scaled[inst.id]
        source = np.array([e[1] for e in entries])
        target = np.array([positions[anchor_key(inst.id, e[0])] for e in entries])
        rigid = similarity_fit(source, target, allow_scale=False)
        check = similarity_fit(source, target, allow_scale=True)
        inst.R, inst.t = rigid.R, rigid.t
        inst.manual = False
        if rep is not None:
            rep.placed = True
            rep.manual = False
            rep.shape_rms = rigid.rms
            rep.implied_scale = check.scale
            center = project.instance_center(inst)
            rep.center = None if center is None else center.tolist()
            optical = project.instance_optical(inst)
            if optical is not None:
                rep.optical_center = optical[0].tolist()
                rep.optical_forward = optical[1].tolist()
            if abs(check.scale - 1.0) > 0.05:
                rep.notes.append(
                    f"this body solved {abs(check.scale - 1.0) * 100:.0f}% "
                    f"{'larger' if check.scale > 1 else 'smaller'} than its model. Either an "
                    "anchor is marked on the wrong feature, or this is not the object the model "
                    "describes."
                )

    return True, description


def build_rig_report(project: Project, report: SolveReport) -> RigReport:
    """Baselines and relative angles between the placed objects.

    For a camera array this is the answer to the actual question. It is reported
    pairwise rather than as absolute poses because the pairwise numbers are what
    survive the arbitrary choice of world origin, and because a rig spec is
    usually written that way: "cameras 120 mm apart, toed in 8 degrees".
    """
    rig = RigReport()
    placed = [
        inst
        for inst in project.placed_object_instances()
        if inst.R is not None and inst.t is not None and report.instances.get(inst.id, InstanceReport(id=inst.id)).placed
    ]
    rig.n_instances = len(placed)
    units = {project.template_of(i).units for i in placed if project.template_of(i) is not None}
    rig.units = "m" if len(units) else "model units"

    def station(inst) -> tuple[np.ndarray, np.ndarray | None]:
        """Where to measure this object from: its lens if the template says where
        that is, otherwise the body's own centre and local +Z."""
        optical = project.instance_optical(inst)
        if optical is not None:
            return optical
        center = project.instance_center(inst)
        return (center if center is not None else inst.t), inst.R[:, 2]

    stations = {inst.id: station(inst) for inst in placed}
    if len(placed) > 1:
        centers = np.array([stations[i.id][0] for i in placed])
        rig.extent = float(
            np.max(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2))
        )
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            ca, fa = stations[a.id]
            cb, fb = stations[b.id]
            entry: dict[str, Any] = {
                "a": a.label or a.id,
                "b": b.label or b.id,
                "distance": float(np.linalg.norm(cb - ca)),
            }
            if fa is not None and fb is not None:
                cos = float(np.clip(np.asarray(fa) @ np.asarray(fb), -1.0, 1.0))
                entry["angle_deg"] = math.degrees(math.acos(cos))
            rig.pairs.append(entry)

    if rig.n_instances and not any(
        project.template_of(i) is not None and project.template_of(i).optical is not None
        for i in placed
    ):
        rig.notes.append(
            "These are body placements, not camera poses. Set an optical frame on the template "
            "-- the lens centre and the direction it faces -- and the layout is reported as real "
            "camera extrinsics instead."
        )
    return rig


# --------------------------------------------------------------------------
# stage 3 -- joint bundle adjustment
# --------------------------------------------------------------------------


class _Problem:
    """Parameter packing, residuals and Jacobian sparsity for the joint refine."""

    def __init__(
        self,
        project: Project,
        intrinsics: dict[str, Intrinsics],
        image_ids: list[str],
        point_ids: list[str],
    ):
        self.project = project
        self.s = project.settings
        self.image_ids = image_ids
        self.point_ids = point_ids
        self.img_index = {iid: k for k, iid in enumerate(image_ids)}
        self.pt_index = {pid: k for k, pid in enumerate(point_ids)}

        self.group_ids = sorted({project.image(i).group for i in image_ids})
        self.group_of_image = {i: project.image(i).group for i in image_ids}

        # ---- known-object anchor marks, grouped by (image, instance)
        #
        # Grouped that way because every observation in one such block shares
        # both a camera pose and a body pose, so the projection vectorises and
        # the Jacobian block structure falls out directly.
        anchor_locals = project.anchor_world_points()
        grouped: dict[tuple[str, str], list[tuple[str, np.ndarray, list[float]]]] = {}
        for iid in image_ids:
            im = project.image(iid)
            for key, uv in im.anchor_obs.items():
                local = anchor_locals.get(key)
                if local is None:
                    continue
                inst_id, name = split_anchor_key(key)
                inst = project.instance(inst_id)
                if inst is None or inst.R is None or inst.t is None:
                    continue  # unplaced body: nothing to refine its anchors against
                grouped.setdefault((iid, inst_id), []).append((key, local, list(uv)))

        self.anchor_blocks: list[tuple[str, str, np.ndarray, np.ndarray, list[str]]] = []
        for (iid, inst_id), entries in sorted(grouped.items()):
            entries.sort()
            self.anchor_blocks.append(
                (
                    iid,
                    inst_id,
                    np.array([e[1] for e in entries], dtype=float),
                    np.array([e[2] for e in entries], dtype=float),
                    [e[0] for e in entries],
                )
            )
        self.anchor_instances = {inst_id for _i, inst_id, _l, _u, _k in self.anchor_blocks}
        self.n_anchor_obs = sum(len(block[3]) for block in self.anchor_blocks)

        # ---- parameter layout
        self.n_img = len(image_ids)
        self.off_images = 0
        cursor = 6 * self.n_img
        self.group_slots: dict[str, dict[str, int]] = {}
        for gid in self.group_ids:
            g = project.groups[gid]
            slots: dict[str, int] = {}
            if self.s.refine_focal and not g.lock_focal:
                slots["f"] = cursor
                cursor += 1
            if not g.lock_principal_point:
                slots["cx"] = cursor
                slots["cy"] = cursor + 1
                cursor += 2
            if not g.lock_k1:
                slots["k1"] = cursor
                cursor += 1
            self.group_slots[gid] = slots
        self.off_points = cursor
        cursor += 3 * len(point_ids)

        # ---- known-object instances: 6 parameters each, and none at all for the
        # anchors themselves. An anchor is not a free point -- its world position
        # is dictated by the model's geometry and the body's pose, which is the
        # entire reason a known object constrains so much more than the same
        # number of clicks on free features. Twelve anchors on one body cost 6
        # unknowns, not 36.
        self.instance_ids = [
            inst.id
            for inst in project.placed_object_instances()
            if inst.R is not None and inst.t is not None and inst.id in self.anchor_instances
        ]
        self.instance_slots: dict[str, int] = {}
        self.off_instances = cursor
        for inst_id in self.instance_ids:
            if not project.instance(inst_id).locked:
                self.instance_slots[inst_id] = cursor
                cursor += 6
        self.fixed_instances = {
            inst_id: (project.instance(inst_id).R.copy(), project.instance(inst_id).t.copy())
            for inst_id in self.instance_ids
        }
        self.n_params = cursor

        # ---- fixed values for whatever is locked
        self.fixed: dict[str, Intrinsics] = {}
        for gid in self.group_ids:
            g = project.groups[gid]
            any_img = next(i for i in image_ids if self.group_of_image[i] == gid)
            im = project.image(any_img)
            self.fixed[gid] = Intrinsics(
                f=g.f if g.f is not None else intrinsics[any_img].f,
                cx=g.cx if g.cx is not None else im.width / 2.0,
                cy=g.cy if g.cy is not None else im.height / 2.0,
                k1=g.k1,
            )

        # ---- observation tables, grouped per image for vectorised projection
        self.obs: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.obs_pids: dict[str, list[str]] = {}
        for iid in image_ids:
            im = project.image(iid)
            pids = [p for p in im.observations if p in self.pt_index]
            if not pids:
                continue
            uv = np.array([im.observations[p] for p in pids], dtype=float)
            idx = np.array([self.pt_index[p] for p in pids], dtype=int)
            self.obs.append((iid, idx, uv))
            self.obs_pids[iid] = pids
        self.n_obs = sum(len(idx) for _, idx, _ in self.obs)

        # ---- control lines, as homogeneous image lines plus their world axis
        self.lines: list[tuple[str, int, np.ndarray, float, Any]] = []
        for iid in image_ids:
            im = project.image(iid)
            for ln in im.axis_lines:
                p0 = np.array([ln.p0[0], ln.p0[1], 1.0])
                p1 = np.array([ln.p1[0], ln.p1[1], 1.0])
                L = ln.length()
                if L < 1e-6:
                    continue
                self.lines.append((iid, ln.axis, np.cross(p0, p1), L, ln))

        # Held, not recomputed. ``all_constraints()`` builds fresh objects for
        # the nominals every call, so a second call would produce an equal list
        # of different objects -- and the report, which matches by identity to
        # find out which constraints actually reached the solve, would conclude
        # that none of the nominals had.
        self.declared = project.all_constraints()
        self.constraints = [
            c
            for c in self.declared
            if (
                (isinstance(c, AxisLockConstraint) and c.point in self.pt_index)
                or (
                    isinstance(c, DistanceConstraint)
                    and c.point_a in self.pt_index
                    and c.point_b in self.pt_index
                )
            )
        ]

        # ---- per-observation weights, and one sigma per known object
        #
        # Culling is done by zeroing a weight, never by removing the mark from
        # these tables. The parameter layout, the residual offsets and the
        # Jacobian sparsity then stay byte-for-byte identical between passes, so
        # a second solve cannot silently shift an index -- and a dropped mark is
        # still there to be reported back to the user.
        # A mark flagged uncertain keeps contributing, at a wider sigma. The
        # flag lives on the point and the line rather than on the observation,
        # because that is how a person thinks about it: "I could not see that
        # corner properly" is a fact about the feature, not about one photo of
        # it. The same array carries both this and the outlier cull, so an
        # uncertain mark that also turns out to be wrong simply reaches zero.
        unc = 1.0 / max(float(self.s.uncertain_factor), 1e-6)
        self.obs_weight = [
            np.array([unc if self._uncertain_point(pid) else 1.0 for pid in self.obs_pids[iid]])
            for iid, _idx, _uv in self.obs
        ]
        self.obs_confident = [w >= 1.0 for w in self.obs_weight]
        self.line_weight = np.array(
            [unc if ln.uncertain else 1.0 for _iid, _axis, _line, _L, ln in self.lines]
        )
        self.line_confident = self.line_weight >= 1.0
        self.line_errs = np.zeros(len(self.lines))
        """Filled by the first bundle pass; zeros mean nothing has been measured yet."""
        self.anchor_weight = [np.ones(len(b[3])) for b in self.anchor_blocks]
        self.anchor_confident = [np.ones(len(b[3]), dtype=bool) for b in self.anchor_blocks]
        self.sigma_point = float(self.s.sigma_px)
        self.anchor_sigma = {
            inst_id: float(self.s.sigma_anchor_px) for inst_id in self.anchor_instances
        }

        # Marks per body, used to stop a densely marked object outvoting
        # everything else. One body is 6 unknowns no matter how many anchors
        # carry it, so N marks contribute 2N residuals against 6 parameters that
        # cannot flex to absorb them: twenty anchors on one body otherwise swamp
        # ten tie points. Scaling the residuals by (ref/N)^(1/4) lets a body's
        # influence grow like sqrt(N) instead of N -- more marks still pin the
        # pose better, they just stop compounding.
        counts: dict[str, int] = {}
        for _iid, inst_id, _local, uv, _keys in self.anchor_blocks:
            counts[inst_id] = counts.get(inst_id, 0) + len(uv)
        self.anchor_count_scale = {inst_id: 1.0 for inst_id in self.anchor_instances}
        ref = float(self.s.anchor_count_ref)
        if ref > 0:
            for inst_id, n in counts.items():
                if n > ref:
                    self.anchor_count_scale[inst_id] = (ref / n) ** 0.25
        self.anchor_row_scale = np.array(
            [
                self.anchor_count_scale[inst_id]
                for _iid, inst_id, _local, uv, _keys in self.anchor_blocks
                for _ in range(2 * len(uv))
            ],
            dtype=float,
        )

        # ---- residual layout, as explicit offsets
        #
        # Named rather than recomputed at each use: there are now six blocks, and
        # an off-by-one in an implicit cursor would silently mix anchor residuals
        # into the depth barrier.
        self.r_reproj = 2 * self.n_obs
        self.r_anchor = 2 * self.n_anchor_obs
        self.r_lines = len(self.lines)
        self.r_constraints = len(self.constraints)
        self.r_depth = self.n_obs + self.n_anchor_obs
        self.r_gauge = 6 * self.n_img + len(self.group_ids) + 6 * len(self.instance_slots)

        self.o_reproj = 0
        self.o_anchor = self.o_reproj + self.r_reproj
        self.o_lines = self.o_anchor + self.r_anchor
        self.o_cons = self.o_lines + self.r_lines
        self.o_depth = self.o_cons + self.r_constraints
        self.o_gauge = self.o_depth + self.r_depth
        self.n_residuals = self.o_gauge + self.r_gauge
        self.z_min = 1e-3
        self.depth_weight = 100.0
        """Barrier stiffness for points drifting behind a camera. Scene scale is
        normalised to about 1 by the initialiser, so this is firm but bounded."""

    def _uncertain_point(self, pid: str) -> bool:
        pt = self.project.point(pid)
        return bool(pt is not None and pt.uncertain)

    # ------------------------------------------------------------------ pack

    def pack(self) -> np.ndarray:
        x = np.zeros(self.n_params)
        for iid in self.image_ids:
            im = self.project.image(iid)
            k = self.img_index[iid]
            x[6 * k : 6 * k + 3] = so3_log(im.R)
            x[6 * k + 3 : 6 * k + 6] = im.t
        for gid, slots in self.group_slots.items():
            cam = self.fixed[gid]
            for name, pos in slots.items():
                x[pos] = getattr(cam, name)
        for pid, k in self.pt_index.items():
            x[self.off_points + 3 * k : self.off_points + 3 * k + 3] = self.project.point(pid).xyz
        for inst_id, pos in self.instance_slots.items():
            R, t = self.fixed_instances[inst_id]
            x[pos : pos + 3] = so3_log(R)
            x[pos + 3 : pos + 6] = t
        self.x0 = x.copy()
        return x

    def instance_pose_at(self, x: np.ndarray, inst_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Object-to-world pose of one instance: ``X_world = R @ p_local + t``.

        Note the opposite sense from an image's world-to-camera ``R``.
        """
        pos = self.instance_slots.get(inst_id)
        if pos is None:
            return self.fixed_instances[inst_id]
        return so3_exp(x[pos : pos + 3]), x[pos + 3 : pos + 6]

    def intrinsics_at(self, x: np.ndarray, gid: str) -> Intrinsics:
        base = self.fixed[gid]
        slots = self.group_slots[gid]
        return Intrinsics(
            f=x[slots["f"]] if "f" in slots else base.f,
            cx=x[slots["cx"]] if "cx" in slots else base.cx,
            cy=x[slots["cy"]] if "cy" in slots else base.cy,
            k1=x[slots["k1"]] if "k1" in slots else base.k1,
        )

    def unpack(self, x: np.ndarray) -> tuple[dict, dict, np.ndarray, dict]:
        poses = {}
        for iid in self.image_ids:
            k = self.img_index[iid]
            poses[iid] = (so3_exp(x[6 * k : 6 * k + 3]), x[6 * k + 3 : 6 * k + 6])
        cams = {gid: self.intrinsics_at(x, gid) for gid in self.group_ids}
        X = x[self.off_points : self.off_points + 3 * len(self.point_ids)].reshape(-1, 3)
        objects = {inst_id: self.instance_pose_at(x, inst_id) for inst_id in self.instance_ids}
        return poses, cams, X, objects

    # ------------------------------------------------------------- residuals

    def _project_block(
        self, Xw: np.ndarray, R: np.ndarray, t: np.ndarray, cam: Intrinsics
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixels for world points ``Xw``, plus their raw camera-space depths.

        Depth is clamped for the projection and the violation penalised in its own
        residual block, so a point that drifts behind a camera stays finite and
        gets pushed back rather than producing a NaN that takes the solve with it.
        """
        Xc = Xw @ R.T + t
        z = Xc[:, 2]
        z_eff = np.maximum(z, self.z_min)
        xn, yn = Xc[:, 0] / z_eff, Xc[:, 1] / z_eff
        if cam.k1 != 0.0:
            r2 = xn * xn + yn * yn
            scale = 1.0 + cam.k1 * r2
            xn, yn = xn * scale, yn * scale
        return np.stack([cam.f * xn + cam.cx, cam.f * yn + cam.cy], axis=1), z

    def residuals(self, x: np.ndarray) -> np.ndarray:
        poses, cams, X, objects = self.unpack(x)
        out = np.zeros(self.n_residuals)
        sp = self.sigma_point
        sl = self.s.sigma_line_px

        cursor = self.o_reproj
        depth_cursor = self.o_depth
        for (iid, idx, uv), w in zip(self.obs, self.obs_weight):
            R, t = poses[iid]
            cam = cams[self.group_of_image[iid]]
            pix, z = self._project_block(X[idx], R, t, cam)
            n = len(idx)
            out[cursor : cursor + 2 * n : 2] = w * (pix[:, 0] - uv[:, 0]) / sp
            out[cursor + 1 : cursor + 2 * n : 2] = w * (pix[:, 1] - uv[:, 1]) / sp
            cursor += 2 * n
            # a culled mark drops its depth barrier too: it is no longer evidence
            # about where that point is, so it has no business pushing on it
            out[depth_cursor : depth_cursor + n] = (
                w * np.maximum(0.0, self.z_min - z) * self.depth_weight
            )
            depth_cursor += n

        # known-object anchors: same reprojection error, but the world point is
        # the model's own geometry carried by the body's pose rather than a free
        # parameter, so these residuals refine 6 unknowns per body no matter how
        # many anchors are marked
        cursor = self.o_anchor
        for (iid, inst_id, local, uv, _keys), w in zip(self.anchor_blocks, self.anchor_weight):
            R, t = poses[iid]
            R_obj, t_obj = objects[inst_id]
            cam = cams[self.group_of_image[iid]]
            pix, z = self._project_block(local @ R_obj.T + t_obj, R, t, cam)
            n = len(uv)
            # this body's own sigma, not the tie-point one: looser to begin with,
            # and raised further by the second pass if this body's marks have
            # shown they deserve it
            sa = self.anchor_sigma[inst_id]
            out[cursor : cursor + 2 * n : 2] = w * (pix[:, 0] - uv[:, 0]) / sa
            out[cursor + 1 : cursor + 2 * n : 2] = w * (pix[:, 1] - uv[:, 1]) / sa
            cursor += 2 * n
            out[depth_cursor : depth_cursor + n] = (
                w * np.maximum(0.0, self.z_min - z) * self.depth_weight
            )
            depth_cursor += n

        # control lines: the world axis direction must lie in the plane through
        # the camera centre and the drawn image line
        cursor = self.o_lines
        for (iid, axis, line, L, _ln), w in zip(self.lines, self.line_weight):
            R, _ = poses[iid]
            cam = cams[self.group_of_image[iid]]
            nrm = cam.K().T @ line
            nn = np.linalg.norm(nrm)
            if nn < 1e-12:
                out[cursor] = 0.0
            else:
                sin_angle = float((nrm / nn) @ R[:, axis])
                # (L/2) * sin(angle) is the perpendicular pixel slip at the endpoints
                out[cursor] = w * (L / 2.0) * sin_angle / sl
            cursor += 1

        cursor = self.o_cons
        for c in self.constraints:
            if isinstance(c, AxisLockConstraint):
                k = self.pt_index[c.point]
                out[cursor] = (X[k, c.axis] - c.value) / c.sigma
            else:
                a = X[self.pt_index[c.point_a]]
                b = X[self.pt_index[c.point_b]]
                out[cursor] = (float(np.linalg.norm(a - b)) - c.value) / c.sigma
            cursor += 1

        cursor = self.o_gauge
        # weak gauge prior: translation and scale are otherwise free (and rotation
        # too, if no image has control lines). Far too weak to bias the fit.
        w = self.s.gauge_weight
        n_pose = 6 * self.n_img
        out[cursor : cursor + n_pose] = w * (x[:n_pose] - self.x0[:n_pose])
        cursor += n_pose
        for gid in self.group_ids:
            slots = self.group_slots[gid]
            if "f" in slots:
                out[cursor] = w * (x[slots["f"]] - self.x0[slots["f"]]) / max(self.fixed[gid].f, 1.0)
            cursor += 1
        # Instance poses get the same weak prior. A body seen in only one photo
        # has unobservable depth; without this its parameters wander freely and
        # the normal equations go singular.
        for inst_id in self.instance_ids:
            pos = self.instance_slots.get(inst_id)
            if pos is None:
                continue
            out[cursor : cursor + 6] = w * (x[pos : pos + 6] - self.x0[pos : pos + 6])
            cursor += 6

        # Robust kernel on the clicked data only -- reprojections, anchors and
        # control lines. Constraints, the depth barrier and the gauge prior keep
        # their full weight.
        out[: self.o_cons] = _huberize(out[: self.o_cons], self.s.huber_f_scale)
        # Per-body count normalisation goes on *after* the kernel, deliberately:
        # the kernel asks "is this mark an outlier for its class?", which must be
        # judged against that class's sigma, while this asks "how much should one
        # body's evidence count?". Folding them together would let a densely
        # marked body hide its own blunders under the knee.
        if self.r_anchor:
            out[self.o_anchor : self.o_anchor + self.r_anchor] *= self.anchor_row_scale
        return out

    # -------------------------------------------------------------- sparsity

    def jac_sparsity(self) -> sparse.csr_matrix:
        S = sparse.lil_matrix((self.n_residuals, self.n_params), dtype=np.int8)

        row = self.o_reproj
        depth_row = self.o_depth
        for iid, idx, _ in self.obs:
            k = self.img_index[iid]
            slots = list(self.group_slots[self.group_of_image[iid]].values())
            for j in idx:
                pc = self.off_points + 3 * j
                for r in (row, row + 1):
                    S[r, 6 * k : 6 * k + 6] = 1
                    S[r, pc : pc + 3] = 1
                    for pos in slots:
                        S[r, pos] = 1
                row += 2
                S[depth_row, 6 * k : 6 * k + 6] = 1
                S[depth_row, pc : pc + 3] = 1
                depth_row += 1

        # anchor rows touch the camera pose, the shared intrinsics and the body's
        # 6 pose parameters -- and no point block at all, since the anchor has no
        # free position of its own
        row = self.o_anchor
        for iid, inst_id, _local, uv, _keys in self.anchor_blocks:
            k = self.img_index[iid]
            slots = list(self.group_slots[self.group_of_image[iid]].values())
            obj = self.instance_slots.get(inst_id)
            for _ in range(len(uv)):
                for r in (row, row + 1):
                    S[r, 6 * k : 6 * k + 6] = 1
                    for pos in slots:
                        S[r, pos] = 1
                    if obj is not None:
                        S[r, obj : obj + 6] = 1
                row += 2
                S[depth_row, 6 * k : 6 * k + 6] = 1
                if obj is not None:
                    S[depth_row, obj : obj + 6] = 1
                depth_row += 1

        row = self.o_lines
        for iid, _axis, _line, _L, _ln in self.lines:
            k = self.img_index[iid]
            S[row, 6 * k : 6 * k + 3] = 1  # rotation only; translation cannot tilt a direction
            for pos in self.group_slots[self.group_of_image[iid]].values():
                S[row, pos] = 1
            row += 1

        row = self.o_cons
        for c in self.constraints:
            if isinstance(c, AxisLockConstraint):
                pc = self.off_points + 3 * self.pt_index[c.point]
                S[row, pc : pc + 3] = 1
            else:
                for pid in (c.point_a, c.point_b):
                    pc = self.off_points + 3 * self.pt_index[pid]
                    S[row, pc : pc + 3] = 1
            row += 1

        row = self.o_gauge
        for i in range(6 * self.n_img):
            S[row + i, i] = 1
        row += 6 * self.n_img
        for gid in self.group_ids:
            slots = self.group_slots[gid]
            if "f" in slots:
                S[row, slots["f"]] = 1
            else:
                S[row, 0] = 1  # placeholder so the row is not structurally empty
            row += 1
        for inst_id in self.instance_ids:
            pos = self.instance_slots.get(inst_id)
            if pos is None:
                continue
            for i in range(6):
                S[row + i, pos + i] = 1
            row += 6

        return S.tocsr()

    def write_back(self, x: np.ndarray) -> None:
        poses, cams, X, objects = self.unpack(x)
        for iid in self.image_ids:
            im = self.project.image(iid)
            im.R, im.t = poses[iid]
        for gid, cam in cams.items():
            g = self.project.groups[gid]
            g.f, g.cx, g.cy, g.k1 = cam.f, cam.cx, cam.cy, cam.k1
            # keep the locked-parameter baseline in step, or a later pack() would
            # reload pre-solve intrinsics over the refined ones
            self.fixed[gid] = cam.copy()
        for pid, k in self.pt_index.items():
            self.project.point(pid).xyz = X[k]
        for inst_id, (R_obj, t_obj) in objects.items():
            inst = self.project.instance(inst_id)
            inst.R, inst.t = R_obj, t_obj
            # keep the locked-pose baseline in step, as for intrinsics above
            self.fixed_instances[inst_id] = (R_obj.copy(), t_obj.copy())


def _line_errors(problem: _Problem, x: np.ndarray) -> np.ndarray:
    """Each control line's endpoint slip in pixels: how far the drawn segment
    misses the direction the solved rotation says that axis points."""
    poses, cams, _X, _objects = problem.unpack(x)
    out = np.zeros(len(problem.lines))
    for i, (iid, axis, line, L, _ln) in enumerate(problem.lines):
        R, _t = poses[iid]
        nrm = cams[problem.group_of_image[iid]].K().T @ line
        nn = np.linalg.norm(nrm)
        out[i] = 0.0 if nn < 1e-12 else abs((L / 2.0) * float((nrm / nn) @ R[:, axis]))
    return out


def _observation_errors(
    problem: _Problem, x: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reprojection error in pixels for every clicked mark, block by block.

    Deliberately routed through the problem's own ``_project_block`` rather than
    the generic projector: it is the same code path the residuals use, distortion
    and depth clamp included, so a mark can never be judged an outlier by a
    camera model slightly different from the one that placed it.
    """
    poses, cams, X, objects = problem.unpack(x)

    point_errs = []
    for iid, idx, uv in problem.obs:
        R, t = poses[iid]
        pix, _z = problem._project_block(X[idx], R, t, cams[problem.group_of_image[iid]])
        point_errs.append(np.linalg.norm(pix - uv, axis=1))

    anchor_errs = []
    for iid, inst_id, local, uv, _keys in problem.anchor_blocks:
        R, t = poses[iid]
        R_obj, t_obj = objects[inst_id]
        pix, _z = problem._project_block(
            local @ R_obj.T + t_obj, R, t, cams[problem.group_of_image[iid]]
        )
        anchor_errs.append(np.linalg.norm(pix - uv, axis=1))

    return point_errs, anchor_errs


def _robust_scale(errs: np.ndarray, floor: float) -> float:
    """A sigma read off the data itself, never below the stated click accuracy.

    Median-based rather than RMS. The point of measuring is to find out how well
    this set of marks agrees; letting one 40-pixel blunder inflate the estimate
    would excuse exactly the mark we want to catch. The 0.8493 converts the
    median of a 2D radial error back to a per-axis sigma (a Rayleigh
    distribution's median is 1.177 sigma).

    The floor matters as much as the estimate: a body whose three marks happen
    to agree to a quarter pixel has not earned a quarter-pixel sigma, it has had
    good luck. Settings state what a click is worth; the data is only allowed to
    argue that it is worth less.
    """
    if errs.size == 0:
        return float(floor)
    return max(float(floor), 0.8493 * float(np.median(errs)))


def _reweight_instances(
    problem: _Problem, anchor_errs: list[np.ndarray], report: SolveReport
) -> bool:
    """Give every known object a sigma taken from its own residuals.

    This is the answer to "a badly placed object makes the whole solve worse".
    A body cannot flex: its anchors are nailed to the model's geometry, so marks
    that disagree with the photos have nowhere to put that disagreement except
    the camera poses. Measuring the disagreement and widening that body's sigma
    by exactly as much lets it keep the influence it has earned and no more --
    the cameras stop being dragged over to meet a body that was never marked
    that well, while a crisply marked body loses nothing.
    """
    s = problem.s
    by_instance: dict[str, list[np.ndarray]] = {}
    for (_iid, inst_id, _local, _uv, _keys), errs in zip(problem.anchor_blocks, anchor_errs):
        by_instance.setdefault(inst_id, []).append(errs)

    changed = False
    for inst_id, parts in sorted(by_instance.items()):
        sigma = _robust_scale(np.concatenate(parts), s.sigma_anchor_px)
        if sigma > problem.anchor_sigma[inst_id] * 1.01:
            changed = True
        problem.anchor_sigma[inst_id] = sigma
        rep = report.instances.setdefault(inst_id, InstanceReport(id=inst_id))
        rep.sigma_px = sigma
        if sigma > s.sigma_anchor_px * 1.5:
            rep.notes.append(
                f"weighted at {sigma:.1f} px rather than the {s.sigma_anchor_px:g} px a body "
                "normally gets, because its own anchor marks disagree by that much. It still "
                "contributes, it just no longer claims an accuracy it has not shown"
            )
    return changed


def _cull_outliers(
    problem: _Problem,
    point_errs: list[np.ndarray],
    anchor_errs: list[np.ndarray],
    report: SolveReport,
) -> bool:
    """Drop the marks the robust kernel can only soften.

    Huber bends a gross blunder's pull down to a constant, which is not the same
    as removing it: a mark 80 px out still tugs forever with the force of one at
    the knee. Past some point that is not noise to be down-weighted, it is a
    click on the wrong corner, and the honest thing is to take it out and say so.

    Guarded, because culling can starve the thing it is trying to clean: a track
    needs 2 views, an image needs 2 marks, a body needs 3 anchors. Worst marks go
    first, each one re-checked against what its removal would leave behind, and
    the whole pass is capped at a fifth of the data -- if more than that looks
    like an outlier the problem is not the marks, and quietly deleting them would
    only hide it.
    """
    s = problem.s
    # The scale is read off the *confident* marks only. Including the uncertain
    # ones would let them widen the very threshold they are supposed to be
    # judged against more harshly, which is backwards.
    confident = [
        e[c] for e, c in zip(point_errs, problem.obs_confident) if c.any()
    ]
    flat = np.concatenate(confident) if confident else np.zeros(0)
    point_sigma = _robust_scale(flat, s.sigma_px)

    # A mark flagged uncertain is held to a tighter threshold, which is what
    # "more likely to be dropped" has to mean in practice: it is already
    # down-weighted, so judging it at the same number of sigma as a confident
    # mark would make the flag *protect* it from the cull.
    factor = max(float(s.uncertain_factor), 1.0)
    doubt_sigma = max(s.cull_sigma / factor, 2.0)

    track_count: dict[str, int] = {}
    image_count: dict[str, int] = {}
    inst_count: dict[str, int] = {}
    axis_count: dict[tuple[str, int], int] = {}
    candidates: list[tuple[float, str, int, int, str, str, float]] = []

    for b, ((iid, _idx, _uv), errs) in enumerate(zip(problem.obs, point_errs)):
        sure = problem.obs_confident[b]
        for j, (pid, e) in enumerate(zip(problem.obs_pids[iid], errs)):
            track_count[pid] = track_count.get(pid, 0) + 1
            image_count[iid] = image_count.get(iid, 0) + 1
            limit = (s.cull_sigma if sure[j] else doubt_sigma) * point_sigma
            if e > limit:
                candidates.append((float(e / point_sigma), "point", b, j, iid, pid, float(e)))

    for b, ((iid, inst_id, _l, _uv, keys), errs) in enumerate(
        zip(problem.anchor_blocks, anchor_errs)
    ):
        sigma = problem.anchor_sigma[inst_id]
        for j, (key, e) in enumerate(zip(keys, errs)):
            image_count[iid] = image_count.get(iid, 0) + 1
            inst_count[inst_id] = inst_count.get(inst_id, 0) + 1
            if e > s.cull_sigma * sigma:
                candidates.append((float(e / sigma), "anchor", b, j, iid, key, float(e)))

    sure_lines = problem.line_errs[problem.line_confident] if len(problem.lines) else np.zeros(0)
    line_sigma = _robust_scale(sure_lines, s.sigma_line_px)
    for i, (iid, axis, _line, _L, _ln) in enumerate(problem.lines):
        axis_count[(iid, axis)] = axis_count.get((iid, axis), 0) + 1
        limit = (s.cull_sigma if problem.line_confident[i] else doubt_sigma) * line_sigma
        if problem.line_errs[i] > limit:
            candidates.append(
                (
                    float(problem.line_errs[i] / line_sigma),
                    "line",
                    i,
                    axis,
                    iid,
                    f"{AXIS_NAMES[axis]} line",
                    float(problem.line_errs[i]),
                )
            )

    if not candidates:
        return False

    candidates.sort(key=lambda c: -c[0])
    budget = max(1, int(0.2 * (problem.n_obs + problem.n_anchor_obs + len(problem.lines))))
    dropped: list[tuple[str, str, float, float]] = []
    starved = 0

    for n_sigma, kind, b, j, iid, label, err in candidates:
        if len(dropped) >= budget:
            break
        if kind != "line" and image_count.get(iid, 0) <= 2:
            starved += 1
            continue
        if kind == "line":
            # a vanishing point needs two lines; dropping to one leaves that
            # axis of this photo's rotation resting on a single drag
            if axis_count.get((iid, j), 0) <= 2:
                starved += 1
                continue
            axis_count[(iid, j)] -= 1
            problem.line_weight[b] = 0.0
            report.images.setdefault(iid, ImageReport(id=iid)).n_culled += 1
            dropped.append((iid, label, err, n_sigma))
            continue
        if kind == "point":
            if track_count.get(label, 0) <= 2:
                starved += 1
                continue
            track_count[label] -= 1
            problem.obs_weight[b][j] = 0.0
        else:
            inst_id = problem.anchor_blocks[b][1]
            if inst_count.get(inst_id, 0) <= 3:
                starved += 1
                continue
            inst_count[inst_id] -= 1
            problem.anchor_weight[b][j] = 0.0
            report.instances.setdefault(inst_id, InstanceReport(id=inst_id)).n_culled += 1
        image_count[iid] -= 1
        report.images.setdefault(iid, ImageReport(id=iid)).n_culled += 1
        dropped.append((iid, label, err, n_sigma))

    if not dropped:
        if starved:
            report.warnings.append(
                f"{starved} mark(s) look like outliers but could not be dropped without leaving "
                "a point, a photo or an object with too few marks to solve. Mark more features "
                "in those photos, or re-click the bad ones -- the solve is carrying them."
            )
        return False

    report.n_culled = len(dropped)
    worst = sorted(dropped, key=lambda d: -d[2])[:4]
    listed = ", ".join(f"{label} in {iid} ({err:.0f} px)" for iid, label, err, _n in worst)
    report.messages.append(
        f"dropped {len(dropped)} mark(s) beyond {s.cull_sigma:g} sigma and re-solved without "
        f"them; worst: {listed}"
    )
    report.warnings.append(
        f"{len(dropped)} mark(s) disagreed with everything else badly enough to be left out of "
        "the final solve. They are still drawn in the residual view so you can see them -- "
        "check each is on the feature you meant, then re-solve."
    )
    if starved:
        report.warnings.append(
            f"a further {starved} mark(s) look just as bad but were kept, because dropping them "
            "would leave a point, a photo or an object with too few marks to solve at all."
        )
    return True


def _solve_once(problem: _Problem, x0: np.ndarray, max_iterations: int):
    return least_squares(
        problem.residuals,
        x0,
        jac_sparsity=problem.jac_sparsity(),
        method="trf",
        # the robust kernel is already baked into the residuals, see _huberize
        loss="linear",
        x_scale="jac",
        max_nfev=max_iterations,
        verbose=0,
    )


def bundle_adjust(
    project: Project,
    intrinsics: dict[str, Intrinsics],
    image_ids: list[str],
    point_ids: list[str],
    report: SolveReport,
    max_iterations: int = 200,
) -> tuple[_Problem, np.ndarray]:
    """Refine everything jointly, then do it again knowing what went wrong.

    Two passes rather than one, because the weights that matter can only be
    measured from a solution: how well each body's marks actually agree, and
    which individual clicks are blunders rather than noise. The first pass is
    the measurement, the second is the answer. Both start from the same
    initialisation as far as the gauge prior is concerned, so it still pulls
    towards where the linear stage put things rather than towards pass one's
    mistakes.
    """
    problem = _Problem(project, intrinsics, image_ids, point_ids)
    x0 = problem.pack()

    result = _solve_once(problem, x0, max_iterations)
    outcome = "converged" if result.status > 0 else "hit the iteration limit"
    report.messages.append(
        f"bundle adjustment: {result.nfev} evaluations, cost {result.cost:.4g}, {outcome}"
    )

    s = project.settings
    changed = False
    if s.reweight_instances or s.cull_sigma > 0:
        point_errs, anchor_errs = _observation_errors(problem, result.x)
        problem.line_errs = _line_errors(problem, result.x)
        if s.reweight_instances and problem.anchor_blocks:
            changed |= _reweight_instances(problem, anchor_errs, report)
        if s.cull_sigma > 0:
            changed |= _cull_outliers(problem, point_errs, anchor_errs, report)

    if changed:
        # warm-started from the first pass: the reweighting is a correction to a
        # solution that is already close, not a fresh problem
        second = _solve_once(problem, result.x, max_iterations)
        outcome = "converged" if second.status > 0 else "hit the iteration limit"
        report.messages.append(
            f"second pass after reweighting: {second.nfev} evaluations, "
            f"cost {second.cost:.4g}, {outcome}"
        )
        result = second

    problem.write_back(result.x)
    return problem, result.x


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _fill_statistics(
    project: Project, problem: _Problem, x: np.ndarray, report: SolveReport
) -> None:
    """Per-image and per-point residuals -- the feedback loop that tells the user
    which photo's control lines to go and fix.

    Takes the solver's own parameter vector rather than re-packing, so these
    numbers are guaranteed to describe the solution that was actually found.
    """
    from .geometry import project as project_points

    poses, cams, X, objects = problem.unpack(x)
    sq_all: list[float] = []
    line_sq_all: list[float] = []

    per_image: dict[str, list[tuple[str, float]]] = {}
    per_point: dict[str, list[float]] = {}
    per_instance: dict[str, list[float]] = {}

    def record(
        iid: str,
        label: str,
        observed: np.ndarray,
        reprojected: np.ndarray,
        used: bool = True,
    ) -> float | None:
        """One observation's error, folded into every table that wants it.

        Anchor marks go through here on equal terms with tie points: they are
        clicks in that photo, so they belong in that photo's RMS, and a
        mis-marked body corner should raise the same alarm a mis-clicked tie
        point does.

        A culled mark is still drawn and still listed, flagged -- the user has to
        see the 40-pixel arrow to know what to go and re-click, and a cull that
        hides its own evidence is just data loss. It is kept out of every summary
        statistic though, so the RMS describes the fit that was actually used
        rather than one the solver had already stopped believing.
        """
        err = float(np.linalg.norm(reprojected - observed))
        if not np.isfinite(err):
            return None
        report.residual_lines.append(
            {
                "image": iid,
                "point": label,
                "observed": [float(observed[0]), float(observed[1])],
                "reprojected": [float(reprojected[0]), float(reprojected[1])],
                "error_px": err,
                "culled": not used,
            }
        )
        if not used:
            return None
        per_image.setdefault(iid, []).append((label, err))
        sq_all.append(err**2)
        return err

    for (iid, idx, uv), w in zip(problem.obs, problem.obs_weight):
        R, t = poses[iid]
        cam = cams[problem.group_of_image[iid]]
        proj = project_points(R, t, cam, X[idx])
        for pid, observed, reprojected, keep in zip(problem.obs_pids[iid], uv, proj, w > 0):
            err = record(iid, pid, observed, reprojected, bool(keep))
            if err is not None:
                per_point.setdefault(pid, []).append(err)

    for (iid, inst_id, local, uv, keys), w in zip(problem.anchor_blocks, problem.anchor_weight):
        R, t = poses[iid]
        cam = cams[problem.group_of_image[iid]]
        R_obj, t_obj = objects[inst_id]
        proj = project_points(R, t, cam, local @ R_obj.T + t_obj)
        for key, observed, reprojected, keep in zip(keys, uv, proj, w > 0):
            err = record(iid, key, observed, reprojected, bool(keep))
            if err is not None:
                per_instance.setdefault(inst_id, []).append(err)

    for iid in problem.image_ids:
        rep = report.images.setdefault(iid, ImageReport(id=iid))
        rep.positioned = True
        rep.focal_px = cams[problem.group_of_image[iid]].f
        entries = per_image.get(iid, [])
        rep.n_observations = len(entries)
        if entries:
            errs = np.array([e for _label, e in entries])
            rep.rms_px = float(np.sqrt(np.mean(errs**2)))
            rep.max_px = float(errs.max())
            rep.worst_point = entries[int(np.argmax(errs))][0]

    for inst_id, errs in per_instance.items():
        rep = report.instances.setdefault(inst_id, InstanceReport(id=inst_id))
        arr = np.array(errs)
        rep.n_observations = len(arr)
        rep.rms_px = float(np.sqrt(np.mean(arr**2)))
        rep.max_px = float(arr.max())
        # what this body was actually weighted with, which is the number that
        # explains why it did or did not move the cameras
        rep.sigma_px = float(
            problem.anchor_sigma.get(inst_id, project.settings.sigma_anchor_px)
        )

    per_image_lines: dict[str, list[float]] = {}
    for (iid, axis, line, L, _ln), w in zip(problem.lines, problem.line_weight):
        R, _ = poses[iid]
        cam = cams[problem.group_of_image[iid]]
        nrm = cam.K().T @ line
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        err = abs((L / 2.0) * float((nrm / nn) @ R[:, axis]))
        if w <= 0.0:
            continue   # culled: reported as a count, kept out of the RMS
        per_image_lines.setdefault(iid, []).append(err)
        line_sq_all.append(err**2)
    for iid, errs in per_image_lines.items():
        report.images.setdefault(iid, ImageReport(id=iid)).line_rms_px = float(
            np.sqrt(np.mean(np.square(errs)))
        )

    for pid, errs in per_point.items():
        pr = report.points.setdefault(pid, PointReport(id=pid))
        pr.n_observations = len(errs)
        pr.rms_px = float(np.sqrt(np.mean(np.square(errs))))
        pr.max_px = float(max(errs))
        pr.triangulated = True

    report.n_observations = len(sq_all)
    report.rms_reprojection_px = float(np.sqrt(np.mean(sq_all))) if sq_all else 0.0
    report.rms_line_px = float(np.sqrt(np.mean(line_sq_all))) if line_sq_all else 0.0
    report.n_images_positioned = sum(1 for r in report.images.values() if r.positioned)
    report.n_points_triangulated = sum(1 for r in report.points.values() if r.triangulated)
    report.focals = {gid: float(cams[gid].f) for gid in cams}


def _warn_about_weak_nominals(project: Project, report: SolveReport) -> None:
    """A nominal on a two-view point is an assertion, not a measurement.

    Two rays cross at exactly one place, so a point seen in two photos fits
    both of them perfectly wherever it is put -- there is no redundancy left to
    disagree with a nominal. Hand such a point a wrong coordinate and it goes
    there quietly: the reprojection error stays near zero, the datum residual
    stays near zero, and nothing anywhere says the drawing was wrong.

    Three views is where a nominal starts being checkable, and the difference
    is worth saying out loud, because the two cases look identical in every
    number the report otherwise prints.
    """
    weak = []
    for pt in project.points:
        if not pt.nominal:
            continue
        rep = report.points.get(pt.id)
        if rep is not None and rep.triangulated and rep.n_observations < 3:
            weak.append(pt.id)
    if not weak:
        return
    report.warnings.append(
        f"{len(weak)} nominal'd point(s) are marked in only two photos "
        f"({', '.join(sorted(weak)[:6])}): two rays cross in exactly one place, so these points "
        "fit their photos perfectly wherever the nominal puts them. Their nominals steer the "
        "solve but cannot be checked by it -- mark them in a third photo to make the deviation "
        "readout mean something."
    )


def _warn_if_nominals_dictate(report: SolveReport) -> None:
    """Say so when the nominals are not being checked, only obeyed.

    A nominal with a tolerance far tighter than the photographs can argue with
    does not measure anything: the solve simply bends the weakly-determined
    point positions until every datum is satisfied exactly, and asked-versus-
    achieved reads zero no matter how wrong the drawing is. That looks like a
    perfect result and is actually the absence of one.

    The signature is unmistakable once you look for it -- every nominal landing
    at a tiny fraction of its own stated tolerance, across more of them than the
    seven gauge freedoms could absorb. Three nominal'd points can legitimately
    come out near-perfect (nine residuals against seven freedoms leaves almost
    nothing to disagree with); five or more cannot, unless the tolerance is
    doing the talking.
    """
    rows = [c for c in report.constraints if c.source == "nominal" and c.applied]
    if len(rows) < 15:        # fewer than five nominal'd points: see above
        return
    worst = max((c.sigmas_off for c in rows), default=0.0)
    if worst >= 0.1:
        return
    report.warnings.append(
        f"all {len(rows) // 3} point nominals were met to under a tenth of their stated "
        "tolerance. That is tighter than the photographs can argue with, so the nominals are "
        "setting these coordinates rather than being checked against them -- the deviation "
        "readout will show zero however wrong the drawing is. Loosen the tolerance to roughly "
        "what the reconstruction can actually resolve if you want it to measure anything."
    )


def _describe_constraint(c: Constraint) -> str:
    if isinstance(c, AxisLockConstraint):
        return f"{c.point}.{AXIS_NAMES[c.axis]} = {c.value:g}"
    return f"|{c.point_a} - {c.point_b}| = {c.value:g}"


def _scale_bearing_pairs(constraints: list[Constraint]) -> list[tuple[str, str, int | None, float]]:
    """Every separation the constraints state, as ``(point_a, point_b, axis, value)``.

    Distance constraints state one outright. Axis locks state one too, as soon as
    two of them share an axis: two points pinned 150 mm apart in X *is* a
    distance constraint wearing a different hat, which is easy to do by accident
    when locking several datums to known coordinates. A pair locked to the *same*
    coordinate states a plane rather than a distance and says nothing about scale,
    so it drops out here by having a zero separation.
    """
    pairs: list[tuple[str, str, int | None, float]] = []
    for c in constraints:
        if isinstance(c, DistanceConstraint) and c.value > 0:
            pairs.append((c.point_a, c.point_b, None, float(c.value)))

    by_axis: dict[int, list[tuple[str, float]]] = {}
    for c in constraints:
        if isinstance(c, AxisLockConstraint):
            by_axis.setdefault(c.axis, []).append((c.point, float(c.value)))
    for axis, entries in by_axis.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (point_a, value_a), (point_b, value_b) = entries[i], entries[j]
                separation = abs(value_a - value_b)
                if point_a != point_b and separation > 1e-12:
                    pairs.append((point_a, point_b, axis, separation))
    return pairs


def constraints_imply_scale(constraints: list[Constraint]) -> bool:
    return bool(_scale_bearing_pairs(constraints))


def stated_scale(constraints: list[Constraint], lookup) -> tuple[float | None, int]:
    """The factor the geometry would need for the stated figures to come out exact.

    ``lookup(point_id)`` returns that point's position, or None if it has none.

    This has to be measured *before* the bundle adjustment, which is the whole
    subtlety. Afterwards the refinement has already traded the two sources of
    scale off against each other, so a disagreement shows up as a little of each
    being wrong and nothing reads as the culprit. At initialisation the scene is
    in the objects' units and the constraints have not pulled on it yet, so the
    ratio is exactly the disagreement between them.
    """
    log_ratios: list[float] = []
    weights: list[float] = []
    for point_a, point_b, axis, value in _scale_bearing_pairs(constraints):
        a, b = lookup(point_a), lookup(point_b)
        if a is None or b is None:
            continue
        found = abs(float(a[axis] - b[axis])) if axis is not None else float(np.linalg.norm(a - b))
        if found > 1e-12:
            log_ratios.append(math.log(value / found))
            weights.append(value)

    if not log_ratios:
        return None, 0
    w = np.array(weights, dtype=float)
    r = np.array(log_ratios, dtype=float)
    return float(np.exp(float(np.sum(w * r) / np.sum(w)))), len(log_ratios)


def _fill_constraint_statistics(
    project: Project, problem: _Problem, x: np.ndarray, report: SolveReport
) -> None:
    """Asked versus achieved, per constraint, in world units."""
    _poses, _cams, X, _objects = problem.unpack(x)
    applied = {id(c) for c in problem.constraints}
    violations: list[float] = []

    for index, c in enumerate(problem.declared):
        is_lock = isinstance(c, AxisLockConstraint)
        rep = ConstraintReport(
            index=index,
            type="axis_lock" if is_lock else "distance",
            source=getattr(c, "source", "constraint"),
            description=_describe_constraint(c),
            points=[c.point] if is_lock else [c.point_a, c.point_b],
            target=float(c.value),
            sigma=float(c.sigma),
        )
        if id(c) not in applied:
            rep.applied = False
            rep.note = (
                "the tie point(s) this names were not solved, so it had no effect on anything"
            )
            report.constraints.append(rep)
            continue

        if is_lock:
            rep.achieved = float(X[problem.pt_index[c.point], c.axis])
        else:
            a = X[problem.pt_index[c.point_a]]
            b = X[problem.pt_index[c.point_b]]
            rep.achieved = float(np.linalg.norm(a - b))
        rep.error = rep.achieved - rep.target
        rep.sigmas_off = abs(rep.error) / max(rep.sigma, 1e-12)
        if rep.sigmas_off > 3.0:
            rep.note = (
                f"missed by {abs(rep.error):.4g} ({rep.sigmas_off:.0f}x the {rep.sigma:g} "
                "tolerance). Either the marks disagree with this figure, or two constraints "
                "are pulling against each other."
            )
        violations.append(abs(rep.error))
        report.constraints.append(rep)

    report.constraint_worst = max(violations) if violations else 0.0


def solve(project: Project, run_bundle: bool = True) -> SolveReport:
    """Run the whole pipeline and return a report suitable for the UI."""
    report = SolveReport()

    # Before anything else, because the part's identity pose is the definition
    # of the frame the nominals build rather than a value to be refined, and a
    # project reopened from a hand-edited file could carry anything here.
    project.pin_part()

    report.stage = "calibrating images from control lines"
    intrinsics = calibrate_all(project, report)
    if len(intrinsics) < 2:
        report.messages.append(
            f"only {len(intrinsics)} image(s) calibrated. Each image needs 2+ control lines "
            "on each of at least 2 different world axes."
        )
        return report

    stale = project.prune_stale_anchor_obs()
    if stale:
        report.messages.append(
            f"ignored {stale} anchor mark(s) whose object, model or anchor no longer exists"
        )

    report.stage = "checking what is solvable"
    valid_anchors = set(project.anchor_world_points())
    tables = _observation_tables(project, set(intrinsics), valid_anchors)
    all_tracks = {p.id for p in project.points} | valid_anchors

    keep_images, keep_tracks = _prune_to_solvable(tables, all_tracks)
    dropped_images = sorted(set(intrinsics) - set(keep_images))
    for iid in dropped_images:
        report.images.setdefault(iid, ImageReport(id=iid)).notes.append(
            "fewer than 2 usable marks; cannot place this camera (its rotation is still solved)"
        )
    dropped_points = [p.id for p in project.points if p.id not in keep_tracks]
    for pid in dropped_points:
        report.points.setdefault(pid, PointReport(id=pid)).n_observations = len(
            project.observations_of(pid)
        )

    if len(keep_images) < 2:
        report.messages.append(
            "not enough connected images to solve positions. Every image needs at least 2 marks "
            "it shares with the others -- tie points, or anchors on a known object -- and two "
            "shared per pair is the real minimum, which is far less than COLMAP needs but is "
            "not zero."
        )
        return report

    main, keep_tracks, orphans = _largest_connected_component(tables, keep_images, keep_tracks)
    for iid in orphans:
        report.images.setdefault(iid, ImageReport(id=iid)).notes.append(
            "not connected to the main group by any shared mark"
        )
    if orphans:
        report.warnings.append(
            f"{len(orphans)} image(s) form separate groups with no shared marks linking them to "
            "the main group; they cannot share its coordinate frame. Solved the largest group only."
        )
    if len(main) < 2 or len(keep_tracks) < 2:
        report.messages.append("the largest connected group is too small to solve")
        return report

    report.stage = "linear initialisation"
    positions = initialize_structure(project, intrinsics, main, keep_tracks, tables, report)
    if positions is None:
        return report

    # Known objects are placed before the bundle adjustment, not after: their fit
    # is what sets the scene's scale (and, with a locked object, its origin and
    # orientation), and the refinement needs to start from those coordinates.
    report.stage = "placing known objects"
    metric_from_objects, scale_source = fit_instances(project, positions, main, report)

    # Measured here, between the objects setting the scale and the refinement
    # starting, because that is the only moment the two sources of scale can be
    # compared without one having already been dragged towards the other.
    if metric_from_objects:
        report.datum_scale, _n_pairs = stated_scale(project.all_constraints(), positions.get)

    # Last word on the frame, deliberately after the objects have had theirs:
    # where both a known object and a drawing are present, the drawing is what
    # the part's coordinates are defined by. The disagreement between them is
    # not hidden -- ``datum_scale``, measured just above, is exactly that.
    report.stage = "putting the initialisation in part space"
    align_to_nominals(project, positions, main, report)

    point_ids = [tid for tid in keep_tracks if tid not in valid_anchors]
    for pid in point_ids:
        project.point(pid).xyz = positions[pid]

    problem: _Problem
    if run_bundle:
        report.stage = "bundle adjustment"
        problem, x_final = bundle_adjust(project, intrinsics, main, point_ids, report)
    else:
        problem = _Problem(project, intrinsics, main, point_ids)
        x_final = problem.pack()

    report.stage = "done"
    _fill_statistics(project, problem, x_final, report)
    _fill_constraint_statistics(project, problem, x_final, report)

    _warn_if_nominals_dictate(report)
    _warn_about_weak_nominals(project, report)
    report.n_nominals = sum(1 for pt in project.points if pt.nominal)
    report.part_space = project.in_part_space()
    if report.part_space:
        report.messages.append(
            f"in part space: {report.n_nominals} point nominal(s) fix the origin, orientation "
            "and scale, so these coordinates are the part's own. A CAD model of the part goes "
            "in at the identity."
        )
    elif report.n_nominals:
        report.warnings.append(
            f"{report.n_nominals} point(s) carry a nominal, which is not enough to define part "
            "space -- that needs three or more that do not lie in a straight line. The nominals "
            "still constrain the solve, but the frame is not the part's."
        )

    from_constraints = constraints_imply_scale(problem.constraints)
    report.scale_is_metric = bool(metric_from_objects) or from_constraints
    if metric_from_objects and from_constraints:
        report.scale_source = f"{scale_source}, cross-checked against stated distances"
    elif from_constraints:
        report.scale_source = "stated distances"
    elif metric_from_objects:
        report.scale_source = scale_source
    else:
        report.scale_source = "arbitrary"
        report.messages.append(
            "scale is arbitrary. Add one distance constraint between two tie points, or place a "
            "known object, to get real units."
        )

    # Two independent measurements of the same thing, so their disagreement is
    # worth saying out loud rather than averaging away. Units are far and away
    # the likeliest culprit: a model declared in the wrong units is off by a
    # round factor and every residual still looks perfect.
    if metric_from_objects and report.datum_scale is not None:
        ratio = report.datum_scale
        off = abs(ratio - 1.0)
        if off > 0.005:
            # A round factor is the signature of a units error, so say "10x"
            # rather than "90%" -- the percentage is true and hides the clue.
            factor = ratio if ratio > 1.0 else 1.0 / ratio
            amount = (
                f"a factor of {factor:.3g}x"
                if off > 0.2
                else f"{off * 100:.2f}%"
            )
            report.warnings.append(
                f"the figures you stated and the known object geometry disagree about scale by "
                f"{amount} (the stated figures want the scene "
                f"{'larger' if ratio > 1 else 'smaller'}). The bundle adjustment has split the "
                "difference, so neither is now exactly right. A near-round factor almost always "
                "means the model's declared units are wrong -- check those first, then which tie "
                "points the constraints actually name."
            )
        else:
            report.messages.append(
                f"stated distances and known object geometry agree on scale to "
                f"{off * 100:.2f}% -- two independent measurements, so this is a real check."
            )

    hand = [
        inst
        for inst in project.placed_object_instances()
        if inst.manual and inst.R is not None
    ]
    if hand and not report.scale_is_metric:
        report.warnings.append(
            f"{len(hand)} object(s) are placed by hand in a scene with no real units. A CAD model "
            "is in millimetres and this scene is in nothing in particular, so the model cannot be "
            "made to fit whatever its pose is. Set the scale first -- one distance between two tie "
            "points is enough -- then place by eye."
        )

    over = [c for c in report.constraints if c.applied and c.sigmas_off > 3.0]
    if over:
        report.warnings.append(
            f"{len(over)} constraint(s) came out beyond their stated tolerance, worst "
            f"{report.constraint_worst:.4g} in world units: "
            + ", ".join(c.description for c in over[:4])
            + ("…" if len(over) > 4 else "")
            + ". The solve could not satisfy them and the marks at the same time."
        )
    unapplied = [c for c in report.constraints if not c.applied]
    if unapplied:
        report.warnings.append(
            f"{len(unapplied)} constraint(s) did nothing because the tie points they name were "
            "not solved: " + ", ".join(c.description for c in unapplied[:4])
            + ("…" if len(unapplied) > 4 else "")
        )

    report.rig = build_rig_report(project, report)

    # The part's fit, last, because it reads the solved point positions. Any
    # failure to read the model is reported inside the fit report rather than
    # thrown: a missing OBJ file is not a reason to lose a solve that worked.
    project.pin_part()
    if project.part_instance() is not None:
        from .part import part_fit

        fit = part_fit(project)
        report.part_fit = fit.to_dict()
        report.messages.extend(fit.messages)
        report.warnings.extend(fit.warnings)

    report.ok = True
    return report
