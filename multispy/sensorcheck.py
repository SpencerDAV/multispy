"""Asking the photographs whether each sensor is actually where the file says.

:mod:`multispy.sensors` can only ask a calibration file to contradict itself. It
does that well -- a feature outside its own sensor's field of view is a fault
nothing can argue with -- but two things are beyond it by construction. A 180
degree turn about the aim axis moves neither the sight line nor the field of
view. A transform that is wrong by less than the field of view is wrong by less
than the only ruler that module has. For both, the photographs are the only
referee left, and this is them.

The idea is cheap to state. The cameras are solved, so a body at a known pose
projects into every photograph with no search and no optimisation: one matrix
multiply per vertex. Put the sensor where the calibration file says it is, draw
its wireframe into each photo, and go and look.

**The thing this module is built around not trusting is its own score.**
:mod:`multispy.edgefit` documents what a chamfer cost does on a cluttered
background: it finds confident, well-converged, completely wrong alignments,
because a distance transform has no notion of which lines belong to the subject.
The photographs this was written for are of a pickup rear-frame cell -- tube
frame, safety screen, bare specular metal, every surface an edge. A bookshelf is
tidy by comparison. So a low chamfer cost is never a verdict here, and two
independent referees decide instead:

* **Agreement across views.** A sensor that is really there scores well in every
  photograph that can see it. A coincidental alignment with a piece of tube does
  not repeat from a viewpoint sixty degrees away, because the coincidence was a
  property of that one projection. This is the strong referee and it costs
  nothing once stage one has run for every image.
* **A null distribution per photograph.** The same wireframe is scored at a
  spread of deliberately wrong nearby offsets, which measures what "nothing
  there" scores *in this photograph's own clutter*. The reported figure is a
  z-score against that. An absolute chamfer cost is not comparable between a
  clean wall and a tube frame; a z-score is, because the clutter is in both terms.

**A sensor found strongly but displaced is the useful case.** If the body
matches well a few centimetres from where the file puts it, that is a measured
correction to the calibration rather than a failed check, and it is reported as
one. ``edgefit.refine_image_pose`` moves the camera to fit the model; this does
the opposite, holding every solved camera and moving the body, because here the
cameras are the evidence and the body is the hypothesis.

**Where this contradicts the file, the contradiction is the output.** A sensor
the file calls consistent that no photograph can find, or one the file convicts
that every photograph finds exactly where it claims, is worth more than any
number of agreeing verdicts -- it means one of the two methods is wrong about
that sensor, and which one is a question a person can then go and settle in a
minute. They are collected and reported separately rather than averaged in.

Honest limits, measured or structural:

* **Small in frame defeats it.** A 70 x 168 mm body at four metres is a few dozen
  pixels; there is not enough silhouette there to tell from a bracket. Sensors
  under a minimum projected extent are reported undecided, not absent.
* **Sensors look alike.** Every 917-4012 on the cell is the same body, so a
  wireframe landing on the *wrong* sensor of the same model scores beautifully.
  Cross-view agreement does not help, since the neighbour is in every photo too.
  Only the displacement figure catches it, and only if the neighbour is far
  enough away to show up as one.
* **A confident score is not proof.** Everything here is evidence about whether
  edges are present where a body would put them, which is not the same claim as
  the body being there.
* **The roll about the aim axis is barely visible even here.** These bodies are
  70 x 168 mm in section, so a 180 degree turn about the long axis maps the
  outline onto itself and leaves only off-centre detail. A sensor small in frame
  cannot settle the wrong-sweep-face question at all, and this says so rather
  than scoring it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .edgefit import (
    EdgeField,
    build_edge_field,
    depth_buffer,
    faces_of,
    front_facing_edges,
    load_grayscale,
)
from .part import intrinsics_for, part_vertices_world

MIN_EXTENT_PX = 25.0
"""Smallest projected body size, in original image pixels, worth scoring.

Below this there is not enough silhouette to distinguish a sensor from any other
small dark rectangle, and a score computed from a handful of samples is noise
with a number attached. A 70 x 168 mm body at four metres through a phone camera
is around this size, which is why it is a real limit on this cell and not a
formality.
"""

MIN_SAMPLES = 24
"""Fewest wireframe samples that may produce a score at all."""

MIN_VIEWS = 2
"""Fewest photographs that can see a sensor before a verdict is offered.

One photograph cannot distinguish a body from a coincidence -- that is the entire
reason cross-view agreement is the strong referee -- so a sensor seen once is
reported as seen once and not graded.
"""

FOUND_Z = 2.0
"""z-score above which a photograph is taken to have found something.

Against the per-photo null, not against an absolute cost. 2.0 is the usual
two-sigma convention and is chosen rather than measured: what makes a verdict
here is agreement across views, and this threshold only decides what counts as a
vote.
"""

AGREEMENT = 0.6
"""Fraction of seeing photographs that must agree before a sensor is 'found'."""

NULL_DIRECTIONS = np.array(
    [
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0],
    ]
)
"""Directions the null distribution is sampled along.

Axes and diagonals rather than a random spray, so the null is reproducible and
the same every run -- a threshold compared against a null that moves is not a
threshold. Normalised on use.
"""

FOUND = "found"
ABSENT = "absent"
DISPLACED = "displaced"
UNDECIDED = "undecided"


@dataclass
class PhotoScore:
    """One sensor, one photograph."""

    image_id: str
    visible: bool = False
    reason: str = ""
    extent_px: float = 0.0
    """Projected size of the body in original image pixels -- the number that
    decides whether anything here is worth reading."""
    n_samples: int = 0
    cost: float = 0.0
    """Mean truncated chamfer distance in working pixels, at the claimed pose.
    Reported for completeness and never used as a verdict: it is the quantity
    with the documented bias in it."""
    null_mean: float = 0.0
    null_std: float = 0.0
    z: float = 0.0
    """How much better the claimed pose scores than deliberately wrong poses in
    this photograph's own clutter. Positive is better. This is comparable
    between photographs where ``cost`` is not."""
    best_offset: list[float] = field(default_factory=list)
    best_z: float = 0.0
    """The same z-score at the best nearby pose found. If this is much higher
    than ``z`` and the offset is real, the sensor is somewhere the file does not
    say it is."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "visible": self.visible,
            "reason": self.reason,
            "extent_px": self.extent_px,
            "n_samples": self.n_samples,
            "cost": self.cost,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "z": self.z,
            "best_offset": list(self.best_offset),
            "best_z": self.best_z,
        }


@dataclass
class SensorCheck:
    """What every photograph that can see one sensor, taken together, says."""

    instance_id: str
    serial: str = ""
    session: str = ""
    verdict: str = UNDECIDED
    reason: str = ""
    n_views: int = 0
    """Photographs that could see it at all. Everything below rests on this."""
    z_median: float = 0.0
    agreement: float = 0.0
    """Fraction of seeing photographs that found it. The strong referee: a
    coincidence does not repeat across viewpoints."""
    displacement: list[float] = field(default_factory=list)
    displacement_mm: float = 0.0
    """Median offset, in nominal units, at which the body actually matched. Zero
    unless the views agree on a direction -- a displacement that differs photo to
    photo is noise, not a measurement."""
    photos: list[PhotoScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "serial": self.serial,
            "session": self.session,
            "verdict": self.verdict,
            "reason": self.reason,
            "n_views": self.n_views,
            "z_median": self.z_median,
            "agreement": self.agreement,
            "displacement": list(self.displacement),
            "displacement_mm": self.displacement_mm,
            "photos": [p.to_dict() for p in self.photos],
        }


@dataclass
class CheckReport:
    sensors: list[SensorCheck] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    """Where the photographs and the file disagree. The most valuable output in
    the feature, so it is kept apart rather than averaged into a summary."""
    messages: list[str] = field(default_factory=list)
    n_images: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensors": [s.to_dict() for s in self.sensors],
            "contradictions": list(self.contradictions),
            "messages": list(self.messages),
            "n_images": self.n_images,
        }


def _wireframe_samples(
    verts_world: np.ndarray,
    mesh: Any,
    crease_angle: float,
    unit_scale: float,
    image: Any,
    cam: Any,
    field_scale: float,
    spacing_px: float = 6.0,
    max_samples: int = 1200,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Points along the body's visible silhouette, in world metres, plus extent.

    Deliberately a separate sampler from ``edgefit.build_samples``, which places
    the part at the identity: here every body has its own pose, and the whole
    question is what happens when that pose is moved. Visibility is decided once
    from the claimed pose and then held, for the reason that module gives -- a
    sample set that changes with the offset makes the comparison between offsets
    meaningless, which is exactly the comparison being made.
    """
    edges = mesh.feature_edges(crease_angle)
    if len(edges) == 0:
        return None
    R = np.asarray(image.R, dtype=float)
    t = np.asarray(image.t, dtype=float)
    centre = -R.T @ t

    Xc = verts_world @ R.T + t
    in_front = Xc[:, 2] > 1e-6
    if not in_front.any():
        return None
    z = np.maximum(Xc[:, 2], 1e-9)
    uv = np.stack([cam.f * Xc[:, 0] / z + cam.cx, cam.f * Xc[:, 1] / z + cam.cy], axis=1)

    keep = in_front[edges[:, 0]] & in_front[edges[:, 1]]
    edges = edges[keep]
    if len(edges) == 0:
        return None
    lengths = np.linalg.norm(uv[edges[:, 1]] - uv[edges[:, 0]], axis=1) * field_scale
    usable = np.isfinite(lengths) & (lengths > 1.0)
    edges, lengths = edges[usable], lengths[usable]
    if len(edges) == 0:
        return None

    counts = np.clip(np.ceil(lengths / max(spacing_px, 1.0)).astype(int) + 1, 2, 200)
    if counts.sum() > max_samples:
        order = np.argsort(-lengths)
        running, chosen = 0, []
        for i in order:
            if running + counts[i] > max_samples:
                continue
            chosen.append(i)
            running += counts[i]
        if not chosen:
            return None
        chosen = np.array(sorted(chosen), dtype=int)
        edges, counts = edges[chosen], counts[chosen]

    xyz, direction = [], []
    for e, n in zip(edges, counts):
        a, b = verts_world[e[0]], verts_world[e[1]]
        ts = np.linspace(0.0, 1.0, int(n))[:, None]
        xyz.append(a[None, :] * (1 - ts) + b[None, :] * ts)
        d = uv[e[1]] - uv[e[0]]
        direction.append(np.repeat(d[None, :], int(n), axis=0))
    xyz = np.concatenate(xyz)
    direction = np.concatenate(direction)

    buf = depth_buffer(
        verts_world, faces_of(mesh), image.R, image.t, cam,
        image.width, image.height, field_scale,
    )
    if buf is not None:
        Xs = xyz @ R.T + t
        zs = np.maximum(Xs[:, 2], 1e-9)
        us = (cam.f * Xs[:, 0] / zs + cam.cx) * field_scale
        vs = (cam.f * Xs[:, 1] / zs + cam.cy) * field_scale
        iy = np.clip(np.round(vs).astype(int), 0, buf.shape[0] - 1)
        ix = np.clip(np.round(us).astype(int), 0, buf.shape[1] - 1)
        nearest = buf[iy, ix]
        visible = ~np.isfinite(nearest) | (zs <= nearest * 1.004 + 1e-9)
        if visible.sum() >= MIN_SAMPLES:
            xyz, direction = xyz[visible], direction[visible]

    return xyz, direction


def _cost_at(
    xyz: np.ndarray,
    direction: np.ndarray,
    offset: np.ndarray,
    image: Any,
    cam: Any,
    field: EdgeField,
) -> float:
    """Mean truncated chamfer distance with the body shifted by ``offset``.

    The camera is held and the body moves, which is the inverse of what
    ``edgefit.refine_image_pose`` does, and the right way round here: the solved
    cameras are the evidence and the body's pose is the hypothesis under test.
    """
    R = np.asarray(image.R, dtype=float)
    t = np.asarray(image.t, dtype=float)
    Xc = (xyz + offset) @ R.T + t
    z = np.maximum(Xc[:, 2], 1e-9)
    uv = np.stack(
        [cam.f * Xc[:, 0] / z + cam.cx, cam.f * Xc[:, 1] / z + cam.cy], axis=1
    ) * field.scale
    return float(np.mean(field.lookup(uv, direction)))


def check_sensors(
    project: Any,
    *,
    mesh_loader: Callable[[str], Any],
    instances: Sequence[Any] | None = None,
    calibration: Any | None = None,
    null_factor: float = 0.6,
    search_factors: Sequence[float] = (0.15, 0.3, 0.5),
    progress: Callable[[int, int], None] | None = None,
) -> CheckReport:
    """Score every calibrated body in every photograph that can see it.

    ``mesh_loader`` takes a resolved model path and returns a mesh; it is passed
    in rather than imported so that a caller can cache, which matters because a
    real cell has one body shared by many sensors.

    ``calibration`` is an optional :class:`~multispy.sensors.CalibrationReport`.
    When given, every verdict here is compared against the file's verdict for the
    same sensor and the disagreements are collected -- those are what this feature
    is for.
    """
    report = CheckReport()
    bodies = list(instances) if instances is not None else [
        o for o in project.instances if getattr(o, "calibrated", "") and o.R is not None
    ]
    images = [
        im for im in project.images
        if im.enabled and im.R is not None and im.t is not None
    ]
    report.n_images = len(images)
    if not bodies:
        report.messages.append(
            "no calibrated bodies in this project: import a calibration file and place its "
            "sensors before asking the photographs about them"
        )
        return report
    if len(images) < MIN_VIEWS:
        report.messages.append(
            f"{len(images)} solved photograph(s). Cross-view agreement is the referee that "
            f"makes a chamfer score worth anything here, so with fewer than {MIN_VIEWS} this "
            f"cannot offer a verdict on anything"
        )

    fields: dict[str, EdgeField] = {}
    for im in images:
        try:
            gray = load_grayscale(project.resolve_path(im))
            fields[im.id] = build_edge_field(gray, im.width, im.height, project.settings)
        except Exception as exc:                                   # noqa: BLE001
            report.messages.append(f"{im.id}: could not read the photograph ({exc})")

    scale = float(getattr(project, "nominal_scale", 0.001)) or 0.001
    for n, inst in enumerate(bodies):
        template = project.template_of(inst)
        mesh = None
        if template is not None and template.mesh_path:
            try:
                mesh = mesh_loader(project.resolve_mesh_path(template))
            except Exception:                                      # noqa: BLE001
                mesh = None
        check = SensorCheck(
            instance_id=inst.id,
            serial=str(inst.id).replace("sensor_", ""),
            session=getattr(inst, "calibrated", ""),
        )
        if mesh is None or template is None:
            check.reason = (
                "no model for this body, so there is no wireframe to look for. A position "
                "alone cannot be checked against a photograph: a point has no silhouette"
            )
            report.sensors.append(check)
            continue
        _check_one(project, inst, template, mesh, images, fields, check,
                   null_factor, search_factors, scale)
        report.sensors.append(check)
        if progress is not None:
            progress(n + 1, len(bodies))

    _summarise(report, calibration)
    return report


def _check_one(
    project, inst, template, mesh, images, fields, check: SensorCheck,
    null_factor: float, search_factors: Sequence[float], scale: float,
) -> None:
    local = part_vertices_world(template, mesh)
    verts = local @ np.asarray(inst.R, dtype=float).T + np.asarray(inst.t, dtype=float)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) or 1e-3
    dirs = NULL_DIRECTIONS / np.linalg.norm(NULL_DIRECTIONS, axis=1, keepdims=True)

    for im in images:
        field = fields.get(im.id)
        score = PhotoScore(image_id=im.id)
        if field is None:
            score.reason = "the photograph could not be read"
            check.photos.append(score)
            continue

        cam = intrinsics_for(project, im)
        got = _wireframe_samples(
            verts, mesh, template.crease_angle, template.unit_scale, im, cam, field.scale
        )
        if got is None:
            score.reason = "behind the camera, or no edge of it faces this way"
            check.photos.append(score)
            continue
        xyz, direction = got
        if len(xyz) < MIN_SAMPLES:
            score.reason = f"only {len(xyz)} wireframe samples survived visibility"
            check.photos.append(score)
            continue

        R = np.asarray(im.R, dtype=float)
        t = np.asarray(im.t, dtype=float)
        Xc = xyz @ R.T + t
        z = np.maximum(Xc[:, 2], 1e-9)
        uv = np.stack([cam.f * Xc[:, 0] / z + cam.cx, cam.f * Xc[:, 1] / z + cam.cy], axis=1)
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
            & (uv[:, 0] <= im.width - 1) & (uv[:, 1] <= im.height - 1)
        )
        score.n_samples = int(len(xyz))
        score.extent_px = float(
            np.linalg.norm(uv[inside].max(axis=0) - uv[inside].min(axis=0))
        ) if inside.sum() > 2 else 0.0
        if inside.mean() < 0.5:
            score.reason = f"only {inside.mean() * 100:.0f}% of it falls inside the frame"
            check.photos.append(score)
            continue
        if score.extent_px < MIN_EXTENT_PX:
            score.reason = (
                f"{score.extent_px:.0f} px across, under the {MIN_EXTENT_PX:.0f} px below "
                f"which a body has too little silhouette to tell from any other dark rectangle"
            )
            check.photos.append(score)
            continue

        score.visible = True
        score.cost = _cost_at(xyz, direction, np.zeros(3), im, cam, field)

        # The null: the same wireframe, deliberately misplaced, in this
        # photograph's own clutter. Without it a cost of 3.1 px means nothing,
        # because 3.1 is excellent against a wall and unremarkable against a tube
        # frame where something is always within three pixels of anything.
        null = np.array([
            _cost_at(xyz, direction, d * null_factor * diag, im, cam, field) for d in dirs
        ])
        score.null_mean = float(np.mean(null))
        score.null_std = float(np.std(null))
        if score.null_std < 1e-6:
            score.reason = (
                "every wrong pose scored identically, so this photograph cannot distinguish "
                "a right pose from a wrong one"
            )
            score.visible = False
            check.photos.append(score)
            continue
        score.z = (score.null_mean - score.cost) / score.null_std

        best_cost, best_offset = score.cost, np.zeros(3)
        for factor in search_factors:
            for d in dirs:
                offset = d * factor * diag
                got_cost = _cost_at(xyz, direction, offset, im, cam, field)
                if got_cost < best_cost:
                    best_cost, best_offset = got_cost, offset
        score.best_offset = [float(v) for v in best_offset / scale]
        score.best_z = (score.null_mean - best_cost) / score.null_std
        check.photos.append(score)

    _verdict_for(check, scale)


def _verdict_for(check: SensorCheck, scale: float) -> None:
    seen = [p for p in check.photos if p.visible]
    check.n_views = len(seen)
    if len(seen) < MIN_VIEWS:
        check.verdict = UNDECIDED
        why = "; ".join(sorted({p.reason for p in check.photos if p.reason})) or "not visible"
        check.reason = (
            f"seen in {len(seen)} photograph(s), under the {MIN_VIEWS} needed. One photograph "
            f"cannot tell a body from a coincidence, which is the whole reason agreement "
            f"across views is the referee here. Why not seen: {why}"
        )
        return

    z = np.array([p.z for p in seen])
    check.z_median = float(np.median(z))
    check.agreement = float(np.mean(z >= FOUND_Z))

    offsets = np.array([p.best_offset for p in seen], dtype=float)
    improved = np.array([p.best_z - p.z for p in seen])
    median_offset = np.median(offsets, axis=0)
    # A displacement only counts if the views agree on it. One photograph is
    # nearly blind along its own view ray, so a body can be slid a long way in
    # depth and score the same -- which shows up as large offsets that disagree.
    spread = float(np.median(np.linalg.norm(offsets - median_offset, axis=1)))
    consistent = spread < max(0.5 * float(np.linalg.norm(median_offset)), 1e-9)

    if check.agreement >= AGREEMENT and check.z_median >= FOUND_Z:
        if consistent and float(np.linalg.norm(median_offset)) > 1e-9 and np.median(improved) > 1.0:
            check.verdict = DISPLACED
            check.displacement = [float(v) for v in median_offset]
            check.displacement_mm = float(np.linalg.norm(median_offset))
            check.reason = (
                f"found in {check.agreement * 100:.0f}% of the {len(seen)} photographs that "
                f"see it, but consistently offset: every view improves by moving it about "
                f"{check.displacement_mm:.0f}, and they agree on the direction to within "
                f"{spread:.0f}. That is a measured correction to the calibration, not a "
                f"failed check"
            )
            return
        check.verdict = FOUND
        check.reason = (
            f"found in {check.agreement * 100:.0f}% of the {len(seen)} photographs that see "
            f"it, median z {check.z_median:.1f} against each photograph's own clutter. "
            f"Repeating across viewpoints is what makes this worth more than the chamfer "
            f"cost, which on its own is the quantity with the known bias"
        )
        return

    if check.z_median < 1.0 and check.agreement < 0.25:
        check.verdict = ABSENT
        check.reason = (
            f"nothing is there: across {len(seen)} photographs the claimed pose scores no "
            f"better than deliberately wrong poses in the same clutter (median z "
            f"{check.z_median:.1f}). Either the body is not where the file says, or it is "
            f"not on the cell at all"
        )
        return

    check.verdict = UNDECIDED
    check.reason = (
        f"{len(seen)} photographs see it and they disagree: median z {check.z_median:.1f}, "
        f"{check.agreement * 100:.0f}% of them above the bar. That is the pattern of a "
        f"coincidental edge alignment in some views and not others, which is exactly what "
        f"cross-view agreement exists to catch and exactly why no verdict is offered"
    )


def _summarise(report: CheckReport, calibration: Any | None) -> None:
    counts: dict[str, int] = {}
    for s in report.sensors:
        counts[s.verdict] = counts.get(s.verdict, 0) + 1
    report.messages.append(
        "photographic check: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        + f", over {report.n_images} solved photograph(s)"
    )
    report.messages.append(
        "every figure here is evidence that edges are present where a body would put them, "
        "which is a narrower claim than the body being there. Sensors of the same model are "
        "identical, so a wireframe landing on the neighbouring sensor scores just as well in "
        "every view -- cross-view agreement cannot catch that one, and only the displacement "
        "figure can"
    )

    if calibration is None:
        return
    by_serial = {v.serial: v for v in getattr(calibration, "verdicts", [])}
    for s in report.sensors:
        filed = by_serial.get(s.serial)
        if filed is None:
            continue
        if filed.verdict in ("consistent", "edge") and s.verdict == ABSENT:
            report.contradictions.append(
                f"{s.serial}: the file is self-consistent about it, and no photograph can find "
                f"it. The file can only catch a fault bigger than one field of view, so a "
                f"transform wrong by less than that passes there and fails here -- or the body "
                f"has been removed from the cell since it was calibrated"
            )
        elif filed.verdict in ("suspect", "explained") and s.verdict in (FOUND,):
            report.contradictions.append(
                f"{s.serial}: the file convicts it ({filed.worst_feature} at "
                f"{filed.ratio:.1f}x its field of view) and the photographs find it exactly "
                f"where it claims to be. That points at the feature nominal being wrong rather "
                f"than the sensor transform -- the sensor is where the file says, it is the "
                f"thing it was told to measure that is somewhere else"
            )
        elif filed.verdict in ("consistent", "edge") and s.verdict == DISPLACED:
            report.contradictions.append(
                f"{s.serial}: the file is self-consistent about it, but every photograph that "
                f"sees it puts it about {s.displacement_mm:.0f} from where it claims. That is "
                f"below what the file could ever have caught and is a correction worth making"
            )
    if report.contradictions:
        report.messages.append(
            f"{len(report.contradictions)} sensor(s) where the file and the photographs "
            f"disagree. These are the most valuable lines in the report: one of the two "
            f"methods is wrong about that sensor, and a person can settle which in a minute"
        )
