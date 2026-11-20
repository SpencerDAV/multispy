"""The part itself: dropping its CAD model in at the origin, and reporting
whether the reconstruction agrees with it.

Everything else in this tool measures something unknown. This module does the
opposite: it takes a shape that is known in advance, puts it where the drawing
says it is, and asks the photographs whether they agree. That only becomes a
meaningful question once the project is in part space -- once three or more
point nominals have fixed the origin, the orientation and the scale, so that
the solved coordinates *are* the part's own coordinates. Before that the part
has no known pose and the honest answer is to say so rather than to place it
somewhere plausible.

Two things live here:

* :func:`part_fit` -- the deliverable. Per nominal'd point, how far the
  reconstruction ended up from the model's own geometry, plus a summary and the
  units check.

* :func:`place_part` and :func:`project_part_wireframe` -- stage one of the
  auto-fit. There is no search and no optimisation in it, which is the point:
  with the part at the identity and the cameras solved, projecting its
  wireframe into every photograph is one matrix multiply per vertex. If it
  lands on the real part, the whole reconstruction is confirmed by eye in a way
  no table of residuals can manage.

Units are the trap this module exists to guard. A template carries ``units`` and
``unit_scale``; the project carries ``nominal_units`` and ``nominal_scale``;
the two are independent and can disagree. They are *allowed* to disagree -- a
model exported in inches against a drawing dimensioned in millimetres is
perfectly legitimate, because both resolve to metres -- so this module never
rescales anything to make them match. It converts both to metres, compares, and
says loudly when the result looks like a declared-units mistake. The reason for
the noise is specific: a factor of 1000 applied silently would leave every
deviation in the report reading as a clean zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Intrinsics, project as project_points, similarity_fit
from .mesh import Mesh, load_mesh
from .project import ImageView, ObjectInstance, ObjectTemplate, Project

def jsonable(value: Any) -> Any:
    """``value`` with every non-finite float turned into ``None``.

    Both reports here use NaN to mean "not measured", which is the right thing
    in Python and an outright bug at the boundary: ``json.dumps`` writes a bare
    ``NaN``, which is not JSON, and the browser's ``JSON.parse`` rejects the
    whole response -- so one unmeasured surface distance on one unsolved point
    took down the entire solve result. Converted here, in ``to_dict``, because
    that is the one place every payload passes through.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


SURFACE_CHUNK = 4096
"""Triangles per block in the point-to-surface sweep, to bound peak memory on a
production mesh without making the loop slow."""


# --------------------------------------------------------------------------
# point to surface
# --------------------------------------------------------------------------


def _point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Distance from one point to each of ``tri`` (an ``(n, 3, 3)`` array).

    The full seven-region clamp rather than a distance-to-plane, because a
    nominal'd feature is usually on an edge or a corner of the part and the
    plane of the nearest facet is nowhere near it there -- a plane distance
    would read zero for a point floating off the end of a face.
    """
    a, b, c = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    ab, ac = b - a, c - a

    def dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", u, v)

    def safe(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return num / np.where(np.abs(den) > 1e-30, den, 1.0)

    ap = p[None, :] - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p[None, :] - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p[None, :] - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    closest = np.empty_like(a)
    done = np.zeros(len(a), dtype=bool)

    def assign(mask: np.ndarray, value: np.ndarray) -> None:
        use = mask & ~done
        if use.any():
            closest[use] = value[use]
            done[use] = True

    # The seven regions in Ericson's order. They are mutually exclusive in exact
    # arithmetic, so the sequence only decides degenerate ties -- but keeping it
    # canonical means a sliver triangle cannot fall through to the interior
    # branch, whose barycentric denominator is what goes to zero there.
    assign((d1 <= 0) & (d2 <= 0), a)
    assign((d3 >= 0) & (d4 <= d3), b)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + safe(d1, d1 - d3)[:, None] * ab)
    assign((d6 >= 0) & (d5 <= d6), c)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + safe(d2, d2 - d6)[:, None] * ac)
    assign(
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
        b + safe(d4 - d3, (d4 - d3) + (d5 - d6))[:, None] * (c - b),
    )
    total = va + vb + vc
    assign(
        np.ones(len(a), dtype=bool),
        a + safe(vb, total)[:, None] * ab + safe(vc, total)[:, None] * ac,
    )

    return np.linalg.norm(p[None, :] - closest, axis=1)


def surface_distances(mesh: Mesh, points: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Shortest distance from each of ``points`` to the mesh surface.

    ``scale`` multiplies the mesh's own coordinates, which is how model units
    become world metres. Brute force over the triangles in blocks: the number of
    query points here is the number of nominal'd features, which is a handful,
    so an acceleration structure would be more code than it saves.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if mesh.n_faces == 0 or len(points) == 0:
        return np.full(len(points), float("nan"))
    verts = np.asarray(mesh.vertices, dtype=float) * float(scale)
    faces = np.asarray(mesh.faces, dtype=int)
    out = np.full(len(points), np.inf)
    for start in range(0, len(faces), SURFACE_CHUNK):
        tri = verts[faces[start : start + SURFACE_CHUNK]]
        for i, p in enumerate(points):
            out[i] = min(out[i], float(_point_triangle_distance(p, tri).min()))
    return out


def nearest_vertices(mesh: Mesh, points: np.ndarray, scale: float = 1.0):
    """``(indices, distances)`` of the nearest model vertex to each point."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    verts = np.asarray(mesh.vertices, dtype=float) * float(scale)
    if len(verts) == 0 or len(points) == 0:
        return np.zeros(len(points), dtype=int), np.full(len(points), float("nan"))
    d = np.linalg.norm(points[:, None, :] - verts[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    return idx, d[np.arange(len(points)), idx]


# --------------------------------------------------------------------------
# the fit report
# --------------------------------------------------------------------------


@dataclass
class PartFitPoint:
    """One nominal'd point, checked three ways.

    Against the drawing (``deviation``), against the model's surface
    (``surface_distance``), and -- before any photograph is involved -- the
    drawing against the model (``nominal_surface_distance``). The three answer
    different questions and a project can fail any one of them while passing the
    other two, which is why none of them is dropped in favour of a single
    number.
    """

    point: str
    label: str = ""
    nominal: list[float] = field(default_factory=list)
    """Where the drawing says it is, in the project's nominal units."""
    solved: list[float] = field(default_factory=list)
    """Where the reconstruction put it, same units."""
    deviation: list[float] = field(default_factory=list)
    deviation_mag: float = 0.0
    """``solved - nominal``. The same quantity the constraint table reports, kept
    here so one table answers "does the part fit" without cross-referencing."""
    surface_distance: float = float("nan")
    """How far the solved position sits from the nearest point on the model's
    surface. This is what the CAD model adds: the constraint table can only
    check a point against the three numbers typed for it, while this checks it
    against the whole shape -- including the parts of the shape nobody typed."""
    nominal_surface_distance: float = float("nan")
    """How far the *drawing's* coordinate sits from the model's surface.

    Involves no photograph at all, and is the first thing to read. If this is
    large the nominal is not on the part, so neither the deviation above nor
    anything downstream of it means anything -- the drawing and the CAD model
    disagree, or one of them is in the wrong units."""
    vertex_distance: float = float("nan")
    vertex: list[float] | None = None
    """Nearest model vertex to the solved position, and how far away it is.
    Useful when the nominal was meant to name a corner: a surface distance of
    zero with a large vertex distance means the point landed on the right face
    but the wrong place on it."""
    n_views: int = 0
    checkable: bool = False
    """Marked in three or more photographs. Two rays cross in exactly one place,
    so a two-view point sits wherever its nominal puts it and its deviation is
    structurally zero however wrong the drawing is."""
    note: str = ""


@dataclass
class PartFitReport:
    ok: bool = False
    stage: str = "not started"
    template: str = ""
    template_name: str = ""
    n_points: int = 0
    n_checkable: int = 0

    # ---- units, reported whether or not they look wrong
    units: str = ""
    unit_scale: float = 0.0
    nominal_units: str = ""
    nominal_scale: float = 0.0
    unit_ratio: float = 1.0
    """``unit_scale / nominal_scale``. 1.0 when the two declarations agree."""
    units_agree: bool = True
    units_mismatch: bool = False
    """The two declarations disagree *and* the part's size says the model is the
    one that is wrong. Kept separate from ``units_agree`` because a model in
    inches against a drawing in millimetres disagrees legitimately."""

    # ---- summary deviations, in nominal units
    rms_deviation: float = 0.0
    max_deviation: float = 0.0
    worst_point: str | None = None
    rms_surface: float = 0.0
    max_surface: float = 0.0
    implied_scale: float = 1.0
    """Scale of the best similarity from the nominals to the solved positions.
    1.0 means the reconstruction agrees with the drawing about size."""
    model_span: float = 0.0
    """The model's bounding-box diagonal in nominal units, after its own
    declared scale is applied."""
    nominal_span: float = 0.0
    """The nominals' bounding-box diagonal, same units. Comparable to
    ``model_span`` without needing any point to correspond to any vertex, which
    is what makes it able to catch a factor-of-1000 blunder on its own."""
    span_ratio: float = 0.0

    points: list[PartFitPoint] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in vars(self).items() if k != "points"}
        d["points"] = [vars(p) for p in self.points]
        return jsonable(d)


def _load_part_mesh(project: Project, template: ObjectTemplate) -> Mesh:
    path = project.resolve_mesh_path(template)
    if path is None:
        raise ValueError("the part's template has no model file")
    if not Path(path).exists():
        raise FileNotFoundError(f"the part's model file is missing: {path}")
    return load_mesh(path)


def part_fit(project: Project, mesh: Mesh | None = None) -> PartFitReport:
    """How well the reconstruction agrees with the part's own CAD geometry.

    Refuses rather than guesses in the two cases where an answer would be
    misleading: no part designated, and a project that is not in part space. In
    the second case the part's pose is simply not known -- there is nothing
    wrong with the CAD model or the photographs, the frame just is not the
    part's yet -- and placing it at the origin anyway would produce a table of
    enormous deviations that says nothing about the part.
    """
    report = PartFitReport()
    inst = project.part_instance()
    if inst is None:
        report.stage = "no part designated"
        report.messages.append(
            "No CAD model is marked as the part. Add the part's own model and mark it as the "
            "part; in part space it needs no anchors and no placement -- it goes in at the origin."
        )
        return report

    template = project.template_of(inst)
    if template is None:
        report.stage = "the part's template is missing"
        report.warnings.append(f"template {inst.template!r} is not in this project")
        return report

    report.template = template.id
    report.template_name = template.name or template.id
    report.units = template.units
    report.unit_scale = float(template.unit_scale)
    report.nominal_units = project.nominal_units
    report.nominal_scale = float(project.nominal_scale)
    ratio = project.part_unit_ratio() or 1.0
    report.unit_ratio = float(ratio)
    report.units_agree = bool(abs(math.log(max(ratio, 1e-30))) < 1e-6)

    if mesh is None:
        try:
            mesh = _load_part_mesh(project, template)
        except (ValueError, FileNotFoundError) as exc:
            report.stage = "the part's model could not be read"
            report.warnings.append(str(exc))
            return report

    nominal_scale = max(float(project.nominal_scale), 1e-30)
    lo, hi = mesh.bounds()
    report.model_span = float(np.linalg.norm(hi - lo) * template.unit_scale / nominal_scale)

    # The units verdict comes first and is loud, because every number below it
    # inherits the mistake. It is stated as a factor rather than a percentage on
    # purpose: "1000x" names the cause, "99.9% too small" hides it.
    if not report.units_agree:
        factor = ratio if ratio > 1.0 else 1.0 / ratio
        report.warnings.append(
            f"the part's model declares {template.units} ({template.unit_scale:g} m per model "
            f"unit) while this project's nominals are in {project.nominal_units} "
            f"({project.nominal_scale:g} m each) -- a factor of {factor:.6g}x between them. "
            "Nothing here has been rescaled to hide that. If the two declarations are both "
            "correct (a model in inches against a drawing in millimetres, say) this is fine and "
            "the deviations below are real; if one of them is wrong, every deviation below is "
            "meaningless and the wireframe will land the same factor out in every photograph."
        )

    nominal_points = [p for p in project.points if p.nominal and len(p.nominal) == 3]
    report.n_points = len(nominal_points)
    if not project.in_part_space():
        report.stage = "not in part space"
        report.warnings.append(
            f"{report.n_points} point nominal(s) is not enough to put this project in part space, "
            "which needs three or more that do not lie in a straight line. Until then the part has "
            "no known pose, so there is nothing to check it against -- this is not a problem with "
            "the model."
        )
        return report

    rows: list[PartFitPoint] = []
    nom_world: list[np.ndarray] = []
    solved_world: list[np.ndarray] = []
    for pt in nominal_points:
        if any(v is None for v in pt.nominal):
            continue
        nom = np.array([float(v) for v in pt.nominal], dtype=float)
        row = PartFitPoint(point=pt.id, label=pt.label, nominal=nom.tolist())
        row.n_views = len(project.observations_of(pt.id))
        row.checkable = row.n_views >= 3
        nom_m = nom * nominal_scale
        row.nominal_surface_distance = float(
            surface_distances(mesh, nom_m, template.unit_scale)[0] / nominal_scale
        )
        if pt.xyz is None:
            row.note = "not solved -- it is not in the connected group the solve reached"
            rows.append(row)
            continue
        solved = np.asarray(pt.xyz, dtype=float)
        row.solved = (solved / nominal_scale).tolist()
        dev = (solved - nom_m) / nominal_scale
        row.deviation = dev.tolist()
        row.deviation_mag = float(np.linalg.norm(dev))
        row.surface_distance = float(
            surface_distances(mesh, solved, template.unit_scale)[0] / nominal_scale
        )
        vi, vd = nearest_vertices(mesh, solved, template.unit_scale)
        row.vertex_distance = float(vd[0] / nominal_scale)
        row.vertex = (np.asarray(mesh.vertices[vi[0]], dtype=float)).tolist()
        if not row.checkable:
            row.note = (
                f"marked in {row.n_views} photo(s): its deviation is structurally near zero and "
                "checks nothing. Mark it in a third photo."
            )
        nom_world.append(nom_m)
        solved_world.append(solved)
        rows.append(row)

    report.points = rows
    report.n_checkable = sum(1 for r in rows if r.checkable and r.solved)
    solved_rows = [r for r in rows if r.solved]
    if solved_rows:
        devs = np.array([r.deviation_mag for r in solved_rows])
        report.rms_deviation = float(np.sqrt(np.mean(devs**2)))
        report.max_deviation = float(devs.max())
        report.worst_point = solved_rows[int(np.argmax(devs))].point
        surf = np.array([r.surface_distance for r in solved_rows], dtype=float)
        finite = surf[np.isfinite(surf)]
        if len(finite):
            report.rms_surface = float(np.sqrt(np.mean(finite**2)))
            report.max_surface = float(finite.max())

    if len(nom_world) >= 3:
        fit = similarity_fit(np.array(nom_world), np.array(solved_world), allow_scale=True)
        if not fit.degenerate and fit.scale > 0:
            report.implied_scale = float(fit.scale)
        P = np.array(nom_world) / nominal_scale
        report.nominal_span = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))

    # The size cross-check. Independent of any correspondence between a nominal
    # and a vertex, which is what lets it accuse the model's declared units
    # rather than merely noting that two declarations differ: the nominals are
    # read off the same drawing the part was made to, so if the model is a
    # thousand times their extent, it is the model that is wrong.
    if report.nominal_span > 0 and report.model_span > 0:
        report.span_ratio = report.model_span / report.nominal_span
        if report.span_ratio > 1.0 / 0.04:
            report.units_mismatch = True
            report.warnings.append(
                f"the part model measures {report.model_span:.4g} {project.nominal_units} across "
                f"while the nominal'd features span {report.nominal_span:.4g} -- the model is "
                f"{report.span_ratio:.4g}x the size of the thing it is supposed to be. The "
                f"declared model units ({template.units}) are almost certainly wrong. Fix those "
                "before reading any deviation here: a wrong unit scale is not a fit error and the "
                "numbers below cannot tell you anything while it stands."
            )
        elif report.span_ratio < 0.5:
            report.warnings.append(
                f"the nominal'd features span {report.nominal_span:.4g} {project.nominal_units} "
                f"but the whole model is only {report.model_span:.4g} across. Either the model is "
                "a sub-assembly of the thing the drawing dimensions, or its declared units are "
                "wrong."
            )

    if abs(report.implied_scale - 1.0) > 0.02:
        report.warnings.append(
            f"the reconstruction came out {report.implied_scale:.4f}x the size the nominals state. "
            "In part space that should be 1.0000 by construction, so this means the nominals are "
            "fighting each other or one of them is on the wrong feature."
        )

    bad_nominals = [
        r
        for r in rows
        if np.isfinite(r.nominal_surface_distance)
        and r.nominal_surface_distance > max(0.02 * max(report.model_span, 1e-9), 1e-9)
    ]
    if bad_nominals and not report.units_mismatch:
        report.warnings.append(
            f"{len(bad_nominals)} nominal(s) do not lie on the model's surface, worst "
            f"{max(r.nominal_surface_distance for r in bad_nominals):.4g} "
            f"{project.nominal_units} off it: "
            + ", ".join(r.point for r in bad_nominals[:5])
            + ". That comparison involves no photograph, so it is the drawing and the CAD model "
            "disagreeing with each other. Settle that first."
        )

    if report.n_checkable < 3:
        report.warnings.append(
            f"only {report.n_checkable} of {len(solved_rows)} nominal'd point(s) are marked in "
            "three or more photographs. The rest sit exactly where their nominals put them "
            "whatever the part really looks like, so the summary below is based on fewer "
            "independent checks than it appears to be."
        )

    report.ok = bool(solved_rows)
    report.stage = "done" if report.ok else "no nominal'd point was solved"
    if report.ok:
        report.messages.append(
            f"the part fits {len(solved_rows)} nominal'd point(s) to "
            f"{report.rms_deviation:.4g} {project.nominal_units} RMS "
            f"({report.max_deviation:.4g} worst, at {report.worst_point}), and sits "
            f"{report.rms_surface:.4g} {project.nominal_units} RMS off the model's surface."
        )
    return report


# --------------------------------------------------------------------------
# stage one -- place from the nominals
# --------------------------------------------------------------------------


@dataclass
class ImagePlacement:
    """What the part's wireframe does in one photograph."""

    image: str
    positioned: bool = False
    """That camera has a solved pose. Without one there is nothing to project
    through and the part cannot be drawn in this photo at all."""
    n_vertices: int = 0
    n_in_front: int = 0
    n_inside: int = 0
    """Model vertices landing inside the frame. A part that projects entirely
    off the edge is not a failure of the placement -- the photograph just does
    not show it -- but it is worth distinguishing from one that is there."""
    coverage: float = 0.0
    """Fraction of the frame's diagonal the projection spans. A part filling the
    frame can be checked by eye; one three pixels across cannot."""
    bbox: list[float] | None = None
    centre_px: list[float] | None = None
    mark_rms_px: float | None = None
    """Reprojection RMS of this photo's own clicked marks, carried along so the
    before/after of a stage-two refinement has something to be measured against."""
    note: str = ""


@dataclass
class PartPlacement:
    ok: bool = False
    stage: str = "not started"
    part_space: bool = False
    pinned: bool = False
    """The part's pose was set to the identity by this call. Always true when a
    part exists and the project is in part space -- it is recorded because the
    whole claim of stage one is that no search happened."""
    images: list[ImagePlacement] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in vars(self).items() if k != "images"}
        d["images"] = [vars(i) for i in self.images]
        return jsonable(d)


def intrinsics_for(project: Project, image: ImageView) -> Intrinsics:
    """The camera this image was solved with, from its group.

    Falls back to the image's own centre for a principal point and to a 50 mm
    equivalent focal if the group has none, so a wireframe can still be drawn
    for an image whose group never got a focal length -- flagged, not silently.
    """
    group = project.group_of(image)
    f = group.f if group.f is not None else 1.2 * max(image.width, image.height)
    cx, cy = group.principal_point_for(image.width, image.height)
    return Intrinsics(f=float(f), cx=cx, cy=cy, k1=float(group.k1))


def part_vertices_world(template: ObjectTemplate, mesh: Mesh) -> np.ndarray:
    """The part's vertices in world metres.

    One multiply, because the part is at the identity. Written out as a function
    anyway so that the one place where ``unit_scale`` is applied to the part is
    findable, and so a future non-identity part pose has somewhere to go.
    """
    return np.asarray(mesh.vertices, dtype=float) * float(template.unit_scale)


def project_part_wireframe(
    project: Project,
    image: ImageView,
    mesh: Mesh,
    template: ObjectTemplate,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(uv, visible)``: every part vertex in this image's pixels, and whether
    it is in front of the camera.

    ``edges`` is accepted only so callers that already thinned the wireframe can
    pass the subset they intend to use; it does not change the projection.
    """
    if image.R is None or image.t is None:
        raise ValueError(f"image {image.id} has no solved pose")
    cam = intrinsics_for(project, image)
    Xw = part_vertices_world(template, mesh)
    Xc = Xw @ np.asarray(image.R, dtype=float).T + np.asarray(image.t, dtype=float)
    visible = Xc[:, 2] > 1e-6
    uv = project_points(np.asarray(image.R, dtype=float), np.asarray(image.t, dtype=float), cam, Xw)
    return uv, visible


def mark_reprojection_rms(project: Project, image: ImageView) -> float | None:
    """RMS reprojection error of this image's clicked tie points.

    The independent referee for stage two. It is computed here rather than read
    out of a solve report because a refinement changes the pose *after* the
    solve, so the report's number describes a pose that no longer exists.
    """
    if image.R is None or image.t is None:
        return None
    cam = intrinsics_for(project, image)
    pts, obs = [], []
    for pid, uv in image.observations.items():
        pt = project.point(pid)
        if pt is None or pt.xyz is None:
            continue
        pts.append(np.asarray(pt.xyz, dtype=float))
        obs.append([float(uv[0]), float(uv[1])])
    if not pts:
        return None
    proj = project_points(
        np.asarray(image.R, dtype=float), np.asarray(image.t, dtype=float), cam, np.array(pts)
    )
    err = np.linalg.norm(proj - np.array(obs), axis=1)
    err = err[np.isfinite(err)]
    if not len(err):
        return None
    return float(np.sqrt(np.mean(err**2)))


def place_part(project: Project, mesh: Mesh | None = None) -> PartPlacement:
    """Stage one: put the part at the origin and report where it lands.

    No search, no image data, no optimisation. In part space the part's pose is
    already known -- that is what part space *is* -- so this is geometry and
    nothing else, which is why it is the half of the auto-fit that can be relied
    on. Everything it reports is a consequence of the cameras the solve already
    produced.
    """
    out = PartPlacement()
    inst = project.part_instance()
    if inst is None:
        out.stage = "no part designated"
        out.messages.append("Mark one of the models as the part first.")
        return out

    template = project.template_of(inst)
    if template is None:
        out.stage = "the part's template is missing"
        out.warnings.append(f"template {inst.template!r} is not in this project")
        return out

    out.part_space = project.in_part_space()
    if not out.part_space:
        out.stage = "not in part space"
        out.warnings.append(
            "The part is only at the origin because the nominals put the reconstruction in the "
            "part's own frame, and they have not. Give three or more points -- not in a straight "
            "line -- the coordinates the drawing states, solve, and the part places itself."
        )
        return out

    if mesh is None:
        try:
            mesh = _load_part_mesh(project, template)
        except (ValueError, FileNotFoundError) as exc:
            out.stage = "the part's model could not be read"
            out.warnings.append(str(exc))
            return out

    project.pin_part()
    out.pinned = True

    ratio = project.part_unit_ratio() or 1.0
    if abs(math.log(max(ratio, 1e-30))) > 1e-6:
        out.warnings.append(
            f"the model's declared units ({template.units}) and the nominals' "
            f"({project.nominal_units}) differ by a factor of "
            f"{(ratio if ratio > 1 else 1 / ratio):.6g}x. The wireframe below is drawn at the "
            "model's declared scale, unchanged, so if that declaration is wrong it will miss the "
            "part by exactly that factor -- which is the easiest thing in this whole tool to spot "
            "by eye and the hardest to spot in a table."
        )

    n_off_frame = 0
    for im in project.images:
        entry = ImagePlacement(image=im.id, n_vertices=int(mesh.n_vertices))
        if im.R is None or im.t is None:
            entry.note = "no solved pose for this camera, so the part cannot be drawn in it"
            out.images.append(entry)
            continue
        entry.positioned = True
        uv, visible = project_part_wireframe(project, im, mesh, template)
        entry.n_in_front = int(visible.sum())
        good = visible & np.isfinite(uv).all(axis=1)
        if good.any():
            P = uv[good]
            inside = (
                (P[:, 0] >= 0) & (P[:, 0] < im.width) & (P[:, 1] >= 0) & (P[:, 1] < im.height)
            )
            entry.n_inside = int(inside.sum())
            lo, hi = P.min(axis=0), P.max(axis=0)
            entry.bbox = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
            centre = P.mean(axis=0)
            entry.centre_px = [float(centre[0]), float(centre[1])]
            diag = math.hypot(im.width, im.height)
            entry.coverage = float(np.linalg.norm(hi - lo) / diag) if diag else 0.0
        entry.mark_rms_px = mark_reprojection_rms(project, im)
        if entry.n_inside == 0:
            n_off_frame += 1
            entry.note = (
                "the part projects entirely outside this frame. Either this photo does not show "
                "it, or this camera's pose is wrong"
            )
        elif entry.coverage < 0.02:
            entry.note = (
                f"the part spans only {entry.coverage * 100:.1f}% of the frame here -- too small "
                "to confirm the fit by eye"
            )
        out.images.append(entry)

    placed = [i for i in out.images if i.positioned]
    out.ok = bool(placed)
    out.stage = "done" if out.ok else "no camera has a solved pose yet"
    if out.ok:
        out.messages.append(
            f"the part's wireframe is projected into {len(placed)} photo(s) from its known pose at "
            "the origin. No anchors were marked and no search was run: with the reconstruction in "
            "part space this is one matrix multiply per vertex. If the wireframe lands on the real "
            "part in every photo, the cameras and the nominals agree with the CAD model."
        )
    if n_off_frame:
        out.warnings.append(
            f"{n_off_frame} photo(s) have the part entirely off-frame. If those photos clearly do "
            "show the part, their camera poses are wrong, not the part's."
        )
    unpositioned = [i.image for i in out.images if not i.positioned]
    if unpositioned:
        out.warnings.append(
            f"{len(unpositioned)} photo(s) have no solved pose, so the part cannot be drawn in "
            "them: " + ", ".join(unpositioned[:6])
        )
    return out
