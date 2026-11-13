"""Stage two of the auto-fit: sliding the part's projected wireframe onto the
edges that are actually in the photograph.

Stage one (:mod:`multispy.part`) is geometry and cannot be wrong in any way the
solve was not already wrong. This is a different kind of claim entirely. It
looks at pixels, decides which of them are edges, and moves a camera pose to
make the wireframe sit on them. On a clean background against a part with strong
silhouettes that works well. On a busy background it can find a confident,
well-converged, completely wrong alignment -- a bookshelf behind the part is a
grid of long straight high-contrast lines, and a chamfer objective has no notion
of which lines belong to the subject.

**What it is for, measured rather than assumed.** On a synthetic scene of a
machined plate with ground truth available, the alignment's own accuracy floor
is about half a *working* pixel, which is the grid of the Euclidean distance
transform: halving the working resolution doubled the error, exactly as a
quantisation limit does. The consequences are not what one would hope:

* A camera pose five to ten pixels off is pulled back to within about 1.5 px.
  That is the case this is worth having -- a visibly misplaced wireframe becomes
  a correctly placed one, and the photo's own clicked marks confirm it
  independently (9.9 px to 1.6 px, on marks that were no part of the objective).
* A camera pose that is already good is made **worse**. From a solved pose
  sitting 0.5 px from the truth, the alignment confidently moved it to 1.9 px
  while reporting that it had improved the edge fit threefold -- because by its
  own measure it had. The bundle adjustment is simply more accurate than a
  distance transform on a pixel grid, and no amount of tuning changes that.
* A pose tens of pixels out is beyond it: the correction is bounded on purpose,
  and an unbounded one would just find different edges.

So this is a **recovery** tool, not a refinement, and it is named one in the UI.
The thing that makes it safe to ship is not the chamfer cost -- that cost is the
quantity with the bias in it -- but the referee below, which gets the final say
and in testing separated the three cases above cleanly every time.

So the whole module is built around not being trusted:

* **It is off by default.** ``Settings.edge_refine`` gates it, and it is applied
  one image at a time rather than to a project.
* **Every pose it replaces is kept.** ``ImageView.R_unrefined`` holds what the
  solve produced, so an undo is exact and needs no re-solve.
* **The decision is made by a referee it did not optimise, not by its own
  score.** Acceptance requires that the image's own clicked marks do not
  reproject worse. Those marks were no part of the objective, so they are an
  independent measurement -- and since the chamfer cost was measured to be
  biased, they are not a safety net but the actual decision rule. An image with
  no clicked marks therefore cannot be refined at all: there would be nothing to
  adjudicate, and the thing asking for the change is the thing with the known
  bias.
* **It refuses rather than guesses.** Too few wireframe samples finding any edge
  at all, a correction larger than a correction should be, an improvement inside
  the noise: each returns the original pose and says why.

Three techniques do the actual work of resisting clutter, and all three matter:

1. **Orientation binning.** A separate distance field per edge direction. A
   vertical window frame is a strong edge but it is not a candidate for a
   horizontal wireframe edge, and this is what lets the cost say so. It is the
   single biggest difference between this working and not working on a real
   photograph.
2. **Truncation.** Chamfer distances are clipped. Unclipped, one wireframe edge
   that found nothing drags the entire pose towards whatever distant clutter
   happens to be nearest; clipped, an edge with no support simply stops voting.
3. **Back-face culling.** Only edges with a front-facing neighbour are used. A
   feature-edge wireframe contains the far side of the part, which has no
   support in the image and would otherwise be fitted to the near side's edges.

The dependencies are numpy, scipy and pillow, which is what the project already
has: a Gaussian blur and Sobel gradients from ``scipy.ndimage``, its Euclidean
distance transform for the chamfer field, and ``scipy.optimize.least_squares``
over the six pose parameters.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from .geometry import Intrinsics, so3_exp
from .mesh import Mesh
from .part import intrinsics_for, jsonable, mark_reprojection_rms, part_vertices_world
from .project import ImageView, ObjectTemplate, Project, Settings

WORKING_MAX_PX = 2000
"""Longest side of the image the gradients are computed on.

Downscaling is partly for speed and partly because a full-resolution photograph
is full of texture that reads as edges at the pixel level and as nothing at the
scale of the part's silhouette. But it is also the accuracy limit: the distance
transform lives on this grid, so the alignment cannot be better than about half
a pixel of it. Halving this doubled the measured error. Raising it further buys
less than it costs, because the texture comes back."""

SAMPLE_SPACING_PX = 4.0
"""Spacing of samples along a projected wireframe edge, in working pixels."""

MAX_SAMPLES = 4000
"""Cap on total samples per image, so a dense wireframe cannot make one
evaluation slow enough to matter."""

MIN_SUPPORTED_FRACTION = 0.25
"""Fraction of samples that must find an edge inside the truncation radius
before a refinement is considered to have evidence behind it. Below this the
part is not visibly in the photograph as far as the gradients are concerned, and
the honest answer is to leave the pose alone."""


# --------------------------------------------------------------------------
# the edge field
# --------------------------------------------------------------------------


@dataclass
class EdgeField:
    """Orientation-binned chamfer distance fields for one photograph.

    ``dist[k]`` is the distance, in working pixels, from each pixel to the
    nearest *detected edge whose own direction falls in bin k*. Looking a
    wireframe edge up in the bin matching its direction is what makes the cost
    care about agreement rather than mere proximity.
    """

    scale: float
    """Working pixels per original image pixel."""
    width: int
    height: int
    dist: np.ndarray
    """``(bins, height, width)`` float32."""
    n_edge_px: int
    edge_fraction: float
    truncate: float
    bins: int

    def lookup(self, uv: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Truncated distance for sample points ``uv`` with edge directions
        ``direction``, both in working pixels.

        Each sample reads its own bin and the two either side. The widening is
        deliberate: a bin boundary is an artefact of the binning, and a wireframe
        edge three degrees from one should not fall off a cliff. ``order=1``
        interpolation keeps the field smooth enough for a finite-difference
        Jacobian to mean something.
        """
        ang = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), math.pi)
        k = np.floor(ang / math.pi * self.bins).astype(int) % self.bins
        coords = np.stack([uv[:, 1], uv[:, 0]])
        best = np.full(len(uv), self.truncate, dtype=float)
        for offset in (-1, 0, 1):
            kk = (k + offset) % self.bins
            for b in np.unique(kk):
                sel = kk == b
                if not sel.any():
                    continue
                d = ndimage.map_coordinates(
                    self.dist[b], coords[:, sel], order=1, mode="nearest"
                )
                best[sel] = np.minimum(best[sel], d)
        # Off-frame samples carry the full truncated cost rather than being
        # dropped. Dropping them would make "slide the part out of the picture"
        # a cost-free move, and it is the one move that trivially minimises a
        # chamfer objective.
        off = (
            (uv[:, 0] < 0)
            | (uv[:, 1] < 0)
            | (uv[:, 0] > self.width - 1)
            | (uv[:, 1] > self.height - 1)
        )
        best[off] = self.truncate
        return np.minimum(best, self.truncate)


def load_grayscale(source: bytes | str | Path) -> np.ndarray:
    """One photograph as a float array in [0, 1].

    Pillow is used for decoding only. EXIF orientation is applied, because the
    marks were made on the image as the UI displayed it and a sideways array
    would put every edge at right angles to where the wireframe expects it.
    """
    from PIL import Image, ImageOps

    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def build_edge_field(
    gray: np.ndarray,
    original_width: int,
    original_height: int,
    settings: Settings | None = None,
) -> EdgeField:
    """Detect edges and build the orientation-binned distance fields.

    A Sobel gradient with non-maximum suppression and a percentile threshold --
    Canny without the hysteresis, which is the part that needs two thresholds
    nobody can pick for an arbitrary photograph. A percentile rather than an
    absolute level because exposure varies enormously between photographs and a
    threshold tuned on one of them is wrong on the next.
    """
    s = settings or Settings()
    bins = max(2, int(s.edge_orientation_bins))

    h0, w0 = gray.shape
    scale = min(1.0, WORKING_MAX_PX / max(h0, w0))
    if scale < 1.0:
        gray = ndimage.zoom(gray, scale, order=1)
    h, w = gray.shape
    # Relate the working image to the ORIGINAL pixel coordinates the marks and
    # the projection live in, not to the decoded array: a photo is served
    # downscaled for display and may have been decoded at any size.
    sx = w / float(original_width or w)
    sy = h / float(original_height or h)

    blurred = ndimage.gaussian_filter(gray, max(float(s.edge_sigma_px), 0.1))
    gy = ndimage.sobel(blurred, axis=0)
    gx = ndimage.sobel(blurred, axis=1)
    mag = np.hypot(gx, gy)

    # Non-maximum suppression along the gradient, so a soft edge becomes one
    # pixel wide instead of a band. Without it the distance transform measures
    # the distance to the edge's blur, which is systematically short.
    norm = np.maximum(mag, 1e-12)
    ux, uy = gx / norm, gy / norm
    rows, cols = np.indices(mag.shape, dtype=np.float32)
    fwd = ndimage.map_coordinates(mag, [rows + uy, cols + ux], order=1, mode="nearest")
    back = ndimage.map_coordinates(mag, [rows - uy, cols - ux], order=1, mode="nearest")
    peak = (mag >= fwd) & (mag >= back)

    pct = float(np.clip(s.edge_percentile, 50.0, 99.9))
    level = float(np.percentile(mag[peak] if peak.any() else mag, pct))
    edges = peak & (mag > max(level, 1e-6))

    # Bin by the EDGE direction, which is perpendicular to the gradient.
    edge_ang = np.mod(np.arctan2(-ux, uy), math.pi)
    which = np.floor(edge_ang / math.pi * bins).astype(np.int16) % bins

    truncate = max(float(s.edge_truncate_px), 1.0) * max(scale, 1e-6)
    dist = np.empty((bins, h, w), dtype=np.float32)
    for b in range(bins):
        mask = edges & (which == b)
        if not mask.any():
            dist[b] = truncate
            continue
        dist[b] = np.minimum(
            ndimage.distance_transform_edt(~mask).astype(np.float32), truncate
        )

    return EdgeField(
        scale=float(0.5 * (sx + sy)),
        width=w,
        height=h,
        dist=dist,
        n_edge_px=int(edges.sum()),
        edge_fraction=float(edges.mean()),
        truncate=float(truncate),
        bins=bins,
    )


# --------------------------------------------------------------------------
# which wireframe edges to use, and where to sample them
# --------------------------------------------------------------------------


def front_facing_edges(
    mesh: Mesh, edges: np.ndarray, camera_centre: np.ndarray, scale: float
) -> np.ndarray:
    """``edges`` filtered to those with at least one front-facing neighbour.

    A feature-edge wireframe describes the whole part, far side included, and
    the far side has no support in the photograph. Fitting it to the near side's
    edges is one of the easier ways to converge somewhere confidently wrong, and
    this is the cheap half of the fix -- it does not handle self-occlusion, which
    on a mostly-convex machined part is the smaller problem.
    """
    verts = np.asarray(mesh.vertices, dtype=float) * float(scale)
    faces = np.asarray(mesh.faces, dtype=int)
    if len(faces) == 0 or len(edges) == 0:
        return edges
    normals = mesh.face_normals()
    centroids = verts[faces].mean(axis=1)
    toward = camera_centre[None, :] - centroids
    front = np.einsum("ij,ij->i", normals, toward) > 0.0

    by_edge: dict[tuple[int, int], bool] = {}
    for fi, (a, b, c) in enumerate(faces):
        f = bool(front[fi])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            by_edge[key] = by_edge.get(key, False) or f

    keep = []
    for i, (u, v) in enumerate(edges):
        key = (int(u), int(v)) if u < v else (int(v), int(u))
        if by_edge.get(key, True):
            keep.append(i)
    return edges[np.array(keep, dtype=int)] if keep else edges[:0]


OCCLUSION_FACE_LIMIT = 40000
"""Above this many triangles the depth buffer is skipped.

Rasterising is linear in faces and this is a per-image cost on a UI action. A
part dense enough to exceed it keeps the back-face cull, which is the cheaper
half of the same job."""


def depth_buffer(
    verts: np.ndarray,
    faces: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cam: Intrinsics,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray | None:
    """Camera-space depth of the nearest surface at each working pixel.

    Written because back-face culling turned out not to be enough, and the
    measurement was unambiguous: on a clean synthetic render of this very part,
    at the *correct* pose, the chamfer cost sat at 7 px rather than the 1 px a
    supported wireframe should give, and the optimiser could reduce it by
    drifting the pose until the unsupported edges found something. Those edges
    were front-facing and occluded -- the pad's far creases seen through the
    pad. A wireframe sample with nothing to match is not neutral: it sits on the
    truncation plateau and votes for whichever direction gives it support.

    ``inf`` where nothing covers the pixel.
    """
    if len(faces) == 0 or len(faces) > OCCLUSION_FACE_LIMIT:
        return None
    w = max(int(round(width * scale)), 1)
    h = max(int(round(height * scale)), 1)
    Xc = verts @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (cam.f * Xc[:, 0] / z + cam.cx) * scale
        v = (cam.f * Xc[:, 1] / z + cam.cy) * scale
    buf = np.full((h, w), np.inf)

    for tri in faces:
        if not np.all(z[tri] > 1e-9):
            continue   # crosses the camera plane: no sane raster for it
        ux, vy, zz = u[tri], v[tri], z[tri]
        x0 = max(int(np.floor(ux.min())), 0)
        x1 = min(int(np.ceil(ux.max())) + 1, w)
        y0 = max(int(np.floor(vy.min())), 0)
        y1 = min(int(np.ceil(vy.max())) + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        area = (ux[1] - ux[0]) * (vy[2] - vy[0]) - (ux[2] - ux[0]) * (vy[1] - vy[0])
        if abs(area) < 1e-9:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((ux[1] - px) * (vy[2] - py) - (ux[2] - px) * (vy[1] - py)) / area
        w1 = ((ux[2] - px) * (vy[0] - py) - (ux[0] - px) * (vy[2] - py)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        # Interpolate 1/z and invert: depth is linear in the reciprocal across a
        # projected triangle, and interpolating z itself bows the surface.
        inv = w0 / zz[0] + w1 / zz[1] + w2 / zz[2]
        depth = np.where(inv > 1e-12, 1.0 / np.where(inv > 1e-12, inv, 1.0), np.inf)
        block = buf[y0:y1, x0:x1]
        np.minimum(block, np.where(inside, depth, np.inf), out=block)
    return buf


@dataclass
class _Samples:
    """Wireframe sample points, as model-local world coordinates plus the
    direction of the edge each came from.

    Sampled once in 3D and re-projected at every pose evaluation, rather than
    resampled in 2D each time: a sample set that changes with the parameters
    makes the cost discontinuous and the Jacobian a fiction.
    """

    xyz: np.ndarray
    edge_index: np.ndarray
    endpoints: np.ndarray
    n_edges: int


def build_samples(
    project: Project,
    image: ImageView,
    mesh: Mesh,
    template: ObjectTemplate,
    field_scale: float,
    spacing_px: float = SAMPLE_SPACING_PX,
) -> _Samples | None:
    """Points along the part's visible wireframe, in world metres."""
    if image.R is None or image.t is None:
        return None
    verts = part_vertices_world(template, mesh)
    edges = mesh.feature_edges(template.crease_angle)
    if len(edges) == 0:
        return None
    centre = -np.asarray(image.R, dtype=float).T @ np.asarray(image.t, dtype=float)
    edges = front_facing_edges(mesh, edges, centre, template.unit_scale)
    if len(edges) == 0:
        return None

    cam = intrinsics_for(project, image)
    from .geometry import project as project_points

    uv = project_points(
        np.asarray(image.R, dtype=float), np.asarray(image.t, dtype=float), cam, verts
    )
    Xc = verts @ np.asarray(image.R, dtype=float).T + np.asarray(image.t, dtype=float)
    in_front = Xc[:, 2] > 1e-6

    # How many samples each edge gets is decided once, from the starting pose,
    # so the sample count cannot drift with the parameters.
    lengths = np.linalg.norm(uv[edges[:, 1]] - uv[edges[:, 0]], axis=1) * field_scale
    usable = in_front[edges[:, 0]] & in_front[edges[:, 1]] & np.isfinite(lengths) & (lengths > 1.0)
    edges = edges[usable]
    lengths = lengths[usable]
    if len(edges) == 0:
        return None
    counts = np.clip(np.ceil(lengths / max(spacing_px, 1.0)).astype(int) + 1, 2, 400)
    if counts.sum() > MAX_SAMPLES:
        order = np.argsort(-lengths)
        running, keep = 0, []
        for i in order:
            if running + counts[i] > MAX_SAMPLES:
                continue
            keep.append(i)
            running += counts[i]
        keep = np.array(sorted(keep), dtype=int)
        edges, counts = edges[keep], counts[keep]

    xyz, index = [], []
    for i, (e, n) in enumerate(zip(edges, counts)):
        a, b = verts[e[0]], verts[e[1]]
        ts = np.linspace(0.0, 1.0, int(n))[:, None]
        xyz.append(a[None, :] * (1 - ts) + b[None, :] * ts)
        index.append(np.full(int(n), i, dtype=int))
    xyz = np.concatenate(xyz)
    index = np.concatenate(index)

    # Drop samples the part itself hides. Visibility is decided once, from the
    # starting pose, and then held: a sample set that changes with the
    # parameters makes the cost discontinuous and its Jacobian a fiction. The
    # starting pose is the solve's, which is the best one available -- if it is
    # wrong enough for this to pick the wrong samples, it is also too wrong for
    # stage two to be the right tool.
    buf = depth_buffer(
        verts, faces_of(mesh), image.R, image.t, cam, image.width, image.height, field_scale
    )
    if buf is not None:
        Xs = xyz @ np.asarray(image.R, dtype=float).T + np.asarray(image.t, dtype=float)
        zs = Xs[:, 2]
        us = (cam.f * Xs[:, 0] / np.maximum(zs, 1e-9) + cam.cx) * field_scale
        vs = (cam.f * Xs[:, 1] / np.maximum(zs, 1e-9) + cam.cy) * field_scale
        iy = np.clip(np.round(vs).astype(int), 0, buf.shape[0] - 1)
        ix = np.clip(np.round(us).astype(int), 0, buf.shape[1] - 1)
        nearest = buf[iy, ix]
        # A tolerance proportional to depth, because what is being compared is a
        # rasterised surface against a point on its own silhouette: an edge is
        # by construction at the same depth as the face it bounds, and a fixed
        # tolerance in metres is either useless far away or too generous close.
        visible = ~np.isfinite(nearest) | (zs <= nearest * 1.004 + 1e-9)
        if visible.sum() >= 20:
            xyz, index = xyz[visible], index[visible]

    return _Samples(
        xyz=xyz,
        edge_index=index,
        endpoints=np.stack([verts[edges[:, 0]], verts[edges[:, 1]]], axis=1),
        n_edges=len(edges),
    )


def faces_of(mesh: Mesh) -> np.ndarray:
    return np.asarray(mesh.faces, dtype=int)


# --------------------------------------------------------------------------
# the refinement
# --------------------------------------------------------------------------


@dataclass
class RefineResult:
    image: str
    attempted: bool = False
    accepted: bool = False
    stage: str = "not started"
    reason: str = ""
    """Why it was rejected, in the user's terms. Empty when accepted."""
    cost_before: float = float("nan")
    cost_after: float = float("nan")
    """What was minimised: the RMS of the *softened* chamfer residual, in
    original image pixels. Comparable before and after, and not dominated by
    the samples that found no edge at all -- which is the whole reason it is
    softened rather than raw. Not the number to quote to a person."""
    median_before: float = float("nan")
    median_after: float = float("nan")
    """The number to quote to a person: the median distance, in original pixels,
    from a wireframe sample to the nearest image edge running the same way. This
    is "how far off is the wireframe"; ``cost`` is "what the optimiser was
    pushing on", and they are deliberately different quantities."""
    improvement: float = 0.0
    """Fractional drop in chamfer cost. Positive means stage two liked it."""
    support_before: float = 0.0
    support_after: float = 0.0
    """Fraction of samples that found any edge at all inside the truncation
    radius. The number that says whether there was evidence, as opposed to a
    well-converged fit to nothing."""
    mark_rms_before: float | None = None
    mark_rms_after: float | None = None
    """The referee: reprojection RMS of this photo's own clicked marks, which
    were not in the objective."""
    shift_px: float = 0.0
    """How far the projected wireframe's centre moved, in original pixels."""
    rotation_deg: float = 0.0
    n_samples: int = 0
    n_edges: int = 0
    edge_fraction: float = 0.0
    """Fraction of working pixels classified as edges. A busy photograph reads
    high here, and a high value is the warning sign for this whole method."""
    R: list[list[float]] | None = None
    t: list[float] | None = None
    """The refined pose, whether or not it was accepted. Returned even when
    rejected so the UI can offer it as something to look at rather than
    silently discarding it."""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(vars(self))


def _cost_fn(
    samples: _Samples,
    edge_field: EdgeField,
    cam: Intrinsics,
    R0,
    t0,
    soften: float = 3.0,
):
    """Chamfer residuals as a function of a pose delta.

    Returns ``(residuals, raw, project_at)``: the softened residual the
    optimiser sees, the raw truncated distances for reporting, and the
    projection itself. The two distances are kept apart on purpose -- the
    softened one is what must be minimised to avoid being led by the samples
    that found nothing, and the raw one is what a person means by "how far is
    the wireframe from the edges".
    """
    R0 = np.asarray(R0, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    soften = max(float(soften), 1e-3)

    def project_at(x: np.ndarray):
        R = so3_exp(x[:3]) @ R0
        t = t0 + x[3:]
        Xc = samples.xyz @ R.T + t
        z = np.maximum(Xc[:, 2], 1e-6)
        uv = np.stack(
            [cam.f * Xc[:, 0] / z + cam.cx, cam.f * Xc[:, 1] / z + cam.cy], axis=1
        )
        ends = samples.endpoints.reshape(-1, 3) @ R.T + t
        ze = np.maximum(ends[:, 2], 1e-6)
        euv = np.stack(
            [cam.f * ends[:, 0] / ze + cam.cx, cam.f * ends[:, 1] / ze + cam.cy], axis=1
        ).reshape(-1, 2, 2)
        return uv, euv, R, t

    def raw(x: np.ndarray) -> np.ndarray:
        uv, euv, _R, _t = project_at(x)
        direction = euv[:, 1, :] - euv[:, 0, :]
        norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        direction = direction / norm
        return edge_field.lookup(uv * edge_field.scale, direction[samples.edge_index])

    def residuals(x: np.ndarray) -> np.ndarray:
        d = raw(x)
        return d / np.sqrt(1.0 + (d / soften) ** 2)

    return residuals, raw, project_at


def refine_image_pose(
    project: Project,
    image: ImageView,
    mesh: Mesh,
    template: ObjectTemplate,
    gray: np.ndarray | None = None,
    image_source: bytes | str | Path | None = None,
    edge_field: EdgeField | None = None,
) -> RefineResult:
    """Nudge one camera's pose so the part's wireframe sits on this photo's edges.

    Returns a result whose ``accepted`` says whether the project was changed. The
    pose is written back only when it both improves the chamfer cost and leaves
    the image's clicked marks no worse -- and the pose it replaced is kept on the
    image, so putting it back is exact.
    """
    s = project.settings
    out = RefineResult(image=image.id)

    if not s.edge_refine:
        out.stage = "disabled"
        out.reason = (
            "edge refinement is switched off for this project. It is off by default because it "
            "reads the photograph rather than the geometry, and on a busy background it can land "
            "confidently on the wrong edges."
        )
        return out
    if image.R is None or image.t is None:
        out.stage = "no pose"
        out.reason = "this camera has no solved pose, so there is nothing to refine"
        return out
    if project.part_instance() is None or not project.in_part_space():
        out.stage = "not in part space"
        out.reason = (
            "the part has no known pose yet, so there is no wireframe to slide. Stage one has to "
            "work before stage two means anything."
        )
        return out

    if edge_field is None:
        if gray is None:
            if image_source is None:
                image_source = project.resolve_path(image)
            gray = load_grayscale(image_source)
        edge_field = build_edge_field(gray, image.width, image.height, s)
    out.edge_fraction = edge_field.edge_fraction

    samples = build_samples(project, image, mesh, template, edge_field.scale)
    if samples is None or len(samples.xyz) < 20:
        out.stage = "no usable wireframe"
        out.reason = (
            "too little of the part's wireframe projects in front of this camera to align "
            "anything. Check stage one's placement in this photo first."
        )
        return out
    out.n_samples = int(len(samples.xyz))
    out.n_edges = int(samples.n_edges)
    out.attempted = True

    cam = intrinsics_for(project, image)
    R0 = np.asarray(image.R, dtype=float)
    t0 = np.asarray(image.t, dtype=float)
    soften = float(s.edge_soften_px) * max(edge_field.scale, 1e-9)
    residuals, raw, project_at = _cost_fn(samples, edge_field, cam, R0, t0, soften)

    def score(x: np.ndarray) -> tuple[float, float, float]:
        d = raw(x)
        r = d / np.sqrt(1.0 + (d / soften) ** 2)
        inv = 1.0 / max(edge_field.scale, 1e-9)
        support = float(np.mean(d < edge_field.truncate * 0.95))
        return (
            float(np.sqrt(np.mean(r**2))) * inv,
            support,
            float(np.median(d)) * inv,
        )

    x0 = np.zeros(6)
    out.cost_before, out.support_before, out.median_before = score(x0)
    out.mark_rms_before = mark_reprojection_rms(project, image)

    # No marks, no refinement. This is not caution for its own sake: the chamfer
    # objective was measured to be biased by about half a working pixel, so on a
    # pose that is already good it asks confidently for a change that makes
    # things worse. The clicked marks are the only thing in the picture that can
    # tell that case from the one worth accepting, and without them the decision
    # would be left entirely to the quantity with the known bias.
    if out.mark_rms_before is None:
        out.stage = "no marks to check against"
        out.reason = (
            "this photo has no solved tie points marked in it, so there is nothing independent to "
            "check an alignment against. The edge fit alone is not enough: it is biased by roughly "
            "half a pixel and will happily report an improvement while moving the pose the wrong "
            "way. Mark two or three tie points in this photo and it becomes decidable."
        )
        return out

    if out.support_before < MIN_SUPPORTED_FRACTION:
        out.stage = "no support"
        out.reason = (
            f"only {out.support_before * 100:.0f}% of the wireframe is within "
            f"{s.edge_truncate_px:g} px of any image edge of a matching direction. There is "
            "nothing here for the part to align to -- either it is not visible in this photo, or "
            "stage one has it in the wrong place by more than this stage is allowed to correct."
        )
        return out

    # Bounded, and tightly. This is a correction to a pose the bundle adjustment
    # already produced, not a search for one: the translation bound is a few
    # percent of the camera's distance to the part and the rotation bound a
    # couple of degrees. Anything beyond that is not a refinement, it is a
    # different answer, and a different answer from an objective this unreliable
    # is exactly what must not be allowed through.
    distance = float(np.linalg.norm(samples.xyz.mean(axis=0) - (-R0.T @ t0)))
    dt_bound = max(0.05 * max(distance, 1e-6), 1e-6)
    rot_bound = math.radians(3.0)
    bounds = (
        np.array([-rot_bound] * 3 + [-dt_bound] * 3),
        np.array([rot_bound] * 3 + [dt_bound] * 3),
    )

    try:
        result = least_squares(
            residuals,
            x0,
            bounds=bounds,
            # The chamfer field is piecewise-linear and truncated, so analytic
            # derivatives do not exist at the clip and the step has to be large
            # enough to see across a pixel of the field rather than the noise
            # inside one.
            diff_step=1e-3,
            x_scale=np.array([rot_bound] * 3 + [dt_bound] * 3),
            max_nfev=200,
            verbose=0,
        )
    except Exception as exc:  # noqa: BLE001 - a failed optimise must not lose a pose
        out.stage = "optimiser failed"
        out.reason = f"the alignment did not run: {type(exc).__name__}: {exc}"
        return out

    out.cost_after, out.support_after, out.median_after = score(result.x)
    uv_after, _ends, R_new, t_new = project_at(result.x)
    uv_before, _e0, _R0, _t0 = project_at(x0)
    out.shift_px = float(np.linalg.norm(uv_after.mean(axis=0) - uv_before.mean(axis=0)))
    out.rotation_deg = float(math.degrees(np.linalg.norm(result.x[:3])))
    out.R = R_new.tolist()
    out.t = t_new.tolist()
    out.improvement = (
        0.0
        if not np.isfinite(out.cost_before) or out.cost_before <= 0
        else float((out.cost_before - out.cost_after) / out.cost_before)
    )

    # ---- the gates, in the order that produces the most useful refusal
    if out.improvement < float(s.edge_min_improvement):
        out.stage = "no improvement"
        out.reason = (
            f"the best alignment found only improves the edge fit by {out.improvement * 100:.1f}% "
            f"(threshold {s.edge_min_improvement * 100:.0f}%), so the solve's pose was already as "
            "good as this can tell. Kept the original."
        )
        # Two very different situations produce this, and the median says which.
        # A pose that was already right leaves the wireframe a pixel or two from
        # its edges; a photograph whose background is brighter than the part
        # leaves it tens of pixels away, because the percentile threshold spent
        # itself on the background and the part's own edges never made the cut.
        # That is the expected failure mode and it is worth naming, since the
        # setting that fixes it is not one anyone would guess at.
        if np.isfinite(out.median_before) and out.median_before > 0.4 * s.edge_truncate_px:
            out.reason += (
                f" The wireframe is {out.median_before:.0f} px from the nearest matching edge, "
                "which is too far to be a good pose -- the part's own edges are probably not "
                f"being detected at all. A background brighter than the part does that: the "
                f"{s.edge_percentile:g}th-percentile threshold goes to the background and the "
                "part's creases fall under it. Lower the percentile and try again, and if that "
                "does not help, this photo is not one this method can work on."
            )
        return out
    if out.shift_px > float(s.edge_max_shift_px):
        out.stage = "moved too far"
        out.reason = (
            f"the alignment wanted to move the part {out.shift_px:.0f} px in this photo, past the "
            f"{s.edge_max_shift_px:g} px a correction is allowed. A move that large means it found "
            "a different set of edges rather than the same ones more precisely. Kept the original."
        )
        return out

    # The referee. Applied by actually moving the camera and measuring, because
    # the marks' reprojection depends on the pose in a way nothing in the
    # objective tracked.
    prev_R, prev_t = image.R, image.t
    image.R, image.t = R_new, t_new
    out.mark_rms_after = mark_reprojection_rms(project, image)
    if (
        out.mark_rms_before is not None
        and out.mark_rms_after is not None
        and out.mark_rms_after > out.mark_rms_before + float(s.edge_max_reproj_worsening_px)
    ):
        image.R, image.t = prev_R, prev_t
        out.stage = "rejected by the marks"
        out.reason = (
            f"the edge fit improved by {out.improvement * 100:.0f}% but this photo's own clicked "
            f"marks got worse, from {out.mark_rms_before:.2f} px to {out.mark_rms_after:.2f} px. "
            "Those marks were not part of what was minimised, so they are an independent check, "
            "and they say the wireframe went onto the wrong edges. Kept the original."
        )
        return out

    if not image.edge_refined:
        image.R_unrefined, image.t_unrefined = prev_R, prev_t
    image.edge_refined = True
    out.accepted = True
    out.stage = "accepted"
    # Better is not the same as good. A pose tens of pixels out can be improved
    # and still be tens of pixels out, and the number that says so is the one
    # that was not optimised.
    if out.mark_rms_after is not None and out.mark_rms_after > 3.0:
        out.reason = (
            f"accepted, but this photo's marks still reproject at {out.mark_rms_after:.1f} px. "
            "The alignment improved the pose without fixing it -- look at the wireframe, and "
            "expect to re-solve rather than to refine again."
        )
    return out


def revert_image_pose(image: ImageView) -> bool:
    """Put back the pose the solve produced. Returns whether anything changed."""
    if not image.edge_refined or image.R_unrefined is None or image.t_unrefined is None:
        return False
    image.R, image.t = image.R_unrefined, image.t_unrefined
    image.R_unrefined, image.t_unrefined = None, None
    image.edge_refined = False
    return True


@dataclass
class RefineReport:
    ok: bool = False
    enabled: bool = False
    results: list[RefineResult] = field(default_factory=list)
    n_accepted: int = 0
    n_rejected: int = 0
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in vars(self).items() if k != "results"}
        d["results"] = [r.to_dict() for r in self.results]
        return jsonable(d)


def refine_images(
    project: Project,
    mesh: Mesh,
    template: ObjectTemplate,
    image_ids: list[str],
    source_for=None,
) -> RefineReport:
    """Try the refinement on the named images, one at a time.

    ``source_for`` is a callable from image id to bytes or a path, so the served
    build and the browser build can each supply pixels their own way. Each image
    is independent: one rejection says nothing about the next, and the report
    lists every outcome rather than a summary, because "accepted on four of
    seven" is the useful answer and an average is not.
    """
    report = RefineReport(enabled=bool(project.settings.edge_refine))
    if not report.enabled:
        report.warnings.append(
            "edge refinement is off. Turn it on in the solve settings if you want it; stage one "
            "needs nothing switched on."
        )
        return report

    for iid in image_ids:
        image = project.image(iid)
        if image is None:
            report.warnings.append(f"no such image: {iid}")
            continue
        source = None if source_for is None else source_for(iid)
        try:
            result = refine_image_pose(
                project, image, mesh, template, image_source=source
            )
        except Exception as exc:  # noqa: BLE001 - one unreadable photo is not a failed run
            result = RefineResult(image=iid, stage="failed", reason=f"{type(exc).__name__}: {exc}")
        report.results.append(result)

    report.n_accepted = sum(1 for r in report.results if r.accepted)
    report.n_rejected = sum(1 for r in report.results if not r.accepted)
    report.ok = True
    busy = [r for r in report.results if r.edge_fraction > 0.12]
    if busy:
        report.warnings.append(
            f"{len(busy)} photo(s) have a lot of the frame classified as edges, which is what a "
            "busy background looks like to this method. Check those alignments by eye before "
            "trusting them -- a wrong confident alignment is the failure mode here."
        )
    report.messages.append(
        f"{report.n_accepted} pose(s) accepted, {report.n_rejected} left as the solve had them. "
        "Every accepted pose kept its original, so each one can be put back exactly."
    )
    return report
