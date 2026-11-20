"""More correspondences, found by geometry rather than by appearance.

A solved project has something an ordinary structure-from-motion run spends
all its effort trying to acquire: it already knows where the cameras are.  That
turns matching from a search into a test.  A feature in one photograph does not
have to be recognised somewhere in the next one -- it has to be *on a specific
line*, at a depth inside the scene, seen from an angle wide enough to mean
anything.  Appearance is then only the tie-breaker among the handful of points
that survive, instead of the sole evidence it is in a cold matcher.

That distinction is not an optimisation here, it is the whole reason this can
work at all on this scene.  The rear-module cell contains twelve identical
copies of one sensor body, five of another, and a row of indistinguishable
tubes and clamps.  Every one of those copies is a perfect appearance match for
every other, so a descriptor-based matcher does not have a hard problem, it has
an *unsolvable* one: the right answer and eleven wrong ones are the same
picture.  What separates them is that eleven of the twelve are in the wrong
place, and a known pose is exactly the instrument that says so.  The epipolar
test below rejects a wrong-copy match by hundreds of pixels while the patch
correlation cheerfully endorses it.

So the order matters, and it is the reverse of the usual one:

1.  Shi-Tomasi corners per photograph, kept on a grid so they cannot all pile
    onto the one high-contrast decal.
2.  For every image pair, the known poses give an essential matrix.  Candidate
    partners are the corners within a few pixels of the epipolar line whose
    triangulation lands inside the scene's own bounding box, at a ray angle
    wide enough for the depth to mean something.
3.  Only then, appearance: normalised cross-correlation over a patch the known
    poses have *pre-warped*, using the plane-induced homography at the depth
    the candidate implies.  Two photographs of the same corner from 40 degrees
    apart do not look alike under a square window, and a raw NCC would throw
    away most of the real matches; warped by the geometry already in hand, they
    do look alike, which is the second thing solved poses buy.
4.  Mutual best match, plus a uniqueness margin over the runner-up, which is
    what catches a repeated texture lying *along* the epipolar line -- the one
    direction the geometry cannot separate.
5.  Tracks by union-find, kept at three or more views, re-triangulated jointly,
    and thinned over a grid so the bundle gets evidence spread across the frame
    rather than four hundred points on one knurled grip.

What comes out is added to the project as ordinary tie points, so the bundle
needs no changes and nothing downstream has to learn a new kind of evidence.

What it was worth on the scene it was written for, measured rather than hoped
for, because ``*-HANDOFF.md`` is not tracked and this is the only durable place
to put it.  On ``projects/rearmod3`` -- 8 photographs, 28 placed bodies, 582
anchor marks -- a harvest adds 105 points and takes the tie-point observations
from 113 to 447, roughly quadrupling the free-point evidence holding the
cameras.  In-solve reprojection RMS falls from 9.17 px to 7.35 px.

**Held-out error does not move at all.**  Cross-validation over the anchor
marks (``tools/crossval.py``, which holds marks out of the solve entirely
rather than culling them) gives 17.83 px before and 18.04 px after at four
folds, and 19.42 px before against 19.55 px after over twelve -- worse both
times, by less than the measurement can resolve.  Freeing ``lock_focal`` and
``lock_k1`` changes nothing either way: every one of the six variants lands
between 19.42 and 19.68, while changing only the fold seed moves the *same*
baseline from 17.83 to 19.42.  The differences between variants are seven
times smaller than the noise in the measurement of them.  Freeing the principal
point as well widens the band only to 19.80, and its worst row is the one with
the most evidence *and* the most freedom: the harvest supplies enough
observations to let more intrinsics float without anything breaking, and
floating them buys nothing, because the error they would have to absorb is not
in the camera model.

Anyone reading the in-solve figure alone would conclude this feature worked,
and would be wrong; that is exactly the trap ``crossval.py`` was written to
catch, and this module walked straight into it.

The reason is visible in a second measurement.  Hold out harvested tie-point
marks instead of anchor marks and they reproject at a median of 5.6 px --
statistically the same as the hand-marked tie points (6.2 px) and the anchor
marks (5.6 px) in the same solve.  So the harvest did not add *better* evidence
than the project already had, it added *more of the same*, and the error that
remains is common to every kind of mark rather than random per mark.  More
observations average down random error; they do nothing to a systematic one.
Adding correspondence was the right experiment and the answer was no.

Where it may still earn its place: a scene whose cameras are held by far fewer
marks than this one, and resilience rather than accuracy -- see the handoff for
the mis-marked-body test, where the question is not whether the focal is better
but whether it stays put when somebody clicks a body onto the wrong features.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# `_Cam` and `triangulate_views` live in geometry now: the solve places a
# known object against finished cameras and needs both. Imported under the
# old private name so nothing in this module reads differently.
from .geometry import Intrinsics, SolvedCamera as _Cam, triangulate_views
from .project import Project, TiePoint

__all__ = [
    "HarvestSettings",
    "HarvestReport",
    "Track",
    "detect_corners",
    "harvest",
    "add_tracks_to_project",
]


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


@dataclass
class HarvestSettings:
    """Every threshold the harvest turns on, in one place and all named.

    The defaults are the ones the rig fixture was measured with.  They are
    deliberately strict: a harvest that adds a thousand points of which fifty
    are wrong is worse than one that adds two hundred that are all right, because
    the bundle has no way to tell which fifty and the cull will blame whatever
    disagrees with them.
    """

    # ---- detection
    max_corners_per_image: int = 6000
    """Cap per photograph, after grid thinning.

    Generous, because the binding constraint on this method is not how many
    matches a pair yields but how often the *same* physical corner is detected
    in three photographs at once.  Repeatability falls off fast with the
    budget: on the rig fixture, halving the corner count did not halve the
    tracks, it cut the three-view ones by rather more."""
    corner_grid: int = 24
    """Detection grid, cells along the image's longer side.  Corners are kept
    per cell, so a single high-contrast region cannot spend the whole budget."""
    corner_min_distance: float = 6.0
    """Non-maximum suppression radius, in pixels of the full-size photograph."""
    corner_quality: float = 0.002
    """Shi-Tomasi response floor, as a fraction of the image's best corner."""
    detect_max_px: int = 2200
    """Longest side the detector works at.  Corner positions are scaled back to
    full resolution afterwards, with the sub-pixel peak fit done at working
    scale.  12 megapixels of iPhone sensor carries no corner detail a 2200 px
    working image misses, and costs four times the gradient arithmetic."""

    # ---- geometric gate
    epipolar_px: float = 8.0
    """How far a candidate may sit from the epipolar line, in pixels, measured
    in both images.

    Set from the data rather than from taste, and the measurement is worth
    keeping.  The 33 hand-marked tie points on the rig fixture are known-good
    correspondences; run them through this same test against the project's own
    poses and their epipolar residual has a median of 3.9 px, a 75th percentile
    of 7.3 and a 90th of 10.9.  A 3 px band -- which looks like the obvious
    choice, and was the first thing tried -- therefore throws away 62% of the
    true matches before appearance is ever consulted.

    That residual is not sloppiness, it is the honest sum of the pose error and
    the marking error, and it is the budget a gate has to live inside.  Widening
    to 8 px costs surprisingly little, because the thing this gate exists to
    reject is not a few pixels off the line: the wrong copy of an identical
    sensor body is *hundreds* of pixels off it.  Going from 3 px to 8 px roughly
    triples the raw candidates and leaves that rejection entirely intact."""
    min_triangulation_deg: float = 5.0
    """Below this the depth is guesswork: the two rays are nearly the same ray,
    so the point slides a long way for a sub-pixel change in either mark."""
    depth_margin: float = 0.25
    """Fractional slack on the scene's existing extent, for the bounding box a
    triangulation has to land inside."""

    # ---- appearance
    patch: int = 11
    """Correlation window, odd, in pixels of the working image."""
    ncc_min: float = 0.80
    """Floor on the warped correlation score."""
    uniqueness: float = 0.80
    """Lowe's ratio on ``1 - ncc``: the best match has to beat the runner-up by
    this factor or neither is believed.  The guard against a repeated texture
    lying along the epipolar line, where geometry has nothing to say."""

    # ---- tracks
    min_views: int = 3
    """Tracks seen in fewer photographs than this are dropped.  Two views
    triangulate to a point that fits both of them perfectly by construction and
    so tells the bundle nothing it did not already know."""
    extend_px: float = 14.0
    """Search radius when a track is projected into a photograph it was not
    matched in, in pixels.

    Track *extension* is what makes three-view tracks common rather than rare.
    Pairwise matching alone produced 280 components on the rig fixture of which
    24 reached three views: two independent pairs only chain when the detector
    happened to fire on the same physical corner in both, and across an eight
    photograph spread it usually does not.  Projecting a two-view point into
    the remaining cameras asks the question directly instead of waiting for it
    to be answered by coincidence -- which is, again, only possible because the
    poses are known.

    The radius is wider than ``epipolar_px`` because this is a point prediction
    rather than a line: it carries the triangulation's own error along the ray
    as well as the pose error across it.  Still nowhere near wide enough to
    reach a different copy of the same body."""
    extend_rounds: int = 2
    """Extension passes.  The second is not redundant: a track promoted from
    two views to three re-triangulates to a better point, and the better point
    reaches a fourth camera the first pass predicted too far off."""
    max_track_reproj_px: float = 10.0
    """A track whose joint triangulation misses any of its own marks by more
    than this is internally inconsistent -- usually one view matched onto a
    different copy of the same body -- and is dropped whole.

    Chosen against the same measurement as ``epipolar_px``, and for the same
    reason.  The known-good hand-marked points do not reproject through these
    poses to a pixel; they reproject to several -- median 4.7 px, 90th
    percentile 11.4.  A 2 px ceiling does not select the correct tracks, it
    selects the ones whose errors happened to cancel -- on the first run it left
    four tracks out of twenty-four, which is not strictness but a sampling
    accident.

    There is a deeper trap in this threshold than yield, and it is the reason
    not to tighten it back.  The residual is measured against *the poses being
    harvested from*, so a tight ceiling keeps the tracks that agree with the
    current answer and throws away the ones that disagree -- which is to say it
    selects for evidence that will not change anything.  A filter that can only
    confirm is not a filter.  Correctness here has to come from the parts of the
    pipeline that do not ask the current poses for permission: the epipolar
    gate, the uniqueness margin, the warped correlation and the requirement that
    three independent views agree.  At 10 px this still rejects over half of
    what reaches it, and nearly all of that is beyond 14 px as well -- a genuine
    bimodality, right tracks against chains onto the wrong copy, rather than a
    threshold cutting through the middle of one distribution."""
    max_points: int = 400
    """Global cap on harvested tie points.  Each one costs the bundle three
    parameters, and past a few hundred the marginal point is buying redundancy
    the solve already has."""
    spread_grid: int = 12
    """Thinning grid, cells along the image's longer side."""
    max_per_cell: int = 6
    """Harvested points allowed per grid cell per photograph."""

    # ---- bookkeeping
    label_prefix: str = "h"
    """Harvested tie points are labelled ``h0001``..., so they can be found,
    counted and deleted again without guessing from the id."""
    uncertain: bool = False
    """Flag the harvested points uncertain, which weights them
    ``Settings.uncertain_factor`` times looser and culls them sooner.  Off by
    default: a correlation peak located to a fraction of a pixel is not a
    hesitant click, and saying it is would waste the one advantage these points
    have over the anchor marks."""


@dataclass
class Track:
    """One harvested 3D point and the marks that produced it."""

    xyz: np.ndarray
    obs: dict[str, np.ndarray]  # image id -> (u, v) in full-resolution pixels
    reproj_px: float
    angle_deg: float
    ncc: float

    @property
    def n_views(self) -> int:
        return len(self.obs)


@dataclass
class HarvestReport:
    tracks: list[Track] = field(default_factory=list)
    corners: dict[str, int] = field(default_factory=dict)
    pair_matches: dict[tuple[str, str], int] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_raw_tracks: int = 0
    n_after_views: int = 0
    n_after_reproj: int = 0
    n_kept: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"corners: {sum(self.corners.values())} over {len(self.corners)} photo(s)",
            f"pairs with matches: {sum(1 for v in self.pair_matches.values() if v)}"
            f" of {len(self.pair_matches)}",
            f"mutual matches: {sum(self.pair_matches.values())}",
            f"tracks: {self.n_raw_tracks} raw -> {self.n_after_views} with "
            f"enough views -> {self.n_after_reproj} consistent -> {self.n_kept} kept",
            f"elapsed: {self.seconds:.1f}s",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# camera bookkeeping
# --------------------------------------------------------------------------


def cameras_of(project: Project) -> dict[str, _Cam]:
    """Every enabled image that carries a solved pose."""
    out: dict[str, _Cam] = {}
    for im in project.active_images():
        if im.R is None or im.t is None:
            continue
        g = project.groups.get(im.group)
        if g is None or g.f is None:
            continue
        cx, cy = g.principal_point_for(im.width, im.height)
        cam = Intrinsics(float(g.f), cx, cy, float(g.k1))
        out[im.id] = _Cam(im.id, np.asarray(im.R, float), np.asarray(im.t, float),
                          cam, int(im.width), int(im.height))
    return out


def scene_points(project: Project) -> np.ndarray:
    """Everything the project already believes a world position for.

    Both kinds count: a tie point's triangulated ``xyz`` and every anchor of
    every placed body, carried through that body's pose.  Together they are the
    only statement available about how big the scene is and where it is, which
    is what the depth gate needs.
    """
    pts: list[np.ndarray] = []
    for p in project.points:
        if p.xyz is not None:
            pts.append(np.asarray(p.xyz, dtype=float))
    for inst in project.placed_object_instances():
        if inst.R is None or inst.t is None:
            continue
        template = project.template_of(inst)
        if template is None:
            continue
        local = np.array(list(template.anchor_points().values()), dtype=float)
        if len(local):
            pts.extend(local @ np.asarray(inst.R, float).T + np.asarray(inst.t, float))
    return np.array(pts, dtype=float) if pts else np.zeros((0, 3))


# --------------------------------------------------------------------------
# corner detection
# --------------------------------------------------------------------------


def detect_corners(
    gray: np.ndarray,
    settings: HarvestSettings | None = None,
    *,
    full_width: int | None = None,
    full_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shi-Tomasi corners, grid-thinned, with a sub-pixel peak fit.

    Shi-Tomasi (the smaller eigenvalue of the structure tensor) rather than
    Harris's determinant-minus-trace form, because the eigenvalue is in the same
    units whatever the contrast and a single quality fraction therefore means
    the same thing on a dark photograph and a bright one.

    Thinning is done per grid cell rather than by a global threshold, and that
    is the part that matters for a bundle.  A global threshold on this scene
    returns a dense mat of corners on the one printed label and nothing from the
    machined surfaces around it, which is a lot of evidence about one place and
    none about the frame.

    Returns ``(uv, response)`` with ``uv`` in full-resolution pixels.
    """
    s = settings or HarvestSettings()
    gray = np.asarray(gray, dtype=np.float32)
    h0, w0 = gray.shape
    full_w = float(full_width or w0)
    full_h = float(full_height or h0)

    scale = min(1.0, s.detect_max_px / float(max(h0, w0)))
    work = ndimage.zoom(gray, scale, order=1) if scale < 1.0 else gray
    h, w = work.shape

    blurred = ndimage.gaussian_filter(work, 1.0)
    gy = ndimage.sobel(blurred, axis=0)
    gx = ndimage.sobel(blurred, axis=1)
    # structure tensor, smoothed over a window a little wider than the gradient
    a = ndimage.gaussian_filter(gx * gx, 1.5)
    b = ndimage.gaussian_filter(gx * gy, 1.5)
    c = ndimage.gaussian_filter(gy * gy, 1.5)
    tr = a + c
    det = a * c - b * b
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    response = tr / 2.0 - disc  # smaller eigenvalue

    peak = float(response.max())
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros((0, 2)), np.zeros(0)

    radius = max(1, int(round(s.corner_min_distance * scale)))
    local_max = ndimage.maximum_filter(response, size=2 * radius + 1)
    mask = (response >= local_max) & (response > s.corner_quality * peak)
    # a one-pixel frame is useless anyway and the sub-pixel fit needs neighbours
    mask[:2, :] = mask[-2:, :] = mask[:, :2] = mask[:, -2:] = False

    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((0, 2)), np.zeros(0)
    vals = response[ys, xs]

    # ---- sub-pixel: a separable quadratic through the response peak.  Worth the
    # four extra lookups -- an integer corner carries up to half a pixel of
    # quantisation noise, which is a third of the click accuracy the solve
    # assumes and would be spent for nothing.
    def _offset(minus, plus, mid):
        denom = minus - 2.0 * mid + plus
        off = np.where(np.abs(denom) > 1e-12, 0.5 * (minus - plus) / np.where(
            np.abs(denom) > 1e-12, denom, 1.0), 0.0)
        return np.clip(off, -0.5, 0.5)

    dx = _offset(response[ys, xs - 1], response[ys, xs + 1], vals)
    dy = _offset(response[ys - 1, xs], response[ys + 1, xs], vals)

    # ---- grid thinning
    cells = max(1, int(s.corner_grid))
    cell = max(h, w) / float(cells)
    gi = np.minimum((ys / cell).astype(int), cells)
    gj = np.minimum((xs / cell).astype(int), cells)
    key = gi * (cells + 1) + gj
    per_cell = max(1, s.max_corners_per_image // max(1, len(np.unique(key))))

    order = np.lexsort((-vals, key))
    keep = np.zeros(len(order), dtype=bool)
    seen: dict[int, int] = {}
    for idx in order:
        k = int(key[idx])
        n = seen.get(k, 0)
        if n < per_cell:
            keep[idx] = True
            seen[k] = n + 1
    sel = order[keep[order]]
    if len(sel) > s.max_corners_per_image:
        sel = sel[np.argsort(-vals[sel])[: s.max_corners_per_image]]

    sx = full_w / float(w)
    sy = full_h / float(h)
    uv = np.stack([(xs[sel] + dx[sel] + 0.5) * sx - 0.5,
                   (ys[sel] + dy[sel] + 0.5) * sy - 0.5], axis=1)
    return uv, vals[sel]


# --------------------------------------------------------------------------
# triangulation
# --------------------------------------------------------------------------


def _triangulate_pair(cam_a: _Cam, cam_b: _Cam, xa: np.ndarray, xb: np.ndarray):
    """Midpoint of the common perpendicular, vectorised over N ray pairs.

    Returns ``(X, angle_rad, za, zb)``.  The midpoint rather than the DLT
    because the gate downstream wants the ray separation anyway, and because
    when two rays nearly coincide the midpoint degrades gracefully while the
    DLT's smallest singular vector does not.
    """
    da = xa @ cam_a.R  # (N,3) world directions, = R^T x
    db = xb @ cam_b.R
    da = da / np.linalg.norm(da, axis=1, keepdims=True)
    db = db / np.linalg.norm(db, axis=1, keepdims=True)

    ca, cb = cam_a.center, cam_b.center
    w0 = ca - cb
    a = np.ones(len(da))
    b = np.einsum("ij,ij->i", da, db)
    d = np.einsum("ij,j->i", da, w0)
    e = np.einsum("ij,j->i", db, w0)
    denom = a - b * b
    safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    sc = (b * e - d) / safe
    tc = (e - b * d) / safe
    pa = ca + sc[:, None] * da
    pb = cb + tc[:, None] * db
    X = 0.5 * (pa + pb)

    angle = np.arccos(np.clip(np.abs(b), -1.0, 1.0))
    za = (X @ cam_a.R.T + cam_a.t)[:, 2]
    zb = (X @ cam_b.R.T + cam_b.t)[:, 2]
    return X, angle, za, zb


# --------------------------------------------------------------------------
# pairwise matching
# --------------------------------------------------------------------------


def _sample(img: np.ndarray, yx: np.ndarray) -> np.ndarray:
    """Bilinear lookup; out-of-range comes back as NaN so callers can drop it."""
    vals = ndimage.map_coordinates(img, yx, order=1, mode="constant", cval=np.nan)
    return vals


def _patch_grid(size: int) -> np.ndarray:
    r = size // 2
    g = np.arange(-r, r + 1, dtype=float)
    dy, dx = np.meshgrid(g, g, indexing="ij")
    return np.stack([dx.ravel(), dy.ravel()], axis=1)  # (P,2) as (dx, dy)


def _warped_ncc(
    work_a: np.ndarray, work_b: np.ndarray,
    scale_a: float, scale_b: float,
    cam_a: _Cam, cam_b: _Cam,
    uv_a: np.ndarray, uv_b: np.ndarray,
    X: np.ndarray,
    patch: int,
    chunk: int = 20000,
) -> np.ndarray:
    """Correlation between patches the known geometry has already aligned.

    For each candidate, the plane through its triangulated point that is
    fronto-parallel to camera A induces a homography onto camera B.  Its
    derivative at the match is the affine map taking A's neighbourhood onto B's,
    which is the rotation, the shear and the scale change between the two views
    of that surface -- all of it known in advance, none of it searched for.

    Without the warp this scene correlates badly for an uninteresting reason:
    the photographs are forty-odd degrees apart, so a square window in A covers
    a visibly different piece of the world than the same square in B, and the
    real matches score no better than the accidental ones.  The geometry removes
    that difference before the comparison instead of asking the comparison to
    tolerate it.
    """
    n = len(X)
    out = np.full(n, -1.0)
    grid = _patch_grid(patch)  # (P,2) in B's working pixels
    P = len(grid)

    R_rel = cam_b.R @ cam_a.R.T
    t_rel = cam_b.t - R_rel @ cam_a.t
    Ka = cam_a.cam.K()
    Kb = cam_b.cam.K()
    Ka_inv = cam_a.cam.K_inv()

    # depth of each point in A, which is the plane's distance
    za = (X @ cam_a.R.T + cam_a.t)[:, 2]

    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        sl = slice(start, stop)
        m = stop - start
        z = za[sl]
        bad = np.abs(z) < 1e-9
        z = np.where(bad, 1.0, z)

        # H = Kb (R_rel + t_rel n^T / z) Ka^-1, with n = (0,0,1) in A's frame
        M = np.broadcast_to(R_rel, (m, 3, 3)).copy()
        M[:, :, 2] += t_rel[None, :] / z[:, None]
        H = Kb @ M @ Ka_inv  # (m,3,3)

        pa = np.concatenate([uv_a[sl], np.ones((m, 1))], axis=1)
        q = np.einsum("nij,nj->ni", H, pa)
        qz = np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])
        # d(pixel)/d(uv_a), the affine part of H at this point
        J = (H[:, :2, :2] * qz[:, None, None]
             - q[:, :2, None] * H[:, 2:3, :2]) / (qz * qz)[:, None, None]
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        singular = ~np.isfinite(det) | (np.abs(det) < 1e-9) | bad
        safe_det = np.where(singular, 1.0, det)
        Jinv = np.empty_like(J)
        Jinv[:, 0, 0] = J[:, 1, 1] / safe_det
        Jinv[:, 1, 1] = J[:, 0, 0] / safe_det
        Jinv[:, 0, 1] = -J[:, 0, 1] / safe_det
        Jinv[:, 1, 0] = -J[:, 1, 0] / safe_det

        # B-side samples: the candidate's own neighbourhood, at working scale
        gb = grid[None, :, :] / scale_b  # (1,P,2) full-res offsets
        b_uv = uv_b[sl][:, None, :] + gb
        b_yx = np.stack([(b_uv[..., 1] + 0.5) * scale_b - 0.5,
                         (b_uv[..., 0] + 0.5) * scale_b - 0.5], axis=0)
        vb = _sample(work_b, b_yx.reshape(2, -1)).reshape(m, P)

        # A-side samples: the same B neighbourhood pulled back through the warp
        delta_a = np.einsum("nij,pj->npi", Jinv, grid / scale_b)  # full-res A offsets
        a_uv = uv_a[sl][:, None, :] + delta_a
        a_yx = np.stack([(a_uv[..., 1] + 0.5) * scale_a - 0.5,
                         (a_uv[..., 0] + 0.5) * scale_a - 0.5], axis=0)
        va = _sample(work_a, a_yx.reshape(2, -1)).reshape(m, P)

        ok = np.isfinite(va).all(axis=1) & np.isfinite(vb).all(axis=1) & ~singular
        if not ok.any():
            continue
        A = va[ok]
        B = vb[ok]
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(B, axis=1)
        good = (na > 1e-6) & (nb > 1e-6)
        scores = np.full(int(ok.sum()), -1.0)
        scores[good] = np.einsum("ij,ij->i", A[good], B[good]) / (na[good] * nb[good])
        res = np.full(m, -1.0)
        res[ok] = scores
        out[sl] = res
    return out


def match_pair(
    cam_a: _Cam, cam_b: _Cam,
    uv_a: np.ndarray, uv_b: np.ndarray,
    work_a: np.ndarray, work_b: np.ndarray,
    scale_a: float, scale_b: float,
    bounds: tuple[np.ndarray, np.ndarray],
    s: HarvestSettings,
) -> list[tuple[int, int, float, np.ndarray]]:
    """Mutual matches between two photographs, geometry first.

    Returns ``(index in A, index in B, ncc, world point)``.
    """
    if not len(uv_a) or not len(uv_b):
        return []

    xa = cam_a.normalize(uv_a)  # (Na,3)
    xb = cam_b.normalize(uv_b)  # (Nb,3)

    R_rel = cam_b.R @ cam_a.R.T
    t_rel = cam_b.t - R_rel @ cam_a.t
    baseline = float(np.linalg.norm(t_rel))
    if baseline < 1e-9:
        return []
    E = np.cross(np.eye(3), t_rel) @ R_rel  # [t]_x R

    # Epipolar distance in BOTH images.  One direction alone is not symmetric --
    # a line nearly through the epipole in B is a tight test there and a loose
    # one in A -- and a pair that only passes in the easy direction is exactly
    # the pair most likely to be a coincidence.
    #
    # Chunked over A, because the full table is Na x Nb and three of them are
    # needed at once: at the corner budget this runs at that is a gigabyte of
    # float64 for a test whose answer is a few thousand index pairs.
    lb = xa @ E.T                      # (Na,3) lines in B's normalised plane
    la = xb @ E                        # (Nb,3) lines in A's normalised plane
    nb_ = np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12)
    na_ = np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12)
    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    step = max(1, int(4_000_000 / max(len(xb), 1)))
    for start in range(0, len(xa), step):
        stop = min(len(xa), start + step)
        num = lb[start:stop] @ xb.T    # (chunk,Nb)  x_b^T E x_a
        absnum = np.abs(num)
        ok = ((absnum / nb_[start:stop, None] * cam_b.cam.f < s.epipolar_px)
              & (absnum / na_[None, :] * cam_a.cam.f < s.epipolar_px))
        r, c = np.nonzero(ok)
        if len(r):
            rows_a.append(r + start)
            rows_b.append(c)
    if not rows_a:
        return []
    ia = np.concatenate(rows_a)
    ib = np.concatenate(rows_b)

    X, angle, za, zb = _triangulate_pair(cam_a, cam_b, xa[ia], xb[ib])
    lo, hi = bounds
    inside = (
        (za > 0) & (zb > 0)
        & (angle >= math.radians(s.min_triangulation_deg))
        & np.all(X >= lo[None, :], axis=1) & np.all(X <= hi[None, :], axis=1)
    )
    ia, ib, X = ia[inside], ib[inside], X[inside]
    if not len(ia):
        return []

    ncc = _warped_ncc(work_a, work_b, scale_a, scale_b, cam_a, cam_b,
                      uv_a[ia], uv_b[ib], X, s.patch)

    good = ncc > s.ncc_min
    ia, ib, X, ncc = ia[good], ib[good], X[good], ncc[good]
    if not len(ia):
        return []

    # ---- best and runner-up, each way round.  The same score array serves both
    # directions, so the mutual test costs nothing beyond two sorts.
    def _best_and_second(keys, scores):
        order = np.lexsort((-scores, keys))      # by key, then best score first
        k = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = k[1:] != k[:-1]
        pos = np.nonzero(first)[0]               # where each key's run starts
        best_idx = order[pos]
        nxt = np.minimum(pos + 1, len(order) - 1)
        has_second = (pos + 1 < len(order)) & (k[nxt] == k[pos])
        second = np.where(has_second, scores[order[nxt]], -1.0)
        return best_idx, second

    best_a, second_a = _best_and_second(ia, ncc)
    best_b, second_b = _best_and_second(ib, ncc)

    # Lowe's ratio on 1 - ncc.  A runner-up nearly as good as the winner means
    # the texture repeats along the line, which is the one ambiguity the poses
    # cannot resolve, so neither candidate is believed.
    def _unique(best_idx, second):
        d1 = 1.0 - ncc[best_idx]
        d2 = 1.0 - np.where(second < -0.5, -1.0, second)
        return (second < -0.5) | (d1 < s.uniqueness * d2)

    keep_a = set(best_a[_unique(best_a, second_a)].tolist())
    keep_b = set(best_b[_unique(best_b, second_b)].tolist())
    mutual = sorted(keep_a & keep_b)
    return [(int(ia[k]), int(ib[k]), float(ncc[k]), X[k]) for k in mutual]


# --------------------------------------------------------------------------
# tracks
# --------------------------------------------------------------------------


class _Union:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _extend_tracks(
    tracks: list[dict],
    cams: dict[str, _Cam],
    corners: dict[str, np.ndarray],
    work: dict[str, np.ndarray],
    scales: dict[str, float],
    trees: dict,
    s: HarvestSettings,
) -> int:
    """Project each track into the cameras it is missing and look for it there.

    The step that turns a pile of two-view matches into a set of three- and
    four-view tracks.  Pairwise matching can only link two photographs when the
    corner detector independently fired on the same physical feature in both,
    and chaining to a third needs it to fire a third time; that compound
    coincidence is rare enough that the first implementation of this module
    produced 280 pair matches and 24 usable tracks from them.

    Asking directly is both cheaper and stricter.  A two-view track already has
    a 3D position, so the question in the third photograph is not "what does
    this look like" but "is there a corner within fourteen pixels of *here*",
    and the answer is checked by the same warped correlation and the same
    uniqueness margin as a pairwise match.  The reference view is the one
    looking from the most similar direction, because that is the patch most
    likely to survive the warp.

    Returns the number of observations added.
    """
    ids = sorted(cams)
    jobs: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for t_i, tr in enumerate(tracks):
        X = tr["X"]
        members = tr["members"]
        for tgt in ids:
            if tgt in members:
                continue
            cam = cams[tgt]
            uv, z = cam.project(X[None, :])
            if z[0] <= 0:
                continue
            u, v = uv[0]
            if not (0 <= u < cam.width and 0 <= v < cam.height):
                continue
            near = trees[tgt].query_ball_point([u, v], s.extend_px)
            if not near:
                continue
            # reference view: smallest angle between the two lines of sight
            d_t = X - cam.center
            d_t = d_t / max(float(np.linalg.norm(d_t)), 1e-12)
            best_ref, best_cos = None, -2.0
            for iid in members:
                d_r = X - cams[iid].center
                c = float(d_r @ d_t / max(np.linalg.norm(d_r), 1e-12))
                if c > best_cos:
                    best_ref, best_cos = iid, c
            if best_ref is None:
                continue
            for c in near:
                jobs.setdefault((best_ref, tgt), []).append((t_i, members[best_ref], int(c)))

    # ---- score every candidate, one batch per ordered image pair
    proposals: dict[tuple[int, str], list[tuple[float, int]]] = {}
    for (ref, tgt), entries in jobs.items():
        t_idx = np.array([e[0] for e in entries])
        ref_idx = np.array([e[1] for e in entries])
        tgt_idx = np.array([e[2] for e in entries])
        X = np.array([tracks[i]["X"] for i in t_idx])
        ncc = _warped_ncc(
            work[ref], work[tgt], scales[ref], scales[tgt], cams[ref], cams[tgt],
            corners[ref][ref_idx], corners[tgt][tgt_idx], X, s.patch,
        )
        for ti, ci, score in zip(t_idx, tgt_idx, ncc):
            proposals.setdefault((int(ti), tgt), []).append((float(score), int(ci)))

    # ---- best candidate per (track, camera), with the same uniqueness margin
    claims: dict[tuple[str, int], tuple[float, int]] = {}
    for (t_i, tgt), cand in proposals.items():
        cand.sort(reverse=True)
        best, corner = cand[0]
        if best <= s.ncc_min:
            continue
        if len(cand) > 1:
            second = cand[1][0]
            if (1.0 - best) >= s.uniqueness * (1.0 - max(second, -1.0)):
                continue
        key = (tgt, corner)
        if key not in claims or claims[key][0] < best:
            claims[key] = (best, t_i)

    added = 0
    for (tgt, corner), (score, t_i) in claims.items():
        tr = tracks[t_i]
        if tgt in tr["members"]:
            continue
        tr["members"][tgt] = corner
        tr["ncc"].append(score)
        added += 1

    if added:
        for tr in tracks:
            members = list(tr["members"].items())
            if len(members) >= 2:
                X, worst, angle = triangulate_views(
                    [cams[i] for i, _ in members], [corners[i][k] for i, k in members]
                )
                if np.isfinite(worst):
                    tr["X"] = X
    return added


def _thin_over_frame(tracks: list[Track], cams: dict[str, _Cam], s: HarvestSettings) -> list[Track]:
    """Keep the best tracks subject to a per-cell quota in every photograph.

    A greedy pass down a quality ordering, where quality is views first and then
    reprojection: a four-view track is worth more to the bundle than a
    three-view one at the same residual, and no amount of extra correlation
    score makes up for a view it does not have.

    The quota is what the bundle actually needs.  Four hundred points clustered
    on one knurled surface constrain a camera's pose about as well as four do,
    because they all pull in the same direction; the same four hundred spread
    over the frame pin rotation and focal length as well as position.
    """
    cells = max(1, int(s.spread_grid))
    used: dict[tuple[str, int, int], int] = {}
    order = sorted(tracks, key=lambda tr: (-tr.n_views, tr.reproj_px, -tr.ncc))
    kept: list[Track] = []
    for tr in order:
        if len(kept) >= s.max_points:
            break
        slots = []
        full = False
        for iid, uv in tr.obs.items():
            cam = cams[iid]
            cell = max(cam.width, cam.height) / float(cells)
            key = (iid, int(uv[0] / cell), int(uv[1] / cell))
            if used.get(key, 0) >= s.max_per_cell:
                full = True
                break
            slots.append(key)
        if full:
            continue
        for key in slots:
            used[key] = used.get(key, 0) + 1
        kept.append(tr)
    return kept


# --------------------------------------------------------------------------
# the harvest
# --------------------------------------------------------------------------


def harvest(
    project: Project,
    settings: HarvestSettings | None = None,
    *,
    images: dict[str, np.ndarray] | None = None,
    progress=None,
) -> HarvestReport:
    """Find correspondences in a solved project and return them as tracks.

    ``images`` lets a caller supply already-decoded greyscale arrays, keyed by
    image id, which is what the tests do; otherwise each photograph is read from
    disk with the project's own loader so that EXIF orientation is applied
    exactly as it was when the marks were made.

    Nothing is written to the project here -- see :func:`add_tracks_to_project`.
    Keeping the two apart is what makes it possible to look at what a harvest
    found, with its scores and its triangulation residuals, before deciding to
    put it into a file the server will happily save over.
    """
    s = settings or HarvestSettings()
    report = HarvestReport()
    t_start = time.time()

    cams = cameras_of(project)
    if len(cams) < 2:
        report.messages.append(
            "fewer than two photographs carry a solved pose, so there is no geometry "
            "to match against -- solve the project first"
        )
        return report

    pts = scene_points(project)
    if len(pts) < 4:
        report.messages.append(
            "the project states fewer than four world points, which is not enough to "
            "say how big the scene is; the depth gate needs that"
        )
        return report
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    pad = s.depth_margin * max(float(span.max()), 1e-6)
    bounds = (lo - pad, hi + pad)

    # ---- decode and detect
    work: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    corners: dict[str, np.ndarray] = {}
    for iid, cam in sorted(cams.items()):
        if images is not None and iid in images:
            gray = np.asarray(images[iid], dtype=np.float32)
        else:
            from .edgefit import load_grayscale
            gray = load_grayscale(project.resolve_path(project.image(iid)))
        h0, w0 = gray.shape
        scale = min(1.0, s.detect_max_px / float(max(h0, w0)))
        work[iid] = (ndimage.zoom(gray, scale, order=1) if scale < 1.0 else gray)
        # the working array's own scale relative to the pixel frame the marks
        # live in, which is the *stated* size, not whatever Pillow decoded
        scales[iid] = work[iid].shape[1] / float(cam.width)
        # detection runs on the same working array -- it would otherwise do this
        # zoom a second time -- and maps its peaks back to the stated frame
        uv, _resp = detect_corners(work[iid], s, full_width=cam.width, full_height=cam.height)
        corners[iid] = uv
        report.corners[iid] = len(uv)
        if progress:
            progress(f"{iid}: {len(uv)} corners")

    # ---- pairwise
    ids = sorted(cams)
    uf = _Union()
    scores: dict[tuple[str, int], list[float]] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            matches = match_pair(
                cams[a], cams[b], corners[a], corners[b],
                work[a], work[b], scales[a], scales[b], bounds, s,
            )
            report.pair_matches[(a, b)] = len(matches)
            if progress:
                progress(f"{a}-{b}: {len(matches)} matches")
            for ka, kb, ncc, _X in matches:
                uf.union((a, ka), (b, kb))
                scores.setdefault((a, ka), []).append(ncc)
                scores.setdefault((b, kb), []).append(ncc)

    # ---- tracks
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)
    report.n_raw_tracks = len(groups)

    seeds: list[dict] = []
    n_conflict = 0
    for members in groups.values():
        seen = {iid for iid, _ in members}
        if len(seen) != len(members):
            # two corners in one photograph merged into one track: a chain of
            # pairwise matches walked onto a different copy of the same body
            # somewhere.  The chain cannot be untangled from here and the whole
            # component is suspect, so it goes.
            n_conflict += 1
            continue
        X, worst, _angle = triangulate_views(
            [cams[iid] for iid, _ in members], [corners[iid][k] for iid, k in members]
        )
        if not np.isfinite(worst):
            continue
        seeds.append({
            "members": {iid: k for iid, k in members},
            "X": X,
            "ncc": [v for iid, k in members for v in scores.get((iid, k), [])],
        })
    if n_conflict:
        report.messages.append(
            f"{n_conflict} track(s) merged two corners in the same photograph and were "
            "dropped whole -- that is what a chain through a repeated body looks like"
        )

    # ---- extension: ask the cameras a track has not reached whether it is there
    from scipy.spatial import cKDTree

    trees = {iid: cKDTree(uv) if len(uv) else None for iid, uv in corners.items()}
    if all(t is not None for t in trees.values()):
        for round_i in range(max(0, int(s.extend_rounds))):
            n_added = _extend_tracks(seeds, cams, corners, work, scales, trees, s)
            if progress:
                progress(f"extension pass {round_i + 1}: +{n_added} observation(s)")
            report.messages.append(
                f"extension pass {round_i + 1} added {n_added} observation(s)"
            )
            if not n_added:
                break

    tracks: list[Track] = []
    for tr in seeds:
        members = list(tr["members"].items())
        if len(members) < s.min_views:
            continue
        X, worst, angle = triangulate_views(
            [cams[i] for i, _ in members], [corners[i][k] for i, k in members]
        )
        if not np.isfinite(worst):
            continue
        tracks.append(Track(
            xyz=X,
            obs={iid: corners[iid][k] for iid, k in members},
            reproj_px=float(worst),
            angle_deg=float(angle),
            ncc=float(np.mean(tr["ncc"])) if tr["ncc"] else 0.0,
        ))
    report.n_after_views = len(tracks)

    lo_b, hi_b = bounds
    tracks = [
        tr for tr in tracks
        if tr.reproj_px <= s.max_track_reproj_px
        and tr.angle_deg >= s.min_triangulation_deg
        and bool(np.all(tr.xyz >= lo_b) and np.all(tr.xyz <= hi_b))
    ]
    report.n_after_reproj = len(tracks)

    tracks = _thin_over_frame(tracks, cams, s)
    report.n_kept = len(tracks)
    report.tracks = tracks
    report.seconds = time.time() - t_start

    if not tracks:
        report.warnings.append(
            "no correspondence survived.  Loosen --epipolar or --ncc-min, but check "
            "first that the project really is solved: a harvest against wrong poses "
            "finds nothing, which is the correct behaviour and not a bug"
        )
    return report


def add_tracks_to_project(
    project: Project, tracks: list[Track], settings: HarvestSettings | None = None
) -> list[str]:
    """Write tracks into the project as ordinary tie points.

    Ordinary on purpose.  The bundle already knows how to weight a tie point,
    cull one, triangulate one and report one; a harvested point that announced
    itself as a new kind of evidence would need all of that written twice, and
    the second copy would be the one with the bug in it.  The only mark they
    carry is the label, so they can be found and removed again.
    """
    s = settings or HarvestSettings()
    added: list[str] = []
    for n, tr in enumerate(tracks, 1):
        pid = project.next_point_id()
        project.points.append(TiePoint(
            id=pid,
            label=f"{s.label_prefix}{n:04d}",
            xyz=np.asarray(tr.xyz, dtype=float),
            uncertain=bool(s.uncertain),
        ))
        for iid, uv in tr.obs.items():
            im = project.image(iid)
            if im is not None:
                im.observations[pid] = [float(uv[0]), float(uv[1])]
        added.append(pid)
    return added


def remove_harvested(project: Project, settings: HarvestSettings | None = None) -> int:
    """Delete every harvested point and its marks.  The undo for the above."""
    s = settings or HarvestSettings()
    doomed = {p.id for p in project.points if p.label.startswith(s.label_prefix)
              and p.label[len(s.label_prefix):].isdigit()}
    if not doomed:
        return 0
    project.points = [p for p in project.points if p.id not in doomed]
    for im in project.images:
        for pid in doomed & set(im.observations):
            im.observations.pop(pid, None)
    return len(doomed)
