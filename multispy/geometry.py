"""Camera and vanishing-point geometry.

Conventions (OpenCV / COLMAP, so exports need no axis gymnastics):

* Image coordinates: x right, y down, origin at the top-left image corner.
* Camera coordinates: x right, y down, z forward into the scene.
* ``R`` is always the **world-to-camera** rotation, so ``x_cam = R @ X_world + t``.
  The columns of ``R`` are therefore the world axis directions expressed in
  camera coordinates -- which is exactly what vanishing points give us.
* Pixel projection: ``u = f * x/z + cx``, ``v = f * y/z + cy``.

Vanishing points are carried in **homogeneous** form ``(p, q, r)`` throughout.
A world axis parallel to the image plane has its vanishing point at infinity
(``r == 0``), which happens constantly in architectural photos, and the
homogeneous form handles it without special cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

AXIS_NAMES = ("X", "Y", "Z")


# --------------------------------------------------------------------------
# rotation helpers
# --------------------------------------------------------------------------


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle vector -> rotation matrix."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        # second-order expansion keeps the Jacobian well behaved near zero
        K = _skew(w)
        return np.eye(3) + K + 0.5 * (K @ K)
    axis = w / theta
    K = _skew(axis)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector."""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if abs(math.pi - theta) < 1e-6:
        # near pi the antisymmetric part vanishes; recover the axis from R + I
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            axis = A[:, k] / axis[k]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return axis * theta
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (theta / (2.0 * math.sin(theta)))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@dataclass
class SimilarityFit:
    """Least-squares similarity mapping one point set onto another."""

    scale: float
    R: np.ndarray
    t: np.ndarray
    rms: float
    """Residual after the fit, in target units. For a known object this is the
    single most informative number available: it says how far the marked points
    are from being that rigid shape at all, independently of any camera."""
    n: int
    collinearity: float
    """Ratio of the 2nd to the 1st singular value of the centred source points.
    Near zero means they lie on a line, which leaves the rotation about that
    line completely free -- the fit will still return *a* pose, just not a
    determined one."""

    @property
    def degenerate(self) -> bool:
        return self.n < 3 or self.collinearity < 0.02

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return self.scale * (points @ self.R.T) + self.t


def similarity_fit(
    source: np.ndarray, target: np.ndarray, *, allow_scale: bool = True
) -> SimilarityFit:
    """Umeyama: the similarity (or rigid) transform best taking ``source`` to ``target``.

    Used two ways when placing a known object. First with ``allow_scale=True``,
    where the recovered scale is pure diagnostic gold: the model's true size is
    known, so a scale that comes back as 1000 or 0.0394 is not a geometry problem
    but a declared-units problem, and saying so beats silently delivering a
    layout that is correct in shape and wrong by a factor of 1000. Then again
    with ``allow_scale=False`` to get the rigid pose the solver actually carries.
    """
    source = np.atleast_2d(np.asarray(source, dtype=float))
    target = np.atleast_2d(np.asarray(target, dtype=float))
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("source and target must be matching (N,3) arrays")
    n = len(source)
    if n < 2:
        raise ValueError("need at least 2 point pairs to fit a pose")

    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    S, T = source - mu_s, target - mu_t

    var_s = float(np.sum(S * S) / n)
    sigma = (T.T @ S) / n
    U, D, Vt = np.linalg.svd(sigma)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ correction @ Vt

    scale = 1.0
    if allow_scale and var_s > 1e-18:
        scale = float(np.trace(np.diag(D) @ correction) / var_s)
        if scale <= 0.0:  # a mirrored configuration; keep it rigid rather than flip
            scale = 1.0
    t = mu_t - scale * (R @ mu_s)

    residual = target - (scale * (source @ R.T) + t)
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))) if n else 0.0

    singular_source = np.linalg.svd(S, compute_uv=False)
    collinearity = (
        float(singular_source[1] / singular_source[0])
        if len(singular_source) > 1 and singular_source[0] > 1e-18
        else 0.0
    )
    return SimilarityFit(scale=scale, R=R, t=t, rms=rms, n=n, collinearity=collinearity)


def orthonormalize(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix to ``M`` (polar decomposition, det = +1)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return U @ D @ Vt


# --------------------------------------------------------------------------
# intrinsics
# --------------------------------------------------------------------------


@dataclass
class Intrinsics:
    """Pinhole intrinsics with one optional radial term (COLMAP SIMPLE_RADIAL)."""

    f: float
    cx: float
    cy: float
    k1: float = 0.0

    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    def K_inv(self) -> np.ndarray:
        return np.array(
            [
                [1.0 / self.f, 0.0, -self.cx / self.f],
                [0.0, 1.0 / self.f, -self.cy / self.f],
                [0.0, 0.0, 1.0],
            ]
        )

    def copy(self) -> "Intrinsics":
        return Intrinsics(self.f, self.cx, self.cy, self.k1)


def project(R: np.ndarray, t: np.ndarray, cam: Intrinsics, X: np.ndarray) -> np.ndarray:
    """Project world points ``X`` (N,3) to pixels (N,2). Points at or behind the
    camera plane come back as NaN so callers can drop them explicitly."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Xc = X @ R.T + t
    z = Xc[:, 2]
    bad = z <= 1e-9
    z_safe = np.where(bad, 1.0, z)
    xn = Xc[:, 0] / z_safe
    yn = Xc[:, 1] / z_safe
    if cam.k1 != 0.0:
        r2 = xn * xn + yn * yn
        s = 1.0 + cam.k1 * r2
        xn, yn = xn * s, yn * s
    uv = np.stack([cam.f * xn + cam.cx, cam.f * yn + cam.cy], axis=1)
    uv[bad] = np.nan
    return uv


# --------------------------------------------------------------------------
# vanishing points from marked line segments
# --------------------------------------------------------------------------


@dataclass
class VanishingPoint:
    """A vanishing point in homogeneous pixel coordinates."""

    v: np.ndarray  # (3,) unit-norm homogeneous
    condition: float  # ratio of 2nd-smallest to smallest eigenvalue; higher is better
    n_segments: int
    residual_px: float  # RMS perpendicular distance of segment endpoints to the VP pencil
    covariance: np.ndarray | None = None  # 2x2 positional covariance, px^2
    distance_px: float = 0.0  # from the image centre
    median_length_px: float = 0.0

    @property
    def at_infinity(self) -> bool:
        # bool() because v is a numpy array and numpy.bool_ is not JSON serialisable
        return bool(abs(self.v[2]) < 1e-9)

    def inhomogeneous(self) -> np.ndarray | None:
        if self.at_infinity:
            return None
        return self.v[:2] / self.v[2]

    def sigma_px(self) -> tuple[float, float]:
        """1-sigma positional spread, major and minor axis, in pixels."""
        if self.covariance is None:
            return (float("inf"), float("inf"))
        w = np.linalg.eigvalsh(self.covariance)
        return (float(np.sqrt(max(w[1], 0.0))), float(np.sqrt(max(w[0], 0.0))))


def vp_covariance(
    v_inhom: np.ndarray, segments: np.ndarray, sigma_px: float, max_sigma_px: float
) -> np.ndarray:
    """Positional covariance of a vanishing point, given endpoint marking error.

    A segment whose endpoints each wobble by ``sigma`` pins the line tightly
    between them and ever more loosely outside: at a point ``t`` segment-lengths
    from the first endpoint the perpendicular wobble is
    ``sigma * sqrt((1-t)^2 + t^2)``, which grows linearly once ``t >> 1``.

    That is the whole story of vanishing-point fragility. A vanishing point 30
    segment-lengths away inherits roughly 42x the marking error, perpendicular
    to the line. Each segment constrains the vanishing point only across its own
    direction, so a bundle of near-parallel segments leaves it almost free to
    slide along them -- which is exactly the direction the focal length depends
    on. ``max_sigma_px`` floors the information so that unconstrained direction
    stays finite rather than infinite.
    """
    A = np.zeros((2, 2))
    for seg in np.atleast_2d(np.asarray(segments, dtype=float)):
        p0, p1 = seg[0:2], seg[2:4]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        t = float((v_inhom - p0) @ u) / L
        s = sigma_px * math.sqrt((1.0 - t) ** 2 + t * t)
        A += np.outer(n, n) / max(s, 1e-9) ** 2
    A += np.eye(2) / (max_sigma_px**2)
    return np.linalg.inv(A)


@dataclass
class FocalObservability:
    """How well a pair of axes pins down the focal length.

    ``relative_sigma`` is the fractional 1-sigma uncertainty: 0.02 means the
    focal length is pinned to about +-2%.
    """

    axis_a: int
    axis_b: int
    focal_px: float | None
    relative_sigma: float | None
    limiting_axis: int | None
    note: str

    @property
    def verdict(self) -> str:
        if self.focal_px is None or self.relative_sigma is None:
            return "unusable"
        if self.relative_sigma < 0.02:
            return "good"
        if self.relative_sigma < 0.08:
            return "weak"
        return "poor"


def focal_observability(
    vp_a: VanishingPoint,
    vp_b: VanishingPoint,
    axis_a: int,
    axis_b: int,
    cx: float,
    cy: float,
) -> FocalObservability:
    """Propagate vanishing-point uncertainty into the focal length estimate.

    The focal comes from ``f^2 = -(d_a . d_b)`` with ``d`` measured from the
    principal point, so ``var(f^2) = d_a^T C_b d_a + d_b^T C_a d_b`` and
    ``sigma_f / f = sqrt(var(f^2)) / (2 f^2)``.

    The practical consequence is counter-intuitive: a vanishing point far from
    centre makes the *other* axis's uncertainty matter more, because ``d_a``
    multiplies ``C_b``. A near-infinite vanishing point is not a free win, it is
    an amplifier.
    """
    if vp_a.at_infinity or vp_b.at_infinity:
        return FocalObservability(
            axis_a, axis_b, None, None, None,
            "one axis vanishes at infinity, so this pair says nothing about focal length",
        )
    da = vp_a.inhomogeneous() - np.array([cx, cy])
    db = vp_b.inhomogeneous() - np.array([cx, cy])
    f_sq = -float(da @ db)
    if f_sq <= 0.0:
        return FocalObservability(
            axis_a, axis_b, None, None, None,
            "these axes are not perpendicular as marked, so the focal length comes out imaginary",
        )
    f = math.sqrt(f_sq)
    Ca = vp_a.covariance if vp_a.covariance is not None else np.eye(2) * 1e12
    Cb = vp_b.covariance if vp_b.covariance is not None else np.eye(2) * 1e12
    contrib_a = float(db @ Ca @ db)  # uncertainty coming from axis a
    contrib_b = float(da @ Cb @ da)
    rel = math.sqrt(max(contrib_a + contrib_b, 0.0)) / (2.0 * f_sq)
    limiting = axis_a if contrib_a > contrib_b else axis_b
    return FocalObservability(
        axis_a, axis_b, f, rel, limiting,
        f"focal {f:.0f} px +-{rel * 100:.1f}%, limited by the {AXIS_NAMES[limiting]} lines",
    )


def _norm_transform(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Similarity transform putting pixel coords in roughly [-1, 1].

    Conditioning of the vanishing-point eigenproblem is terrible in raw pixels
    (entries spanning 1 to 10^7), so everything is solved normalized.
    """
    s = max(width, height) / 2.0
    T = np.array([[1.0 / s, 0.0, -width / (2.0 * s)], [0.0, 1.0 / s, -height / (2.0 * s)], [0.0, 0.0, 1.0]])
    T_inv = np.array([[s, 0.0, width / 2.0], [0.0, s, height / 2.0], [0.0, 0.0, 1.0]])
    return T, T_inv


def estimate_vanishing_point(
    segments: np.ndarray, width: int, height: int, sigma_px: float = 1.5
) -> VanishingPoint:
    """Least-squares vanishing point of a bundle of line segments.

    ``segments`` is (N, 4) as ``[x0, y0, x1, y1]`` in pixels. Needs at least two
    non-parallel segments; a single segment leaves the vanishing point free to
    slide along it.

    Each segment contributes its line ``l``, normalised so that ``l . v`` reads
    as a perpendicular pixel distance, and weighted by segment length -- a 2px
    endpoint slip on a short segment says far less about the direction than the
    same slip on a long one.
    """
    segments = np.atleast_2d(np.asarray(segments, dtype=float))
    if len(segments) < 2:
        raise ValueError("need at least 2 segments to fix a vanishing point")

    T, T_inv = _norm_transform(width, height)
    p0 = np.hstack([segments[:, 0:2], np.ones((len(segments), 1))]) @ T.T
    p1 = np.hstack([segments[:, 2:4], np.ones((len(segments), 1))]) @ T.T

    lines = np.cross(p0, p1)
    nrm = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    if np.any(nrm < 1e-12):
        raise ValueError("a segment has zero length")
    lines = lines / nrm

    lengths = np.linalg.norm(p1[:, :2] / p1[:, 2:3] - p0[:, :2] / p0[:, 2:3], axis=1)
    M = (lines * lengths[:, None]).T @ lines

    evals, evecs = np.linalg.eigh(M)
    v_norm = evecs[:, 0]
    condition = float(evals[1] / evals[0]) if evals[0] > 1e-18 else float("inf")

    # RMS residual, reported back in pixels so the UI can flag sloppy marking
    s = max(width, height) / 2.0
    resid = float(np.sqrt(np.mean((lines @ v_norm) ** 2)) * s)

    v = T_inv @ v_norm  # points transform back by T^-1
    v = v / max(np.linalg.norm(v), 1e-300)

    vp = VanishingPoint(
        v=v,
        condition=condition,
        n_segments=len(segments),
        residual_px=resid,
        median_length_px=float(np.median(np.hypot(segments[:, 2] - segments[:, 0],
                                                  segments[:, 3] - segments[:, 1]))),
    )
    if not vp.at_infinity:
        inhom = vp.inhomogeneous()
        vp.distance_px = float(np.hypot(inhom[0] - width / 2.0, inhom[1] - height / 2.0))
        vp.covariance = vp_covariance(
            inhom, segments, sigma_px, max_sigma_px=50.0 * max(width, height)
        )
    return vp


def _observability_advice(analysis: "AxisAnalysis") -> str:
    """Say what to actually go and do about a poorly determined focal length."""
    best = analysis.best
    if best is None or best.limiting_axis is None:
        return "Mark two axes with at least two control lines each."
    axis = best.limiting_axis
    vp = analysis.axes.get(axis)
    if vp is None:
        return ""
    name = AXIS_NAMES[axis]
    if vp.at_infinity or vp.distance_px > 20000:
        return (
            f"The {name} lines are so nearly parallel in the image that their vanishing point is "
            f"effectively at infinity, which carries no focal information. You need a viewpoint "
            f"with visible convergence along {name}, or a different pair of axes."
        )
    ratio = vp.distance_px / max(vp.median_length_px, 1.0)
    return (
        f"The {name} vanishing point is {vp.distance_px:,.0f} px away while the {name} control "
        f"lines are only {vp.median_length_px:.0f} px long, a {ratio:.0f}x lever -- so one pixel "
        f"of marking error moves it about {ratio:.0f} px. Longer {name} lines, further apart, "
        f"are what help."
    )


@dataclass
class AxisAnalysis:
    """Everything the observability check knows about one image."""

    axes: dict[int, VanishingPoint] = field(default_factory=dict)
    pairs: list[FocalObservability] = field(default_factory=list)
    best: FocalObservability | None = None
    problems: list[str] = field(default_factory=list)


def analyse_axes(
    axis_segments: dict[int, np.ndarray],
    width: int,
    height: int,
    *,
    sigma_px: float = 1.5,
    cx: float | None = None,
    cy: float | None = None,
) -> AxisAnalysis:
    """Can this image's control lines determine a focal length, and how well?

    Answerable from four control lines on two axes, long before a full markup
    exists -- which is the point. Nothing here needs a solve.
    """
    out = AxisAnalysis()
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy

    for axis, segs in sorted(axis_segments.items()):
        segs = np.atleast_2d(np.asarray(segs, dtype=float))
        if len(segs) < 2:
            if len(segs):
                out.problems.append(
                    f"{AXIS_NAMES[axis]}: only {len(segs)} control line, needs 2 to fix a "
                    "vanishing point"
                )
            continue
        try:
            out.axes[axis] = estimate_vanishing_point(segs, width, height, sigma_px)
        except ValueError as exc:
            out.problems.append(f"{AXIS_NAMES[axis]}: {exc}")

    axes = sorted(out.axes)
    if len(axes) < 2:
        out.problems.append("need 2 axes with 2+ control lines each before focal length is defined")
        return out

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a, b = axes[i], axes[j]
            out.pairs.append(focal_observability(out.axes[a], out.axes[b], a, b, cx, cy))

    usable = [p for p in out.pairs if p.relative_sigma is not None]
    if usable:
        out.best = min(usable, key=lambda p: p.relative_sigma)
    return out


# --------------------------------------------------------------------------
# intrinsics and rotation from vanishing points
# --------------------------------------------------------------------------


def focal_from_two_vps(v1: np.ndarray, v2: np.ndarray, cx: float, cy: float) -> float | None:
    """Focal length from two orthogonal vanishing points and a known principal point.

    From the image of the absolute conic, ``v1^T w v2 = 0`` with
    ``w ~ [[1,0,-cx],[0,1,-cy],[-cx,-cy,cx^2+cy^2+f^2]]``. Returns None when the
    geometry cannot determine a focal length: either vanishing point at infinity
    (the constraint then says something about the principal point instead), or a
    sign that would need an imaginary focal length (the usual symptom of
    mis-assigned axes or badly drawn control lines).
    """
    p1, q1, r1 = (float(x) for x in v1)
    p2, q2, r2 = (float(x) for x in v2)
    if abs(r1) < 1e-9 or abs(r2) < 1e-9:
        return None
    num = (
        p1 * p2
        + q1 * q2
        - cx * (p1 * r2 + r1 * p2)
        - cy * (q1 * r2 + r1 * q2)
        + (cx * cx + cy * cy) * r1 * r2
    )
    f_sq = -num / (r1 * r2)
    if f_sq <= 0.0 or not math.isfinite(f_sq):
        return None
    return math.sqrt(f_sq)


def principal_point_from_three_vps(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
) -> np.ndarray | None:
    """Principal point as the orthocenter of the triangle of three orthogonal
    vanishing points. Classical, and classically noise-sensitive -- prefer
    locking the principal point to the image centre unless the control lines are
    very clean."""
    pts = []
    for v in (v1, v2, v3):
        if abs(float(v[2])) < 1e-9:
            return None
        pts.append(np.asarray(v[:2], dtype=float) / float(v[2]))
    a, b, c = pts
    # orthocenter: intersection of two altitudes
    A = np.array([b - c, a - c])
    rhs = np.array([a @ (b - c), b @ (a - c)])
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None


def direction_from_vp(v: np.ndarray, cam: Intrinsics) -> np.ndarray:
    """Unit camera-space direction of the world axis with vanishing point ``v``.

    Sign is ambiguous: ``d`` and ``-d`` share the same vanishing point, so the
    caller must disambiguate from how the control lines were drawn.
    """
    d = cam.K_inv() @ np.asarray(v, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("degenerate vanishing point")
    return d / n


def resolve_direction_sign(d: np.ndarray, v: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Flip ``d`` so it agrees with the direction the control lines were drawn in.

    Both ``+d`` and ``-d`` share one vanishing point, so the image alone cannot
    settle the sign -- but the *drawn* direction can. Writing a point travelling
    along ``d`` as ``X(s) = X0 + s*d``, its image satisfies
    ``p(s) - vp = C / depth(s)``, so the image approaches the vanishing point
    exactly when depth grows with ``s``, i.e. when the camera-space direction has
    a positive z component.

    So: canonicalise ``d`` to point away from the camera first (otherwise the
    test below is comparing against the arbitrary sign that came out of the
    vanishing point eigenvector), then flip if the user drew the other way.

    A vanishing point at infinity is a separate case: there ``d`` is
    proportional to the homogeneous ``(p, q, 0)``, so its sign already tracks
    ``v``, and the drawn displacement is compared with ``(p, q)`` directly.
    """
    segments = np.atleast_2d(np.asarray(segments, dtype=float))
    v = np.asarray(v, dtype=float)
    at_infinity = abs(v[2]) < 1e-9
    if not at_infinity and d[2] < 0.0:
        d = -d

    score = 0.0
    for seg in segments:
        p0, p1 = seg[0:2], seg[2:4]
        drawn = p1 - p0
        toward = v[:2] if at_infinity else v[:2] / v[2] - 0.5 * (p0 + p1)
        n = np.linalg.norm(toward)
        if n < 1e-12:
            continue
        score += float(drawn @ (toward / n))
    return d if score >= 0.0 else -d


@dataclass
class Calibration:
    """Result of calibrating one image from its control lines alone."""

    intrinsics: Intrinsics
    R: np.ndarray
    axes_used: tuple[int, ...]
    vps: dict[int, VanishingPoint] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    orthogonality_error_deg: float = 0.0
    observability: AxisAnalysis | None = None
    focal_relative_sigma: float | None = None


def calibrate_image(
    axis_segments: dict[int, np.ndarray],
    width: int,
    height: int,
    *,
    solve_principal_point: bool = False,
    fixed_focal: float | None = None,
    sigma_px: float = 1.5,
) -> Calibration:
    """fSpy, essentially: control lines for 2-3 world axes -> focal + rotation.

    ``axis_segments`` maps axis index (0=X, 1=Y, 2=Z) to an (N,4) array of
    segments. Two axes with >= 2 segments each is the minimum.

    This is the *initialiser*. Its estimates are noisy by nature -- a distant
    vanishing point moves hundreds of pixels for a two-pixel slip on a control
    line -- and the whole point of the bundle adjustment downstream is that tie
    points across images outvote that noise.
    """
    warnings: list[str] = []
    vps: dict[int, VanishingPoint] = {}
    for axis, segs in sorted(axis_segments.items()):
        segs = np.atleast_2d(np.asarray(segs, dtype=float))
        if len(segs) < 2:
            warnings.append(f"axis {AXIS_NAMES[axis]}: only {len(segs)} control line(s), need 2")
            continue
        vp = estimate_vanishing_point(segs, width, height, sigma_px)
        if vp.condition < 1.05:
            warnings.append(
                f"axis {AXIS_NAMES[axis]}: control lines are nearly parallel in the image, "
                "so the vanishing point is poorly localised"
            )
        vps[axis] = vp

    if len(vps) < 2:
        raise ValueError("need usable control lines for at least 2 axes")

    cx, cy = width / 2.0, height / 2.0
    axes = tuple(sorted(vps))

    if solve_principal_point and len(axes) >= 3:
        pp = principal_point_from_three_vps(*(vps[a].v for a in axes[:3]))
        if pp is None:
            warnings.append("could not solve principal point from 3 vanishing points; using image centre")
        else:
            off = math.hypot(pp[0] - width / 2.0, pp[1] - height / 2.0) / max(width, height)
            if off > 0.25:
                warnings.append(
                    f"solved principal point is {off * 100:.0f}% of the frame off-centre, "
                    "which usually means a mis-drawn control line; using image centre"
                )
            else:
                cx, cy = float(pp[0]), float(pp[1])

    observability = AxisAnalysis(axes=vps)
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a, b = axes[i], axes[j]
            observability.pairs.append(focal_observability(vps[a], vps[b], a, b, cx, cy))
    usable = [p for p in observability.pairs if p.relative_sigma is not None]
    if usable:
        observability.best = min(usable, key=lambda p: p.relative_sigma)
    focal_sigma = observability.best.relative_sigma if observability.best else None

    if fixed_focal is not None:
        f = float(fixed_focal)
    else:
        if not usable:
            raise ValueError(
                "no vanishing point pair yields a real focal length -- check that the control "
                "lines are assigned to the right axes and that they really are parallel in 3D"
            )
        # Inverse-variance weighting, not a plain median: a pair whose vanishing
        # points sit near the image is worth orders of magnitude more than one
        # relying on a near-infinite vanishing point, and averaging them as
        # equals throws that away.
        weights = np.array([1.0 / max(p.relative_sigma, 1e-6) ** 2 for p in usable])
        values = np.array([p.focal_px for p in usable])
        f = float(np.sum(weights * values) / np.sum(weights))
        if len(usable) > 1:
            spread = (values.max() - values.min()) / f
            if spread > 0.25:
                warnings.append(
                    f"focal estimates from different axis pairs disagree by {spread * 100:.0f}% "
                    f"(weighted toward the {AXIS_NAMES[usable[int(np.argmax(weights))].axis_a]}"
                    f"x{AXIS_NAMES[usable[int(np.argmax(weights))].axis_b]} pair, the best "
                    f"determined at +-{min(p.relative_sigma for p in usable) * 100:.1f}%)"
                )
        if focal_sigma is not None and focal_sigma > 0.08:
            warnings.append(
                f"focal length is barely observable from these control lines "
                f"(+-{focal_sigma * 100:.0f}%). "
                + _observability_advice(observability)
            )

    cam = Intrinsics(f=f, cx=cx, cy=cy)

    dirs: dict[int, np.ndarray] = {}
    for axis, vp in vps.items():
        d = direction_from_vp(vp.v, cam)
        dirs[axis] = resolve_direction_sign(d, vp.v, axis_segments[axis])

    a0, a1 = axes[0], axes[1]
    d0, d1 = dirs[a0], dirs[a1]
    ortho_err = math.degrees(math.asin(max(-1.0, min(1.0, abs(float(d0 @ d1))))))
    if ortho_err > 10.0:
        warnings.append(
            f"axes {AXIS_NAMES[a0]} and {AXIS_NAMES[a1]} come out {ortho_err:.0f} deg from "
            "perpendicular; the focal length absorbing that error is why single-image solves drift"
        )

    cols: dict[int, np.ndarray] = {a0: d0, a1: d1}
    missing = ({0, 1, 2} - {a0, a1}).pop()
    # Right-handed completion. X*Y=Z and Y*Z=X hold directly; X*Z=-Y needs the flip.
    order = sorted([a0, a1])
    cross = np.cross(cols[order[0]], cols[order[1]])
    cross = cross / max(np.linalg.norm(cross), 1e-12)
    if order == [0, 2]:
        cross = -cross
    cols[missing] = cross

    if missing in dirs:
        drawn = dirs[missing]
        if float(drawn @ cols[missing]) < 0.0:
            warnings.append(
                f"axis {AXIS_NAMES[missing]} control lines were drawn opposite to the "
                "right-handed completion of the other two axes; flip them or expect a mirrored solve"
            )

    R = orthonormalize(np.column_stack([cols[0], cols[1], cols[2]]))
    return Calibration(
        intrinsics=cam,
        R=R,
        axes_used=axes,
        vps=vps,
        warnings=warnings,
        orthogonality_error_deg=ortho_err,
        observability=observability,
        focal_relative_sigma=focal_sigma,
    )
