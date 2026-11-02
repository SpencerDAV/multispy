"""Validation against a synthetic Manhattan-world scene with known ground truth.

Builds a cuboid, places cameras around it, generates exactly what a user would
click -- control lines along visible box edges and tie points on visible corners
-- adds realistic click noise, then checks that the solve recovers the truth.

The last test is the one that matters for the whole premise of the tool: it
compares the focal length error from independent single-image solves (what fSpy
gives you) against the error after the joint solve.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_synthetic.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.geometry import Intrinsics, calibrate_image, project as project_points
from multispy.project import (
    AxisLine,
    AxisLockConstraint,
    CameraGroup,
    DistanceConstraint,
    ImageView,
    Project,
    TiePoint,
)
from multispy.solve import solve

WIDTH, HEIGHT = 4000, 3000
TRUE_FOCAL = 3200.0
BOX = np.array([10.0, 6.0, 4.0])  # x, y, z extents; z is up


# --------------------------------------------------------------------------
# synthetic scene
# --------------------------------------------------------------------------


def look_at(center: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0])):
    """World-to-camera rotation for an x-right, y-down, z-forward camera."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.array([x, y, z])  # rows are the camera axes in world coordinates
    t = -R @ center
    return R, t


def box_points() -> tuple[np.ndarray, np.ndarray]:
    """Points on the surface of the cuboid, with their outward face normals."""
    pts, normals = [], []
    nx, ny, nz = 3, 3, 3
    xs = np.linspace(0, BOX[0], nx)
    ys = np.linspace(0, BOX[1], ny)
    zs = np.linspace(0, BOX[2], nz)
    for face, normal in (
        ("y0", [0, -1, 0]),
        ("x0", [-1, 0, 0]),
        ("z1", [0, 0, 1]),
        ("y1", [0, 1, 0]),
        ("x1", [1, 0, 0]),
    ):
        if face == "y0":
            grid = [(x, 0.0, z) for x in xs for z in zs]
        elif face == "y1":
            grid = [(x, BOX[1], z) for x in xs for z in zs]
        elif face == "x0":
            grid = [(0.0, y, z) for y in ys for z in zs]
        elif face == "x1":
            grid = [(BOX[0], y, z) for y in ys for z in zs]
        else:
            grid = [(x, y, BOX[2]) for x in xs for y in ys]
        for g in grid:
            pts.append(g)
            normals.append(normal)

    pts = np.array(pts, dtype=float)
    normals = np.array(normals, dtype=float)
    # drop duplicates along shared edges, keeping the first normal seen
    _, keep = np.unique(np.round(pts, 6), axis=0, return_index=True)
    keep = np.sort(keep)
    return pts[keep], normals[keep]


def camera_rig() -> list[tuple[np.ndarray, np.ndarray]]:
    """Cameras on an arc in front of and around the box, all roughly eye height."""
    target = BOX / 2.0
    poses = []
    # kept inside +-65 degrees: past that the front face is so foreshortened
    # that no two visible points share an axis-aligned edge long enough to draw
    for angle_deg, dist, height in (
        (-62.0, 18.0, 1.7),
        (-38.0, 15.0, 2.2),
        (-12.0, 14.0, 1.5),
        (16.0, 16.0, 2.8),
        (41.0, 19.0, 1.6),
        (64.0, 17.0, 2.4),
    ):
        a = math.radians(angle_deg)
        center = target + np.array([math.sin(a) * dist, -math.cos(a) * dist, 0.0])
        center[2] = height
        poses.append(look_at(center, target))
    return poses


def build_project(
    seed: int = 7,
    tie_noise_px: float = 1.0,
    line_noise_px: float = 1.5,
    max_lines_per_axis: int = 3,
) -> tuple[Project, dict]:
    """Generate a project containing only what a user could actually have clicked."""
    rng = np.random.default_rng(seed)
    pts, normals = box_points()
    poses = camera_rig()
    cam_true = Intrinsics(f=TRUE_FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)

    project = Project(name="synthetic")
    project.groups["cam0"] = CameraGroup(id="cam0")
    for i in range(len(pts)):
        project.points.append(TiePoint(id=f"p{i}", label=f"corner {i}"))

    truth = {"R": [], "t": [], "centers": [], "points": pts, "focal": TRUE_FOCAL}

    for ci, (R, t) in enumerate(poses):
        center = -R.T @ t
        uv = project_points(R, t, cam_true, pts)

        inside = (
            np.isfinite(uv[:, 0])
            & (uv[:, 0] > 20)
            & (uv[:, 0] < WIDTH - 20)
            & (uv[:, 1] > 20)
            & (uv[:, 1] < HEIGHT - 20)
        )
        # a user can only click a point on a face turned toward the camera
        facing = np.einsum("ij,ij->i", normals, center[None, :] - pts) > 0.0
        visible = inside & facing

        image = ImageView(
            id=f"img{ci}",
            path=f"photos/synthetic_{ci:02d}.jpg",
            width=WIDTH,
            height=HEIGHT,
            group="cam0",
        )

        for pi in np.flatnonzero(visible):
            noisy = uv[pi] + rng.normal(0.0, tie_noise_px, 2)
            image.observations[f"p{pi}"] = [float(noisy[0]), float(noisy[1])]

        # control lines along box edges: pairs of visible points differing in
        # exactly one world coordinate, longest first
        vis_idx = np.flatnonzero(visible)
        for axis in range(3):
            candidates = []
            for a_i in range(len(vis_idx)):
                for b_i in range(a_i + 1, len(vis_idx)):
                    a, b = vis_idx[a_i], vis_idx[b_i]
                    delta = pts[b] - pts[a]
                    others = [k for k in range(3) if k != axis]
                    if abs(delta[axis]) < 1e-6:
                        continue
                    if any(abs(delta[k]) > 1e-6 for k in others):
                        continue
                    lo, hi = (a, b) if delta[axis] > 0 else (b, a)
                    length = float(np.linalg.norm(uv[hi] - uv[lo]))
                    if length < 90.0:
                        continue
                    candidates.append((length, lo, hi))
            candidates.sort(reverse=True)
            for _length, lo, hi in candidates[:max_lines_per_axis]:
                p0 = uv[lo] + rng.normal(0.0, line_noise_px, 2)
                p1 = uv[hi] + rng.normal(0.0, line_noise_px, 2)
                image.axis_lines.append(
                    AxisLine(axis=axis, p0=[float(p0[0]), float(p0[1])], p1=[float(p1[0]), float(p1[1])])
                )

        project.images.append(image)
        truth["R"].append(R)
        truth["t"].append(t)
        truth["centers"].append(center)

    truth["centers"] = np.array(truth["centers"])
    return project, truth


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------


def umeyama(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) best mapping A onto B.

    Translation and scale are gauge freedoms of the solve unless the user pins
    them with constraints, so recovered geometry has to be compared up to a
    similarity.
    """
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - mu_a, B - mu_b
    H = A0.T @ B0 / len(A)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = (U @ D @ Vt).T
    var_a = float(np.mean(np.sum(A0**2, axis=1)))
    s = float(np.trace(np.diag(S) @ D) / var_a) if var_a > 1e-18 else 1.0
    t = mu_b - s * R @ mu_a
    return s, R, t


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    cos = (np.trace(R_est @ R_true.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_solve_recovers_ground_truth():
    project, truth = build_project()
    report = solve(project)
    assert report.ok, f"solve failed at stage {report.stage}: {report.messages}"
    assert report.n_images_positioned == 6, report.images
    assert report.rms_reprojection_px < 2.5, report.rms_reprojection_px

    focal = report.focals["cam0"]
    assert abs(focal - TRUE_FOCAL) / TRUE_FOCAL < 0.02, focal

    for im, R_true in zip(project.images, truth["R"]):
        assert rotation_error_deg(im.R, R_true) < 1.0, (im.id, rotation_error_deg(im.R, R_true))

    centers = np.array([im.center() for im in project.images])
    s, R, t = umeyama(centers, truth["centers"])
    aligned = (s * (R @ centers.T).T) + t
    err = np.linalg.norm(aligned - truth["centers"], axis=1)
    extent = float(np.linalg.norm(BOX))
    assert err.max() / extent < 0.02, err / extent


def test_distance_constraint_sets_metric_scale():
    project, truth = build_project()
    pts = truth["points"]
    # a known 10 m baseline along the bottom front edge of the box
    a = int(np.argmin(np.linalg.norm(pts - np.array([0.0, 0.0, 0.0]), axis=1)))
    b = int(np.argmin(np.linalg.norm(pts - np.array([BOX[0], 0.0, 0.0]), axis=1)))
    true_dist = float(np.linalg.norm(pts[b] - pts[a]))
    project.constraints.append(
        DistanceConstraint(point_a=f"p{a}", point_b=f"p{b}", value=true_dist, sigma=0.005)
    )
    project.constraints.append(AxisLockConstraint(point=f"p{a}", axis=0, value=0.0, sigma=0.005))
    project.constraints.append(AxisLockConstraint(point=f"p{a}", axis=1, value=0.0, sigma=0.005))
    project.constraints.append(AxisLockConstraint(point=f"p{a}", axis=2, value=0.0, sigma=0.005))

    report = solve(project)
    assert report.ok, report.messages
    assert report.scale_is_metric

    solved = {p.id: p.xyz for p in project.points if p.xyz is not None}
    got = float(np.linalg.norm(solved[f"p{b}"] - solved[f"p{a}"]))
    assert abs(got - true_dist) / true_dist < 0.01, (got, true_dist)

    # with scale and origin pinned, distances should now be right everywhere
    ids = sorted(solved)
    est = np.array([solved[i] for i in ids])
    tru = np.array([pts[int(i[1:])] for i in ids])
    s, R, t = umeyama(est, tru)
    assert abs(s - 1.0) < 0.02, s


def test_joint_solve_beats_independent_single_image_solves():
    """The premise of the tool, measured.

    Each image's control lines alone give a focal length determined by a handful
    of clicks with no redundancy. Sharing one focal length across images and
    tying them together with corresponding points should cut that error a lot.
    """
    project, truth = build_project()

    single_image = []
    for im in project.images:
        segs = {a: s for a, s in im.axis_segments().items() if len(s) >= 2}
        if len(segs) < 2:
            continue
        try:
            calib = calibrate_image(segs, im.width, im.height)
        except ValueError:
            continue
        single_image.append(calib.intrinsics.f)

    assert len(single_image) >= 4, "expected most synthetic images to calibrate standalone"
    single_err = np.abs(np.array(single_image) - TRUE_FOCAL) / TRUE_FOCAL

    report = solve(project)
    assert report.ok, report.messages
    joint_err = abs(report.focals["cam0"] - TRUE_FOCAL) / TRUE_FOCAL

    print(f"\n  single-image focal error: median {np.median(single_err) * 100:5.2f}%  "
          f"worst {single_err.max() * 100:5.2f}%  spread {single_err.std() * 100:5.2f}%")
    print(f"  joint solve focal error:  {joint_err * 100:5.2f}%")

    assert joint_err < np.median(single_err), (joint_err, np.median(single_err))


def test_focal_observability_brackets_the_real_error():
    """The predicted uncertainty has to be honest, or it is worse than no warning.

    Checks both directions: it must not be so loose that it never warns, and not
    so tight that the error it predicts is smaller than the error made.
    """
    from multispy.geometry import analyse_axes

    project, _ = build_project()
    per_image = []
    for im in project.images:
        a = analyse_axes(im.axis_segments(), im.width, im.height, sigma_px=1.5)
        assert a.best is not None, f"{im.id} produced no usable axis pair"
        per_image.append(a.best.relative_sigma)

    # the synthetic box has real convergence, so at least one view is tightly pinned
    assert min(per_image) < 0.03, per_image

    report = solve(project)
    assert report.ok, report.messages
    assert report.focal_sigma_pct is not None
    predicted = report.focal_sigma_pct / 100.0
    actual = abs(report.focals["cam0"] - TRUE_FOCAL) / TRUE_FOCAL

    # not wildly over-confident ...
    assert actual <= 4 * predicted + 0.005, (actual, predicted)
    # ... and not so pessimistic that a good set still looks hopeless
    assert predicted < 0.08, predicted


def test_observability_flags_weak_control_lines():
    """Same directions, far less leverage: the prediction must get much worse."""
    from multispy.geometry import analyse_axes

    project, _ = build_project()
    im = project.images[0]
    good = analyse_axes(im.axis_segments(), im.width, im.height, sigma_px=1.5)

    stubs = {}
    for axis, segs in im.axis_segments().items():
        out = []
        for s in segs:
            mid = np.array([(s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0])
            half = np.array([s[2] - s[0], s[3] - s[1]]) * 0.02
            out.append([*(mid - half), *(mid + half)])
        stubs[axis] = np.array(out)
    weak = analyse_axes(stubs, im.width, im.height, sigma_px=1.5)

    assert weak.best is not None
    assert weak.best.relative_sigma > 10 * good.best.relative_sigma, (
        good.best.relative_sigma, weak.best.relative_sigma)
    assert weak.best.verdict in ("weak", "poor", "unusable"), weak.best.verdict
    assert good.best.verdict == "good", good.best.verdict


def test_reports_what_is_unsolvable():
    """An image with no tie points should be named, not silently dropped."""
    project, _ = build_project()
    lonely = project.images[-1]
    lonely.observations.clear()

    report = solve(project)
    assert report.ok, report.messages
    assert report.n_images_positioned == 5
    notes = report.images[lonely.id].notes
    assert any("cannot place this camera" in n for n in notes), notes


def test_project_round_trip(tmp_path=None):
    import tempfile

    project, _ = build_project()
    solve(project)
    out = Path(tmp_path or tempfile.mkdtemp()) / "rt.multispy"
    project.save(out)
    reloaded = Project.load(out)

    assert len(reloaded.images) == len(project.images)
    assert reloaded.groups["cam0"].f is not None
    for a, b in zip(project.images, reloaded.images):
        assert len(a.axis_lines) == len(b.axis_lines)
        assert a.observations.keys() == b.observations.keys()
        assert (a.R is None) == (b.R is None)
        if a.R is not None:
            assert np.allclose(a.R, b.R)
            assert np.allclose(a.t, b.t)


def test_exports_run():
    import tempfile

    from multispy import export

    project, _ = build_project()
    report = solve(project)
    assert report.ok
    out = Path(tempfile.mkdtemp())
    project.path = out / "p.multispy"

    export.export_colmap(project, out / "sparse")
    assert (out / "sparse" / "cameras.txt").exists()
    assert (out / "sparse" / "images.txt").read_text().count("\n") > 6
    export.export_transforms_json(project, out / "transforms.json")
    export.export_blender(project, out / "import_cameras.py")
    export.export_ply(project, out / "tiepoints.ply")
    assert (out / "tiepoints.ply").read_text().startswith("ply")


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_solve_recovers_ground_truth,
        test_distance_constraint_sets_metric_scale,
        test_joint_solve_beats_independent_single_image_solves,
        test_focal_observability_brackets_the_real_error,
        test_observability_flags_weak_control_lines,
        test_reports_what_is_unsolvable,
        test_project_round_trip,
        test_exports_run,
    ]
    failures = 0
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")

    print("\n--- detail ---")
    project, truth = build_project()
    report = solve(project)
    print(f"stage: {report.stage}  ok: {report.ok}")
    for m in report.messages:
        print(f"  note: {m}")
    for w in report.warnings:
        print(f"  warn: {w}")
    print(f"focal: solved {report.focals.get('cam0', float('nan')):.1f} px, true {TRUE_FOCAL:.1f} px")
    print(f"tie points triangulated: {report.n_points_triangulated}, observations: {report.n_observations}")
    print(f"reprojection RMS: {report.rms_reprojection_px:.3f} px   control-line RMS: {report.rms_line_px:.3f} px")
    print(f"linear init conditioning: {report.init_condition:.1f}")
    print(f"{'image':8} {'obs':>4} {'reproj rms':>11} {'max':>7} {'line rms':>9} {'rot err':>9}")
    for im, R_true in zip(project.images, truth["R"]):
        r = report.images[im.id]
        rot = rotation_error_deg(im.R, R_true) if im.R is not None else float("nan")
        print(f"{im.id:8} {r.n_observations:4d} {r.rms_px:10.3f}px {r.max_px:6.2f} {r.line_rms_px:8.3f}px {rot:8.3f}deg")

    placed = [i for i, im in enumerate(project.images) if im.center() is not None]
    centers = np.array([project.images[i].center() for i in placed])
    truth_centers = truth["centers"][placed]
    s, R, t = umeyama(centers, truth_centers)
    aligned = (s * (R @ centers.T).T) + t
    err = np.linalg.norm(aligned - truth_centers, axis=1)
    print(f"camera position error after similarity alignment: "
          f"mean {err.mean():.4f} m, max {err.max():.4f} m (box diagonal {np.linalg.norm(BOX):.2f} m)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
