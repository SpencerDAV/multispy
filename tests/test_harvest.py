"""The tie-point harvester, against a synthetic scene whose answer is known.

Renders a textured corner-of-a-room from several viewpoints by ray casting, so
the images are geometrically exact and every pixel's world point can be looked
up afterwards.  That lookup is what makes the interesting test possible: a
harvested track is *correct* when every one of its marks ray-casts to the same
physical point, and wrong when they do not, which can be checked outright
rather than inferred from a residual.

Two of the walls are deliberately given the *same* texture.  That is the whole
problem this module exists for, in miniature.  The rear-module cell it was
written for contains twelve identical copies of one sensor body and five of
another, and any matcher working from appearance alone has twelve equally good
answers and no way to choose.  Here, the two walls are pixel-for-pixel
identical and in completely different places, so a track that mixes them is
detectable and must not survive.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_harvest.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.harvest import (  # noqa: E402
    HarvestSettings,
    add_tracks_to_project,
    cameras_of,
    detect_corners,
    harvest,
    remove_harvested,
    scene_points,
)
from multispy.project import AxisLine, CameraGroup, ImageView, Project, TiePoint  # noqa: E402

WIDTH, HEIGHT = 1000, 750
FOCAL = 950.0
L = 4.0          # room half-extent, metres
H = 3.0          # wall height
TEX_CELL = 0.05  # world metres per texture cell


# --------------------------------------------------------------------------
# a synthetic room, rendered by ray casting
# --------------------------------------------------------------------------


def _texture(seed: int, n: int = 420) -> np.ndarray:
    """Blobby greyscale with corners at every scale, from upsampled noise.

    Bilinear upsampling of white noise is a cheap way to get a field that is
    smooth enough to interpolate and interesting enough to corner-detect, which
    pure noise is not (every pixel is a corner, so none of them is).
    """
    rng = np.random.default_rng(seed)
    coarse = ndimage.zoom(rng.random((n // 20, n // 20)), 20.0, order=1)
    fine = ndimage.zoom(rng.random((n // 6, n // 6)), 6.0, order=1)
    out = 0.65 * coarse[:n, :n] + 0.35 * fine[:n, :n]
    return out.astype(np.float32)


class Room:
    """Three mutually perpendicular textured planes, and a ray caster for them.

    ``floor`` and ``wall_x`` carry different textures; ``wall_y`` repeats
    ``wall_x``'s exactly, which is the duplicated body this is all about.
    """

    def __init__(self) -> None:
        tex_floor = _texture(11)
        tex_wall = _texture(22)
        # (origin, axis_a, axis_b, extent_a, extent_b, texture)
        self.planes = [
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
             L, L, tex_floor),
            (np.array([0.0, 0.0, 0.0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
             L, H, tex_wall),
            (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
             L, H, tex_wall),  # the identical copy
        ]

    def cast(self, origin: np.ndarray, dirs: np.ndarray):
        """Nearest hit of each ray.  Returns (hit_mask, plane_index, point, grey)."""
        n_rays = len(dirs)
        best_t = np.full(n_rays, np.inf)
        best_plane = np.full(n_rays, -1, dtype=int)
        best_grey = np.zeros(n_rays, dtype=np.float32)
        for k, (p0, ea, eb, la, lb, tex) in enumerate(self.planes):
            nrm = np.cross(ea, eb)
            denom = dirs @ nrm
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((p0 - origin) @ nrm) / denom
            ok = np.isfinite(t) & (t > 1e-6) & (t < best_t)
            if not ok.any():
                continue
            P = origin + t[ok, None] * dirs[ok]
            a = (P - p0) @ ea
            b = (P - p0) @ eb
            inside = (a >= 0) & (a < la) & (b >= 0) & (b < lb)
            idx = np.nonzero(ok)[0][inside]
            if not len(idx):
                continue
            ta, tb = a[inside] / TEX_CELL, b[inside] / TEX_CELL
            h, w = tex.shape
            grey = ndimage.map_coordinates(
                tex, np.stack([np.clip(tb, 0, h - 1.001), np.clip(ta, 0, w - 1.001)]),
                order=1, mode="nearest",
            )
            best_t[idx] = t[ok][inside]
            best_plane[idx] = k
            best_grey[idx] = grey
        hit = best_plane >= 0
        point = np.zeros((n_rays, 3))
        point[hit] = origin + best_t[hit, None] * dirs[hit]
        return hit, best_plane, point, best_grey

    def render(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        C = -R.T @ t
        u, v = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
        xn = (u.ravel() - WIDTH / 2.0) / FOCAL
        yn = (v.ravel() - HEIGHT / 2.0) / FOCAL
        dirs = np.stack([xn, yn, np.ones_like(xn)], axis=1) @ R
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        hit, _plane, _point, grey = self.cast(C, dirs)
        img = np.where(hit, grey, 0.08).reshape(HEIGHT, WIDTH)
        return img.astype(np.float32)

    def lookup(self, R: np.ndarray, t: np.ndarray, uv: np.ndarray):
        """World point a single image mark actually lies on.  The ground truth."""
        C = -R.T @ t
        xn = (uv[0] - WIDTH / 2.0) / FOCAL
        yn = (uv[1] - HEIGHT / 2.0) / FOCAL
        d = np.array([[xn, yn, 1.0]]) @ R
        d /= np.linalg.norm(d)
        hit, plane, point, _g = self.cast(C, d)
        return bool(hit[0]), int(plane[0]), point[0]


def look_at(center: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0])):
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.array([x, y, z])
    return R, -R @ center


VIEWS = [
    np.array([5.2, 4.6, 2.6]),
    np.array([6.4, 2.4, 2.2]),
    np.array([3.0, 6.2, 2.8]),
    np.array([5.6, 5.4, 1.3]),
    np.array([4.0, 4.0, 4.6]),
]
TARGET = np.array([1.5, 1.5, 0.8])


def build_scene(*, with_lines: bool = False):
    """A project with solved poses, the rendered images, and the room itself."""
    room = Room()
    project = Project(name="harvest-synthetic")
    project.groups["cam0"] = CameraGroup(
        id="cam0", f=FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0, k1=0.0, lock_focal=True
    )
    images: dict[str, np.ndarray] = {}
    for i, c in enumerate(VIEWS):
        R, t = look_at(c, TARGET)
        im = ImageView(id=f"img{i}", path=f"synthetic_{i}.png",
                       width=WIDTH, height=HEIGHT, group="cam0")
        im.R, im.t = R, t
        if with_lines:
            for axis, a, b in ((0, [0, 0, 0], [L, 0, 0]), (0, [0, L, 0], [L, L, 0]),
                               (1, [0, 0, 0], [0, L, 0]), (1, [L, 0, 0], [L, L, 0]),
                               (2, [0, 0, 0], [0, 0, H]), (2, [L, 0, 0], [L, 0, H])):
                p0 = _project_point(R, t, np.array(a, float))
                p1 = _project_point(R, t, np.array(b, float))
                im.axis_lines.append(AxisLine(axis=axis, p0=list(p0), p1=list(p1)))
        project.images.append(im)
        images[im.id] = room.render(R, t)

    # a handful of stated world points, so the depth gate knows the scene's size
    for k, xyz in enumerate([[0, 0, 0], [L, 0, 0], [0, L, 0], [L, L, 0],
                             [0, 0, H], [L, 0, H], [0, L, H]]):
        pt = TiePoint(id=f"p{k}", xyz=np.array(xyz, dtype=float))
        project.points.append(pt)
        for im in project.images:
            uv = _project_point(im.R, im.t, pt.xyz)
            if 0 <= uv[0] < WIDTH and 0 <= uv[1] < HEIGHT:
                im.observations[pt.id] = [float(uv[0]), float(uv[1])]
    return project, images, room


def _project_point(R, t, X):
    Xc = R @ X + t
    return np.array([FOCAL * Xc[0] / Xc[2] + WIDTH / 2.0,
                     FOCAL * Xc[1] / Xc[2] + HEIGHT / 2.0])


def fast_settings(**kw) -> HarvestSettings:
    """Tighter than the shipped defaults, and deliberately so.

    Those defaults are sized for rearmod3, whose poses carry several pixels of
    real error -- the epipolar band is 8 px because the project's own
    hand-marked points scatter that far.  Here the poses are exact to machine
    precision, so the same slack would only admit coincidences and make the
    correctness test weaker than it looks.  ``ncc_min`` and ``uniqueness`` are
    left at their real values: those are the gates under test.
    """
    base = dict(max_corners_per_image=1500, corner_grid=16, patch=9,
                epipolar_px=2.0, extend_px=6.0, max_track_reproj_px=2.0,
                max_points=400, max_per_cell=8, min_triangulation_deg=5.0)
    base.update(kw)
    return HarvestSettings(**base)


def world_spread(tracks, project, room):
    """How far apart, in metres, the marks on each track really are.

    Each mark is ray-cast against the ground-truth room and the spread of the
    resulting world points is reported.  Nothing here consults the track's own
    triangulation, which is the point: a track that mixes the two identical
    walls triangulates to a perfectly plausible residual and is still two
    different places.

    Judged on distance alone and not on which plane was hit.  The planes meet
    along shared edges, and a ray striking a seam is assigned to one or other
    of them essentially at random -- a track on the corner line is then
    reported as crossing planes while every one of its marks is within a
    centimetre of the same point.  That is the classifier being wrong, not the
    harvest, and it cost an afternoon to establish, so: distance.
    """
    out = []
    for tr in tracks:
        pts = []
        for iid, uv in tr.obs.items():
            im = project.image(iid)
            ok, _plane, P = room.lookup(im.R, im.t, uv)
            if not ok:
                pts = []
                break
            pts.append(P)
        if not pts:
            out.append(float("inf"))
            continue
        a = np.array(pts)
        out.append(float(np.max(np.linalg.norm(a - a.mean(axis=0), axis=1))))
    return np.array(out)


def plane_of(track, project, room) -> int:
    """Which surface a track sits on, by majority of its marks."""
    votes = []
    for iid, uv in track.obs.items():
        im = project.image(iid)
        ok, plane, _P = room.lookup(im.R, im.t, uv)
        if ok:
            votes.append(plane)
    return max(set(votes), key=votes.count) if votes else -1


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_corners_spread_over_the_frame():
    """Detection must not spend its whole budget on the most textured region."""
    _project, images, _room = build_scene()
    img = images["img0"]
    s = fast_settings()
    uv, resp = detect_corners(img, s, full_width=WIDTH, full_height=HEIGHT)
    assert len(uv) >= 250, f"only {len(uv)} corners on a fully textured frame"
    assert len(uv) <= s.max_corners_per_image, "the cap was not applied"
    assert np.all(uv[:, 0] >= 0) and np.all(uv[:, 0] < WIDTH), "corner outside the frame"
    assert np.all(uv[:, 1] >= 0) and np.all(uv[:, 1] < HEIGHT), "corner outside the frame"
    assert np.all(resp > 0), "a corner was kept with a non-positive response"

    cells = 8
    cu = (uv[:, 0] / (WIDTH / cells)).astype(int)
    cv = (uv[:, 1] / (HEIGHT / cells)).astype(int)
    occupied = len(set(zip(cu.tolist(), cv.tolist())))
    assert occupied >= 48, f"corners occupy only {occupied} of 64 cells"

    # sub-pixel: positions must not all be integers
    frac = np.abs(uv - np.round(uv))
    assert float(frac.max()) > 0.05, "no sub-pixel refinement happened at all"


def test_harvest_finds_tracks_that_are_actually_the_same_point():
    """The headline claim: tracks are correct, including across identical walls."""
    project, images, room = build_scene()
    report = harvest(project, fast_settings(), images=images)
    assert report.n_kept >= 40, f"only {report.n_kept} tracks on a richly textured scene"
    assert all(tr.n_views >= 3 for tr in report.tracks), "a track below min_views survived"

    spread = world_spread(report.tracks, project, room)
    # A mark swapped onto the identical copy of a wall is of the order of a
    # metre away from the right one.  A mark that is simply a pixel or two off
    # is about a centimetre away, since the cameras stand several metres back.
    # The gap between those two numbers is two orders of magnitude, so this
    # threshold is not a tuned one and does not need to be.
    wrong = int((spread > 0.25).sum())
    assert wrong == 0, (
        f"{wrong} of {len(report.tracks)} tracks mix marks on different physical "
        f"points, the worst by {float(spread.max()):.2f} m"
    )
    assert float(np.median(spread)) < 0.02, (
        f"tracks are internally loose: median mark spread {float(np.median(spread)):.3f} m"
    )

    # and the geometry it reports about them is honest
    worst = max(tr.reproj_px for tr in report.tracks)
    assert worst <= 2.0, f"a kept track reprojects {worst:.1f} px from its own marks"
    assert min(tr.angle_deg for tr in report.tracks) >= 5.0, "a sliver track survived"


def test_both_identical_walls_are_harvested():
    """Not solved by refusing to look at the ambiguous thing.

    A matcher could score perfectly on the test above by rejecting every
    correspondence on the duplicated walls.  That is not the behaviour wanted:
    the point of knowing the poses is to use the repeated surfaces, not to
    avoid them.
    """
    project, images, room = build_scene()
    report = harvest(project, fast_settings(), images=images)
    spread = world_spread(report.tracks, project, room)
    planes = [plane_of(tr, project, room)
              for tr, sp in zip(report.tracks, spread) if sp <= 0.25]
    counts = {k: planes.count(k) for k in (0, 1, 2)}
    assert counts[1] >= 5 and counts[2] >= 5, (
        f"the duplicated walls were avoided rather than resolved: {counts}"
    )


def test_a_wrong_pose_finds_nothing():
    """Geometry is doing the work, not appearance.

    If a camera's pose is wrong, every epipolar line through it is wrong, and
    the harvest should lose that camera rather than quietly match on looks.  A
    matcher that kept its yield here would be one whose filters are decorative.
    """
    project, images, _room = build_scene()
    s = fast_settings()
    before = harvest(project, s, images=images)
    n_before = sum(1 for tr in before.tracks if "img1" in tr.obs)
    assert n_before >= 5, "the control case found nothing, so this proves nothing"

    im = project.image("img1")
    axis = np.array([0.03, -0.05, 0.02])  # a few degrees of pose error
    theta = float(np.linalg.norm(axis))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    im.R = (np.eye(3) + math.sin(theta) / theta * K
            + (1 - math.cos(theta)) / theta**2 * (K @ K)) @ im.R

    after = harvest(project, s, images=images)
    n_after = sum(1 for tr in after.tracks if "img1" in tr.obs)
    assert n_after <= 0.25 * n_before, (
        f"a mis-posed camera still contributed {n_after} tracks against {n_before} "
        "correct ones -- the geometric gate is not what is selecting the matches"
    )


def test_unsolved_project_is_refused_rather_than_guessed_at():
    project, images, _room = build_scene()
    for im in project.images:
        im.R = im.t = None
    report = harvest(project, fast_settings(), images=images)
    assert not report.tracks, "matched against poses that do not exist"
    assert report.messages, "refused silently, with nothing said about why"


def test_adding_and_removing_is_a_round_trip():
    project, images, _room = build_scene()
    report = harvest(project, fast_settings(), images=images)
    assert report.tracks

    n_points = len(project.points)
    n_obs = {im.id: len(im.observations) for im in project.images}

    added = add_tracks_to_project(project, report.tracks)
    assert len(added) == len(report.tracks)
    assert len(project.points) == n_points + len(added)
    assert len(set(p.id for p in project.points)) == len(project.points), "duplicate point id"
    for im in project.images:
        assert len(im.observations) >= n_obs[im.id]
    per_point = {pid: sum(1 for im in project.images if pid in im.observations)
                 for pid in added}
    assert min(per_point.values()) >= 3, "a harvested point reached the project under-observed"

    # and it survives being written out and read back
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "round-trip.multispy"
        project.save(path)
        reloaded = Project.load(str(path))
    assert len(reloaded.points) == len(project.points)
    assert sum(len(i.observations) for i in reloaded.images) == \
        sum(len(i.observations) for i in project.images)

    removed = remove_harvested(project)
    assert removed == len(added), f"removed {removed} of {len(added)}"
    assert len(project.points) == n_points
    assert {im.id: len(im.observations) for im in project.images} == n_obs


def test_harvested_points_survive_a_solve_and_hold_the_focal():
    """The output has to be usable evidence, not just plausible numbers.

    Harvested points go in as ordinary tie points, so the real test is that an
    ordinary solve accepts them: most of them should come through the outlier
    cull alive, and the focal length recovered from a scene held mostly by
    harvested correspondence should be the one the images were rendered with.
    """
    from multispy.solve import solve

    project, images, _room = build_scene(with_lines=True)
    report = harvest(project, fast_settings(), images=images)
    assert len(report.tracks) >= 30, f"only {len(report.tracks)} tracks to solve with"
    project.groups["cam0"].lock_focal = False
    project.groups["cam0"].f = None  # make the control lines measure it

    added = add_tracks_to_project(project, report.tracks)
    rep = solve(project, run_bundle=True)
    assert rep.ok, f"the solve failed outright: {rep.messages}"

    f = rep.focals.get("cam0", float("nan"))
    assert abs(f - FOCAL) / FOCAL < 0.02, (
        f"focal came back {f:.0f} px against a rendered {FOCAL:.0f} px"
    )

    alive = sum(1 for pid in added
                if sum(1 for im in project.images if pid in im.observations) >= 2)
    assert alive >= 0.8 * len(added), (
        f"only {alive} of {len(added)} harvested points were still carried after the "
        "solve; the bundle is rejecting most of what the harvest produced"
    )
    # Where the bundle put them, compared with where the harvest said -- but
    # only up to a similarity.  This project states no distance and no nominal,
    # so its origin, orientation and scale are gauge: the solve normalises to a
    # working scale of its own and the two point clouds are not in the same
    # units at all.  Comparing them raw measures the gauge, which is nothing.
    from multispy.geometry import similarity_fit

    pairs = [(np.asarray(project.point(pid).xyz, float), tr.xyz)
             for pid, tr in zip(added, report.tracks)
             if project.point(pid) is not None and project.point(pid).xyz is not None]
    fit = similarity_fit(np.array([a for a, _ in pairs]),
                         np.array([b for _, b in pairs]), allow_scale=True)
    assert fit.rms < 0.05 * L, (
        f"the bundle moved the harvested points {fit.rms:.3f} m relative to each "
        f"other, in a room {L:.0f} m across -- they are not a consistent cloud"
    )


def test_scene_points_sees_bodies_as_well_as_tie_points():
    project, _images, _room = build_scene()
    pts = scene_points(project)
    assert len(pts) == len(project.points), "stated tie points were not all counted"
    assert len(cameras_of(project)) == len(VIEWS)


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_corners_spread_over_the_frame,
        test_scene_points_sees_bodies_as_well_as_tie_points,
        test_unsolved_project_is_refused_rather_than_guessed_at,
        test_harvest_finds_tracks_that_are_actually_the_same_point,
        test_both_identical_walls_are_harvested,
        test_a_wrong_pose_finds_nothing,
        test_adding_and_removing_is_a_round_trip,
        test_harvested_points_survive_a_solve_and_hold_the_focal,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok    {fn.__name__}")

    print("\n--- detail ---")
    project, images, room = build_scene()
    report = harvest(project, fast_settings(), images=images)
    print(report.summary())
    spread = world_spread(report.tracks, project, room)
    print(f"tracks whose marks are the same physical point: "
          f"{int((spread <= 0.25).sum())} of {len(report.tracks)}; "
          f"mark spread median {float(np.median(spread)):.4f} m, "
          f"max {float(spread.max()):.4f} m")
    if report.tracks:
        views: dict[int, int] = {}
        for tr in report.tracks:
            views[tr.n_views] = views.get(tr.n_views, 0) + 1
        print("by view count: " + ", ".join(f"{k}:{v}" for k, v in sorted(views.items())))
        print(f"worst triangulation residual: "
              f"{max(tr.reproj_px for tr in report.tracks):.2f} px")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
