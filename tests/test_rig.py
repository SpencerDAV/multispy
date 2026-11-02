"""Validation of the known-object workflow, on the case it was built for:
recovering the layout of a camera array from a handheld pass around it.

The scene is a rig -- an axis-aligned frame with five identical camera bodies
bolted to its top, each toed in by a different angle. A handheld camera walks
around it taking six photos. What a user would actually mark is generated and
nothing else:

* control lines along the *frame's* edges, which are the only world-parallel
  edges in the scene (a toed-in camera body has none, by definition);
* tie points on the frame's corners;
* anchors on each camera body's corners, which is what places the bodies.

The assertions lean on quantities that do not depend on where the world origin
happened to land -- baselines between the cameras and the angles between their
optical axes -- because those are what a rig specification is written in, and
because they are the honest thing to check when the solve is only determined up
to a similarity. The locked-object test then checks the case where that freedom
is removed and absolute poses become meaningful.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_rig.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.geometry import Intrinsics, project as project_points, similarity_fit
from multispy.project import (
    AxisLine,
    AxisLockConstraint,
    CameraGroup,
    DistanceConstraint,
    ImageView,
    ObjectAnchor,
    ObjectInstance,
    ObjectTemplate,
    OpticalFrame,
    Project,
    TiePoint,
    anchor_key,
)
from multispy.solve import constraints_imply_scale, solve

WIDTH, HEIGHT = 4000, 3000
TRUE_FOCAL = 3200.0

FRAME = np.array([1.20, 0.40, 0.30])
"""The rig's mounting frame, axis-aligned at the origin, in metres. Its edges are
what the control lines are drawn along."""

BODY = np.array([100.0, 60.0, 40.0])
"""One camera body, in millimetres -- the units a CAD export would carry."""
UNIT_SCALE = 0.001

N_CAMERAS = 5
SPACING = 0.225
TOE_IN_DEG = 6.0
PITCH_DEG = -4.0

# Body-local axes follow the camera convention: +x right, +y down, +z along the
# optical axis, with the local origin at the entrance pupil on the front face.
# So the box spans z in [-depth, 0] and the optical frame is the origin itself.
BODY_CORNERS = np.array(
    [
        [sx * BODY[0] / 2.0, sy * BODY[1] / 2.0, sz * BODY[2]]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 0.0)
    ]
)
CORNER_NAMES = [
    f"c{'r' if sx > 0 else 'l'}{'d' if sy > 0 else 'u'}{'f' if sz == 0 else 'b'}"
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
    for sz in (-1.0, 0.0)
]


# --------------------------------------------------------------------------
# the rig
# --------------------------------------------------------------------------


def look_at(center: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0])):
    """World-to-camera rotation for an x-right, y-down, z-forward camera."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.array([x, y, z])
    return R, -R @ center


def body_pose(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth object-to-world pose of one camera in the array.

    The cameras sit in a row along the frame's top and converge on a point in
    front of it, which is what a real array for depth or light-field capture
    looks like and, more to the point, gives every body a different orientation
    so a solve that merely got the positions right cannot pass.
    """
    yaw = math.radians((index - (N_CAMERAS - 1) / 2.0) * -TOE_IN_DEG)
    pitch = math.radians(PITCH_DEG)
    forward = np.array(
        [math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)]
    )
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])

    x = (index - (N_CAMERAS - 1) / 2.0) * SPACING
    t = np.array([FRAME[0] / 2.0 + x, FRAME[1], FRAME[2] + BODY[1] / 2.0 * UNIT_SCALE])
    return R, t


def frame_points() -> tuple[np.ndarray, np.ndarray]:
    """Corners of the mounting frame, each with the three outward face normals
    that meet there."""
    pts, normals = [], []
    for sx in (0.0, 1.0):
        for sy in (0.0, 1.0):
            for sz in (0.0, 1.0):
                pts.append(np.array([sx, sy, sz]) * FRAME)
                normals.append(
                    np.array(
                        [
                            [sx * 2 - 1, 0.0, 0.0],
                            [0.0, sy * 2 - 1, 0.0],
                            [0.0, 0.0, sz * 2 - 1],
                        ]
                    )
                )
    return np.array(pts), np.array(normals)


def corner_visibility(
    corners: np.ndarray, normals: np.ndarray, center: np.ndarray, min_faces: int
) -> np.ndarray:
    """Which corners of a convex box a person could pick out from ``center``.

    A corner of a box is visible as soon as *one* of the three faces meeting at
    it is turned toward the camera -- requiring all three would hide the corners
    of the very face you are looking at. ``min_faces=2`` is the stricter reading:
    the corner is not merely visible but sits inside the silhouette, where it
    reads as a crisp corner rather than a point on an edge, which is what makes
    it worth clicking.
    """
    to_camera = center[None, :] - corners
    facing = np.einsum("nfi,ni->nf", normals, to_camera) > 0.0
    return facing.sum(axis=1) >= min_faces


def viewpoints() -> list[tuple[np.ndarray, np.ndarray]]:
    """Six handheld photos arcing around the front of the rig."""
    target = np.array([FRAME[0] / 2.0, FRAME[1] / 2.0, FRAME[2]])
    poses = []
    for angle_deg, dist, height in (
        (-58.0, 1.45, 0.55),
        (-33.0, 1.30, 0.80),
        (-10.0, 1.25, 0.42),
        (14.0, 1.35, 0.85),
        (37.0, 1.30, 0.48),
        (60.0, 1.50, 0.72),
    ):
        a = math.radians(angle_deg)
        center = target + np.array([math.sin(a) * dist, -math.cos(a) * dist, 0.0])
        center[2] = height
        poses.append(look_at(center, target))
    return poses


def build_rig_project(
    seed: int = 11,
    anchor_noise_px: float = 1.0,
    tie_noise_px: float = 1.0,
    line_noise_px: float = 1.5,
    with_optical_frame: bool = True,
) -> tuple[Project, dict]:
    """Generate the rig project with only marks a person could plausibly place."""
    rng = np.random.default_rng(seed)
    cam_true = Intrinsics(f=TRUE_FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)
    frame_pts, frame_normals = frame_points()
    poses = viewpoints()

    project = Project(name="camera-array")
    project.groups["cam0"] = CameraGroup(id="cam0")

    template = ObjectTemplate(
        id="tpl0",
        name="array camera body",
        units="mm",
        unit_scale=UNIT_SCALE,
        anchors=[
            ObjectAnchor(name=name, xyz=corner.tolist())
            for name, corner in zip(CORNER_NAMES, BODY_CORNERS)
        ],
        optical=OpticalFrame(center=[0.0, 0.0, 0.0], forward=[0.0, 0.0, 1.0], up=[0.0, -1.0, 0.0])
        if with_optical_frame
        else None,
    )
    project.templates["tpl0"] = template

    truth: dict = {"bodies": [], "focal": TRUE_FOCAL, "R": [], "t": []}
    for i in range(N_CAMERAS):
        project.instances.append(
            ObjectInstance(id=f"obj{i}", template="tpl0", label=f"cam{i}")
        )
        R_obj, t_obj = body_pose(i)
        truth["bodies"].append((R_obj, t_obj))

    for i, p in enumerate(frame_pts):
        project.points.append(TiePoint(id=f"p{i}", label=f"frame {i}"))

    # the six body faces, as corner-index triples sharing each corner, so a
    # corner counts as markable only when the body is actually turned toward
    # the camera there
    corner_face_normals = np.array(
        [
            [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, -1.0 if sz < 0 else 1.0]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 0.0)
        ]
    )

    for ci, (R, t) in enumerate(poses):
        center = -R.T @ t
        image = ImageView(
            id=f"img{ci}",
            path=f"photos/rig_{ci:02d}.jpg",
            width=WIDTH,
            height=HEIGHT,
            group="cam0",
        )
        truth["R"].append(R)
        truth["t"].append(t)

        def inside(uv: np.ndarray) -> np.ndarray:
            return (
                np.isfinite(uv[:, 0])
                & (uv[:, 0] > 25)
                & (uv[:, 0] < WIDTH - 25)
                & (uv[:, 1] > 25)
                & (uv[:, 1] < HEIGHT - 25)
            )

        # ---- tie points on the frame corners
        uv_frame = project_points(R, t, cam_true, frame_pts)
        visible_frame = inside(uv_frame) & corner_visibility(
            frame_pts, frame_normals, center, min_faces=1
        )
        for pi in np.flatnonzero(visible_frame):
            noisy = uv_frame[pi] + rng.normal(0.0, tie_noise_px, 2)
            image.observations[f"p{pi}"] = [float(noisy[0]), float(noisy[1])]

        # ---- control lines along the frame's axis-aligned edges
        vis_idx = np.flatnonzero(visible_frame)
        for axis in range(3):
            candidates = []
            for a_i in range(len(vis_idx)):
                for b_i in range(a_i + 1, len(vis_idx)):
                    a, b = vis_idx[a_i], vis_idx[b_i]
                    delta = frame_pts[b] - frame_pts[a]
                    if abs(delta[axis]) < 1e-6:
                        continue
                    if any(abs(delta[k]) > 1e-6 for k in range(3) if k != axis):
                        continue
                    lo, hi = (a, b) if delta[axis] > 0 else (b, a)
                    length = float(np.linalg.norm(uv_frame[hi] - uv_frame[lo]))
                    if length < 80.0:
                        continue
                    candidates.append((length, lo, hi))
            candidates.sort(reverse=True)
            for _length, lo, hi in candidates[:3]:
                p0 = uv_frame[lo] + rng.normal(0.0, line_noise_px, 2)
                p1 = uv_frame[hi] + rng.normal(0.0, line_noise_px, 2)
                image.axis_lines.append(
                    AxisLine(axis=axis, p0=[float(p0[0]), float(p0[1])], p1=[float(p1[0]), float(p1[1])])
                )

        # ---- anchors on each camera body
        for i in range(N_CAMERAS):
            R_obj, t_obj = truth["bodies"][i]
            world = (BODY_CORNERS * UNIT_SCALE) @ R_obj.T + t_obj
            world_normals = np.einsum("ij,nfj->nfi", R_obj, corner_face_normals)
            uv_body = project_points(R, t, cam_true, world)
            ok = inside(uv_body) & corner_visibility(world, world_normals, center, min_faces=2)
            for k, name in enumerate(CORNER_NAMES):
                if not ok[k]:
                    continue
                noisy = uv_body[k] + rng.normal(0.0, anchor_noise_px, 2)
                image.anchor_obs[anchor_key(f"obj{i}", name)] = [
                    float(noisy[0]),
                    float(noisy[1]),
                ]

        project.images.append(image)

    return project, truth


# --------------------------------------------------------------------------
# ground-truth comparisons
# --------------------------------------------------------------------------


def true_optical(truth: dict) -> tuple[np.ndarray, np.ndarray]:
    """Entrance-pupil positions and optical-axis directions, from the truth."""
    centers = np.array([t for _R, t in truth["bodies"]])
    forwards = np.array([R[:, 2] for R, _t in truth["bodies"]])
    return centers, forwards


def solved_optical(project: Project, report) -> tuple[np.ndarray, np.ndarray]:
    centers, forwards = [], []
    for inst in project.instances:
        rep = report.instances[inst.id]
        assert rep.placed, (inst.id, rep.notes)
        centers.append(rep.optical_center if rep.optical_center is not None else inst.t)
        forwards.append(rep.optical_forward if rep.optical_forward is not None else inst.R[:, 2])
    return np.array(centers, dtype=float), np.array(forwards, dtype=float)


def pairwise(values: np.ndarray) -> np.ndarray:
    n = len(values)
    return np.array(
        [np.linalg.norm(values[j] - values[i]) for i in range(n) for j in range(i + 1, n)]
    )


def pairwise_angles(forwards: np.ndarray) -> np.ndarray:
    n = len(forwards)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = float(np.clip(forwards[i] @ forwards[j], -1.0, 1.0))
            out.append(math.degrees(math.acos(cos)))
    return np.array(out)


def most_observed_anchors(project: Project, instance_id: str, count: int) -> list[str]:
    """The anchors of one body seen in the most photos, which is what a careful
    user would end up with and what makes a trimmed-down test deterministic."""
    tally: dict[str, int] = {}
    for im in project.images:
        for key in im.anchor_obs:
            if key.startswith(f"{instance_id}/"):
                tally[key] = tally.get(key, 0) + 1
    return [k for k, _n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:count]]


def shared_anchor_pair(project: Project, instance_id: str) -> tuple[str, str, int]:
    """Two anchors of one body that appear together in as many photos as
    possible, so swapping them is a mistake the solve genuinely has to notice."""
    keys = most_observed_anchors(project, instance_id, 99)
    best = (None, None, 0)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            together = sum(
                1 for im in project.images if keys[i] in im.anchor_obs and keys[j] in im.anchor_obs
            )
            if together > best[2]:
                best = (keys[i], keys[j], together)
    return best


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_known_objects_make_the_scale_metric():
    """Five known bodies and no distance constraint anywhere should still give
    real units -- that is the entire point of a known object."""
    project, _truth = build_rig_project()
    assert not project.constraints

    report = solve(project)
    assert report.ok, report.messages
    assert report.scale_is_metric, report.scale_source
    assert "object geometry" in report.scale_source, report.scale_source
    assert "mm" in report.scale_source, report.scale_source


def test_rig_layout_recovered():
    """Baselines and relative toe-in angles, which is what a rig spec states."""
    project, truth = build_rig_project()
    report = solve(project)
    assert report.ok, report.messages

    centers, forwards = solved_optical(project, report)
    true_centers, true_forwards = true_optical(truth)

    baseline_err = np.abs(pairwise(centers) - pairwise(true_centers))
    angle_err = np.abs(pairwise_angles(forwards) - pairwise_angles(true_forwards))

    assert baseline_err.max() < 0.004, (baseline_err.max(), pairwise(true_centers))

    # adjacent spacing is the headline number for an array
    adjacent = np.array([np.linalg.norm(centers[i + 1] - centers[i]) for i in range(N_CAMERAS - 1)])
    assert np.all(np.abs(adjacent - SPACING) < 0.004), adjacent

    # Orientation is the weak axis and deliberately held to a looser bound. A
    # body's rotation is read off a lever only as long as the body itself -- here
    # 100 mm, seen from 1.3 m -- so it inherits roughly (distance / body size)
    # times the angular error of a position. Positions land near 1 mm; angles
    # near a degree. Marking a bigger object, or shooting closer, is what moves
    # this number, not more photos of the same geometry.
    assert angle_err.max() < 2.0, (angle_err.max(), pairwise_angles(true_forwards))


def test_bodies_fit_their_model():
    """A body's anchors must actually form the shape the model claims."""
    project, _truth = build_rig_project()
    report = solve(project)
    assert report.ok, report.messages

    for inst in project.instances:
        rep = report.instances[inst.id]
        assert rep.placed, (inst.id, rep.notes)
        assert rep.n_anchors_used >= 3, (inst.id, rep.n_anchors_used)
        # the body is 100 mm across; a shape error of 3 mm would be 3%
        assert rep.shape_rms < 0.003, (inst.id, rep.shape_rms)
        assert abs(rep.implied_scale - 1.0) < 0.05, (inst.id, rep.implied_scale)
        assert rep.rms_px < 3.0, (inst.id, rep.rms_px)


def test_mismarked_anchor_shows_up_as_shape_error():
    """Marking a body corner on the wrong corner should be *reported*, not
    absorbed into a plausible-looking reconstruction.

    This is the diagnostic free tie points cannot offer: swapping two features
    in a point cloud just moves two points, but swapping two corners of a known
    body contradicts a shape that was fixed in advance.
    """
    project, _truth = build_rig_project()
    good = solve(project)
    assert good.ok, good.messages
    baseline_shape = good.instances["obj2"].shape_rms

    project, _truth = build_rig_project()
    # swap two anchors of one body wherever both were marked
    a, b, together = shared_anchor_pair(project, "obj2")
    assert together >= 2, together
    swaps = 0
    for im in project.images:
        if a in im.anchor_obs and b in im.anchor_obs:
            im.anchor_obs[a], im.anchor_obs[b] = im.anchor_obs[b], im.anchor_obs[a]
            swaps += 1
    assert swaps >= 2, swaps

    report = solve(project)
    assert report.ok, report.messages
    bad = report.instances["obj2"]
    assert bad.shape_rms > 10 * max(baseline_shape, 1e-6), (bad.shape_rms, baseline_shape)
    # and the other bodies should be untouched
    assert report.instances["obj0"].shape_rms < 0.003, report.instances["obj0"].shape_rms


def test_locked_object_defines_the_world_frame():
    """Locking one body removes all seven gauge freedoms, so absolute poses
    become meaningful and comparable to the truth directly."""
    project, truth = build_rig_project()
    R0, t0 = truth["bodies"][0]
    project.instances[0].R = R0.copy()
    project.instances[0].t = t0.copy()
    project.instances[0].locked = True

    report = solve(project)
    assert report.ok, report.messages
    assert "locked object" in report.scale_source, report.scale_source

    centers, forwards = solved_optical(project, report)
    true_centers, true_forwards = true_optical(truth)

    position_err = np.linalg.norm(centers - true_centers, axis=1)
    angle_err = np.array(
        [
            math.degrees(math.acos(float(np.clip(forwards[i] @ true_forwards[i], -1.0, 1.0))))
            for i in range(N_CAMERAS)
        ]
    )
    # Absolute error grows with distance from the locked body, because the whole
    # array hangs off that one body's orientation -- and a body's orientation is
    # the weakest thing in this geometry. 900 mm away, half a degree of frame
    # rotation is already 8 mm. Lock the best-determined body, and prefer a large
    # one if you have the choice.
    assert position_err.max() < 0.015, position_err
    assert angle_err.max() < 2.0, angle_err


def test_rig_report_lists_the_layout():
    project, _truth = build_rig_project()
    report = solve(project)
    assert report.ok, report.messages

    rig = report.rig
    assert rig.n_instances == N_CAMERAS
    assert len(rig.pairs) == N_CAMERAS * (N_CAMERAS - 1) // 2
    assert all("angle_deg" in p for p in rig.pairs), rig.pairs
    assert abs(rig.extent - SPACING * (N_CAMERAS - 1)) < 0.006, rig.extent
    # with an optical frame set, it must not warn about reporting shells
    assert not rig.notes, rig.notes


def test_without_optical_frame_it_says_so():
    """A template with no optical frame places the shell, and the report should
    admit that rather than implying camera poses."""
    project, _truth = build_rig_project(with_optical_frame=False)
    report = solve(project)
    assert report.ok, report.messages
    assert report.rig.n_instances == N_CAMERAS
    assert any("body placements, not camera poses" in n for n in report.rig.notes), report.rig.notes


def test_two_anchors_is_not_a_placement():
    """Two anchors leave the body free to spin about the line joining them, and
    that has to be said rather than silently returning one of the poses."""
    project, _truth = build_rig_project()
    keep = set(most_observed_anchors(project, "obj3", 2))
    assert len(keep) == 2, keep
    for im in project.images:
        for key in list(im.anchor_obs):
            if key.startswith("obj3/") and key not in keep:
                del im.anchor_obs[key]

    report = solve(project)
    assert report.ok, report.messages
    rep = report.instances["obj3"]
    assert rep.n_anchors_used == 2, rep.n_anchors_used
    assert any("collinear" in n or "spin" in n for n in rep.notes), rep.notes
    # the other four must still solve
    assert sum(1 for r in report.instances.values() if r.placed) >= 4


def test_constraints_report_asked_vs_achieved():
    """A stated figure is the one number the user asserts rather than clicks, so
    the solve has to say whether it was met."""
    project, _truth = build_rig_project()
    # frame corners are p0..p7 over (sx, sy, sz); p0 is the origin, p4 is 1.2 m
    # along X from it
    project.constraints.append(AxisLockConstraint(point="p0", axis=0, value=0.0))
    project.constraints.append(AxisLockConstraint(point="p0", axis=1, value=0.0))
    project.constraints.append(AxisLockConstraint(point="p0", axis=2, value=0.0))
    project.constraints.append(
        DistanceConstraint(point_a="p0", point_b="p4", value=float(FRAME[0]))
    )

    report = solve(project)
    assert report.ok, report.messages
    assert len(report.constraints) == 4, report.constraints

    for rep in report.constraints:
        assert rep.applied, (rep.description, rep.note)
        assert abs(rep.error) < 0.006, (rep.description, rep.target, rep.achieved)

    distance = next(c for c in report.constraints if c.type == "distance")
    assert abs(distance.achieved - FRAME[0]) < 0.006, distance.achieved
    assert abs(distance.error) == abs(distance.achieved - distance.target)

    # stated figures and object geometry are two independent measurements here,
    # and both are right, so they must agree
    assert report.datum_scale is not None
    assert abs(report.datum_scale - 1.0) < 0.01, report.datum_scale
    assert "cross-checked" in report.scale_source, report.scale_source


def test_constraint_naming_an_unsolved_point_is_reported_not_silent():
    project, _truth = build_rig_project()
    project.points.append(TiePoint(id="pX", label="never marked"))
    project.constraints.append(AxisLockConstraint(point="pX", axis=0, value=5.0))

    report = solve(project)
    assert report.ok, report.messages
    rep = next(c for c in report.constraints if c.points == ["pX"])
    assert not rep.applied
    assert "no effect" in rep.note, rep.note
    assert any("did nothing" in w for w in report.warnings), report.warnings


def test_units_error_shows_up_as_scale_disagreement():
    """The trap this check exists for: a model declared in the wrong units is
    out by a round factor while every residual still looks perfect.

    Stating one true distance is enough to catch it, but only because the
    comparison is made before the refinement has traded the two against each
    other -- afterwards the solve has split the difference and neither source
    reads as the culprit.
    """
    project, _truth = build_rig_project()
    # the model is in millimetres; declare it centimetres
    project.templates["tpl0"].units = "cm"
    project.templates["tpl0"].unit_scale = 0.01
    project.constraints.append(
        DistanceConstraint(point_a="p0", point_b="p4", value=float(FRAME[0]))
    )

    report = solve(project)
    assert report.ok, report.messages
    assert report.datum_scale is not None
    # objects claim the scene is 10x bigger than it is, so the true 1.2 m frame
    # measures 12 m and the stated figure wants a tenth of that
    assert 0.09 < report.datum_scale < 0.11, report.datum_scale
    assert any("disagree about scale" in w for w in report.warnings), report.warnings

    # and with the units right, the same project agrees
    project, _truth = build_rig_project()
    project.constraints.append(
        DistanceConstraint(point_a="p0", point_b="p4", value=float(FRAME[0]))
    )
    good = solve(project)
    assert good.ok, good.messages
    assert abs(good.datum_scale - 1.0) < 0.01, good.datum_scale
    assert not any("disagree about scale" in w for w in good.warnings), good.warnings


def test_axis_locks_alone_imply_a_scale():
    """Two datums locked to different coordinates on one axis state a distance,
    whether or not anyone thought of it that way."""
    project, _truth = build_rig_project()
    project.constraints.append(AxisLockConstraint(point="p0", axis=0, value=0.0))
    project.constraints.append(AxisLockConstraint(point="p4", axis=0, value=float(FRAME[0])))

    report = solve(project)
    assert report.ok, report.messages
    assert report.datum_scale is not None, "a pair of X locks states a separation"
    assert abs(report.datum_scale - 1.0) < 0.01, report.datum_scale

    # locking two points to the *same* coordinate states a plane, not a distance
    plane = build_rig_project()[0]
    plane.constraints.append(AxisLockConstraint(point="p0", axis=2, value=0.0))
    plane.constraints.append(AxisLockConstraint(point="p4", axis=2, value=0.0))
    assert not constraints_imply_scale(plane.constraints)


def test_round_trip_keeps_objects():
    import json
    import tempfile

    project, _truth = build_rig_project()
    report = solve(project)
    assert report.ok, report.messages

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig.multispy"
        project.save(path)
        again = Project.load(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

    assert set(again.templates) == set(project.templates)
    assert len(again.instances) == len(project.instances)
    template = again.templates["tpl0"]
    assert len(template.anchors) == len(CORNER_NAMES)
    assert template.units == "mm" and template.unit_scale == UNIT_SCALE
    assert template.optical is not None
    for before, after in zip(project.instances, again.instances):
        assert np.allclose(before.R, after.R), before.id
        assert np.allclose(before.t, after.t), before.id
    total = sum(len(im.anchor_obs) for im in again.images)
    assert total == sum(len(im.anchor_obs) for im in project.images), total

    # a version-1 project, from before objects existed, must still load
    raw["version"] = 1
    raw.pop("templates")
    raw.pop("instances")
    for im in raw["images"]:
        im.pop("anchor_obs")
    plain = Project.from_dict(raw)
    assert plain.templates == {} and plain.instances == []
    assert all(im.anchor_obs == {} for im in plain.images)


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_known_objects_make_the_scale_metric,
        test_rig_layout_recovered,
        test_bodies_fit_their_model,
        test_mismarked_anchor_shows_up_as_shape_error,
        test_locked_object_defines_the_world_frame,
        test_rig_report_lists_the_layout,
        test_without_optical_frame_it_says_so,
        test_two_anchors_is_not_a_placement,
        test_constraints_report_asked_vs_achieved,
        test_constraint_naming_an_unsolved_point_is_reported_not_silent,
        test_units_error_shows_up_as_scale_disagreement,
        test_axis_locks_alone_imply_a_scale,
        test_round_trip_keeps_objects,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")

    print("\n--- detail ---")
    project, truth = build_rig_project()
    report = solve(project)
    print(f"stage: {report.stage}  ok: {report.ok}")
    for m in report.messages:
        print(f"  note: {m}")
    for w in report.warnings:
        print(f"  warn: {w}")
    if report.ok:
        n_anchor = sum(len(im.anchor_obs) for im in project.images)
        print(f"focal: solved {report.focals.get('cam0', 0):.1f} px, true {TRUE_FOCAL:.1f} px")
        print(f"marks: {n_anchor} anchor observations, {report.n_observations} total")
        print(f"reprojection RMS: {report.rms_reprojection_px:.3f} px")
        print(f"scale: {report.scale_source}")
        centers, forwards = solved_optical(project, report)
        true_centers, true_forwards = true_optical(truth)
        print(f"\n{'body':<8}{'shape rms':>11}{'scale':>8}{'rms px':>9}{'anchors':>9}")
        for inst in project.instances:
            r = report.instances[inst.id]
            print(
                f"{r.label or r.id:<8}{r.shape_rms * 1000:9.2f}mm{r.implied_scale:8.4f}"
                f"{r.rms_px:9.2f}{r.n_anchors_used:5d}/{r.n_anchors_marked}"
            )
        print(f"\n{'pair':<14}{'solved':>10}{'true':>10}{'error':>10}")
        k = 0
        for i in range(N_CAMERAS):
            for j in range(i + 1, N_CAMERAS):
                d = np.linalg.norm(centers[j] - centers[i])
                dt = np.linalg.norm(true_centers[j] - true_centers[i])
                print(f"cam{i}-cam{j:<9}{d * 1000:8.1f}mm{dt * 1000:8.1f}mm{(d - dt) * 1000:8.2f}mm")
                k += 1
        ang = pairwise_angles(forwards)
        true_ang = pairwise_angles(true_forwards)
        print(
            f"\ntoe-in angle error: max {np.abs(ang - true_ang).max():.3f} deg "
            f"(true adjacent {true_ang[0]:.2f} deg)"
        )
        fit = similarity_fit(centers, true_centers)
        print(f"residual scale after alignment: {fit.scale:.5f}  (1.0 means units are right)")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
