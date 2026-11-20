"""Control lines pointing at a direction the solve works out for itself.

A direction group lets someone mark a cable tray in four photographs without
knowing which way the tray runs. The geometry is a small change to an existing
residual; the risk is entirely in the bookkeeping around it, because a free
parameter that looks like a constraint is worse than no constraint at all. These
tests are mostly about that.

What is being defended here:

* **A group recovers a direction nobody named.** Against a synthetic scene with
  a known non-axis direction, the solve has to come back pointing along it. If
  this fails the feature is decoration.

* **A group confined to one photograph is refused, not fitted.** This is the
  subtle one, and the obvious reading of it is wrong. Two lines in one photo
  *do* determine the direction -- that is a vanishing point -- so the failure is
  not a shortage of data. It is an exact invariance: every residual in a group
  enters through ``m = R @ d``, so turning the camera and counter-turning ``d``
  leaves all of them unchanged, and the group cannot prefer any rotation however
  many lines it holds. A fitted one-image group would therefore report a
  confident direction and a clean residual while contributing nothing, which is
  the one outcome worth building a refusal for.

* **A two-line group is reported as contributing nothing.** 2 unknowns per
  group, 1 residual per line, so two lines is break-even. The user's reason for
  wanting this feature ("there are lots of these, and an edge beats a point") is
  true per *line* and false per *group*, and a panel that showed only the line
  count would let someone add a dozen two-line groups and believe they helped.

* **The end you drew from makes no difference.** An axis line's drag direction
  is load-bearing -- it resolves the vanishing point's sign -- so carrying that
  habit over would be a reasonable thing to assume and a waste of care. Checked
  twice: algebraically, that reversing a segment only negates its residual, and
  end to end, that a project with every grouped line reversed solves to the same
  place.

* **An axis-only project is untouched.** Two ways, because this is where a
  generalised residual usually leaks. A regression lock on the solved numbers of
  the synthetic project, captured from the build before groups existed; and a
  refused group added to a copy of a project, which must leave every solved
  figure bit-for-bit identical rather than merely close.

* **A group cannot bootstrap a calibration, and the report says so.** Rotation
  and focal come from vanishing points, which need two axes whose directions are
  already known. A photo carrying nothing but grouped lines still will not
  calibrate, and someone who has just drawn six of them into it needs to be told
  that rather than left to wonder.

* **A changed group makes a measured origin read stale.** The origin radius is a
  reading about a particular set of marks. Edit a group and it describes data it
  was not measured on. The converse matters as much: an axis-only project's
  fingerprint must not move merely because the build gained a capability that
  project does not use.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_dirgroup.py
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy import dirgroup, uncertainty
from multispy.dirgroup import (
    MIN_PAIR_SINE,
    DirectionGroup,
    contribution_note,
    direction_at,
    initial_direction,
    plane_normal,
    tangent_basis,
)
from multispy.geometry import Intrinsics, project as project_points
from multispy.project import AxisLine, CameraGroup, ImageView, Project, TiePoint
from multispy.solve import _Problem, calibrate_all, solve

from test_synthetic import BOX, HEIGHT, TRUE_FOCAL, WIDTH, build_project

# A direction aligned with none of the world axes and not in any of their
# planes, so recovering it cannot be an accident of a degenerate case.
FREE_DIR = np.array([0.6, -0.45, 0.66])
FREE_DIR = FREE_DIR / np.linalg.norm(FREE_DIR)

_PERP_A = np.cross(FREE_DIR, np.array([0.0, 0.0, 1.0]))
_PERP_A = _PERP_A / np.linalg.norm(_PERP_A)
_PERP_B = np.cross(FREE_DIR, _PERP_A)

# Where a tray parallel to FREE_DIR might plausibly run, as offsets from the box
# centre perpendicular to it. Several, because not every one of them projects
# into every camera.
_BASES = [
    BOX / 2.0,
    BOX / 2.0 + 3.0 * _PERP_A,
    BOX / 2.0 - 3.0 * _PERP_A,
    BOX / 2.0 + 2.0 * _PERP_B,
    BOX / 2.0 - 2.0 * _PERP_B,
    BOX / 2.0 + 2.0 * _PERP_A + 2.0 * _PERP_B,
    BOX / 2.0 - 2.0 * _PERP_A - 2.0 * _PERP_B,
]

SEGMENT_LENGTH = 7.0
"""Metres of tray drawn per line. Long enough that the projected segment clears
the minimum below in every camera of the rig, which is what makes the fixture
deterministic rather than seed-dependent."""


# --------------------------------------------------------------------------
# fixture: a known non-axis direction marked across chosen photos
# --------------------------------------------------------------------------


def _true_camera(truth: dict, index: int) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
    return (
        truth["R"][index],
        truth["t"][index],
        Intrinsics(f=TRUE_FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0),
    )


def _segment(truth: dict, index: int, base: np.ndarray, rng, noise_px: float):
    """Pixel endpoints of a segment along ``FREE_DIR``, or ``None`` if unusable.

    Drawn from the *true* pose and the *true* focal, which is the point: the
    fixture is what a user could have clicked on a real photograph of this
    scene, and the solve has to get back to the direction without being told it.
    """
    R, t, cam = _true_camera(truth, index)
    ends = np.array([base - 0.5 * SEGMENT_LENGTH * FREE_DIR, base + 0.5 * SEGMENT_LENGTH * FREE_DIR])
    uv = project_points(R, t, cam, ends)
    if not np.all(np.isfinite(uv)):
        return None
    if uv[:, 0].min() < 30 or uv[:, 0].max() > WIDTH - 30:
        return None
    if uv[:, 1].min() < 30 or uv[:, 1].max() > HEIGHT - 30:
        return None
    if float(np.linalg.norm(uv[1] - uv[0])) < 150.0:
        return None
    return uv + rng.normal(0.0, noise_px, uv.shape)


def add_free_lines(
    project: Project,
    truth: dict,
    images: list[int],
    group_id: str = "d0",
    label: str = "tray 1",
    per_image: int = 1,
    seed: int = 11,
    noise_px: float = 1.5,
) -> Project:
    """Mark ``per_image`` lines along ``FREE_DIR`` in each named photo."""
    rng = np.random.default_rng(seed)
    if not any(g.id == group_id for g in project.direction_groups):
        project.direction_groups.append(DirectionGroup(id=group_id, label=label))
    for index in images:
        drawn = 0
        for base in _BASES:
            if drawn >= per_image:
                break
            uv = _segment(truth, index, base, rng, noise_px)
            if uv is None:
                continue
            project.images[index].axis_lines.append(
                AxisLine(
                    # the axis field is meaningless on a grouped line and is
                    # deliberately set to something wrong here, so a code path
                    # that quietly read it would fail loudly rather than work
                    axis=1,
                    p0=[float(uv[0][0]), float(uv[0][1])],
                    p1=[float(uv[1][0]), float(uv[1][1])],
                    direction_group=group_id,
                )
            )
            drawn += 1
        assert drawn == per_image, (
            f"fixture could not draw {per_image} line(s) in img{index}, only {drawn}"
        )
    return project


def group_report(report, group_id: str = "d0"):
    found = [g for g in report.direction_groups if g.id == group_id]
    assert found, f"no report for direction group {group_id!r} in {report.direction_groups}"
    return found[0]


def solved_digest(project: Project) -> str:
    """Every solved number in the project, as raw float bytes.

    Raw bytes rather than a rounded comparison, because the claim being defended
    is that an axis-only project is *untouched* -- not that it comes out close.
    """
    h = hashlib.sha1()
    for im in sorted(project.images, key=lambda i: i.id):
        h.update(im.id.encode())
        for arr in (im.R, im.t):
            h.update(b"-" if arr is None else np.asarray(arr, dtype=float).tobytes())
    for pt in sorted(project.points, key=lambda p: p.id):
        h.update(pt.id.encode())
        h.update(b"-" if pt.xyz is None else np.asarray(pt.xyz, dtype=float).tobytes())
    for gid, g in sorted(project.groups.items()):
        h.update(gid.encode())
        h.update(np.array([g.f or 0.0, g.cx or 0.0, g.cy or 0.0, g.k1], dtype=float).tobytes())
    return h.hexdigest()


AXIS_ONLY_DIGEST = "75fdfe76a927a9219a683e886a11992b172772ec"
"""``solved_digest`` of the unmodified synthetic project.

A regression lock, captured from the working tree immediately *before* direction
groups were added to ``project.py`` and ``solve.py``. Generalising the control
line residual is the kind of change that leaks into the axis path by a changed
residual offset or a reordered table, and "the tests still pass" does not catch a
solve that moved in the twelfth decimal. If this test fails, the axis-only path
is not what it was, whatever else looks fine.

Regenerate deliberately and never casually::

    .venv\\Scripts\\python.exe -c "import sys; sys.path[:0]=['.','tests']; \\
        from test_dirgroup import *; from test_synthetic import build_project; \\
        p,_=build_project(); solve(p); print(solved_digest(p))"

Regenerated once, from ``1db384d1``, on 2026-11-20, when the bundle stopped
writing a principal point it had never solved back into the camera group.
``solved_digest`` hashes ``g.cx or 0.0`` next to the poses and the points, so a
group that now keeps ``None`` where it used to be handed the image centre moves
the hash without anything geometric having happened.

That it was only the bookkeeping was checked rather than assumed: putting the
old write-back back -- setting each group's ``cx``/``cy`` to its images' centre
after the solve -- returns this digest to ``1db384d1`` exactly, so every pose,
every point and the focal length are bit-identical across the change. Do not
read that check as a reason to wave the next one through. It is the reason this
constant moved, and the only one.

**Run this file with no thread limits set.** The hash also moves with the BLAS
thread count, which is the cross-machine fragility above reproduced in a single
shell: on the machine this was pinned on, clean ``main`` hashed to ``1db384d1``
at the default thread count and to ``60c6fe32`` under ``OMP_NUM_THREADS``,
``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS`` of 1, 2, 4 or 8 alike. A
regeneration done with them set pins the constant to the wrong number, and a
test run with them set fails for a reason that is not in the code.
"""

AXIS_ONLY_FINGERPRINT = "8f93a3f84f34910e"
"""``uncertainty.fingerprint`` of ``small_project()`` with no groups anywhere.

Captured the same way and for the same reason. A stored origin radius must not
start reading ``(stale)`` on every existing project merely because the build
gained a capability those projects do not use."""


def small_project() -> Project:
    """A tiny hand-built project, for the tests that only need marks and lines."""
    project = Project(name="fingerprint fixture")
    project.groups["cam0"] = CameraGroup(id="cam0")
    for i in range(3):
        project.points.append(TiePoint(id=f"p{i}", label=f"p{i}"))
    for k in range(2):
        im = ImageView(id=f"img{k}", path=f"photos/{k}.jpg", width=4000, height=3000)
        im.axis_lines = [
            AxisLine(axis=0, p0=[100.0 + k, 200.0], p1=[900.0, 240.0]),
            AxisLine(axis=0, p0=[120.0, 800.0], p1=[950.0, 770.0]),
            AxisLine(axis=2, p0=[300.0, 100.0], p1=[310.0, 900.0]),
            AxisLine(axis=2, p0=[700.0, 120.0], p1=[688.0, 880.0]),
        ]
        im.observations = {f"p{i}": [400.0 + 60 * i + 5 * k, 500.0 + 30 * i] for i in range(3)}
        project.images.append(im)
    return project


# --------------------------------------------------------------------------
# the parameterisation
# --------------------------------------------------------------------------


def test_the_tangent_basis_is_orthonormal_for_every_direction_tried():
    rng = np.random.default_rng(0)
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
    ]
    directions += [v / np.linalg.norm(v) for v in rng.normal(size=(40, 3))]
    for d in directions:
        u, v = tangent_basis(d)
        assert abs(float(u @ d)) < 1e-12, d
        assert abs(float(v @ d)) < 1e-12, d
        assert abs(float(u @ v)) < 1e-12, d
        assert abs(float(u @ u) - 1.0) < 1e-12, d
        assert abs(float(v @ v) - 1.0) < 1e-12, d


def test_a_direction_stays_a_unit_vector_however_far_the_solver_pushes_it():
    """Why the update is a rotation on the sphere and not a normalised offset.

    A least-squares solver early in a trust region will try a step far larger
    than anything geometric, and a parameterisation that only behaves near zero
    hands it a direction that is no longer a direction.
    """
    d = np.array([0.3, -0.5, 0.81])
    d = d / np.linalg.norm(d)
    u, v = tangent_basis(d)
    for a, b in [(0.0, 0.0), (1e-9, 0.0), (0.3, -0.2), (50.0, -31.0), (-1e4, 1e4)]:
        got = direction_at(d, u, v, a, b)
        assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-12, (a, b, got)


def test_the_two_parameters_read_as_an_angle_in_radians():
    d = np.array([0.0, 0.0, 1.0])
    u, v = tangent_basis(d)
    for a, b in [(0.1, 0.0), (0.0, -0.25), (0.3, 0.4)]:
        got = direction_at(d, u, v, a, b)
        moved = math.acos(max(-1.0, min(1.0, float(got @ d))))
        assert abs(moved - math.hypot(a, b)) < 1e-12, (a, b, moved)


# --------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------


def test_the_direction_starts_from_the_two_view_planes_that_cross_most_squarely():
    """A near-parallel pair would hand the bundle noise dressed as a direction.

    Three planes all containing +Z: two of them nearly coincide, and the third
    crosses them squarely. The squarest pair has to win, because the direction
    from the near-parallel pair is a cross product of two almost identical
    vectors and points wherever the last decimal place says.
    """
    truth = np.array([0.0, 0.0, 1.0])
    n_a = np.array([1.0, 0.0, 0.0])
    n_b = np.array([math.cos(0.01), math.sin(0.01), 0.0])
    n_c = np.array([0.0, 1.0, 0.0])
    d, sine, pair = initial_direction([n_a, n_b, n_c])
    assert d is not None
    assert abs(abs(float(d @ truth)) - 1.0) < 1e-12, d
    assert sine > 0.99, sine
    assert set(pair) in ({0, 2}, {1, 2}), pair


def test_a_group_whose_views_are_all_nearly_parallel_gives_no_starting_direction():
    n_a = np.array([1.0, 0.0, 0.0])
    n_b = np.array([math.cos(0.01), math.sin(0.01), 0.0])
    d, sine, _pair = initial_direction([n_a, n_b])
    assert d is None
    assert sine < MIN_PAIR_SINE, sine
    # the sine is still reported, so the caller can say how close it came
    assert sine > 0.0


def test_one_line_cannot_start_a_direction_at_all():
    d, sine, pair = initial_direction([np.array([1.0, 0.0, 0.0])])
    assert d is None and sine == 0.0 and pair is None


def test_a_plane_normal_is_the_drawn_line_carried_into_the_world():
    """The direction a grouped line claims must lie in its own view plane.

    Checked against a case with an answer known by hand: a camera at the
    identity looking down +Z, and a horizontal image line through the principal
    point. That plane contains the world X axis, so a normal perpendicular to X
    is the only correct answer.
    """
    cam = Intrinsics(f=1000.0, cx=500.0, cy=400.0)
    p0 = np.array([100.0, 400.0, 1.0])
    p1 = np.array([900.0, 400.0, 1.0])
    n = plane_normal(np.eye(3), cam.K(), np.cross(p0, p1))
    assert n is not None
    assert abs(float(n @ np.array([1.0, 0.0, 0.0]))) < 1e-12, n
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-12


def test_a_zero_length_drag_has_no_plane_rather_than_a_nonsense_one():
    cam = Intrinsics(f=1000.0, cx=500.0, cy=400.0)
    p = np.array([100.0, 400.0, 1.0])
    assert plane_normal(np.eye(3), cam.K(), np.cross(p, p)) is None


# --------------------------------------------------------------------------
# sign
# --------------------------------------------------------------------------


def test_reversing_a_segment_only_negates_its_residual():
    """Algebraically, which is why the UI need not ask which end you drew from.

    ``cross(p1, p0) == -cross(p0, p1)``, the residual is linear in that, and the
    solve squares it. An axis line cannot do this -- its drag direction resolves
    the vanishing point's sign -- so the simplification is specific to a group
    and worth stating as a fact rather than a hope.
    """
    cam = Intrinsics(f=3200.0, cx=2000.0, cy=1500.0)
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    p0 = np.array([1200.0, 900.0, 1.0])
    p1 = np.array([2600.0, 1450.0, 1.0])
    forward = plane_normal(R, cam.K(), np.cross(p0, p1))
    backward = plane_normal(R, cam.K(), np.cross(p1, p0))
    assert np.allclose(forward, -backward, atol=0.0, rtol=0.0)
    d = FREE_DIR
    assert abs(float(forward @ d) + float(backward @ d)) < 1e-15


def test_the_end_you_drew_from_makes_no_difference_to_the_solve():
    drawn, truth = build_project()
    add_free_lines(drawn, truth, images=[0, 2, 4], per_image=2)
    reversed_ = copy.deepcopy(drawn)
    flipped = 0
    for im in reversed_.images:
        for ln in im.axis_lines:
            if ln.direction_group is not None:
                ln.p0, ln.p1 = ln.p1, ln.p0
                flipped += 1
    assert flipped == 6, flipped

    a = solve(drawn)
    b = solve(reversed_)
    assert a.ok and b.ok, (a.stage, b.stage)

    d_a = np.array(group_report(a).direction, dtype=float)
    d_b = np.array(group_report(b).direction, dtype=float)
    # up to sign, because a group has no positive end: that is the whole point
    angle = math.degrees(math.acos(min(1.0, abs(float(d_a @ d_b)))))
    assert angle < 0.01, angle
    for im_a, im_b in zip(drawn.images, reversed_.images):
        assert np.allclose(im_a.R, im_b.R, atol=1e-7), im_a.id
    assert abs(a.rms_line_px - b.rms_line_px) < 1e-7


# --------------------------------------------------------------------------
# recovering a direction nobody named
# --------------------------------------------------------------------------


def test_a_group_recovers_a_direction_nobody_named():
    """The whole premise, against a direction aligned with no world axis.

    The axis lines pin the rotations to the true world frame, so the recovered
    direction is directly comparable to the truth -- no similarity alignment
    needed, and no way for a wrong answer to hide behind a gauge freedom.
    """
    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 2, 4], per_image=2)
    report = solve(project)
    assert report.ok, (report.stage, report.messages)

    grp = group_report(report)
    assert grp.identifiable, grp.note
    assert grp.n_lines == 6 and grp.n_images == 3, (grp.n_lines, grp.n_images)
    assert grp.net_constraints == 4, grp.net_constraints

    d = np.array(grp.direction, dtype=float)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-9
    angle = math.degrees(math.acos(min(1.0, abs(float(d @ FREE_DIR)))))
    assert angle < 3.0, f"recovered direction is {angle:.2f} degrees off"

    # and it is written back onto the project, so the panel can show it
    stored = project.direction_group("d0").vector()
    assert stored is not None
    assert abs(abs(float(stored @ d)) - 1.0) < 1e-9

    # the lines agree with the direction about as well as axis lines do
    assert grp.rms_px < 6.0, grp.rms_px


def test_the_direction_is_refined_by_the_bundle_and_not_merely_initialised():
    """The starting direction is already decent, which is a trap.

    ``initial_direction`` picks the squarest pair of view planes, and on clean
    data that alone lands within a tenth of a degree. A group whose two
    parameters never actually moved would therefore pass a recovery test while
    contributing nothing, and the bug would be invisible. So the initialisation
    is measured against the answer: the bundle has to improve on it.
    """
    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 2, 4], per_image=2)
    report = solve(project)
    assert report.ok

    point_ids = [p.id for p in project.points if p.xyz is not None]
    problem = _Problem(project, {}, [im.id for im in project.images], point_ids)
    started = problem.dir_base["d0"]
    finished = np.array(group_report(report).direction, dtype=float)

    def off(d):
        return math.degrees(math.acos(min(1.0, abs(float(np.asarray(d) @ FREE_DIR)))))

    assert off(started) > 0.02, f"initialisation was already {off(started):.4f} deg -- "
    assert off(finished) < 0.5 * off(started), (
        f"the bundle left the direction where the initialisation put it "
        f"({off(started):.4f} -> {off(finished):.4f} degrees off truth)"
    )


def test_a_group_tightens_the_rotations_it_spans():
    """The claim the whole feature rests on, measured rather than assumed.

    Degrade the axis lines until the rotations are visibly wrong, then mark one
    unknown direction cleanly across every photo. The group cannot fix a focal
    length and cannot move a camera, but it ties the rotations of the photos it
    spans together about one shared axis, and that has to show up as a smaller
    error against ground truth. If this test ever fails the feature is costing
    two unknowns a group and buying nothing.
    """
    from test_synthetic import rotation_error_deg

    plain, truth = build_project(line_noise_px=12.0)
    grouped = copy.deepcopy(plain)
    add_free_lines(grouped, truth, images=[0, 1, 2, 3, 4, 5], per_image=2, noise_px=1.0)

    a, b = solve(plain), solve(grouped)
    assert a.ok and b.ok, (a.stage, b.stage)

    def rot_errs(project):
        return [
            rotation_error_deg(im.R, R)
            for im, R in zip(project.images, truth["R"])
            if im.R is not None
        ]

    before, after = rot_errs(plain), rot_errs(grouped)
    # at this much line noise a photo or two may fail to calibrate at all, which
    # a group cannot help with -- it is the same set either way that matters
    assert len(before) == len(after) >= 5, (len(before), len(after))
    assert np.mean(after) < 0.9 * np.mean(before), (
        f"mean rotation error {np.mean(before):.4f} -> {np.mean(after):.4f} degrees; the group "
        "did not tighten the rotations it spans"
    )
    assert max(after) < max(before)

    grp = group_report(b)
    assert grp.net_constraints >= 6, grp.net_constraints


def test_two_groups_at_once_each_find_their_own_direction():
    """Nothing in the bookkeeping ties two groups together, so this is the test
    that would catch it if something did -- a shared tangent basis, a parameter
    offset computed once, a report keyed by the wrong id."""
    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 2, 4], per_image=2)

    second = np.array([-0.35, 0.82, 0.45])
    second = second / np.linalg.norm(second)
    global FREE_DIR
    first_dir = FREE_DIR
    try:
        FREE_DIR = second
        add_free_lines(project, truth, images=[1, 3, 5], group_id="d1", label="conduit", per_image=2)
    finally:
        FREE_DIR = first_dir

    report = solve(project)
    assert report.ok, (report.stage, report.messages)
    for gid, want in (("d0", first_dir), ("d1", second)):
        grp = group_report(report, gid)
        assert grp.identifiable, (gid, grp.note)
        d = np.array(grp.direction, dtype=float)
        angle = math.degrees(math.acos(min(1.0, abs(float(d @ want)))))
        assert angle < 4.0, f"{gid} is {angle:.2f} degrees off"


# --------------------------------------------------------------------------
# refusals and honest counting
# --------------------------------------------------------------------------


def test_a_group_confined_to_one_photo_is_refused_rather_than_fitted():
    """Three lines, one photograph, and the right answer is still "no".

    The direction is perfectly recoverable from those three lines -- they meet at
    a vanishing point. It constrains the camera by nothing, because the solve can
    turn the camera and counter-turn the direction with no residual changing at
    all. Fitting it would report a confident direction for a group that is doing
    no work.
    """
    project, truth = build_project()
    add_free_lines(project, truth, images=[1], per_image=3)
    report = solve(project)
    assert report.ok, (report.stage, report.messages)

    grp = group_report(report)
    assert not grp.identifiable
    assert grp.direction is None
    assert grp.n_lines == 3 and grp.n_images == 1
    assert grp.net_constraints == 0
    assert "one photo" in grp.note, grp.note
    assert any("direction group" in w for w in report.warnings), report.warnings


def test_a_refused_group_leaves_every_solved_number_bit_for_bit_unchanged():
    """A refusal has to mean ignored, not down-weighted.

    Bit-for-bit rather than within a tolerance, because a refused group that
    still reached the parameter vector or the residual table would move the
    answer in the last few decimals and look like nothing at all.
    """
    plain, truth = build_project()
    with_refused = copy.deepcopy(plain)
    add_free_lines(with_refused, truth, images=[1], per_image=3)

    a = solve(plain)
    b = solve(with_refused)
    assert a.ok and b.ok
    assert solved_digest(plain) == solved_digest(with_refused)
    assert a.rms_reprojection_px == b.rms_reprojection_px
    assert a.rms_line_px == b.rms_line_px


def test_a_two_line_group_is_reported_as_contributing_nothing():
    """2 unknowns, 2 residuals. It fits itself exactly and says nothing.

    Reported rather than refused: it is not wrong, and a user part-way through
    marking a third photo should see the group they are building. What they must
    not see is a line count that implies it is already helping.
    """
    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 3], per_image=1)
    report = solve(project)
    assert report.ok, (report.stage, report.messages)

    grp = group_report(report)
    assert grp.identifiable, grp.note
    assert grp.n_lines == 2 and grp.n_images == 2
    assert grp.net_constraints == 0
    assert "break-even" in grp.note, grp.note


def test_the_contribution_note_counts_lines_and_not_groups():
    assert "nothing" in contribution_note(2, 2, 0.9)
    assert "break-even" in contribution_note(2, 2, 0.9)
    assert "1 net constraint" in contribution_note(3, 2, 0.9)
    assert "3 net constraint" in contribution_note(5, 3, 0.9)
    assert "one photo" in contribution_note(4, 1, 0.9)
    assert "no lines" in contribution_note(0, 0, 0.0)


def test_a_badly_conditioned_group_is_named_as_such_rather_than_merely_survived():
    """Low conditioning means another line from the same spot will not help, and
    a photo looking across the edge will. That is a different piece of advice
    from "add more lines", so it needs a different sentence."""
    note = contribution_note(6, 3, 0.1)
    assert "weakly observed" in note, note
    assert "across" in note, note
    assert "weakly observed" not in contribution_note(6, 3, 0.9)


# --------------------------------------------------------------------------
# what a group is not allowed to do
# --------------------------------------------------------------------------


def test_a_grouped_line_is_not_offered_to_the_calibration():
    """``axis_segments`` is what ``calibrate_all`` reads, and a direction that is
    itself a bundle output is no use to a vanishing point."""
    im = ImageView(id="img0", path="a.jpg", width=4000, height=3000)
    im.axis_lines = [
        AxisLine(axis=0, p0=[0.0, 0.0], p1=[100.0, 10.0]),
        AxisLine(axis=0, p0=[0.0, 50.0], p1=[100.0, 65.0]),
        AxisLine(axis=1, p0=[0.0, 0.0], p1=[10.0, 100.0], direction_group="d0"),
        AxisLine(axis=1, p0=[50.0, 0.0], p1=[65.0, 100.0], direction_group="d0"),
    ]
    segs = im.axis_segments()
    assert set(segs) == {0}, segs
    assert len(segs[0]) == 2


def test_a_photo_carrying_only_grouped_lines_still_does_not_calibrate():
    """And the report says why, because this is exactly where someone who has
    just drawn six tray lines into a photo would otherwise sit and wonder."""
    project, truth = build_project()
    bare = project.images[2]
    bare.axis_lines = []
    add_free_lines(project, truth, images=[2], per_image=3)
    add_free_lines(project, truth, images=[0, 4], per_image=1)

    from multispy.solve import SolveReport

    report = SolveReport()
    intrinsics = calibrate_all(project, report)
    assert bare.id not in intrinsics
    notes = " ".join(report.images[bare.id].notes)
    assert "need 2 axes" in notes, notes
    assert "direction group" in notes, notes


def test_a_line_naming_a_group_that_no_longer_exists_is_ignored_and_reported():
    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 2, 4], per_image=2)
    project.direction_groups = []          # as deleting the group from the panel leaves it
    report = solve(project)
    assert report.ok, (report.stage, report.messages)
    assert not report.direction_groups
    assert any("no longer exists" in w for w in report.warnings), report.warnings


# --------------------------------------------------------------------------
# the axis path, untouched
# --------------------------------------------------------------------------


def test_an_axis_only_project_solves_exactly_as_it_did_before_groups_existed():
    project, _truth = build_project()
    report = solve(project)
    assert report.ok, (report.stage, report.messages)
    assert not report.direction_groups
    assert solved_digest(project) == AXIS_ONLY_DIGEST, (
        "the axis-only solve has moved. Generalising the control line residual "
        "leaked into the axis path -- see the docstring on AXIS_ONLY_DIGEST before "
        "regenerating it."
    )


def test_an_axis_only_project_has_the_residual_layout_it_always_had():
    """The offsets, not just the answer. A new block inserted in the middle of
    the residual vector is the mechanism by which the numbers above would move,
    so it is worth asserting where it is rather than only that it was harmless."""
    project, _truth = build_project()
    report = solve(project)
    assert report.ok
    point_ids = [p.id for p in project.points if p.xyz is not None]
    problem = _Problem(project, {}, [im.id for im in project.images], point_ids)
    assert problem.r_dirlines == 0
    assert problem.o_cons == problem.o_lines + problem.r_lines
    assert not problem.dir_group_ids
    # and no parameters were set aside for a direction nobody declared
    assert problem.off_points + 3 * len(point_ids) == problem.n_params - 6 * len(
        problem.instance_slots
    )


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_groups_and_the_per_line_reference_survive_a_round_trip():
    project = small_project()
    project.direction_groups.append(DirectionGroup(id="d0", label="tray 1", d=[0.6, -0.45, 0.66]))
    project.direction_groups.append(DirectionGroup(id="d1", label="conduit"))
    project.images[0].axis_lines.append(
        AxisLine(axis=0, p0=[10.0, 20.0], p1=[500.0, 90.0], direction_group="d0")
    )
    project.images[1].axis_lines.append(
        AxisLine(axis=0, p0=[15.0, 25.0], p1=[505.0, 95.0], direction_group="d0", uncertain=True)
    )

    back = Project.from_dict(project.to_dict())
    assert [g.id for g in back.direction_groups] == ["d0", "d1"]
    assert back.direction_group("d0").label == "tray 1"
    assert np.allclose(back.direction_group("d0").vector(), np.array([0.6, -0.45, 0.66]) / 1.0
                       / np.linalg.norm([0.6, -0.45, 0.66]))
    assert back.direction_group("d1").d is None
    assert back.images[0].axis_lines[-1].direction_group == "d0"
    assert back.images[1].axis_lines[-1].direction_group == "d0"
    assert back.images[1].axis_lines[-1].uncertain is True
    assert back.images[0].axis_lines[0].direction_group is None


def test_a_project_file_written_before_groups_existed_still_loads():
    project = small_project()
    raw = project.to_dict()
    raw.pop("direction_groups", None)
    for im in raw["images"]:
        for ln in im["axis_lines"]:
            ln.pop("direction_group", None)
    back = Project.from_dict(raw)
    assert back.direction_groups == []
    assert all(
        ln.direction_group is None for im in back.images for ln in im.axis_lines
    )
    assert len(back.images[0].axis_lines) == 4


# --------------------------------------------------------------------------
# staleness
# --------------------------------------------------------------------------


def test_changing_a_group_makes_a_measured_origin_read_stale():
    """A stored origin radius describes one set of marks. Move a line into a
    group, or out of one, and it describes data it was never measured on."""
    project = small_project()
    project.direction_groups.append(DirectionGroup(id="d0", label="tray 1"))
    project.images[0].axis_lines.append(
        AxisLine(axis=0, p0=[10.0, 20.0], p1=[500.0, 90.0], direction_group="d0")
    )
    project.images[1].axis_lines.append(
        AxisLine(axis=0, p0=[15.0, 25.0], p1=[505.0, 95.0], direction_group="d0")
    )
    before = uncertainty.fingerprint(project)

    moved = copy.deepcopy(project)
    moved.direction_groups.append(DirectionGroup(id="d1", label="conduit"))
    moved.images[1].axis_lines[-1].direction_group = "d1"
    assert uncertainty.fingerprint(moved) != before

    released = copy.deepcopy(project)
    released.images[1].axis_lines[-1].direction_group = None
    assert uncertainty.fingerprint(released) != before

    renamed = copy.deepcopy(project)
    renamed.direction_group("d0").label = "tray 2"
    assert uncertainty.fingerprint(renamed) == before, (
        "a label is not evidence, so renaming a group must not invalidate a reading"
    )


def test_an_axis_only_project_keeps_the_fingerprint_it_had():
    """The converse, and the reason the group id enters the digest only when a
    line has one: a measured origin must not go stale on every existing project
    merely because this build understands something that project does not use."""
    assert uncertainty.fingerprint(small_project()) == AXIS_ONLY_FINGERPRINT


def test_a_solved_direction_is_not_itself_evidence():
    """``DirectionGroup.d`` is an output, like ``ImageView.R``. Re-solving writes
    a new one, and if that entered the fingerprint the reading would invalidate
    itself every time the project was solved."""
    project = small_project()
    project.direction_groups.append(DirectionGroup(id="d0", label="tray 1"))
    before = uncertainty.fingerprint(project)
    project.direction_group("d0").d = [0.6, -0.45, 0.66]
    assert uncertainty.fingerprint(project) == before


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def test_the_report_is_serialisable_for_the_browser():
    import json

    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 2, 4], per_image=2)
    report = solve(project)
    blob = json.dumps(report.to_dict())
    assert '"direction_groups"' in blob
    assert '"net_constraints"' in blob


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    project, truth = build_project()
    add_free_lines(project, truth, images=[0, 2, 4], per_image=2)
    add_free_lines(project, truth, images=[1, 5], group_id="d1", label="two-line group")
    add_free_lines(project, truth, images=[3], group_id="d2", label="one photo only", per_image=3)
    report = solve(project)
    print("\nthree groups on the synthetic scene, one of each kind")
    print(f"{'group':<8}{'lines':>6}{'photos':>7}{'net':>5}{'cond':>7}{'rms px':>8}  note")
    for g in report.direction_groups:
        print(f"{g.id:<8}{g.n_lines:>6}{g.n_images:>7}{g.net_constraints:>5}"
              f"{g.conditioning:>7.2f}{g.rms_px:>8.2f}  {g.note[:58]}")
    d = report.direction_groups[0].direction
    if d is not None:
        off = math.degrees(math.acos(min(1.0, abs(float(np.array(d) @ FREE_DIR)))))
        print(f"\nd0 recovered {off:.2f} degrees from a direction never stated. "
              f"Axis-line RMS and group RMS are the same measure: {report.rms_line_px:.2f} px overall.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
