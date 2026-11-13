"""The part's own CAD model: placing it at the origin, checking the fit, and
sliding it onto the edges of a photograph.

The scene is a machined plate with a raised pad, dimensioned in millimetres from
a datum corner, photographed from six viewpoints. What makes it different from
``test_rig``'s scene is that the tie points carry *nominals* -- the coordinates
the drawing states -- so the solve lands in the part's own coordinate frame
rather than an arbitrary one. At that point the part needs no anchors and no
hand placement: it is at the origin by definition, and its wireframe lands in
every photograph from pure geometry.

What is actually being defended here:

* **Stage one is geometry, not a fit.** The projected wireframe is compared
  against the ground-truth projection of the same vertices through the true
  camera, and it has to agree to a fraction of a pixel. There is no search in it
  to go wrong.

* **A units mismatch is reported, never applied.** A model declared in metres
  against a drawing in millimetres is off by exactly 1000. The dangerous
  outcome is not a bad number, it is a *good* one -- silently rescaling would
  make every deviation in the report read as a clean zero. So the test asserts
  the report says 1000x, says which declaration it doubts, and does not quietly
  fix it.

* **Stage two cannot make a good pose worse.** The honest risk of an edge
  alignment is a confident fit to the wrong edges, so the test starts from the
  true pose on a photograph with a deliberately busy background and requires
  that the pose comes back unharmed -- and separately, with every acceptance
  gate switched off, that the optimiser's own answer from a correct start is
  still within a pixel of correct.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_part.py
"""

from __future__ import annotations

import io
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.edgefit import (
    build_edge_field,
    build_samples,
    load_grayscale,
    refine_image_pose,
    refine_images,
    revert_image_pose,
)
from multispy.geometry import Intrinsics, project as project_points
from multispy.mesh import load_mesh
from multispy.part import (
    intrinsics_for,
    mark_reprojection_rms,
    part_fit,
    place_part,
    project_part_wireframe,
    surface_distances,
)
from multispy.project import (
    AxisLine,
    CameraGroup,
    ImageView,
    ObjectTemplate,
    Project,
    TiePoint,
    UNIT_SCALES,
)
from multispy.solve import solve

WIDTH, HEIGHT = 2400, 1800
TRUE_FOCAL = 2100.0

PLATE = (120.0, 80.0, 20.0)
"""The part, in millimetres, with the drawing's datum at one corner -- which is
how a real part is dimensioned and, more to the point, means the solved origin
has to land somewhere specific rather than anywhere convenient."""
PAD = (46.0, 30.0, 8.0)
PAD_AT = (34.0, 25.0)

SCRATCH = Path(tempfile.mkdtemp(prefix="multispy_part_"))


# --------------------------------------------------------------------------
# the part
# --------------------------------------------------------------------------


def plate_obj() -> str:
    """The plate plus its pad, as an OBJ in millimetres."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def box(x0, y0, z0, x1, y1, z1):
        i = len(verts) + 1
        for z in (z0, z1):
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                verts.append((x, y, z))
        faces.extend(
            [
                (i + 3, i + 2, i + 1, i),               # bottom, outward -z
                (i + 4, i + 5, i + 6, i + 7),           # top
                (i, i + 1, i + 5, i + 4),
                (i + 1, i + 2, i + 6, i + 5),
                (i + 2, i + 3, i + 7, i + 6),
                (i + 3, i, i + 4, i + 7),
            ]
        )

    box(0, 0, 0, PLATE[0], PLATE[1], PLATE[2])
    box(
        PAD_AT[0], PAD_AT[1], PLATE[2],
        PAD_AT[0] + PAD[0], PAD_AT[1] + PAD[1], PLATE[2] + PAD[2],
    )
    return (
        "".join(f"v {x:g} {y:g} {z:g}\n" for x, y, z in verts)
        + "".join("f " + " ".join(str(i) for i in f) + "\n" for f in faces)
    )


def part_mesh_path() -> Path:
    path = SCRATCH / "plate.obj"
    if not path.exists():
        path.write_text(plate_obj(), encoding="utf-8")
    return path


DATUM_CORNERS = np.array(
    [
        [0.0, 0.0, 0.0],
        [PLATE[0], 0.0, 0.0],
        [PLATE[0], PLATE[1], 0.0],
        [0.0, PLATE[1], 0.0],
        [0.0, 0.0, PLATE[2]],
        [PLATE[0], 0.0, PLATE[2]],
        [PLATE[0], PLATE[1], PLATE[2]],
        [0.0, PLATE[1], PLATE[2]],
        [PAD_AT[0], PAD_AT[1], PLATE[2] + PAD[2]],
        [PAD_AT[0] + PAD[0], PAD_AT[1] + PAD[1], PLATE[2] + PAD[2]],
    ]
)
"""The features a person would both mark in the photos and read off the drawing:
the plate's eight corners and two diagonally opposite corners of the pad."""

CORNER_NORMALS = np.array(
    [
        [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]],
        [[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]],
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]],
        [[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]],
        [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]],
        [[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]],
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
        [[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
        [[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]],
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
    ]
)

PLATE_EDGES = [
    (0, 1, 0), (3, 2, 0), (4, 5, 0), (7, 6, 0),
    (0, 3, 1), (1, 2, 1), (4, 7, 1), (5, 6, 1),
    (0, 4, 2), (1, 5, 2), (2, 6, 2), (3, 7, 2),
]
"""``(from, to, axis)`` along the plate's own edges, in the positive axis
direction. In part space the part's axes *are* the world axes, which is exactly
what makes the control lines and the nominals talk about the same frame."""


def look_at(center: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0])):
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.array([x, y, z])
    return R, -R @ center


def viewpoints() -> list[tuple[np.ndarray, np.ndarray]]:
    """Six photos arcing over the plate, all looking at the middle of its top."""
    target = np.array([PLATE[0] / 2.0, PLATE[1] / 2.0, PLATE[2]]) * 0.001
    out = []
    for angle_deg, dist_mm, height_mm in (
        (-62.0, 300.0, 150.0),
        (-34.0, 265.0, 205.0),
        (-8.0, 250.0, 130.0),
        (18.0, 275.0, 220.0),
        (44.0, 260.0, 145.0),
        (68.0, 310.0, 190.0),
        (152.0, 295.0, 125.0),
        (205.0, 285.0, 170.0),
    ):
        a = math.radians(angle_deg)
        centre = target + np.array(
            [math.sin(a) * dist_mm, -math.cos(a) * dist_mm, 0.0]
        ) * 0.001
        centre[2] = height_mm * 0.001
        out.append(look_at(centre, target))
    return out


def visible_corners(centre_m: np.ndarray, min_faces: int = 2) -> np.ndarray:
    """Which datum corners a person could pick out from here.

    Two of the three faces meeting at a corner must be turned toward the camera,
    which is the stricter reading: it keeps corners that are merely on the
    silhouette's far edge out of the marks, since those are not what anyone
    clicks.
    """
    to_camera = centre_m[None, :] - DATUM_CORNERS * 0.001
    facing = np.einsum("nfi,ni->nf", CORNER_NORMALS, to_camera) > 0.0
    return facing.sum(axis=1) >= min_faces


def build_part_project(
    seed: int = 7,
    tie_noise_px: float = 0.7,
    line_noise_px: float = 1.0,
    nominals_on: int = 8,
    nominal_sigma: float = 0.5,
    template_units: str = "mm",
    with_part: bool = True,
) -> tuple[Project, dict]:
    """A part-space project: nominals from the drawing, and the part's own model.

    ``nominal_sigma`` defaults to half a millimetre -- loose enough that the
    nominals place the part without dictating the answer, which is the whole
    distinction the nominals-dictate warning exists to police.
    """
    rng = np.random.default_rng(seed)
    cam_true = Intrinsics(f=TRUE_FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)
    poses = viewpoints()

    project = Project(name="machined-plate", nominal_units="mm", nominal_scale=0.001)
    project.groups["cam0"] = CameraGroup(id="cam0")
    project.settings.sigma_nominal = nominal_sigma

    for i, xyz in enumerate(DATUM_CORNERS):
        pt = TiePoint(id=f"p{i}", label=f"corner {i}")
        if i < nominals_on:
            pt.nominal = [float(v) for v in xyz]
        project.points.append(pt)

    if with_part:
        project.templates["tpl0"] = ObjectTemplate(
            id="tpl0",
            name="machined plate",
            mesh_path=str(part_mesh_path()),
            units=template_units,
            unit_scale=UNIT_SCALES.get(template_units, 0.001),
        )
        project.set_part("tpl0")

    truth: dict = {"R": [], "t": [], "focal": TRUE_FOCAL, "cam": cam_true}
    world = DATUM_CORNERS * 0.001

    for ci, (R, t) in enumerate(poses):
        centre = -R.T @ t
        image = ImageView(
            id=f"img{ci}",
            path=f"photos/plate_{ci:02d}.jpg",
            width=WIDTH,
            height=HEIGHT,
            group="cam0",
        )
        truth["R"].append(R)
        truth["t"].append(t)

        uv = project_points(R, t, cam_true, world)
        inside = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] > 30)
            & (uv[:, 0] < WIDTH - 30)
            & (uv[:, 1] > 30)
            & (uv[:, 1] < HEIGHT - 30)
        )
        seen = inside & visible_corners(centre, min_faces=2)
        for i in np.flatnonzero(seen):
            noisy = uv[i] + rng.normal(0.0, tie_noise_px, 2)
            image.observations[f"p{i}"] = [float(noisy[0]), float(noisy[1])]

        # A control line only needs its two ends to be *on* the silhouette, not
        # inside it: you can draw along the top edge of a plate from any angle
        # that shows the top edge, while clicking the corner itself as a point
        # wants the corner to read crisply, which is the stricter test above.
        drawable = inside & visible_corners(centre, min_faces=1)
        for a, b, axis in PLATE_EDGES:
            if not (drawable[a] and drawable[b]):
                continue
            if np.linalg.norm(uv[b] - uv[a]) < 60.0:
                continue
            p0 = uv[a] + rng.normal(0.0, line_noise_px, 2)
            p1 = uv[b] + rng.normal(0.0, line_noise_px, 2)
            image.axis_lines.append(
                AxisLine(
                    axis=axis,
                    p0=[float(p0[0]), float(p0[1])],
                    p1=[float(p1[0]), float(p1[1])],
                )
            )
        project.images.append(image)

    return project, truth


_solved_cache: dict = {}


def solved_project(**kwargs):
    """A solved part-space project, memoised -- the solve takes a few seconds and
    most of the assertions below want the same one.

    Each caller gets its own copy, round-tripped through the project format.
    Several of the tests below designate a different part, clear it, or move a
    camera, and handing them all the same object made one test's mutation into
    the next test's confusing failure. The round trip is free here and doubles
    as a persistence check on every field these tests touch.
    """
    key = tuple(sorted(kwargs.items()))
    if key not in _solved_cache:
        project, truth = build_part_project(**kwargs)
        report = solve(project)
        _solved_cache[key] = (project.to_dict(), project.path, truth, report)
    raw, path, truth, report = _solved_cache[key]
    return Project.from_dict(raw, path=path), truth, report


# --------------------------------------------------------------------------
# rendering a photograph to align against
# --------------------------------------------------------------------------


def render_photo(
    project: Project,
    image: ImageView,
    R: np.ndarray,
    t: np.ndarray,
    mesh,
    clutter: int = 0,
    seed: int = 3,
) -> bytes:
    """A shaded render of the part through a given camera, as JPEG bytes.

    A real render rather than a drawn wireframe, because the thing being tested
    is an *edge detector*: lines painted straight onto a flat background would
    hand the alignment a clean, one-pixel-wide, perfectly placed target and
    prove nothing about a photograph. Shading the faces makes the edges emerge
    from contrast the way they do in a photo, silhouette and crease alike.

    ``clutter`` adds straight high-contrast lines in the background -- the
    bookshelf-behind-the-part failure mode, which is what the orientation
    binning and the truncation exist to survive.
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    scale = 0.5
    w, h = int(WIDTH * scale), int(HEIGHT * scale)
    img = Image.new("L", (w, h), 70)
    draw = ImageDraw.Draw(img)

    for _ in range(clutter):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, math.pi)
        length = rng.uniform(0.3, 1.2) * w
        x1, y1 = x0 + math.cos(ang) * length, y0 + math.sin(ang) * length
        draw.line(
            [(x0, y0), (x1, y1)],
            fill=int(rng.integers(140, 235)),
            width=int(rng.integers(2, 6)),
        )

    cam = intrinsics_for(project, image)
    verts = np.asarray(mesh.vertices, dtype=float) * 0.001
    faces = np.asarray(mesh.faces, dtype=int)
    normals = mesh.face_normals()
    centre = -np.asarray(R).T @ np.asarray(t)
    uv = project_points(np.asarray(R), np.asarray(t), cam, verts) * scale

    centroids = verts[faces].mean(axis=1)
    toward = centre[None, :] - centroids
    front = np.einsum("ij,ij->i", normals, toward) > 0.0
    depth = np.linalg.norm(toward, axis=1)
    light = np.array([0.35, -0.55, 0.76])
    light = light / np.linalg.norm(light)

    for fi in np.argsort(-depth):   # painter's algorithm, far faces first
        if not front[fi]:
            continue
        poly = [tuple(uv[v]) for v in faces[fi]]
        if not all(np.isfinite(p).all() for p in poly):
            continue
        shade = 0.5 + 0.5 * abs(float(normals[fi] @ light))
        draw.polygon(poly, fill=int(np.clip(120 + 125 * shade, 0, 255)))

    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0.0, 2.5, arr.shape)      # sensor noise, so nothing is exact
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def perturb(R: np.ndarray, t: np.ndarray, rot_deg: float, shift_m: float, seed: int = 5):
    """A pose knocked slightly off, for showing what the refinement does."""
    from multispy.geometry import so3_exp

    rng = np.random.default_rng(seed)
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * math.radians(rot_deg)
    d = rng.normal(size=3)
    d = d / np.linalg.norm(d) * shift_m
    return so3_exp(w) @ np.asarray(R), np.asarray(t) + d


# --------------------------------------------------------------------------
# the fit report
# --------------------------------------------------------------------------


def test_the_solve_reaches_part_space():
    _project, _truth, report = solved_project()
    assert report.ok, f"solve failed at {report.stage}"
    assert report.part_space, "eight nominal'd corners should put this in part space"
    assert report.n_nominals == 8, report.n_nominals
    assert report.rms_reprojection_px < 1.5, report.rms_reprojection_px


def test_the_part_fits_its_own_reconstruction():
    project, _truth, report = solved_project()
    fit = part_fit(project)
    assert fit.ok, f"{fit.stage}: {fit.warnings}"
    assert fit.n_points == 8, fit.n_points
    assert fit.rms_deviation < 1.0, f"{fit.rms_deviation:.4f} mm RMS off the drawing"
    assert fit.max_deviation < 2.5, f"{fit.max_deviation:.4f} mm worst"
    # Every nominal is a corner of the plate, so the drawing and the model must
    # agree before any photograph is involved. This is the check that costs
    # nothing and catches the worst mistakes.
    worst_nominal = max(r.nominal_surface_distance for r in fit.points)
    assert worst_nominal < 1e-6, f"a nominal sits {worst_nominal:.4g} mm off the model"
    assert abs(fit.implied_scale - 1.0) < 0.02, fit.implied_scale


def test_the_solve_report_carries_the_fit():
    _project, _truth, report = solved_project()
    assert report.part_fit is not None, "a designated part should be reported on by the solve"
    assert report.part_fit["ok"], report.part_fit["stage"]
    assert report.part_fit["n_points"] == 8
    assert "points" in report.part_fit and len(report.part_fit["points"]) == 8


def test_the_reconstruction_lands_on_the_models_surface():
    project, _truth, _report = solved_project()
    fit = part_fit(project)
    # The thing the constraint table cannot say: the solved points agree with
    # the whole shape, not merely with the three numbers typed for each of them.
    assert fit.rms_surface < 1.0, f"{fit.rms_surface:.4f} mm RMS off the surface"
    assert fit.max_surface < 2.5, f"{fit.max_surface:.4f} mm worst"


def test_a_wrong_nominal_shows_up_against_the_model():
    """A corner given the wrong drawing coordinate is caught by the geometry.

    Not by the reprojection residuals, which stay under a pixel: the solve
    simply bends the weakly-determined positions to suit. The model's surface
    is what has an opinion about it.
    """
    project, _truth = build_part_project(seed=7)
    bad = project.point("p6")
    bad.nominal = [bad.nominal[0], bad.nominal[1], bad.nominal[2] + 9.0]
    solve(project)
    fit = part_fit(project)
    row = next(r for r in fit.points if r.point == "p6")
    assert row.nominal_surface_distance > 5.0, (
        f"a nominal 9 mm off the part read as {row.nominal_surface_distance:.3f} mm off its surface"
    )
    assert any("do not lie on the model" in w for w in fit.warnings), fit.warnings


def test_a_two_view_nominal_is_reported_as_uncheckable():
    project, _truth, _report = solved_project()
    fit = part_fit(project)
    for row in fit.points:
        assert row.checkable == (row.n_views >= 3), row
        if row.solved and not row.checkable:
            assert "checks nothing" in row.note, row.note


def test_no_part_is_not_a_failed_fit():
    project, _truth = build_part_project(with_part=False)
    fit = part_fit(project)
    assert not fit.ok
    assert fit.stage == "no part designated"
    assert not fit.warnings, "a project with no part has nothing wrong with it"


def test_not_in_part_space_refuses_rather_than_guessing():
    project, _truth = build_part_project(nominals_on=2)
    solve(project)
    fit = part_fit(project)
    assert not fit.ok
    assert fit.stage == "not in part space"
    assert any("part space" in w for w in fit.warnings), fit.warnings
    # and the fit does not invent a deviation table for an unknown pose
    assert not fit.points


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------


def test_a_units_mismatch_is_reported_as_a_factor_not_scaled_away():
    """The trap: a model declared in metres against a drawing in millimetres.

    Rescaling it to fit would make every deviation in the report read zero,
    which looks like a flawless result and is the complete absence of one. So
    the report has to name the factor, accuse the model's declaration, and leave
    the geometry alone.
    """
    project, _truth = build_part_project(template_units="m")
    solve(project)
    fit = part_fit(project)
    assert not fit.units_agree, "mm nominals against an m model should not read as agreeing"
    assert abs(fit.unit_ratio - 1000.0) < 1e-6, fit.unit_ratio
    assert fit.units_mismatch, "a model 1000x the part's size should be called out"
    assert any("1000" in w for w in fit.warnings), fit.warnings
    assert any("declared model units" in w for w in fit.warnings), fit.warnings
    assert fit.span_ratio > 100.0, fit.span_ratio
    # nothing was rescaled: the model is still the size it declared
    assert fit.model_span > 1e5, f"model span {fit.model_span:g} looks silently corrected"


def test_agreeing_units_say_nothing():
    project, _truth, _report = solved_project()
    fit = part_fit(project)
    assert fit.units_agree and not fit.units_mismatch
    assert fit.unit_ratio == 1.0
    assert not any("factor of" in w for w in fit.warnings), fit.warnings


def test_legitimately_different_units_are_flagged_but_not_accused():
    """Inches against a millimetre drawing is a real possibility, not an error.

    The part is modelled at 1/25.4 the coordinate values so that it is the same
    physical object; the two declarations disagree and both are right. The report
    should say they differ and stop there -- the accusation is reserved for the
    case where the part's own size says which one is wrong.
    """
    path = SCRATCH / "plate_inch.obj"
    text = plate_obj().splitlines()
    scaled = []
    for line in text:
        if line.startswith("v "):
            x, y, z = (float(v) / 25.4 for v in line.split()[1:])
            scaled.append(f"v {x:g} {y:g} {z:g}")
        else:
            scaled.append(line)
    path.write_text("\n".join(scaled) + "\n", encoding="utf-8")

    project, _truth, _report = solved_project()
    project = Project.from_dict(project.to_dict(), path=project.path)
    template = project.templates["tpl0"]
    template.mesh_path = str(path)
    template.units, template.unit_scale = "in", UNIT_SCALES["in"]
    fit = part_fit(project)
    assert not fit.units_agree, "in against mm does differ"
    assert not fit.units_mismatch, (
        f"an inch model of a millimetre part is the same object; span ratio {fit.span_ratio:.4f}"
    )
    assert any("factor of" in w for w in fit.warnings), fit.warnings
    assert fit.rms_deviation < 1.0, fit.rms_deviation


# --------------------------------------------------------------------------
# stage one
# --------------------------------------------------------------------------


def test_the_part_is_pinned_at_the_identity():
    project, _truth, _report = solved_project()
    inst = project.part_instance()
    assert inst is not None and inst.is_part
    assert np.allclose(inst.R, np.eye(3)), inst.R
    assert np.allclose(inst.t, 0.0), inst.t
    assert inst.locked and not inst.manual
    # and it costs the bundle nothing: no anchors, so no parameters
    assert not any(
        key.startswith(inst.id + "/") for im in project.images for key in im.anchor_obs
    )


def test_only_one_instance_can_be_the_part():
    project, _truth, _report = solved_project()
    project.templates["tpl1"] = ObjectTemplate(id="tpl1", name="other", units="mm")
    first = project.part_instance().id
    second = project.set_part("tpl1")
    assert second.id == first, "designating a new part should reuse the one instance"
    assert sum(1 for o in project.instances if o.is_part) == 1


def test_stage_one_places_the_part_in_every_photo():
    project, _truth, _report = solved_project()
    placement = place_part(project)
    assert placement.ok, f"{placement.stage}: {placement.warnings}"
    assert placement.part_space and placement.pinned
    assert len(placement.images) == len(project.images)
    for entry in placement.images:
        assert entry.positioned, f"{entry.image}: {entry.note}"
        assert entry.n_inside > 0, f"{entry.image} has the part off-frame: {entry.note}"
        assert entry.coverage > 0.2, f"{entry.image} covers only {entry.coverage:.3f}"


def test_stage_one_wireframe_matches_the_truth_projection():
    """No search is involved, so this should be exact to the solve's own error."""
    project, truth, _report = solved_project()
    mesh = load_mesh(part_mesh_path())
    template = project.part_template()
    cam_true = truth["cam"]
    verts = np.asarray(mesh.vertices, dtype=float) * 0.001

    worst = 0.0
    for i, im in enumerate(project.images):
        uv, visible = project_part_wireframe(project, im, mesh, template)
        want = project_points(truth["R"][i], truth["t"][i], cam_true, verts)
        d = np.linalg.norm(uv[visible] - want[visible], axis=1)
        worst = max(worst, float(d.max()))
    assert worst < 6.0, f"the projected wireframe is {worst:.2f} px from the truth"


def test_stage_one_refuses_outside_part_space():
    project, _truth = build_part_project(nominals_on=2)
    solve(project)
    placement = place_part(project)
    assert not placement.ok
    assert placement.stage == "not in part space"
    assert not placement.pinned, "nothing should be pinned when the frame is not the part's"


def test_clearing_the_part_leaves_no_phantom_object():
    project, _truth, _report = solved_project()
    project = Project.from_dict(project.to_dict(), path=project.path)
    assert project.clear_part()
    assert project.part_instance() is None
    assert not project.instances, "a cleared part must not linger as an object at the origin"
    assert "tpl0" in project.templates, "clearing the part keeps the model"


def test_the_part_survives_a_round_trip():
    project, _truth, _report = solved_project()
    again = Project.from_dict(project.to_dict(), path=project.path)
    inst = again.part_instance()
    assert inst is not None and inst.is_part and inst.template == "tpl0"
    assert np.allclose(inst.R, np.eye(3)) and np.allclose(inst.t, 0.0)


# --------------------------------------------------------------------------
# stage two
# --------------------------------------------------------------------------


def _refine_setup(clutter: int = 26, seed: int = 3):
    """A solved project with real rendered photographs written to disk."""
    project, truth, _report = solved_project()
    project = Project.from_dict(project.to_dict(), path=SCRATCH / "project.multispy")
    mesh = load_mesh(part_mesh_path())
    template = project.part_template()
    photos = SCRATCH / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(project.images):
        data = render_photo(
            project, im, truth["R"][i], truth["t"][i], mesh, clutter=clutter, seed=seed + i
        )
        (photos / f"plate_{i:02d}.jpg").write_bytes(data)
    return project, truth, mesh, template


def test_the_edge_detector_finds_the_part():
    project, _truth, _mesh, _template = _refine_setup()
    gray = load_grayscale(project.resolve_path(project.images[0]))
    field = build_edge_field(gray, WIDTH, HEIGHT, project.settings)
    assert field.n_edge_px > 200, field.n_edge_px
    assert 0.0 < field.edge_fraction < 0.2, field.edge_fraction
    assert field.dist.shape[0] == project.settings.edge_orientation_bins


def test_stage_two_is_off_unless_asked():
    project, _truth, mesh, template = _refine_setup()
    assert not project.settings.edge_refine, "edge refinement must not be on by default"
    result = refine_image_pose(project, project.images[0], mesh, template)
    assert not result.accepted and result.stage == "disabled"
    report = refine_images(project, mesh, template, ["img0"])
    assert not report.enabled and report.n_accepted == 0


def test_stage_two_does_not_make_a_good_pose_worse():
    """The one thing it must never do.

    Started from the pose the solve produced, on photographs with a deliberately
    cluttered background, every acceptance gate live. Whatever the optimiser
    thinks it has found, the pose that comes out must not reproject this photo's
    own clicked marks any worse than the pose that went in.
    """
    project, _truth, mesh, template = _refine_setup()
    project.settings.edge_refine = True
    before = {
        im.id: (np.array(im.R), np.array(im.t), mark_reprojection_rms(project, im))
        for im in project.images
    }
    report = refine_images(
        project, mesh, template, [im.id for im in project.images]
    )
    assert report.ok and report.enabled
    tolerance = project.settings.edge_max_reproj_worsening_px
    for result in report.results:
        im = project.image(result.image)
        was_R, was_t, was_rms = before[result.image]
        now = mark_reprojection_rms(project, im)
        assert now is not None and was_rms is not None
        assert now <= was_rms + tolerance + 1e-9, (
            f"{result.image}: marks went from {was_rms:.3f} to {now:.3f} px "
            f"({result.stage}: {result.reason})"
        )
        if not result.accepted:
            assert np.allclose(im.R, was_R) and np.allclose(im.t, was_t), (
                f"{result.image} was rejected but its pose moved anyway"
            )
            assert result.reason, "a rejection has to say why"


def wireframe_truth_error(project, image, mesh, template, truth_R, truth_t) -> float:
    """Median pixel distance from the part's projected wireframe to where the
    ground-truth pose would put it.

    The only honest measure of whether an alignment helped. The chamfer cost
    cannot be used for this: it is the quantity being minimised, and it is the
    quantity with the bias in it.
    """
    cam = intrinsics_for(project, image)
    verts = np.asarray(mesh.vertices, dtype=float) * 0.001
    want = project_points(np.asarray(truth_R), np.asarray(truth_t), cam, verts)
    got = project_points(
        np.asarray(image.R, dtype=float), np.asarray(image.t, dtype=float), cam, verts
    )
    d = np.linalg.norm(got - want, axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d))


def test_the_alignment_is_less_accurate_than_the_solve():
    """The limitation, written down as a test so it cannot be quietly forgotten.

    With every acceptance gate removed, started from the solve's own pose, the
    chamfer alignment moves *away* from ground truth while reporting that it has
    improved the edge fit. This is not a bug to be fixed: the distance transform
    lives on the working pixel grid and cannot resolve better than about half of
    one, while the bundle adjustment on a well-marked project is already better
    than that. It is why stage two is a recovery tool and not a refinement, and
    why the clicked marks rather than the chamfer cost decide.

    If this ever starts failing because the alignment became *more* accurate
    than the solve, that is good news and the gates should be revisited -- but it
    has to be a deliberate change rather than a surprise.
    """
    project, truth, mesh, template = _refine_setup(clutter=0)
    project.settings.edge_refine = True
    project.settings.edge_min_improvement = -1.0
    project.settings.edge_max_reproj_worsening_px = 1e9
    project.settings.edge_max_shift_px = 1e9

    rows = []
    for i, im in enumerate(project.images):
        before = wireframe_truth_error(project, im, mesh, template, truth["R"][i], truth["t"][i])
        result = refine_image_pose(project, im, mesh, template)
        if not result.attempted:
            continue
        after = wireframe_truth_error(project, im, mesh, template, truth["R"][i], truth["t"][i])
        rows.append((im.id, before, after, result))

    assert rows, "nothing was attempted, so nothing was measured"
    assert all(r.median_after < r.median_before for _i, _b, _a, r in rows), (
        "the alignment did not reduce the edge distance it was minimising"
    )
    worse = [iid for iid, before, after, _r in rows if after > before]
    assert len(worse) >= len(rows) - 1, (
        "the alignment is now about as accurate as the solve on a good pose, which it was not "
        f"when the gates were written: {[(i, round(b, 2), round(a, 2)) for i, b, a, _r in rows]}"
    )
    assert all(
        r.mark_rms_after > r.mark_rms_before for _i, _b, _a, r in rows
    ), "the referee did not notice"


def test_stage_two_recovers_a_pose_a_few_pixels_off():
    """The case stage two is actually for, measured against ground truth.

    A camera knocked a few pixels out -- enough that the wireframe is visibly
    beside the part, which is how anyone notices in the first place -- is pulled
    back to within about a pixel and a half, and this photo's own clicked marks
    agree independently. That agreement is the whole argument for shipping this:
    the marks were no part of what was minimised.
    """
    project, truth, mesh, template = _refine_setup(clutter=26)
    project.settings.edge_refine = True

    recovered = 0
    for i in (0, 3, 5):
        im = project.images[i]
        good_R, good_t = np.array(im.R), np.array(im.t)
        im.R, im.t = perturb(good_R, good_t, rot_deg=0.3, shift_m=0.0015, seed=11 + i)
        before = wireframe_truth_error(project, im, mesh, template, truth["R"][i], truth["t"][i])
        assert before > 3.0, f"{im.id}: the perturbation only moved it {before:.2f} px"
        result = refine_image_pose(project, im, mesh, template)
        after = wireframe_truth_error(project, im, mesh, template, truth["R"][i], truth["t"][i])
        assert result.accepted, f"{im.id} not accepted -- {result.stage}: {result.reason}"
        assert after < before / 2.0, (
            f"{im.id}: {before:.2f} px off before, {after:.2f} px after"
        )
        assert result.mark_rms_after < result.mark_rms_before / 2.0, (
            f"{im.id}: the marks do not agree it improved "
            f"({result.mark_rms_before:.2f} -> {result.mark_rms_after:.2f} px)"
        )
        recovered += 1
    assert recovered == 3


def test_an_image_with_no_marks_cannot_be_refined():
    """No referee, no refinement.

    The chamfer cost is biased, so leaving it to decide alone is how a good pose
    gets quietly made worse. A photo with no clicked tie points has nothing to
    adjudicate with, and the honest answer is to refuse.
    """
    project, _truth, mesh, template = _refine_setup(clutter=0)
    project.settings.edge_refine = True
    im = project.images[0]
    was_R, was_t = np.array(im.R), np.array(im.t)
    im.observations = {}
    result = refine_image_pose(project, im, mesh, template)
    assert not result.accepted
    assert result.stage == "no marks to check against", result.stage
    assert np.allclose(im.R, was_R) and np.allclose(im.t, was_t)


def test_stage_two_is_exactly_reversible():
    project, _truth, mesh, template = _refine_setup(clutter=0)
    project.settings.edge_refine = True
    project.settings.edge_min_improvement = -1.0
    im = project.images[0]
    was_R, was_t = np.array(im.R), np.array(im.t)
    result = refine_image_pose(project, im, mesh, template)
    if not result.accepted:
        # nothing to revert, which is itself an acceptable outcome
        assert not im.edge_refined
        return
    assert im.edge_refined and im.R_unrefined is not None
    assert revert_image_pose(im)
    assert np.allclose(im.R, was_R) and np.allclose(im.t, was_t), "an undo must be exact"
    assert not im.edge_refined and im.R_unrefined is None
    assert not revert_image_pose(im), "a second undo has nothing to do"


def test_a_pose_with_nothing_to_align_to_is_refused():
    """A blank photograph has no edges, and the honest answer is to say so."""
    from PIL import Image

    project, _truth, mesh, template = _refine_setup()
    project.settings.edge_refine = True
    blank = SCRATCH / "photos" / "blank.jpg"
    Image.new("L", (WIDTH // 2, HEIGHT // 2), 128).save(blank, quality=90)
    im = project.images[0]
    im.path = str(blank)
    was_R = np.array(im.R)
    result = refine_image_pose(project, im, mesh, template)
    assert not result.accepted, result.stage
    assert result.stage in {"no support", "no improvement"}, result.stage
    assert np.allclose(im.R, was_R)


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_the_solve_reaches_part_space,
        test_the_part_fits_its_own_reconstruction,
        test_the_solve_report_carries_the_fit,
        test_the_reconstruction_lands_on_the_models_surface,
        test_a_wrong_nominal_shows_up_against_the_model,
        test_a_two_view_nominal_is_reported_as_uncheckable,
        test_no_part_is_not_a_failed_fit,
        test_not_in_part_space_refuses_rather_than_guessing,
        test_a_units_mismatch_is_reported_as_a_factor_not_scaled_away,
        test_agreeing_units_say_nothing,
        test_legitimately_different_units_are_flagged_but_not_accused,
        test_the_part_is_pinned_at_the_identity,
        test_only_one_instance_can_be_the_part,
        test_stage_one_places_the_part_in_every_photo,
        test_stage_one_wireframe_matches_the_truth_projection,
        test_stage_one_refuses_outside_part_space,
        test_clearing_the_part_leaves_no_phantom_object,
        test_the_part_survives_a_round_trip,
        test_the_edge_detector_finds_the_part,
        test_stage_two_is_off_unless_asked,
        test_stage_two_does_not_make_a_good_pose_worse,
        test_the_alignment_is_less_accurate_than_the_solve,
        test_stage_two_recovers_a_pose_a_few_pixels_off,
        test_an_image_with_no_marks_cannot_be_refined,
        test_stage_two_is_exactly_reversible,
        test_a_pose_with_nothing_to_align_to_is_refused,
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

    project, _truth, report = solved_project()
    fit = part_fit(project)
    print("\n--- the part against its reconstruction ---")
    print(f"part space: {report.part_space}   reprojection RMS {report.rms_reprojection_px:.3f} px")
    print(f"model declares {fit.units} ({fit.unit_scale:g} m/unit); "
          f"nominals are {fit.nominal_units} ({fit.nominal_scale:g} m/unit); "
          f"ratio {fit.unit_ratio:g}")
    print(f"span: model {fit.model_span:.1f} mm vs nominal'd features "
          f"{fit.nominal_span:.1f} mm (ratio {fit.span_ratio:.3f})")
    print(f"\n{'point':<7}{'views':>6}{'dev mm':>9}{'surface':>9}{'nom-surf':>10}  nominal")
    for row in fit.points:
        print(f"{row.point:<7}{row.n_views:>6}{row.deviation_mag:>9.3f}"
              f"{row.surface_distance:>9.3f}{row.nominal_surface_distance:>10.3f}  "
              f"{np.round(row.nominal, 1).tolist()}")
    print(f"\nRMS {fit.rms_deviation:.3f} mm from the drawing, "
          f"{fit.rms_surface:.3f} mm from the model's surface; "
          f"implied scale {fit.implied_scale:.5f}x")

    placement = place_part(project)
    print("\n--- stage one: the part projected from its known pose ---")
    print(f"{'image':<8}{'in front':>10}{'inside':>8}{'coverage':>10}{'mark RMS':>10}")
    for entry in placement.images:
        rms = "—" if entry.mark_rms_px is None else f"{entry.mark_rms_px:.2f}"
        print(f"{entry.image:<8}{entry.n_in_front:>10}{entry.n_inside:>8}"
              f"{entry.coverage * 100:>9.1f}%{rms:>10}")

    print("\n--- stage two: edge alignment, every gate live ---")
    sub, _t, mesh, template = _refine_setup()
    sub.settings.edge_refine = True
    refine = refine_images(sub, mesh, template, [im.id for im in sub.images])
    print(f"{'image':<8}{'cost':>14}{'wireframe px':>16}{'shift':>8}{'marks':>14}  outcome")
    for r in refine.results:
        marks = (
            "—"
            if r.mark_rms_before is None
            else f"{r.mark_rms_before:.2f}->{(r.mark_rms_after if r.mark_rms_after is not None else r.mark_rms_before):.2f}"
        )
        print(f"{r.image:<8}{r.cost_before:>6.2f}->{r.cost_after:<7.2f}"
              f"{r.median_before:>7.2f}->{r.median_after:<8.2f}"
              f"{r.shift_px:>8.1f}{marks:>14}  {r.stage}")
    for m in refine.messages:
        print(f"  {m}")
    for w in refine.warnings:
        print(f"  ! {w}")

    print(f"\nscratch files in {SCRATCH}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
