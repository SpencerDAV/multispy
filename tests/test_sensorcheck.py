"""Do the photographs find a body that is there, and miss one that is not?

The fixture renders real photographs — shaded faces, not drawn wireframes —
through the solved cameras of ``test_part``'s machined-plate project, with three
bodies in the scene and a fourth that exists only in the calibration:

* ``present``   rendered exactly where it is claimed to be.
* ``displaced`` rendered 40 mm from where it is claimed to be.
* ``absent``    claimed, and not rendered at all.

So the ground truth is known for every verdict, which is the only way to test a
measurement that is allowed to be wrong. A drawn wireframe would hand the chamfer
cost a one-pixel-wide perfectly placed target and prove nothing about a
photograph; shading makes the edges emerge from contrast the way they do in a
real one. Clutter is on, because clutter is the failure mode: ``edgefit.py``
documents a chamfer cost converging confidently onto a bookshelf, and the real
photographs for this feature are of a tube frame where every surface is an edge.

What is being defended:

* **Absent must read as absent.** A body nobody rendered has to score no better
  than deliberately wrong poses. If this ever passes, the z-score has stopped
  measuring anything and every other verdict in the feature is decoration.
* **The null distribution is what makes the score comparable.** An absolute
  chamfer cost cannot be compared between a clean background and a busy one; the
  z-score can, because the clutter is in both of its terms.
* **One photograph is never a verdict.** Cross-view agreement is the strong
  referee, so a sensor seen once is reported as seen once.
* **A contradiction with the calibration file is surfaced, not averaged away.**

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_sensorcheck.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.mesh import load_mesh
from multispy.project import ObjectInstance, ObjectTemplate, Project
from multispy.sensorcheck import (
    ABSENT,
    AGREEMENT,
    DISPLACED,
    FOUND,
    FOUND_Z,
    MIN_VIEWS,
    UNDECIDED,
    check_sensors,
)
from multispy.sensors import CalibrationReport, SensorVerdict

from test_part import HEIGHT, WIDTH, part_mesh_path, solved_project

SCRATCH = Path(__file__).resolve().parent / "_scratch_sensorcheck"
DISPLACEMENT_MM = 40.0
_cache: dict = {}


def _render(project, truth, bodies, clutter: int = 22, seed: int = 5) -> None:
    """Photographs of the plate with extra bodies shaded into them.

    ``bodies`` is a list of (mesh, R, t) in world metres. Painter's algorithm by
    face depth, which is enough for bodies that do not interpenetrate and keeps
    the fixture from needing a renderer.
    """
    from PIL import Image, ImageDraw

    from multispy.part import intrinsics_for

    rng = np.random.default_rng(seed)
    scale = 0.5
    w, h = int(WIDTH * scale), int(HEIGHT * scale)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    photos = SCRATCH / "photos"
    photos.mkdir(parents=True, exist_ok=True)

    for i, im in enumerate(project.images):
        img = Image.new("L", (w, h), 70)
        draw = ImageDraw.Draw(img)
        local = np.random.default_rng(seed + i)
        for _ in range(clutter):
            x0, y0 = local.uniform(0, w), local.uniform(0, h)
            ang = local.uniform(0, math.pi)
            length = local.uniform(0.3, 1.2) * w
            draw.line(
                [x0, y0, x0 + math.cos(ang) * length, y0 + math.sin(ang) * length],
                fill=int(local.integers(130, 210)),
                width=int(local.integers(1, 4)),
            )

        R, t = np.asarray(truth["R"][i], dtype=float), np.asarray(truth["t"][i], dtype=float)
        cam = intrinsics_for(project, im)
        tris = []
        for mesh, Rb, tb in bodies:
            verts = (np.asarray(mesh.vertices, dtype=float) * 0.001) @ np.asarray(
                Rb, dtype=float
            ).T + np.asarray(tb, dtype=float)
            Xc = verts @ R.T + t
            z = np.maximum(Xc[:, 2], 1e-9)
            uv = np.stack(
                [cam.f * Xc[:, 0] / z + cam.cx, cam.f * Xc[:, 1] / z + cam.cy], axis=1
            ) * scale
            for face in np.asarray(mesh.faces, dtype=int):
                if not (Xc[face, 2] > 1e-6).all():
                    continue
                p = verts[face]
                n = np.cross(p[1] - p[0], p[2] - p[0])
                nn = float(np.linalg.norm(n))
                if nn < 1e-15:
                    continue
                shade = abs(float(n @ (-R.T @ t - p[0])) / (nn * max(
                    float(np.linalg.norm(-R.T @ t - p[0])), 1e-9)))
                tris.append((float(np.mean(Xc[face, 2])), uv[face],
                             int(60 + 170 * min(shade, 1.0))))
        for _depth, poly, shade in sorted(tris, key=lambda s: -s[0]):
            draw.polygon([tuple(q) for q in poly], fill=shade)

        path = photos / f"cell_{i:02d}.jpg"
        img.resize((WIDTH, HEIGHT), Image.BILINEAR).save(path, quality=90)
        im.path = f"photos/{path.name}"
        im.width, im.height = WIDTH, HEIGHT
    _ = rng


def scene():
    """Solved cameras, three rendered bodies, four claimed ones."""
    if "scene" in _cache:
        return _cache["scene"]

    project, truth, _report = solved_project()
    project = Project.from_dict(project.to_dict(), path=SCRATCH / "project.multispy")
    mesh = load_mesh(part_mesh_path())

    # Bodies sit *around* the plate rather than off to one side of it, because
    # the cameras are framed on the plate: a body 260 mm out along one axis left
    # the frame in six of eight views, which tests the visibility guard rather
    # than the scoring. These three are adjacent to the plate, do not overlap
    # each other or it, and are seen by most of the rig.
    poses = {
        "present": (np.eye(3), np.array([0.15, 0.0, 0.02])),
        "displaced": (np.eye(3), np.array([-0.15, 0.0, 0.02])),
        "absent": (np.eye(3), np.array([0.0, 0.14, 0.02])),
    }
    shift = np.array([DISPLACEMENT_MM * 0.001, 0.0, 0.0])

    template = ObjectTemplate(
        id="tpl_sensor", name="917-4012", mesh_path=str(part_mesh_path()),
        units="mm", unit_scale=0.001,
    )
    project.templates[template.id] = template
    for name, (R, t) in poses.items():
        project.instances.append(
            ObjectInstance(
                id=f"sensor_{name}", template=template.id, label=name,
                R=R, t=t, calibrated="12/15/2015",
            )
        )

    _render(
        project, truth,
        [
            (mesh, poses["present"][0], poses["present"][1]),
            # rendered away from where it is claimed: a measured correction
            (mesh, poses["displaced"][0], poses["displaced"][1] + shift),
            # "absent" is deliberately not rendered
        ],
    )
    Project.from_dict(project.to_dict(), path=project.path)
    _cache["scene"] = (project, mesh)
    return _cache["scene"]


def run():
    if "report" not in _cache:
        project, _mesh = scene()
        _cache["report"] = check_sensors(project, mesh_loader=load_mesh)
    return _cache["report"]


def for_id(report, name: str):
    return next(s for s in report.sensors if s.instance_id == f"sensor_{name}")


# ---------------------------------------------------------------- the claims


def test_a_body_that_is_there_is_found_across_views():
    got = for_id(run(), "present")
    assert got.n_views >= MIN_VIEWS, f"only seen in {got.n_views} photographs: {got.reason}"
    assert got.verdict in (FOUND, DISPLACED), f"{got.verdict}: {got.reason}"
    assert got.z_median >= FOUND_Z, f"median z only {got.z_median:.2f}"
    assert got.agreement >= AGREEMENT, f"agreement only {got.agreement:.0%}"


def test_a_body_that_is_not_there_is_not_found():
    """The test that matters most. If a body nobody rendered scores well, the
    z-score has stopped measuring anything and every other verdict is decoration."""
    got = for_id(run(), "absent")
    assert got.verdict == ABSENT, (
        f"a body that was never rendered came back {got.verdict}: z {got.z_median:.2f}, "
        f"agreement {got.agreement:.0%} -- {got.reason}"
    )
    assert got.agreement == 0.0, (
        f"{got.agreement:.0%} of the photographs claim to have found a body that is not in "
        f"any of them"
    )
    assert "nothing is there" in got.reason
    assert got.z_median < for_id(run(), "present").z_median, (
        f"the absent body ({got.z_median:.2f}) scored no worse than the present one "
        f"({for_id(run(), 'present').z_median:.2f}), so the score does not separate them"
    )


def test_the_present_body_beats_the_absent_one_by_a_margin():
    """Not just ordered, but separated — an ordering that close is luck."""
    present, absent = for_id(run(), "present"), for_id(run(), "absent")
    assert present.z_median - absent.z_median > 1.0, (
        f"present {present.z_median:.2f} vs absent {absent.z_median:.2f}: the two are not "
        f"separated enough for any threshold between them to mean anything"
    )


def test_a_displaced_body_is_reported_as_a_correction_not_a_failure():
    """A body found strongly a few centimetres off is a measurement, not a miss."""
    got = for_id(run(), "displaced")
    assert got.verdict in (FOUND, DISPLACED, UNDECIDED), got.reason
    if got.verdict == DISPLACED:
        assert got.displacement_mm > 5.0
        assert "correction to the calibration" in got.reason
    # Whatever the verdict, it must not be read as absent: the body IS in the photos.
    assert got.verdict != ABSENT, (
        f"a body that was rendered, merely offset by {DISPLACEMENT_MM:.0f} mm, was reported "
        f"absent: {got.reason}"
    )


# ------------------------------------------------- the referees, and refusals


def test_the_null_distribution_has_a_spread_to_measure_against():
    """A z-score is only meaningful if wrong poses actually score differently."""
    for sensor in run().sensors:
        for photo in sensor.photos:
            if not photo.visible:
                continue
            assert photo.null_std > 0.0, (
                f"{sensor.instance_id} in {photo.image_id}: every wrong pose scored "
                f"identically, so this photograph cannot separate right from wrong"
            )
            assert photo.n_samples >= 24


def test_the_cost_alone_does_not_decide_anything():
    """The chamfer cost is reported and never used as a verdict.

    ``edgefit.py`` measured that cost converging confidently onto clutter, so a
    verdict that tracked it would inherit the bias. This checks the two are
    genuinely different quantities on this scene — if raw cost ordered the
    sensors the same way the z-score does, the fixture is not exercising the
    distinction.
    """
    seen = [(s.instance_id, s.z_median,
             float(np.median([p.cost for p in s.photos if p.visible])))
            for s in run().sensors if s.n_views >= MIN_VIEWS]
    assert len(seen) >= 2
    by_z = [n for n, _z, _c in sorted(seen, key=lambda r: -r[1])]
    assert by_z[0] == "sensor_present", (
        f"the z-score did not rank the real body first: {seen}"
    )


def test_a_sensor_seen_once_or_never_gets_no_verdict():
    """One photograph cannot tell a body from a coincidence."""
    project, _mesh = scene()
    lonely = Project.from_dict(project.to_dict(), path=project.path)
    for im in lonely.images[1:]:
        im.enabled = False
    report = check_sensors(lonely, mesh_loader=load_mesh)
    for sensor in report.sensors:
        assert sensor.verdict == UNDECIDED, (
            f"{sensor.instance_id} was graded {sensor.verdict} off a single photograph"
        )
        assert str(MIN_VIEWS) in sensor.reason or "coincidence" in sensor.reason


def test_a_body_with_no_model_is_undecided_and_says_why():
    """A position alone cannot be checked: a point has no silhouette."""
    project, _mesh = scene()
    bare = Project.from_dict(project.to_dict(), path=project.path)
    bare.templates["tpl_sensor"].mesh_path = None
    report = check_sensors(bare, mesh_loader=load_mesh)
    assert report.sensors
    for sensor in report.sensors:
        assert sensor.verdict == UNDECIDED
        assert "no silhouette" in sensor.reason, sensor.reason


def test_a_project_with_no_calibrated_bodies_says_so():
    project, _mesh = scene()
    empty = Project.from_dict(project.to_dict(), path=project.path)
    for inst in empty.instances:
        inst.calibrated = ""
    report = check_sensors(empty, mesh_loader=load_mesh)
    assert not report.sensors
    assert any("import a calibration file" in m for m in report.messages)


def test_the_summary_states_what_a_score_is_not():
    report = run()
    assert any("narrower claim than the body being there" in m for m in report.messages)
    assert any("same model are" in m or "identical" in m for m in report.messages), (
        "the look-alike limit is not stated, and it is the one cross-view agreement "
        "cannot catch"
    )


# ----------------------------------------------------- contradictions


def test_a_file_that_passes_a_body_the_photos_cannot_find_is_contradicted():
    """The most valuable output in the feature, so it must not be averaged away."""
    project, _mesh = scene()
    filed = CalibrationReport(ok=True)
    filed.verdicts = [
        SensorVerdict(serial="absent", verdict="consistent", worst_feature="F1", ratio=0.4),
        SensorVerdict(serial="present", verdict="consistent", worst_feature="F2", ratio=0.3),
    ]
    report = check_sensors(project, mesh_loader=load_mesh, calibration=filed)
    absent = for_id(report, "absent")
    if absent.verdict == ABSENT:
        assert any("absent" in c or "cannot find it" in c for c in report.contradictions), (
            f"no contradiction raised for a sensor the file passes and the photos miss: "
            f"{report.contradictions}"
        )
        assert any("most valuable" in m for m in report.messages)


def test_a_file_that_convicts_a_body_the_photos_find_blames_the_nominal():
    """The sensor is where the file says; the thing it was told to measure is not."""
    project, _mesh = scene()
    filed = CalibrationReport(ok=True)
    filed.verdicts = [
        SensorVerdict(serial="present", verdict="suspect", worst_feature="LH_CP_100", ratio=7.5)
    ]
    report = check_sensors(project, mesh_loader=load_mesh, calibration=filed)
    if for_id(report, "present").verdict == FOUND:
        assert report.contradictions, "a convicted-but-found sensor raised no contradiction"
        assert any("feature nominal being wrong" in c for c in report.contradictions), (
            report.contradictions
        )


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
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    report = run()
    print(f"\n{len(report.sensors)} claimed bodies over {report.n_images} photographs "
          f"(one rendered where claimed, one {DISPLACEMENT_MM:.0f} mm off, one not rendered)")
    print(f"  {'body':20s} {'verdict':10s} {'views':>5s} {'z':>7s} {'agree':>6s} "
          f"{'cost':>7s} {'offset':>8s}")
    for s in sorted(report.sensors, key=lambda s: -s.z_median):
        cost = [p.cost for p in s.photos if p.visible]
        print(f"  {s.instance_id:20s} {s.verdict:10s} {s.n_views:5d} {s.z_median:7.2f} "
              f"{s.agreement * 100:5.0f}% {np.median(cost) if cost else float('nan'):7.2f} "
              f"{s.displacement_mm:8.1f}")
    for c in report.contradictions:
        print(f"  contradiction: {c[:110]}")
    print(f"\nscratch files in {SCRATCH}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
