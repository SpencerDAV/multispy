"""Render the camera-array scene as photos, and write a demo project for it.

This is the known-object workflow end to end: a rig frame with five identical
camera bodies bolted to its top, photographed by hand from six viewpoints. The
bodies are placed by fitting their CAD model to anchors marked on their corners,
which is what makes the array's layout come out in millimetres with no tape
measure anywhere in the process.

Two flavours:

* default -- fully marked, so you can press Solve and compare against the truth
  printed at the end.
* ``--blank`` -- the same photos, the same model, the same five objects and
  their anchors, but no marks. This is the one to practise on: the model and its
  anchor names are set up, so the job is purely to click each body's corners.

    .venv\\Scripts\\python.exe demo\\make_rig_demo.py
    .venv\\Scripts\\python.exe demo\\make_rig_demo.py --blank --out demo_rig_blank
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multispy.geometry import Intrinsics, project as project_points  # noqa: E402
from tests.test_rig import (  # noqa: E402
    BODY,
    BODY_CORNERS,
    CORNER_NAMES,
    FRAME,
    HEIGHT,
    N_CAMERAS,
    SPACING,
    TOE_IN_DEG,
    TRUE_FOCAL,
    UNIT_SCALE,
    WIDTH,
    body_pose,
    build_rig_project,
    viewpoints,
)

LIGHT = np.array([0.35, -0.72, 0.6])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# the 6 faces of a box as corner-index quads, for a corner array ordered
# [sx][sy][sz] with each index running 0 then 1
QUADS = [
    ((0, 1, 3, 2), (-1, 0, 0)),
    ((4, 6, 7, 5), (1, 0, 0)),
    ((0, 4, 5, 1), (0, -1, 0)),
    ((2, 3, 7, 6), (0, 1, 0)),
    ((0, 2, 6, 4), (0, 0, -1)),
    ((1, 5, 7, 3), (0, 0, 1)),
]

FRAME_TINT = (150, 152, 158)
BODY_TINT = (58, 60, 66)
LENS_TINT = (24, 26, 30)


def frame_corners() -> np.ndarray:
    return np.array(
        [[sx, sy, sz] for sx in (0.0, 1.0) for sy in (0.0, 1.0) for sz in (0.0, 1.0)]
    ) * FRAME


def body_corners_world(index: int) -> np.ndarray:
    R, t = body_pose(index)
    return (BODY_CORNERS * UNIT_SCALE) @ R.T + t


def collect_faces(center: np.ndarray) -> list[tuple]:
    """Every front-facing quad in the scene, with its tint and local axes."""
    out = []

    def add(corners: np.ndarray, tint, rotation: np.ndarray | None, tag: str) -> None:
        for idx, normal in QUADS:
            quad = corners[list(idx)]
            n = np.array(normal, dtype=float)
            if rotation is not None:
                n = rotation @ n
            centroid = quad.mean(axis=0)
            if n @ (center - centroid) <= 0.0:
                continue
            out.append((quad, tint, n, tag))

    add(frame_corners(), FRAME_TINT, None, "frame")
    for i in range(N_CAMERAS):
        R, _t = body_pose(i)
        add(body_corners_world(i), BODY_TINT, R, f"body{i}")
    return out


def render_view(R: np.ndarray, t: np.ndarray, cam: Intrinsics) -> Image.Image:
    """Flat-shaded boxes over a studio floor, painted back to front."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (158, 166, 178))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(HEIGHT * 0.46), WIDTH, HEIGHT], fill=(104, 102, 100))

    center = -R.T @ t
    faces = collect_faces(center)
    ordered = []
    for quad, tint, normal, tag in faces:
        depth = float((R @ quad.mean(axis=0) + t)[2])
        ordered.append((depth, quad, tint, normal, tag))
    ordered.sort(key=lambda e: -e[0])

    for _depth, quad, tint, normal, tag in ordered:
        uv = project_points(R, t, cam, quad)
        if not np.all(np.isfinite(uv)):
            continue
        shade = 0.40 + 0.60 * max(0.0, float(normal @ LIGHT))
        fill = tuple(int(c * shade) for c in tint)
        poly = [tuple(p) for p in uv]
        draw.polygon(poly, fill=fill)

        # a crisp dark edge on every face: the corners have to be clickable, and
        # a flat-shaded box with no edges is guesswork to mark
        edge = tuple(int(c * shade * 0.45) for c in tint)
        draw.line(poly + [poly[0]], fill=edge, width=4 if tag == "frame" else 3)

        # the lens, so the front of each body is unmistakable
        if tag.startswith("body") and float(normal @ LIGHT) > -2.0:
            index = int(tag[4:])
            R_obj, t_obj = body_pose(index)
            if np.allclose(R_obj @ np.array([0.0, 0.0, 1.0]), normal, atol=1e-6):
                radius = BODY[1] * 0.3 * UNIT_SCALE
                ring = []
                for k in range(16):
                    a = 2.0 * math.pi * k / 16
                    local = np.array([math.cos(a) * radius, math.sin(a) * radius, 0.0])
                    ring.append(R_obj @ (local / UNIT_SCALE * UNIT_SCALE) + t_obj)
                ruv = project_points(R, t, cam, np.array(ring))
                if np.all(np.isfinite(ruv)):
                    draw.polygon([tuple(p) for p in ruv], fill=LENS_TINT)

    return img


def save_with_exif(img: Image.Image, path: Path) -> None:
    """Write the photo with a 35mm-equivalent focal length, rounded the way a
    real camera records it."""
    equivalent_mm = TRUE_FOCAL * 36.0 / WIDTH
    data = Image.Exif()
    data[0x010F] = "MultiSpy"
    data[0x0110] = "Synthetic Camera"
    data[0x8769] = {
        0x920A: round(equivalent_mm, 1),
        0xA405: int(round(equivalent_mm)),
        0xA002: WIDTH,
        0xA003: HEIGHT,
    }
    img.save(path, quality=90, exif=data)


def write_body_obj(path: Path) -> None:
    """The camera body as an OBJ in millimetres, matching the template's local
    frame: +x right, +y down, +z along the optical axis, origin at the lens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MultiSpy demo: one array camera body, millimetres\n"]
    lines += [f"v {x:g} {y:g} {z:g}\n" for x, y, z in BODY_CORNERS]
    for idx, _normal in QUADS:
        lines.append("f " + " ".join(str(i + 1) for i in idx) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_rig", help="output folder, relative to the repo root")
    ap.add_argument("--blank", action="store_true", help="strip the marks for practice")
    args = ap.parse_args()

    out_dir = (ROOT / args.out).resolve()
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)

    project, truth = build_rig_project()
    cam = Intrinsics(f=TRUE_FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)

    write_body_obj(out_dir / "models" / "array_camera_body.obj")
    project.templates["tpl0"].mesh_path = "models/array_camera_body.obj"
    project.templates["tpl0"].name = "array camera body"
    project.templates["tpl0"].optical.fov_deg = 62.0

    for image, (R, t) in zip(project.images, zip(truth["R"], truth["t"])):
        path = out_dir / image.path
        path.parent.mkdir(parents=True, exist_ok=True)
        save_with_exif(render_view(R, t, cam), path)
        print(f"  rendered {path.relative_to(ROOT)}")

    if args.blank:
        for image in project.images:
            image.axis_lines.clear()
            image.observations.clear()
            image.anchor_obs.clear()
            image.R = None
            image.t = None
        project.points.clear()
        for inst in project.instances:
            inst.R, inst.t = None, None
            inst.locked = False
        project.groups["cam0"].f = None
        project.name = "demo_rig_blank"

    project_path = out_dir / "project.multispy"
    project.save(project_path)

    print(f"\nwrote {project_path}")
    print(f"  model: models/array_camera_body.obj, {BODY[0]:g} x {BODY[1]:g} x {BODY[2]:g} mm")
    print(f"  {N_CAMERAS} camera bodies, {SPACING * 1000:g} mm apart, toed in {TOE_IN_DEG:g} deg each")
    print(f"  frame {FRAME[0]:g} x {FRAME[1]:g} x {FRAME[2]:g} m, true focal {TRUE_FOCAL:.0f} px")
    if args.blank:
        print(
            "\nThe model, its 8 anchors and the 5 objects are set up; the marks are not.\n"
            "  1. Lines mode: trace the frame's edges, 2-3 per axis (X along its length,\n"
            "     Z vertical).\n"
            "  2. Objects mode (O): pick a body in the Known objects panel and click its\n"
            "     corners. The next anchor follows automatically.\n"
            "  3. Solve, then press D for the 3D view and check the layout."
        )
    print(f"\nopen it with:\n  .venv\\Scripts\\python.exe -m multispy \"{project_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
