"""Render the synthetic test scene as actual photos, and write a demo project.

Two flavours:

* default -- the project arrives fully marked, so you can press Solve and see a
  known-good result immediately. Useful for checking an install.
* ``--blank`` -- the same photos with no marks at all. This is the one to
  practise the workflow on, because the geometry is known: the box is
  10 x 6 x 4 units, world axes along its edges.

    .venv\\Scripts\\python.exe demo\\make_demo.py
    .venv\\Scripts\\python.exe demo\\make_demo.py --blank --out demo_blank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multispy.geometry import Intrinsics, project as project_points  # noqa: E402
from tests.test_synthetic import BOX, HEIGHT, TRUE_FOCAL, WIDTH, build_project  # noqa: E402

LIGHT = np.array([0.4, -0.75, 0.53])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# the 6 faces as corner-index quads of the unit cube, with outward normals
CUBE = np.array(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float
)
FACES = [
    ((0, 1, 3, 2), (-1, 0, 0)),
    ((4, 6, 7, 5), (1, 0, 0)),
    ((0, 4, 5, 1), (0, -1, 0)),
    ((2, 3, 7, 6), (0, 1, 0)),
    ((0, 2, 6, 4), (0, 0, -1)),
    ((1, 5, 7, 3), (0, 0, 1)),
]
BASE_TINT = (196, 188, 176)


def face_quad(idx: tuple[int, ...]) -> np.ndarray:
    return CUBE[list(idx)] * BOX


def render_view(R: np.ndarray, t: np.ndarray, cam: Intrinsics) -> Image.Image:
    """Flat-shaded box with a window grid, painted back to front."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (124, 146, 170))
    draw = ImageDraw.Draw(img)

    # ground
    draw.rectangle([0, int(HEIGHT * 0.52), WIDTH, HEIGHT], fill=(96, 92, 86))

    center = -R.T @ t
    ordered = []
    for idx, normal in FACES:
        quad = face_quad(idx)
        normal = np.array(normal, dtype=float)
        centroid = quad.mean(axis=0)
        if normal @ (center - centroid) <= 0:  # back face
            continue
        depth = float((R @ centroid + t)[2])
        ordered.append((depth, idx, normal, quad))
    ordered.sort(reverse=True)

    for _depth, idx, normal, quad in ordered:
        uv = project_points(R, t, cam, quad)
        if not np.all(np.isfinite(uv)):
            continue
        shade = 0.42 + 0.58 * max(0.0, float(normal @ LIGHT))
        fill = tuple(int(c * shade) for c in BASE_TINT)
        draw.polygon([tuple(p) for p in uv], fill=fill)

        # window grid, giving real features to click and real edges to trace
        a, b, c, d = quad
        e1, e2 = b - a, d - a
        n1 = max(2, int(round(np.linalg.norm(e1) / 1.6)))
        n2 = max(2, int(round(np.linalg.norm(e2) / 1.6)))
        pane = tuple(int(c0 * shade * 0.52) for c0 in (120, 140, 165))
        for i in range(n1):
            for j in range(n2):
                u0, u1 = (i + 0.28) / n1, (i + 0.72) / n1
                v0, v1 = (j + 0.28) / n2, (j + 0.72) / n2
                corners = np.array([
                    a + e1 * u0 + e2 * v0,
                    a + e1 * u1 + e2 * v0,
                    a + e1 * u1 + e2 * v1,
                    a + e1 * u0 + e2 * v1,
                ])
                puv = project_points(R, t, cam, corners)
                if np.all(np.isfinite(puv)):
                    draw.polygon([tuple(p) for p in puv], fill=pane)

        edge = tuple(int(c0 * shade * 0.35) for c0 in BASE_TINT)
        draw.line([tuple(p) for p in uv] + [tuple(uv[0])], fill=edge, width=5)

    return img


def save_with_exif(img: Image.Image, path: Path) -> None:
    """Write the photo with the focal length recorded the way a real camera does.

    Only the 35mm equivalent, rounded to a whole millimetre, with no sensor-size
    tags -- which is what most phones and drones actually record. The rounding
    is deliberate: 3200 px on a 4000 px frame is 28.8 mm, stored as 29, so
    reading it back gives 3222 px. That ~0.7% is the honest accuracy of the EXIF
    route, and the demo should show it rather than hide it.
    """
    equivalent_mm = TRUE_FOCAL * 36.0 / WIDTH
    sub = {
        0x920A: round(equivalent_mm, 1),        # FocalLength
        0xA405: int(round(equivalent_mm)),      # FocalLengthIn35mmFilm
        0xA002: WIDTH,                          # ExifImageWidth
        0xA003: HEIGHT,                         # ExifImageHeight
    }
    data = Image.Exif()
    data[0x010F] = "MultiSpy"                   # Make
    data[0x0110] = "Synthetic Camera"           # Model
    data[0x8769] = sub                          # Exif sub-IFD
    img.save(path, quality=90, exif=data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo", help="output folder, relative to the repo root")
    ap.add_argument("--blank", action="store_true", help="strip all marks for practice")
    args = ap.parse_args()

    out_dir = (ROOT / args.out).resolve()
    photos = out_dir / "photos"
    photos.mkdir(parents=True, exist_ok=True)

    project, truth = build_project()
    cam = Intrinsics(f=TRUE_FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)

    for image, R, t in zip(project.images, truth["R"], truth["t"]):
        path = out_dir / image.path
        path.parent.mkdir(parents=True, exist_ok=True)
        save_with_exif(render_view(R, t, cam), path)
        print(f"  rendered {path.relative_to(ROOT)}")

    if args.blank:
        for image in project.images:
            image.axis_lines.clear()
            image.observations.clear()
            image.R = None
            image.t = None
        project.points.clear()
        project.constraints.clear()
        project.groups["cam0"].f = None
        project.name = "demo_blank"

    project_path = out_dir / "project.multispy"
    project.save(project_path)
    print(f"\nwrote {project_path}")
    print(f"true focal {TRUE_FOCAL:.0f} px, box {BOX[0]:g} x {BOX[1]:g} x {BOX[2]:g} units")
    print(f"\nopen it with:\n  .venv\\Scripts\\python.exe -m multispy \"{project_path}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
