"""Add automatically found tie points to a solved project.

    .venv\\Scripts\\python.exe tools\\harvest.py projects/rearmod3
    .venv\\Scripts\\python.exe tools\\harvest.py projects/rearmod3 --dry-run
    .venv\\Scripts\\python.exe tools\\harvest.py projects/rearmod3 --out scratch/r3.multispy
    .venv\\Scripts\\python.exe tools\\harvest.py projects/rearmod3 --undo

The project must already be solved: the harvest reads the poses it finds and
matches against them, so running it on an unsolved project finds nothing.  That
is the correct behaviour rather than a failure -- see :mod:`multispy.harvest`
for why known poses are the whole method and not an accelerant.

``--out`` writes a copy and leaves the original alone, which is what to use
while measuring.  Writing in place is fine for a project you then open in the
app; it is a bad idea for one the server already has open, because the server
saves the whole project back on every edit and will overwrite this.

By default the harvest is followed by a solve, so the numbers printed are the
ones the project would show.  Compare them with ``tools/crossval.py`` before
believing them: in-solve RMS falls when evidence is added whether or not the
answer improved, which is the entire reason that tool exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multispy import solve as S  # noqa: E402
from multispy.harvest import (  # noqa: E402
    HarvestSettings,
    add_tracks_to_project,
    harvest,
    remove_harvested,
)
from multispy.project import Project  # noqa: E402


def resolve(path: Path) -> Path:
    if path.is_dir():
        found = sorted(path.glob("*.multispy"))
        if not found:
            raise SystemExit(f"no .multispy file in {path}")
        return found[0]
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("project", type=Path, help="project folder or .multispy file")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the result here instead of over the input")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what was found and write nothing")
    ap.add_argument("--undo", action="store_true",
                    help="remove previously harvested points and stop")
    ap.add_argument("--no-solve", action="store_true",
                    help="skip the solve after harvesting")
    ap.add_argument("--quiet", action="store_true", help="no per-pair progress")

    d = HarvestSettings()
    ap.add_argument("--max-points", type=int, default=d.max_points)
    ap.add_argument("--min-views", type=int, default=d.min_views)
    ap.add_argument("--epipolar", type=float, default=d.epipolar_px,
                    help="epipolar band half-width in px (default: %(default)s)")
    ap.add_argument("--min-angle", type=float, default=d.min_triangulation_deg,
                    help="minimum triangulation angle in degrees")
    ap.add_argument("--ncc-min", type=float, default=d.ncc_min)
    ap.add_argument("--uniqueness", type=float, default=d.uniqueness,
                    help="Lowe ratio against the runner-up; lower is stricter")
    ap.add_argument("--corners", type=int, default=d.max_corners_per_image)
    ap.add_argument("--patch", type=int, default=d.patch)
    ap.add_argument("--max-reproj", type=float, default=d.max_track_reproj_px)
    ap.add_argument("--per-cell", type=int, default=d.max_per_cell)
    ap.add_argument("--uncertain", action="store_true",
                    help="flag harvested points uncertain (weighted looser)")
    args = ap.parse_args()

    path = resolve(args.project)
    project = Project.load(str(path))

    if args.undo:
        n = remove_harvested(project)
        print(f"removed {n} harvested point(s)")
        if not args.dry_run:
            out = project.save(args.out or path)
            print(f"wrote {out}")
        return 0

    s = HarvestSettings(
        max_points=args.max_points,
        min_views=args.min_views,
        epipolar_px=args.epipolar,
        min_triangulation_deg=args.min_angle,
        ncc_min=args.ncc_min,
        uniqueness=args.uniqueness,
        max_corners_per_image=args.corners,
        patch=args.patch,
        max_track_reproj_px=args.max_reproj,
        max_per_cell=args.per_cell,
        uncertain=args.uncertain,
    )

    before = len(project.points)
    report = harvest(project, s, progress=None if args.quiet else lambda m: print(f"  {m}"))
    print()
    print(report.summary())
    for m in report.messages:
        print(f"  - {m}")
    for w in report.warnings:
        print(f"  ! {w}")

    if report.tracks:
        views = {}
        for tr in report.tracks:
            views[tr.n_views] = views.get(tr.n_views, 0) + 1
        spread = ", ".join(f"{k} views: {v}" for k, v in sorted(views.items()))
        worst = max(tr.reproj_px for tr in report.tracks)
        print(f"  kept by view count -- {spread}")
        print(f"  triangulation residual: worst {worst:.2f} px over all kept tracks")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if not report.tracks:
        return 1

    added = add_tracks_to_project(project, report.tracks, s)
    print(f"\n{before} tie point(s) -> {len(project.points)} ({len(added)} harvested)")

    if not args.no_solve:
        rep = S.solve(project, run_bundle=True)
        print(f"solve: ok={rep.ok}  reprojection RMS {rep.rms_reprojection_px:.2f} px  "
              f"control lines {rep.rms_line_px:.2f} px")
        for gid, f in rep.focals.items():
            print(f"  camera group {gid}: focal {f:.1f} px")
        for w in rep.warnings[:4]:
            print(f"  ! {w}")

    out = project.save(args.out or path)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
