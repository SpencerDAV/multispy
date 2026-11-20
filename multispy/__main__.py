"""Command line entry point.

    python -m multispy <folder-of-photos>      start a project and open the UI
    python -m multispy <project.multispy>      reopen a project
    python -m multispy solve <project>         headless solve, print the report
    python -m multispy export <project> [dir]  headless export
    python -m multispy rig <project>           print the solved object layout
    python -m multispy exif <project>          what the photos claim about focal length
    python -m multispy regroup <project>      split the camera groups by lens and zoom
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import exif as exif_focal
from . import export as exporters
from .project import Project
from .server import DEFAULT_PROJECTS_ROOT, default_project, load_or_create, serve
from .solve import solve as run_solve


def _print_report(report) -> None:
    print(f"stage: {report.stage}   ok: {report.ok}")
    for m in report.messages:
        print(f"  - {m}")
    for w in report.warnings:
        print(f"  ! {w}")
    if not report.ok:
        return
    print(
        f"\n{report.n_images_positioned} image(s) positioned, "
        f"{report.n_points_triangulated} tie point(s) triangulated, "
        f"{report.n_observations} observation(s)"
    )
    for gid, f in report.focals.items():
        print(f"  camera group {gid}: focal {f:.1f} px")
    print(f"  reprojection RMS {report.rms_reprojection_px:.2f} px")
    print(f"  control-line RMS {report.rms_line_px:.2f} px")
    print(f"  scale: {report.scale_source}")
    print(f"\n{'image':<12}{'obs':>5}{'rms px':>9}{'max px':>9}{'lines px':>10}  notes")
    for iid, rep in sorted(report.images.items(), key=lambda kv: -kv[1].rms_px):
        note = rep.notes[0] if rep.notes else ""
        print(
            f"{iid:<12}{rep.n_observations:5d}{rep.rms_px:9.2f}{rep.max_px:9.2f}"
            f"{rep.line_rms_px:10.2f}  {note}"
        )

    if report.instances:
        print(
            f"\n{'object':<14}{'anchors':>9}{'rms px':>9}{'shape':>11}{'size':>8}  notes"
        )
        for oid, rep in sorted(report.instances.items()):
            note = rep.notes[0] if rep.notes else ("" if rep.placed else "not placed")
            shape = f"{rep.shape_rms * 1000:7.2f}mm" if rep.placed else "       --"
            size = f"{rep.implied_scale:8.4f}" if rep.placed else "      --"
            print(
                f"{(rep.label or oid):<14}{rep.n_anchors_used:4d}/{rep.n_anchors_marked:<4}"
                f"{rep.rms_px:9.2f}{shape}{size}  {note[:48]}"
            )
    _print_constraints(report)
    _print_rig(report.rig)


def _print_constraints(report) -> None:
    """Asked versus achieved. The one table where the numbers are things the
    user asserted rather than clicked, so a miss means something definite."""
    if not report.constraints:
        return
    print(f"\n{'constraint':<26}{'asked':>13}{'got':>13}{'error':>13}  ")
    for c in report.constraints:
        if not c.applied:
            print(f"{c.description:<26}{c.target:13.5f}{'--':>13}{'--':>13}  {c.note[:44]}")
            continue
        flag = "  !" if c.sigmas_off > 3.0 else ""
        print(
            f"{c.description:<26}{c.target:13.5f}{c.achieved:13.5f}{c.error:+13.5f}{flag}"
        )
    if report.datum_scale is not None:
        off = (report.datum_scale - 1.0) * 100.0
        print(
            f"\n  scale implied by those figures: {report.datum_scale:.5f}x "
            f"({off:+.2f}% against the scale the known objects set)"
        )


def _print_rig(rig) -> None:
    if not rig.n_instances:
        return
    print(f"\nrig layout: {rig.n_instances} object(s), extent {rig.extent:.4f} {rig.units}")
    for note in rig.notes:
        print(f"  ! {note}")
    if rig.pairs:
        print(f"  {'pair':<24}{'distance':>12}{'angle':>10}")
        for p in rig.pairs:
            angle = f"{p['angle_deg']:9.3f}d" if "angle_deg" in p else "        --"
            print(f"  {p['a'] + ' - ' + p['b']:<24}{p['distance']:12.5f}{angle}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(prog="multispy", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="open the marking UI (default)")
    p_serve.add_argument(
        "target",
        nargs="?",
        help="folder of photos, or a .multispy project. Omit to reopen the most "
             "recent project (or start an empty one) and add photos in the app.",
    )
    p_serve.add_argument("--port", type=int, default=8777)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.add_argument(
        "--projects", default=None, help="folder holding projects (default: ./projects)"
    )

    p_solve = sub.add_parser("solve", help="solve without the UI")
    p_solve.add_argument("project")
    p_solve.add_argument("--no-bundle", action="store_true", help="stop after linear init")

    p_export = sub.add_parser("export", help="export a solved project")
    p_export.add_argument("project")
    p_export.add_argument("out_dir", nargs="?", default=None)
    p_export.add_argument(
        "--step-shape",
        choices=exporters.STEP_SHAPES,
        default="none",
        help=(
            "how the STEP export carries CAD bodies: none (the default -- the "
            "tree and the poses, no geometry, read by every seat), wireframe "
            "(feature edges, AP214), brep (faceted solid, AP214, small parts "
            "only), tessellated (AP242, every triangle, compact, but CATIA V5 "
            "cannot read it)"
        ),
    )
    p_export.add_argument(
        "--colmap-db",
        default=None,
        metavar="PATH",
        help=(
            "a COLMAP database to take the sparse model's IMAGE_IDs from. "
            "COLMAP numbers photographs as its extractor happens to reach "
            "them, and point_triangulator refuses a model numbered any other "
            "way, so the order is: export, run feature_extractor over the "
            "exported images/, then export again with this flag"
        ),
    )
    p_export.add_argument(
        "--step-units",
        choices=exporters.STEP_UNITS,
        default="mm",
        help="units the STEP file is written in (default mm)",
    )

    p_blend = sub.add_parser(
        "blend", help="build a packed .blend of the solved scene (needs Blender)"
    )
    p_blend.add_argument("project")
    p_blend.add_argument("out", nargs="?", default=None, help="default: export/scene.blend")
    p_blend.add_argument(
        "--blender",
        default=None,
        help=(
            "the Blender executable. Without it: $MULTISPY_BLENDER, then PATH, "
            "then the newest installed"
        ),
    )

    p_regroup = sub.add_parser(
        "regroup", help="split the camera groups by what EXIF says each photo was shot on"
    )
    p_regroup.add_argument("project")
    p_regroup.add_argument(
        "--dry-run", action="store_true", help="say what would move, change nothing"
    )

    p_exif = sub.add_parser("exif", help="show what EXIF says about focal length")
    p_exif.add_argument("project")

    p_rig = sub.add_parser("rig", help="print the solved layout of the known objects")
    p_rig.add_argument("project")
    p_rig.add_argument("--json", action="store_true", help="dump the full payload instead")

    # bare `python -m multispy [path]` means serve.
    #
    # Taken from the subparsers rather than written out again: the literal list
    # that used to be here had to be edited in step with every new command, and
    # the failure when it was not is silent and baffling -- the new command's
    # name is read as a path, `serve` is prepended, and argparse then rejects
    # the command's own arguments as unrecognised extras on `serve`.
    if not argv or argv[0] not in set(sub.choices) | {"-h", "--help"}:
        argv.insert(0, "serve")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "serve":
        root = Path(args.projects).expanduser() if args.projects else DEFAULT_PROJECTS_ROOT
        root.mkdir(parents=True, exist_ok=True)
        project = load_or_create(args.target) if args.target else default_project(root)
        serve(
            project,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            projects_root=root,
        )
        return 0

    if args.command == "solve":
        project = Project.load(args.project)
        report = run_solve(project, run_bundle=not args.no_bundle)
        project.save()
        _print_report(report)
        return 0 if report.ok else 1

    if args.command == "blend":
        from . import blendfile

        project = Project.load(args.project)
        out = Path(args.out) if args.out else project.path.parent / "export" / "scene.blend"
        try:
            exe = blendfile.find_blender(args.blender)
        except blendfile.BlenderNotFound as exc:
            print(exc)
            return 1
        version = ".".join(str(v) for v in blendfile.blender_version(exe)) or "unknown version"
        print(f"building with {exe} ({version})")
        try:
            written = blendfile.build_blend(project, out, blender=exe)
        except (blendfile.BlenderFailed, ValueError) as exc:
            print(exc)
            return 1
        print(f"{written}  ({written.stat().st_size / 1e6:.1f} MB)")
        return 0

    if args.command == "regroup":
        project = Project.load(args.project)
        before = {im.id: im.group for im in project.images}
        moved = exif_focal.assign_camera_groups(project)
        by_group: dict[str, list[str]] = {}
        for im in project.images:
            by_group.setdefault(im.group, []).append(im.id)
        for gid in sorted(by_group):
            g = exif_focal.summarise_group(project, gid)
            focal = f"{g.focal_px:.0f} px" if g.focal_px else "no EXIF focal"
            print(f"{gid}: {len(by_group[gid])} photo(s), {focal}")
            print(f"    {' '.join(by_group[gid])}")
        if not moved:
            print()
            print("nothing to move -- the grouping already matches the photographs")
            return 0
        print()
        print(f"{len(moved)} photo(s) move:")
        for iid, gid in sorted(moved.items()):
            print(f"    {iid}: {before[iid]} -> {gid}")
        if args.dry_run:
            print()
            print("--dry-run: nothing written")
            return 0
        project.save()
        print()
        print(f"written to {project.path}. Re-solve to pick up the new grouping.")
        return 0

    if args.command == "exif":
        project = Project.load(args.project)
        images = project.active_images()
        if not images:
            print("no images in this project")
            return 1
        for gid in sorted({im.group for im in images}):
            g = exif_focal.summarise_group(project, gid)
            print(f"camera group {gid}: {g.n_with_focal}/{g.n_images} image(s) carry a usable focal length")
            if g.focal_px:
                shape = next((im.width, im.height) for im in images if im.group == gid)
                eq = exif_focal.focal_px_to_35mm(g.focal_px, *shape)
                print(f"  {g.focal_px:.1f} px  (~{eq:.0f} mm equivalent)  "
                      f"via {g.source}  [{g.confidence}]")
            if g.note:
                print(f"  {g.note}")
            for e in g.per_image:
                value = f"{e.focal_px:9.1f} px" if e.focal_px else "        -- "
                print(f"    {e.image_id:<10}{value}  {e.confidence:<12}{e.note[:72]}")
        return 0

    if args.command == "rig":
        project = Project.load(args.project)
        payload = exporters.rig_payload(project)
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        if not payload["n_objects"]:
            print(
                "no known objects are placed in this project. Add a model, mark 3+ of its "
                "anchors in 2+ photos each, and solve."
            )
            return 1
        print(
            f"{payload['n_objects']} object(s), extent {payload['extent']:.4f} "
            f"{payload['world_units']}  (models in {', '.join(payload['model_units'])})"
        )
        print(f"measured from the {payload['measured_from']}")
        print(f"\n{'object':<16}{'x':>11}{'y':>11}{'z':>11}   optical axis")
        for row in payload["objects"]:
            c = row["center"] or [0.0, 0.0, 0.0]
            optical = row.get("optical")
            axis = (
                "  ".join(f"{v:+.4f}" for v in optical["forward"]) if optical else "(body only)"
            )
            print(f"{row['label']:<16}{c[0]:11.5f}{c[1]:11.5f}{c[2]:11.5f}   {axis}")
        print(f"\n{'pair':<26}{'distance':>12}{'angle':>10}")
        for p in payload["pairs"]:
            angle = f"{p['angle_deg']:9.3f}d" if "angle_deg" in p else "        --"
            print(f"{p['a'] + ' - ' + p['b']:<26}{p['distance']:12.5f}{angle}")
        return 0

    project = Project.load(args.project)
    out_dir = Path(args.out_dir) if args.out_dir else project.path.parent / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        exporters.export_colmap(
            project, out_dir / "sparse", out_dir / "images", database=args.colmap_db
        )
    )
    print(exporters.export_blender(project, out_dir / "import_cameras.py", out_dir / "images"))
    print(
        exporters.export_transforms_json(project, out_dir / "transforms.json", out_dir / "images")
    )
    print(exporters.export_ply(project, out_dir / "tiepoints.ply"))
    print(
        exporters.export_step(
            project,
            out_dir / "scene.step",
            shape=args.step_shape,
            units=args.step_units,
        )
    )
    if project.placed_instances():
        print(exporters.export_rig(project, out_dir / "rig.json"))
        print(exporters.export_rig_csv(project, out_dir / "rig.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
