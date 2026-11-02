"""Exporters.

The solve is only useful if it lands somewhere, so the outputs deliberately
target the tools already in the pipeline: a COLMAP sparse model (which Meshroom,
GLOMAP and ``ns-process-data`` all read), a Blender scene script, a nerfstudio
``transforms.json``, and a PLY of the tie points for a quick sanity look.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geometry import Intrinsics, project as project_points
from .project import Project

# CV camera (x right, y down, z forward) -> OpenGL/Blender camera (x right, y up, z back)
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion as (w, x, y, z), COLMAP's ordering."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def camera_to_world_gl(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 camera-to-world matrix in the Blender / nerfstudio axis convention."""
    M = np.eye(4)
    M[:3, :3] = R.T @ CV_TO_GL
    M[:3, 3] = -R.T @ t
    return M


def _solved_images(project: Project) -> list:
    return [
        im
        for im in project.active_images()
        if im.R is not None and im.t is not None
    ]


def optical_world_to_camera(
    center: np.ndarray, forward: np.ndarray, up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera pose for a placed object's optical frame.

    Builds the usual CV camera basis (x right, y down, z forward) from the
    template's declared axis and up vector, so a solved camera body can be
    exported as a real camera rather than a box.
    """
    z = np.asarray(forward, dtype=float)
    z = z / max(np.linalg.norm(z), 1e-12)
    y = -np.asarray(up, dtype=float)  # the camera's y points down
    y = y - z * float(y @ z)  # make it exactly perpendicular to the axis
    if np.linalg.norm(y) < 1e-9:
        # up parallel to the optical axis: pick any perpendicular rather than fail
        fallback = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = fallback - z * float(fallback @ z)
    y = y / max(np.linalg.norm(y), 1e-12)
    x = np.cross(y, z)  # x = y * z for a right-handed x-right/y-down/z-forward frame
    R = np.array([x, y, z])
    return R, -R @ np.asarray(center, dtype=float)


def _instance_rows(project: Project) -> list[dict]:
    """One row per placed object: its pose, and its camera pose where known."""
    rows = []
    for inst in project.placed_instances():
        template = project.template_of(inst)
        if template is None:
            continue
        center = project.instance_center(inst)
        row: dict = {
            "id": inst.id,
            "label": inst.label or inst.id,
            "template": inst.template,
            "template_name": template.name or inst.template,
            "units": template.units,
            "unit_scale": template.unit_scale,
            "locked": inst.locked,
            "R": inst.R.tolist(),
            "t": inst.t.tolist(),
            "center": None if center is None else center.tolist(),
        }
        optical = project.instance_optical(inst)
        if optical is not None:
            eye, forward = optical
            up_local = np.array(template.optical.up, dtype=float)
            R_wc, t_wc = optical_world_to_camera(eye, forward, inst.R @ up_local)
            row["optical"] = {
                "center": eye.tolist(),
                "forward": forward.tolist(),
                "up": (inst.R @ up_local).tolist(),
                "fov_deg": template.optical.fov_deg,
                "R_world_to_camera": R_wc.tolist(),
                "t_world_to_camera": t_wc.tolist(),
                "quaternion_wxyz": rotmat_to_quat(R_wc).tolist(),
            }
        rows.append(row)
    return rows


def _intrinsics_for(project: Project, image) -> Intrinsics:
    g = project.group_of(image)
    return Intrinsics(
        f=g.f if g.f is not None else max(image.width, image.height),
        cx=g.cx if g.cx is not None else image.width / 2.0,
        cy=g.cy if g.cy is not None else image.height / 2.0,
        k1=g.k1,
    )


# --------------------------------------------------------------------------
# COLMAP sparse model (text format)
# --------------------------------------------------------------------------


def export_colmap(project: Project, out_dir: str | Path) -> Path:
    """Write ``cameras.txt`` / ``images.txt`` / ``points3D.txt``.

    This is the format everything downstream speaks, so a hand-marked solve can
    feed Meshroom's dense stage, ``ns-process-data``, or a COLMAP bundle
    adjustment that refines what we produced here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = _solved_images(project)
    if not images:
        raise ValueError("nothing solved yet")

    group_ids = sorted({im.group for im in images})
    cam_id_of = {gid: i + 1 for i, gid in enumerate(group_ids)}
    img_id_of = {im.id: i + 1 for i, im in enumerate(images)}
    points = [p for p in project.points if p.xyz is not None]
    pt_id_of = {p.id: i + 1 for i, p in enumerate(points)}

    # per-image ordered observation list; COLMAP tracks reference these indices
    obs_order: dict[str, list[str]] = {}
    for im in images:
        obs_order[im.id] = [pid for pid in im.observations if pid in pt_id_of]

    with (out_dir / "cameras.txt").open("w", encoding="utf-8") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for gid in group_ids:
            im = next(i for i in images if i.group == gid)
            cam = _intrinsics_for(project, im)
            fh.write(
                f"{cam_id_of[gid]} SIMPLE_RADIAL {im.width} {im.height} "
                f"{cam.f:.10f} {cam.cx:.10f} {cam.cy:.10f} {cam.k1:.10g}\n"
            )

    with (out_dir / "images.txt").open("w", encoding="utf-8") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in images:
            q = rotmat_to_quat(im.R)
            name = Path(im.path).name
            fh.write(
                f"{img_id_of[im.id]} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                f"{im.t[0]:.10f} {im.t[1]:.10f} {im.t[2]:.10f} {cam_id_of[im.group]} {name}\n"
            )
            parts = []
            for pid in obs_order[im.id]:
                uv = im.observations[pid]
                parts.append(f"{uv[0]:.4f} {uv[1]:.4f} {pt_id_of[pid]}")
            fh.write(" ".join(parts) + "\n")

    with (out_dir / "points3D.txt").open("w", encoding="utf-8") as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p in points:
            track, errors = [], []
            for im in images:
                if p.id not in obs_order[im.id]:
                    continue
                idx = obs_order[im.id].index(p.id)
                track.append(f"{img_id_of[im.id]} {idx}")
                cam = _intrinsics_for(project, im)
                proj = project_points(im.R, im.t, cam, p.xyz[None, :])[0]
                if np.all(np.isfinite(proj)):
                    errors.append(float(np.linalg.norm(proj - np.array(im.observations[p.id]))))
            err = float(np.mean(errors)) if errors else 0.0
            fh.write(
                f"{pt_id_of[p.id]} {p.xyz[0]:.10f} {p.xyz[1]:.10f} {p.xyz[2]:.10f} "
                f"200 200 200 {err:.6f} " + " ".join(track) + "\n"
            )

    return out_dir


# --------------------------------------------------------------------------
# nerfstudio / instant-ngp transforms.json
# --------------------------------------------------------------------------


def export_transforms_json(project: Project, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = _solved_images(project)
    if not images:
        raise ValueError("nothing solved yet")

    frames = []
    for im in images:
        cam = _intrinsics_for(project, im)
        frames.append(
            {
                "file_path": im.path.replace("\\", "/"),
                "fl_x": cam.f,
                "fl_y": cam.f,
                "cx": cam.cx,
                "cy": cam.cy,
                "w": im.width,
                "h": im.height,
                "k1": cam.k1,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "transform_matrix": camera_to_world_gl(im.R, im.t).tolist(),
            }
        )

    out_path.write_text(
        json.dumps({"camera_model": "OPENCV", "frames": frames}, indent=2), encoding="utf-8"
    )
    return out_path


# --------------------------------------------------------------------------
# rig layout -- the deliverable when the placed objects are the cameras
# --------------------------------------------------------------------------


def _stations(project: Project) -> list[tuple[str, str, np.ndarray, np.ndarray | None]]:
    """``(id, label, measure-from point, optical axis)`` per placed object."""
    out = []
    for inst in project.placed_instances():
        optical = project.instance_optical(inst)
        if optical is not None:
            out.append((inst.id, inst.label or inst.id, optical[0], optical[1]))
            continue
        center = project.instance_center(inst)
        out.append((inst.id, inst.label or inst.id, center if center is not None else inst.t, None))
    return out


def rig_payload(project: Project) -> dict:
    """Every placed object's pose, plus the pairwise layout.

    Pairwise distances and angles sit alongside the absolute poses because they
    are what a rig is actually specified by, and because they do not depend on
    where the arbitrary world origin ended up -- so they are directly comparable
    against a drawing or a spec sheet, while the absolute poses are only
    meaningful if an object was locked.
    """
    stations = _stations(project)
    pairs = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            _ida, label_a, ca, fa = stations[i]
            _idb, label_b, cb, fb = stations[j]
            entry: dict = {"a": label_a, "b": label_b, "distance": float(np.linalg.norm(cb - ca))}
            if fa is not None and fb is not None:
                entry["angle_deg"] = math.degrees(
                    math.acos(float(np.clip(np.asarray(fa) @ np.asarray(fb), -1.0, 1.0)))
                )
            pairs.append(entry)

    extent = 0.0
    if len(stations) > 1:
        centers = np.array([s[2] for s in stations])
        extent = float(np.max(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)))

    units = {project.template_of(i).units for i in project.placed_instances() if project.template_of(i)}
    return {
        "project": project.name,
        "n_objects": len(stations),
        "model_units": sorted(units),
        "world_units": "m" if units else "arbitrary",
        "extent": extent,
        "measured_from": "optical centre where the template declares one, body centre otherwise",
        "objects": _instance_rows(project),
        "pairs": pairs,
    }


def export_rig(project: Project, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rig_payload(project), indent=2), encoding="utf-8")
    return out_path


def export_rig_csv(project: Project, out_path: str | Path) -> Path:
    """The layout as a flat table, one row per object, for a spreadsheet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,label,x,y,z,axis_x,axis_y,axis_z,yaw_deg,pitch_deg,locked"]
    for inst_id, label, center, forward in _stations(project):
        inst = project.instance(inst_id)
        if forward is None:
            axis = ["", "", ""]
            yaw = pitch = ""
        else:
            axis = [f"{v:.9f}" for v in forward]
            # yaw about world Z, pitch from the world XY plane: the two numbers a
            # rig drawing usually quotes
            yaw = f"{math.degrees(math.atan2(float(forward[0]), float(forward[1]))):.4f}"
            pitch = f"{math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0)))):.4f}"
        lines.append(
            f"{inst_id},{label},"
            + ",".join(f"{v:.9f}" for v in center)
            + ","
            + ",".join(axis)
            + f",{yaw},{pitch},{int(bool(inst.locked)) if inst else 0}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# Blender
# --------------------------------------------------------------------------


BLENDER_TEMPLATE = '''"""Generated by MultiSpy. Run inside Blender: Scripting > Open > Run Script."""

import bpy
import mathutils
from pathlib import Path

CAMERAS = {cameras}
POINTS = {points}
OBJECTS = {objects}
SENSOR_WIDTH_MM = 36.0
COLLECTION_NAME = {collection!r}

# fresh collection so re-running does not pile up duplicates
if COLLECTION_NAME in bpy.data.collections:
    old = bpy.data.collections[COLLECTION_NAME]
    for obj in list(old.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old)
coll = bpy.data.collections.new(COLLECTION_NAME)
bpy.context.scene.collection.children.link(coll)

for entry in CAMERAS:
    data = bpy.data.cameras.new(entry["name"])
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = SENSOR_WIDTH_MM
    data.lens = entry["f_px"] * SENSOR_WIDTH_MM / entry["width"]
    data.shift_x = -(entry["cx"] / entry["width"] - 0.5)
    data.shift_y = (entry["cy"] / entry["height"] - 0.5) * entry["height"] / entry["width"]

    img_path = Path(entry["image"])
    if img_path.exists():
        data.show_background_images = True
        bg = data.background_images.new()
        bg.image = bpy.data.images.load(str(img_path), check_existing=True)
        bg.display_depth = "BACK"
        bg.alpha = 0.85

    obj = bpy.data.objects.new(entry["name"], data)
    obj.matrix_world = mathutils.Matrix(entry["matrix"])
    coll.objects.link(obj)

if POINTS:
    mesh = bpy.data.meshes.new("MultiSpy_TiePoints")
    mesh.from_pydata([tuple(p) for p in POINTS], [], [])
    mesh.update()
    obj = bpy.data.objects.new("MultiSpy_TiePoints", mesh)
    coll.objects.link(obj)


def import_model(path):
    """Import one mesh file, returning the objects it brought in.

    The operator names moved in Blender 4.0, so each format tries the new name
    and falls back to the old one rather than demanding a particular version.
    """
    suffix = Path(path).suffix.lower()
    before = set(bpy.data.objects)
    try:
        if suffix == ".obj":
            try:
                bpy.ops.wm.obj_import(filepath=path)
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=path)
        elif suffix == ".stl":
            try:
                bpy.ops.wm.stl_import(filepath=path)
            except AttributeError:
                bpy.ops.import_mesh.stl(filepath=path)
        elif suffix == ".ply":
            try:
                bpy.ops.wm.ply_import(filepath=path)
            except AttributeError:
                bpy.ops.import_mesh.ply(filepath=path)
        else:
            print("MultiSpy: no importer for %s" % suffix)
            return []
    except Exception as exc:
        print("MultiSpy: could not import %s (%s)" % (path, exc))
        return []
    return [o for o in bpy.data.objects if o not in before]


for entry in OBJECTS:
    placement = mathutils.Matrix(entry["matrix"])
    brought = []
    if entry.get("mesh"):
        if Path(entry["mesh"]).exists():
            brought = import_model(entry["mesh"])
        else:
            print("MultiSpy: model file missing: %s" % entry["mesh"])

    if brought:
        for obj in brought:
            # compose with whatever transform the file itself carried, so a
            # multi-part model keeps its internal arrangement
            obj.matrix_world = placement @ obj.matrix_world
            for other in list(obj.users_collection):
                other.objects.unlink(obj)
            coll.objects.link(obj)
            if len(brought) > 1:
                obj.name = "%s_%s" % (entry["name"], obj.name)
            else:
                obj.name = entry["name"]
    else:
        # no mesh available: an empty still shows where the body was placed
        empty = bpy.data.objects.new(entry["name"], None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = max(entry.get("display_size") or 0.1, 1e-4)
        empty.matrix_world = placement
        coll.objects.link(empty)

    optical = entry.get("optical")
    if optical:
        data = bpy.data.cameras.new(entry["name"] + "_cam")
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = SENSOR_WIDTH_MM
        data.lens = optical["lens_mm"]
        data.display_size = max(entry.get("display_size") or 0.1, 1e-4) * 2.0
        obj = bpy.data.objects.new(entry["name"] + "_cam", data)
        obj.matrix_world = mathutils.Matrix(optical["matrix"])
        coll.objects.link(obj)

scene = bpy.context.scene
if CAMERAS:
    scene.render.resolution_x = CAMERAS[0]["width"]
    scene.render.resolution_y = CAMERAS[0]["height"]
print(
    "MultiSpy: imported %d cameras, %d tie points, %d placed object(s)"
    % (len(CAMERAS), len(POINTS), len(OBJECTS))
)
'''


def export_blender(project: Project, out_path: str | Path) -> Path:
    """Write a Blender script that builds the cameras, with each photo wired up as
    that camera's background image so the alignment can be eyeballed immediately."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = _solved_images(project)
    if not images:
        raise ValueError("nothing solved yet")

    cameras = []
    for im in images:
        cam = _intrinsics_for(project, im)
        cameras.append(
            {
                "name": f"MS_{Path(im.path).stem}",
                "image": str(project.resolve_path(im)),
                "width": im.width,
                "height": im.height,
                "f_px": cam.f,
                "cx": cam.cx,
                "cy": cam.cy,
                "matrix": camera_to_world_gl(im.R, im.t).tolist(),
            }
        )
    points = [p.xyz.tolist() for p in project.points if p.xyz is not None]

    objects = []
    for inst in project.placed_instances():
        template = project.template_of(inst)
        if template is None:
            continue
        mesh_path = project.resolve_mesh_path(template)
        # fold the model's units into the placement matrix, since the mesh file
        # is in its own units and the world is in metres
        placement = np.eye(4)
        placement[:3, :3] = inst.R * template.unit_scale
        placement[:3, 3] = inst.t
        anchors = list(template.anchor_points().values())
        span = (
            float(np.max(np.ptp(np.array(anchors), axis=0))) if len(anchors) > 1 else 0.1
        )
        entry: dict = {
            "name": f"MS_{inst.label or inst.id}",
            "mesh": str(mesh_path) if mesh_path is not None else None,
            "matrix": placement.tolist(),
            "display_size": max(span, 1e-3),
        }
        optical = project.instance_optical(inst)
        if optical is not None:
            eye, forward = optical
            up_world = inst.R @ np.array(template.optical.up, dtype=float)
            R_wc, t_wc = optical_world_to_camera(eye, forward, up_world)
            fov = template.optical.fov_deg or 50.0
            entry["optical"] = {
                "matrix": camera_to_world_gl(R_wc, t_wc).tolist(),
                # a placeholder lens: the array camera's own focal length is not
                # something marking its body could ever reveal, and nothing in
                # the solve depends on it -- it only sizes the frustum drawn here
                "lens_mm": float(36.0 / (2.0 * math.tan(math.radians(fov) / 2.0))),
                "fov_deg": fov,
            }
        objects.append(entry)

    out_path.write_text(
        BLENDER_TEMPLATE.format(
            cameras=json.dumps(cameras, indent=4),
            points=json.dumps(points),
            objects=json.dumps(objects, indent=4),
            collection=f"MultiSpy_{project.name}",
        ),
        encoding="utf-8",
    )
    return out_path


# --------------------------------------------------------------------------
# PLY
# --------------------------------------------------------------------------


def export_ply(project: Project, out_path: str | Path, include_cameras: bool = True) -> Path:
    """Tie points, camera centres and placed-object anchors, each in its own
    colour, for a quick look in MeshLab or CloudCompare."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for p in project.points:
        if p.xyz is not None:
            verts.append((p.xyz, (230, 230, 230)))
    if include_cameras:
        for im in _solved_images(project):
            c = im.center()
            if c is not None:
                verts.append((c, (255, 90, 60)))
    for inst in project.placed_instances():
        template = project.template_of(inst)
        if template is None:
            continue
        for local in template.anchor_points().values():
            verts.append((inst.R @ local + inst.t, (90, 200, 255)))
        optical = project.instance_optical(inst)
        if optical is not None:
            verts.append((optical[0], (255, 220, 60)))

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(verts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for xyz, rgb in verts:
            fh.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")
    return out_path
