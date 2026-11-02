"""The operations behind the UI, independent of how the request arrived.

Two front ends use these. :mod:`multispy.server` wraps them in a local HTTP
server. The browser runs them directly: under Pyodide the whole package is
loaded into the page and the UI calls straight into Python with no network in
between.

Keeping the operations here rather than inside the request handler is what lets
the static build be *the same application* rather than a reimplementation of it.
The only thing the browser replaces is the transport, and even the filesystem
carries over, because Pyodide gives Python a real (virtual) one -- so photos and
models are written and read exactly as they are on disk.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from . import exif, export
from .geometry import AXIS_NAMES, _observability_advice, analyse_axes
from .mesh import MESH_SUFFIXES, UNIT_SCALES, load_mesh, suggest_units
from .project import CameraGroup, ImageView, ObjectTemplate, Project
from .solve import solve

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
DISPLAY_MAX_DIM = 2400
"""Photos are served downscaled for display. All marked coordinates stay in
original-image pixels; the client scales by width/display_width."""

MAX_UPLOAD_BYTES = 120 * 1024 * 1024

_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ()")

EXPORT_FORMATS = ("colmap", "blender", "transforms", "ply", "rig")

SHARED_ROUTES = frozenset(
    {
        "/api/state",
        "/api/solve",
        "/api/exif",
        "/api/observability",
        "/api/mesh",
        "/api/rig",
        "/api/remove_image",
    }
)
"""Routes :meth:`Workspace.handle` answers, and both front ends therefore share.

Everything outside this set needs something only one of them has: a projects
directory on disk, a folder to export into, or a browser download.
"""


# --------------------------------------------------------------------------
# filenames and files
# --------------------------------------------------------------------------


def sanitize_filename(name: str, allowed: set[str] | None = None) -> str | None:
    """Reduce an uploaded name to a bare, safe filename, or None if unusable.

    Uploads arrive with a client-supplied name, so this drops any directory part
    and keeps only known-safe characters -- nothing from the request gets to
    choose where a file lands.
    """
    base = Path(str(name or "")).name
    cleaned = "".join(c for c in base if c in _SAFE_CHARS).strip(" .")
    if not cleaned or Path(cleaned).suffix.lower() not in (allowed or IMAGE_SUFFIXES):
        return None
    return cleaned[:160]


def unique_path(folder: Path, name: str) -> Path:
    """A free path in ``folder``, suffixing ``-2``, ``-3`` ... on collision."""
    target = folder / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    for i in range(2, 10000):
        candidate = folder / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise OSError("could not find a free filename")


def project_owns(project: Project, path: Path) -> bool:
    """True if ``path`` is a photo this project copied into its own folder.

    Only these may be deleted from disk; photos referenced from somewhere else
    on the machine are the user's and are merely unlinked from the set.
    """
    if project.path is None:
        return False
    try:
        photos = (project.path.parent / "photos").resolve()
        return path.resolve().is_relative_to(photos)
    except (OSError, ValueError):
        return False


def image_dimensions(path: Path) -> tuple[int, int]:
    """Dimensions as displayed, i.e. after applying EXIF orientation.

    Getting this wrong is a classic source of silent misalignment: the browser
    honours the EXIF rotation tag, so a project that stored the raw dimensions
    would have every click offset.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return int(img.width), int(img.height)


def add_images_from_folder(project: Project, folder: str | Path) -> list[ImageView]:
    folder = Path(folder).expanduser()
    existing = {project.resolve_path(im) for im in project.images}
    added = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.resolve() in existing:
            continue
        try:
            width, height = image_dimensions(path)
        except Exception:  # noqa: BLE001 - unreadable file, just skip it
            continue
        try:
            rel = path.resolve().relative_to(project.path.parent.resolve()).as_posix()
        except (ValueError, AttributeError):
            rel = str(path.resolve())
        view = ImageView(
            id=project.next_image_id(), path=rel, width=width, height=height, group="cam0"
        )
        project.images.append(view)
        added.append(view)
    return added


def list_projects(root: Path) -> list[dict]:
    out = []
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            f = folder / "project.multispy"
            if not f.is_file():
                continue
            n_images = 0
            try:
                n_images = len(json.loads(f.read_text(encoding="utf-8")).get("images", []))
            except Exception:  # noqa: BLE001 - a broken project should still list
                pass
            out.append({
                "name": folder.name,
                "path": str(f),
                "images": n_images,
                "modified": f.stat().st_mtime,
            })
    out.sort(key=lambda p: -p["modified"])
    return out


def display_jpeg(path: Path, cache: dict | None = None) -> tuple[bytes, int]:
    """Downscaled JPEG for the canvas, plus its width, memoised by file mtime."""
    key = (str(path), path.stat().st_mtime, DISPLAY_MAX_DIM)
    if cache is not None and key in cache:
        data = cache[key]
        return data, _jpeg_width(data)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((DISPLAY_MAX_DIM, DISPLAY_MAX_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()
        width = img.width
    if cache is not None:
        if len(cache) > 64:
            cache.clear()
        cache[key] = data
    return data, width


def _jpeg_width(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as img:
        return int(img.width)


def thumbnail_jpeg(path: Path, size: int = 220, cache: dict | None = None) -> bytes:
    key = (str(path), path.stat().st_mtime, size)
    if cache is not None and key in cache:
        return cache[key]
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
    if cache is not None:
        cache[key] = data
    return data


# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------


def project_payload(project: Project) -> dict:
    """The project as the client holds it, plus where its photos actually are."""
    data = project.to_dict()
    data["_path"] = str(project.path) if project.path else None
    data["_images_meta"] = {}
    for im in project.images:
        path = project.resolve_path(im)
        data["_images_meta"][im.id] = {
            "exists": path.exists(),
            "absolute": str(path),
            "name": path.name,
        }
    return data


def exif_payload(project: Project) -> dict:
    """What EXIF says about each camera group's focal length."""
    return {
        "groups": {
            gid: exif.summarise_group(project, gid).to_dict()
            for gid in sorted({im.group for im in project.active_images()})
        }
    }


def observability_payload(body: dict) -> dict:
    """Can these control lines determine a focal length, and how well?

    Stateless and cheap, so the UI can call it while the user is still drawing --
    the whole point is to answer before a full markup exists.
    """
    width = int(body.get("width") or 0)
    height = int(body.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("need image dimensions")

    by_axis: dict[int, list[list[float]]] = {}
    for ln in body.get("axis_lines") or []:
        by_axis.setdefault(int(ln["axis"]), []).append(
            [float(ln["p0"][0]), float(ln["p0"][1]), float(ln["p1"][0]), float(ln["p1"][1])]
        )
    segments = {a: np.array(v, dtype=float) for a, v in by_axis.items()}

    analysis = analyse_axes(segments, width, height, sigma_px=float(body.get("sigma_px") or 2.0))
    axes_out = {}
    for axis, vp in analysis.axes.items():
        major, _minor = vp.sigma_px()
        axes_out[str(axis)] = {
            "name": AXIS_NAMES[axis],
            "lines": int(vp.n_segments),
            "median_length_px": float(vp.median_length_px),
            "at_infinity": bool(vp.at_infinity),
            "distance_px": float(vp.distance_px),
            "lever": float(vp.distance_px / vp.median_length_px) if vp.median_length_px else None,
            "sigma_major_px": None if major == float("inf") else major,
            "residual_px": vp.residual_px,
        }
    pairs_out = [
        {
            "name": f"{AXIS_NAMES[p.axis_a]}x{AXIS_NAMES[p.axis_b]}",
            "focal_px": p.focal_px,
            "relative_sigma": p.relative_sigma,
            "verdict": p.verdict,
            "limiting": None if p.limiting_axis is None else AXIS_NAMES[p.limiting_axis],
            "note": p.note,
        }
        for p in analysis.pairs
    ]
    best = None
    if analysis.best is not None:
        best = next(p for p in pairs_out if p["note"] == analysis.best.note)
    return {
        "axes": axes_out,
        "pairs": pairs_out,
        "best": best,
        "problems": analysis.problems,
        "advice": _observability_advice(analysis) if analysis.best else "",
    }


def mesh_payload(path: Path, cache: dict | None = None) -> dict:
    """Wireframe and pickable corners for one model, memoised by mtime."""
    key = (str(path), path.stat().st_mtime)
    if cache is not None and key in cache:
        return cache[key]
    mesh = load_mesh(path)
    units, why = suggest_units(mesh)
    payload = mesh.display_payload()
    payload["suggested_units"] = units
    payload["units_note"] = why
    if cache is not None:
        if len(cache) > 8:
            cache.clear()
        cache[key] = payload
    return payload


def write_exports(project: Project, out_dir: str | Path, formats=None) -> dict[str, str]:
    """Every requested export, into ``out_dir``. Returns what was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = list(formats or EXPORT_FORMATS)
    written: dict[str, str] = {}
    if "colmap" in formats:
        written["colmap"] = str(export.export_colmap(project, out_dir / "sparse"))
    if "blender" in formats:
        written["blender"] = str(export.export_blender(project, out_dir / "import_cameras.py"))
    if "transforms" in formats:
        written["transforms"] = str(
            export.export_transforms_json(project, out_dir / "transforms.json")
        )
    if "ply" in formats:
        written["ply"] = str(export.export_ply(project, out_dir / "tiepoints.ply"))
    # only worth writing when there is something placed to describe
    if "rig" in formats and project.placed_instances():
        written["rig"] = str(export.export_rig(project, out_dir / "rig.json"))
        written["rig_csv"] = str(export.export_rig_csv(project, out_dir / "rig.csv"))
    return written


def export_zip(project: Project, formats=None, work_dir: str | Path = "/tmp/ms_export") -> bytes:
    """Every export as one zip, for a front end with nowhere to put a folder.

    The browser has no directory to drop files into, so the exports are written
    to a scratch directory and packed. Same writers, same output -- only the
    delivery differs.
    """
    work_dir = Path(work_dir)
    if work_dir.exists():
        for item in sorted(work_dir.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
    work_dir.mkdir(parents=True, exist_ok=True)
    write_exports(project, work_dir, formats)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(work_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(work_dir).as_posix())
    return buf.getvalue()


# --------------------------------------------------------------------------
# a workspace, and the routes over it
# --------------------------------------------------------------------------


class Workspace:
    """One open project and the folder its files live in.

    The folder is real on disk under the local server, and virtual under
    Pyodide. Nothing here cares which, which is the point.
    """

    def __init__(self, project: Project, root: Path | None = None, autosave: bool = True):
        self.project = project
        self.root = Path(root) if root is not None else (
            project.path.parent if project.path else Path(".")
        )
        self.autosave = autosave
        self.display_cache: dict[Any, bytes] = {}
        self.mesh_cache: dict[Any, dict] = {}
        self.last_report: dict | None = None

    # ---------------------------------------------------------------- files

    def adopt(self, raw: dict) -> Project:
        """Replace the open project with the client's version of it."""
        incoming = Project.from_dict(raw, path=self.project.path)
        incoming.path = self.project.path
        self.project = incoming
        return incoming

    def save(self) -> str | None:
        if not self.autosave or self.project.path is None:
            return None
        return str(self.project.save())

    def add_photo(self, name: str, data: bytes) -> dict:
        """Copy one photo into the project and register it."""
        safe = sanitize_filename(name, IMAGE_SUFFIXES)
        if safe is None:
            raise ValueError(
                "unsupported or unsafe filename -- use "
                + ", ".join(sorted(IMAGE_SUFFIXES))
            )
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

        photos = self.root / "photos"
        photos.mkdir(parents=True, exist_ok=True)
        target = unique_path(photos, safe)
        target.write_bytes(data)
        try:
            width, height = image_dimensions(target)
        except Exception as exc:  # noqa: BLE001 - not actually a decodable image
            target.unlink(missing_ok=True)
            raise ValueError(f"{safe} is not a readable image") from exc

        view = ImageView(
            id=self.project.next_image_id(),
            path=f"photos/{target.name}",
            width=width,
            height=height,
            group="cam0",
        )
        self.project.images.append(view)
        self.save()
        return {"image": view.id, "name": target.name, "width": width, "height": height}

    def add_model(self, name: str, data: bytes) -> dict:
        """Copy one CAD model in, and create a template for it."""
        safe = sanitize_filename(name, MESH_SUFFIXES)
        if safe is None:
            raise ValueError(
                "unsupported model file -- use "
                + ", ".join(sorted(MESH_SUFFIXES))
                + ". STEP, IGES and other parametric CAD formats need a geometry kernel to "
                "read; convert once in Blender or FreeCAD and the units carry over."
            )
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"model is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

        models = self.root / "models"
        models.mkdir(parents=True, exist_ok=True)
        target = unique_path(models, safe)
        target.write_bytes(data)
        try:
            payload = mesh_payload(target, self.mesh_cache)
        except Exception as exc:  # noqa: BLE001 - unreadable or unsupported geometry
            target.unlink(missing_ok=True)
            raise ValueError(f"could not read {safe}: {exc}") from exc

        template_id = self.project.next_template_id()
        units = payload["suggested_units"]
        self.project.templates[template_id] = ObjectTemplate(
            id=template_id,
            name=Path(safe).stem,
            mesh_path=f"models/{target.name}",
            units=units,
            unit_scale=UNIT_SCALES.get(units, 0.001),
        )
        self.save()
        return {
            "project": project_payload(self.project),
            "template": template_id,
            "mesh": payload,
        }

    def photo_bytes(self, image_id: str, thumb: bool = False) -> bytes:
        image = self.project.image(image_id)
        if image is None:
            raise KeyError("unknown image")
        path = self.project.resolve_path(image)
        if not path.exists():
            raise FileNotFoundError(f"missing file: {path}")
        if thumb:
            return thumbnail_jpeg(path, cache=self.display_cache)
        return display_jpeg(path, self.display_cache)[0]

    # --------------------------------------------------------------- routes

    def handle(self, route: str, body: dict | None = None) -> dict:
        """Dispatch one UI request. Raises ValueError/KeyError on bad input.

        The route names are the same strings the HTTP server answers, so the
        client can target either front end without knowing which it has.
        """
        body = body or {}

        if route == "GET /api/state":
            return {"project": project_payload(self.project), "report": self.last_report}

        if route == "/api/state":
            self.adopt(body["project"])
            return {"saved": self.save()}

        if route == "/api/solve":
            project = self.adopt(body["project"])
            report = solve(project, run_bundle=bool(body.get("bundle", True)))
            self.last_report = report.to_dict()
            if body.get("save", True):
                self.save()
            return {"project": project_payload(self.project), "report": self.last_report}

        if route == "/api/exif":
            # read-only: a question should not move the open project
            return exif_payload(Project.from_dict(body["project"], path=self.project.path))

        if route == "/api/observability":
            return observability_payload(body)

        if route == "/api/mesh":
            project = Project.from_dict(body["project"], path=self.project.path)
            template = project.templates.get(body.get("template"))
            if template is None:
                raise KeyError("unknown template")
            path = project.resolve_mesh_path(template)
            if path is None:
                raise ValueError("this object has no model file")
            if not path.exists():
                raise FileNotFoundError(f"model file is missing: {path}")
            return {"template": template.id, "mesh": mesh_payload(path, self.mesh_cache)}

        if route == "/api/rig":
            project = Project.from_dict(body["project"], path=self.project.path)
            return export.rig_payload(project)

        if route == "/api/remove_image":
            project = self.adopt(body["project"])
            image = project.image(body.get("image_id"))
            if image is None:
                raise KeyError("unknown image")
            resolved = project.resolve_path(image)
            owned = project_owns(project, resolved) or self.project.path is None
            project.images = [im for im in project.images if im.id != image.id]
            deleted = False
            if body.get("delete_file") and owned:
                try:
                    resolved.unlink(missing_ok=True)
                    deleted = True
                except OSError:
                    pass
            self.save()
            return {"project": project_payload(self.project), "deleted_file": deleted}

        raise KeyError(f"unknown route {route!r}")


def bundle_zip(workspace: Workspace) -> bytes:
    """The whole workspace as one zip: the project file, its photos and models.

    This is what "save" has to mean on a front end with no disk. Saving the
    marks alone would reopen into a project every one of whose photos is
    missing, which is not a saved project -- so the files travel with it.
    """
    project = workspace.project
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.multispy", json.dumps(project.to_dict(), indent=2))
        for folder in ("photos", "models"):
            base = workspace.root / folder
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    zf.write(path, f"{folder}/{path.relative_to(base).as_posix()}")
    return buf.getvalue()


def open_bundle(workspace: Workspace, data: bytes) -> Project:
    """Replace a workspace from a zip written by :func:`bundle_zip`.

    Members are checked against the workspace root before anything is written:
    the archive is user-supplied, and a crafted entry could otherwise climb out
    of it.
    """
    root = workspace.root
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if "project.multispy" not in names:
            raise ValueError("this zip has no project.multispy in it")
        for name in names:
            if name.endswith("/"):
                continue
            target = (root / name).resolve()
            if not str(target).startswith(str(root.resolve())):
                raise ValueError(f"refusing an archive entry that escapes the project: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))

    project = Project.load(root / "project.multispy")
    project.path = root / "project.multispy"
    workspace.project = project
    workspace.display_cache.clear()
    workspace.mesh_cache.clear()
    workspace.last_report = None
    return project


def blank_workspace(root: str | Path = "/workspace", name: str = "untitled") -> Workspace:
    """An empty project in a fresh folder, ready to have photos dropped on it.

    Used by the browser build, where there is no projects directory to reopen
    from and every session starts clean. The folder is emptied first, so
    starting a new project does not inherit the previous one's photos.
    """
    root = Path(root)
    if root.exists():
        for item in sorted(root.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    project = Project(name=name, path=root / "project.multispy")
    project.groups["cam0"] = CameraGroup(id="cam0")
    return Workspace(project, root=root, autosave=True)
