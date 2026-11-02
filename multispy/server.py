"""Local HTTP server backing the marking UI.

Deliberately stdlib-only for the web layer, so the install stays at numpy +
scipy + pillow. Binds to localhost and will only serve image files that the
open project actually references.

The client holds the project as JSON and posts the whole thing back on save or
solve. Projects are small (a few hundred clicks), so this avoids a whole
incremental-sync layer for no benefit.
"""

from __future__ import annotations

import io
import json
import mimetypes
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

from . import exif, export
from .docpage import render_page
from .geometry import AXIS_NAMES, _observability_advice, analyse_axes
from struct import error as struct_error

from .mesh import MESH_SUFFIXES, UNIT_SCALES, load_mesh, suggest_units
from .project import CameraGroup, ImageView, ObjectTemplate, Project
from .solve import solve

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
DISPLAY_MAX_DIM = 2400
"""Photos are served downscaled for display. All marked coordinates stay in
original-image pixels; the client scales by width/display_width."""

DEFAULT_PROJECTS_ROOT = Path(__file__).resolve().parent.parent / "projects"
MAX_UPLOAD_BYTES = 120 * 1024 * 1024

_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ()")


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


class _State:
    """Server-wide mutable state, guarded for the threading server."""

    def __init__(self, project: Project, projects_root: Path | None = None):
        self.project = project
        self.projects_root = Path(projects_root or DEFAULT_PROJECTS_ROOT)
        self.lock = threading.Lock()
        self.display_cache: dict[tuple[str, float, int], bytes] = {}
        self.mesh_cache: dict[tuple[str, float], dict] = {}
        """Parsed CAD models, memoised by file mtime. A production model can run
        to hundreds of thousands of triangles, and re-parsing it on every orbit
        of the 3D view would be unusable."""
        self.last_report: dict | None = None

    def switch_to(self, project: Project) -> None:
        self.project = project
        self.last_report = None  # a report from another project would be nonsense
        self.display_cache.clear()
        self.mesh_cache.clear()


def default_project(projects_root: Path) -> Project:
    """What to open when launched with no path: the most recently touched
    project under the root, or a fresh empty one ready to have photos dropped
    onto it."""
    existing = list_projects(projects_root)
    if existing:
        return Project.load(existing[0]["path"])
    folder = projects_root / "untitled"
    folder.mkdir(parents=True, exist_ok=True)
    project = Project(name="untitled", path=folder / "project.multispy")
    project.groups["cam0"] = CameraGroup(id="cam0")
    project.save()
    return project


def load_or_create(target: str | Path) -> Project:
    """Open a ``.multispy`` file, or start a project from a folder of photos."""
    target = Path(target).expanduser()
    if target.is_dir():
        project_path = target / "project.multispy"
        if project_path.exists():
            return Project.load(project_path)
        project = Project(name=target.name, path=project_path)
        add_images_from_folder(project, target)
        project.save()
        return project
    if target.suffix != ".multispy" and target.suffix.lower() in IMAGE_SUFFIXES:
        target = target.parent
        return load_or_create(target)
    if target.exists():
        return Project.load(target)
    project = Project(name=target.stem, path=target)
    project.save()
    return project


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


def display_jpeg(state: _State, image: ImageView) -> tuple[bytes, int]:
    """Downscaled JPEG for the canvas, plus its width, memoised by file mtime."""
    path = state.project.resolve_path(image)
    key = (str(path), path.stat().st_mtime, DISPLAY_MAX_DIM)
    cached = state.display_cache.get(key)
    if cached is not None:
        return cached, _jpeg_width(cached)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((DISPLAY_MAX_DIM, DISPLAY_MAX_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()
        width = img.width
    if len(state.display_cache) > 64:
        state.display_cache.clear()
    state.display_cache[key] = data
    return data, width


def _jpeg_width(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as img:
        return int(img.width)


def thumbnail_jpeg(state: _State, image: ImageView, size: int = 220) -> bytes:
    path = state.project.resolve_path(image)
    key = (str(path), path.stat().st_mtime, size)
    cached = state.display_cache.get(key)
    if cached is not None:
        return cached
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
    state.display_cache[key] = data
    return data


class Handler(BaseHTTPRequestHandler):
    state: _State

    server_version = "MultiSpy"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # ------------------------------------------------------------- utilities

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _project_payload(self) -> dict:
        st = self.state
        data = st.project.to_dict()
        data["_path"] = str(st.project.path) if st.project.path else None
        data["_images_meta"] = {}
        for im in st.project.images:
            path = st.project.resolve_path(im)
            data["_images_meta"][im.id] = {
                "exists": path.exists(),
                "absolute": str(path),
                "name": path.name,
            }
        return data

    # ------------------------------------------------------------------- GET

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        route, query = url.path, parse_qs(url.query)
        try:
            if route in ("/", "/index.html"):
                self._serve_static("index.html")
            elif route.startswith("/web/"):
                self._serve_static(route[len("/web/") :])
            elif route == "/docs" or route == "/docs/":
                self.send_response(302)
                self.send_header("Location", "/docs/tutorial")
                self.end_headers()
            elif route.startswith("/docs/"):
                self._serve_doc(route[len("/docs/") :])
            elif route == "/api/state":
                with self.state.lock:
                    self._send_json(
                        {"project": self._project_payload(), "report": self.state.last_report}
                    )
            elif route == "/api/projects":
                with self.state.lock:
                    self._send_json({
                        "root": str(self.state.projects_root),
                        "current": str(self.state.project.path) if self.state.project.path else None,
                        "projects": list_projects(self.state.projects_root),
                    })
            elif route == "/api/photo":
                self._serve_photo(query, thumb=False)
            elif route == "/api/thumb":
                self._serve_photo(query, thumb=True)
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _serve_doc(self, name: str) -> None:
        """The project's own Markdown docs, rendered for reading in the browser."""
        page = render_page(name.strip("/").lower())
        if page is None:
            self._send_json({"error": f"no such document: {name}"}, 404)
            return
        self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_static(self, rel: str) -> None:
        # keep the static handler inside web/ regardless of what was requested
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), ctype)

    def _serve_photo(self, query: dict, thumb: bool) -> None:
        image_id = (query.get("id") or [""])[0]
        with self.state.lock:
            image = self.state.project.image(image_id)
            if image is None:
                self._send_json({"error": "unknown image"}, 404)
                return
            path = self.state.project.resolve_path(image)
            if not path.exists():
                self._send_json({"error": f"missing file: {path}"}, 404)
                return
            if thumb:
                data = thumbnail_jpeg(self.state, image)
            else:
                data, _ = display_jpeg(self.state, image)
        self._send_bytes(data, "image/jpeg", cache=True)

    # ------------------------------------------------------------------ POST

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            # uploads carry raw image bytes rather than JSON, so they branch first
            if url.path == "/api/upload":
                self._post_upload(parse_qs(url.query))
                return
            if url.path == "/api/upload_model":
                self._post_upload_model(parse_qs(url.query))
                return
            body = self._read_json()
            if url.path == "/api/state":
                self._post_state(body)
            elif url.path == "/api/solve":
                self._post_solve(body)
            elif url.path == "/api/export":
                self._post_export(body)
            elif url.path == "/api/add_folder":
                self._post_add_folder(body)
            elif url.path == "/api/observability":
                self._post_observability(body)
            elif url.path == "/api/exif":
                self._post_exif(body)
            elif url.path == "/api/remove_image":
                self._post_remove_image(body)
            elif url.path == "/api/new_project":
                self._post_new_project(body)
            elif url.path == "/api/open_project":
                self._post_open_project(body)
            elif url.path == "/api/mesh":
                self._post_mesh(body)
            elif url.path == "/api/rig":
                self._post_rig(body)
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _adopt(self, raw: dict) -> Project:
        """Replace the in-memory project with the client's version."""
        incoming = Project.from_dict(raw, path=self.state.project.path)
        incoming.path = self.state.project.path
        self.state.project = incoming
        return incoming

    def _post_state(self, body: dict) -> None:
        with self.state.lock:
            project = self._adopt(body["project"])
            path = project.save()
        self._send_json({"saved": str(path)})

    def _post_solve(self, body: dict) -> None:
        with self.state.lock:
            project = self._adopt(body["project"])
            report = solve(project, run_bundle=bool(body.get("bundle", True)))
            self.state.last_report = report.to_dict()
            if body.get("save", True):
                project.save()
            self._send_json(
                {"project": self._project_payload(), "report": self.state.last_report}
            )

    def _post_export(self, body: dict) -> None:
        with self.state.lock:
            project = self._adopt(body["project"])
            base = project.path.parent if project.path else Path.cwd()
            out_dir = Path(body.get("out_dir") or (base / "export"))
            out_dir.mkdir(parents=True, exist_ok=True)
            written = {}
            formats = body.get("formats") or ["colmap", "blender", "transforms", "ply", "rig"]
            if "colmap" in formats:
                written["colmap"] = str(export.export_colmap(project, out_dir / "sparse"))
            if "blender" in formats:
                written["blender"] = str(
                    export.export_blender(project, out_dir / "import_cameras.py")
                )
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
        self._send_json({"written": written, "out_dir": str(out_dir)})

    def _post_add_folder(self, body: dict) -> None:
        with self.state.lock:
            project = self._adopt(body["project"])
            folder = body.get("folder")
            if not folder:
                self._send_json({"error": "no folder given"}, 400)
                return
            if not Path(folder).expanduser().is_dir():
                self._send_json({"error": f"not a folder: {folder}"}, 400)
                return
            added = add_images_from_folder(project, folder)
            project.save()
            self._send_json(
                {"project": self._project_payload(), "added": [im.id for im in added]}
            )

    def _post_exif(self, body: dict) -> None:
        """What EXIF says about each camera group's focal length.

        Read-only: it builds a throwaway Project from the client's state rather
        than adopting it, since asking a question should not move the server's
        idea of the open project.
        """
        with self.state.lock:
            project = Project.from_dict(body["project"], path=self.state.project.path)
            groups = {
                gid: exif.summarise_group(project, gid).to_dict()
                for gid in sorted({im.group for im in project.active_images()})
            }
        self._send_json({"groups": groups})

    def _post_observability(self, body: dict) -> None:
        """Can these control lines determine a focal length, and how well?

        Stateless and cheap, so the UI can call it while the user is still
        drawing -- the whole point is to answer before a full markup exists.
        """
        width = int(body.get("width") or 0)
        height = int(body.get("height") or 0)
        if width <= 0 or height <= 0:
            self._send_json({"error": "need image dimensions"}, 400)
            return
        by_axis: dict[int, list[list[float]]] = {}
        for ln in body.get("axis_lines") or []:
            by_axis.setdefault(int(ln["axis"]), []).append(
                [float(ln["p0"][0]), float(ln["p0"][1]), float(ln["p1"][0]), float(ln["p1"][1])]
            )
        segments = {a: np.array(v, dtype=float) for a, v in by_axis.items()}

        analysis = analyse_axes(
            segments, width, height, sigma_px=float(body.get("sigma_px") or 2.0)
        )
        axes_out = {}
        for axis, vp in analysis.axes.items():
            major, minor = vp.sigma_px()
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
        self._send_json({
            "axes": axes_out,
            "pairs": pairs_out,
            "best": best,
            "problems": analysis.problems,
            "advice": _observability_advice(analysis) if analysis.best else "",
        })

    def _read_exact(self, length: int) -> bytes:
        chunks, remaining = [], length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _post_upload(self, query: dict) -> None:
        """One photo per request, raw bytes in the body.

        Deliberately not multipart: one file per request keeps the server
        stdlib-only and lets the client show real per-file progress. The client
        saves the project before uploading, because this handler has no copy of
        the client's edits to merge.
        """
        name = sanitize_filename((query.get("name") or [""])[0])
        if name is None:
            self._send_json({"error": "unsupported or unsafe filename"}, 400)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json({"error": "empty upload"}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json({"error": f"file is larger than {MAX_UPLOAD_BYTES // (1024*1024)} MB"}, 413)
            return
        data = self._read_exact(length)
        if len(data) != length:
            self._send_json({"error": "upload truncated"}, 400)
            return

        with self.state.lock:
            project = self.state.project
            if project.path is None:
                self._send_json({"error": "project has no folder to save photos into"}, 400)
                return
            photos = project.path.parent / "photos"
            photos.mkdir(parents=True, exist_ok=True)
            target = unique_path(photos, name)
            target.write_bytes(data)
            try:
                width, height = image_dimensions(target)
            except Exception:  # noqa: BLE001 - not actually a decodable image
                target.unlink(missing_ok=True)
                self._send_json({"error": f"{name} is not a readable image"}, 400)
                return
            view = ImageView(
                id=project.next_image_id(),
                path=f"photos/{target.name}",
                width=width,
                height=height,
                group="cam0",
            )
            project.images.append(view)
            project.save()
            self._send_json({"image": view.id, "name": target.name, "width": width, "height": height})

    def _post_remove_image(self, body: dict) -> None:
        with self.state.lock:
            project = self._adopt(body["project"])
            image = project.image(body.get("image_id"))
            if image is None:
                self._send_json({"error": "unknown image"}, 404)
                return
            resolved = project.resolve_path(image)
            owned = project_owns(project, resolved)
            project.images = [im for im in project.images if im.id != image.id]

            deleted = False
            if body.get("delete_file"):
                if not owned:
                    self._send_json(
                        {"error": "that photo lives outside the project folder; "
                                  "removed from the set but not deleted from disk"},
                        400,
                    )
                    return
                try:
                    resolved.unlink(missing_ok=True)
                    deleted = True
                except OSError as exc:
                    self._send_json({"error": f"could not delete the file: {exc}"}, 500)
                    return

            project.save()
            self._send_json({"project": self._project_payload(), "deleted_file": deleted})

    def _post_new_project(self, body: dict) -> None:
        raw = str(body.get("name") or "").strip()
        folder_name = "".join(c for c in raw if c in _SAFE_CHARS).strip(" .")[:80]
        if not folder_name:
            self._send_json({"error": "give the project a name"}, 400)
            return
        with self.state.lock:
            root = self.state.projects_root
            root.mkdir(parents=True, exist_ok=True)
            folder = root / folder_name
            if (folder / "project.multispy").exists():
                self._send_json({"error": f"a project called {folder_name!r} already exists"}, 409)
                return
            folder.mkdir(parents=True, exist_ok=True)
            project = Project(name=folder_name, path=folder / "project.multispy")
            project.groups["cam0"] = CameraGroup(id="cam0")
            project.save()
            self.state.switch_to(project)
            self._send_json({"project": self._project_payload()})

    # --------------------------------------------------------- known objects

    def _mesh_payload(self, path: Path) -> dict:
        """Wireframe and pickable corners for one model, memoised by mtime."""
        key = (str(path), path.stat().st_mtime)
        cached = self.state.mesh_cache.get(key)
        if cached is not None:
            return cached
        mesh = load_mesh(path)
        units, why = suggest_units(mesh)
        payload = mesh.display_payload()
        payload["suggested_units"] = units
        payload["units_note"] = why
        if len(self.state.mesh_cache) > 8:
            self.state.mesh_cache.clear()
        self.state.mesh_cache[key] = payload
        return payload

    def _post_upload_model(self, query: dict) -> None:
        """One CAD model per request, raw bytes in the body.

        As with photos, the client saves the project before uploading: this
        handler has no copy of the client's edits to merge, and it needs to add
        a template of its own.
        """
        name = sanitize_filename((query.get("name") or [""])[0], MESH_SUFFIXES)
        if name is None:
            self._send_json(
                {
                    "error": "unsupported model file -- use "
                    + ", ".join(sorted(MESH_SUFFIXES))
                    + ". STEP, IGES and other parametric CAD formats need a geometry kernel to "
                    "read; convert once in Blender or FreeCAD and the units carry over."
                },
                400,
            )
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json({"error": "empty upload"}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(
                {"error": f"model is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}, 413
            )
            return
        data = self._read_exact(length)
        if len(data) != length:
            self._send_json({"error": "upload truncated"}, 400)
            return

        with self.state.lock:
            project = self.state.project
            if project.path is None:
                self._send_json({"error": "project has no folder to save models into"}, 400)
                return
            models = project.path.parent / "models"
            models.mkdir(parents=True, exist_ok=True)
            target = unique_path(models, name)
            target.write_bytes(data)
            try:
                payload = self._mesh_payload(target)
            except (ValueError, OSError, struct_error) as exc:
                target.unlink(missing_ok=True)
                self._send_json({"error": f"could not read {name}: {exc}"}, 400)
                return

            template_id = project.next_template_id()
            units = payload["suggested_units"]
            project.templates[template_id] = ObjectTemplate(
                id=template_id,
                name=Path(name).stem,
                mesh_path=f"models/{target.name}",
                units=units,
                unit_scale=UNIT_SCALES.get(units, 0.001),
            )
            project.save()
            self._send_json(
                {
                    "project": self._project_payload(),
                    "template": template_id,
                    "mesh": payload,
                }
            )

    def _post_mesh(self, body: dict) -> None:
        """Geometry for one template. Read-only, and resolved against the
        client's own state so an unsaved unit change is still honoured."""
        with self.state.lock:
            project = Project.from_dict(body["project"], path=self.state.project.path)
            template = project.templates.get(body.get("template"))
            if template is None:
                self._send_json({"error": "unknown template"}, 404)
                return
            path = project.resolve_mesh_path(template)
            if path is None:
                self._send_json({"error": "this object has no model file"}, 404)
                return
            if not path.exists():
                self._send_json({"error": f"model file is missing: {path}"}, 404)
                return
            try:
                payload = self._mesh_payload(path)
            except (ValueError, OSError, struct_error) as exc:
                self._send_json({"error": f"could not read the model: {exc}"}, 400)
                return
        self._send_json({"template": template.id, "mesh": payload})

    def _post_rig(self, body: dict) -> None:
        """The layout of the placed objects, without re-solving -- so editing an
        optical frame updates the numbers immediately."""
        with self.state.lock:
            project = Project.from_dict(body["project"], path=self.state.project.path)
            self._send_json(export.rig_payload(project))

    def _post_open_project(self, body: dict) -> None:
        path = Path(str(body.get("path") or "")).expanduser()
        if path.suffix != ".multispy" or not path.is_file():
            self._send_json({"error": f"not a project file: {path}"}, 400)
            return
        with self.state.lock:
            self.state.switch_to(Project.load(path))
            self._send_json({"project": self._project_payload()})


def serve(
    project: Project,
    host: str = "127.0.0.1",
    port: int = 8777,
    open_browser: bool = True,
    projects_root: Path | None = None,
):
    Handler.state = _State(project, projects_root)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"MultiSpy serving {project.path}")
    print(f"  {len(project.images)} image(s), {len(project.points)} tie point(s)")
    print(f"  open {url}   (ctrl+c to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return httpd
