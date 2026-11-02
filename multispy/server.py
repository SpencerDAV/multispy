"""Local HTTP server backing the marking UI.

Deliberately stdlib-only for the web layer, so the install stays at numpy +
scipy + pillow. Binds to localhost and will only serve image files that the
open project actually references.

The client holds the project as JSON and posts the whole thing back on save or
solve. Projects are small (a few hundred clicks), so this avoids a whole
incremental-sync layer for no benefit.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import webapp
from .docpage import render_page
from .project import CameraGroup, Project
from .webapp import IMAGE_SUFFIXES, MAX_UPLOAD_BYTES, add_images_from_folder, list_projects

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DEFAULT_PROJECTS_ROOT = Path(__file__).resolve().parent.parent / "projects"

# The operations themselves live in webapp.py, which the browser build uses
# directly. This module is transport: HTTP in, JSON out, and nothing else.
__all__ = ["serve", "default_project", "load_or_create", "DEFAULT_PROJECTS_ROOT"]


class _State:
    """Server-wide mutable state, guarded for the threading server.

    The open project and its caches live in a :class:`webapp.Workspace`, which
    is the same object the browser build drives. Everything here is the bit a
    browser does not need: a projects directory on disk, and a lock, because
    this front end is threaded and that one is not.
    """

    def __init__(self, project: Project, projects_root: Path | None = None):
        self.workspace = webapp.Workspace(project)
        self.projects_root = Path(projects_root or DEFAULT_PROJECTS_ROOT)
        self.lock = threading.Lock()

    @property
    def project(self) -> Project:
        return self.workspace.project

    @project.setter
    def project(self, value: Project) -> None:
        self.workspace.project = value

    @property
    def display_cache(self) -> dict:
        return self.workspace.display_cache

    @property
    def mesh_cache(self) -> dict:
        return self.workspace.mesh_cache

    @property
    def last_report(self) -> dict | None:
        return self.workspace.last_report

    @last_report.setter
    def last_report(self, value: dict | None) -> None:
        self.workspace.last_report = value

    def switch_to(self, project: Project) -> None:
        # a fresh workspace drops the caches and the previous report with it,
        # since a report from another project would be nonsense
        self.workspace = webapp.Workspace(project)


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
        return webapp.project_payload(self.state.project)

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
                    self._send_json(self.state.workspace.handle("GET /api/state"))
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
                # The page references its scripts relatively, so the same markup
                # works unchanged on a static host served from a subpath. Serve
                # those names out of web/ here too.
                rel = route.lstrip("/")
                if rel and "/" not in rel and (WEB_DIR / rel).is_file():
                    self._serve_static(rel)
                else:
                    self._send_json({"error": "not found"}, 404)
        except (FileNotFoundError, KeyError) as exc:
            self._send_json({"error": str(exc).strip("'")}, 404)
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
            data = self.state.workspace.photo_bytes(image_id, thumb=thumb)
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
            # Everything the browser build also answers goes straight to the
            # shared workspace. What stays here is the server-only part: a
            # projects directory, a folder on disk to export into, and photos
            # linked from elsewhere on the machine.
            if url.path in webapp.SHARED_ROUTES:
                with self.state.lock:
                    self._send_json(self.state.workspace.handle(url.path, body))
            elif url.path == "/api/export":
                self._post_export(body)
            elif url.path == "/api/add_folder":
                self._post_add_folder(body)
            elif url.path == "/api/new_project":
                self._post_new_project(body)
            elif url.path == "/api/open_project":
                self._post_open_project(body)
            else:
                self._send_json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, 404)
        except KeyError as exc:
            self._send_json({"error": str(exc).strip("'")}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _post_export(self, body: dict) -> None:
        """Server-only: the browser has no folder to write into, so it asks for
        a zip from the same writers instead."""
        with self.state.lock:
            project = self.state.workspace.adopt(body["project"])
            base = project.path.parent if project.path else Path.cwd()
            out_dir = Path(body.get("out_dir") or (base / "export"))
            written = webapp.write_exports(project, out_dir, body.get("formats"))
        self._send_json({"written": written, "out_dir": str(out_dir)})

    def _post_add_folder(self, body: dict) -> None:
        """Server-only: referencing photos where they already sit on disk."""
        with self.state.lock:
            project = self.state.workspace.adopt(body["project"])
            folder = body.get("folder")
            if not folder:
                raise ValueError("no folder given")
            if not Path(folder).expanduser().is_dir():
                raise ValueError(f"not a folder: {folder}")
            added = add_images_from_folder(project, folder)
            project.save()
            self._send_json(
                {"project": self._project_payload(), "added": [im.id for im in added]}
            )

    def _read_exact(self, length: int) -> bytes:
        chunks, remaining = [], length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_upload_body(self) -> bytes:
        """The raw bytes of a one-file upload.

        Deliberately not multipart: one file per request keeps the server
        stdlib-only and lets the client show real per-file progress. The client
        saves the project before uploading, because this handler has no copy of
        the client's edits to merge.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty upload")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        data = self._read_exact(length)
        if len(data) != length:
            raise ValueError("upload truncated")
        return data

    def _post_upload(self, query: dict) -> None:
        data = self._read_upload_body()
        with self.state.lock:
            self._send_json(
                self.state.workspace.add_photo((query.get("name") or [""])[0], data)
            )

    def _post_upload_model(self, query: dict) -> None:
        data = self._read_upload_body()
        with self.state.lock:
            self._send_json(
                self.state.workspace.add_model((query.get("name") or [""])[0], data)
            )

    def _post_new_project(self, body: dict) -> None:
        raw = str(body.get("name") or "").strip()
        folder_name = "".join(c for c in raw if c in webapp._SAFE_CHARS).strip(" .")[:80]
        if not folder_name:
            raise ValueError("give the project a name")
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

    def _post_open_project(self, body: dict) -> None:
        path = Path(str(body.get("path") or "")).expanduser()
        if path.suffix != ".multispy" or not path.is_file():
            raise ValueError(f"not a project file: {path}")
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
