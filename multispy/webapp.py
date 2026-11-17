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

from . import (
    cadguard,
    calibfile,
    exif,
    export,
    nominalcheck,
    sensorcheck,
    sensors,
    uncertainty,
    viewpoint,
)
from .geometry import AXIS_NAMES, _observability_advice, analyse_axes
from .mesh import FEATURE_ANGLE_DEG, MESH_SUFFIXES, UNIT_SCALES, load_mesh, suggest_units
from .part import PartFitReport, part_fit, place_part
from .project import CameraGroup, ImageView, ObjectTemplate, Project
from .solve import solve

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
DISPLAY_MAX_DIM = 2400
"""Photos are served downscaled for display. All marked coordinates stay in
original-image pixels; the client scales by width/display_width."""

MAX_UPLOAD_BYTES = 120 * 1024 * 1024

_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ()")

EXPORT_FORMATS = ("colmap", "blender", "transforms", "ply", "step", "rig")

SHARED_ROUTES = frozenset(
    {
        "/api/state",
        "/api/solve",
        "/api/exif",
        "/api/observability",
        "/api/mesh",
        "/api/rig",
        "/api/remove_image",
        "/api/set_part",
        "/api/part_fit",
        "/api/place_part",
        "/api/edge_refine",
        "/api/edge_revert",
        "/api/origin_uncertainty",
        "/api/check_calibration",
        "/api/check_sensors_in_photos",
        "/api/check_nominal_shape",
        "/api/viewpoints",
        "/api/check_cad",
        "/api/use_calibration_nominals",
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
    # Whether the stored origin sphere still describes this data. Computed here
    # rather than in the client so the digest has exactly one implementation.
    stored = project.origin_uncertainty
    data["_origin_stale"] = bool(
        stored and stored.get("fingerprint") != uncertainty.fingerprint(project)
    )
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
        # a grouped line has no world axis to analyse: its direction is an output
        # of the bundle, so it cannot contribute to a vanishing point here any
        # more than it can in ``calibrate_all``
        if ln.get("direction_group"):
            continue
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


def mesh_payload(path: Path, cache: dict | None = None,
                 angle_deg: float = FEATURE_ANGLE_DEG) -> dict:
    """Wireframe and pickable corners for one model, memoised by mtime.

    The crease angle is part of the key. The same file at two angles is two
    different wireframes, and handing back the cached one would quietly ignore
    the setting that was just changed -- which looks exactly like the setting
    not working.
    """
    key = (str(path), path.stat().st_mtime, float(angle_deg))
    if cache is not None and key in cache:
        return cache[key]
    mesh = load_mesh(path)
    units, why = suggest_units(mesh)
    payload = mesh.display_payload(angle_deg=angle_deg)
    payload["suggested_units"] = units
    payload["units_note"] = why
    if cache is not None:
        if len(cache) > 8:
            cache.clear()
        cache[key] = payload
    return payload


def write_exports(
    project: Project,
    out_dir: str | Path,
    formats=None,
    step_shape: str = "tessellated",
) -> dict[str, str]:
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
    if "step" in formats:
        written["step"] = str(
            export.export_step(project, out_dir / "scene.step", shape=step_shape)
        )
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


def step_bytes(
    project: Project,
    shape: str = "tessellated",
    work_dir: str | Path = "/tmp/ms_step",
) -> bytes:
    """The STEP scene on its own, for a front end with nowhere to put a folder.

    Written through the same scratch directory as :func:`export_zip` and for the
    same reason, but handed back bare rather than packed: a zip holding one file
    is an extra step between the user and the thing they asked for, and a
    browser can download a ``.step`` as readily as a ``.zip``.

    ``shape`` is exposed here, unlike in :func:`write_exports`, because the
    browser build has no command line to fall back to. Somebody whose CAD seat
    will not read AP242 tessellated geometry has to be able to ask for the
    wireframe from the page itself, or the export is simply broken for them with
    no way out.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = export.export_step(project, work_dir / "scene.step", shape=shape)
    data = out.read_bytes()
    out.unlink()  # a scratch file, and the next call wants a clean name
    return data


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
        self.geometry_cache: dict[Any, Any] = {}
        """Loaded :class:`multispy.mesh.Mesh` objects, keyed by path and mtime.

        Separate from ``mesh_cache``, which holds the thinned *display* payload
        the client draws. The part's fit and its edge alignment need the real
        triangles -- the display payload has had its edges sorted, trimmed and
        its faces dropped entirely -- and re-parsing a production OBJ on every
        request is the difference between a fit report that feels instant and
        one that does not."""
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

    def geometry_for(self, project: Project, template: ObjectTemplate):
        """The template's real triangle mesh, memoised by file mtime."""
        path = project.resolve_mesh_path(template)
        if path is None:
            raise ValueError("this object has no model file")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model file is missing: {path}")
        key = (str(path), path.stat().st_mtime)
        if key not in self.geometry_cache:
            if len(self.geometry_cache) > 4:
                self.geometry_cache.clear()
            self.geometry_cache[key] = load_mesh(path)
        return self.geometry_cache[key]

    def part_geometry(self, project: Project):
        """``(instance, template, mesh)`` for the designated part.

        Raises rather than returning a sentinel: every caller here needs all
        three, and a route that answered "no part" by returning half a tuple
        would push that check into four places instead of one.
        """
        inst = project.part_instance()
        if inst is None:
            raise KeyError("no model is marked as the part in this project")
        template = project.template_of(inst)
        if template is None:
            raise KeyError(f"the part's template {inst.template!r} is missing")
        return inst, template, self.geometry_for(project, template)

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
            return {
                "template": template.id,
                "mesh": mesh_payload(path, self.mesh_cache, template.crease_angle),
            }

        # The origin's confidence sphere. Its own route, and never folded into
        # /api/solve, because it costs tens of re-solves: a solve that silently
        # got 40x slower would be a worse bargain than an estimate you ask for.
        if route == "/api/origin_uncertainty":
            project = self.adopt(body["project"])
            n = int(body.get("n") or uncertainty.DEFAULT_SAMPLES)
            n = max(1, min(n, 512))
            result = uncertainty.origin_uncertainty(project, n=n, seed=int(body.get("seed") or 0))
            project.origin_uncertainty = result.to_dict()
            self.save()
            return {"project": project_payload(self.project), "uncertainty": result.to_dict()}

        # Reading a sensor calibration file and judging it. Takes the file's text
        # rather than a path, because the browser build has no filesystem to hand
        # one over from, and the text is what an upload already produces.
        #
        # Deliberately read-only: it does not touch the project. Turning imported
        # poses into nominals and then judging the fit by how well those nominals
        # were met would be circular -- the trap uncertainty.py's docstring names
        # -- so looking at the file and believing it are separate actions.
        if route == "/api/check_calibration":
            project = Project.from_dict(body["project"], path=self.project.path)
            imported = calibfile.read_export(body.get("text") or "")
            s = project.settings
            report = sensors.diagnose(
                imported.sensors,
                tol=float(s.sensor_tol),
                warn=float(s.sensor_warn),
                units=project.nominal_units,
                depth_axis=(
                    int(body["depth_axis"]) if body.get("depth_axis") is not None else None
                ),
            )
            return {
                "import": imported.to_dict(),
                "report": report.to_dict(),
                "grade": report.grade(float(s.sensor_tol), float(s.sensor_warn)),
            }

        # Asking the photographs about the bodies a calibration import placed.
        # Its own route and not folded into the import, because it reads every
        # photograph and builds an edge field per image -- seconds, not
        # milliseconds -- and because the two are different claims: the import
        # asks whether the file contradicts itself, and this asks whether the
        # cell agrees with it.
        if route == "/api/check_sensors_in_photos":
            project = self.adopt(body["project"])
            text = body.get("calibration_text") or ""
            filed = None
            if text:
                imported = calibfile.read_export(text)
                filed = sensors.diagnose(
                    imported.sensors,
                    tol=float(project.settings.sensor_tol),
                    warn=float(project.settings.sensor_warn),
                    units=project.nominal_units,
                )
            result = sensorcheck.check_sensors(
                project,
                mesh_loader=lambda path: load_mesh(path),
                calibration=filed,
            )
            return {"check": result.to_dict()}

        # Read-only, and deliberately not folded into /api/solve: the comparison
        # only means anything against a solve the nominals are NOT driving, so it
        # runs its own with them stripped out. Folding it in would have left it
        # reading the very numbers it is supposed to be auditing.
        if route == "/api/check_nominal_shape":
            project = Project.from_dict(body["project"], path=self.project.path)
            report = nominalcheck.check_nominal_shape(
                project,
                tol_sigmas=float(
                    body.get("tol_sigmas") or nominalcheck.DEFAULT_TOL_SIGMAS
                ),
                warn_sigmas=float(
                    body.get("warn_sigmas") or nominalcheck.DEFAULT_WARN_SIGMAS
                ),
            )
            return {"shape": report.to_dict(), "verdict": report.verdict}

        # Read-only and deliberately cheap: no re-solve, no Monte Carlo. It reads
        # the poses as they stand and apportions the information they already
        # carry, which is why it can run on every solve without being asked for.
        # The weak direction it apportions along comes from the stored origin
        # measurement rather than from anything computed here -- see the module
        # docstring for why those are different questions.
        if route == "/api/viewpoints":
            project = Project.from_dict(body["project"], path=self.project.path)
            report = viewpoint.viewpoint_report(project)
            return {"viewpoints": report.to_dict(), "verdict": report.verdict}

        # Two full solves, so this is the expensive one on this list -- and its own
        # route rather than part of /api/solve for that reason, and for a better
        # one: /api/solve saves by default, and a check that answers "did the
        # bodies make this worse" must not be able to write the worse answer to
        # disk on its way to telling you about it. Read-only, like the shape
        # check, and for the same reason: the comparison only means anything
        # against a solve the bodies are NOT in, so it runs its own.
        #
        # ``apply`` is separate and explicit. Putting the body-free cameras back
        # is a real edit to the project, and it is not something to do as a side
        # effect of looking.
        if route == "/api/check_cad":
            project = Project.from_dict(body["project"], path=self.project.path)
            report = cadguard.check_cad_bodies(project)
            out: dict[str, Any] = {"cad": report.to_dict(), "verdict": report.verdict}
            if body.get("apply"):
                project = self.adopt(body["project"])
                out["applied"] = cadguard.restore_without_bodies(project, report)
                self.save()
                out["project"] = project_payload(self.project)
            return out

        # Adopting the file's feature coordinates as nominals you can pick from.
        # A separate route from /api/check_calibration, which stays read-only:
        # looking at a calibration file and taking its word for where things are
        # ought to be two deliberate actions, not one. This is the second.
        #
        # The candidates are stored, not applied. Nothing becomes a point's
        # nominal until somebody picks it for that point, because which clicked
        # feature IS which named feature is a judgement only the person marking
        # the photographs can make.
        if route == "/api/use_calibration_nominals":
            project = self.adopt(body["project"])
            imported = calibfile.read_export(body.get("text") or "")
            filed = sensors.diagnose(
                imported.sensors,
                tol=float(project.settings.sensor_tol),
                warn=float(project.settings.sensor_warn),
                units=project.nominal_units,
            )
            # The project's own default goes in because a single-axis feature
            # with no stated limits still has axes the cell does not inspect,
            # and widening them has to widen something.
            candidates = sensors.nominal_candidates(
                imported.sensors,
                filed,
                default_sigma=float(project.settings.sigma_nominal),
            )
            project.calibration_nominals = [c.to_dict() for c in candidates]
            self.save()
            usable = sum(1 for c in candidates if c.usable)
            return {
                "project": project_payload(self.project),
                "n_candidates": len(candidates),
                "n_usable": usable,
                "warnings": imported.warnings,
                "report": filed.to_dict(),
            }

        if route == "/api/rig":
            project = Project.from_dict(body["project"], path=self.project.path)
            return export.rig_payload(project)

        # ---- the part: designating it, placing it, and checking the fit
        #
        # Designation is its own route rather than a field the client sets in
        # the project payload, because it has invariants to enforce -- one part
        # at most, pinned at the identity -- and a flag the client could set
        # directly would put those in the client as well as here.
        if route == "/api/set_part":
            project = self.adopt(body["project"])
            template_id = body.get("template")
            if template_id is None:
                changed = project.clear_part()
                self.save()
                return {"project": project_payload(self.project), "cleared": changed}
            inst = project.set_part(str(template_id), label=str(body.get("label") or ""))
            self.save()
            out: dict[str, Any] = {
                "project": project_payload(self.project),
                "instance": inst.id,
                "template": inst.template,
            }
            # The units check is the one thing worth answering at the moment of
            # designation rather than at the next solve: it needs no solve, and
            # a model declared in the wrong units is best caught before anyone
            # reads a wireframe that is a thousand times too big.
            out["fit"] = part_fit(project).to_dict()
            return out

        if route == "/api/place_part":
            project = self.adopt(body["project"])
            _inst, template, mesh = self.part_geometry(project)
            placement = place_part(project, mesh=mesh)
            self.save()
            return {
                "project": project_payload(self.project),
                "placement": placement.to_dict(),
                "fit": part_fit(project, mesh=mesh).to_dict(),
            }

        if route == "/api/part_fit":
            # read-only: asking how well the part fits must not move anything
            project = Project.from_dict(body["project"], path=self.project.path)
            try:
                _inst, _template, mesh = self.part_geometry(project)
            except (KeyError, ValueError, FileNotFoundError):
                mesh = None
            return {"fit": part_fit(project, mesh=mesh).to_dict()}

        if route == "/api/edge_refine":
            from .edgefit import refine_images

            project = self.adopt(body["project"])
            _inst, template, mesh = self.part_geometry(project)
            ids = [str(i) for i in (body.get("images") or [])]
            if not ids:
                raise ValueError("name at least one image to refine")
            # The display JPEG, not the original file: the gradients are
            # computed on a downscaled working image anyway, and the field maps
            # itself back to original pixels from the image's stored dimensions.
            refine = refine_images(
                project, mesh, template, ids, source_for=lambda iid: self.photo_bytes(iid)
            )
            self.save()
            return {
                "project": project_payload(self.project),
                "refine": refine.to_dict(),
                "fit": part_fit(project, mesh=mesh).to_dict(),
            }

        if route == "/api/edge_revert":
            from .edgefit import revert_image_pose

            project = self.adopt(body["project"])
            ids = [str(i) for i in (body.get("images") or [])]
            if not ids:
                ids = [im.id for im in project.images if im.edge_refined]
            reverted = [iid for iid in ids if (im := project.image(iid)) and revert_image_pose(im)]
            self.save()
            return {"project": project_payload(self.project), "reverted": reverted}

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
