"""Project data model and JSON persistence.

A project is one ``.multispy`` JSON file plus the photos it points at. Image
paths are stored relative to the project file so a project folder stays
portable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 2
"""Version 2 adds object templates and instances. Version 1 files load unchanged
-- they simply have none."""

ANCHOR_SEP = "/"
"""Separates instance id from anchor name in an observation key.

Anchor observations live in the same flat ``{key: [u, v]}`` shape as tie points
so that the solver's pruning, connectivity and linear-initialisation stages can
treat both alike, which is what keeps this feature out of three different code
paths. Anchor names are sanitised to keep the key unambiguous.
"""


def anchor_key(instance_id: str, anchor_name: str) -> str:
    return f"{instance_id}{ANCHOR_SEP}{anchor_name}"


def split_anchor_key(key: str) -> tuple[str, str]:
    instance_id, _sep, anchor_name = key.partition(ANCHOR_SEP)
    return instance_id, anchor_name


def clean_anchor_name(name: str) -> str:
    """Anchor names index into observation keys, so the separator cannot appear."""
    cleaned = str(name or "").replace(ANCHOR_SEP, "-").strip()
    return cleaned[:60] or "anchor"


@dataclass
class AxisLine:
    """A control line segment, drawn along a world axis.

    Direction matters: ``p0 -> p1`` is taken to point along the *positive* axis,
    which is how the sign ambiguity of the vanishing point gets resolved.
    """

    axis: int  # 0=X, 1=Y, 2=Z
    p0: list[float]
    p1: list[float]

    def length(self) -> float:
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))


@dataclass
class CameraGroup:
    """A set of images sharing intrinsics -- i.e. shot on one camera at one zoom.

    Grouping is what makes the whole approach stable: one focal length
    constrained by every image's control lines *and* every tie point beats N
    independent focal lengths each constrained by four clicks.
    """

    id: str
    f: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float = 0.0
    lock_focal: bool = False
    lock_principal_point: bool = True
    lock_k1: bool = True


@dataclass
class ImageView:
    id: str
    path: str
    width: int
    height: int
    group: str = "cam0"
    enabled: bool = True
    axis_lines: list[AxisLine] = field(default_factory=list)
    observations: dict[str, list[float]] = field(default_factory=dict)
    anchor_obs: dict[str, list[float]] = field(default_factory=dict)
    """Known-object anchors marked in this photo, keyed ``"<instance>/<anchor>"``.

    Kept separate from ``observations`` because these are not free points -- their
    world positions are dictated by the instance's pose and the model's geometry.
    """
    # solved extrinsics, world-to-camera
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def axis_segments(self) -> dict[int, np.ndarray]:
        """Control lines regrouped as ``{axis: (N,4) array}`` for the geometry code."""
        out: dict[int, list[list[float]]] = {}
        for ln in self.axis_lines:
            out.setdefault(ln.axis, []).append([ln.p0[0], ln.p0[1], ln.p1[0], ln.p1[1]])
        return {a: np.array(v, dtype=float) for a, v in out.items()}

    def center(self) -> np.ndarray | None:
        """Camera centre in world coordinates."""
        if self.R is None or self.t is None:
            return None
        return -self.R.T @ self.t


@dataclass
class TiePoint:
    id: str
    label: str = ""
    xyz: np.ndarray | None = None


@dataclass
class ObjectAnchor:
    """A point on a known object's surface, in the model's own local coordinates
    and its own units.

    Only features a person can unambiguously identify in a photograph are worth
    marking -- body corners, mount-hole centres, the rim of a lens. The middle of
    a flat face is unmarkable because nothing in the photo locates it.
    """

    name: str
    xyz: list[float]


@dataclass
class OpticalFrame:
    """Where a camera actually looks, relative to the body the CAD model describes.

    Without this, solving a camera array tells you where each plastic shell sits,
    which is not what anyone wants to know. With it, the solve reports the real
    extrinsics of every camera in the rig: entrance pupil and optical axis.

    ``forward`` and ``up`` are directions in model-local coordinates; ``center``
    is a position in model-local units.
    """

    center: list[float]
    forward: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    up: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    fov_deg: float | None = None
    """Nominal horizontal field of view, if known. Only ever used for drawing a
    frustum at the right size -- nothing in the solve depends on it, since the
    photos being marked come from a different camera entirely."""


@dataclass
class ObjectTemplate:
    """A known object: its mesh, its real size, and the points marked on it.

    One template, many instances -- which is the whole point for a camera array.
    Ten identical cameras share one template, so measuring the body once pins all
    ten placements, and every anchor marked in every photo refines the same
    shared geometry.
    """

    id: str
    name: str = ""
    mesh_path: str | None = None
    units: str = "mm"
    unit_scale: float = 0.001
    """World units per model unit. Kept explicitly alongside ``units`` so an
    unusual scale factor survives a round trip."""
    anchors: list[ObjectAnchor] = field(default_factory=list)
    optical: OpticalFrame | None = None

    def anchor(self, name: str) -> ObjectAnchor | None:
        return next((a for a in self.anchors if a.name == name), None)

    def anchor_points(self) -> dict[str, np.ndarray]:
        """Anchors in world units, keyed by name."""
        return {
            a.name: np.array(a.xyz, dtype=float) * self.unit_scale for a in self.anchors
        }


@dataclass
class ObjectInstance:
    """One placement of a template in the scene, with a 6-DoF pose to solve for.

    ``R`` and ``t`` map model-local world-unit coordinates into the scene:
    ``X_world = R @ p_local + t``. Note this is object-to-world, the opposite
    sense from an image's world-to-camera ``R``.
    """

    id: str
    template: str
    label: str = ""
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    locked: bool = False
    """Hold this placement fixed during the bundle adjustment. Useful once one
    body is trusted and you want it to define the frame for the rest."""


@dataclass
class AxisLockConstraint:
    """Pins one world coordinate of a point. Three of these pin an origin, and a
    shared value across several points states a plane (``z = 0`` for a floor)."""

    point: str
    axis: int
    value: float = 0.0
    sigma: float = 0.01
    type: str = "axis_lock"


@dataclass
class DistanceConstraint:
    """A known real-world distance between two tie points. One of these turns the
    solve from arbitrary-scale into metric."""

    point_a: str
    point_b: str
    value: float
    sigma: float = 0.01
    type: str = "distance"


Constraint = AxisLockConstraint | DistanceConstraint


@dataclass
class Settings:
    sigma_px: float = 1.5
    """Expected click accuracy for tie points, in pixels."""
    sigma_line_px: float = 2.0
    """Expected accuracy of control line endpoints, in pixels."""
    huber_f_scale: float = 3.0
    """Robust loss knee, in multiples of sigma. Clicks beyond this get downweighted."""
    solve_principal_point: bool = False
    """Try the 3-vanishing-point orthocenter during initialisation. Noisy; off by default."""
    refine_focal: bool = True
    gauge_weight: float = 1e-3
    """Weak prior holding the solution near its initialisation in the 7 gauge
    directions. Too small to bias anything, large enough to keep the normal
    equations full rank when the user has supplied no constraints."""


@dataclass
class Project:
    name: str = "untitled"
    images: list[ImageView] = field(default_factory=list)
    groups: dict[str, CameraGroup] = field(default_factory=dict)
    points: list[TiePoint] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    templates: dict[str, ObjectTemplate] = field(default_factory=dict)
    instances: list[ObjectInstance] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
    path: Path | None = None

    # ---------------------------------------------------------------- lookups

    def image(self, image_id: str) -> ImageView | None:
        return next((im for im in self.images if im.id == image_id), None)

    def point(self, point_id: str) -> TiePoint | None:
        return next((p for p in self.points if p.id == point_id), None)

    def instance(self, instance_id: str) -> ObjectInstance | None:
        return next((o for o in self.instances if o.id == instance_id), None)

    def template_of(self, instance: ObjectInstance) -> ObjectTemplate | None:
        return self.templates.get(instance.template)

    def next_instance_id(self) -> str:
        used = {o.id for o in self.instances}
        i = 0
        while f"obj{i}" in used:
            i += 1
        return f"obj{i}"

    def next_template_id(self) -> str:
        i = 0
        while f"tpl{i}" in self.templates:
            i += 1
        return f"tpl{i}"

    # -------------------------------------------------------- known objects

    def anchor_world_points(self) -> dict[str, np.ndarray]:
        """Every valid anchor observation key mapped to its model-local position
        in world units.

        Keys naming a missing instance, a missing template or a deleted anchor
        are simply absent, which is how stale marks get ignored rather than
        crashing a solve.
        """
        out: dict[str, np.ndarray] = {}
        for inst in self.instances:
            template = self.template_of(inst)
            if template is None:
                continue
            for name, xyz in template.anchor_points().items():
                out[anchor_key(inst.id, name)] = xyz
        return out

    def anchor_observations_of(self, key: str) -> list[tuple[ImageView, np.ndarray]]:
        return [
            (im, np.array(im.anchor_obs[key], dtype=float))
            for im in self.active_images()
            if key in im.anchor_obs
        ]

    def merged_observations(self, image: ImageView, valid: set[str] | None = None) -> dict[str, list[float]]:
        """Tie points and anchor marks as one table.

        The solver's pruning, connectivity and linear-initialisation stages all
        want "every 2D mark in this photo that refers to a 3D position", with no
        interest in which kind it is. Giving them one table is what lets known
        objects reuse those stages untouched instead of duplicating them.
        """
        merged = dict(image.observations)
        for key, uv in image.anchor_obs.items():
            if valid is None or key in valid:
                merged[key] = uv
        return merged

    def instance_center(self, instance: ObjectInstance) -> np.ndarray | None:
        """World position of a placed instance, taken as its anchor centroid.

        The anchor centroid rather than the mesh's: it needs no mesh file to be
        loadable, and it is the part of the object the solve actually saw.
        """
        template = self.template_of(instance)
        if template is None or instance.R is None or instance.t is None:
            return None
        points = list(template.anchor_points().values())
        local = np.mean(points, axis=0) if points else np.zeros(3)
        return instance.R @ local + instance.t

    def instance_optical(self, instance: ObjectInstance) -> tuple[np.ndarray, np.ndarray] | None:
        """World-space entrance pupil and unit optical axis of a placed instance,
        or None if its template declares no optical frame.

        This is the difference between knowing where a camera's shell is and
        knowing where the camera looks, which is the only version of the answer
        a camera array is actually after.
        """
        template = self.template_of(instance)
        if template is None or template.optical is None:
            return None
        if instance.R is None or instance.t is None:
            return None
        center_local = np.array(template.optical.center, dtype=float) * template.unit_scale
        forward = np.array(template.optical.forward, dtype=float)
        norm = float(np.linalg.norm(forward))
        if norm < 1e-12:
            return None
        return instance.R @ center_local + instance.t, instance.R @ (forward / norm)

    def placed_instances(self) -> list[ObjectInstance]:
        return [o for o in self.instances if o.R is not None and o.t is not None]

    def prune_stale_anchor_obs(self) -> int:
        """Drop anchor marks whose instance, template or anchor has gone. Returns
        how many were removed."""
        valid = set(self.anchor_world_points())
        removed = 0
        for im in self.images:
            stale = [k for k in im.anchor_obs if k not in valid]
            for k in stale:
                del im.anchor_obs[k]
            removed += len(stale)
        return removed

    def group_of(self, image: ImageView) -> CameraGroup:
        if image.group not in self.groups:
            self.groups[image.group] = CameraGroup(id=image.group)
        return self.groups[image.group]

    def active_images(self) -> list[ImageView]:
        return [im for im in self.images if im.enabled]

    def observations_of(self, point_id: str) -> list[tuple[ImageView, np.ndarray]]:
        out = []
        for im in self.active_images():
            if point_id in im.observations:
                out.append((im, np.array(im.observations[point_id], dtype=float)))
        return out

    def next_point_id(self) -> str:
        used = {p.id for p in self.points}
        i = 0
        while f"p{i}" in used:
            i += 1
        return f"p{i}"

    def next_image_id(self) -> str:
        used = {im.id for im in self.images}
        i = 0
        while f"img{i}" in used:
            i += 1
        return f"img{i}"

    def resolve_path(self, image: ImageView) -> Path:
        p = Path(image.path)
        if p.is_absolute() or self.path is None:
            return p
        return (self.path.parent / p).resolve()

    def resolve_mesh_path(self, template: ObjectTemplate) -> Path | None:
        if not template.mesh_path:
            return None
        p = Path(template.mesh_path)
        if p.is_absolute() or self.path is None:
            return p
        return (self.path.parent / p).resolve()

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "name": self.name,
            "settings": asdict(self.settings),
            "groups": {gid: asdict(g) for gid, g in self.groups.items()},
            "images": [
                {
                    "id": im.id,
                    "path": im.path,
                    "width": im.width,
                    "height": im.height,
                    "group": im.group,
                    "enabled": im.enabled,
                    "axis_lines": [asdict(ln) for ln in im.axis_lines],
                    "observations": im.observations,
                    "anchor_obs": im.anchor_obs,
                    "R": im.R.tolist() if im.R is not None else None,
                    "t": im.t.tolist() if im.t is not None else None,
                    "notes": im.notes,
                }
                for im in self.images
            ],
            "points": [
                {
                    "id": p.id,
                    "label": p.label,
                    "xyz": p.xyz.tolist() if p.xyz is not None else None,
                }
                for p in self.points
            ],
            "constraints": [asdict(c) for c in self.constraints],
            "templates": {
                tid: {
                    "id": t.id,
                    "name": t.name,
                    "mesh_path": t.mesh_path,
                    "units": t.units,
                    "unit_scale": t.unit_scale,
                    "anchors": [asdict(a) for a in t.anchors],
                    "optical": asdict(t.optical) if t.optical is not None else None,
                }
                for tid, t in self.templates.items()
            },
            "instances": [
                {
                    "id": o.id,
                    "template": o.template,
                    "label": o.label,
                    "locked": o.locked,
                    "R": o.R.tolist() if o.R is not None else None,
                    "t": o.t.tolist() if o.t is not None else None,
                }
                for o in self.instances
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: Path | None = None) -> "Project":
        version = d.get("version", FORMAT_VERSION)
        if version > FORMAT_VERSION:
            raise ValueError(
                f"project file is format version {version}, this build understands {FORMAT_VERSION}"
            )

        settings = Settings(**{k: v for k, v in d.get("settings", {}).items() if k in Settings.__annotations__})
        groups = {
            gid: CameraGroup(**{k: v for k, v in g.items() if k in CameraGroup.__annotations__})
            for gid, g in d.get("groups", {}).items()
        }

        images = []
        for raw in d.get("images", []):
            im = ImageView(
                id=raw["id"],
                path=raw["path"],
                width=int(raw["width"]),
                height=int(raw["height"]),
                group=raw.get("group", "cam0"),
                enabled=bool(raw.get("enabled", True)),
                axis_lines=[AxisLine(axis=int(l["axis"]), p0=list(l["p0"]), p1=list(l["p1"])) for l in raw.get("axis_lines", [])],
                observations={k: list(v) for k, v in raw.get("observations", {}).items()},
                anchor_obs={k: list(v) for k, v in raw.get("anchor_obs", {}).items()},
                notes=list(raw.get("notes", [])),
            )
            if raw.get("R") is not None:
                im.R = np.array(raw["R"], dtype=float)
            if raw.get("t") is not None:
                im.t = np.array(raw["t"], dtype=float)
            images.append(im)

        points = []
        for raw in d.get("points", []):
            pt = TiePoint(id=raw["id"], label=raw.get("label", ""))
            if raw.get("xyz") is not None:
                pt.xyz = np.array(raw["xyz"], dtype=float)
            points.append(pt)

        constraints: list[Constraint] = []
        for raw in d.get("constraints", []):
            kind = raw.get("type")
            if kind == "axis_lock":
                constraints.append(
                    AxisLockConstraint(
                        point=raw["point"],
                        axis=int(raw["axis"]),
                        value=float(raw.get("value", 0.0)),
                        sigma=float(raw.get("sigma", 0.01)),
                    )
                )
            elif kind == "distance":
                constraints.append(
                    DistanceConstraint(
                        point_a=raw["point_a"],
                        point_b=raw["point_b"],
                        value=float(raw["value"]),
                        sigma=float(raw.get("sigma", 0.01)),
                    )
                )
            else:
                raise ValueError(f"unknown constraint type {kind!r}")

        templates: dict[str, ObjectTemplate] = {}
        for tid, raw in (d.get("templates") or {}).items():
            optical_raw = raw.get("optical")
            templates[tid] = ObjectTemplate(
                id=raw.get("id", tid),
                name=raw.get("name", ""),
                mesh_path=raw.get("mesh_path"),
                units=raw.get("units", "mm"),
                unit_scale=float(raw.get("unit_scale", 0.001)),
                anchors=[
                    ObjectAnchor(name=clean_anchor_name(a["name"]), xyz=[float(v) for v in a["xyz"]])
                    for a in raw.get("anchors", [])
                ],
                optical=None
                if not optical_raw
                else OpticalFrame(
                    center=[float(v) for v in optical_raw["center"]],
                    forward=[float(v) for v in optical_raw.get("forward", [0.0, 0.0, 1.0])],
                    up=[float(v) for v in optical_raw.get("up", [0.0, -1.0, 0.0])],
                    fov_deg=None
                    if optical_raw.get("fov_deg") is None
                    else float(optical_raw["fov_deg"]),
                ),
            )

        instances: list[ObjectInstance] = []
        for raw in d.get("instances", []):
            inst = ObjectInstance(
                id=raw["id"],
                template=raw["template"],
                label=raw.get("label", ""),
                locked=bool(raw.get("locked", False)),
            )
            if raw.get("R") is not None:
                inst.R = np.array(raw["R"], dtype=float)
            if raw.get("t") is not None:
                inst.t = np.array(raw["t"], dtype=float)
            instances.append(inst)

        return cls(
            name=d.get("name", "untitled"),
            images=images,
            groups=groups,
            points=points,
            constraints=constraints,
            templates=templates,
            instances=instances,
            settings=settings,
            path=path,
        )

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        # write to a sibling temp file first so an interrupted save can't eat the project
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path=path)
