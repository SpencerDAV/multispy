"""Triangle-mesh loading, for CAD models of objects with known real-world size.

An object whose geometry you already know is the strongest constraint this tool
can take. Three marked points on a free-floating tie point cost 3 unknowns each
and buy nothing; three marked points on a known body cost 6 unknowns *for the
whole body* and pin the scene's scale outright. That is what makes a camera-array
layout solvable from a handheld pass: every camera in the rig is the same known
object, and placing it is a 6-parameter fit, not a point cloud.

Formats
-------
OBJ, STL (ascii and binary) and PLY (ascii and binary little-endian) are read
directly. Native CAD interchange formats -- STEP, IGES, SLDPRT -- are *not*: they
are boundary representations needing a geometry kernel, which would dwarf the
numpy + scipy + pillow install. Convert once in Blender or FreeCAD and export
OBJ or STL.

Units
-----
None of these formats record units, which is exactly the trap: an STL exported
from SolidWorks in millimetres and one in metres are byte-identical apart from
the numbers. So the scale factor is never guessed from the file -- the caller
states it, and :func:`suggest_units` only offers an opinion for the UI to show.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MESH_SUFFIXES = {".obj", ".stl", ".ply"}

WELD_TOLERANCE = 1e-6
"""Vertices closer than this (relative to the model's size) are one vertex.

STL has no vertex table at all -- every triangle carries its own three corners --
so without welding, a cube arrives as 36 unrelated points and has no edges to
speak of. Everything downstream (feature edges, corner detection, anchor
picking) depends on this step.
"""

FEATURE_ANGLE_DEG = 18.0
"""Dihedral angle above which an edge is a real edge rather than mesh tessellation.

A cylinder's side is a band of near-coplanar quads; drawing every one of them
turns a lens barrel into a solid blob. Keeping only edges where the surface
actually creases gives the clean CAD-looking wireframe you can pick corners off.
"""


@dataclass
class AnchorCandidate:
    """A point on a model that a person could pick out in a photograph."""

    xyz: list[float]
    kind: str
    """``corner`` (a sharp vertex), ``centre`` (the middle of a round or
    symmetric outline), or ``virtual`` (where two edges would meet if a fillet
    had not rounded them off)."""
    score: float
    note: str = ""
    vertex: int | None = None
    """Index into ``vertices`` for a ``corner``; None for the constructed kinds,
    which are not vertices of the mesh at all."""

    def to_dict(self) -> dict:
        return {
            "xyz": [round(float(v), 6) for v in self.xyz],
            "kind": self.kind,
            "score": round(float(self.score), 4),
            "note": self.note,
            "vertex": self.vertex,
        }


@dataclass
class Mesh:
    """A welded triangle soup, plus the derived structure the UI and solver need."""

    vertices: np.ndarray  # (N, 3) float, in the file's own units
    faces: np.ndarray  # (M, 3) int, indices into vertices
    source: str = ""

    # ---------------------------------------------------------------- basics

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)

    @property
    def n_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.n_vertices:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size()))

    def centroid(self) -> np.ndarray:
        """Area-weighted centroid, which for a CAD body is a far more stable
        reference than the mean of its vertices -- tessellated curves pile
        hundreds of vertices onto one fillet and drag a plain mean towards it."""
        if not self.n_faces:
            return self.vertices.mean(axis=0) if self.n_vertices else np.zeros(3)
        tri = self.vertices[self.faces]
        centers = tri.mean(axis=1)
        areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        total = float(areas.sum())
        if total < 1e-18:
            return centers.mean(axis=0)
        return (centers * areas[:, None]).sum(axis=0) / total

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(lengths, 1e-18)

    # -------------------------------------------------------------- topology

    def feature_edges(self, angle_deg: float = FEATURE_ANGLE_DEG) -> np.ndarray:
        """Edges worth drawing: open boundaries, plus creases sharper than ``angle_deg``.

        Returns an (E, 2) index array.
        """
        if not self.n_faces:
            return np.zeros((0, 2), dtype=np.int64)

        normals = self.face_normals()
        # map each undirected edge to the faces using it
        owners: dict[tuple[int, int], list[int]] = {}
        for fi, (a, b, c) in enumerate(self.faces):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(u), int(v)) if u < v else (int(v), int(u))
                owners.setdefault(key, []).append(fi)

        cos_limit = math.cos(math.radians(angle_deg))
        keep: list[tuple[int, int]] = []
        for key, faces in owners.items():
            if len(faces) == 1:
                keep.append(key)  # boundary edge: always a real silhouette
                continue
            # a non-manifold edge (3+ faces) is a crease by construction
            if len(faces) > 2:
                keep.append(key)
                continue
            if float(normals[faces[0]] @ normals[faces[1]]) < cos_limit:
                keep.append(key)
        return np.array(sorted(keep), dtype=np.int64) if keep else np.zeros((0, 2), dtype=np.int64)

    def corner_vertices(self, angle_deg: float = FEATURE_ANGLE_DEG) -> np.ndarray:
        """Vertices where three or more feature edges meet -- i.e. the points a
        person can actually identify and click in a photograph.

        This is the whole usability question for marking a CAD model: the middle
        of a flat face is unmarkable because nothing locates it, while a box
        corner is unambiguous from any angle. Returns indices, ordered so the
        sharpest corners come first.
        """
        edges = self.feature_edges(angle_deg)
        if not len(edges):
            return np.zeros(0, dtype=np.int64)

        incident: dict[int, list[int]] = {}
        for a, b in edges:
            incident.setdefault(int(a), []).append(int(b))
            incident.setdefault(int(b), []).append(int(a))

        scored: list[tuple[float, int]] = []
        for vi, neighbours in incident.items():
            if len(neighbours) < 3:
                continue
            # Sharpness: how far from coplanar the incident edge directions are.
            # Three edges spanning 3D (a box corner) score high; a fan of edges
            # lying in one plane (a crease running across a face) scores ~0.
            dirs = self.vertices[neighbours] - self.vertices[vi]
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / np.maximum(norms, 1e-18)
            singular = np.linalg.svd(dirs, compute_uv=False)
            sharpness = float(singular[2] / max(singular[0], 1e-18)) if len(singular) >= 3 else 0.0
            scored.append((sharpness, vi))

        scored.sort(reverse=True)
        return np.array([vi for _s, vi in scored], dtype=np.int64)

    # ------------------------------------------------- planar regions

    def planar_regions(self, angle_deg: float = FEATURE_ANGLE_DEG) -> list[list[int]]:
        """Faces grouped into flat patches, as lists of face indices.

        Almost everything worth marking on a machined part is a feature of some
        flat face -- the rim of a bored hole, the outline of a sensor window, the
        top of a bolt head. Finding those faces first is what lets the two kinds
        of anchor a vertex scan cannot see fall out of one piece of machinery.
        """
        if not self.n_faces:
            return []
        normals = self.face_normals()
        tri = self.vertices[self.faces]
        offsets = np.einsum("fi,fi->f", normals, tri[:, 0, :])

        shared: dict[tuple[int, int], list[int]] = {}
        for fi, (a, b, c) in enumerate(self.faces):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(u), int(v)) if u < v else (int(v), int(u))
                shared.setdefault(key, []).append(fi)

        neighbours: dict[int, list[int]] = {}
        cos_limit = math.cos(math.radians(angle_deg))
        tolerance = max(self.diagonal(), 1e-9) * 1e-4
        for faces in shared.values():
            if len(faces) != 2:
                continue
            f, g = faces
            if float(normals[f] @ normals[g]) < cos_limit:
                continue
            # same plane, not merely parallel: two sides of a thin plate agree on
            # direction and are a long way from being one surface
            if abs(float(offsets[f] - offsets[g])) > tolerance:
                continue
            neighbours.setdefault(f, []).append(g)
            neighbours.setdefault(g, []).append(f)

        seen = np.zeros(self.n_faces, dtype=bool)
        regions: list[list[int]] = []
        for start in range(self.n_faces):
            if seen[start]:
                continue
            seen[start] = True
            group, stack = [start], [start]
            while stack:
                cur = stack.pop()
                for nxt in neighbours.get(cur, ()):
                    if not seen[nxt]:
                        seen[nxt] = True
                        group.append(nxt)
                        stack.append(nxt)
            regions.append(group)
        return regions

    def region_loops(self, faces: list[int]) -> list[list[int]]:
        """Closed vertex loops bounding one flat patch."""
        used: dict[tuple[int, int], int] = {}
        for fi in faces:
            a, b, c = (int(v) for v in self.faces[fi])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                used[key] = used.get(key, 0) + 1
        border = [k for k, n in used.items() if n == 1]
        if not border:
            return []

        links: dict[int, list[int]] = {}
        for u, v in border:
            links.setdefault(u, []).append(v)
            links.setdefault(v, []).append(u)

        loops: list[list[int]] = []
        unvisited = set(border)
        while unvisited:
            u, v = next(iter(unvisited))
            unvisited.discard((u, v))
            loop = [u, v]
            while True:
                tail = loop[-1]
                nxt = None
                for cand in links.get(tail, ()):
                    key = (tail, cand) if tail < cand else (cand, tail)
                    if key in unvisited:
                        nxt = cand
                        unvisited.discard(key)
                        break
                if nxt is None:
                    break
                if nxt == loop[0]:
                    break  # closed
                loop.append(nxt)
            if len(loop) >= 3:
                loops.append(loop)
        return loops

    def nearest_vertex(self, point: np.ndarray) -> int:
        """Index of the vertex closest to ``point``, for snapping a picked anchor."""
        if not self.n_vertices:
            raise ValueError("mesh has no vertices")
        d = np.linalg.norm(self.vertices - np.asarray(point, dtype=float), axis=1)
        return int(np.argmin(d))

    # ------------------------------------------------- anchor candidates

    def _scored_corners(self, angle_deg: float) -> list[tuple[float, int]]:
        """Sharp vertices, scored by how markable they actually are.

        Sharpness alone is not enough, and ranking by it is what buries a sensor
        outline under a handful of fasteners: every vertex of a hex head is a
        perfectly sharp corner, and none of them is identifiable in a photograph,
        because you cannot tell which of the six you clicked. So the score is
        scaled by the length of the shortest edge meeting there -- tessellation
        detail is short by definition, and real features are not.
        """
        edges = self.feature_edges(angle_deg)
        if not len(edges):
            return []
        diag = self.diagonal() or 1.0
        lengths = np.linalg.norm(self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], axis=1)

        incident: dict[int, list[tuple[int, float]]] = {}
        for (a, b), L in zip(edges, lengths):
            incident.setdefault(int(a), []).append((int(b), float(L)))
            incident.setdefault(int(b), []).append((int(a), float(L)))

        scored: list[tuple[float, int]] = []
        for vi, links in incident.items():
            if len(links) < 3:
                continue
            dirs = self.vertices[[n for n, _L in links]] - self.vertices[vi]
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = dirs / np.maximum(norms, 1e-18)
            singular = np.linalg.svd(dirs, compute_uv=False)
            if len(singular) < 3 or singular[0] <= 1e-18:
                continue
            sharpness = float(singular[2] / singular[0])
            shortest = min(L for _n, L in links)
            scale = min(1.0, shortest / (0.05 * diag))
            scored.append((sharpness * scale, vi))
        scored.sort(reverse=True)
        return scored

    def anchor_candidates(
        self, max_candidates: int = 300, angle_deg: float = FEATURE_ANGLE_DEG
    ) -> list["AnchorCandidate"]:
        """Points on this model a person could actually find in a photograph.

        Three kinds, because a vertex scan alone misses the two that matter most
        on a machined part:

        * **centre** -- the middle of a bored hole, a bolt head, a round port.
          Rotationally symmetric, so it reads the same from every angle, and a
          person can place it by eye at the centre of the visible ellipse. It is
          often the single best anchor on a part, and it is not a vertex at all.
        * **virtual** -- where two straight edges would meet if a fillet had not
          rounded them off. The sharp corner does not exist in the geometry, but
          it is well defined, and it is more stable than the rounded corner it
          replaces, whose apparent position slides with viewpoint and lighting.
        * **corner** -- a genuinely sharp vertex, scored as above.
        """
        diag = self.diagonal() or 1.0
        out: list[AnchorCandidate] = []
        covered: set[int] = set()

        for faces in self.planar_regions(angle_deg):
            for loop in self.region_loops(faces):
                pts = self.vertices[loop]

                found = _loop_centre(pts)
                if found is not None:
                    centre, radius, roundness = found
                    if radius > diag * 0.004:
                        # a rim is better described by its centre than by the
                        # vertices around it, so stop offering those separately
                        covered.update(int(v) for v in loop)
                        size = min(1.0, radius / (0.04 * diag))
                        out.append(
                            AnchorCandidate(
                                xyz=centre.tolist(),
                                kind="centre",
                                score=1.4 * roundness * (0.4 + 0.6 * size),
                                note=f"centre of a {2 * radius:.1f}-across feature",
                            )
                        )

                for xyz, gap, reach in _virtual_corners(pts, diag):
                    out.append(
                        AnchorCandidate(
                            xyz=xyz.tolist(),
                            kind="virtual",
                            score=1.2 * min(1.0, reach / (0.15 * diag)),
                            note=(
                                "where two edges would meet; the corner itself is "
                                f"rounded off by about {gap:.1f}"
                            ),
                        )
                    )

        for score, vi in self._scored_corners(angle_deg):
            if vi in covered:
                continue
            out.append(
                AnchorCandidate(
                    xyz=self.vertices[vi].tolist(), kind="corner", score=score, vertex=vi
                )
            )

        out.sort(key=lambda c: -c.score)
        return _dedupe_candidates(out, tolerance=diag * 1e-3)[:max_candidates]

    # ----------------------------------------------------------- for the UI

    def display_payload(self, max_edges: int = 6000, max_corners: int = 400) -> dict:
        """Wireframe and pickable corners, trimmed to something a browser can
        orbit at 60 fps.

        A production CAD model runs to hundreds of thousands of triangles, which
        is both unsendable and unreadable. Feature edges already cut that by one
        or two orders of magnitude; what survives is thinned longest-first, since
        the long edges are the ones that carry the shape.
        """
        edges = self.feature_edges()
        if len(edges) > max_edges:
            v = self.vertices
            lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
            edges = edges[np.argsort(-lengths)[:max_edges]]

        picks = self.anchor_candidates(max_candidates=max_corners)
        lo, hi = self.bounds()
        return {
            "vertices": self.vertices.round(6).tolist(),
            "edges": edges.tolist(),
            "picks": [p.to_dict() for p in picks],
            "pick_counts": {
                kind: sum(1 for p in picks if p.kind == kind)
                for kind in ("centre", "virtual", "corner")
            },
            "bounds": [lo.tolist(), hi.tolist()],
            "size": self.size().tolist(),
            "centroid": self.centroid().round(6).tolist(),
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_feature_edges": int(len(edges)),
            "source": self.source,
        }


# --------------------------------------------------------------------------
# anchor candidates from a boundary loop
# --------------------------------------------------------------------------


def _loop_centre(pts: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """``(centre, mean radius, roundness)`` if this outline has a usable centre.

    Roundness is the loop's area over that of the circle through it: 1.0 for a
    circle, 0.83 for a hexagon, 0.64 for a square. All three have a centre worth
    marking -- what the test rejects is a long thin outline, whose centre is
    poorly defined along its length and not something anyone can point at.
    """
    pts = np.asarray(pts, dtype=float)
    # Five sides minimum. A genuine round feature always has more than that once
    # tessellated, while a four-sided loop is usually just a flat facet -- the
    # side wall of a hex head, say -- whose middle is not a feature of anything.
    # Rectangular outlines lose nothing by this: their corners say far more than
    # their centre does.
    if len(pts) < 5:
        return None

    middle = pts.mean(axis=0)
    centred = pts - middle
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[2]
    spread = float(np.max(np.linalg.norm(centred, axis=1)))
    if spread < 1e-12:
        return None
    if float(np.max(np.abs(centred @ normal))) > 0.02 * spread:
        return None  # not flat enough to have a centre in any useful sense

    ex, ey = vt[0], vt[1]
    xy = np.stack([centred @ ex, centred @ ey], axis=1)
    x, y = xy[:, 0], xy[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area2 = float(cross.sum())
    if abs(area2) < 1e-18:
        return None
    cx = float(((x + xn) * cross).sum() / (3.0 * area2))
    cy = float(((y + yn) * cross).sum() / (3.0 * area2))

    centre = middle + cx * ex + cy * ey
    radii = np.linalg.norm(pts - centre, axis=1)
    mean_r = float(radii.mean())
    if mean_r < 1e-12:
        return None
    if float(radii.std() / mean_r) > 0.30:
        return None

    roundness = abs(area2) / 2.0 / (math.pi * mean_r * mean_r)
    if roundness < 0.55:
        return None
    return centre, mean_r, float(min(roundness, 1.0))


def _straight_runs(
    pts: np.ndarray, angle_tol_deg: float = 7.0
) -> list[tuple[int, int, np.ndarray, float]]:
    """Maximal straight stretches of a closed loop, as ``(start, end, dir, length)``.

    A fillet arrives as a short burst of many tiny segments between two long
    straight runs, which is exactly what makes it findable.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 3:
        return []
    segments = []
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i]
        L = float(np.linalg.norm(d))
        if L > 1e-12:
            segments.append((i, d / L, L))
    if not segments:
        return []

    cos_tol = math.cos(math.radians(angle_tol_deg))
    runs: list[list] = []
    first, last, direction, total = segments[0][0], segments[0][0], segments[0][1], segments[0][2]
    for i, d, L in segments[1:]:
        if float(d @ direction) >= cos_tol:
            total += L
            last = i
        else:
            runs.append([first, (last + 1) % n, direction, total])
            first, last, direction, total = i, i, d, L
    runs.append([first, (last + 1) % n, direction, total])

    # the loop wraps, so the last run may be the same straight stretch as the first
    if len(runs) > 1 and float(runs[-1][2] @ runs[0][2]) >= cos_tol:
        runs[0][0] = runs[-1][0]
        runs[0][3] += runs[-1][3]
        runs.pop()
    return [(a, b, d, L) for a, b, d, L in runs]


def _virtual_corners(
    pts: np.ndarray, diag: float
) -> list[tuple[np.ndarray, float, float]]:
    """Where consecutive straight runs would meet, as ``(point, gap, reach)``.

    ``gap`` is how far the meeting point sits past the ends of the runs -- the
    fillet radius, in effect -- and is what distinguishes a rounded-off corner
    from two edges that simply happen to point at each other.
    """
    pts = np.asarray(pts, dtype=float)
    runs = [r for r in _straight_runs(pts) if r[3] >= diag * 0.02]
    if len(runs) < 2:
        return []

    out: list[tuple[np.ndarray, float, float]] = []
    pairs = len(runs) if len(runs) > 2 else 1
    for k in range(pairs):
        a = runs[k]
        b = runs[(k + 1) % len(runs)]
        da, db = a[2], b[2]
        axis = np.cross(da, db)
        denom = float(axis @ axis)
        if denom < 1e-12:
            continue
        if math.degrees(math.acos(min(1.0, abs(float(da @ db))))) < 20.0:
            continue

        pa, pb = pts[a[0]], pts[b[0]]
        t = float(np.cross(pb - pa, db) @ axis) / denom
        point = pa + t * da

        gap = max(
            float(np.linalg.norm(point - pts[a[1]])),
            float(np.linalg.norm(point - pts[b[0]])),
        )
        reach = float(min(a[3], b[3]))
        if gap > 0.6 * reach:
            continue
        # Nothing was rounded off here: the runs already meet at a vertex, and
        # the vertex scan reports it as the sharp corner it is. Claiming it as a
        # reconstructed one would be a lie about the geometry.
        if gap <= 0.02 * reach:
            continue
        out.append((point, gap, reach))
    return out


def _dedupe_candidates(
    candidates: list[AnchorCandidate], tolerance: float
) -> list[AnchorCandidate]:
    """Keep the best-scoring candidate in any one spot.

    A corner reached from two adjoining faces, or a centre found from both sides
    of a through-hole, is one anchor and should be offered once.
    """
    grid = max(tolerance, 1e-12)
    occupied: set[tuple[int, int, int]] = set()
    kept: list[AnchorCandidate] = []
    for candidate in candidates:
        key = tuple(int(v) for v in np.round(np.asarray(candidate.xyz) / grid))
        if any(
            (key[0] + dx, key[1] + dy, key[2] + dz) in occupied
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ):
            continue
        occupied.add(key)  # type: ignore[arg-type]
        kept.append(candidate)
    return kept


# --------------------------------------------------------------------------
# welding
# --------------------------------------------------------------------------


def weld(vertices: np.ndarray, faces: np.ndarray, tolerance: float = WELD_TOLERANCE) -> Mesh:
    """Merge coincident vertices and drop degenerate faces.

    The tolerance is relative to the model's own diagonal, so it behaves the same
    on a 30 mm model and a 30 m one.
    """
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(vertices):
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    span = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    grid = max(span * tolerance, 1e-12)
    keys = np.round(vertices / grid).astype(np.int64)
    _unique, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    # keep the first occurrence's actual coordinates rather than the rounded key,
    # so welding never moves geometry
    order = np.argsort(first_index)
    remap = np.empty(len(first_index), dtype=np.int64)
    remap[order] = np.arange(len(order))
    new_vertices = vertices[first_index[order]]
    new_faces = remap[inverse.reshape(-1)][faces.reshape(-1)].reshape(-1, 3)

    a, b, c = new_faces[:, 0], new_faces[:, 1], new_faces[:, 2]
    alive = (a != b) & (b != c) & (a != c)
    return Mesh(new_vertices, new_faces[alive])


def triangulate(polygon: list[int]) -> list[tuple[int, int, int]]:
    """Fan-triangulate a convex-ish polygon. CAD exporters emit quads and n-gons
    for planar faces, where a fan is exact."""
    return [(polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1)]


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def load_obj(path: Path) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx: list[int] = []
                for token in line.split()[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    i = int(raw)
                    # OBJ is 1-based, and negative indices count back from the end
                    idx.append(i - 1 if i > 0 else len(vertices) + i)
                if len(idx) >= 3:
                    faces.extend(triangulate(idx))
    if not vertices:
        raise ValueError("no vertices found -- is this really an OBJ file?")
    return weld(np.array(vertices), np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64))


_STL_FACET = re.compile(r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")


def load_stl(path: Path) -> Mesh:
    data = path.read_bytes()
    # An ascii STL starts with "solid", but so do some binary ones written by
    # careless exporters -- so trust the arithmetic instead: a binary file's
    # length is exactly 84 + 50 * triangles.
    binary = True
    if len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        binary = len(data) == 84 + 50 * count
    if not binary:
        text = data.decode("utf-8", errors="replace")
        coords = [(float(a), float(b), float(c)) for a, b, c in _STL_FACET.findall(text)]
        if not coords or len(coords) % 3:
            raise ValueError("malformed ascii STL: vertex count is not a multiple of 3")
        verts = np.array(coords)
    else:
        (count,) = struct.unpack_from("<I", data, 80)
        if len(data) < 84 + 50 * count:
            raise ValueError("truncated binary STL")
        raw = np.frombuffer(data, dtype=np.uint8, count=50 * count, offset=84).reshape(count, 50)
        # each 50-byte record is normal(3f) + 3 vertices(3f each) + attribute(uint16)
        verts = raw[:, 12:48].copy().view("<f4").reshape(count * 3, 3).astype(float)

    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    return weld(verts, faces)


def load_ply(path: Path) -> Mesh:
    data = path.read_bytes()
    marker = b"end_header"
    cut = data.find(marker)
    if cut < 0:
        raise ValueError("no PLY header found")
    header = data[:cut].decode("ascii", errors="replace")
    body_start = data.find(b"\n", cut) + 1

    fmt = "ascii"
    if "binary_little_endian" in header:
        fmt = "binary_le"
    elif "binary_big_endian" in header:
        raise ValueError("big-endian binary PLY is not supported -- re-export as ascii or OBJ")

    n_vertices = n_faces = 0
    vertex_props: list[tuple[str, str]] = []
    face_count_type = "uchar"
    face_index_type = "int"
    current = None
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                n_vertices = int(parts[2])
            elif current == "face":
                n_faces = int(parts[2])
        elif parts[0] == "property" and current == "vertex" and parts[1] != "list":
            vertex_props.append((parts[1], parts[2]))
        elif parts[0] == "property" and current == "face" and parts[1] == "list":
            face_count_type, face_index_type = parts[2], parts[3]

    names = [name for _t, name in vertex_props]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PLY vertices have no {axis} property")
    xi, yi, zi = names.index("x"), names.index("y"), names.index("z")

    if fmt == "ascii":
        tokens = data[body_start:].decode("ascii", errors="replace").split()
        pos = 0
        stride = len(vertex_props)
        vertices = np.empty((n_vertices, 3))
        for i in range(n_vertices):
            row = tokens[pos : pos + stride]
            vertices[i] = (float(row[xi]), float(row[yi]), float(row[zi]))
            pos += stride
        faces: list[tuple[int, int, int]] = []
        for _ in range(n_faces):
            k = int(tokens[pos])
            idx = [int(v) for v in tokens[pos + 1 : pos + 1 + k]]
            pos += 1 + k
            if len(idx) >= 3:
                faces.extend(triangulate(idx))
    else:
        np_of = {
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
        }
        try:
            dtype = np.dtype([(f"f{i}", np_of[t]) for i, (t, _n) in enumerate(vertex_props)])
        except KeyError as exc:
            raise ValueError(f"unsupported PLY property type {exc}") from exc
        block = np.frombuffer(data, dtype=dtype, count=n_vertices, offset=body_start)
        vertices = np.stack(
            [block[f"f{xi}"].astype(float), block[f"f{yi}"].astype(float), block[f"f{zi}"].astype(float)],
            axis=1,
        )
        offset = body_start + dtype.itemsize * n_vertices
        count_dt, index_dt = np.dtype(np_of[face_count_type]), np.dtype(np_of[face_index_type])
        faces = []
        for _ in range(n_faces):
            k = int(np.frombuffer(data, dtype=count_dt, count=1, offset=offset)[0])
            offset += count_dt.itemsize
            idx = np.frombuffer(data, dtype=index_dt, count=k, offset=offset).astype(int).tolist()
            offset += index_dt.itemsize * k
            if len(idx) >= 3:
                faces.extend(triangulate(idx))

    return weld(vertices, np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64))


def load_mesh(path: str | Path) -> Mesh:
    """Load OBJ, STL or PLY. Raises ValueError with something actionable on failure."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in MESH_SUFFIXES:
        if suffix in {".step", ".stp", ".iges", ".igs", ".sldprt", ".f3d", ".3dm"}:
            raise ValueError(
                f"{suffix} is a parametric CAD format and needs a geometry kernel to read. "
                "Open it in Blender or FreeCAD and export OBJ or STL -- the mesh is all this "
                "needs, and the export keeps the original units."
            )
        raise ValueError(f"unsupported mesh format {suffix!r}; use OBJ, STL or PLY")

    loader = {".obj": load_obj, ".stl": load_stl, ".ply": load_ply}[suffix]
    mesh = loader(path)
    if not mesh.n_vertices:
        raise ValueError(f"{path.name} contains no geometry")
    mesh.source = path.name
    return mesh


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

UNIT_SCALES = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
"""Metres per file unit. World coordinates come out in metres when a template
declares its units, which is what makes a camera-array layout directly readable."""


def suggest_units(mesh: Mesh) -> tuple[str, str]:
    """A guess at the file's units, and why -- strictly advisory.

    Mesh files carry no unit information, so this reasons from the only evidence
    there is: how big the numbers are. It is offered to the user to confirm, and
    never applied silently, because getting it wrong scales the entire solved
    layout by 1000x while every residual stays perfect.
    """
    d = mesh.diagonal()
    if d <= 0:
        return "m", "empty model"
    if d > 300:
        return "mm", f"diagonal is {d:.0f} units, which looks like millimetres"
    if d > 20:
        return "mm", (
            f"diagonal is {d:.0f} units -- millimetres if this is a {d / 10:.0f} cm object, "
            f"but check: centimetres would make it {d / 100:.1f} m and inches {d * 2.54 / 100:.1f} m"
        )
    if d > 0.3:
        return "m", f"diagonal is {d:.2f} units, which looks like metres"
    return "m", f"diagonal is only {d:.4f} units -- check the export scale"
