"""What a CAD model offers as anchor points, and why those and not others.

The point being defended here is that a vertex scan is not enough. On a real
machined part the two best anchors are usually not vertices at all:

* the centre of a bored hole or a bolt head, which is rotationally symmetric and
  so reads the same from every angle;
* the corner a fillet rounded off, which is well defined as the meeting of two
  straight edges even though no vertex sits there.

Meanwhile a vertex scan on its own is actively misleading: every vertex of a hex
head is a perfectly sharp corner and not one of them can be identified in a
photograph, because you cannot tell which of the six you clicked. A part with a
handful of fasteners will bury its own sensor outline under them.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_anchors.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.mesh import load_mesh

# the test part, in millimetres: a plate carrying one sharp-cornered pad, one
# filleted pad, and six hex bolt heads
PLATE = (120.0, 80.0, 10.0)
SHARP_PAD = (36.0, 24.0)
ROUND_PAD = (30.0, 18.0, 3.0)  # width, height, fillet radius
ROUND_PAD_AT = (0.0, 28.0)
BOLTS = ((-52, -32), (52, -32), (-52, 32), (52, 32), (-52, 0), (52, 0))
BOLT_R = 4.0


def _plate_obj() -> str:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    def box(cx, cy, z0, z1, w, h):
        i = len(verts) + 1
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append((cx + sx * w / 2, cy + sy * h / 2, z0))
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append((cx + sx * w / 2, cy + sy * h / 2, z1))
        faces.extend([
            (i, i + 1, i + 2, i + 3), (i + 4, i + 7, i + 6, i + 5),
            (i, i + 4, i + 5, i + 1), (i + 1, i + 5, i + 6, i + 2),
            (i + 2, i + 6, i + 7, i + 3), (i + 3, i + 7, i + 4, i),
        ])

    def rounded(cx, cy, z0, z1, w, h, r, seg=6):
        ring = []
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            ccx, ccy = cx + sx * (w / 2 - r), cy + sy * (h / 2 - r)
            base = {(1, 1): 0, (-1, 1): 90, (-1, -1): 180, (1, -1): 270}[(sx, sy)]
            for k in range(seg + 1):
                a = math.radians(base + 90 * k / seg)
                ring.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
        i0 = len(verts) + 1
        for p in ring:
            verts.append((p[0], p[1], z1))
        for p in ring:
            verts.append((p[0], p[1], z0))
        n = len(ring)
        faces.append(tuple(range(i0, i0 + n)))
        faces.append(tuple(range(i0 + n, i0 + 2 * n))[::-1])
        for k in range(n):
            a, b = i0 + k, i0 + (k + 1) % n
            faces.append((a, b, b + n, a + n))

    def hex_head(cx, cy, z0, z1, r, seg=6):
        i0 = len(verts) + 1
        for z in (z1, z0):
            for k in range(seg):
                a = 2 * math.pi * k / seg
                verts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
        faces.append(tuple(range(i0, i0 + seg)))
        faces.append(tuple(range(i0 + seg, i0 + 2 * seg))[::-1])
        for k in range(seg):
            a, b = i0 + k, i0 + (k + 1) % seg
            faces.append((a, b, b + seg, a + seg))

    box(0, 0, 0, PLATE[2], PLATE[0], PLATE[1])
    box(0, 0, PLATE[2], PLATE[2] + 2, *SHARP_PAD)
    rounded(ROUND_PAD_AT[0], ROUND_PAD_AT[1], PLATE[2], PLATE[2] + 2, *ROUND_PAD)
    for bx, by in BOLTS:
        hex_head(bx, by, PLATE[2], PLATE[2] + 4, BOLT_R)

    return (
        "".join(f"v {x:g} {y:g} {z:g}\n" for x, y, z in verts)
        + "".join("f " + " ".join(str(i) for i in f) + "\n" for f in faces)
    )


def _write(text: str, suffix: str = ".obj") -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="multispy_anchor_")) / f"part{suffix}"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def _load_plate():
    return load_mesh(_write(_plate_obj()))


def _near(candidates, xyz, tol=0.5):
    target = np.asarray(xyz, dtype=float)
    return [c for c in candidates if np.linalg.norm(np.asarray(c.xyz) - target) <= tol]


# --------------------------------------------------------------------------


def test_filleted_corners_are_offered_at_all():
    """The symptom this exists for: a rounded-off corner has no vertex, so a
    vertex scan offers nothing on the one feature you most want to mark."""
    mesh = _load_plate()
    candidates = mesh.anchor_candidates()

    # the corners the filleted pad would have if it were not filleted
    cx, cy = ROUND_PAD_AT
    w, h, _r = ROUND_PAD
    wanted = [
        (cx + sx * w / 2, cy + sy * h / 2, PLATE[2] + 2)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    for corner in wanted:
        found = _near(candidates, corner)
        assert found, f"no candidate at the filleted corner {corner}"
        assert found[0].kind == "virtual", (corner, found[0].kind)

    # and the old vertex-only scan genuinely could not see them
    sharp = {tuple(np.round(mesh.vertices[i], 3)) for i in mesh.corner_vertices()}
    for corner in wanted:
        assert not any(
            np.linalg.norm(np.asarray(v) - np.asarray(corner)) < 1.0 for v in sharp
        ), f"expected no sharp vertex near {corner}"


def test_bolt_heads_become_centres_not_a_flood_of_rim_vertices():
    """A hex head's vertices are sharp and unmarkable; its centre is neither."""
    mesh = _load_plate()
    candidates = mesh.anchor_candidates()

    for bx, by in BOLTS:
        top = _near(candidates, (bx, by, PLATE[2] + 4))
        assert top, f"no centre on the bolt head at {(bx, by)}"
        assert top[0].kind == "centre", top[0].kind

    # none of the rim vertices should be offered separately
    rim = [
        c for c in candidates
        if any(
            abs(math.hypot(c.xyz[0] - bx, c.xyz[1] - by) - BOLT_R) < 0.3
            for bx, by in BOLTS
        )
    ]
    assert not rim, f"{len(rim)} bolt rim vertices still offered"

    # and the list as a whole is no longer dominated by fasteners
    assert len(candidates) < 60, len(candidates)


def test_a_genuinely_sharp_corner_is_called_a_corner():
    """Reporting a real vertex as a reconstructed one would misdescribe the
    geometry, and the note would claim a fillet that is not there."""
    mesh = _load_plate()
    candidates = mesh.anchor_candidates()
    for sx in (-1, 1):
        for sy in (-1, 1):
            at = (sx * SHARP_PAD[0] / 2, sy * SHARP_PAD[1] / 2, PLATE[2] + 2)
            found = _near(candidates, at)
            assert found, f"no candidate at the sharp pad corner {at}"
            assert found[0].kind == "corner", (at, found[0].kind, found[0].note)
            assert found[0].vertex is not None


def test_plate_corners_survive():
    mesh = _load_plate()
    candidates = mesh.anchor_candidates()
    for sx in (-1, 1):
        for sy in (-1, 1):
            at = (sx * PLATE[0] / 2, sy * PLATE[1] / 2, PLATE[2])
            assert _near(candidates, at), f"no candidate at the plate corner {at}"


def test_a_long_thin_outline_has_no_usable_centre():
    """A centre is only worth offering where it is actually locatable. Halfway
    along a slot is not a point anyone can pick out."""
    verts = [(-50, -5, 0), (50, -5, 0), (50, 5, 0), (-50, 5, 0)]
    obj = "".join(f"v {x} {y} {z}\n" for x, y, z in verts) + "f 1 2 3 4\n"
    mesh = load_mesh(_write(obj))
    assert not [c for c in mesh.anchor_candidates() if c.kind == "centre"]


def test_a_plain_box_is_unchanged():
    """The simple case must not acquire noise: eight corners, nothing invented."""
    size = np.array([100.0, 60.0, 40.0])
    corners = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float
    ) * size
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    obj = (
        "".join(f"v {x:g} {y:g} {z:g}\n" for x, y, z in corners)
        + "".join("f " + " ".join(str(i + 1) for i in q) + "\n" for q in quads)
    )
    mesh = load_mesh(_write(obj))
    candidates = mesh.anchor_candidates()
    assert len(candidates) == 8, len(candidates)
    assert all(c.kind == "corner" for c in candidates), [c.kind for c in candidates]


def test_payload_carries_the_breakdown():
    mesh = _load_plate()
    payload = mesh.display_payload()
    assert "picks" in payload and payload["picks"]
    counts = payload["pick_counts"]
    assert counts["centre"] and counts["virtual"] and counts["corner"], counts
    first = payload["picks"][0]
    assert set(first) == {"xyz", "kind", "score", "note", "vertex"}
    scores = [p["score"] for p in payload["picks"]]
    assert scores == sorted(scores, reverse=True), "candidates must be best-first"


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_filleted_corners_are_offered_at_all,
        test_bolt_heads_become_centres_not_a_flood_of_rim_vertices,
        test_a_genuinely_sharp_corner_is_called_a_corner,
        test_plate_corners_survive,
        test_a_long_thin_outline_has_no_usable_centre,
        test_a_plain_box_is_unchanged,
        test_payload_carries_the_breakdown,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")

    print("\n--- what the test part offers ---")
    mesh = _load_plate()
    candidates = mesh.anchor_candidates()
    print(f"{mesh.n_vertices} vertices, {mesh.n_faces} triangles")
    print(f"vertex scan alone: {len(mesh.corner_vertices())} sharp vertices")
    print(f"anchor candidates: {len(candidates)}")
    for kind in ("centre", "virtual", "corner"):
        group = [c for c in candidates if c.kind == kind]
        print(f"  {kind:<8}{len(group):>3}")
    print(f"\n{'kind':<9}{'score':>7}  position")
    for c in candidates[:10]:
        print(f"{c.kind:<9}{c.score:>7.3f}  {np.round(c.xyz, 1).tolist()}  {c.note[:38]}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
