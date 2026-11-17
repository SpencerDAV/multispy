"""Validation of the STEP scene export, against a parser that is not its own.

The export writes ISO 10303-21 by hand, so the one thing worth testing is the
thing a CAD system will do with the file: parse it, resolve every reference,
and recover the poses. Checking it by asking :mod:`multispy.stepfile` what it
meant would test nothing at all -- a missing bracket or a swapped axis is
exactly the kind of fault that round-trips perfectly through the code that
caused it.

So this file carries its own Part 21 reader. It is small and it is independent,
which between them are the whole point: it tokenises the file from scratch,
rebuilds the assembly tree out of the five entities a placement is spread
across, and compares the poses it recovers against the scene that went in.

The scene is synthetic and built here, in full. Nothing is read from
``projects/`` -- that directory is gitignored and holds a user's own photos, so
a test depending on it cannot pass in a clean checkout.

The faults this has actually caught, kept as a record of what is worth
asserting:

* an item list written without its brackets, which made every
  ``SHAPE_REPRESENTATION`` in the file carry the wrong parameter count. The file
  still loaded -- OpenCASCADE recovered the geometry and reported the error
  only on a full check -- so nothing short of counting parameters would have
  noticed.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_step.py
"""

from __future__ import annotations

import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy import export
from multispy.project import (
    CameraGroup,
    ImageView,
    ObjectAnchor,
    ObjectInstance,
    ObjectTemplate,
    OpticalFrame,
    Project,
    TiePoint,
)

# --------------------------------------------------------------------------
# a Part 21 reader, deliberately written from the standard and not from the
# writer: entity ids, typed parameters, nested lists, complex instances
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Ref:
    """``#12`` -- a reference to another entity."""

    id: int


@dataclass(frozen=True)
class Enum:
    """``.T.`` or ``.MILLI.`` -- a Part 21 enumeration or boolean."""

    name: str


@dataclass(frozen=True)
class Typed:
    """``LENGTH_MEASURE(1.5)`` -- a typed parameter inside another entity."""

    name: str
    args: tuple


@dataclass
class Entity:
    id: int
    parts: dict[str, tuple]
    """Keyword to parameter tuple. A simple entity has one key; a complex
    instance -- ``( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )`` --
    has one per subtype, which is the form every unit and context in the file
    takes."""

    @property
    def name(self) -> str:
        return next(iter(self.parts))

    @property
    def args(self) -> tuple:
        return self.parts[self.name]


_TOKEN = re.compile(
    r"""
    (?P<string>'(?:''|[^'])*')
  | (?P<ref>\#\d+)
  | (?P<enum>\.[A-Z0-9_]*\.)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<real>[-+]?(?:\d+\.\d*(?:E[-+]?\d+)?|\d*\.\d+(?:E[-+]?\d+)?|\d+E[-+]?\d+))
  | (?P<int>[-+]?\d+)
  | (?P<punct>[(),$*=;])
    """,
    re.VERBOSE,
)


def _tokenise(src: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    n = len(src)
    while pos < n:
        if src[pos] in " \t\r\n":
            pos += 1
            continue
        if src.startswith("/*", pos):
            end = src.find("*/", pos + 2)
            if end < 0:
                raise ValueError("unterminated comment")
            pos = end + 2
            continue
        m = _TOKEN.match(src, pos)
        if m is None:
            raise ValueError(f"cannot tokenise at {src[pos:pos + 40]!r}")
        kind = m.lastgroup
        out.append((kind, m.group()))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, value: str) -> None:
        kind, got = self.take()
        if got != value:
            raise ValueError(f"expected {value!r}, got {got!r}")

    def value(self):
        kind, tok = self.take()
        if kind == "string":
            return tok[1:-1].replace("''", "'")
        if kind == "ref":
            return Ref(int(tok[1:]))
        if kind == "enum":
            return Enum(tok.strip("."))
        if kind == "real":
            return float(tok)
        if kind == "int":
            return int(tok)
        if tok == "$":
            return None
        if tok == "*":
            return "*"
        if tok == "(":
            items = []
            if self.peek()[1] == ")":
                self.take()
                return items
            while True:
                items.append(self.value())
                kind, sep = self.take()
                if sep == ")":
                    return items
                if sep != ",":
                    raise ValueError(f"expected , or ) in list, got {sep!r}")
        if kind == "name":
            self.expect("(")
            return Typed(tok, tuple(self.arglist()))
        raise ValueError(f"unexpected token {tok!r}")

    def arglist(self) -> list:
        """Parameters up to and including the closing bracket."""
        if self.peek()[1] == ")":
            self.take()
            return []
        out = []
        while True:
            out.append(self.value())
            kind, sep = self.take()
            if sep == ")":
                return out
            if sep != ",":
                raise ValueError(f"expected , or ) in arguments, got {sep!r}")


def parse_step(text: str) -> tuple[dict[str, str], dict[int, Entity]]:
    """A Part 21 file as its header strings and its entities by id."""
    if not text.startswith("ISO-10303-21;"):
        raise ValueError("missing ISO-10303-21 magic")
    if not text.rstrip().endswith("END-ISO-10303-21;"):
        raise ValueError("missing END-ISO-10303-21 terminator")
    head, _, rest = text.partition("HEADER;")
    if not _:
        raise ValueError("no HEADER section")
    header_src, _, rest = rest.partition("ENDSEC;")
    data_src, _, tail = rest.partition("DATA;")[2].partition("ENDSEC;")
    if "END-ISO-10303-21;" not in tail:
        raise ValueError("DATA section is not closed before the terminator")

    header: dict[str, str] = {}
    for item in re.finditer(r"(FILE_[A-Z_]+)\s*\(", header_src):
        start = item.end()
        depth = 1
        pos = start
        while depth:
            if header_src[pos] == "(":
                depth += 1
            elif header_src[pos] == ")":
                depth -= 1
            pos += 1
        header[item.group(1)] = header_src[start : pos - 1]

    entities: dict[int, Entity] = {}
    parser = _Parser(_tokenise(data_src))
    while parser.i < len(parser.tokens):
        kind, tok = parser.take()
        if tok == ";":
            continue
        if kind != "ref":
            raise ValueError(f"expected an entity id, got {tok!r}")
        eid = int(tok[1:])
        parser.expect("=")
        parts: dict[str, tuple] = {}
        if parser.peek()[1] == "(":
            parser.take()  # a complex instance: subtypes until the closer
            while parser.peek()[1] != ")":
                sub_kind, sub = parser.take()
                if sub_kind != "name":
                    raise ValueError(f"expected a subtype name, got {sub!r}")
                parser.expect("(")
                parts[sub] = tuple(parser.arglist())
            parser.take()
        else:
            name_kind, name = parser.take()
            if name_kind != "name":
                raise ValueError(f"expected an entity name, got {name!r}")
            parser.expect("(")
            parts[name] = tuple(parser.arglist())
        parser.expect(";")
        if eid in entities:
            raise ValueError(f"#{eid} is defined twice")
        entities[eid] = Entity(id=eid, parts=parts)
    return header, entities


# --------------------------------------------------------------------------
# reading the model back out
# --------------------------------------------------------------------------

#: Parameter counts for every entity this exporter writes. A reader checks
#: these against the schema and complains; since the suite cannot carry the
#: schema, it carries the counts.
ARITY = {
    "APPLICATION_CONTEXT": 1,
    "APPLICATION_PROTOCOL_DEFINITION": 4,
    "PRODUCT_CONTEXT": 3,
    "PRODUCT_DEFINITION_CONTEXT": 3,
    "UNCERTAINTY_MEASURE_WITH_UNIT": 4,
    "CARTESIAN_POINT": 2,
    "DIRECTION": 2,
    "AXIS2_PLACEMENT_3D": 4,
    "POLYLINE": 2,
    "GEOMETRIC_CURVE_SET": 2,
    "COORDINATES_LIST": 3,
    "TRIANGULATED_FACE": 7,
    "TESSELLATED_SHELL": 3,
    "PLANE": 2,
    "POLY_LOOP": 2,
    "FACE_OUTER_BOUND": 3,
    "FACE_SURFACE": 4,
    "CLOSED_SHELL": 2,
    "FACETED_BREP": 2,
    "PRODUCT": 4,
    "PRODUCT_DEFINITION_FORMATION": 3,
    "PRODUCT_DEFINITION": 4,
    "PRODUCT_DEFINITION_SHAPE": 3,
    "SHAPE_DEFINITION_REPRESENTATION": 2,
    "SHAPE_REPRESENTATION": 3,
    "TESSELLATED_SHAPE_REPRESENTATION": 3,
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE": 6,
    "ITEM_DEFINED_TRANSFORMATION": 4,
    "REPRESENTATION_RELATIONSHIP": 4,
    "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION": 1,
    "SHAPE_REPRESENTATION_RELATIONSHIP": 0,
    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION": 2,
    "COLOUR_RGB": 4,
    "DRAUGHTING_PRE_DEFINED_CURVE_FONT": 1,
    "CURVE_STYLE": 4,
    "FILL_AREA_STYLE_COLOUR": 2,
    "FILL_AREA_STYLE": 2,
    "SURFACE_STYLE_FILL_AREA": 1,
    "SURFACE_SIDE_STYLE": 2,
    "SURFACE_STYLE_USAGE": 2,
    "PRESENTATION_STYLE_ASSIGNMENT": 1,
    "STYLED_ITEM": 3,
    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION": 3,
    "PRODUCT_RELATED_PRODUCT_CATEGORY": 3,
    "GEOMETRIC_REPRESENTATION_CONTEXT": 1,
    "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT": 1,
    "GLOBAL_UNIT_ASSIGNED_CONTEXT": 1,
    "REPRESENTATION_CONTEXT": 2,
    "LENGTH_UNIT": 0,
    "PLANE_ANGLE_UNIT": 0,
    "SOLID_ANGLE_UNIT": 0,
    "NAMED_UNIT": 1,
    "SI_UNIT": 2,
    "CONVERSION_BASED_UNIT": 2,
    "LENGTH_MEASURE_WITH_UNIT": 2,
    "DIMENSIONAL_EXPONENTS": 7,
}


class Model:
    """The parsed file, with the lookups the assertions want."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.text = path.read_text(encoding="utf-8")
        self.header, self.entities = parse_step(self.text)

    def of_type(self, name: str) -> list[Entity]:
        return [e for e in self.entities.values() if name in e.parts]

    def one(self, name: str) -> Entity:
        hits = self.of_type(name)
        if len(hits) != 1:
            raise AssertionError(f"expected exactly one {name}, found {len(hits)}")
        return hits[0]

    def deref(self, value):
        return self.entities[value.id] if isinstance(value, Ref) else value

    # ------------------------------------------------------------ geometry

    def point(self, ref) -> np.ndarray:
        e = self.deref(ref)
        return np.array(e.parts["CARTESIAN_POINT"][1], dtype=float)

    def direction(self, ref) -> np.ndarray:
        e = self.deref(ref)
        return np.array(e.parts["DIRECTION"][1], dtype=float)

    def placement(self, ref) -> tuple[np.ndarray, np.ndarray]:
        """An AXIS2_PLACEMENT_3D as ``(R, t)``.

        Built the way the schema's ``build_axes`` does -- z from the axis, x
        from the reference direction made perpendicular to it, y from their
        cross product -- so a swapped axis in the writer shows up as a rotation
        that does not match the scene.
        """
        e = self.deref(ref)
        origin, axis, ref_dir = e.parts["AXIS2_PLACEMENT_3D"][1:]
        t = self.point(origin)
        z = self.direction(axis)
        z = z / np.linalg.norm(z)
        x = self.direction(ref_dir)
        x = x - z * float(x @ z)
        x = x / np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z]), t

    # ------------------------------------------------------------ assembly

    def products(self) -> dict[str, Entity]:
        out: dict[str, Entity] = {}
        for e in self.of_type("PRODUCT"):
            out[e.parts["PRODUCT"][0]] = e
        return out

    def placements(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Every occurrence in the tree, by its name, as ``(R, t)``.

        This is the five-entity chain walked backwards: the occurrence names the
        node, and its pose is only reachable through the shape of that
        occurrence, the context-dependent shape that refers to it, the
        representation relationship that carries the transform, and finally the
        transform's target placement.
        """
        by_occurrence: dict[int, str] = {}
        for e in self.of_type("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
            by_occurrence[e.id] = e.parts["NEXT_ASSEMBLY_USAGE_OCCURRENCE"][0]

        shape_of: dict[int, str] = {}
        for e in self.of_type("PRODUCT_DEFINITION_SHAPE"):
            target = e.parts["PRODUCT_DEFINITION_SHAPE"][2]
            if isinstance(target, Ref) and target.id in by_occurrence:
                shape_of[e.id] = by_occurrence[target.id]

        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for e in self.of_type("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
            rel_ref, pds_ref = e.parts["CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"]
            name = shape_of.get(pds_ref.id)
            if name is None:
                raise AssertionError(f"#{e.id} refers to a shape with no occurrence")
            rel = self.deref(rel_ref)
            transform = rel.parts["REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"][0]
            idt = self.deref(transform)
            source, target = idt.parts["ITEM_DEFINED_TRANSFORMATION"][2:]
            R0, t0 = self.placement(source)
            if not np.allclose(R0, np.eye(3)) or not np.allclose(t0, 0.0):
                raise AssertionError(f"{name} is placed from a non-identity origin")
            if name in out:
                raise AssertionError(f"{name} is placed twice")
            out[name] = self.placement(target)
        return out

    def representation_items(self, name: str) -> list[Entity]:
        """The items of the shape representation belonging to product ``name``."""
        product = self.products()[name]
        formation = next(
            e
            for e in self.of_type("PRODUCT_DEFINITION_FORMATION")
            if e.parts["PRODUCT_DEFINITION_FORMATION"][2].id == product.id
        )
        definition = next(
            e
            for e in self.of_type("PRODUCT_DEFINITION")
            if e.parts["PRODUCT_DEFINITION"][2].id == formation.id
        )
        shape_of = next(
            e
            for e in self.of_type("PRODUCT_DEFINITION_SHAPE")
            if isinstance(e.parts["PRODUCT_DEFINITION_SHAPE"][2], Ref)
            and e.parts["PRODUCT_DEFINITION_SHAPE"][2].id == definition.id
        )
        sdr = next(
            e
            for e in self.of_type("SHAPE_DEFINITION_REPRESENTATION")
            if e.parts["SHAPE_DEFINITION_REPRESENTATION"][0].id == shape_of.id
        )
        rep = self.deref(sdr.parts["SHAPE_DEFINITION_REPRESENTATION"][1])
        key = next(k for k in rep.parts if k.endswith("SHAPE_REPRESENTATION"))
        return [self.deref(r) for r in rep.parts[key][1]]


# --------------------------------------------------------------------------
# the scene, built here in full
# --------------------------------------------------------------------------

CUBE_OBJ = """# a 40 mm cube, written here so the suite carries its own model
v 0 0 0
v 40 0 0
v 40 40 0
v 0 40 0
v 0 0 40
v 40 0 40
v 40 40 40
v 0 40 40
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""
CUBE_FACES = 12
CUBE_VERTICES = 8

N_CAMERAS = 4
FOCAL_PX = 2000.0
WIDTH, HEIGHT = 4000, 3000


def look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A world-to-camera pose looking from ``eye`` at ``target``."""
    z = target - eye
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    y = -(up - z * float(up @ z))
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.array([x, y, z])
    return R, -R @ eye


def build_scene(root: Path) -> Project:
    """A solved scene with every kind of thing the exporter knows about."""
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "cube.obj").write_text(CUBE_OBJ, encoding="utf-8")

    project = Project(name="stepcase", nominal_units="mm", nominal_scale=0.001)
    project.path = root / "project.multispy"
    project.groups["cam0"] = CameraGroup(
        id="cam0", f=FOCAL_PX, cx=WIDTH / 2.0, cy=HEIGHT / 2.0
    )

    for i in range(N_CAMERAS):
        angle = 2.0 * math.pi * i / N_CAMERAS
        eye = np.array([0.8 * math.cos(angle), 0.8 * math.sin(angle), 0.35])
        R, t = look_at(eye, np.zeros(3))
        project.images.append(
            ImageView(
                id=f"img{i}",
                path=f"photos/IMG_{i:04d}.jpg",
                group="cam0",
                width=WIDTH,
                height=HEIGHT,
                R=R,
                t=t,
            )
        )

    # five tie points, three of them carrying a drawing nominal; one states a
    # single axis only, which must not be exported as a position
    for i in range(5):
        xyz = np.array([0.05 * i, 0.02 * i, 0.1 + 0.01 * i])
        point = TiePoint(id=f"p{i}", label=f"P{i}", xyz=xyz)
        if i < 3:
            point.nominal = [50.0 * i, 20.0 * i, 100.0 + 10.0 * i]
        if i == 4:
            point.nominal = [None, 17.0, None]
        project.points.append(point)

    project.templates["tplpart"] = ObjectTemplate(
        id="tplpart",
        name="widget",
        mesh_path="models/cube.obj",
        units="mm",
        unit_scale=0.001,
        anchors=[
            ObjectAnchor(name="c0", xyz=[0.0, 0.0, 0.0]),
            ObjectAnchor(name="c6", xyz=[40.0, 40.0, 40.0]),
        ],
    )
    project.templates["tplcam"] = ObjectTemplate(
        id="tplcam",
        name="array_camera",
        mesh_path="models/cube.obj",
        units="mm",
        unit_scale=0.001,
        anchors=[ObjectAnchor(name="a", xyz=[0.0, 0.0, 0.0])],
        optical=OpticalFrame(
            center=[20.0, 20.0, 40.0], forward=[0.0, 0.0, 1.0], up=[0.0, -1.0, 0.0], fov_deg=60.0
        ),
    )

    # the part: pinned at the origin by definition
    project.set_part("tplpart", label="widget")
    project.pin_part()

    # two placements of one template, which must share a single mesh
    for i in range(2):
        angle = 0.3 + 0.7 * i
        project.instances.append(
            ObjectInstance(
                id=f"obj{i}",
                template="tplcam",
                label=f"cam_body_{i}",
                R=np.array(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                t=np.array([0.2 + 0.1 * i, -0.15, 0.05 * i]),
            )
        )
    return project


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def check(condition: bool, message: str) -> list[str]:
    return [] if condition else [message]


def test_every_reference_resolves(model: Model) -> list[str]:
    """A dangling ``#n`` is the one fault that makes a file unopenable."""
    bad: list[str] = []
    seen = 0

    def walk(value, owner: int):
        nonlocal seen
        if isinstance(value, Ref):
            seen += 1
            if value.id not in model.entities:
                bad.append(f"#{owner} refers to #{value.id}, which is not in the file")
        elif isinstance(value, Typed):
            for item in value.args:
                walk(item, owner)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, owner)

    for entity in model.entities.values():
        for args in entity.parts.values():
            walk(args, entity.id)
    bad += check(seen > 0, "the file contains no entity references at all")
    return bad


def test_parameter_counts(model: Model) -> list[str]:
    """Every entity carries the number of parameters its schema declares.

    The regression this is here for: item lists written without their
    brackets, which flattened a three-parameter representation into five and
    passed every other check in this file.
    """
    bad: list[str] = []
    for entity in model.entities.values():
        for name, args in entity.parts.items():
            want = ARITY.get(name)
            if want is None:
                bad.append(f"#{entity.id} is a {name}, which this test does not know")
            elif len(args) != want:
                bad.append(
                    f"#{entity.id} {name} has {len(args)} parameters, expected {want}"
                )
    return bad


def test_header(model: Model, schema: str) -> list[str]:
    bad = check(
        schema in model.header.get("FILE_SCHEMA", ""),
        f"FILE_SCHEMA does not name {schema}: {model.header.get('FILE_SCHEMA')!r}",
    )
    bad += check(
        "'2;1'" in model.header.get("FILE_DESCRIPTION", ""),
        "FILE_DESCRIPTION is missing the '2;1' conformance class",
    )
    return bad


def test_tree(model: Model, project: Project, units: str) -> list[str]:
    """Every camera, point, nominal and body is a named node at the right pose."""
    scale = 1.0 / {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}[units]
    tol = 1e-5 * scale
    placed = model.placements()
    bad: list[str] = []

    for im in project.images:
        name = f"cam_{Path(im.path).stem}"
        if name not in placed:
            bad.append(f"{name} is not in the assembly")
            continue
        R, t = placed[name]
        bad += check(
            np.allclose(R, im.R.T, atol=1e-9),
            f"{name} has the wrong orientation",
        )
        bad += check(
            np.allclose(t, im.center() * scale, atol=tol),
            f"{name} is at {t}, expected {im.center() * scale}",
        )

    for point in project.points:
        if point.xyz is None:
            continue
        name = point.label or point.id
        if name not in placed:
            bad.append(f"tie point {name} is not in the assembly")
            continue
        _R, t = placed[name]
        bad += check(
            np.allclose(t, point.xyz * scale, atol=tol),
            f"tie point {name} is at {t}, expected {point.xyz * scale}",
        )

    stated = [
        p
        for p in project.points
        if p.nominal and len(p.nominal) == 3 and all(v is not None for v in p.nominal)
    ]
    for point in stated:
        name = f"{point.label or point.id}_nominal"
        if name not in placed:
            bad.append(f"nominal {name} is not in the assembly")
            continue
        _R, t = placed[name]
        want = np.array(point.nominal, dtype=float) * project.nominal_scale * scale
        bad += check(
            np.allclose(t, want, atol=tol),
            f"nominal {name} is at {t}, expected {want}",
        )
    bad += check(
        not any(
            name.startswith("P4_nominal") for name in placed
        ),
        "a nominal stating only one axis was exported as a position",
    )

    for inst in project.placed_instances():
        name = f"{inst.label} (part)" if inst.is_part else inst.label
        if name not in placed:
            bad.append(f"body {name} is not in the assembly")
            continue
        R, t = placed[name]
        bad += check(np.allclose(R, inst.R, atol=1e-9), f"body {name} is mis-rotated")
        bad += check(
            np.allclose(t, inst.t * scale, atol=tol),
            f"body {name} is at {t}, expected {inst.t * scale}",
        )

    bad += check("MS_Origin" in placed, "the world origin triad is missing")
    return bad


def test_templates_are_shared(model: Model, project: Project) -> list[str]:
    """Two placements of one template must carry one mesh, not two.

    The case this is for is a camera array: ten identical bodies, and a file
    that repeats the mesh ten times is ten times the size for nothing.
    """
    products = model.products()
    bad = check(
        "array_camera" in products,
        "the shared template is not a product in its own right",
    )
    placed = model.placements()
    bad += check(
        "cam_body_0" in placed and "cam_body_1" in placed,
        "the two instances of the shared template are not both placed",
    )
    bad += check(
        len([e for e in model.of_type("PRODUCT") if e.parts["PRODUCT"][0] == "array_camera"])
        == 1,
        "the shared template was written as more than one product",
    )
    return bad


def test_tessellated_mesh(model: Model, project: Project) -> list[str]:
    """The triangle block says what it holds, and indexes inside it."""
    bad: list[str] = []
    coords = model.of_type("COORDINATES_LIST")
    bad += check(bool(coords), "no COORDINATES_LIST in a tessellated export")
    for entity in coords:
        _name, declared, points = entity.parts["COORDINATES_LIST"]
        bad += check(
            declared == len(points),
            f"#{entity.id} declares {declared} points and lists {len(points)}",
        )
        bad += check(
            all(len(p) == 3 for p in points),
            f"#{entity.id} has a coordinate that is not a triple",
        )
        bad += check(
            declared == CUBE_VERTICES,
            f"#{entity.id} holds {declared} points, expected the cube's {CUBE_VERTICES}",
        )

    faces = model.of_type("TRIANGULATED_FACE")
    bad += check(bool(faces), "no TRIANGULATED_FACE in a tessellated export")
    for entity in faces:
        args = entity.parts["TRIANGULATED_FACE"]
        clist = model.deref(args[1])
        npoints = clist.parts["COORDINATES_LIST"][1]
        pnmax, normals, link, pnindex, triangles = args[2:]
        bad += check(
            pnmax == npoints,
            f"#{entity.id} pnmax is {pnmax} against {npoints} coordinates, which "
            "breaks the rule that an empty pnindex requires the two to agree",
        )
        bad += check(pnindex == [], f"#{entity.id} has a pnindex this writer never sets")
        bad += check(link is None, f"#{entity.id} has a geometric_link it cannot honour")
        bad += check(normals == [], f"#{entity.id} has normals this writer never sets")
        bad += check(
            len(triangles) == CUBE_FACES,
            f"#{entity.id} has {len(triangles)} triangles, expected {CUBE_FACES}",
        )
        flat = [i for tri in triangles for i in tri]
        bad += check(
            all(len(tri) == 3 for tri in triangles),
            f"#{entity.id} has a triangle that is not a triple",
        )
        bad += check(
            min(flat) >= 1 and max(flat) <= npoints,
            f"#{entity.id} indexes outside 1..{npoints}, so the list is not 1-based",
        )

    shells = model.of_type("TESSELLATED_SHELL")
    bad += check(
        len(shells) == len({t.id for t in faces}) and bool(shells),
        "every triangulated face should sit in exactly one tessellated shell",
    )
    # the body is in a representation that declares itself tessellated
    reps = model.of_type("TESSELLATED_SHAPE_REPRESENTATION")
    bad += check(
        bool(reps),
        "tessellated geometry is not in a TESSELLATED_SHAPE_REPRESENTATION",
    )
    return bad


def test_mesh_is_in_file_units(model: Model, project: Project, units: str) -> list[str]:
    """A 40 mm cube is 40 in a millimetre file and 1.5748 in an inch one."""
    per_unit = {"mm": 1.0, "cm": 0.1, "m": 0.001, "in": 1.0 / 25.4, "ft": 1.0 / 304.8}[
        units
    ]
    want = 40.0 * per_unit
    bad: list[str] = []
    for entity in model.of_type("COORDINATES_LIST"):
        pts = np.array(entity.parts["COORDINATES_LIST"][2], dtype=float)
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        bad += check(
            abs(span - want) < 1e-6 * max(want, 1.0),
            f"#{entity.id} spans {span} in {units}, expected {want}",
        )
    return bad


def test_wireframe(model: Model, project: Project) -> list[str]:
    """Wireframe mode writes edges and no surfaces at all."""
    bad = check(
        not model.of_type("TRIANGULATED_FACE") and not model.of_type("FACE_SURFACE"),
        "wireframe mode wrote face geometry",
    )
    # a cube's twelve edges are all creases, so they must all be there
    body = model.representation_items("widget")
    sets = [e for e in body if "GEOMETRIC_CURVE_SET" in e.parts]
    bad += check(bool(sets), "the body has no curve set in wireframe mode")
    if sets:
        lines = [
            model.deref(r)
            for r in sets[0].parts["GEOMETRIC_CURVE_SET"][1]
        ]
        bad += check(
            len(lines) == 12,
            f"the cube came out with {len(lines)} feature edges, expected 12",
        )
    return bad


def test_faceted(model: Model, project: Project) -> list[str]:
    """Brep mode writes a real closed solid, one planar face per triangle."""
    bad: list[str] = []
    breps = model.of_type("FACETED_BREP")
    bad += check(bool(breps), "brep mode wrote no FACETED_BREP")
    for entity in breps:
        shell = model.deref(entity.parts["FACETED_BREP"][1])
        faces = shell.parts["CLOSED_SHELL"][1]
        bad += check(
            len(faces) == CUBE_FACES,
            f"#{entity.id} has {len(faces)} faces, expected {CUBE_FACES}",
        )
        for face_ref in faces:
            face = model.deref(face_ref)
            bounds = face.parts["FACE_SURFACE"][1]
            bad += check(len(bounds) == 1, f"#{face.id} has {len(bounds)} bounds")
            loop = model.deref(model.deref(bounds[0]).parts["FACE_OUTER_BOUND"][1])
            verts = loop.parts["POLY_LOOP"][1]
            bad += check(
                len(verts) == 3, f"#{loop.id} is a loop of {len(verts)}, expected 3"
            )
            # the loop's points must lie in the face's own plane, or the solid
            # is self-contradictory however well it parses
            plane = model.deref(face.parts["FACE_SURFACE"][2])
            R, origin = model.placement(plane.parts["PLANE"][1])
            normal = R[:, 2]
            for v in verts:
                off = abs(float((model.point(v) - origin) @ normal))
                if off > 1e-6:
                    bad.append(f"#{loop.id} has a point {off} off its own plane")
                    break
    return bad


def test_colours(model: Model, project: Project) -> list[str]:
    """Each kind of thing carries its own colour, and they are all used."""
    bad: list[str] = []
    rgb = {
        tuple(
            round(c * 255.0)
            for c in e.parts["COLOUR_RGB"][1:]
        )
        for e in model.of_type("COLOUR_RGB")
    }
    for kind in ("camera", "point", "nominal", "origin", "part", "object", "anchor"):
        want = export.STEP_COLOURS[kind]
        bad += check(want in rgb, f"the {kind} colour {want} is not in the file")
    styled = model.of_type("STYLED_ITEM")
    bad += check(bool(styled), "nothing in the file is styled")
    presentation = model.of_type(
        "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
    )
    bad += check(
        len(presentation) == 1,
        "the styled items are not collected into one presentation",
    )
    if presentation:
        held = {r.id for r in presentation[0].args[1]}
        bad += check(
            held == {e.id for e in styled},
            "some styled items are not in the presentation",
        )
    return bad


def test_units(model: Model, units: str) -> list[str]:
    """The unit context says what the numbers mean."""
    bad: list[str] = []
    si = [e for e in model.of_type("SI_UNIT") if "LENGTH_UNIT" in e.parts]
    imperial = {"in": ("INCH", 25.4), "ft": ("FOOT", 304.8)}
    if units in imperial:
        want_name, want_mm = imperial[units]
        conv = [e for e in model.of_type("CONVERSION_BASED_UNIT")]
        bad += check(bool(conv), f"a {units} file carries no CONVERSION_BASED_UNIT")
        if conv:
            name = conv[0].parts["CONVERSION_BASED_UNIT"][0]
            factor = model.deref(conv[0].parts["CONVERSION_BASED_UNIT"][1])
            measure = factor.parts["LENGTH_MEASURE_WITH_UNIT"][0]
            bad += check(
                name == want_name,
                f"the conversion unit is called {name!r}, expected {want_name!r}",
            )
            bad += check(
                isinstance(measure, Typed) and abs(measure.args[0] - want_mm) < 1e-12,
                f"one {units} is stated as {measure}, not {want_mm} mm",
            )
            # the unit it converts against must itself be the millimetre
            base = model.deref(factor.parts["LENGTH_MEASURE_WITH_UNIT"][1])
            prefix, metre = base.parts["SI_UNIT"]
            bad += check(
                isinstance(prefix, Enum)
                and prefix.name == "MILLI"
                and isinstance(metre, Enum)
                and metre.name == "METRE",
                f"the conversion is against {prefix} {metre}, not the millimetre",
            )
    else:
        prefix = {"mm": "MILLI", "cm": "CENTI", "m": None}[units]
        bad += check(len(si) == 1, f"expected one length SI_UNIT, found {len(si)}")
        if si:
            got, metre = si[0].parts["SI_UNIT"]
            bad += check(
                isinstance(metre, Enum) and metre.name == "METRE",
                f"the length unit is not a metre: {metre}",
            )
            if prefix is None:
                bad += check(got is None, f"metres should carry no prefix, got {got}")
            else:
                bad += check(
                    isinstance(got, Enum) and got.name == prefix,
                    f"expected the {prefix} prefix, got {got}",
                )
    context = model.of_type("GLOBAL_UNIT_ASSIGNED_CONTEXT")
    bad += check(len(context) == 1, "there is not exactly one unit context")
    if context:
        bad += check(
            len(context[0].parts["GLOBAL_UNIT_ASSIGNED_CONTEXT"][0]) == 3,
            "the unit context does not assign length, angle and solid angle",
        )
    return bad


def test_refusals(root: Path, project: Project) -> list[str]:
    """What it declines to do, and whether it says why."""
    bad: list[str] = []

    def refuses(fn, fragment: str, what: str):
        try:
            fn()
        except ValueError as exc:
            if fragment not in str(exc):
                bad.append(f"{what} raised {exc!r}, which does not mention {fragment!r}")
        else:
            bad.append(f"{what} was accepted")

    refuses(
        lambda: export.export_step(project, root / "x.step", shape="solid"),
        "shape must be one of",
        "an unknown shape mode",
    )
    refuses(
        lambda: export.export_step(project, root / "x.step", units="furlong"),
        "units must be one of",
        "an unknown unit",
    )
    refuses(
        lambda: export.export_step(Project(name="empty"), root / "x.step"),
        "nothing solved yet",
        "an empty project",
    )
    refuses(
        lambda: export.export_step(
            project, root / "x.step", shape="brep", faceted_limit=4
        ),
        "past the 4",
        "a mesh over the faceted limit",
    )
    return bad


def test_missing_mesh_still_places_the_body(root: Path) -> list[str]:
    """A body whose model file has gone keeps its node and its pose.

    Losing the pose because the mesh moved would be the worst of both: the
    measurement is in the file and the file does not say where it was.
    """
    project = build_scene(root)
    project.templates["tplcam"].mesh_path = "models/gone.obj"
    out = export.export_step(project, root / "missing.step")
    model = Model(out)
    placed = model.placements()
    bad = check(
        "cam_body_0" in placed and "cam_body_1" in placed,
        "a body with no mesh file lost its placement",
    )
    bad += check(
        "array_camera" in model.products(),
        "a body with no mesh file lost its node",
    )
    items = model.representation_items("array_camera")
    bad += check(
        any("GEOMETRIC_CURVE_SET" in e.parts for e in items),
        "a body with no mesh kept neither its anchors nor its optical frame",
    )
    return bad


def test_front_end_paths(root: Path) -> list[str]:
    """The two routes the buttons take, which no CAD check would reach.

    The STEP button is not a thin wrapper over ``export_step``: in server mode
    it asks for one format out of the set, and in the browser it asks for the
    bytes with no folder to put them in. Both deliver a different thing from
    what the CLI does, and both are what a user actually presses.
    """
    from multispy import webapp

    project = build_scene(root)
    bad: list[str] = []

    # server mode: one format, so one file -- not the whole export set
    out = root / "one"
    written = webapp.write_exports(project, out, formats=["step"])
    bad += check(
        list(written) == ["step"],
        f"asking for step alone wrote {list(written)}",
    )
    landed = sorted(p.name for p in out.iterdir())
    bad += check(
        landed == ["scene.step"],
        f"asking for step alone left {landed} in the folder",
    )
    bad += check(
        "AP242" in Model(out / "scene.step").header["FILE_SCHEMA"],
        "the default shape did not write AP242",
    )

    # the shape has to survive the trip, or the shift-click does nothing
    webapp.write_exports(project, out, formats=["step"], step_shape="wireframe")
    model = Model(out / "scene.step")
    bad += check(
        "AUTOMOTIVE_DESIGN" in model.header["FILE_SCHEMA"],
        "step_shape='wireframe' still wrote the AP242 default",
    )
    bad += check(
        not model.of_type("TRIANGULATED_FACE"),
        "step_shape='wireframe' still wrote tessellated geometry",
    )

    # browser mode: bytes, no folder, and callable more than once
    work = root / "bytes"
    first = webapp.step_bytes(project, work_dir=work)
    second = webapp.step_bytes(project, work_dir=work)
    bad += check(
        first.startswith(b"ISO-10303-21;"),
        "step_bytes did not return a Part 21 file",
    )
    bad += check(
        first == second,
        "step_bytes gave a different answer the second time it was called",
    )
    wire = webapp.step_bytes(project, "wireframe", work_dir=work)
    bad += check(
        b"AUTOMOTIVE_DESIGN" in wire and b"AP242" not in wire,
        "step_bytes ignored the shape it was given",
    )
    bad += check(
        not list(work.glob("*.step")),
        "step_bytes left its scratch file behind",
    )
    # and what it returns has to be a real file, not just bytes that start right
    scratch = root / "from_bytes.step"
    scratch.write_bytes(first)
    bad += test_every_reference_resolves(Model(scratch))
    return bad


def test_optical_frame(root: Path) -> list[str]:
    """A declared optical frame is drawn where the template says it is."""
    project = build_scene(root)
    out = export.export_step(project, root / "optical.step")
    model = Model(out)
    items = model.representation_items("array_camera")
    sets = [e for e in items if "GEOMETRIC_CURVE_SET" in e.parts]
    bad = check(
        len(sets) == 2,
        f"expected an anchor set and an optical set, found {len(sets)}",
    )
    # the frustum's apex is the entrance pupil, in file units
    apexes = []
    for s in sets:
        for ref in s.parts["GEOMETRIC_CURVE_SET"][1]:
            line = model.deref(ref)
            if "POLYLINE" not in line.parts:
                continue
            pts = [model.point(p) for p in line.parts["POLYLINE"][1]]
            apexes.append(pts[0])
    want = np.array([20.0, 20.0, 40.0])  # mm, and the file is in mm
    bad += check(
        any(np.allclose(a, want, atol=1e-6) for a in apexes),
        f"no frustum apex at the declared pupil {want}",
    )
    return bad


# --------------------------------------------------------------------------


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="multispy_step_"))
    failures: list[str] = []
    try:
        project = build_scene(root)

        print("tessellated, in millimetres")
        out = export.export_step(project, root / "scene.step")
        model = Model(out)
        for name, result in (
            ("references resolve", test_every_reference_resolves(model)),
            ("parameter counts", test_parameter_counts(model)),
            ("header", test_header(model, "AP242")),
            ("assembly tree", test_tree(model, project, "mm")),
            ("shared templates", test_templates_are_shared(model, project)),
            ("tessellated mesh", test_tessellated_mesh(model, project)),
            ("mesh in file units", test_mesh_is_in_file_units(model, project, "mm")),
            ("colours", test_colours(model, project)),
            ("units", test_units(model, "mm")),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result
        print(f"  {out.stat().st_size / 1024:.1f} KiB, {len(model.entities)} entities")

        for units in ("cm", "m", "in", "ft"):
            print(f"tessellated, in {units}")
            out = export.export_step(project, root / f"scene_{units}.step", units=units)
            model = Model(out)
            for name, result in (
                ("references resolve", test_every_reference_resolves(model)),
                ("parameter counts", test_parameter_counts(model)),
                ("assembly tree", test_tree(model, project, units)),
                ("mesh in file units", test_mesh_is_in_file_units(model, project, units)),
                ("units", test_units(model, units)),
            ):
                print(f"  {'ok  ' if not result else 'FAIL'} {name}")
                failures += result

        print("wireframe")
        out = export.export_step(project, root / "wire.step", shape="wireframe")
        model = Model(out)
        for name, result in (
            ("references resolve", test_every_reference_resolves(model)),
            ("parameter counts", test_parameter_counts(model)),
            ("header", test_header(model, "AUTOMOTIVE_DESIGN")),
            ("assembly tree", test_tree(model, project, "mm")),
            ("edges only", test_wireframe(model, project)),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result

        print("brep")
        out = export.export_step(project, root / "brep.step", shape="brep")
        model = Model(out)
        for name, result in (
            ("references resolve", test_every_reference_resolves(model)),
            ("parameter counts", test_parameter_counts(model)),
            ("header", test_header(model, "AUTOMOTIVE_DESIGN")),
            ("assembly tree", test_tree(model, project, "mm")),
            ("faceted solid", test_faceted(model, project)),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result

        print("placements only")
        out = export.export_step(project, root / "none.step", shape="none")
        model = Model(out)
        for name, result in (
            ("references resolve", test_every_reference_resolves(model)),
            ("parameter counts", test_parameter_counts(model)),
            ("assembly tree", test_tree(model, project, "mm")),
            (
                "no body geometry",
                check(
                    not model.of_type("TRIANGULATED_FACE")
                    and not model.of_type("FACETED_BREP"),
                    "shape='none' wrote body geometry",
                ),
            ),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result

        print("edges")
        for name, result in (
            ("refusals", test_refusals(root, project)),
            ("a missing mesh file", test_missing_mesh_still_places_the_body(root)),
            ("optical frame", test_optical_frame(root)),
            ("the front end's two routes", test_front_end_paths(root)),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    for message in failures:
        print(f"  fail: {message}")
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
