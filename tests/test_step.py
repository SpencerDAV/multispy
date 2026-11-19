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
    ReferencePlane,
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
    "OPEN_SHELL": 2,
    "SHELL_BASED_SURFACE_MODEL": 2,
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
    # a cube's twelve edges are all creases, so they must all be there -- and
    # they live in the body node now, not in the template that locates it
    body = model.representation_items("widget_body")
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


def test_geometry_is_not_shared(model: Model) -> list[str]:
    """No geometric entity belongs to two shape representations at once.

    A representation_item belongs to one representation, and in an assembly
    every representation sits in its own coordinate system -- so a
    CARTESIAN_POINT reachable from two of them is a point claiming to be in
    two frames at the same time. The writer dedupes coordinates, which is how
    this got in: every drawing nominal is a cross at its own origin, so all of
    them shared one set of six points and differed only in their placement.
    OpenCASCADE recovered the scene from that file regardless. CATIA V5 read
    it to the end of the progress bar and then closed.
    """

    def reachable(entity: Entity) -> list[int]:
        out: list[int] = []

        def walk(value):
            if isinstance(value, Ref):
                out.append(value.id)
            elif isinstance(value, Typed):
                for item in value.args:
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        for args in entity.parts.values():
            walk(args)
        return out

    reps = [
        e
        for e in model.entities.values()
        if {"SHAPE_REPRESENTATION", "TESSELLATED_SHAPE_REPRESENTATION"} & set(e.parts)
    ]
    bad = check(bool(reps), "the file holds no shape representation")
    owners: dict[int, set[int]] = {}
    for rep in reps:
        # (name, items, context): only the items are the representation's own
        stack = [r.id for r in rep.args[1] if isinstance(r, Ref)]
        seen: set[int] = set()
        while stack:
            eid = stack.pop()
            if eid in seen or eid not in model.entities:
                continue
            seen.add(eid)
            stack += reachable(model.entities[eid])
        for eid in seen:
            owners.setdefault(eid, set()).add(rep.id)
    shared = sorted(eid for eid, who in owners.items() if len(who) > 1)
    for eid in shared[:5]:
        bad.append(
            "#%d (%s) is reachable from %d shape representations"
            % (eid, model.entities[eid].name, len(owners[eid]))
        )
    bad += check(
        not shared,
        f"{len(shared)} entities are shared between shape representations",
    )
    return bad


def test_styles_apply_to_their_items(model: Model) -> list[str]:
    """A curve is styled with a pen, a surface with a fill, neither with both.

    A SURFACE_STYLE_USAGE assigned to a GEOMETRIC_CURVE_SET is a style for
    something the item has none of. Presentation is the last thing a reader
    does with a file, which is why this one reads as a crash at the end of an
    import that looked like it was working.
    """
    curves = {"POLYLINE", "GEOMETRIC_CURVE_SET", "LINE", "CIRCLE"}
    surfaces = {"FACETED_BREP", "TESSELLATED_SHELL", "SHELL_BASED_SURFACE_MODEL"}
    styled = model.of_type("STYLED_ITEM")
    bad = check(bool(styled), "nothing in the file is styled")
    for item in styled:
        _name, assignments, target = item.args
        applied: set[str] = set()
        for assignment in assignments:
            for style in model.deref(assignment).args[0]:
                applied |= set(model.deref(style).parts)
        subject = model.deref(target)
        kinds = set(subject.parts)
        if kinds & curves and "SURFACE_STYLE_USAGE" in applied:
            bad.append(f"#{item.id} gives a surface style to a {subject.name}")
        if kinds & surfaces and "CURVE_STYLE" in applied:
            bad.append(f"#{item.id} gives a curve style to a {subject.name}")
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
    out = export.export_step(project, root / "missing.step", shape="tessellated")
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
    kept = [
        name
        for name in ("array_camera_anchors", "array_camera_optical")
        if name in model.products()
    ]
    bad += check(
        bool(kept),
        "a body with no mesh kept neither its anchors nor its optical frame",
    )
    bad += check(
        "array_camera_body" not in model.products(),
        "a template with no mesh file still produced a body node",
    )

    # ...and it has to SAY so. This is the half that was missing, and it is
    # the half the user met: the export succeeded, the file opened, every
    # camera and every anchor cross was in it, and the bodies simply were not.
    # Nothing anywhere named the reason, so the file read as an exporter that
    # cannot carry meshes rather than as one that could not find them.
    reported: list[str] = []
    out2 = export.export_step(
        project, root / "missing_reported.step", shape="tessellated", skipped=reported
    )
    bad += check(
        any("array_camera" in line for line in reported),
        f"a skipped body was not reported: {reported}",
    )
    bad += check(
        any("missing" in line for line in reported),
        f"the report does not say the file is missing: {reported}",
    )
    header = out2.read_text(encoding="utf-8", errors="replace").split("DATA;")[0]
    bad += check(
        "WARNING" in header and "array_camera" in header,
        "the STEP header does not warn that a body has no geometry in it",
    )

    # and the same export with its mesh present must stay quiet, or the
    # warning is noise and will be ignored on the day it matters
    clean: list[str] = []
    ok = build_scene(root)
    out3 = export.export_step(
        ok, root / "present.step", shape="tessellated", skipped=clean
    )
    bad += check(not clean, f"a healthy export reported skipped bodies: {clean}")
    bad += check(
        "WARNING" not in out3.read_text(encoding="utf-8", errors="replace").split("DATA;")[0],
        "a healthy export warns about missing geometry anyway",
    )
    return bad


def test_bodies_are_swappable(model: Model, project: Project) -> list[str]:
    """A body is a leaf of its own, under the transform that located it.

    This is the structure the export exists for. The mesh sits at the identity
    in the model file's own coordinates, and the one thing between it and part
    space is a rigid transform on the node above. Flatten those two together
    and replacing a CAD model with a better one of the same part means
    rebuilding the placement by hand -- which is the measurement, and the only
    expensive thing in the file.
    """
    bad: list[str] = []
    placed = model.placements()
    products = model.products()

    for template in ("widget", "array_camera"):
        body = f"{template}_body"
        bad += check(body in products, f"{body} is not a product of its own")
        bad += check(body in placed, f"{body} is not placed under its template")
        if body not in placed:
            continue
        R, t = placed[body]
        # identity, or the geometry is not in the model's own frame any more
        bad += check(
            np.allclose(R, np.eye(3), atol=1e-12) and np.allclose(t, 0.0, atol=1e-12),
            f"{body} sits at {t} rather than at its model origin",
        )
        # and the body node holds geometry while the node above holds none
        items = model.representation_items(template)
        geometry = [
            e
            for e in items
            if any(
                k in e.parts
                for k in (
                    "TESSELLATED_SHELL",
                    "FACETED_BREP",
                    "GEOMETRIC_CURVE_SET",
                    "SHELL_BASED_SURFACE_MODEL",
                )
            )
        ]
        bad += check(
            not geometry,
            f"{template} carries geometry itself; it should only carry its children",
        )

    # the mesh is in model-local coordinates, so a 40 mm cube is still 40 mm
    # wide inside its body node however its instance was rotated
    for entity in model.of_type("COORDINATES_LIST"):
        pts = np.array(entity.parts["COORDINATES_LIST"][2], dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        bad += check(
            np.allclose(lo, 0.0, atol=1e-6) and np.allclose(hi, 40.0, atol=1e-6),
            f"#{entity.id} spans {lo}..{hi}; the cube is no longer in its own frame",
        )
    return bad


def test_unplaced_templates(root: Path) -> list[str]:
    """A template nobody has placed is still in the file, and says so."""
    project = build_scene(root)
    project.templates["spare"] = ObjectTemplate(
        id="spare",
        name="spare_bracket",
        mesh_path="models/cube.obj",
        units="mm",
        unit_scale=0.001,
    )
    out = export.export_step(project, root / "unplaced.step", shape="tessellated")
    model = Model(out)
    placed = model.placements()
    bad = check(
        "spare_bracket (unplaced)" in placed,
        f"an unplaced template is missing from the tree: {sorted(placed)}",
    )
    if "spare_bracket (unplaced)" in placed:
        R, t = placed["spare_bracket (unplaced)"]
        bad += check(
            np.allclose(R, np.eye(3)) and np.allclose(t, 0.0),
            "an unplaced template was given a pose it has not got",
        )
    bad += check(
        "spare_bracket_body" in model.products(),
        "an unplaced template brought no body with it",
    )

    # and it can be turned off, for a file that states only what was measured
    strict = Model(
        export.export_step(
            project, root / "strict.step", shape="tessellated", unplaced=False
        )
    )
    bad += check(
        not any("unplaced" in name for name in strict.placements()),
        "unplaced=False still wrote unplaced templates",
    )
    return bad


def test_reference_planes(root: Path) -> list[str]:
    """Datums come through as real planar surfaces, where the drawing puts them."""
    project = build_scene(root)
    project.reference_planes = [
        ReferencePlane(id="pl0", label="datum A", normal=[0, 0, 1], offset=995.0),
        ReferencePlane(id="pl1", label="datum B", normal=[1, 0, 0], offset=1510.0, size=400.0),
        ReferencePlane(id="pl2", normal=[1, 1, 0], offset=100.0, size=50.0),
    ]
    out = export.export_step(project, root / "planes.step")
    model = Model(out)
    bad: list[str] = []

    faces = model.of_type("SHELL_BASED_SURFACE_MODEL")
    bad += check(
        len(faces) == 3, f"expected 3 reference surfaces, found {len(faces)}"
    )
    products = model.products()
    for name in ("datum A", "datum B"):
        bad += check(name in products, f"{name} is not a named product")

    # the unnamed one falls back to stating itself
    bad += check(
        any("=" in name or "n=" in name for name in products),
        "a plane with no label did not fall back to describing itself",
    )

    # geometry: nominal units (mm here) land on the same numbers as a nominal
    by_name = {}
    for entity in model.of_type("PLANE"):
        R, origin = model.placement(entity.parts["PLANE"][1])
        by_name[entity.parts["PLANE"][0]] = (R[:, 2], origin)

    for name, normal, point in (
        ("datum A", [0, 0, 1], [0, 0, 995]),
        ("datum B", [1, 0, 0], [1510, 0, 0]),
    ):
        if name not in by_name:
            bad.append(f"{name} has no PLANE surface")
            continue
        n, o = by_name[name]
        bad += check(np.allclose(n, normal, atol=1e-9), f"{name} normal is {n}")
        bad += check(
            abs(float((o - np.array(point, dtype=float)) @ np.array(normal, dtype=float)))
            < 1e-6,
            f"{name} passes through {o}, not {point}",
        )

    # a bounded face, or there is nothing for CAD to select
    for entity in model.of_type("FACE_OUTER_BOUND"):
        loop = model.deref(entity.parts["FACE_OUTER_BOUND"][1])
        if "POLY_LOOP" not in loop.parts:
            continue
        verts = loop.parts["POLY_LOOP"][1]
        if len(verts) == 4:
            break
    else:
        bad.append("no four-cornered face bound; the planes are not patches")

    # size: the one given 400 must measure 800 across
    sized = None
    for entity in model.of_type("FACE_SURFACE"):
        if entity.parts["FACE_SURFACE"][0] != "datum B":
            continue
        bound = model.deref(entity.parts["FACE_SURFACE"][1][0])
        loop = model.deref(bound.parts["FACE_OUTER_BOUND"][1])
        pts = np.array([model.point(v) for v in loop.parts["POLY_LOOP"][1]])
        sized = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    bad += check(
        sized is not None and abs(sized - 800.0) < 1e-6,
        f"datum B measures {sized} across, expected 800",
    )

    # and they can be left out
    without = Model(export.export_step(project, root / "noplanes.step", planes=False))
    bad += check(
        not without.of_type("SHELL_BASED_SURFACE_MODEL"),
        "planes=False still wrote reference surfaces",
    )
    return bad


def test_drawn_plane(root: Path) -> list[str]:
    """A surface drawn on the photographs: fitted to its corners, exported as them.

    The three things that can go quietly wrong, in the order they would bite.
    The fit has to find the plane the corners actually lie in, including its
    tilt -- an axis-aligned answer would look plausible and be wrong. The
    corners have to come out of the fit exactly coplanar, because a CAD face
    cannot be nearly planar and the alternative to projecting them is a warped
    or rejected surface. And the face in the file has to be bounded by those
    four corners rather than by a tidy square around their middle, which for a
    fence would reach past both ends.
    """
    from multispy.solve import SolveReport, fit_drawn_planes

    project = build_scene(root)
    project.nominal_scale = 0.001
    project.nominal_units = "mm"

    # a deliberately tilted square, one metre across, with one corner 2 mm
    # proud of the other three -- so there is something for the fit to resolve
    # and something for it to report
    n = np.array([0.0, -0.3, 1.0])
    n /= np.linalg.norm(n)
    u = np.array([1.0, 0.0, 0.0])
    u -= n * (u @ n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    o = np.array([0.5, 0.2, 1.0])
    truth = [o - u * 0.5 - v * 0.5, o + u * 0.5 - v * 0.5,
             o + u * 0.5 + v * 0.5, o - u * 0.5 + v * 0.5]
    marked = list(truth)
    marked[2] = marked[2] + n * 0.002

    ids = []
    for i, c in enumerate(marked):
        pid = f"corner{i}"
        project.points.append(TiePoint(id=pid, label=pid, xyz=np.asarray(c)))
        ids.append(pid)
    project.reference_planes = [ReferencePlane(id="pl0", label="fence", corners=ids)]

    report = SolveReport()
    fitted = fit_drawn_planes(project, report)
    plane = project.reference_planes[0]
    bad = check(fitted == 1, f"fitted {fitted} drawn plane(s), expected 1")
    bad += check(plane.is_drawn, "a plane with four corners did not report as drawn")

    # Within a couple of milliradians of the truth, and not tighter: the fit is
    # through the *marked* corners, one of which is 2 mm proud, so the best
    # plane through them genuinely is tilted about a milliradian off the one
    # they were generated from. Either sign passes -- which way a fitted
    # surface faces is decided by where the cameras are, and this fixture has
    # no opinion about that.
    tilt = math.degrees(math.acos(min(1.0, abs(float(plane.unit_normal() @ n)))))
    bad += check(
        tilt < 0.2,
        f"fitted normal {plane.unit_normal()} is {tilt:.3f}° off the plane the corners "
        f"lie in ({n}); an axis-aligned answer would look like this",
    )
    bad += check(
        plane.flatness is not None and abs(plane.flatness - 0.5) < 0.05,
        f"flatness came out {plane.flatness}, expected about 0.5 mm",
    )

    # every exported corner must lie exactly on the surface that was reported
    un = plane.unit_normal()
    off = [
        abs(float(np.asarray(c) * project.nominal_scale @ un)
            - plane.offset * project.nominal_scale)
        for c in plane.quad
    ]
    bad += check(
        max(off) < 1e-9,
        f"a corner sits {max(off):.2e} m off its own plane; the face is not planar",
    )

    out = export.export_step(project, root / "drawn.step")
    text = out.read_text(encoding="utf-8")
    bad += check("PLANE('fence'" in text, "the drawn surface is not named in the file")

    loop = re.search(r"POLY_LOOP\('',\(([^)]*)\)\)", text)
    bad += check(loop is not None, "no bounding loop was written for the drawn surface")
    if loop is not None:
        coords = []
        for ref in loop.group(1).split(","):
            vid = ref.strip().lstrip("#")
            vp = re.search(rf"#{vid}\s*=\s*VERTEX_POINT\('',#(\d+)\)", text)
            pid = vp.group(1) if vp else vid
            m = re.search(rf"#{pid}\s*=\s*CARTESIAN_POINT\('',\(([^)]*)\)\)", text)
            if m:
                coords.append([float(x) for x in m.group(1).split(",")])
        bad += check(
            len(coords) == 4, f"the face is bounded by {len(coords)} corners, expected 4"
        )
        for want in plane.quad:
            bad += check(
                any(np.allclose(want, got, atol=1e-6) for got in coords),
                f"corner {want} is not in the exported face, so it was squared off",
            )
    return bad


def test_drawn_plane_refuses_what_it_cannot_measure(root: Path) -> list[str]:
    """The two ways a drawn surface can look perfect while being nonsense.

    Both were found by reading the numbers rather than by anything failing.

    A corner that did not triangulate used to be skipped, so a four-corner
    panel missing one came out as a *triangle* -- and three points are exactly
    coplanar by definition, so it reported a flatness of zero. Wrong about its
    own shape, claiming perfection while doing it.

    Corners strung out along a line are the same trap by another route: every
    plane containing that line fits them exactly, so the residual is zero and
    the normal is whatever the arithmetic happened to land on. Neither case can
    be caught by looking at the residual, because the residual is the thing
    they fake.
    """
    from multispy.solve import SolveReport, fit_drawn_planes

    bad: list[str] = []

    # --- a corner with no position
    project = build_scene(root)
    ids = []
    for i, c in enumerate([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]):
        pid = f"gap{i}"
        # the third corner never triangulated
        xyz = None if i == 2 else np.array(c, dtype=float)
        project.points.append(TiePoint(id=pid, label=pid, xyz=xyz))
        ids.append(pid)
    project.reference_planes = [ReferencePlane(id="pl0", label="gappy", corners=ids)]
    report = SolveReport()
    fit_drawn_planes(project, report)
    plane = project.reference_planes[0]
    bad += check(
        plane.quad is None,
        "a surface with an unsolved corner was fitted anyway, as a triangle",
    )
    bad += check(
        plane.flatness is None,
        f"it also reported a flatness of {plane.flatness}, which nothing measured",
    )
    bad += check(
        any("gap2" in w for w in report.warnings),
        f"nothing named the missing corner; warnings were {report.warnings}",
    )

    # --- corners along a line
    project = build_scene(root)
    direction = np.array([1.0, 0.3, 0.1])
    direction /= np.linalg.norm(direction)
    ids = []
    for i, t in enumerate([0.0, 0.33, 0.66, 1.0]):
        pid = f"rail{i}"
        xyz = np.array([0.0, 0.0, 1.0]) + direction * t
        xyz = xyz + np.array([0.0, 0.0, 0.0005 * (i % 2)])   # half a millimetre of wobble
        project.points.append(TiePoint(id=pid, label=pid, xyz=xyz))
        ids.append(pid)
    project.reference_planes = [ReferencePlane(id="pl0", label="rail", corners=ids)]
    report = SolveReport()
    fit_drawn_planes(project, report)
    plane = project.reference_planes[0]
    bad += check(
        plane.quad is None,
        "corners lying along a line were fitted to a plane, which cannot be determined",
    )
    bad += check(
        any("line" in w for w in report.warnings),
        f"nothing said why; warnings were {report.warnings}",
    )
    return bad


def test_drawn_planes_share_corners(root: Path) -> list[str]:
    """Two surfaces meeting along an edge share that edge's corner points.

    The alternative is marking one physical feature twice and hoping the two
    marks triangulate to the same place, which they do not: the seam between
    the panels is then as wide as the two clicks disagree. Sharing the point
    removes that error.

    It does *not* make the edge exact, and this checks the honest version:
    each surface still projects the shared corner onto its own plane to keep
    its own face flat, so two surfaces that are not quite coplanar put it in
    two places. What is asserted here is that the gap is bounded by how far
    the planes are from each other, not by a second independent click.
    """
    from multispy.solve import SolveReport, fit_drawn_planes

    project = build_scene(root)
    # two panels folded along a shared edge, at a slight angle to each other
    shared = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])]
    left = [np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 1.0, 1.0])]
    right = [np.array([1.0, 0.0, 1.05]), np.array([1.0, 1.0, 1.05])]

    named = {}
    for name, xyz in [
        ("s0", shared[0]), ("s1", shared[1]),
        ("l0", left[0]), ("l1", left[1]),
        ("r0", right[0]), ("r1", right[1]),
    ]:
        project.points.append(TiePoint(id=name, label=name, xyz=xyz))
        named[name] = xyz

    project.reference_planes = [
        ReferencePlane(id="pl0", label="left", corners=["l0", "s0", "s1", "l1"]),
        ReferencePlane(id="pl1", label="right", corners=["s0", "r0", "r1", "s1"]),
    ]
    report = SolveReport()
    fitted = fit_drawn_planes(project, report)
    bad = check(fitted == 2, f"fitted {fitted} surfaces, expected 2")

    bad += check(
        project.plane_corner_points() == {"l0", "l1", "r0", "r1", "s0", "s1"},
        "a point shared by two surfaces should be named once in the corner set",
    )

    a, b = project.reference_planes
    if a.quad and b.quad:
        for pid in ("s0", "s1"):
            pa = np.asarray(a.quad[a.corners.index(pid)])
            pb = np.asarray(b.quad[b.corners.index(pid)])
            seam = float(np.linalg.norm(pa - pb)) * project.nominal_scale
            # both panels are flat and pass through the shared edge, so the
            # projections must land on top of each other
            bad += check(
                seam < 1e-6,
                f"{pid} sits {seam:.2e} m apart in the two faces that share it",
            )
    return bad


def test_drawn_plane_is_not_evidence(root: Path) -> list[str]:
    """A drawn surface's corners are kept out of the solve that measures the part.

    This is the property that lets someone sketch a fence without paying for
    it. The corners are marked like tie points and are deliberately not treated
    like them: they are excluded from the bundle and triangulated against the
    finished cameras afterwards, so four corners clicked roughly cannot move a
    camera. Without it, adding a decoration to a 7-photo project took its
    reprojection RMS from 1.41 px to 82 px.
    """
    project = build_scene(root)
    ids = [p.id for p in project.points[:4]]
    bad = check(
        len(ids) == 4, f"the fixture has {len(project.points)} points, need 4 for this"
    )
    if len(ids) < 4:
        return bad

    bad += check(
        project.plane_corner_points() == set(),
        "a project with no drawn planes claims to have corner points",
    )
    project.reference_planes = [ReferencePlane(id="pl0", label="fence", corners=ids)]
    bad += check(
        project.plane_corner_points() == set(ids),
        f"plane_corner_points gave {project.plane_corner_points()}, expected {set(ids)}",
    )

    # the solve's own track set is built by subtracting exactly this
    tracks = {p.id for p in project.points} - project.plane_corner_points()
    bad += check(
        not (tracks & set(ids)),
        "a drawn plane's corners are still in the set the bundle solves",
    )
    return bad


def test_planes_round_trip(root: Path) -> list[str]:
    """A plane survives the project file, which is where the UI leaves it."""
    project = build_scene(root)
    project.reference_planes = [
        ReferencePlane(id="pl0", label="datum A", normal=[0, 0, 1], offset=995.0),
        ReferencePlane(id="pl1", label="", normal=[0, 1, 0], offset=-612.5, size=250.0),
    ]
    back = Project.from_dict(project.to_dict())
    bad = check(
        len(back.reference_planes) == 2,
        f"{len(back.reference_planes)} planes survived the file, expected 2",
    )
    for before, after in zip(project.reference_planes, back.reference_planes):
        bad += check(
            (before.id, before.label, before.offset, before.size)
            == (after.id, after.label, after.offset, after.size),
            f"{before.id} came back as {after}",
        )
        bad += check(
            np.allclose(before.unit_normal(), after.unit_normal()),
            f"{before.id} changed direction in the file",
        )
    bad += check(
        back.reference_planes[0].describe("mm") == "Z = 995 mm",
        f"describe() says {back.reference_planes[0].describe('mm')!r}",
    )
    bad += check(
        back.next_plane_id() == "pl2",
        f"next_plane_id gave {back.next_plane_id()}, which collides",
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
    default = Model(out / "scene.step")
    bad += check(
        "AUTOMOTIVE_DESIGN" in default.header["FILE_SCHEMA"],
        "the default shape did not write AP214, which is the one every seat reads",
    )
    bad += check(
        not default.of_type("TRIANGULATED_FACE")
        and not default.of_type("FACETED_BREP"),
        "the default shape wrote body geometry",
    )

    # the shape has to survive the trip, or the shift-click does nothing
    webapp.write_exports(project, out, formats=["step"], step_shape="wireframe")
    model = Model(out / "scene.step")
    bad += check(
        "AUTOMOTIVE_DESIGN" in model.header["FILE_SCHEMA"],
        "step_shape='wireframe' did not write AP214",
    )
    bad += check(
        not model.of_type("TRIANGULATED_FACE"),
        "step_shape='wireframe' still wrote tessellated geometry",
    )
    bad += check(
        len(model.entities) > len(default.entities),
        "step_shape='wireframe' carried no more than the body-less default",
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
    sets = []
    for node in ("array_camera_anchors", "array_camera_optical"):
        if node not in model.products():
            continue
        sets += [
            e
            for e in model.representation_items(node)
            if "GEOMETRIC_CURVE_SET" in e.parts
        ]
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
        out = export.export_step(project, root / "scene.step", shape="tessellated")
        model = Model(out)
        for name, result in (
            ("references resolve", test_every_reference_resolves(model)),
            ("parameter counts", test_parameter_counts(model)),
            ("header", test_header(model, "AP242")),
            ("assembly tree", test_tree(model, project, "mm")),
            ("shared templates", test_templates_are_shared(model, project)),
            ("tessellated mesh", test_tessellated_mesh(model, project)),
            ("bodies are swappable", test_bodies_are_swappable(model, project)),
            ("mesh in file units", test_mesh_is_in_file_units(model, project, "mm")),
            ("colours", test_colours(model, project)),
            ("geometry is not shared", test_geometry_is_not_shared(model)),
            ("styles apply", test_styles_apply_to_their_items(model)),
            ("units", test_units(model, "mm")),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result
        print(f"  {out.stat().st_size / 1024:.1f} KiB, {len(model.entities)} entities")

        for units in ("cm", "m", "in", "ft"):
            print(f"tessellated, in {units}")
            out = export.export_step(
                project, root / f"scene_{units}.step", shape="tessellated", units=units
            )
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
            ("geometry is not shared", test_geometry_is_not_shared(model)),
            ("styles apply", test_styles_apply_to_their_items(model)),
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
            ("geometry is not shared", test_geometry_is_not_shared(model)),
            ("styles apply", test_styles_apply_to_their_items(model)),
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
            ("geometry is not shared", test_geometry_is_not_shared(model)),
        ):
            print(f"  {'ok  ' if not result else 'FAIL'} {name}")
            failures += result

        print("edges")
        for name, result in (
            ("refusals", test_refusals(root, project)),
            ("a missing mesh file", test_missing_mesh_still_places_the_body(root)),
            ("optical frame", test_optical_frame(root)),
            ("the front end's two routes", test_front_end_paths(root)),
            ("unplaced templates", test_unplaced_templates(root)),
            ("reference planes", test_reference_planes(root)),
            ("drawn planes", test_drawn_plane(root)),
            ("drawn planes refuse bad corners",
             test_drawn_plane_refuses_what_it_cannot_measure(root)),
            ("drawn planes share corners", test_drawn_planes_share_corners(root)),
            ("drawn planes are not evidence", test_drawn_plane_is_not_evidence(root)),
            ("planes survive the file", test_planes_round_trip(root)),
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
