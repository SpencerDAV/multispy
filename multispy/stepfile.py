"""A small ISO 10303-21 (STEP) writer.

Enough of Part 21 to put a whole solved scene into one file: a named assembly
tree, rigid placements, wireframe curves, tessellated triangle meshes, faceted
solids and per-item colour. No geometry kernel, no dependency -- the same
numpy-and-nothing-else footing as the rest of the package.

Why write STEP by hand rather than lean on a kernel: the scene this exports has
no parametric geometry in it. Camera frustums are line segments, tie points are
crosses, and the CAD bodies arrive as triangle soup from STL or OBJ, because
that is what :mod:`multispy.mesh` can read. A kernel would add half a gigabyte
of dependency to emit entities that are, in the end, lists of coordinates.

Two schemas are supported, and the difference matters in practice:

``ap214``
    AUTOMOTIVE_DESIGN. Wireframe and faceted solids only. Every CAD system in
    existence reads it, which is the point.
``ap242``
    Needed for tessellated geometry (``TRIANGULATED_FACE``), which is the only
    compact way to carry a 180,000-triangle body. Read by OpenCASCADE 7.5+
    (so FreeCAD, and most viewers), NX, Creo and 3DEXPERIENCE; support in
    older mid-range CAD is patchy, which is why :mod:`multispy.export` offers
    the wireframe fallback rather than assuming. **CATIA V5 is not on that
    list** -- its STEP reader has no AP242 tessellated geometry at all -- so a
    V5 seat wants ``shape="wireframe"`` or ``shape="brep"``.

Entity attribute order follows the merged AIM schema; the tessellated entities
in particular are easy to get subtly wrong, so the shapes written here are
``TRIANGULATED_FACE(name, coordinates, pnmax, normals, geometric_link, pnindex,
triangles)`` with ``pnindex`` empty and ``pnmax`` equal to the point count, which
is what the schema's WR1/WR2 rules demand of that combination.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

PRECISION = 9
"""Significant digits written for a real.

Nine, because the file is in millimetres and the scene can be metres across: at
1e4 mm a nine-digit real still resolves a micron, and the solve's own
uncertainty is three orders of magnitude coarser than that. Writing full
repr() precision instead would inflate a 180k-triangle body by a third for
digits that are pure float noise."""

WRAP_ITEMS = 6
"""Coordinate triples or triangles per output line inside a long list.

Part 21 permits very long lines but plenty of readers have historically been
unhappy past a few thousand characters, and a single 180k-point
``COORDINATES_LIST`` is ten megabytes on one line. Wrapping while the list is
built costs nothing; wrapping it afterwards would mean scanning ten megabytes
for quote-safe commas."""

SCHEMAS = {
    "ap214": (
        "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }",
        "core data for automotive mechanical design processes",
        "automotive_design",
        2000,
    ),
    "ap242": (
        # Edition 1 (442 1 1 4) on purpose, not edition 3. Nothing written
        # here is newer than edition 1 -- tessellated geometry is already in
        # ISO 10303-242:2014 -- and a reader handed a schema identifier it
        # does not know does not degrade gracefully: CATIA V5 parses the whole
        # file and then falls over on the way out.
        "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }",
        "managed model based 3d engineering",
        "managed_model_based_3d_engineering",
        2014,
    ),
}

SI_LENGTH = {
    "mm": ".MILLI.",
    "cm": ".CENTI.",
    "m": "$",
}
"""SI prefix per metric unit."""

IMPERIAL_LENGTH = {
    "in": ("INCH", 25.4),
    "ft": ("FOOT", 304.8),
}
"""Name and millimetres per unit for the units that are not SI.

These get a ``CONVERSION_BASED_UNIT`` stated as a ratio against the millimetre
rather than a prefix. The unit set matches the rest of the package deliberately:
:data:`multispy.project.UNIT_SCALES` offers feet, so an export asked for in
feet should write feet rather than fail halfway down."""

STEP_UNITS = (*SI_LENGTH, *IMPERIAL_LENGTH)
"""Every unit a STEP file can be written in here."""

#: Length a reader should treat as the same point, per unit -- a micron.
CONFUSION = {"mm": 1e-3, "cm": 1e-4, "m": 1e-6, "in": 4e-5, "ft": 3e-6}


# --------------------------------------------------------------------------
# primitive formatting
# --------------------------------------------------------------------------


def real(x: float) -> str:
    """A float as a Part 21 real, which must always carry a decimal point.

    ``0`` is not a real in Part 21; ``0.`` is. Getting this wrong produces a
    file that parses as integers and then fails validation deep inside a
    reader, with an error message about something else entirely.
    """
    s = f"{float(x):.{PRECISION}G}"
    if "E" in s:
        mant, _, exp = s.partition("E")
        if "." not in mant:
            mant += "."
        return f"{mant}E{exp}"
    return s if "." in s else s + "."


def text(s: str) -> str:
    """A string as a Part 21 string literal.

    Backslash introduces Part 21's escape sequences and the apostrophe ends the
    literal, so both have to be doubled or escaped; anything outside ASCII goes
    through the ``\\X2\\`` encoding, because a raw UTF-8 byte in a Part 21
    string is not portable and a CAD part named with an umlaut is not unusual.
    """
    out = []
    for ch in str(s):
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append(f"\\X2\\{ord(ch):04X}\\X0\\")
    return "'" + "".join(out) + "'"


def refs(ids: Iterable[int]) -> str:
    """An entity-reference list, ``(#1,#2,#3)``."""
    return "(" + ",".join(f"#{i}" for i in ids) + ")"


def _wrapped(parts: Sequence[str], per_line: int = WRAP_ITEMS) -> str:
    """A parenthesised Part 21 list, broken every ``per_line`` items."""
    if len(parts) <= per_line:
        return "(" + ",".join(parts) + ")"
    chunks = [
        ",".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line)
    ]
    return "(" + ",\n  ".join(chunks) + ")"


# --------------------------------------------------------------------------
# the assembly node a caller hands back to place()
# --------------------------------------------------------------------------


@dataclass
class Component:
    """One named product in the assembly, with its own shape representation.

    ``items`` stays open after the component is made: a parent collects one
    extra axis placement per child placed under it, because the schema wants
    the transform's target to be an item of the parent's own representation.
    So the SHAPE_REPRESENTATION entity is reserved now and written at
    :meth:`StepFile.dumps` time, once the tree is complete.
    """

    name: str
    product: int
    definition: int
    shape: int
    """Reserved id of this component's SHAPE_REPRESENTATION."""
    origin: int
    """Its local origin placement, the source side of any child transform."""
    items: list[int] = field(default_factory=list)
    tessellated: bool = False


class StepFile:
    """A Part 21 file under construction.

    Entities are appended in creation order and may reference ids that do not
    exist yet -- Part 21 has no ordering requirement, which is what makes the
    reserve-and-fill trick above legal.
    """

    def __init__(
        self,
        name: str = "scene",
        *,
        schema: str = "ap242",
        units: str = "mm",
        description: str = "MultiSpy scene",
        author: str = "MultiSpy",
    ) -> None:
        if schema not in SCHEMAS:
            raise ValueError(f"unknown schema {schema!r}; use 'ap214' or 'ap242'")
        if units not in STEP_UNITS:
            raise ValueError(
                f"unsupported STEP unit {units!r}; use one of {', '.join(STEP_UNITS)}"
            )
        self.name = name
        self.schema = schema
        self.units = units
        self.description = description
        self.author = author
        self._lines: list[str | None] = []
        self._points: dict[tuple, int] = {}
        self._dirs: dict[tuple, int] = {}
        self._item_kinds: dict[int, str] = {}
        self._products: list[int] = []
        self._components: list[Component] = []
        self._styles: list[int] = []
        self._colours: dict[tuple[int, int, int], int] = {}
        self._font: int | None = None
        self._finalised = False
        self._build_context()

    # ------------------------------------------------------------ low level

    def add(self, body: str) -> int:
        self._lines.append(body)
        return len(self._lines)

    def reserve(self) -> int:
        self._lines.append(None)
        return len(self._lines)

    def fill(self, entity_id: int, body: str) -> None:
        self._lines[entity_id - 1] = body

    def _kind(self, entity_id: int, kind: str) -> int:
        """Remember whether an item is drawn with a pen or filled.

        Only :meth:`colour` needs it, and it needs it badly enough to be worth
        the dict: see there.
        """
        self._item_kinds[entity_id] = kind
        return entity_id

    # -------------------------------------------------------------- context

    def _build_context(self) -> None:
        _, ctx_desc, protocol, year = SCHEMAS[self.schema]
        self.app_context = self.add(f"APPLICATION_CONTEXT({text(ctx_desc)})")
        self.add(
            "APPLICATION_PROTOCOL_DEFINITION('international standard',"
            f"{text(protocol)},{year},#{self.app_context})"
        )
        self.product_context = self.add(
            f"PRODUCT_CONTEXT('',#{self.app_context},'mechanical')"
        )
        self.definition_context = self.add(
            f"PRODUCT_DEFINITION_CONTEXT('part definition',#{self.app_context},'design')"
        )

        if self.units in IMPERIAL_LENGTH:
            # not SI, so it is stated as a ratio against the millimetre -- the
            # dimensional exponents spell out "this is a length"
            label, per_mm = IMPERIAL_LENGTH[self.units]
            mm = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
            factor = self.add(
                f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE({real(per_mm)}),#{mm})"
            )
            dims = self.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
            length = self.add(
                "( CONVERSION_BASED_UNIT(%s,#%d) LENGTH_UNIT() NAMED_UNIT(#%d) )"
                % (text(label), factor, dims)
            )
        else:
            length = self.add(
                "( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(%s,.METRE.) )"
                % SI_LENGTH[self.units]
            )
        angle = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        solid = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        # the accuracy a reader uses to decide whether two coordinates are the
        # same point: a micron in whatever unit the file is written in
        tol = CONFUSION[self.units]
        uncert = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({real(tol)}),#{length},"
            "'distance_accuracy_value','confusion accuracy')"
        )
        self.geom_context = self.add(
            "( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
            f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncert})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) "
            "REPRESENTATION_CONTEXT('','3D') )"
        )

    # ------------------------------------------------------------- geometry

    def _end_scope(self) -> None:
        """Close the current representation: stop reusing its geometry.

        :meth:`point` and :meth:`direction` dedupe, which is what keeps a
        welded mesh from writing every vertex three times -- but a
        representation_item belongs to exactly one representation, and in an
        assembly every representation is in its own coordinate system. A
        CARTESIAN_POINT shared between two components is a point claiming to
        be in two frames at once. It is easy to do by accident here: the
        drawing nominals are all a cross at their component's origin, so
        without this they shared one set of six points between them and
        differed only in their placement. OpenCASCADE forgives that. CATIA V5
        reads the whole file and then dies building the assembly.

        So the caches live for one representation and are dropped at its
        boundary -- the end of :meth:`component` and of :meth:`place`, the two
        points at which the next entity written belongs to a different
        representation than the last.
        """
        self._points.clear()
        self._dirs.clear()

    def point(self, xyz, name: str = "") -> int:
        """A CARTESIAN_POINT, reused when the same coordinate comes back.

        Welded meshes share vertices between triangles, and a faceted solid
        names every vertex of every face: without this the 8,400-triangle
        bracket writes 25,000 points instead of 4,200. Reuse is confined to
        one representation; see :meth:`_end_scope`.
        """
        x, y, z = (float(v) for v in xyz)
        key = (name, round(x, PRECISION), round(y, PRECISION), round(z, PRECISION))
        hit = self._points.get(key)
        if hit is not None:
            return hit
        eid = self.add(
            f"CARTESIAN_POINT({text(name)},({real(x)},{real(y)},{real(z)}))"
        )
        self._points[key] = eid
        return eid

    def direction(self, v, name: str = "") -> int:
        x, y, z = (float(c) for c in v)
        key = (name, round(x, PRECISION), round(y, PRECISION), round(z, PRECISION))
        hit = self._dirs.get(key)
        if hit is not None:
            return hit
        eid = self.add(f"DIRECTION({text(name)},({real(x)},{real(y)},{real(z)}))")
        self._dirs[key] = eid
        return eid

    def axis(self, origin=(0.0, 0.0, 0.0), z=(0.0, 0.0, 1.0), x=(1.0, 0.0, 0.0)) -> int:
        """An AXIS2_PLACEMENT_3D: origin, local z (the axis), local x (ref_direction).

        STEP takes z and x rather than a matrix, and projects x onto the plane
        normal to z itself, so a very slightly non-orthogonal rotation -- which
        is every rotation that has been through a solve -- is accepted as is.
        """
        return self.add(
            "AXIS2_PLACEMENT_3D('',#%d,#%d,#%d)"
            % (self.point(origin), self.direction(z), self.direction(x))
        )

    def placement_from(self, R, t) -> int:
        """An axis placement for the pose ``X_parent = R @ x_local + t``."""
        R = np.asarray(R, dtype=float)
        return self.axis(np.asarray(t, dtype=float), R[:, 2], R[:, 0])

    def polyline(self, pts: Sequence[Sequence[float]], name: str = "") -> int:
        ids = [self.point(p) for p in pts]
        return self._kind(self.add(f"POLYLINE({text(name)},{refs(ids)})"), "curve")

    def curve_set(self, elements: Sequence[int], name: str = "") -> int:
        """A GEOMETRIC_CURVE_SET: loose curves and points with no topology.

        The right container for a frustum or a point marker -- there is no
        solid there to pretend about, and every reader back to AP203 takes it.
        """
        return self._kind(
            self.add(f"GEOMETRIC_CURVE_SET({text(name)},{refs(elements)})"), "curve"
        )

    def cross(self, size: float, center=(0.0, 0.0, 0.0)) -> list[int]:
        """Three axis-aligned segments through ``center``.

        A bare CARTESIAN_POINT is legal in a curve set but several readers
        import it as nothing visible at all, so a measured point is drawn as
        well as stated.
        """
        c = np.asarray(center, dtype=float)
        out = []
        for axis in range(3):
            d = np.zeros(3)
            d[axis] = size / 2.0
            out.append(self.polyline([c - d, c + d]))
        return out

    # -------------------------------------------------- tessellated meshes

    def mesh_tessellated(
        self, verts: np.ndarray, tris: np.ndarray, name: str = ""
    ) -> int:
        """A triangle mesh as one AP242 TESSELLATED_SHELL.

        Two entities plus the coordinate block, whatever the triangle count,
        which is what makes a six-figure-triangle body a practical thing to put
        in a STEP file at all.
        """
        if self.schema != "ap242":
            raise ValueError("tessellated geometry needs the ap242 schema")
        verts = np.asarray(verts, dtype=float).reshape(-1, 3)
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if not len(verts) or not len(tris):
            raise ValueError("empty mesh")
        coords = _wrapped(
            [f"({real(x)},{real(y)},{real(z)})" for x, y, z in verts], WRAP_ITEMS
        )
        clist = self.add(f"COORDINATES_LIST('',{len(verts)},{coords})")
        # Part 21 lists are 1-based, so every index shifts up by one; an
        # off-by-one here reads as a mesh that is almost right, which is worse
        # than one that fails to load
        faces = _wrapped(
            [f"({a + 1},{b + 1},{c + 1})" for a, b, c in tris], WRAP_ITEMS
        )
        face = self.add(
            f"TRIANGULATED_FACE({text(name)},#{clist},{len(verts)},(),$,(),{faces})"
        )
        return self._kind(
            self.add(f"TESSELLATED_SHELL({text(name)},(#{face}),$)"), "surface"
        )

    # -------------------------------------------------------- faceted solid

    def mesh_faceted(self, verts: np.ndarray, tris: np.ndarray, name: str = "") -> int:
        """A triangle mesh as a FACETED_BREP: real faces, read by everything.

        Each triangle costs a plane, its placement and a bounded face -- about
        nine entities -- against two for the whole mesh in tessellated form.
        That is the trade: universal support for a file one to two orders of
        magnitude larger, which is why the caller has to ask for it.
        """
        verts = np.asarray(verts, dtype=float).reshape(-1, 3)
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if not len(verts) or not len(tris):
            raise ValueError("empty mesh")
        faces: list[int] = []
        for a, b, c in tris:
            pa, pb, pc = verts[a], verts[b], verts[c]
            n = np.cross(pb - pa, pc - pa)
            norm = float(np.linalg.norm(n))
            if norm < 1e-15:
                continue  # a degenerate facet has no plane and no area to lose
            n = n / norm
            # any direction in the facet's plane will do for the plane's x
            ref = pb - pa
            ref = ref / max(float(np.linalg.norm(ref)), 1e-15)
            plane = self.add(f"PLANE('',#{self.axis(pa, n, ref)})")
            loop = self.add(
                "POLY_LOOP('',%s)"
                % refs([self.point(pa), self.point(pb), self.point(pc)])
            )
            bound = self.add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            faces.append(self.add(f"FACE_SURFACE('',(#{bound}),#{plane},.T.)"))
        if not faces:
            raise ValueError("mesh has no non-degenerate triangles")
        shell = self.add(f"CLOSED_SHELL('',{refs(faces)})")
        return self._kind(self.add(f"FACETED_BREP({text(name)},#{shell})"), "surface")

    # ------------------------------------------------------ bounded surfaces

    def plane_face(self, origin, normal, half: float, name: str = "", u=None) -> int:
        """A square patch of plane, as a real surface rather than a drawn outline.

        A reference surface has to be something a CAD user can select, dimension
        to, offset from and project onto. Four lines in the shape of a rectangle
        are none of those things, so this is a genuine ``FACE_SURFACE`` on a
        ``PLANE``, wrapped in the open shell that makes it a surface model and
        not a claim to be a solid.

        The patch is finite because a STEP face must be; ``half`` is how far it
        reaches from ``origin`` in each direction.
        """
        n = np.asarray(normal, dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        if u is None:
            seed = np.eye(3)[int(np.argmin(np.abs(n)))]
        else:
            seed = np.asarray(u, dtype=float)
        u = seed - n * float(seed @ n)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        v = np.cross(n, u)
        o = np.asarray(origin, dtype=float)

        corners = [
            o - u * half - v * half,
            o + u * half - v * half,
            o + u * half + v * half,
            o - u * half + v * half,
        ]
        return self._patch(corners, o, n, u, name)

    def polygon_face(self, corners, name: str = "", normal=None) -> int:
        """A plane patch bounded by the corners given, rather than a square.

        For a surface that was drawn on the photographs instead of typed off a
        print: it has four measured corners, so the face in the file is those
        corners and not a square centred on them. A fence exported as a tidy
        square around its own middle would be a worse answer than the one that
        was measured, and would reach past the ends of the real thing.

        The corners are expected to be coplanar already -- the solve projects
        them onto the fitted plane precisely so that this can be true -- and
        the plane's own origin is taken as their centroid.
        """
        pts = [np.asarray(c, dtype=float) for c in corners]
        if len(pts) < 3:
            raise ValueError("a face needs at least three corners")
        o = np.mean(pts, axis=0)
        if normal is None:
            _u, _s, vt = np.linalg.svd(np.asarray(pts) - o)
            n = vt[2]
        else:
            n = np.asarray(normal, dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        u = pts[1] - pts[0]
        u = u - n * float(u @ n)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        return self._patch(pts, o, n, u, name)

    def _patch(self, corners, o, n, u, name: str) -> int:
        """The shared tail of both: a bounded face on a plane, in an open shell.

        An open shell and not a closed one, because a surface model is a claim
        about a surface; a closed shell would be a claim to bound a volume,
        which a single datum face does not.
        """
        plane = self.add(f"PLANE({text(name)},#{self.axis(o, n, u)})")
        loop = self.add("POLY_LOOP('',%s)" % refs([self.point(c) for c in corners]))
        bound = self.add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        face = self.add(f"FACE_SURFACE({text(name)},(#{bound}),#{plane},.T.)")
        shell = self.add(f"OPEN_SHELL('',(#{face}))")
        return self._kind(
            self.add(f"SHELL_BASED_SURFACE_MODEL({text(name)},(#{shell}))"), "surface"
        )

    # ------------------------------------------------------------ structure

    def component(
        self, name: str, items: Sequence[int] = (), *, tessellated: bool = False
    ) -> Component:
        """A named product with its own shape representation.

        Named, because the name is half the point of this exporter: a scene
        whose tree reads ``p4``, ``cam_IMG_0431``, ``917-4007`` is one somebody
        can navigate, and a flat bag of curves is not.
        """
        product = self.add(
            f"PRODUCT({text(name)},{text(name)},'',(#{self.product_context}))"
        )
        formation = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
        definition = self.add(
            f"PRODUCT_DEFINITION('design','',#{formation},#{self.definition_context})"
        )
        shape_of = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
        shape = self.reserve()
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape_of},#{shape})")
        origin = self.axis()
        comp = Component(
            name=name,
            product=product,
            definition=definition,
            shape=shape,
            origin=origin,
            items=[origin, *items],
            tessellated=tessellated,
        )
        self._products.append(product)
        self._components.append(comp)
        self._end_scope()
        return comp

    def place(
        self,
        parent: Component,
        child: Component,
        R=None,
        t=None,
        name: str | None = None,
    ) -> int:
        """Put ``child`` under ``parent`` at the pose ``R, t``.

        The five-entity dance the schema requires for an assembly placement:
        an occurrence, a shape for that occurrence, the transform, the
        representation relationship and the context-dependent shape that ties
        the two halves together.

        ``name`` names this occurrence rather than the product, which is how
        ten placements of one camera body show up as ten distinguishable nodes
        over a single shared mesh.
        """
        R = np.eye(3) if R is None else np.asarray(R, dtype=float)
        t = np.zeros(3) if t is None else np.asarray(t, dtype=float)
        nauo = self.add(
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE(%s,%s,'',#%d,#%d,$)"
            % (
                text(name or child.name),
                text(name or child.name),
                parent.definition,
                child.definition,
            )
        )
        occurrence = self.add(f"PRODUCT_DEFINITION_SHAPE('Placement','',#{nauo})")
        target = self.placement_from(R, t)
        parent.items.append(target)  # the transform's target lives in the parent
        transform = self.add(
            f"ITEM_DEFINED_TRANSFORMATION('','',#{child.origin},#{target})"
        )
        relationship = self.add(
            "( REPRESENTATION_RELATIONSHIP('','',#%d,#%d) "
            "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#%d) "
            "SHAPE_REPRESENTATION_RELATIONSHIP() )"
            % (child.shape, parent.shape, transform)
        )
        placed = self.add(
            f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relationship},#{occurrence})"
        )
        self._end_scope()
        return placed

    # --------------------------------------------------------------- colour

    def _colour(self, rgb: tuple[int, int, int]) -> int:
        hit = self._colours.get(rgb)
        if hit is not None:
            return hit
        r, g, b = (max(0, min(255, int(c))) / 255.0 for c in rgb)
        eid = self.add(f"COLOUR_RGB('',{real(r)},{real(g)},{real(b)})")
        self._colours[rgb] = eid
        return eid

    def colour(
        self, items: Sequence[int], rgb: tuple[int, int, int], *, width: float = 0.5
    ) -> None:
        """Give geometric items a colour.

        Worth the entities: in a scene holding cameras, tie points, drawing
        nominals and the part itself, colour is the only thing that keeps them
        apart once they are all grey in a CAD tree. Curves get a pen, surfaces
        get a fill, and both kinds can be in one call.

        A curve gets *only* the pen and a surface *only* the fill. Handing
        both to every item assigns a style that does not apply to the thing it
        is assigned to, and presentation is the last thing a reader does: a
        SURFACE_STYLE_USAGE on a GEOMETRIC_CURVE_SET is CATIA V5 importing a
        whole file and then crashing as it finishes.
        """
        if not items:
            return
        col = self._colour(rgb)
        groups: dict[str, list[int]] = {}
        for item in items:
            # an item of unrecorded kind keeps the old both-styles assignment:
            # better an over-broad style than a body that imports uncoloured
            groups.setdefault(self._item_kinds.get(item, "both"), []).append(item)
        for kind, members in groups.items():
            styles: list[int] = []
            if kind in ("curve", "both"):
                if self._font is None:
                    self._font = self.add(
                        "DRAUGHTING_PRE_DEFINED_CURVE_FONT('continuous')"
                    )
                styles.append(
                    self.add(
                        "CURVE_STYLE('',#%d,POSITIVE_LENGTH_MEASURE(%s),#%d)"
                        % (self._font, real(width), col)
                    )
                )
            if kind in ("surface", "both"):
                fill = self.add(f"FILL_AREA_STYLE_COLOUR('',#{col})")
                area = self.add(f"FILL_AREA_STYLE('',(#{fill}))")
                fill_style = self.add(f"SURFACE_STYLE_FILL_AREA(#{area})")
                side = self.add(f"SURFACE_SIDE_STYLE('',(#{fill_style}))")
                styles.append(self.add(f"SURFACE_STYLE_USAGE(.BOTH.,#{side})"))
            assignment = self.add(f"PRESENTATION_STYLE_ASSIGNMENT({refs(styles)})")
            for item in members:
                self._styles.append(
                    self.add(f"STYLED_ITEM('colour',(#{assignment}),#{item})")
                )

    # ---------------------------------------------------------------- output

    def dumps(self) -> str:
        if not self._components:
            raise ValueError("nothing to write: no components")
        if self._finalised:
            raise ValueError("this StepFile has already been written")
        self._finalised = True
        for comp in self._components:
            kind = (
                "TESSELLATED_SHAPE_REPRESENTATION"
                if comp.tessellated
                else "SHAPE_REPRESENTATION"
            )
            self.fill(
                comp.shape,
                "%s(%s,%s,#%d)"
                % (
                    kind,
                    text(comp.name),
                    _wrapped([f"#{i}" for i in comp.items], 8),
                    self.geom_context,
                ),
            )
        self.add(
            f"PRODUCT_RELATED_PRODUCT_CATEGORY('part','',{refs(self._products)})"
        )
        if self._styles:
            self.add(
                "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',%s,#%d)"
                % (_wrapped([f"#{i}" for i in self._styles], 8), self.geom_context)
            )

        stamp = _dt.datetime.now().replace(microsecond=0).isoformat()
        schema = SCHEMAS[self.schema][0]
        head = [
            "ISO-10303-21;",
            "HEADER;",
            f"FILE_DESCRIPTION(({text(self.description)}),'2;1');",
            "FILE_NAME(%s,%s,(%s),(''),%s,%s,'');"
            % (
                text(self.name),
                text(stamp),
                text(self.author),
                text("MultiSpy"),
                text("MultiSpy"),
            ),
            f"FILE_SCHEMA(({text(schema)}));",
            "ENDSEC;",
            "DATA;",
        ]
        # a reserved id that never got filled would be dropped here and leave
        # every reference to it dangling, which is a file that opens as far as
        # the first missing entity -- so say so instead
        empty = [i + 1 for i, line in enumerate(self._lines) if line is None]
        if empty:
            raise ValueError(
                f"{len(empty)} reserved entity id(s) were never filled: "
                f"{empty[:5]}{'...' if len(empty) > 5 else ''}"
            )
        body = [f"#{i + 1} = {line};" for i, line in enumerate(self._lines)]
        return "\n".join(head + body + ["ENDSEC;", "END-ISO-10303-21;", ""])
