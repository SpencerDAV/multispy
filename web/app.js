/* MultiSpy marking UI.
 *
 * Everything is stored and edited in ORIGINAL image pixel coordinates. Photos
 * are served downscaled for display, so each loaded bitmap carries a scale
 * factor; drawing happens in image space via a canvas transform, which keeps
 * hit-testing and persistence in one coordinate system.
 */

const AXIS_COLOR = ['#ff5a5a', '#4ddb7f', '#5a9cff'];
const AXIS_NAME = ['X', 'Y', 'Z'];
/* Direction groups get their own palette, deliberately nothing like the axis
 * colours: a tray line and an X line mean different things to the solve, and
 * the one mistake worth designing against is drawing one where you meant the
 * other. */
const GROUP_COLOR = ['#ffb347', '#c792ea', '#56d4dd', '#f78fb3', '#9ccc65', '#e6b8ff'];
const RESIDUAL_GAIN = 10; // sub-pixel errors are invisible at 1:1
/* Screen pixels a press may wander and still count as a click rather than a
 * drag. A press on a control line is ambiguous -- it could mean "flip this"
 * or "start dragging" -- and only the release tells you which. */
const CLICK_SLOP_PX = 3;

const state = {
  project: null,
  report: null,
  meta: {},
  current: null,
  mode: 'lines',
  axis: 0,
  dirGroup: null,
  /* Direction group a new control line will be drawn into, or null to draw an
   * ordinary axis line. Held beside `axis` rather than folded into it as a
   * fourth value, because the two are not alternatives of the same kind: an
   * axis is a fact about the world, a group is a promise that several lines
   * agree with each other. */
  activePoint: null,
  activeInstance: null,
  activeAnchor: null,
  activeTemplate: null,
  meshes: new Map(), // template id -> geometry payload from the server
  rig: null,
  autoAdvance: true,
  dirty: false,
  bitmaps: new Map(),
  view: { panX: 0, panY: 0, zoom: 1 },
  hover: null,
  drag: null,
  spaceDown: false,
  depthDown: false,   // S held: the depth handle, see nudgeDepth
  depthUsed: false,   // ...and whether that hold actually dragged anything
  axisKey: null,   // 'x' | 'y' | 'z' while held, to turn about that axis
  transformSpace: 'part',   // which frame X/Y/Z name; see turnAxisWorld
  showVP: true,
  showResid: true,
  showExpected: true,
  showShaded: false,
  undoBase: null,
  undoRestoring: false,
  dragUndoDone: false,
  showGrid: false,
  showOrigin: true,
  showModels: true,
  partFit: null,       // the latest fit report, which outlives one solve
  partPlacement: null,
  refine: null,        // the latest edge-alignment outcomes, per image
};

/* Anchors are keyed "<instance>/<anchor>" to match the project format, where
 * they deliberately share the flat shape tie-point observations have. */
const ANCHOR_SEP = '/';
const anchorKey = (instanceId, name) => `${instanceId}${ANCHOR_SEP}${name}`;
const splitAnchorKey = (key) => {
  const i = key.indexOf(ANCHOR_SEP);
  return i < 0 ? [key, ''] : [key.slice(0, i), key.slice(i + 1)];
};
const UNIT_SCALES = { mm: 0.001, cm: 0.01, m: 1, in: 0.0254, ft: 0.3048 };

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

/* ------------------------------------------------------------------ helpers */

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function currentImage() {
  if (!state.project) return null;
  return state.project.images.find((im) => im.id === state.current) || null;
}

function groupOf(im) {
  if (!state.project.groups[im.group]) {
    state.project.groups[im.group] = {
      id: im.group, f: null, cx: null, cy: null, k1: 0,
      lock_focal: false, lock_principal_point: true, lock_k1: true,
    };
  }
  return state.project.groups[im.group];
}

/* ------------------------------------------------------- known objects */

// null-safe: the toolbar is wired up before the first project has loaded
const templates = () =>
  state.project ? state.project.templates || (state.project.templates = {}) : {};
const instanceList = () =>
  state.project ? state.project.instances || (state.project.instances = []) : [];
const templateOf = (inst) => (inst ? templates()[inst.template] || null : null);
const activeInstance = () => instanceList().find((o) => o.id === state.activeInstance) || null;
const activeTemplate = () => templates()[state.activeTemplate] || null;

/** World-unit anchor positions of a template, keyed by name. */
function anchorPoints(template) {
  const out = {};
  for (const a of template.anchors || []) {
    out[a.name] = a.xyz.map((v) => v * template.unit_scale);
  }
  return out;
}

/** How many photos carry this anchor. The number that decides whether it can be
 *  triangulated at all: one photo is not enough. */
function anchorMarkCount(key) {
  if (!state.project) return 0;
  return state.project.images.filter((im) => im.anchor_obs && key in im.anchor_obs).length;
}

/** Every anchor mark clicked on one object, across every photo.
 *
 *  Counted over the marks rather than over the template's anchors, because the
 *  number this answers for is "how many clicks am I setting aside", and the
 *  honest answer to that is clicks, not features. */
function anchorMarkCountFor(instanceId) {
  if (!state.project) return 0;
  const prefix = `${instanceId}${ANCHOR_SEP}`;
  let n = 0;
  for (const im of state.project.images) {
    for (const key of Object.keys(im.anchor_obs || {})) if (key.startsWith(prefix)) n += 1;
  }
  return n;
}

/** Anchors of this object that reach two or more photos.
 *
 *  The number that decides whether a solve can do anything with the body, and
 *  the one worth putting in front of someone who has just finished marking:
 *  anchors marked in a single photo cannot be triangulated, so a body can look
 *  thoroughly marked and still be unplaceable. Three is the floor -- two leave
 *  it free to spin about the line joining them. */
function placeableAnchors(inst) {
  const template = templateOf(inst);
  if (!template) return 0;
  return (template.anchors || [])
    .filter((a) => anchorMarkCount(anchorKey(inst.id, a.name)) >= 2).length;
}

function nextInstanceId() {
  const used = new Set(instanceList().map((o) => o.id));
  let i = 0;
  while (used.has(`obj${i}`)) i++;
  return `obj${i}`;
}

/** Object-local point to world: X = R @ (p * unit_scale) + t. Note this is
 *  object-to-world, the opposite sense from an image's world-to-camera R. */
function objectToWorld(inst, template, local) {
  const s = template.unit_scale;
  const p = [local[0] * s, local[1] * s, local[2] * s];
  const R = inst.R;
  return [
    R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + inst.t[0],
    R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + inst.t[1],
    R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + inst.t[2],
  ];
}

const isPlaced = (inst) => !!(inst && inst.R && inst.t);

/* ------------------------------------------------------------------ the part

 * The part is the object the photographs are OF, as opposed to an object placed
 * in the scene. Once the point nominals have put the reconstruction into the
 * part's own coordinate frame, the part's pose is not unknown: it is the
 * identity, because that frame was built out of the part's own drawing. So it
 * needs no anchors, no hand placement and no search -- it draws itself in every
 * photo the moment the cameras are solved.
 *
 * It is a normal instance carrying `is_part`, which is what makes the wireframe
 * drawing, the 3D view and the save format work on it without a second code
 * path. Everything below is about not letting it be *edited* like a placement. */

const partInstance = () => instanceList().find((o) => o.is_part) || null;
const partTemplate = () => { const i = partInstance(); return i ? templateOf(i) : null; };
const isPartTemplate = (tid) => { const i = partInstance(); return !!i && i.template === tid; };
const partFit = () => (state.partFit || (state.report && state.report.part_fit) || null);

/** Images whose pose was moved by the edge alignment rather than by the solve. */
const refinedImages = () =>
  (state.project ? state.project.images : []).filter((im) => im.edge_refined);

/* -------------------------------------------------------------- pose by hand

 * Dragging a CAD model over a photograph until it fits. The cameras are already
 * solved, so a body's 6-DoF pose in the world fixes its silhouette in every
 * photo at once -- which is what makes this worth doing by eye: an alignment
 * made in one photo is immediately checkable in all the others.
 *
 * Scale is deliberately NOT a handle. Scaling a body about the camera centre is
 * an exact invariance of a single image -- a body 1.25x larger and 1.25x
 * further away projects to identical pixels, to 0.00 px, measured on these very
 * cameras -- so a scale slider would be the depth handle wearing a different
 * label, and it would throw away the one thing a known object is for. The model
 * keeps its CAD size; depth comes from the other photographs.
 */

const mat3 = {
  I: () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  mul: (A, B) => A.map((row) =>
    [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j])),
  apply: (A, v) => [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ],
  /** Rotation of `ang` radians about a unit axis (Rodrigues). */
  axisAngle: (axis, ang) => {
    const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
    const x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
    const c = Math.cos(ang), s = Math.sin(ang), k = 1 - c;
    return [
      [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
      [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
      [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ];
  },
  /** Re-orthonormalise, so a long run of incremental drags cannot drift into
   *  something that is no longer a rotation. Gram-Schmidt is enough here. */
  renorm: (A) => {
    const col = (j) => [A[0][j], A[1][j], A[2][j]];
    const unit = (v) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return v.map((c) => c / n); };
    const dot = (v, w) => v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    const x = unit(col(0));
    const c1 = col(1);
    const d = dot(c1, x);
    const y = unit(c1.map((c, i) => c - d * x[i]));
    const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
    return [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
  },
};

/** Camera centre of a solved image, in world coordinates: C = -R^T t. */
function cameraCentre(im) {
  if (!im.R || !im.t) return null;
  const R = im.R, t = im.t;
  return [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ];
}

/** The camera axes in world coordinates: right, down, forward. These are
 *  exactly the rows of the world-to-camera R. */
const cameraAxes = (im) => ({ right: im.R[0], down: im.R[1], forward: im.R[2] });

/** Where the solved tie points sit, as a centre and a radius. The only thing
 *  available to aim a first guess at: a body dropped at the world origin is
 *  usually nowhere near the subject, and may be behind the camera. */
function sceneExtent() {
  const pts = (state.project.points || []).filter((p) => p.xyz);
  if (!pts.length) return null;
  const c = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p.xyz[k], 0) / pts.length);
  const r = Math.max(...pts.map(
    (p) => Math.hypot(p.xyz[0] - c[0], p.xyz[1] - c[1], p.xyz[2] - c[2])));
  return { centre: c, radius: r || 1 };
}

/** A first pose for a body that has none: world-axis aligned, on this camera's
 *  view ray at the depth of the reconstruction, so it lands in frame at a
 *  sensible size instead of somewhere off-camera. */
function defaultPose(im) {
  const C = cameraCentre(im);
  if (!C) return null;
  const fwd = cameraAxes(im).forward;
  const ext = sceneExtent();
  let depth;
  if (ext) {
    const toCentre = [0, 1, 2].map((k) => ext.centre[k] - C[k]);
    depth = toCentre[0] * fwd[0] + toCentre[1] * fwd[1] + toCentre[2] * fwd[2];
  } else {
    // A project solved from control lines alone has no tie points to aim at,
    // and fitting a CAD model by eye is a reason not to mark any -- so falling
    // back is not a corner case, it is the common path for this mode. The
    // other cameras are the only measurement of scene size left: whoever took
    // the photographs stood about as far from the subject as from each other.
    const others = state.project.images
      .filter((o) => o !== im && o.R && o.t)
      .map((o) => cameraCentre(o))
      .map((c) => Math.hypot(c[0] - C[0], c[1] - C[1], c[2] - C[2]))
      .filter((d) => d > 1e-6)
      .sort((a, b) => a - b);
    depth = others.length ? others[Math.floor(others.length / 2)] : 2.0;
  }
  depth = clamp(depth, 0.2, 50);
  return { R: mat3.I(), t: [0, 1, 2].map((k) => C[k] + fwd[k] * depth) };
}

/** Give this body a pose if it has none, and flag it as hand-set so a solve
 *  will not discard it. False when this camera is not solved yet. */
function ensurePose(inst, im) {
  if (isPlaced(inst)) return true;
  const pose = defaultPose(im);
  if (!pose) return false;
  inst.R = pose.R;
  inst.t = pose.t;
  inst.manual = true;
  inst.pose_in_part_space = partSpaceState().ok;
  return true;
}

/** How much of this body lands inside this photo, as a fraction of its anchors
 *  (or its model's corners when it has no anchors yet). */
function bodyVisibility(inst, im) {
  const template = templateOf(inst);
  if (!template || !isPlaced(inst) || !im.R || !im.t) return 1;
  let local = (template.anchors || []).map((a) => a.xyz);
  if (!local.length) {
    const mesh = meshFor(template.id);
    local = mesh && mesh.vertices ? mesh.vertices.slice(0, 64) : [];
  }
  if (!local.length) return 1;
  let seen = 0;
  for (const p of local) {
    const q = projectWorld(im, objectToWorld(inst, template, p));
    if (q && q.x >= 0 && q.y >= 0 && q.x < im.width && q.y < im.height) seen++;
  }
  return seen / local.length;
}

/** Put the body back on this camera's view axis, keeping its distance,
 *  its rotation and its size.
 *
 *  Needed the moment the work moves to a second photograph. A body aligned in
 *  one photo is very often just outside the frame in the next -- these were
 *  taken metres apart around the subject, so a body a hand's breadth off in
 *  one view is off the edge in another -- and a body that cannot be seen
 *  cannot be dragged, which strands the whole align-snap-repeat loop on its
 *  second step. Only the position across the view changes: the orientation and
 *  the distance are the parts already worked for, and this throws neither
 *  away. */
function bringIntoView(inst, im) {
  const C = cameraCentre(im);
  if (!C || !isPlaced(inst)) return false;
  const fwd = cameraAxes(im).forward;
  const depth = Math.max(0.2, bodyDepth(inst, im) || 2.0);
  const pivot = pivotWorld(inst);
  for (let k = 0; k < 3; k++) inst.t[k] += C[k] + fwd[k] * depth - pivot[k];
  inst.manual = true;
  inst.pose_in_part_space = partSpaceState().ok;
  markDirty();
  return true;
}

/** Depth of a body along this camera's view ray. */
function bodyDepth(inst, im) {
  const C = cameraCentre(im);
  const fwd = cameraAxes(im).forward;
  if (!C) return null;
  const d = [0, 1, 2].map((k) => inst.t[k] - C[k]);
  return d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2];
}

/** Translate parallel to the image plane by a pixel delta.
 *
 *  Scaled through the body's own depth so the drag tracks the pointer at any
 *  distance: a pixel at 6 m is four times the world distance it is at 1.5 m. */
function nudgeInPlane(inst, im, dxPx, dyPx) {
  const g = groupOf(im);
  const depth = bodyDepth(inst, im);
  if (!g.f || depth === null) return;
  const ax = cameraAxes(im);
  const sx = (dxPx * depth) / g.f, sy = (dyPx * depth) / g.f;
  for (let k = 0; k < 3; k++) inst.t[k] += ax.right[k] * sx + ax.down[k] * sy;
}

/** Move the body along the view ray. Multiplicative, because "a bit closer" is
 *  a fraction of the current distance rather than a fixed number of metres.
 *
 *  This is the handle S drives, and it is deliberately depth rather than size.
 *  The two are indistinguishable in a single photograph -- a body 1.25x larger
 *  and 1.25x further away projects to identical pixels -- so the choice is not
 *  about what the eye can see, it is about which quantity the gesture should
 *  write. Depth belongs to THIS placement; size belongs to the MODEL, shared by
 *  every photo and every copy of it. Setting a shared property by eye from one
 *  view is much the more damaging of the two, and the eye-fit is only
 *  scaffolding for the anchor marks anyway: the solve refines placements, so a
 *  depth set by hand is a starting guess it is expected to improve on, where a
 *  unit_scale set by hand quietly rescales work already finished elsewhere.
 *
 *  Size stays settable, from the units dropdown on the model card, where it is
 *  a number that gets typed rather than one that gets dragged. */
function nudgeDepth(inst, im, factor) {
  const depth = bodyDepth(inst, im);
  if (depth === null) return;
  const ax = cameraAxes(im);
  const want = clamp(depth * factor, 0.05, 1e4);
  for (let k = 0; k < 3; k++) inst.t[k] += ax.forward[k] * (want - depth);
}

/** Axis-aligned bounds of a list of world points, or null if there are none. */
function bboxOf(pts) {
  if (!pts || !pts.length) return null;
  return [
    [0, 1, 2].map((k) => Math.min(...pts.map((p) => p[k]))),
    [0, 1, 2].map((k) => Math.max(...pts.map((p) => p[k]))),
  ];
}

/** World bounds of a placed object's actual geometry.
 *
 *  The mesh's own bounds, carried through the placement, rather than anything
 *  derived from ``t``. A CAD file's origin is very often nowhere near the body
 *  it describes -- an assembly zero, a machine datum, a corner of the stock --
 *  and for tooling the gap is routinely larger than the part being positioned,
 *  so ``t`` says almost nothing about where the geometry is. Falls back to the
 *  anchors for a model with no mesh, which are at least on the body. */
function placedBoxWorld(inst, template) {
  const mesh = meshFor(inst.template);
  const pts = [];
  if (mesh && mesh.bounds) {
    const [lo, hi] = mesh.bounds;
    for (let i = 0; i < 8; i++) {
      pts.push(objectToWorld(inst, template, [
        i & 1 ? hi[0] : lo[0],
        i & 2 ? hi[1] : lo[1],
        i & 4 ? hi[2] : lo[2],
      ]));
    }
  } else {
    for (const a of template.anchors || []) pts.push(objectToWorld(inst, template, a.xyz));
  }
  return bboxOf(pts);
}

/** Centre of the model in its own units, from the mesh if there is one and the
 *  anchors otherwise. Only ever a pivot, so the bounding-box centre is the
 *  right answer rather than a centroid weighted by vertex density. */
function modelCentreLocal(template) {
  const mesh = meshFor(template.id);
  if (mesh && mesh.vertices && mesh.vertices.length) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const v of mesh.vertices) {
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], v[k]);
        hi[k] = Math.max(hi[k], v[k]);
      }
    }
    return [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  }
  const anchors = template.anchors || [];
  if (!anchors.length) return [0, 0, 0];
  return [0, 1, 2].map(
    (k) => anchors.reduce((s, a) => s + a.xyz[k], 0) / anchors.length);
}

/** The model's longest side, in its own file units.
 *
 *  What the size box on the model card writes against, so that a fixture can
 *  be sized by typing the one dimension anybody actually knows -- "it is 2.4 m
 *  across" -- instead of a scale factor nobody holds in their head. Mesh
 *  bounds when there is a mesh, the anchor spread otherwise, since a model
 *  with no geometry still has points a real distance apart. */
function modelSpanLocal(template) {
  const mesh = meshFor(template.id);
  const box = mesh && mesh.bounds
    ? mesh.bounds
    : bboxOf((template.anchors || []).map((a) => a.xyz));
  if (!box) return 0;
  return Math.max(box[1][0] - box[0][0], box[1][1] - box[0][1], box[1][2] - box[0][2]);
}

/** Resize the model, keeping every copy of it where it looks like it is.
 *
 *  ``unit_scale`` multiplies model-space coordinates, so changing it alone
 *  moves the geometry along the line from the CAD file's own origin -- and a
 *  tooling origin is routinely further from the body than the body is wide, an
 *  assembly zero or a machine datum rather than anything on the part. A ten
 *  percent resize then throws the body clean out of frame, which makes the
 *  gesture unusable and makes a typed size look like it did something else.
 *  Correcting each placement's translation so the model's own centre stays put
 *  is what makes a resize read as a resize, from any view and at any
 *  orientation.
 *
 *  Returns false when nothing changed, so callers can skip the redraw. */
function setTemplateScale(t, unitScale) {
  /* Refused, not clamped. `clamp(+x || 0, 1e-9, 1e9)` turns a NaN into 1e-9,
   * which is a positive number that passes every later check and silently
   * shrinks the model to nothing -- the one outcome worse than doing nothing
   * at all, because it also drags every placement along with it. */
  const asked = +unitScale;
  const was = +t.unit_scale;
  if (!Number.isFinite(asked) || asked <= 0) return false;
  if (!Number.isFinite(was) || was <= 0) return false;
  const want = clamp(asked, 1e-9, 1e9);
  if (Math.abs(want - was) <= 1e-15 * Math.max(want, was)) return false;
  const c = modelCentreLocal(t);
  const k = was - want;
  for (const inst of instanceList()) {
    if (inst.template !== t.id || !isPlaced(inst)) continue;
    const d = mat3.apply(inst.R, [c[0] * k, c[1] * k, c[2] * k]);
    for (let j = 0; j < 3; j++) inst.t[j] += d[j];
  }
  t.unit_scale = want;
  // A model resized by hand no longer sits on a named unit, and saying "mm"
  // when it is not one is a lie the real-size readout beside it contradicts.
  const named = Object.keys(UNIT_SCALES).find((u) => Math.abs(UNIT_SCALES[u] - want) < 1e-12);
  t.units = named || 'custom';
  markDirty();
  return true;
}

/** Multiply the model's size, for the drag handle. */
function nudgeTemplateScale(t, factor) {
  return setTemplateScale(t, t.unit_scale * factor);
}

/** What the resize gesture and the size box report back: the longest side in
 *  the project's own nominal units, which is the number being set. */
function modelSpanNominal(template) {
  const span = modelSpanLocal(template);
  if (!span) return null;
  return (span * template.unit_scale) / nominalScale();
}

/** World position the body spins about: the centre of its own model, not its
 *  origin. A CAD origin is often a corner or a mount face, and rotating about
 *  that throws the body across the frame on every turn. */
function pivotWorld(inst) {
  const template = templateOf(inst);
  return template
    ? objectToWorld(inst, template, modelCentreLocal(template))
    : inst.t.slice();
}

const WORLD_AXIS = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

/** Which frame X, Y and Z name while a turn is locked to one axis.
 *
 *  Part space is the scene's own frame. Once nominals have landed, those are
 *  the drawing's axes, and they are the frame every number this app reports is
 *  quoted in -- so it is the right default, and it was the wrong only option.
 *  A body mounted at a compound angle has no edge parallel to any of them, and
 *  the turn the eye actually wants on it -- roll it about its own length, bring
 *  its own face round to meet the photograph -- is not on offer at all. Every
 *  such correction then has to be reached by composing two or three turns about
 *  axes that mean nothing to the body, each one disturbing what the last one
 *  got right.
 *
 *  Object space reads the axes straight off the columns of the body's own pose,
 *  so X, Y and Z mean what they meant in the CAD file the model came from. A
 *  rotation about a local axis leaves that axis where it is, so recomputing it
 *  from the current pose on every step of a drag is stable rather than drifting.
 */
function turnAxisWorld(inst, axisKey) {
  const k = { x: 0, y: 1, z: 2 }[axisKey];
  if (k === undefined) return null;
  if (state.transformSpace !== 'object' || !inst || !inst.R) return WORLD_AXIS[axisKey];
  // column k of an object-to-world R is that local axis, expressed in world
  const v = [inst.R[0][k], inst.R[1][k], inst.R[2][k]];
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 1e-9 ? v.map((c) => c / n) : WORLD_AXIS[axisKey];
}

function spaceLabel() {
  return state.transformSpace === 'object' ? 'object' : 'part';
}

/** Cycle the frame X/Y/Z are read in, and say so -- a mode with no visible
 *  state is a mode that turns every later gesture into a guess. */
function cycleTransformSpace() {
  state.transformSpace = state.transformSpace === 'object' ? 'part' : 'object';
  const inst = activeInstance();
  setStatus(state.transformSpace === 'object'
    ? `turning in OBJECT space — X/Y/Z are `
      + `${inst ? (inst.label || inst.id) : 'the body'}'s own axes, as modelled`
    : `turning in PART space — X/Y/Z are the scene's axes, the frame every `
      + `reported number uses`, 'ok');
  updateHint();
  renderPanels();   // the 90-degree buttons carry the frame in their labels
  render();
}

/** Turn by an exact angle about one world axis.
 *
 *  Dragging is for finding an angle; this is for the angles already known.
 *  Fixtures are built square, so a part is very often at some multiple of a
 *  right angle to the scene's own axes, and nudging up to it by hand leaves a
 *  degree or two that no photograph will later remove. */
function rotateSteps(inst, axisKey, deg) {
  const axis = turnAxisWorld(inst, axisKey);
  if (!axis) return;
  rotateInPlace(inst, axis, (deg * Math.PI) / 180);
  inst.manual = true;
  inst.pose_in_part_space = partSpaceState().ok;
  markDirty();
}

/** Rotate about a world-space axis through the body's own centre. */
function rotateInPlace(inst, axis, ang) {
  const pivot = pivotWorld(inst);
  const Rg = mat3.axisAngle(axis, ang);
  inst.R = mat3.renorm(mat3.mul(Rg, inst.R));
  const d = [0, 1, 2].map((k) => inst.t[k] - pivot[k]);
  const moved = mat3.apply(Rg, d);
  for (let k = 0; k < 3; k++) inst.t[k] = pivot[k] + moved[k];
}

/** Anchors of the active instance that this photo does not have yet, in order,
 *  so marking can walk through them without the user choosing each time. */
function unmarkedAnchors(inst, image) {
  const template = templateOf(inst);
  if (!template) return [];
  return (template.anchors || [])
    .map((a) => a.name)
    .filter((name) => !(anchorKey(inst.id, name) in (image.anchor_obs || {})));
}

function setStatus(text, cls) {
  const el = $('status');
  el.textContent = text;
  el.className = cls || '';
}

/* ---------------------------------------------------------------- dialogs */

/* `window.confirm` and `window.prompt` are not dependable here. An embedded
 * webview may suppress them outright -- the Claude desktop browser pane does,
 * and `confirm` then returns false the instant it is called, with no dialog
 * and no error to notice. Every guarded action silently became a no-op: no
 * photo, object or model could be removed, and a project with unsaved changes
 * could not be closed. So the app asks its own questions, in the page, where
 * nothing can switch them off.
 *
 * Both helpers return a promise. `askConfirm` resolves true or false;
 * `askText` resolves the string, or null if the user backed out -- the same
 * contract the native calls had, so the call sites only had to learn to wait.
 */

function dialogHost() {
  let host = $('dialog');
  if (host) return host;
  host = document.createElement('div');
  host.id = 'dialog';
  host.hidden = true;
  host.innerHTML =
    '<div class="box"><div class="msg"></div>'
    + '<input class="val" type="text" hidden>'
    + '<div class="row"><button class="smallbtn cancel">Cancel</button>'
    + '<button class="action primary ok">OK</button></div></div>';
  document.body.appendChild(host);
  return host;
}

/** Ask one question in the page. Blank lines split `text` into paragraphs:
 *  the first is the question, the rest are the consequences. Passing a string
 *  for `value` makes it a text entry instead of a yes/no. */
function askDialog(text, { value = null, ok = 'OK', danger = false } = {}) {
  const host = dialogHost();
  const box = host.querySelector('.box');
  const msg = host.querySelector('.msg');
  const input = host.querySelector('.val');
  const okBtn = host.querySelector('.ok');
  const cancelBtn = host.querySelector('.cancel');
  const asking = value !== null;

  msg.innerHTML = '';
  for (const para of String(text).split(/\n{2,}/)) {
    const p = document.createElement('p');
    p.textContent = para;
    msg.appendChild(p);
  }
  okBtn.textContent = ok;
  okBtn.classList.toggle('danger', !!danger);
  input.hidden = !asking;
  input.value = asking ? String(value) : '';
  host.hidden = false;
  if (asking) { input.focus(); input.select(); } else { okBtn.focus(); }

  return new Promise((resolve) => {
    const done = (answer) => {
      host.hidden = true;
      document.removeEventListener('keydown', onKey, true);
      resolve(answer);
    };
    const accept = () => done(asking ? input.value : true);
    const cancel = () => done(asking ? null : false);
    function onKey(ev) {
      // capture, so the canvas shortcuts -- Enter solves, Escape leaves the
      // 3D view -- never fire underneath an open question
      if (ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); cancel(); }
      else if (ev.key === 'Enter') { ev.preventDefault(); ev.stopPropagation(); accept(); }
    }
    okBtn.onclick = accept;
    cancelBtn.onclick = cancel;
    host.onmousedown = (ev) => { if (ev.target === host) cancel(); };
    box.onmousedown = (ev) => ev.stopPropagation();
    document.addEventListener('keydown', onKey, true);
  });
}

/** Yes or no. Resolves true only if the user says yes. */
const askConfirm = (text, opts) => askDialog(text, { ...(opts || {}), value: null });

/** Ask for a line of text. Resolves null if the user backs out. */
const askText = (text, value = '') =>
  askDialog(text, { value: value == null ? '' : String(value) });

/** Is a question on screen? The canvas shortcuts stand down while one is. */
function dialogOpen() {
  const host = $('dialog');
  return !!host && !host.hidden;
}

/* ------------------------------------------------------------------- undo
 *
 * Whole-project snapshots, not a log of reversible operations. The operations
 * here are not reversible in any useful sense -- a solve rewrites every pose and
 * every triangulated point at once, placing an object rewrites its anchors, and
 * an edge refinement rewrites poses it kept the originals of separately. Writing
 * an inverse for each would be a second implementation of every edit, drifting
 * out of step with the first. A snapshot is the project, which is the one thing
 * every edit already agrees on how to serialise.
 *
 * Cheap enough to be worth it: the project is marks, lines, nominals, instances
 * and settings -- tens of kilobytes. Meshes are not in it; a template carries a
 * path, not geometry, so a forty-thousand-vertex model costs nothing here.
 *
 * `markDirty` is the hook because every edit already calls it, so nothing can
 * add an edit and forget to make it undoable. It fires AFTER the change, which
 * is what makes this work: the snapshot standing in `undoBase` at that moment is
 * the state before it.
 */
const UNDO_LIMIT = 60;
const undoStack = [];
const redoStack = [];

function snapshotProject() {
  try {
    return JSON.stringify(state.project);
  } catch (err) {
    return null;                       // never let a snapshot failure stop an edit
  }
}

function updateUndoUi() {
  const u = $('btn-undo'), r = $('btn-redo');
  if (u) {
    u.disabled = !undoStack.length;
    u.title = undoStack.length ? `Undo (Ctrl+Z) — ${undoStack.length} step(s)` : 'Nothing to undo';
  }
  if (r) {
    r.disabled = !redoStack.length;
    r.title = redoStack.length ? `Redo (Ctrl+Shift+Z) — ${redoStack.length} step(s)` : 'Nothing to redo';
  }
}

function pushUndo() {
  const snap = snapshotProject();
  if (snap === null) return;
  // One step per drag, not one per pointermove: dragging a mark fires markDirty
  // on every mouse event, and sixty undos that each move a point two pixels is
  // the same as no undo at all.
  if (state.drag && state.dragUndoDone) { state.undoBase = snap; return; }
  if (state.undoBase !== null && state.undoBase !== undefined
      && state.undoBase !== snap) {
    undoStack.push(state.undoBase);
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
    redoStack.length = 0;
  }
  if (state.drag) state.dragUndoDone = true;
  state.undoBase = snap;
  updateUndoUi();
}

/** Snapshot before something that replaces the project wholesale.
 *
 *  A solve does not go through `markDirty` -- it adopts a project the server
 *  built -- and a solve that makes things worse is exactly what someone reaches
 *  for undo after. */
function pushUndoBefore(label) {
  const snap = snapshotProject();
  if (snap === null) return;
  undoStack.push(snap);
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack.length = 0;
  state.undoBase = snap;
  state.undoLabel = label || '';
  updateUndoUi();
}

function applySnapshot(snap) {
  const restoring = JSON.parse(snap);
  state.undoRestoring = true;
  try {
    adopt({ project: restoring });
    state.undoBase = snapshotProject();
    state.dirty = true;
    $('dirty').textContent = 'unsaved changes';
    $('dirty').className = 'dirty';
  } finally {
    state.undoRestoring = false;
  }
  buildFilmstrip();
  renderPanels();
  render();
  if (Viewer3D.isOn()) Viewer3D.render();
  updateUndoUi();
}

function doUndo() {
  if (!undoStack.length) { setStatus('nothing to undo'); return; }
  const now = snapshotProject();
  const snap = undoStack.pop();
  if (now !== null) redoStack.push(now);
  applySnapshot(snap);
  setStatus(`undone — ${undoStack.length} step(s) left`, 'ok');
}

function doRedo() {
  if (!redoStack.length) { setStatus('nothing to redo'); return; }
  const now = snapshotProject();
  const snap = redoStack.pop();
  if (now !== null) undoStack.push(now);
  applySnapshot(snap);
  setStatus(`redone — ${redoStack.length} step(s) left`, 'ok');
}

function markDirty() {
  if (!state.undoRestoring) pushUndo();
  // A viewpoint reading describes the marks and poses it ran against. Keeping it
  // across an edit would show a true statement about a project that no longer
  // exists, so it goes rather than ages.
  state.viewpoints = null;
  state.cadCheck = null;
  state.dirty = true;
  $('dirty').textContent = 'unsaved changes';
  $('dirty').className = 'dirty';
}

function clearDirty() {
  state.dirty = false;
  $('dirty').textContent = '';
}

/* --------------------------------------------------------------------- api */

/** One request to whichever front end is behind this page.
 *
 *  Over HTTP when a Python process is serving, or straight into the same Python
 *  running under Pyodide when the page is static. Callers cannot tell, and do
 *  not need to. */
async function api(path, body) {
  return Backend.call(path, body);
}

/** Strip the server-only annotations before posting the project back. */
function projectPayload() {
  const copy = JSON.parse(JSON.stringify(state.project));
  delete copy._path;
  delete copy._images_meta;
  delete copy._origin_stale;
  return copy;
}

function adopt(data) {
  state.viewpoints = null;   // new poses, so the apportionment above them is void
  state.cadCheck = null;   // and the bodies-vs-photographs verdict with it
  state.project = data.project;
  state.meta = data.project._images_meta || {};
  if (data.report !== undefined) state.report = data.report;
  if (!state.current && state.project.images.length) state.current = state.project.images[0].id;

  // a project saved before objects existed has none of these fields
  if (!state.project.templates) state.project.templates = {};
  if (!state.project.instances) state.project.instances = [];
  for (const im of state.project.images) if (!im.anchor_obs) im.anchor_obs = {};

  // keep the selections pointing at things that still exist
  if (!activeInstance()) {
    state.activeInstance = instanceList().length ? instanceList()[0].id : null;
    state.activeAnchor = null;
  }
  if (!activeTemplate()) {
    const ids = Object.keys(templates());
    state.activeTemplate = ids.length ? ids[0] : null;
  }
  ensureMeshes();

  // The baseline the first edit will be pushed onto. Without it the first edit
  // of a session had nothing to compare against and silently produced no undo
  // step -- the one edit most likely to be the mistake you want back.
  state.undoBase = snapshotProject();
  if (typeof updateUndoUi === 'function') updateUndoUi();
}

/** Fetch geometry for any template we do not have yet.
 *
 *  Meshes are kept client-side because the 3D view redraws on every orbit frame
 *  and the model never changes between solves. */
async function ensureMeshes() {
  for (const [tid, template] of Object.entries(templates())) {
    if (!template.mesh_path || state.meshes.has(tid)) continue;
    state.meshes.set(tid, null); // mark in flight so we ask only once
    try {
      const data = await api('/api/mesh', { project: projectPayload(), template: tid });
      state.meshes.set(tid, data.mesh);
    } catch (err) {
      state.meshes.set(tid, { error: err.message });
      console.warn(`model for ${tid}: ${err.message}`);
    }
    renderObjectsPanel();
    syncViewer();
    render();
  }
}

/** Crease angles worth offering, coarsest first. Below about 5 degrees the
 *  triangulation itself starts showing through, which is noise rather than
 *  shape, so the range stops there. */
const CREASE_ANGLES = [
  [30, 'sharp only'],
  [18, 'standard'],
  [12, 'more'],
  [8, 'fine'],
  [5, 'everything'],
];

/** A template's crease angle, defaulting to the server's own. */
const creaseOf = (t) => (t && typeof t.crease_angle === 'number' ? t.crease_angle : 18);

const meshFor = (tid) => {
  const m = state.meshes.get(tid);
  return m && !m.error ? m : null;
};

/** Upload one CAD model; the server creates a template for it. */
async function uploadModel(file) {
  // the endpoint carries no copy of the project, so flush edits first or the
  // reload afterwards would discard them
  if (state.dirty) await saveNow();
  try {
    setStatus(`reading ${file.name}…`);
    const data = await Backend.addFileUpload('model', file);
    state.meshes.set(data.template, data.mesh);
    adopt(data);
    state.activeTemplate = data.template;
    clearDirty();
    const size = data.mesh.size.map((v) => v.toFixed(1)).join(' x ');
    setStatus(
      `${file.name}: ${data.mesh.n_faces} faces, ${size} ${data.mesh.suggested_units}` +
        ` — check the units before marking`,
      'ok'
    );
    renderPanels();
    syncViewer();
    render();
  } catch (err) {
    setStatus(`could not add the model: ${err.message}`, 'bad');
  }
}

/** Pairwise layout of the placed objects, without re-solving. */
async function refreshRig() {
  if (!instanceList().some(isPlaced)) {
    state.rig = null;
    return;
  }
  try {
    state.rig = await api('/api/rig', { project: projectPayload() });
  } catch (err) {
    state.rig = null;
  }
  renderObjectsPanel();
}

async function loadState() {
  adopt(await api('/api/state'));
  buildFilmstrip();
  await selectImage(state.current, true);
  renderPanels();
  showProjectName();
  // the layout table should be there on open, not only after a re-solve
  refreshRig();
}

function showProjectName() {
  $('projname').textContent = state.project.name || 'untitled';
  $('btn-project').title = state.project._path || 'no project file';
}

/** The tutorial and reference, rendered by the server from the project's own
 *  Markdown. A named tab so repeat clicks reuse it instead of piling up, and a
 *  separate window so it can sit beside the app while you follow along. */
function openDocs(page = 'tutorial') {
  window.open(`/docs/${page}`, 'multispy-docs', 'noopener');
}

async function saveNow() {
  try {
    setStatus('saving…');
    await api('/api/state', { project: projectPayload() });
    clearDirty();
    if (Backend.isStatic()) {
      // no disk to save to, so saving is a download -- and it has to carry the
      // photos, or reopening it would restore marks pointing at nothing
      await Backend.saveBundle(state.project.name);
      setStatus('downloaded project + photos — keep it to reopen this work', 'ok');
    } else {
      setStatus('saved', 'ok');
    }
  } catch (err) {
    setStatus(`save failed: ${err.message}`, 'bad');
  }
}

/** Open a .multispy, or a zip of one with its photos. Static mode only. */
async function openProjectFile(file) {
  if (state.dirty && !(await askConfirm(
    'This project has unsaved changes.\n\nOpen anyway? The unsaved changes here are lost.',
    { ok: 'Open anyway', danger: true }))) return;
  try {
    setStatus(`opening ${file.name}…`);
    const data = await Backend.openFile(file);
    for (const bmp of state.bitmaps.values()) bmp.close?.();
    state.bitmaps.clear();
    state.meshes.clear();
    state.current = null;
    state.activePoint = null;
    state.activeInstance = null;
    state.activeAnchor = null;
    state.report = null;
    state.rig = null;
    adopt(data);
    $('projmenu').hidden = true;
    clearDirty();
    showProjectName();
    buildFilmstrip();
    if (state.current) await selectImage(state.current, true);
    renderPanels();
    syncViewer();
    render();

    const missing = Object.values(state.meta).filter((m) => !m.exists).length;
    setStatus(
      missing
        ? `opened — ${missing} photo(s) are not in this file; drop them back in to see them`
        : `opened ${state.project.name}`,
      missing ? 'warn' : 'ok'
    );
  } catch (err) {
    setStatus(`could not open: ${err.message}`, 'bad');
  }
}

async function newStaticProject(name) {
  if (state.dirty && !(await askConfirm(
    'This project has unsaved changes.\n\nStart a new one anyway? The unsaved changes here are lost.',
    { ok: 'Start new', danger: true }))) return;
  await Backend.resetProject(name);
  for (const bmp of state.bitmaps.values()) bmp.close?.();
  state.bitmaps.clear();
  state.meshes.clear();
  state.current = null;
  state.report = null;
  state.rig = null;
  state.activeInstance = null;
  state.activeAnchor = null;
  state.activeTemplate = null;
  adopt(await api('/api/state'));
  $('projmenu').hidden = true;
  clearDirty();
  showProjectName();
  buildFilmstrip();
  renderPanels();
  syncViewer();
  render();
  setStatus('empty project — drop photos onto the canvas to start', 'ok');
}

async function solveNow() {
  try {
    setStatus('solving…');
    // A solve replaces every pose and every triangulated point at once, and it
    // adopts the server's project rather than going through markDirty, so this
    // is the only place it can be made undoable. It is also the edit most worth
    // undoing: one bad object can take a healthy solve down with it.
    pushUndoBefore('solve');
    $('btn-solve').disabled = true;
    const data = await api('/api/solve', { project: projectPayload(), bundle: true, save: true });
    adopt(data);
    clearDirty();
    const r = state.report;
    if (r && r.ok) {
      setStatus(
        `solved: ${r.n_images_positioned} cameras, ${r.n_points_triangulated} points, ` +
        `RMS ${r.rms_reprojection_px.toFixed(2)} px`,
        r.rms_reprojection_px < 3 ? 'ok' : 'warn'
      );
    } else {
      setStatus(`could not solve (${r ? r.stage : 'unknown'}) — see the Solve panel`, 'warn');
    }
    state.rig = null; // the fresh report carries its own layout
    // Same reasoning for the part: the solve moved the very point positions the
    // fit was measured against, so a kept fit report now describes nothing.
    state.partFit = null;
    state.partPlacement = null;
    state.refine = null;
    buildFilmstrip();
    renderPanels();
    syncViewer();
    render();
  } catch (err) {
    setStatus(`solve failed: ${err.message}`, 'bad');
  } finally {
    $('btn-solve').disabled = false;
  }
}

async function exportNow() {
  try {
    setStatus('exporting…');
    // the workspace needs this session's edits before it writes anything
    await api('/api/state', { project: projectPayload() });
    clearDirty();
    const data = await Backend.exportAll(undefined, projectPayload());
    setStatus(`exported to ${data.out_dir}`, 'ok');
  } catch (err) {
    setStatus(`export failed: ${err.message}`, 'bad');
  }
}

/** The STEP scene on its own, which is the one export a CAD seat wants.
 *
 *  Shift picks the wireframe. The default carries every triangle as AP242
 *  tessellated geometry, which is compact and right for anything built on
 *  OpenCASCADE, NX, CATIA or Creo -- and silently unreadable to a CAD system
 *  that does not do AP242, where the file opens with the cameras present and
 *  the bodies missing. That is the one case this needs an escape hatch for,
 *  and in the browser build there is no command line to be the escape hatch.
 */
async function exportStepNow(wireframe) {
  try {
    const shape = wireframe ? 'wireframe' : 'tessellated';
    setStatus(wireframe ? 'writing scene.step (wireframe)…' : 'writing scene.step…');
    await api('/api/state', { project: projectPayload() });
    clearDirty();
    const data = await Backend.exportStep(shape, projectPayload());
    setStatus(`scene.step written to ${data.out_dir}`, 'ok');
  } catch (err) {
    // the browser build runs Python in the page, so a refusal arrives as a
    // whole traceback; the status bar is one line, and the useful line is the
    // last one -- "nothing solved yet" rather than six frames of path
    const lines = String(err.message).trim().split('\n');
    const reason = lines[lines.length - 1].replace(/^[\w.]*(Error|Exception):\s*/, '');
    setStatus(`STEP export failed: ${reason || err.message}`, 'bad');
  }
}

async function addFolder() {
  const folder = await askText('Folder of photos to add:', '');
  if (!folder) return;
  try {
    setStatus('adding photos…');
    adopt(await api('/api/add_folder', { project: projectPayload(), folder }));
    buildFilmstrip();
    if (!currentImage() && state.project.images.length) await selectImage(state.project.images[0].id, true);
    renderPanels();
    setStatus(`${state.project.images.length} image(s) in project`, 'ok');
  } catch (err) {
    setStatus(`could not add: ${err.message}`, 'bad');
  }
}

/* ------------------------------------------------------- photos in and out */

const IMAGE_NAME_RE = /\.(jpe?g|png|tiff?|webp|bmp)$/i;

/** Upload files into the project's own photos folder, one request each.
 *
 *  One file per request rather than multipart: it keeps the server stdlib-only
 *  and gives honest per-file progress and per-file error reporting. */
async function uploadFiles(files) {
  const list = [...files].filter((f) => /^image\//.test(f.type) || IMAGE_NAME_RE.test(f.name));
  const skipped = files.length - list.length;
  if (!list.length) {
    setStatus(`nothing to add — ${skipped} file(s) were not images`, 'warn');
    return;
  }

  // the upload endpoint carries no copy of the project, so flush edits first or
  // the reload afterwards would discard them
  if (state.dirty) await saveNow();

  let row = $('uploadrow');
  if (row) row.innerHTML = `<progress max="${list.length}" value="0"></progress>`;

  let added = 0;
  const failed = [];
  for (const file of list) {
    try {
      await Backend.addFileUpload('photo', file);
      added++;
    } catch (err) {
      failed.push(`${file.name}: ${err.message}`);
    }
    setStatus(`uploading… ${added + failed.length}/${list.length}`);
    const bar = row && row.firstChild;
    if (bar) bar.value = added + failed.length;
  }

  adopt(await api('/api/state'));
  buildFilmstrip();
  if (!currentImage() && state.project.images.length) {
    await selectImage(state.project.images[0].id, true);
  }
  renderPanels();
  render();
  row = $('uploadrow'); // buildFilmstrip rebuilt the strip, so re-find it
  if (row) row.innerHTML = '';

  if (failed.length) {
    setStatus(`added ${added}, ${failed.length} failed — ${failed[0]}`, 'warn');
    console.warn('upload failures:', failed);
  } else {
    setStatus(`added ${added} photo${added === 1 ? '' : 's'}`
      + (skipped ? ` (${skipped} non-image file(s) skipped)` : ''), 'ok');
  }
}

async function removeImage(imageId) {
  const im = state.project.images.find((x) => x.id === imageId);
  if (!im) return;
  const meta = state.meta[imageId] || {};
  const marks = im.axis_lines.length + Object.keys(im.observations).length;
  const owned = /^photos[/\\]/.test(im.path);

  let question = `Remove ${meta.name || im.id} from this project?`;
  if (marks) question += `\n\nIt carries ${marks} mark(s), which will be lost.`;
  if (!(await askConfirm(question, { ok: 'Remove', danger: true }))) return;

  // deleting the file is a separate, explicit decision, and only ever offered
  // for photos this project copied in itself
  let deleteFile = false;
  if (owned) {
    deleteFile = await askConfirm(
      `Also delete the file from disk?\n\n${meta.absolute || im.path}\n\n`
      + `Cancel keeps the file and only unlinks it from the set.`,
      { ok: 'Delete the file', danger: true }
    );
  }

  try {
    const data = await api('/api/remove_image', {
      project: projectPayload(), image_id: imageId, delete_file: deleteFile,
    });
    state.bitmaps.get(imageId)?.close?.();
    state.bitmaps.delete(imageId);
    Backend.forgetPhoto(imageId);
    if (state.current === imageId) state.current = null;
    adopt(data);
    clearDirty();
    buildFilmstrip();
    if (state.current) await selectImage(state.current, true);
    renderPanels();
    render();
    setStatus(data.deleted_file ? 'removed, and deleted the file' : 'removed from the set', 'ok');
  } catch (err) {
    setStatus(`remove failed: ${err.message}`, 'bad');
  }
}

/* ----------------------------------------------------------- project menu */

async function toggleProjectMenu() {
  const menu = $('projmenu');
  if (!menu.hidden) { menu.hidden = true; return; }
  if (Backend.isStatic()) { renderStaticProjectMenu(menu); return; }
  menu.innerHTML = '<div class="head">loading…</div>';
  menu.hidden = false;
  try {
    const data = await api('/api/projects');
    let html = '<div class="head">Projects</div>';
    if (!data.projects.length) {
      html += '<div class="pr"><span class="sub">none yet</span></div>';
    }
    for (const p of data.projects) {
      const cur = p.path === data.current;
      html += `<div class="pr${cur ? ' cur' : ''}" data-path="${escapeHtml(p.path)}">`
            + `<span>${escapeHtml(p.name)}<div class="sub">${p.images} photo(s)</div></span>`
            + `<span class="sub">${cur ? 'open' : ''}</span></div>`;
    }
    html += '<div class="head">New project</div><div class="newrow">'
          + '<input type="text" id="newname" placeholder="name" style="flex:1">'
          + '<button class="smallbtn" id="newgo">create</button></div>'
          + `<div class="path">${escapeHtml(data.current || '')}</div>`;
    menu.innerHTML = html;

    for (const row of menu.querySelectorAll('.pr[data-path]')) {
      row.onclick = () => switchProject('/api/open_project', { path: row.dataset.path });
    }
    const go = () => switchProject('/api/new_project', { name: $('newname').value });
    $('newgo').onclick = go;
    $('newname').onkeydown = (ev) => { if (ev.key === 'Enter') go(); };
    $('newname').focus();
  } catch (err) {
    menu.innerHTML = `<div class="head">${escapeHtml(err.message)}</div>`;
  }
}

/* Static mode has no projects directory to list: a tab has one project, and it
 * lives only until the tab closes. So the menu offers the three things that do
 * make sense there -- take it with you, bring one back, start over -- and says
 * plainly that nothing is being stored anywhere. */
function renderStaticProjectMenu(menu) {
  menu.innerHTML =
    '<div class="head">This project</div>'
    + '<div class="newrow"><input type="text" id="projrename" style="flex:1" '
    + `value="${escapeHtml(state.project ? state.project.name : 'untitled')}" `
    + 'placeholder="name"><button class="smallbtn" id="projdl">download</button></div>'
    + '<div class="msg" style="margin:4px 2px">Downloads a zip with the project and its '
    + 'photos, so you can reopen it later or hand it to someone.</div>'
    + '<div id="samplesec" hidden><div class="head">Samples</div>'
    + '<div id="samplelist"></div></div>'
    + '<div class="head">Open</div>'
    + '<div class="newrow"><button class="smallbtn" id="projopen" style="flex:1">'
    + 'choose a .multispy or .zip…</button></div>'
    + '<div class="head">New</div>'
    + '<div class="newrow"><button class="smallbtn" id="projnew" style="flex:1">'
    + 'start an empty project</button></div>'
    + '<div class="path">Running entirely in this tab. Nothing is uploaded anywhere — '
    + 'your photos never leave the machine, and they are gone when the tab closes, '
    + 'so download the project to keep it.</div>';
  menu.hidden = false;

  const rename = $('projrename');
  rename.onchange = (e) => {
    const name = e.target.value.trim();
    if (name && state.project) { state.project.name = name; markDirty(); showProjectName(); }
  };
  $('projdl').onclick = async () => {
    try {
      await api('/api/state', { project: projectPayload() });
      clearDirty();
      await Backend.saveBundle(state.project.name);
      menu.hidden = true;
      setStatus('downloaded project + photos', 'ok');
    } catch (err) {
      setStatus(`download failed: ${err.message}`, 'bad');
    }
  };
  $('projopen').onclick = () => $('projectinput').click();
  $('projnew').onclick = async () => {
    const name = await askText('Name for the new project:', 'untitled');
    if (name === null) return;
    newStaticProject(name);
  };
  fillSampleList();
}

/* ---------------------------------------------------------------- samples */

/* Project bundles the build shipped alongside the page. Static mode has no
 * projects folder to list, so without these a first visit has nothing in it to
 * look at but an empty canvas. Each is an ordinary `.multispy.zip` -- the same
 * thing "download" produces -- so opening one goes through `openProjectFile`
 * exactly like a bundle someone brings themselves, dirty-check included. */

const MB = (bytes) => (bytes / 1e6).toFixed(1);

/** Wait until Python can actually be called.
 *  `isStatic()` reports which build this is and nothing more -- it answers true
 *  from a config flag, long before Pyodide has finished downloading and
 *  unpacking. The menu is open and clickable during those seconds, and a click
 *  then reached `pyodide.FS` while `pyodide` was still null. `isReady()` is the
 *  signal that actually means something. */
async function whenBackendReady(timeoutMs = 180000) {
  const until = Date.now() + timeoutMs;
  while (!Backend.isReady()) {
    if (Date.now() > until) return false;
    await new Promise((r) => setTimeout(r, 200));
  }
  return true;
}

/** Fill the Samples section, or leave it hidden if there are none.
 *  A build served straight out of `web/` has no manifest beside it, and that is
 *  ordinary rather than an error: the section simply never appears. */
async function fillSampleList() {
  const section = $('samplesec');
  const host = $('samplelist');
  if (!section || !host) return;

  let items;
  try {
    const res = await fetch('samples/index.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    items = await res.json();
  } catch {
    return;
  }
  if (!Array.isArray(items) || !items.length) return;

  host.innerHTML = items.map((s, i) =>
    `<div class="pr" data-sample="${i}"><span>${escapeHtml(s.name)}`
    + `<div class="sub">${s.images} photo(s), ${s.points} tie point(s)</div></span>`
    + `<span class="sub">${MB(s.bytes)} MB</span></div>`
  ).join('');
  for (const row of host.querySelectorAll('[data-sample]')) {
    row.onclick = () => openSample(items[+row.dataset.sample]);
  }
  section.hidden = false;
}

/** Fetch one sample and open it.
 *  These bundles carry their photos, so they run to tens of megabytes and the
 *  app would otherwise sit silent for a long moment on a click. The body is
 *  read in chunks purely so the status line can say how far along it is. */
async function openSample(sample) {
  try {
    setStatus(`downloading ${sample.name} — ${MB(sample.bytes)} MB…`);
    const res = await fetch(`samples/${sample.file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    let blob;
    const total = Number(res.headers.get('content-length')) || sample.bytes || 0;
    if (res.body && total) {
      const reader = res.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        setStatus(`downloading ${sample.name} — ${MB(got)} of ${MB(total)} MB`);
      }
      blob = new Blob(chunks);
    } else {
      blob = await res.blob();
    }

    // the download and Pyodide's own boot overlap, which is the whole reason
    // this waits here rather than refusing the click up front
    if (!Backend.isReady()) {
      setStatus(`${sample.name} downloaded — waiting for Python to finish loading…`);
      if (!(await whenBackendReady())) throw new Error('Python did not finish loading');
    }

    await openProjectFile(new File([blob], sample.file));
  } catch (err) {
    setStatus(`could not open ${sample.name}: ${err.message}`, 'bad');
  }
}

async function switchProject(endpoint, body) {
  if (state.dirty && !(await askConfirm(
    'This project has unsaved changes.\n\nSwitch anyway? The unsaved changes here are lost.',
    { ok: 'Switch anyway', danger: true }))) return;
  try {
    const data = await api(endpoint, body);
    for (const bmp of state.bitmaps.values()) bmp.close?.();
    state.bitmaps.clear();
    state.current = null;
    state.activePoint = null;
    state.report = null;
    state.hover = null;
    adopt(data);
    $('projmenu').hidden = true;
    clearDirty();
    showProjectName();
    buildFilmstrip();
    if (state.current) await selectImage(state.current, true);
    renderPanels();
    render();
    setStatus(
      state.project.images.length
        ? `opened ${state.project.name}`
        : `${state.project.name} is empty — drop photos onto the canvas to start`,
      'ok'
    );
  } catch (err) {
    setStatus(err.message, 'bad');
  }
}

/* ------------------------------------------------------------ drag and drop */

let dragDepth = 0;
const hasFiles = (ev) => ev.dataTransfer && [...ev.dataTransfer.types].includes('Files');

window.addEventListener('dragenter', (ev) => {
  if (!hasFiles(ev)) return;
  ev.preventDefault();
  dragDepth++;
  $('dropzone').classList.add('on');
});
window.addEventListener('dragover', (ev) => {
  if (!hasFiles(ev)) return;
  ev.preventDefault();
  ev.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) { dragDepth = 0; $('dropzone').classList.remove('on'); }
});
const MODEL_NAME_RE = /\.(obj|stl|ply)$/i;

window.addEventListener('drop', async (ev) => {
  if (!hasFiles(ev)) return;
  ev.preventDefault();
  dragDepth = 0;
  $('dropzone').classList.remove('on');
  // photos and CAD models arrive the same way and go to different endpoints,
  // so split them here rather than making the user find two different buttons
  const files = [...ev.dataTransfer.files];
  const models = files.filter((f) => MODEL_NAME_RE.test(f.name));
  const photos = files.filter((f) => !MODEL_NAME_RE.test(f.name));
  if (photos.length) await uploadFiles(photos);
  for (const file of models) await uploadModel(file);
});

/* ------------------------------------------------------------------ bitmaps */

async function bitmapFor(imageId) {
  if (state.bitmaps.has(imageId)) return state.bitmaps.get(imageId);
  const res = await fetch(await Backend.photoSrc(imageId, false));
  if (!res.ok) throw new Error(`photo ${imageId} unavailable`);
  const bmp = await createImageBitmap(await res.blob());
  state.bitmaps.set(imageId, bmp);
  if (state.bitmaps.size > 24) {
    const oldest = state.bitmaps.keys().next().value;
    if (oldest !== imageId) {
      state.bitmaps.get(oldest).close?.();
      state.bitmaps.delete(oldest);
    }
  }
  return bmp;
}

/* ------------------------------------------------------------------- view */

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  render();
  Viewer3D.resize();
}

function fitView() {
  const im = currentImage();
  if (!im) return;
  const rect = canvas.getBoundingClientRect();
  const zoom = Math.min(rect.width / im.width, rect.height / im.height) * 0.96;
  state.view.zoom = zoom;
  state.view.panX = im.width / 2 - rect.width / 2 / zoom;
  state.view.panY = im.height / 2 - rect.height / 2 / zoom;
}

function toImage(ev) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (ev.clientX - rect.left) / state.view.zoom + state.view.panX,
    y: (ev.clientY - rect.top) / state.view.zoom + state.view.panY,
  };
}

const tol = () => 9 / state.view.zoom;

async function selectImage(id, fit) {
  if (!id) return;
  state.current = id;
  try {
    await bitmapFor(id);
  } catch (err) {
    setStatus(err.message, 'bad');
  }
  if (fit) fitView();
  buildFilmstrip();
  renderPanels();
  render();
}

/* ------------------------------------------------- projection (for overlays) */

function projectWorld(im, X) {
  if (!im.R || !im.t) return null;
  const g = groupOf(im);
  const f = g.f, cx = g.cx ?? im.width / 2, cy = g.cy ?? im.height / 2;
  if (!f) return null;
  const R = im.R, t = im.t;
  const zc = R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + t[2];
  if (zc <= 1e-9) return null;
  const xc = R[0][0] * X[0] + R[0][1] * X[1] + R[0][2] * X[2] + t[0];
  const yc = R[1][0] * X[0] + R[1][1] * X[1] + R[1][2] * X[2] + t[1];
  let xn = xc / zc, yn = yc / zc;
  const k1 = g.k1 || 0;
  if (k1) { const s = 1 + k1 * (xn * xn + yn * yn); xn *= s; yn *= s; }
  return { x: f * xn + cx, y: f * yn + cy };
}

/* ------------------------------------------------------- direction groups */

/** The project's direction groups, creating the list if an older file lacks it. */
function dirGroups() {
  if (!state.project) return [];
  if (!Array.isArray(state.project.direction_groups)) state.project.direction_groups = [];
  return state.project.direction_groups;
}

function dirGroupOf(gid) {
  return dirGroups().find((g) => g.id === gid) || null;
}

function groupColor(gid) {
  const i = dirGroups().findIndex((g) => g.id === gid);
  return GROUP_COLOR[(i < 0 ? 0 : i) % GROUP_COLOR.length];
}

function lineColor(ln) {
  return ln.direction_group ? groupColor(ln.direction_group) : AXIS_COLOR[ln.axis];
}

function groupLabel(gid) {
  const g = dirGroupOf(gid);
  return (g && g.label) || gid;
}

/** Control lines on one world axis, which excludes every grouped line.
 *
 *  The same rule as `ImageView.axis_segments` on the Python side, and it has to
 *  be the same: a readiness badge counting tray lines towards "2 axes marked"
 *  would promise a calibration the solve cannot deliver. */
function axisLinesOn(im, axis) {
  return im.axis_lines.filter((l) => l.axis === axis && !l.direction_group);
}

function groupLinesIn(im, gid) {
  return im.axis_lines.filter((l) => l.direction_group === gid);
}

/** Lines and photos a group holds, across the whole project. */
function groupTally(gid) {
  let lines = 0;
  const images = new Set();
  for (const im of state.project.images) {
    const n = groupLinesIn(im, gid).length;
    if (n) { lines += n; images.add(im.id); }
  }
  return { lines, images: images.size };
}

/** Vanishing point of one axis from its control lines, by median pairwise
 *  intersection. Only an overlay -- the solver does this properly. */
function vanishingPoint(im, axis) {
  const lines = axisLinesOn(im, axis);
  if (lines.length < 2) return null;
  const hom = lines.map((l) => [
    l.p0[1] - l.p1[1],
    l.p1[0] - l.p0[0],
    l.p0[0] * l.p1[1] - l.p1[0] * l.p0[1],
  ]);
  const xs = [], ys = [];
  for (let i = 0; i < hom.length; i++) {
    for (let j = i + 1; j < hom.length; j++) {
      const a = hom[i], b = hom[j];
      const w = a[0] * b[1] - a[1] * b[0];
      if (Math.abs(w) < 1e-12) continue;
      xs.push((a[1] * b[2] - a[2] * b[1]) / w);
      ys.push((a[2] * b[0] - a[0] * b[2]) / w);
    }
  }
  if (!xs.length) return null;
  const med = (v) => { const s = [...v].sort((p, q) => p - q); return s[s.length >> 1]; };
  return { x: med(xs), y: med(ys) };
}

/* ----------------------------------------------------------------- drawing */

function render() {
  const im = currentImage();
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0d0f12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!im) return;

  const z = state.view.zoom;
  ctx.setTransform(z * dpr, 0, 0, z * dpr, -state.view.panX * z * dpr, -state.view.panY * z * dpr);

  const bmp = state.bitmaps.get(im.id);
  if (bmp) ctx.drawImage(bmp, 0, 0, im.width, im.height);
  else {
    ctx.fillStyle = '#20242b';
    ctx.fillRect(0, 0, im.width, im.height);
  }

  ctx.strokeStyle = '#555c68';
  ctx.lineWidth = 1 / z;
  ctx.strokeRect(0, 0, im.width, im.height);

  if (state.showGrid) drawGrid(im, z);
  if (state.showOrigin) drawPartOrigin(im, z);
  if (state.showVP) drawVanishingRays(im, z);
  if (state.showModels && state.showShaded) drawObjectShaded(im, z);
  if (state.showModels) drawObjectModels(im, z);
  drawControlLines(im, z);
  if (state.showResid) drawResiduals(im, z);
  if (state.showExpected) drawExpected(im, z);
  drawObservations(im, z);
  drawAnchors(im, z);
  if (state.drag && state.drag.kind === 'newline') drawPendingLine(z);

  $('zoombadge').textContent = `${im.width}x${im.height}  ${(z * 100).toFixed(0)}%`;
  drawFilmstripOverlays();
}

/* ------------------------------------------ the same body, in every photo */

/** Draw each placed body into every thumbnail, live.
 *
 *  This is the answer to a fit that comes right in one photograph and wrong in
 *  the next. A body has ONE pose in the world, so there is nothing to fix
 *  per photo -- and the reason it feels as though there is, is that a single
 *  photograph is very nearly blind to the one thing that matters most here.
 *  Sliding a body along the view ray barely changes what this camera sees; its
 *  apparent size is the only clue, and judging that by eye to better than a few
 *  per cent is not possible. At 2 m a 5% error is 10 cm, and in a photograph
 *  taken 90 degrees away 10 cm is not subtle.
 *
 *  So the fit has to be judged in two photographs at once, and the cheapest way
 *  to do that is to show it in all of them while the hand is still on the
 *  mouse. The thumbnails are small, but the error being hunted is not: a body
 *  at the wrong depth lands somewhere else entirely over there, which reads
 *  perfectly well at 70 px wide. */
function drawFilmstripOverlays() {
  if (!state.project) return;
  const strip = $('filmstrip');
  if (!strip) return;
  const tiles = strip.querySelectorAll('.film');
  const placed = instanceList().filter(isPlaced);
  state.project.images.forEach((im, i) => {
    const tile = tiles[i];
    const cv = tile && tile.querySelector('canvas.filmov');
    if (!cv) return;
    const w = Math.round(tile.clientWidth), h = Math.round(tile.clientHeight);
    if (w < 4 || h < 4) return;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const c = cv.getContext('2d');
    c.clearRect(0, 0, w, h);
    if (!state.showModels || !placed.length || !im.R || !im.t || !im.width) return;
    const s = w / im.width;
    for (const inst of placed) {
      const template = templateOf(inst);
      const mesh = template && meshFor(inst.template);
      if (!mesh || !mesh.edges || !mesh.edges.length) continue;
      // a thumbnail cannot show more than an outline, so spend the edges on
      // being drawn at all rather than on being complete
      const budget = 240;
      const order = edgesByLength(mesh);
      const shown = Math.min(order.length, budget);
      // project only the vertices those edges actually use. This runs for
      // every photo on every frame of a drag, and a real machined part brings
      // tens of thousands of vertices -- projecting all of them seven times a
      // frame to draw 240 lines would make the drag that this view exists to
      // support the one thing it ruins.
      const screen = new Map();
      const at = (idx) => {
        let p = screen.get(idx);
        if (p === undefined) {
          p = projectWorld(im, objectToWorld(inst, template, mesh.vertices[idx]));
          screen.set(idx, p);
        }
        return p;
      };
      c.beginPath();
      for (let k = 0; k < shown; k++) {
        const e = mesh.edges[order[k]];
        const a = at(e[0]), b = at(e[1]);
        if (!a || !b) continue;              // the edge passes behind the camera
        c.moveTo(a.x * s, a.y * s);
        c.lineTo(b.x * s, b.y * s);
      }
      c.strokeStyle = inst.id === state.activeInstance ? '#ffd24ad0' : '#7fd4ff80';
      c.lineWidth = 1;
      c.stroke();
    }
  });
}

function drawControlLines(im, z) {
  im.axis_lines.forEach((ln, idx) => {
    const isHover = state.hover && state.hover.kind === 'line' && state.hover.index === idx;
    const col = lineColor(ln);
    ctx.strokeStyle = col;
    ctx.lineWidth = (isHover ? 3.5 : 2) / z;
    // dashed means flagged uncertain: still in the solve, at a wider sigma,
    // and the first thing dropped if something has to go
    ctx.setLineDash(ln.uncertain ? [7 / z, 5 / z] : []);
    ctx.beginPath();
    ctx.moveTo(ln.p0[0], ln.p0[1]);
    ctx.lineTo(ln.p1[0], ln.p1[1]);
    ctx.stroke();
    ctx.setLineDash([]);

    /* Direction arrow: p0 -> p1 must point along the POSITIVE axis.
     *
     * Drawn only on an axis line, and its absence on a grouped one is the
     * point. A group's residual is odd in its direction and the solve squares
     * it, so which end you dragged from changes nothing -- and an arrowhead
     * there would invite someone to check it, or worse, to correct it. */
    if (!ln.direction_group) {
      const dx = ln.p1[0] - ln.p0[0], dy = ln.p1[1] - ln.p0[1];
      const len = Math.hypot(dx, dy) || 1;
      const a = 13 / z;
      const ux = dx / len, uy = dy / len;
      ctx.beginPath();
      ctx.moveTo(ln.p1[0], ln.p1[1]);
      ctx.lineTo(ln.p1[0] - a * ux + a * 0.45 * uy, ln.p1[1] - a * uy - a * 0.45 * ux);
      ctx.lineTo(ln.p1[0] - a * ux - a * 0.45 * uy, ln.p1[1] - a * uy + a * 0.45 * ux);
      ctx.closePath();
      ctx.fillStyle = col;
      ctx.fill();
    } else if (isHover) {
      // which group this belongs to, where the eye already is
      ctx.fillStyle = col;
      ctx.font = `${13 / z}px sans-serif`;
      ctx.fillText(groupLabel(ln.direction_group),
                   (ln.p0[0] + ln.p1[0]) / 2 + 9 / z, (ln.p0[1] + ln.p1[1]) / 2 - 7 / z);
    }

    for (const [hx, hy, which] of [[ln.p0[0], ln.p0[1], 0], [ln.p1[0], ln.p1[1], 1]]) {
      const hot = state.hover && state.hover.kind === 'handle'
        && state.hover.index === idx && state.hover.which === which;
      ctx.beginPath();
      ctx.arc(hx, hy, (hot ? 6 : 4) / z, 0, Math.PI * 2);
      ctx.fillStyle = hot ? '#fff' : '#0d0f12';
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5 / z;
      ctx.stroke();
    }
  });
}

function drawVanishingRays(im, z) {
  for (let axis = 0; axis < 3; axis++) {
    const vp = vanishingPoint(im, axis);
    if (!vp) continue;
    const lines = axisLinesOn(im, axis);
    ctx.save();
    ctx.setLineDash([6 / z, 6 / z]);
    ctx.strokeStyle = AXIS_COLOR[axis] + '66';
    ctx.lineWidth = 1 / z;
    for (const ln of lines) {
      // extend from whichever endpoint is nearer the vanishing point
      const d0 = Math.hypot(ln.p0[0] - vp.x, ln.p0[1] - vp.y);
      const d1 = Math.hypot(ln.p1[0] - vp.x, ln.p1[1] - vp.y);
      const from = d0 < d1 ? ln.p0 : ln.p1;
      ctx.beginPath();
      ctx.moveTo(from[0], from[1]);
      ctx.lineTo(vp.x, vp.y);
      ctx.stroke();
    }
    ctx.restore();

    const margin = Math.max(im.width, im.height) * 2.5;
    if (vp.x > -margin && vp.x < im.width + margin && vp.y > -margin && vp.y < im.height + margin) {
      ctx.beginPath();
      ctx.arc(vp.x, vp.y, 7 / z, 0, Math.PI * 2);
      ctx.strokeStyle = AXIS_COLOR[axis];
      ctx.lineWidth = 2 / z;
      ctx.stroke();
      ctx.fillStyle = AXIS_COLOR[axis];
      ctx.font = `${14 / z}px sans-serif`;
      ctx.fillText(`VP ${AXIS_NAME[axis]}`, vp.x + 11 / z, vp.y - 9 / z);
    }
  }
}

/* Where the solve expects a point this photograph has no mark for.
 *
 * A point triangulated from other views has a position, and that position
 * projects into every camera -- including the ones nobody has clicked it in.
 * Drawing the prediction is what turns a count of marks into something a person
 * can check: the ghost lands on the feature, or it does not, and when it does
 * not either the correspondence is wrong or the point is not in this view at
 * all. It costs no extra solve. The camera, the intrinsics and the point are
 * already in hand, and this is the same projection the residual vectors use.
 *
 * Grey and dashed because nothing here was observed. A solid ring in this tool
 * means somebody clicked that spot; a prediction borrowing the same mark would
 * be claiming evidence it does not have.
 *
 * Two things it cannot know, both worth holding in mind before trusting a ghost
 * that looks wrong. It cannot know occlusion -- a point on the far side of the
 * part projects onto the near side just as confidently, and a bare tie point has
 * no surface to be hidden by, so blank metal under a ghost is not by itself a
 * verdict. And it is only ever as good as the solve behind it: a photograph
 * whose pose is wrong puts every ghost in the wrong place at once, which is its
 * own useful signal and reads quite differently from one ghost out on its own.
 */
function drawExpected(im, z) {
  if (!im.R || !im.t) return;

  // Marked in at least two photographs, or there is no triangulation behind the
  // position -- a single ray fixes a direction and no depth, so the solver's
  // gauge is what decided where that point sits, and projecting it would dress
  // an arbitrary choice up as a prediction.
  const seen = {};
  for (const other of state.project.images) {
    for (const pid of Object.keys(other.observations)) seen[pid] = (seen[pid] || 0) + 1;
  }

  for (const pt of state.project.points) {
    if (!pt.xyz || pt.xyz.length !== 3) continue;
    if (im.observations[pt.id]) continue;
    if ((seen[pt.id] || 0) < 2) continue;
    const q = projectWorld(im, pt.xyz);
    if (!q) continue;                                   // behind this camera
    const slack = 14 / z;
    if (q.x < -slack || q.x > im.width + slack
        || q.y < -slack || q.y > im.height + slack) continue;

    const active = pt.id === state.activePoint;
    const color = active ? '#c3ccd8' : '#79838f';
    const r = (active ? 8 : 6) / z;

    ctx.setLineDash([3.4 / z, 2.9 / z]);
    ctx.beginPath();
    ctx.arc(q.x, q.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = (active ? 2.2 : 1.5) / z;
    ctx.stroke();
    ctx.setLineDash([]);

    // a thin cross rather than a filled centre: the dot is what a real mark has
    const tick = 2.6 / z;
    ctx.beginPath();
    ctx.moveTo(q.x - tick, q.y); ctx.lineTo(q.x + tick, q.y);
    ctx.moveTo(q.x, q.y - tick); ctx.lineTo(q.x, q.y + tick);
    ctx.lineWidth = 1.1 / z;
    ctx.stroke();

    if (active || z > 0.25) {
      ctx.fillStyle = color;
      ctx.font = `${11 / z}px sans-serif`;
      ctx.fillText((pt.label || pt.id) + ' ?', q.x + r + 4 / z, q.y - r);
    }
  }
}

function drawObservations(im, z) {
  const pointErr = {};
  if (state.report && state.report.residual_lines) {
    for (const rl of state.report.residual_lines) {
      if (rl.image === im.id) pointErr[rl.point] = rl.error_px;
    }
  }
  for (const [pid, uv] of Object.entries(im.observations)) {
    const active = pid === state.activePoint;
    const hot = state.hover && state.hover.kind === 'obs' && state.hover.point === pid;
    const err = pointErr[pid];
    const color = active ? '#ffd24a' : err > 4 ? '#e5604f' : '#4dd0e1';
    const r = (active || hot ? 8 : 6) / z;
    // a dashed ring means the point is flagged uncertain -- the flag is on the
    // point, so it shows in every photo the point appears in, which is the
    // honest picture: you doubted the feature, not one view of it
    const doubt = state.project.points.find((q) => q.id === pid)?.uncertain;
    ctx.setLineDash(doubt ? [3.4 / z, 2.6 / z] : []);
    ctx.beginPath();
    ctx.arc(uv[0], uv[1], r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = (active ? 2.6 : 1.8) / z;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(uv[0], uv[1], 1.4 / z, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // labels pile into an unreadable mat when zoomed out, so keep them for the
    // point being worked on until there is room for all of them
    if (active || hot || z > 0.25) {
      const pt = state.project.points.find((p) => p.id === pid);
      ctx.fillStyle = color;
      ctx.font = `${12 / z}px sans-serif`;
      ctx.fillText((pt && pt.label) || pid, uv[0] + r + 4 / z, uv[1] - r);
    }
  }
}

/** Edge indices for one mesh, longest first.
 *
 *  There is never room to draw every edge of a real part, so the question is
 *  which ones to drop. Taking every Nth -- what this used to do -- is the one
 *  answer that is certainly wrong: it keeps edges by their position in an
 *  array, which has nothing to do with the model, so a part arrives as a
 *  scatter of disconnected segments with half of its outline missing. That is
 *  unmatchable against a photograph, which is the entire job of this overlay.
 *
 *  Length is the proxy that works, and ``display_payload`` already thins the
 *  same way on its way out of the server; this is that rule applied again at
 *  the smaller budget a canvas has. Measured in model space, not on screen:
 *  screen length would be marginally better per frame and would have to be
 *  re-sorted on every one of them, and a drag cannot afford that on a part with
 *  tens of thousands of edges. So it is computed once and kept on the mesh. */
function edgesByLength(mesh) {
  if (mesh._byLength) return mesh._byLength;
  const v = mesh.vertices, e = mesh.edges;
  const len2 = new Float64Array(e.length);
  for (let i = 0; i < e.length; i++) {
    const a = v[e[i][0]], b = v[e[i][1]];
    if (!a || !b) continue;
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    len2[i] = dx * dx + dy * dy + dz * dz;   // squared: same order, no sqrt
  }
  const order = new Array(e.length);
  for (let i = 0; i < e.length; i++) order[i] = i;
  order.sort((p, q) => len2[q] - len2[p]);
  mesh._byLength = order;
  return order;
}

/* Each placed object's model, projected through this camera.
 *
 * This is the check that makes a known-object solve trustworthy in a way no
 * number can: if the model's wireframe lands on the real object in the
 * photograph, the pose is right. A residual table cannot tell you that a body
 * was placed 180 degrees out, and this picture can. */
/* The model as a solid, not an outline.
 *
 * A wireframe of anything boxy is genuinely ambiguous -- the Necker cube is the
 * textbook case and a sensor body is a Necker cube with mounting holes. Two
 * orientations project to the same lines and the eye picks whichever it saw
 * last, so a body placed half a turn out looks placed, which is how one ends up
 * fitted that way behind a residual that raised no objection. Filling the faces
 * removes the choice: the near surface hides the far one and there is only one
 * reading left.
 *
 * Painter's algorithm, back to front by face depth, because canvas 2d has no
 * depth buffer. That is wrong for faces that interpenetrate or that pierce one
 * another, which a single rigid body's own facets do not do -- and two bodies
 * that overlap are drawn in separate passes, so one will win wholesale. Worth
 * knowing before reading an overlap as geometry.
 *
 * Drawn under the wireframe deliberately. The lines are the feature edges, which
 * is what you actually aim a corner at, and burying them under fill would trade
 * one kind of unreadable for another. The fill is kept translucent for the same
 * reason: this is an overlay on a photograph, and a model you cannot see the
 * part through tells you nothing about whether it lines up.
 */
function drawObjectShaded(im, z) {
  if (!im.R || !im.t) return;
  const placed = instanceList().filter(isPlaced);
  if (!placed.length) return;

  const R = im.R;
  const light = [0.35, -0.5, -0.79];            // over the viewer's shoulder

  for (const inst of placed) {
    const template = templateOf(inst);
    const mesh = template && meshFor(inst.template);
    if (!mesh || !mesh.faces || !mesh.faces.length) continue;

    const world = mesh.vertices.map((v) => objectToWorld(inst, template, v));
    const screen = world.map((X) => projectWorld(im, X));

    // depth in THIS camera, so the sort is per view and not per scene
    const depth = world.map((X) =>
      R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + im.t[2]);

    const faces = [];
    for (let i = 0; i < mesh.faces.length; i++) {
      const f = mesh.faces[i];
      const a = screen[f[0]], b = screen[f[1]], c = screen[f[2]];
      if (!a || !b || !c) continue;                       // crosses behind the camera
      // Back faces, by screen winding. Also throws away degenerate slivers,
      // whose normals are noise and whose shading would flicker.
      const area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (area2 >= 0) continue;
      const wa = world[f[0]], wb = world[f[1]], wc = world[f[2]];
      const ux = wb[0] - wa[0], uy = wb[1] - wa[1], uz = wb[2] - wa[2];
      const vx = wc[0] - wa[0], vy = wc[1] - wa[1], vz = wc[2] - wa[2];
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      const lam = Math.abs(nx * light[0] + ny * light[1] + nz * light[2]);
      faces.push({
        a, b, c,
        d: (depth[f[0]] + depth[f[1]] + depth[f[2]]) / 3,
        shade: 0.32 + 0.68 * lam,
      });
    }
    if (!faces.length) continue;
    faces.sort((p, q) => q.d - p.d);                      // far first

    const selected = inst.id === state.activeInstance;
    const base = inst.is_part ? [255, 184, 108]
      : selected ? [255, 210, 74]
      : inst.locked ? [154, 255, 200]
      : [127, 212, 255];
    const alpha = selected || inst.is_part ? 0.5 : 0.34;

    for (const f of faces) {
      const t = f.shade;
      ctx.fillStyle = `rgba(${Math.round(base[0] * t)},${Math.round(base[1] * t)},`
                    + `${Math.round(base[2] * t)},${alpha})`;
      ctx.beginPath();
      ctx.moveTo(f.a.x, f.a.y);
      ctx.lineTo(f.b.x, f.b.y);
      ctx.lineTo(f.c.x, f.c.y);
      ctx.closePath();
      ctx.fill();
    }
  }
}

function drawObjectModels(im, z) {
  if (!im.R || !im.t) return;
  const placed = instanceList().filter(isPlaced);
  if (!placed.length) return;
  const budget = Math.max(400, Math.floor(4000 / placed.length));

  for (const inst of placed) {
    const template = templateOf(inst);
    const mesh = template && meshFor(inst.template);
    if (!mesh || !mesh.edges.length) continue;

    const world = mesh.vertices.map((v) => objectToWorld(inst, template, v));
    const screen = world.map((X) => projectWorld(im, X));
    const order = edgesByLength(mesh);
    const shown = Math.min(order.length, budget);

    ctx.beginPath();
    for (let k = 0; k < shown; k++) {
      const e = mesh.edges[order[k]];
      const a = screen[e[0]];
      const b = screen[e[1]];
      if (!a || !b) continue; // the edge crosses behind the camera
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    const selected = inst.id === state.activeInstance;
    /* The part gets its own colour and a heavier line. It is not one more body
     * among several -- it is the thing in the photograph, drawn from a pose
     * nobody adjusted, and whether its wireframe sits on the real part is the
     * single most informative thing on this screen. */
    ctx.strokeStyle = inst.is_part
      ? (selected ? '#ffb86cdd' : '#ffb86c99')
      : selected ? '#ffd24a99' : inst.locked ? '#9affc866' : '#7fd4ff66';
    ctx.lineWidth = (inst.is_part ? 1.8 : selected ? 1.6 : 1.1) / z;
    ctx.stroke();

    const center = projectWorld(im, objectToWorld(inst, template, [0, 0, 0]));
    if (center && z > 0.12) {
      ctx.fillStyle = inst.is_part ? '#ffb86c' : selected ? '#ffd24a' : '#7fd4ffaa';
      ctx.font = `${12 / z}px sans-serif`;
      const tag = inst.is_part ? `${inst.label || 'part'} — at the origin` : (inst.label || inst.id);
      ctx.fillText(tag, center.x + 6 / z, center.y - 6 / z);
    }
    // No gizmo on the part, ever. Dragging it would be dragging the coordinate
    // frame the nominals define, which is not an edit anyone means to make.
    if (selected && state.mode === 'pose' && !inst.is_part) drawPoseGizmo(inst, im, z);
  }
}

/** The handles on the body being posed: its pivot, and its three model axes.
 *
 *  The axes are not decoration. The one thing an eye cannot judge from a
 *  wireframe is which way round a symmetric body is, and a model 180 degrees
 *  out fits its own silhouette perfectly -- the same failure that put photo 3's
 *  rotation half a turn from the truth behind a respectable residual. Drawn
 *  axes make the orientation something you can read rather than infer. */
function drawPoseGizmo(inst, im, z) {
  const template = templateOf(inst);
  if (!template) return;
  const pivot = pivotWorld(inst);
  const o = projectWorld(im, pivot);
  if (!o) return;

  // an arm a quarter of the body across, so the gizmo scales with the object
  // rather than swamping a small one or vanishing on a large one
  const mesh = meshFor(template.id);
  let span = 0;
  if (mesh && mesh.size) span = Math.max(...mesh.size) * template.unit_scale;
  if (!span) {
    const pts = Object.values(anchorPoints(template));
    for (const a of pts) {
      for (const b of pts) span = Math.max(span, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    }
  }
  const arm = (span || 0.2) * 0.35;

  for (let k = 0; k < 3; k++) {
    const tip = [0, 1, 2].map((j) => pivot[j] + inst.R[j][k] * arm);
    const q = projectWorld(im, tip);
    if (!q) continue;
    ctx.strokeStyle = AXIS_COLOR[k];
    ctx.lineWidth = 2.2 / z;
    ctx.beginPath();
    ctx.moveTo(o.x, o.y);
    ctx.lineTo(q.x, q.y);
    ctx.stroke();
    ctx.fillStyle = AXIS_COLOR[k];
    ctx.beginPath();
    ctx.arc(q.x, q.y, 3.5 / z, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = '#ffd24a';
  ctx.lineWidth = 1.6 / z;
  ctx.beginPath();
  ctx.arc(o.x, o.y, 7 / z, 0, Math.PI * 2);
  ctx.stroke();

  // the axis a turn is locked to, drawn through the body and well past it, so
  // the constraint is something seen rather than remembered
  if (state.axisKey) {
    const k = { x: 0, y: 1, z: 2 }[state.axisKey];
    const reach = Math.max(arm * 4, 0.3);
    // drawn in whichever frame the turn will actually use, so switching space
    // is something seen on the body rather than read off the hint
    const dir = turnAxisWorld(inst, state.axisKey);
    const ends = [-1, 1].map((sgn) => projectWorld(
      im, [0, 1, 2].map((j) => pivot[j] + dir[j] * reach * sgn)));
    if (ends[0] && ends[1]) {
      ctx.save();
      ctx.setLineDash([8 / z, 6 / z]);
      ctx.strokeStyle = AXIS_COLOR[k];
      ctx.lineWidth = 2 / z;
      ctx.beginPath();
      ctx.moveTo(ends[0].x, ends[0].y);
      ctx.lineTo(ends[1].x, ends[1].y);
      ctx.stroke();
      ctx.restore();
    }
  }

  const depth = bodyDepth(inst, im);
  if (depth !== null && z > 0.1) {
    ctx.fillStyle = '#ffd24a';
    ctx.font = `${11 / z}px sans-serif`;
    ctx.fillText(`${depth.toFixed(2)} m`, o.x + 10 / z, o.y + 14 / z);
  }
}

/* Anchor marks, drawn as squares so they never read as tie points. The
 * distinction matters: a tie point is a free position the solve will move, an
 * anchor is a claim about which corner of a known body you clicked. */
function drawAnchors(im, z) {
  const obs = im.anchor_obs || {};
  const errors = {};
  if (state.report && state.report.residual_lines) {
    for (const rl of state.report.residual_lines) {
      if (rl.image === im.id) errors[rl.point] = rl.error_px;
    }
  }

  for (const [key, uv] of Object.entries(obs)) {
    const [instId, name] = splitAnchorKey(key);
    const inst = instanceList().find((o) => o.id === instId);
    const mine = instId === state.activeInstance;
    const active = mine && name === state.activeAnchor;
    const hot = state.hover && state.hover.kind === 'anchor' && state.hover.key === key;
    const err = errors[key];
    const color = active ? '#ffd24a' : err > 4 ? '#e5604f' : mine ? '#4dd0e1' : '#4dd0e188';
    const r = (active || hot ? 8 : 6) / z;

    ctx.strokeStyle = color;
    ctx.lineWidth = (active ? 2.6 : 1.8) / z;
    ctx.beginPath();
    ctx.rect(uv[0] - r, uv[1] - r, r * 2, r * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(uv[0], uv[1], 1.4 / z, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (active || hot || (mine && z > 0.3)) {
      ctx.fillStyle = color;
      ctx.font = `${12 / z}px sans-serif`;
      const label = inst && inst.label ? `${inst.label}·${name}` : name;
      ctx.fillText(label, uv[0] + r + 4 / z, uv[1] - r);
    }
  }
}

/* Culled marks are drawn red and dashed rather than hidden. A cull that
 * disappears is just data loss: the whole point of dropping a mark is to tell
 * the user which click to go and fix, and they can only see that if the bad
 * arrow is still on the photo, visibly distinct from the ones that counted. */
function drawResiduals(im, z) {
  if (!state.report || !state.report.residual_lines) return;
  ctx.lineWidth = 1.6 / z;
  for (const pass of [false, true]) {   // culled ones last, so they draw on top
    ctx.strokeStyle = pass ? '#ff4d4d' : '#ffa23f';
    ctx.setLineDash(pass ? [4 / z, 3 / z] : []);
    for (const rl of state.report.residual_lines) {
      if (rl.image !== im.id || Boolean(rl.culled) !== pass) continue;
      const ox = rl.observed[0], oy = rl.observed[1];
      const dx = (rl.reprojected[0] - ox) * RESIDUAL_GAIN;
      const dy = (rl.reprojected[1] - oy) * RESIDUAL_GAIN;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + dx, oy + dy);
      ctx.stroke();
      if (pass) {   // ring the click itself, so it is findable without the arrow
        ctx.beginPath();
        ctx.arc(ox, oy, 7 / z, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
  ctx.setLineDash([]);
}

/* Grid on the world XY plane, i.e. perpendicular to the Z axis -- so draw
 * vertical edges as Z if you want this to land on the ground.
 *
 * Unless constraints pin an origin, the solved frame's translation and scale
 * are arbitrary, so a grid at literal z = 0 would sit somewhere meaningless.
 * It is anchored to the reconstruction instead: the floor of the point cloud. */
function drawGrid(im, z) {
  const pts = state.project.points.filter((p) => p.xyz);
  if (!im.R || pts.length < 3) return;
  const xs = pts.map((p) => p.xyz[0]), ys = pts.map((p) => p.xyz[1]);
  const zs = [...pts.map((p) => p.xyz[2])].sort((a, b) => a - b);
  const planeZ = zs[Math.floor(zs.length * 0.1)];

  const lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]) || 1;
  const step = Math.pow(10, Math.round(Math.log10(span / 8)));
  const pad = span * 0.5;
  const x0 = Math.floor((lo[0] - pad) / step) * step, x1 = Math.ceil((hi[0] + pad) / step) * step;
  const y0 = Math.floor((lo[1] - pad) / step) * step, y1 = Math.ceil((hi[1] + pad) / step) * step;

  ctx.strokeStyle = '#ffffff38';
  ctx.lineWidth = 1 / z;
  const polyline = (from, to) => {
    const N = 32;
    let prev = null;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const s = i / N;
      const p = projectWorld(im, [
        from[0] + (to[0] - from[0]) * s,
        from[1] + (to[1] - from[1]) * s,
        planeZ,
      ]);
      // a grid line can cross behind the camera; break the stroke there
      if (!p) { prev = null; continue; }
      if (prev) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      prev = p;
    }
    ctx.stroke();
  };
  for (let x = x0; x <= x1 + 1e-9; x += step) polyline([x, y0], [x, y1]);
  for (let y = y0; y <= y1 + 1e-9; y += step) polyline([x0, y], [x1, y]);

  // axis gizmo, parked on the grid under the middle of the points
  const anchor = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, planeZ];
  const o = projectWorld(im, anchor);
  if (o) {
    for (let a = 0; a < 3; a++) {
      const d = [...anchor];
      d[a] += step * 2;
      const e = projectWorld(im, d);
      if (!e) continue;
      ctx.strokeStyle = AXIS_COLOR[a];
      ctx.lineWidth = 3 / z;
      ctx.beginPath();
      ctx.moveTo(o.x, o.y);
      ctx.lineTo(e.x, e.y);
      ctx.stroke();
      ctx.fillStyle = AXIS_COLOR[a];
      ctx.font = `${13 / z}px sans-serif`;
      ctx.fillText(AXIS_NAME[a], e.x + 4 / z, e.y - 4 / z);
    }
  }
}

/* The part's own origin, once the nominals have put the solve in part space.
 *
 * This overlay is either exactly right or has no business being drawn. Without
 * three non-collinear nominals the solved frame's position, orientation and
 * scale are a gauge choice -- zero is wherever the solver happened to leave it
 * -- so putting a crosshair there and calling it the origin would be a lie
 * with a label on it. With them, world zero *is* the drawing's zero, and
 * seeing it land on the datum corner of the real part confirms the frame came
 * out right in a way no residual table can. This is the question the grid
 * cannot answer: that grid is anchored to the floor of the point cloud, not to
 * the part.
 *
 * The axis length is a round number of nominal units rather than a fraction of
 * the scene, and it is written next to the origin, so the triad doubles as a
 * ruler: if the part is 50 mm across and the 50 mm axis spans half the photo,
 * the units are wrong by a factor this says out loud. */
/* How long the origin's axes are drawn, as a round 1/2/5 x 10^n of the
 * drawing's own unit, about a sixth of the point cloud across.
 *
 * Deliberately measured from the points rather than from the whole scene: in
 * the 3D view the cameras stand metres back from a part that is 80 mm across,
 * and sizing the triad to them would quote a dimension the part has nothing
 * to do with. Shared by both overlays so the photograph and the 3D view never
 * put two different numbers on the same triad. */
function partOriginAxis() {
  const unit = nominalScale();          // world metres per nominal unit
  const pts = state.project.points.filter((p) => p.xyz);
  let span = 0;
  for (let a = 0; a < 3; a++) {
    const v = pts.map((p) => p.xyz[a]);
    if (v.length > 1) span = Math.max(span, Math.max(...v) - Math.min(...v));
  }
  const target = ((span || unit * 100) / 6) / unit;
  const pow = Math.pow(10, Math.floor(Math.log10(target || 1)));
  const nice = [1, 2, 5, 10].map((m) => m * pow);
  const L = nice.find((v) => v >= target) || nice[nice.length - 1];
  return { L, len: L * unit, units: nominalUnits() };
}

/* The origin's confidence sphere, or null if it has never been measured.
 *
 * The radius is a measurement and the colour is a judgement, so they are kept
 * apart: the figure is always drawn, and the thresholds that colour it are the
 * project's own. A green sphere against someone else's default thresholds
 * would be the most misleading thing this overlay could show. */
function originSphere() {
  const u = state.project.origin_uncertainty;
  if (!u || !u.n || !(u.radius > 0)) return null;
  const st = state.project.settings || {};
  const good = Number.isFinite(st.origin_good) ? st.origin_good : 5;
  const warn = Number.isFinite(st.origin_warn) ? st.origin_warn : 20;
  const thin = u.n < 8;
  const grade = thin ? 'unknown' : (u.radius <= good ? 'good' : (u.radius <= warn ? 'warn' : 'bad'));
  return {
    radius: u.radius,                       // nominal units
    world: u.radius * nominalScale(),       // metres, for drawing
    grade,
    stale: !!state.project._origin_stale,
    n: u.n,
    sigma: u.sigma || [],
    anisotropy: u.anisotropy || 1,
    sigma_px: u.sigma_px || 0,
    units: u.units || nominalUnits(),
  };
}

const GRADE_COLOR = { good: '#43d17a', warn: '#e8c33a', bad: '#e2553d', unknown: '#9aa0a6' };

/* Enough digits to be read, never more than the estimate has. A sphere
 * measured from a few dozen re-solves is good to a couple of significant
 * figures, so printing 18.0271 would be inventing confidence. */
function fmtRadius(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function drawPartOrigin(im, z) {
  if (!im.R) return;
  if (!partSpaceState().ok) return;

  const { L, len } = partOriginAxis();

  const o = projectWorld(im, [0, 0, 0]);
  if (!o) return;   // behind this camera: there is nothing honest to draw

  // On a close-up the datum corner is often outside the frame. Say so at the
  // edge instead of drawing a triad into the margin.
  if (o.x < 0 || o.y < 0 || o.x > im.width || o.y > im.height) {
    const cx = Math.min(Math.max(o.x, 10 / z), im.width - 10 / z);
    const cy = Math.min(Math.max(o.y, 10 / z), im.height - 10 / z);
    ctx.strokeStyle = '#ffffff99';
    ctx.lineWidth = 1.5 / z;
    ctx.setLineDash([3 / z, 3 / z]);
    ctx.beginPath();
    ctx.arc(cx, cy, 6 / z, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffffff99';
    ctx.font = `${11 / z}px sans-serif`;
    // the marker sits on the border it went out through, so the label has to
    // turn back into the photograph or it is written off the edge too
    const back = o.x > im.width;
    ctx.textAlign = back ? 'right' : 'left';
    ctx.fillText('0,0,0 off frame',
                 cx + (back ? -9 : 9) / z,
                 o.y > im.height ? cy - 10 / z : cy + 4 / z);
    ctx.textAlign = 'left';
    return;
  }

  // Each axis is sampled rather than drawn as one segment: a short axis can
  // still cross behind the camera, and a straight line to a wrapped point
  // would shoot across the photograph.
  for (let a = 0; a < 3; a++) {
    ctx.strokeStyle = AXIS_COLOR[a];
    ctx.lineWidth = 3 / z;
    let prev = null;
    ctx.beginPath();
    for (let i = 0; i <= 24; i++) {
      const w = [0, 0, 0];
      w[a] = len * (i / 24);
      const p = projectWorld(im, w);
      if (!p) { prev = null; continue; }
      if (prev) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
      prev = p;
    }
    ctx.stroke();

    const end = [0, 0, 0];
    end[a] = len;
    const e = projectWorld(im, end);
    if (!e) continue;
    ctx.fillStyle = AXIS_COLOR[a];
    ctx.font = `${13 / z}px sans-serif`;
    ctx.fillText(AXIS_NAME[a], e.x + 4 / z, e.y - 4 / z);
  }

  // The confidence sphere reads as a circle from any camera. Its pixel radius
  // is measured by projecting a point one radius to the camera's right rather
  // than by reaching for the focal length: that way it stays right under
  // whatever projection model the image actually carries.
  const sph = originSphere();
  if (sph) {
    const right = [im.R[0][0], im.R[0][1], im.R[0][2]];
    const edge = projectWorld(im, right.map((c) => c * sph.world));
    if (edge) {
      const rpx = Math.hypot(edge.x - o.x, edge.y - o.y);
      const col = GRADE_COLOR[sph.grade] || GRADE_COLOR.unknown;
      ctx.beginPath();
      ctx.arc(o.x, o.y, rpx, 0, Math.PI * 2);
      ctx.fillStyle = col + '22';
      ctx.fill();
      ctx.strokeStyle = sph.stale ? col + '77' : col;
      ctx.lineWidth = 2 / z;
      if (sph.stale) ctx.setLineDash([5 / z, 4 / z]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // The sphere is drawn true to scale, which on a cell-sized scene is a couple
  // of pixels: millimetres against metres. So the grade rides on the origin
  // marker, which is always visible, and the circle is what you see once you
  // zoom in far enough for it to mean something.
  ctx.strokeStyle = sph ? (GRADE_COLOR[sph.grade] || GRADE_COLOR.unknown) : '#ffffff';
  ctx.lineWidth = (sph ? 2.5 : 1.5) / z;
  ctx.beginPath();
  ctx.arc(o.x, o.y, 5 / z, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = `${11 / z}px sans-serif`;
  const caption = sph
    ? `0,0,0 · ${L} ${nominalUnits()} axes · ±${fmtRadius(sph.radius)} ${sph.units}`
      + (sph.stale ? ' (stale)' : '')
    : `0,0,0 · ${L} ${nominalUnits()} axes`;
  ctx.lineWidth = 3 / z;
  ctx.strokeStyle = '#000000aa';
  ctx.strokeText(caption, o.x + 9 / z, o.y + 15 / z);
  ctx.fillStyle = '#ffffffdd';
  ctx.fillText(caption, o.x + 9 / z, o.y + 15 / z);
}

function drawPendingLine(z) {
  const d = state.drag;
  ctx.strokeStyle = state.dirGroup ? groupColor(state.dirGroup) : AXIS_COLOR[state.axis];
  ctx.lineWidth = 2 / z;
  ctx.setLineDash([4 / z, 4 / z]);
  ctx.beginPath();
  ctx.moveTo(d.from.x, d.from.y);
  ctx.lineTo(d.to.x, d.to.y);
  ctx.stroke();
  ctx.setLineDash([]);
}

/* -------------------------------------------------------------- hit testing */

function distToSegment(p, a, b) {
  const vx = b[0] - a[0], vy = b[1] - a[1];
  const wx = p.x - a[0], wy = p.y - a[1];
  const L2 = vx * vx + vy * vy;
  const t = L2 ? clamp((wx * vx + wy * vy) / L2, 0, 1) : 0;
  return Math.hypot(p.x - (a[0] + t * vx), p.y - (a[1] + t * vy));
}

function hitTest(p) {
  const im = currentImage();
  if (!im) return null;
  const t = tol();

  for (const [pid, uv] of Object.entries(im.observations)) {
    if (Math.hypot(p.x - uv[0], p.y - uv[1]) <= t) return { kind: 'obs', point: pid };
  }
  for (const [key, uv] of Object.entries(im.anchor_obs || {})) {
    if (Math.hypot(p.x - uv[0], p.y - uv[1]) <= t) return { kind: 'anchor', key };
  }
  for (let i = im.axis_lines.length - 1; i >= 0; i--) {
    const ln = im.axis_lines[i];
    if (Math.hypot(p.x - ln.p0[0], p.y - ln.p0[1]) <= t) return { kind: 'handle', index: i, which: 0 };
    if (Math.hypot(p.x - ln.p1[0], p.y - ln.p1[1]) <= t) return { kind: 'handle', index: i, which: 1 };
  }
  for (let i = im.axis_lines.length - 1; i >= 0; i--) {
    const ln = im.axis_lines[i];
    if (distToSegment(p, ln.p0, ln.p1) <= t) return { kind: 'line', index: i };
  }
  return null;
}

/** Reverse a control line, so p0 -> p1 points the other way.
 *
 *  A line carries no arrowhead in the world, only on screen: the direction it
 *  was dragged in is what fixes the sign of its axis. Draw one backwards and
 *  the solver silently gets a mirrored frame -- flip TWO axes in a photo and
 *  the determinant is back to +1, a legal rotation 180 degrees about the
 *  third, with nothing to complain about. Correcting that by hand meant
 *  deleting the line and redrawing it; this is the one-click version. */
function flipLine(im, index) {
  const ln = im.axis_lines[index];
  if (!ln) return;
  if (ln.direction_group) {
    // Nothing to reverse: a group has no positive end. Said rather than done
    // silently, because the click was a reasonable thing to try.
    setStatus(`${groupLabel(ln.direction_group)} lines have no direction to reverse — `
            + `a group's sign cancels in the solve, so either end does`, 'ok');
    return;
  }
  [ln.p0, ln.p1] = [ln.p1, ln.p0];
  markDirty();
  setStatus(`${AXIS_NAME[ln.axis]} line ${index} reversed — the arrow now points the other way`, 'ok');
  renderPanels();
  render();
}

/* ------------------------------------------------------------- tie points */

function newPoint(label) {
  const used = new Set(state.project.points.map((p) => p.id));
  let i = 0;
  while (used.has(`p${i}`)) i++;
  const pt = { id: `p${i}`, label: label || '', xyz: null };
  state.project.points.push(pt);
  state.activePoint = pt.id;
  markDirty();
  return pt;
}

function imagesWithout(pointId) {
  return state.project.images.filter((im) => im.enabled && !(pointId in im.observations));
}

function advanceToMissing(pointId) {
  const list = state.project.images.filter((im) => im.enabled);
  const start = list.findIndex((im) => im.id === state.current);
  for (let k = 1; k <= list.length; k++) {
    const cand = list[(start + k) % list.length];
    if (!(pointId in cand.observations)) {
      selectImage(cand.id, true);
      return true;
    }
  }
  return false;
}

/* -------------------------------------------------------- posing by hand */

/** Hand this body the pose the current drag is building, creating one if the
 *  body has none yet. Returns the instance, or null with a reason on screen. */
function poseTarget(im) {
  const inst = activeInstance();
  if (!inst) {
    setStatus('no object selected - add one in the Known objects panel', 'warn');
    return null;
  }
  if (!im.R || !im.t) {
    setStatus('this photo is not solved yet - draw control lines and press Enter first', 'warn');
    return null;
  }
  if (!ensurePose(inst, im)) {
    setStatus('nothing solved to aim at yet - press Enter to solve the cameras first', 'warn');
    return null;
  }
  return inst;
}

/** Write every anchor of this body into this photo at the position the posed
 *  model predicts.
 *
 *  This is the point of posing by hand. Finding a given corner of a given body
 *  in the seventh photograph is the slow, error-prone part of marking anchors,
 *  and it is exactly what the alignment has already answered: once the model
 *  sits over the object, each anchor has a predicted pixel. Dragging a marker
 *  five pixels onto the true feature is quick and, unlike the eye-fit itself,
 *  what comes out is a measurement the solver can weigh and report on. */
function snapAnchorsHere(im) {
  const inst = activeInstance();
  const template = inst && templateOf(inst);
  if (!inst || !isPlaced(inst) || !template) {
    setStatus('pose the object over this photo first', 'warn');
    return 0;
  }
  const anchors = template.anchors || [];
  if (!anchors.length) {
    setStatus('this model has no anchors yet - pick them in the 3D view (D)', 'warn');
    return 0;
  }
  let written = 0, offscreen = 0;
  for (const a of anchors) {
    const q = projectWorld(im, objectToWorld(inst, template, a.xyz));
    // an anchor behind the camera or outside the frame is not a prediction
    // anybody can nudge, and a marker parked off-canvas is worse than none
    if (!q || q.x < 0 || q.y < 0 || q.x >= im.width || q.y >= im.height) {
      offscreen++;
      continue;
    }
    im.anchor_obs[anchorKey(inst.id, a.name)] = [q.x, q.y];
    written++;
  }
  if (written) markDirty();
  setStatus(
    written
      ? `${written} predicted anchor(s) placed in this photo${offscreen ? `, ${offscreen} off-frame` : ''}`
        + ' - drag each onto the real feature, then press Enter'
      : 'none of this model\u2019s anchors land inside this photo at that pose',
    written ? 'ok' : 'warn'
  );
  renderPanels();
  render();
  return written;
}

/** Drop a hand-set pose. Anchors are left alone: they are measurements, and a
 *  bad eye-fit is no reason to throw them away. */
function clearPose(inst) {
  inst.R = null;
  inst.t = null;
  inst.manual = false;
  inst.pose_in_part_space = false;
  markDirty();
  renderPanels();
  render();
  setStatus(`${inst.label || inst.id}: pose cleared`, 'ok');
}

/* ------------------------------------------------------- anchor marking */

/** What this object still needs before a solve can place it.
 *
 *  Marking anchors writes observations and nothing else: no pose exists until
 *  a solve triangulates them and fits the model. Saying so at the moment the
 *  marks go in is the difference between "nothing happened" and "now solve". */
function placementAdvice(inst) {
  const n = placeableAnchors(inst);
  if (n < 3) {
    return `${n} anchor(s) reach 2+ photos — three in two photos each is the floor `
         + `for a placement, so mark more`;
  }
  return isPlaced(inst)
    ? `${n} anchors reach 2+ photos`
    : `${n} anchors reach 2+ photos — press Enter to solve and place it`;
}

/** Place the active anchor of the active object at ``p``.
 *
 *  Marking walks through a body's anchors automatically -- place a corner and
 *  the next unmarked one on the same body becomes active -- because marking a
 *  known object means working one body at a time, not one feature at a time. */
function placeAnchor(im, p) {
  const inst = activeInstance();
  if (!inst) {
    setStatus('no object selected — add one in the Known objects panel', 'warn');
    return;
  }
  const template = templateOf(inst);
  if (!template || !(template.anchors || []).length) {
    setStatus(
      'this object has no anchors yet — open the 3D view (D), switch to model, and click its corners',
      'warn'
    );
    return;
  }

  let name = state.activeAnchor;
  if (!name || !template.anchors.some((a) => a.name === name)) {
    name = unmarkedAnchors(inst, im)[0] || template.anchors[0].name;
  }
  im.anchor_obs[anchorKey(inst.id, name)] = [p.x, p.y];
  markDirty();

  const remaining = unmarkedAnchors(inst, im);
  if (remaining.length) {
    state.activeAnchor = remaining[0];
  } else {
    state.activeAnchor = name;
    if (state.autoAdvance && !advanceToMissingAnchor(inst)) {
      // marking changes nothing on its own -- the pose comes out of a solve,
      // and without this the last click of a finished body looked like the end
      // of the job rather than the point at which to run one
      setStatus(
        `${inst.label || inst.id}: every anchor marked in every photo — ${placementAdvice(inst)}`,
        'ok'
      );
    }
  }
  renderPanels();
  render();
}

/** Jump to the next photo missing any anchor of this object. */
function advanceToMissingAnchor(inst) {
  const list = state.project.images.filter((im) => im.enabled);
  const start = list.findIndex((im) => im.id === state.current);
  for (let k = 1; k <= list.length; k++) {
    const cand = list[(start + k) % list.length];
    const missing = unmarkedAnchors(inst, cand);
    if (missing.length) {
      state.activeAnchor = missing[0];
      selectImage(cand.id, true);
      return true;
    }
  }
  return false;
}

function deleteInstance(instanceId) {
  state.project.instances = instanceList().filter((o) => o.id !== instanceId);
  for (const im of state.project.images) {
    for (const key of Object.keys(im.anchor_obs || {})) {
      if (splitAnchorKey(key)[0] === instanceId) delete im.anchor_obs[key];
    }
  }
  if (state.activeInstance === instanceId) {
    state.activeInstance = instanceList().length ? instanceList()[0].id : null;
    state.activeAnchor = null;
  }
  markDirty();
}

function deleteAnchorFromTemplate(template, name) {
  template.anchors = (template.anchors || []).filter((a) => a.name !== name);
  // marks referring to it are now meaningless, so drop them rather than leave
  // the solve to quietly ignore them
  for (const im of state.project.images) {
    for (const key of Object.keys(im.anchor_obs || {})) {
      const [instId, anchorName] = splitAnchorKey(key);
      const inst = instanceList().find((o) => o.id === instId);
      if (anchorName === name && inst && inst.template === template.id) {
        delete im.anchor_obs[key];
      }
    }
  }
  if (state.activeAnchor === name) state.activeAnchor = null;
  markDirty();
}

function deletePoint(pointId) {
  state.project.points = state.project.points.filter((p) => p.id !== pointId);
  for (const im of state.project.images) delete im.observations[pointId];
  state.project.constraints = state.project.constraints.filter(
    (c) => c.point !== pointId && c.point_a !== pointId && c.point_b !== pointId
  );
  if (state.activePoint === pointId) state.activePoint = null;
  markDirty();
}

/* ------------------------------------------------------------- interaction */

canvas.addEventListener('contextmenu', (ev) => {
  ev.preventDefault();
  const im = currentImage();
  if (!im) return;
  // in pose mode the right button turns the body, so it must not also delete
  // whatever mark happens to lie under the pointer
  if (state.mode === 'pose') return;
  const hit = hitTest(toImage(ev));
  if (!hit) return;
  if (hit.kind === 'obs') delete im.observations[hit.point];
  else if (hit.kind === 'anchor') delete im.anchor_obs[hit.key];
  else im.axis_lines.splice(hit.index, 1);
  markDirty();
  renderPanels();
  render();
});

canvas.addEventListener('mousedown', (ev) => {
  state.dragUndoDone = false;
  // Going back to the photo means the panels are finished with. The canvas
  // cannot take focus itself, so without this a control clicked earlier keeps
  // it indefinitely -- and a focused text field goes on eating shortcuts.
  if (document.activeElement && document.activeElement !== document.body) {
    document.activeElement.blur();
  }
  const im = currentImage();
  if (!im) return;
  const p = toImage(ev);

  if (ev.button === 1 || state.spaceDown || ev.altKey) {
    state.drag = { kind: 'pan', sx: ev.clientX, sy: ev.clientY,
                   px: state.view.panX, py: state.view.panY };
    return;
  }
  // Pose mode claims the press before hit testing: while posing, the only
  // thing on the canvas meant to be grabbable is the body, and a press that
  // happened to land on a control-line handle would otherwise drag the line.
  // It also claims the right button, which everywhere else deletes a mark --
  // there is nothing here to delete, and turning a body is wanted often
  // enough to deserve a button of its own rather than a modifier.
  if (state.mode === 'pose' && (ev.button === 0 || ev.button === 2)) {
    const inst = poseTarget(im);
    if (inst) {
      // the handle is decided by the button and the keys held at press time,
      // so a drag cannot change meaning half way through
      const depth = ev.button === 0 && state.depthDown;
      // pressing a button while S is held commits that hold to the drag, so
      // even a press that never moves does not fall through to the space switch
      if (depth) state.depthUsed = true;
      state.drag = {
        kind: 'pose',
        inst: inst.id,
        last: p,
        depth,
        // Shift makes it a resize. Deliberately its own gesture rather than a
        // second meaning for the depth drag: in one photograph a body 1.25x
        // larger and 1.25x further away project identically, so the two have
        // to be told apart by the hand, not by the picture. Shift also keeps
        // it out of reach of an ordinary drag, because this one writes the
        // MODEL's size and so changes every copy of it in every photo.
        size: ev.button === 0 && ev.shiftKey && !state.depthDown,
        rotate: ev.button === 2 || ev.ctrlKey || ev.metaKey,
        // Shift on the turn drag rolls the body about the direction the photo
        // is looking. It is the third turn, and without it the two the eye can
        // judge leave one degree of freedom reachable only through the X/Y/Z
        // locks -- which is fine for a part square to the fixture and useless
        // for one that is not.
        roll: (ev.button === 2 || ev.ctrlKey || ev.metaKey) && ev.shiftKey,
        axisKey: state.axisKey,
      };
      renderPanels();
      render();
    }
    return;
  }

  if (ev.button !== 0) return;

  const hit = hitTest(p);
  if (hit && hit.kind === 'obs') {
    state.activePoint = hit.point;
    state.drag = { kind: 'moveobs', point: hit.point };
    renderPanels();
    render();
    return;
  }
  if (hit && hit.kind === 'anchor') {
    const [instId, name] = splitAnchorKey(hit.key);
    state.activeInstance = instId;
    state.activeAnchor = name;
    state.drag = { kind: 'moveanchor', key: hit.key };
    renderPanels();
    render();
    return;
  }
  // A press on a line, or on one of its endpoint handles, means reverse it --
  // and the arrowhead you naturally aim at IS a handle, so both must flip. It
  // only becomes a drag once the pointer passes the slop, and mouseup decides
  // between the two. In lines mode the drag is a new line, so drawing one that
  // starts on top of an existing line still works. Only in the two modes that
  // are about lines: while marking points or anchors, a press near a line is
  // not about the line, and a silent sign change there would be invisible.
  const flipMode = state.mode === 'select' || state.mode === 'lines';

  if (hit && hit.kind === 'handle') {
    state.drag = { kind: 'movehandle', index: hit.index, which: hit.which };
    if (flipMode) Object.assign(state.drag, { flip: hit.index, down: p, moved: false });
    return;
  }
  if (hit && hit.kind === 'line' && flipMode) {
    state.drag = state.mode === 'select'
      ? { kind: 'moveline', index: hit.index, last: p, down: p, moved: false, flip: hit.index }
      : { kind: 'newline', from: p, to: p, down: p, moved: false, flip: hit.index };
    return;
  }

  if (state.mode === 'lines') {
    state.drag = { kind: 'newline', from: p, to: p };
  } else if (state.mode === 'points') {
    if (!state.activePoint) newPoint();
    im.observations[state.activePoint] = [p.x, p.y];
    markDirty();
    const pid = state.activePoint;
    renderPanels();
    render();
    if (state.autoAdvance) advanceToMissing(pid);
  } else if (state.mode === 'objects') {
    placeAnchor(im, p);
  }
});

window.addEventListener('mousemove', (ev) => {
  const im = currentImage();
  if (!im) return;
  const p = toImage(ev);
  const d = state.drag;

  if (!d) {
    const hit = hitTest(p);
    const changed = JSON.stringify(hit) !== JSON.stringify(state.hover);
    state.hover = hit;
    canvas.style.cursor = hit ? 'pointer' : (state.spaceDown ? 'grab' : 'crosshair');
    if (changed) render();
    return;
  }

  if (d.down && !d.moved
      && Math.hypot(p.x - d.down.x, p.y - d.down.y) * state.view.zoom > CLICK_SLOP_PX) {
    d.moved = true;
  }

  if (d.kind === 'pan') {
    state.view.panX = d.px - (ev.clientX - d.sx) / state.view.zoom;
    state.view.panY = d.py - (ev.clientY - d.sy) / state.view.zoom;
  } else if (d.kind === 'newline') {
    d.to = p;
  } else if (d.kind === 'moveobs') {
    im.observations[d.point] = [p.x, p.y];
  } else if (d.kind === 'moveanchor') {
    im.anchor_obs[d.key] = [p.x, p.y];
  } else if (d.kind === 'movehandle') {
    const ln = im.axis_lines[d.index];
    if (ln && (d.flip === undefined || d.moved)) {
      (d.which === 0 ? ln.p0 : ln.p1).splice(0, 2, p.x, p.y);
    }
  } else if (d.kind === 'pose') {
    const inst = instanceList().find((o) => o.id === d.inst);
    if (inst && isPlaced(inst)) {
      const dx = p.x - d.last.x, dy = p.y - d.last.y;
      // Turning, sizing and depth are judged by how far the hand moved on the
      // screen, not by how many photo pixels that crossed: in image units the
      // same gesture means something wildly different zoomed in than zoomed
      // out -- an 80 px drag multiplied a model's size by 103 at one zoom.
      // Moving the body in the image plane is the exception and stays in image
      // units, because there it has to stay under the pointer.
      const sdx = dx * state.view.zoom, sdy = dy * state.view.zoom;
      if (d.depth) {
        // one axis only: placing a body along the view ray is a
        // one-dimensional judgement, and a handle that also responded to
        // vertical movement would change the answer every time the hand
        // wandered off the line
        //
        // right brings it closer, which grows its silhouette -- the same thing
        // this gesture did to the screen when it resized the model, so the hand
        // keeps the habit it already has while the number underneath it becomes
        // the one that belongs to this placement
        state.depthUsed = true;   // this hold was a drag, so its keyup is not a tap
        nudgeDepth(inst, im, Math.exp(-sdx * 0.004));   // halves in ~170 px
        // the distance in metres, live, because that is the number being set --
        // the silhouette on screen cannot distinguish it from size
        setStatus(`${inst.label || inst.id} at `
          + `${bodyDepth(inst, im).toFixed(2)} m`, 'ok');
      } else if (d.size) {
        /* One axis and multiplicative, like depth: "a bit bigger" is a
         * fraction of the current size, and a handle that also answered to
         * vertical movement would change the answer every time the hand
         * wandered. Being a screen-space horizontal drag is also what makes it
         * work from any view -- the gesture never consults the camera, so it
         * means the same thing from the front, from underneath, and with the
         * view rolled over. */
        const t = templateOf(inst);
        if (t && nudgeTemplateScale(t, Math.exp(sdx * 0.004))) {
          const span = modelSpanNominal(t);
          setStatus(`${t.name || t.id} now ${span ? span.toPrecision(4) : '?'} `
            + `${nominalUnits()} on its longest side — this is the MODEL's size, so `
            + `every copy of it changed`, 'warn');
          // the card's size box and the derived readouts are now stale
          state.poseScaleDirty = true;
        }
      } else if (d.rotate && d.axisKey) {
        // One axis at a time, chosen by the key held down. Free rotation is
        // quick to get close with and hopeless to finish with: every drag
        // moves two angles at once, so correcting the one that is wrong
        // disturbs the one that was already right, and a part square to the
        // fixture never quite lands. Constraining it is what makes the last
        // few degrees reachable at all.
        rotateInPlace(inst, turnAxisWorld(inst, d.axisKey), sdx * 0.006);
      } else if (d.roll) {
        // about the direction this photo is looking, so the body spins in the
        // picture plane -- the one turn that reads the same from every camera
        const ax = cameraAxes(im);
        rotateInPlace(inst, ax.forward, sdx * 0.006);
      } else if (d.rotate) {
        const ax = cameraAxes(im);
        /* Horizontal drag yaws about world up, which is what a body standing
         * on a floor actually does; vertical drag tips it about the screen
         * horizontal, which is the only other turn an eye can judge. Half a
         * turn per ~500 px of screen, so a whole revolution is a comfortable
         * two drags rather than a flick.
         *
         * World up is the right default and the wrong only option. It assumes
         * the body has a floor, and inverted tooling does not: looking up at a
         * fixture from underneath, a world-Z yaw turns the body the opposite
         * way round the screen from the hand, and the turn the eye actually
         * wants -- about what is up in THIS picture -- is not on offer at all.
         * So the free tumble the 3D view got is the same switch here: T. */
        const yawAxis = Viewer3D.orbitMode() === 'trackball'
          ? [-ax.down[0], -ax.down[1], -ax.down[2]]
          : [0, 0, 1];
        if (Math.abs(sdx) > 0) rotateInPlace(inst, yawAxis, sdx * 0.006);
        if (Math.abs(sdy) > 0) rotateInPlace(inst, ax.right, sdy * 0.006);
      } else {
        nudgeInPlane(inst, im, dx, dy);
      }
      inst.manual = true;
      inst.pose_in_part_space = partSpaceState().ok;
      d.last = p;
    }
  } else if (d.kind === 'moveline') {
    const ln = im.axis_lines[d.index];
    if (ln && d.moved) {
      const dx = p.x - d.last.x, dy = p.y - d.last.y;
      ln.p0 = [ln.p0[0] + dx, ln.p0[1] + dy];
      ln.p1 = [ln.p1[0] + dx, ln.p1[1] + dy];
      d.last = p;
    }
  }
  render();
});

window.addEventListener('mouseup', () => {
  const im = currentImage();
  const d = state.drag;
  state.drag = null;
  // the size box, the real-size readout and the units-vs-nominal check all
  // move with a resize; rebuilding the panel per frame while dragging would
  // blow focus around, so it happens once, here
  if (state.poseScaleDirty) {
    state.poseScaleDirty = false;
    renderPanels();
    syncViewer();
    refreshRig();
  }
  if (!d || !im) return;

  if (d.kind === 'newline') {
    const len = Math.hypot(d.to.x - d.from.x, d.to.y - d.from.y) * state.view.zoom;
    if (len > 12) {
      const ln = { axis: state.axis, p0: [d.from.x, d.from.y], p1: [d.to.x, d.to.y] };
      if (state.dirGroup && dirGroupOf(state.dirGroup)) ln.direction_group = state.dirGroup;
      im.axis_lines.push(ln);
      markDirty();
      if (ln.direction_group) {
        const t = groupTally(ln.direction_group);
        setStatus(`${groupLabel(ln.direction_group)}: ${t.lines} line(s) across ${t.images} photo(s)`
                + ` — ${groupAdvice(t)}`, t.lines - 2 > 0 ? 'ok' : 'warn');
      }
      renderPanels();
    } else if (d.flip !== undefined) {
      flipLine(im, d.flip);
      return;
    }
  } else if (d.flip !== undefined && !d.moved) {
    flipLine(im, d.flip);
    return;
  } else if (d.kind !== 'pan') {
    markDirty();
    renderPanels();
  }
  render();
});

canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const before = toImage(ev);
  const factor = Math.exp(-ev.deltaY * 0.0014);
  state.view.zoom = clamp(state.view.zoom * factor, 0.02, 40);
  const after = toImage(ev);
  state.view.panX += before.x - after.x;
  state.view.panY += before.y - after.y;
  render();
}, { passive: false });

/* --------------------------------------------------------------- keyboard */

/** Is the user typing into something, rather than merely having left focus on
 *  a control they clicked?
 *
 *  Worth separating, because the two were treated alike and it cost the whole
 *  keyboard. The test used to be "an input, a select or a textarea", which is
 *  also every checkbox in the toolbar and every dropdown in the panels -- so
 *  choosing units after loading a model left focus on that select and silently
 *  killed every shortcut in the app. There was no way back, either: a canvas
 *  takes no focus, so clicking the photo did not clear it. Only a field that
 *  swallows letters has any business swallowing shortcuts. */
function typingInto(el) {
  if (!el) return false;
  if (el.isContentEditable) return true;
  if (el.tagName === 'TEXTAREA') return true;
  if (el.tagName !== 'INPUT') return false;
  const t = (el.type || 'text').toLowerCase();
  return !['checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'range',
           'color'].includes(t);
}

window.addEventListener('keydown', (ev) => {
  if (typingInto(ev.target)) return;
  if (dialogOpen()) return;
  const k = ev.key.toLowerCase();

  if (k === ' ') { state.spaceDown = true; canvas.style.cursor = 'grab'; ev.preventDefault(); return; }
  if ((ev.ctrlKey || ev.metaKey) && k === 'z') {
    ev.preventDefault();
    if (ev.shiftKey) doRedo(); else doUndo();
    return;
  }
  if ((ev.ctrlKey || ev.metaKey) && k === 'y') { ev.preventDefault(); doRedo(); return; }
  if (ev.ctrlKey && k === 's') { ev.preventDefault(); saveNow(); return; }
  if (k === 'enter') { ev.preventDefault(); solveNow(); return; }
  if (k === '?' || (k === '/' && ev.shiftKey)) { ev.preventDefault(); openDocs(); return; }

  if (k === 'escape' && !$('showmenu').hidden) {
    $('showmenu').hidden = true;
    $('btn-show').classList.remove('on');
    return;
  }
  if (k === 'escape' && Viewer3D.isOn()) { toggleViewer(false); return; }
  if (k === 'd') { toggleViewer(!Viewer3D.isOn()); return; }

  /* One switch for every free turn in the app: the 3D orbit and the body's own
   * rotate drag read the same mode, so "horizontal means world Z" or
   * "horizontal means whatever is up on screen" is answered once rather than
   * learned twice. It is bound outside the viewer block because the body
   * gizmo is used with the viewer closed. */
  if (k === 't') {
    Viewer3D.setOrbitMode(Viewer3D.orbitMode() === 'trackball' ? 'turntable' : 'trackball');
    updateViewerBar();
    updateHint();
    setStatus(Viewer3D.orbitMode() === 'trackball'
      ? 'free tumble — a horizontal drag turns about whatever is up on screen, '
        + 'for the view and for the body'
      : 'turntable — a horizontal drag turns about world Z, for the view and '
        + 'for the body', 'ok');
    return;
  }

  /* While the 3D view is up it owns the view keys outright.
   *
   * It covers the whole canvas, so the 2D bindings underneath it act on
   * something nobody can see -- F fitting the invisible photo was the first
   * case of that, and the axis and roll keys below would be the next. Claiming
   * them here, with a return, is also what makes X/Y/Z mean "put the eye on
   * that axis" in the viewer and "turn the body about that axis" on the photo
   * without the two ever colliding. */
  if (Viewer3D.isOn()) {
    if (k === 'f') {
      if (ev.shiftKey) fitSelection(); else Viewer3D.frameAll();
      return;
    }
    if (k === 'x' || k === 'y' || k === 'z') {
      // Shift takes the far side, so the six axis views need three keys. The
      // Z pair is the point of the exercise: Shift+Z looks straight up the +Z
      // axis at the underside of an inverted fixture, which is the view the
      // old pitch clamp could get within a few degrees of and never reach.
      const key = (ev.shiftKey ? '-' : '') + k;
      Viewer3D.setAxisView(key);
      setStatus(`eye on ${key.toUpperCase()}`, 'ok');
      return;
    }
    if (k === 'r') { Viewer3D.levelRoll(); return; }
    if (k === 'q' || k === 'e') {
      Viewer3D.rollBy(((ev.shiftKey ? 15 : 5) * (k === 'e' ? 1 : -1) * Math.PI) / 180);
      return;
    }
  }

  if (k === '1' || k === '2' || k === '3') { setAxis(+k - 1); return; }
  /* 4 steps through the direction groups and back out to the axes, so a set of
   * tray lines can be marked across a photo set without going to the panel
   * between each one. */
  if (k === '4') {
    const ids = dirGroups().map((g) => g.id);
    if (!ids.length) {
      newDirGroup();
    } else {
      const at = ids.indexOf(state.dirGroup);
      if (at + 1 >= ids.length) setAxis(state.axis);
      else setDirGroup(ids[at + 1]);
    }
    return;
  }
  if (k === 'l') { setMode('lines'); return; }
  if (k === 'p') { setMode('points'); return; }
  if (k === 'o') { setMode('objects'); return; }
  // S is the depth handle while posing, so it must not also leave pose mode.
  /* S does two jobs in pose mode, split by whether it was held or tapped.
   *
   * Held, it is the depth handle -- that is what it has always been, it sits
   * beside the other two drag modifiers, and taking it away would cost the one
   * gesture that places a body along the view ray.
   *
   * Tapped, it switches which frame X/Y/Z name. That was free to give: a tap
   * on S did nothing at all before, since the keydown only armed a drag that
   * then never happened. So the modifier keeps its key and the switch gets the
   * key it was asked for, and neither has to be reached for with a chord.
   */
  if (k === 's' && state.mode === 'pose') {
    if (!state.depthDown) { state.depthDown = true; state.depthUsed = false; updateHint(); }
    return;
  }
  /* Objects mode gets the switch too, without the depth handle it has no drag
   * for. Scoping it to pose mode alone was wrong in the way that matters: a
   * body is placed from Objects mode as often as from Pose, S there still meant
   * "go to Select", and the jump is quiet enough to miss -- so the next X-locked
   * turn came out in part space, the switch looked broken, and nothing on screen
   * said why. The cost is the Select shortcut from this one mode, which is a
   * button away; the frame a turn uses is not. */
  if (k === 's' && state.mode === 'objects') { cycleTransformSpace(); return; }
  // H recalls a body that has drifted out of this photo's frame.
  if (k === 'h' && state.mode === 'pose') {
    const inst = activeInstance();
    const im = currentImage();
    if (inst && im && isPlaced(inst) && bringIntoView(inst, im)) {
      setStatus(`${inst.label || inst.id} brought onto this view at `
                + `${bodyDepth(inst, im).toFixed(2)} m — its turn and size are unchanged`,
                'ok');
      renderPanels();
      render();
    } else {
      setStatus('nothing placed to bring into view', 'warn');
    }
    return;
  }
  // X, Y and Z pick a single world axis to turn about, held the same way.
  if (state.mode === 'pose' && (k === 'x' || k === 'y' || k === 'z')) {
    if (state.axisKey !== k) { state.axisKey = k; updateHint(); render(); }
    return;
  }
  // With an axis held, the arrows turn by an exact amount instead of stepping
  // to the next photo: a quarter turn is the commonest correction there is on
  // a part that was modelled in a different orientation from the one it is
  // mounted in, and it is not an angle worth hunting for by hand.
  if (state.mode === 'pose' && state.axisKey
      && (k === 'arrowleft' || k === 'arrowright')) {
    ev.preventDefault();
    const inst = activeInstance();
    if (inst && isPlaced(inst)) {
      rotateSteps(inst, state.axisKey,
                  (ev.shiftKey ? 90 : 15) * (k === 'arrowright' ? 1 : -1));
      // names the frame, because "about X" is the one thing this message could
      // say that does not distinguish the two turns it might have just made
      setStatus(`turned ${ev.shiftKey ? 90 : 15}° about ${spaceLabel()} `
                + `${state.axisKey.toUpperCase()}`, 'ok');
      renderPanels();
      render();
    }
    return;
  }
  if (k === 's') { setMode('select'); return; }
  if (k === 'g') { setMode('pose'); return; }
  if (k === 'n') { newPoint(); renderPanels(); render(); return; }
  if (k === 'v') { $('showvp').click(); return; }
  if (k === 'r') { $('showresid').click(); return; }
  if (k === 'e') { $('showexpected').click(); return; }
  // G used to toggle the grid and now opens pose mode, so this was dead and
  // misleading. The grid keeps its checkbox; a second binding for it would
  // only be another key to collide with.
  if (k === 'm') { $('showmodels').click(); return; }
  if (k === 'h') { $('showshaded').click(); return; }
  if (k === 'f') { fitView(); render(); return; }
  if (k === 'tab') {
    ev.preventDefault();
    if (state.mode === 'objects') {
      const inst = activeInstance();
      if (inst) advanceToMissingAnchor(inst);
    } else if (state.activePoint) {
      advanceToMissing(state.activePoint);
    }
    return;
  }

  if (k === 'arrowleft' || k === '[') { step(-1); return; }
  if (k === 'arrowright' || k === ']') { step(1); return; }

  /* U: "I could not see that clearly."
   *
   * On the hover target rather than on a selection, to match how Delete
   * already works here -- point at the thing and say it. A doubtful mark is
   * not a mark to delete: in a photo with only two lines on an axis it may be
   * all there is, so it keeps contributing at a wider sigma and is simply
   * first in line when the solve decides something has to go. */
  if (k === 'u') {
    const im = currentImage();
    if (im && state.hover) {
      if (state.hover.kind === 'line' || state.hover.kind === 'handle') {
        const ln = im.axis_lines[state.hover.index];
        if (ln) {
          ln.uncertain = !ln.uncertain;
          setStatus(`${AXIS_NAME[ln.axis]} line marked ${ln.uncertain ? 'uncertain' : 'confident'}`, 'ok');
        }
      } else if (state.hover.kind === 'obs') {
        const pt = state.project.points.find((q) => q.id === state.hover.point);
        if (pt) {
          pt.uncertain = !pt.uncertain;
          setStatus(`${pt.label || pt.id} marked ${pt.uncertain ? 'uncertain' : 'confident'}`
                  + ` — this applies to the point in every photo`, 'ok');
        }
      }
      markDirty();
      renderPanels();
      render();
    } else {
      setStatus('hover a control line or a tie point, then press U', 'warn');
    }
    return;
  }

  if (k === 'delete' || k === 'backspace') {
    const im = currentImage();
    if (im && state.hover) {
      if (state.hover.kind === 'obs') delete im.observations[state.hover.point];
      else if (state.hover.kind === 'anchor') delete im.anchor_obs[state.hover.key];
      else if (state.hover.kind === 'line' || state.hover.kind === 'handle') {
        im.axis_lines.splice(state.hover.index, 1);
      }
      state.hover = null;
      markDirty();
      renderPanels();
      render();
    }
  }
});

window.addEventListener('keyup', (ev) => {
  if (ev.key === ' ') { state.spaceDown = false; canvas.style.cursor = 'crosshair'; }
  const ku = ev.key.toLowerCase();
  if (ku === 's' && state.depthDown) {
    const tapped = !state.depthUsed;
    state.depthDown = false;
    state.depthUsed = false;
    if (tapped) cycleTransformSpace(); else updateHint();
  }
  if (state.axisKey === ku) { state.axisKey = null; updateHint(); render(); }
});

function step(delta) {
  const list = state.project.images;
  const i = list.findIndex((im) => im.id === state.current);
  if (i < 0) return;
  selectImage(list[(i + delta + list.length) % list.length].id, true);
}

function setMode(mode) {
  state.mode = mode;
  for (const b of $('modes').children) b.classList.toggle('on', b.dataset.mode === mode);
  // The axis buttons draw control lines and do nothing in any other mode.
  // They used to be dimmed rather than removed, which still spent four slots
  // on four inert buttons; the flexible spacer sits immediately to their
  // right, so hiding them moves nothing else on the bar.
  $('axes').hidden = mode !== 'lines';
  // marking anchors means looking at the photo, so step out of the 3D view
  if ((mode === 'objects' || mode === 'pose')
      && Viewer3D.isOn() && Viewer3D.mode() === 'template') toggleViewer(false);
  // dragging a body you cannot see is not dragging
  if (mode === 'pose' && !state.showModels) $('showmodels').click();
  // S and the axis keys mean something only while posing; leaving must not
  // strand either of them held down
  if (mode !== 'pose') { state.depthDown = false; state.axisKey = null; }
  updateHint();
  render();
}

function setAxis(axis) {
  state.axis = axis;
  state.dirGroup = null;
  syncAxisButtons();
  if (state.mode !== 'lines') setMode('lines');
  updateHint();
}

/** Draw the next control line into a direction group instead of onto an axis.
 *
 *  Passing a group that no longer exists falls back to the axis buttons rather
 *  than leaving the toolbar in a state that would attach new lines to nothing. */
function setDirGroup(gid) {
  const g = gid ? dirGroupOf(gid) : null;
  if (!g) { setAxis(state.axis); return; }
  state.dirGroup = g.id;
  syncAxisButtons();
  if (state.mode !== 'lines') setMode('lines');
  updateHint();
  renderDirGroupPanel();
  render();
}

function syncAxisButtons() {
  for (const b of $('axes').children) {
    const isGroup = b.dataset.axis === 'group';
    b.classList.toggle('on', isGroup ? !!state.dirGroup : !state.dirGroup && +b.dataset.axis === state.axis);
  }
  const btn = $('axisgroup');
  if (btn) {
    btn.textContent = state.dirGroup ? (groupLabel(state.dirGroup).slice(0, 9) || 'Grp') : 'Grp';
    btn.style.color = state.dirGroup ? groupColor(state.dirGroup) : '';
  }
}

/** One line on what a group's counts mean for the solve.
 *
 *  Mirrors `dirgroup.contribution_note` on the Python side. Duplicated rather
 *  than fetched because it has to be right while the user is still drawing,
 *  before any solve has run -- and the arithmetic is two numbers, so the risk
 *  of the two drifting is smaller than the risk of saying nothing until a
 *  solve. */
function groupAdvice(tally) {
  if (tally.images < 2) {
    return tally.lines < 2
      ? 'one line fixes only a plane — draw this edge in another photo'
      : 'all in one photo, so it is ignored: draw this edge in a second photo';
  }
  const net = tally.lines - 2;
  if (net <= 0) return 'break-even — a third line is where it starts to help';
  return `${net} net constraint${net === 1 ? '' : 's'}`;
}

function updateHint() {
  const inst = activeInstance();
  const anchor = state.activeAnchor;
  const objectHint = !inst
    ? `No object selected. Add a model and an object in the <b>Known objects</b> panel.`
    : !anchor
      ? `<b>${escapeHtml(inst.label || inst.id)}</b> has no anchor selected — `
        + `pick its corners on the model in the 3D view (<b>D</b>).`
      : `Click <b>${escapeHtml(anchor)}</b> on <b>${escapeHtml(inst.label || inst.id)}</b>. `
        + `The next unmarked anchor follows automatically; <b>Tab</b> jumps to the next photo. `
        + `${escapeHtml(placementAdvice(inst))}.`;

  const tumble = Viewer3D.orbitMode() === 'trackball';
  const turnNote = tumble
    ? `<b>T</b>: a turn drag now yaws about screen-up — the mode for inverted tooling. `
      + `<b>Shift</b> on it rolls in the picture plane.`
    : `<b>T</b>: a turn drag yaws about world Z — switch it to a free tumble for `
      + `inverted tooling. <b>Shift</b> on it rolls in the picture plane.`;

  const poseHint = !inst
    ? `No object selected. Add a model and an object in the <b>Known objects</b> panel.`
    : state.axisKey
      ? `<b>${state.axisKey.toUpperCase()} axis locked`
        + `${state.transformSpace === 'object' ? ', in OBJECT space' : ''}.</b> `
        + `Right-drag turns about ${spaceLabel()} `
        + `<b>${state.axisKey.toUpperCase()}</b> and nothing else — `
        + (state.transformSpace === 'object'
            ? `the body's own axis, as modelled.`
            : `the scene's axis, the frame every reported number uses.`)
        + ` <b>←</b>/<b>→</b> turn exactly 15°, with <b>Shift</b> exactly 90°. `
        + `Tap <b>S</b> for ${state.transformSpace === 'object' ? 'part' : 'object'} space; `
        + `release ${state.axisKey.toUpperCase()} for free rotation.`
      : state.depthDown
      ? `<b>Distance:</b> drag right to bring it closer, left to push it away. The `
        + `status line shows the distance in metres as you go. This moves <i>this body</i> `
        + `and leaves the model's size alone — but in this one photo the two look the `
        + `same, so settle it against a photo 60–120° away. The model's own size is `
        + `<b>Shift-drag</b>, or the <b>size</b> box in Known objects.`
      : isPlaced(inst) && currentImage() && bodyVisibility(inst, currentImage()) === 0
      ? `<b>${escapeHtml(inst.label || inst.id)} is outside this photo.</b> That is normal `
        + `on the second view — these were taken metres apart, so a body a little off in `
        + `one is off the edge in another. Press <b>H</b> to bring it onto this view at `
        + `the same distance; its turn and size are kept.`
      : !((templateOf(inst) || {}).anchors || []).length
      ? `<b>Left-drag</b> moves it, <b>right-drag</b> turns it, <b>hold S</b> for distance, `
        + `<b>Shift</b>+left-drag for the model's size. But this model has no anchors yet, `
        + `so the fit stays an eye-fit and nothing can refine it — press `
        + `<b>use model features</b> in Known objects to take the ones the model already `
        + `offers.`
      : `<b>Left-drag</b> moves it, <b>right-drag</b> turns it, <b>hold S</b> for distance, `
        + `<b>hold X/Y/Z</b> for one world axis. <b>Shift</b>+left-drag resizes the `
        + `<i>model</i> — every copy, every photo — and <b>Shift</b>+right-drag rolls it. `
        + turnNote + ` `
        + `The body has one pose, so every photo in the filmstrip moves as you drag. When it `
        + `looks right here, <b>snap anchors</b>, then the next photo — <b>Enter</b> `
        + `reconciles them into one measured pose and moves the cameras to match.`;

  const tally = state.dirGroup ? groupTally(state.dirGroup) : null;
  const linesHint = state.dirGroup
    ? `Drag along the same edge you marked as <b>${escapeHtml(groupLabel(state.dirGroup))}</b> `
      + `in the other photos. <b>Either end — it makes no difference</b>, unlike an axis line: `
      + `the solve works this direction out, so it has no positive end to point at. `
      + `So far ${tally.lines} line(s) across ${tally.images} photo(s) — ${groupAdvice(tally)}. `
      + `Right-click to delete, <b>U</b> to flag uncertain. <b>4</b> steps through the groups, `
      + `1/2/3 go back to an axis.`
    : `Drag along an edge parallel to the <b>${AXIS_NAME[state.axis]}</b> axis. `
      + `Draw in the <b>+${AXIS_NAME[state.axis]}</b> direction — the arrow shows which way. `
      + `Click a drawn line to reverse it, right-click to delete, <b>U</b> to flag it `
      + `uncertain. Two per axis minimum, three is better. Keys 1/2/3 switch axis, `
      + `<b>4</b> draws into a direction group instead.`;

  const hints = {
    lines: linesHint,
    points: `Click the same physical feature in each photo it appears in. `
          + `<b>N</b> starts a new point, <b>Tab</b> jumps to the next photo missing it, `
          + `<b>U</b> flags one uncertain. Two shared points per image pair is the minimum.`,
    objects: objectHint,
    pose: poseHint,
    select: `Drag markers or whole control lines; click a line to reverse it. `
          + `Right-click or Delete removes one, <b>U</b> flags it uncertain.`,
  };
  $('hint').innerHTML = hints[state.mode]
    + ` &nbsp;·&nbsp; wheel zoom, space-drag pan, <b>F</b> fit`
    // which frame a turn uses survives every other gesture, so it is shown
    // wherever pose mode is, not only while an axis happens to be held
    + (state.mode === 'pose' || state.mode === 'objects'
        ? ` &nbsp;·&nbsp; turning in <b>${spaceLabel()}</b> space (tap <b>S</b>)` : '')
    + (state.showResid ? ` &nbsp;·&nbsp; residuals ×${RESIDUAL_GAIN}` : '');
}

/* ----------------------------------------------------------------- filmstrip */

function imageStatus(im) {
  const rep = state.report && state.report.images ? state.report.images[im.id] : null;
  const ready = [0, 1, 2].filter((a) => axisLinesOn(im, a).length >= 2);
  if (!im.enabled) return { color: '#555', text: 'off' };
  if (ready.length < 2) return { color: '#777', text: `${ready.length}/2 axes` };
  if (rep && rep.positioned) {
    const rms = rep.rms_px;
    return {
      color: rms < 2 ? 'var(--good)' : rms < 6 ? 'var(--warn)' : 'var(--bad)',
      text: `${rms.toFixed(1)}px`,
    };
  }
  return { color: 'var(--warn)', text: `${Object.keys(im.observations).length} pts` };
}

function buildFilmstrip() {
  const strip = $('filmstrip');
  strip.innerHTML = '';
  for (const im of state.project.images) {
    const st = imageStatus(im);
    const div = document.createElement('div');
    div.className = 'film' + (im.id === state.current ? ' sel' : '') + (im.enabled ? '' : ' off');
    div.innerHTML =
      // not draggable: dragging a tile hands the browser the thumbnail as a
      // file, which the window drop handler then uploads as a new 220px photo
      `<img loading="lazy" alt="" draggable="false">` +
      `<canvas class="filmov"></canvas>` +
      `<button class="rm" title="Remove from this project">&times;</button>` +
      `<div class="meta"><span>${st.text}</span>` +
      `<span class="dot" style="background:${st.color}"></span></div>`;
    // the thumbnail's source is resolved asynchronously, since in static mode it
    // is a blob the Python side has to render first
    Backend.photoSrc(im.id, true)
      .then((src) => {
        const img = div.querySelector('img');
        if (!img) return;
        // the tile has no height until its thumbnail arrives, and a canvas
        // sized against a zero-height tile draws nothing
        img.onload = () => drawFilmstripOverlays();
        img.src = src;
      })
      .catch(() => {});
    div.onclick = () => selectImage(im.id, true);
    div.querySelector('.rm').onclick = (ev) => { ev.stopPropagation(); removeImage(im.id); };
    strip.appendChild(div);
  }

  const add = document.createElement('button');
  add.className = 'smallbtn';
  add.style.width = '100%';
  add.textContent = '+ add photos';
  add.title = 'Upload photos into this project (or just drag them onto the canvas)';
  add.onclick = () => $('fileinput').click();
  strip.appendChild(add);

  // referencing photos in place needs a filesystem to reference; in the browser
  // there is only what has been dropped in
  if (!Backend.isStatic()) {
    const link = document.createElement('button');
    link.className = 'smallbtn';
    link.style.cssText = 'width:100%;margin-top:4px';
    link.textContent = 'link a folder…';
    link.title = 'Reference photos where they already sit on disk, without copying them in';
    link.onclick = addFolder;
    strip.appendChild(link);
  }

  const row = document.createElement('div');
  row.id = 'uploadrow';
  strip.appendChild(row);
}

/* -------------------------------------------------------------------- panels */

function renderPanels() {
  renderSolvePanel();
  renderImagePanel();
  renderDirGroupPanel();
  renderPartPanel();
  renderObjectsPanel();
  renderPointsPanel();
  renderViewpointPanel();
  renderCadPanel();
  renderPlanesPanel();
  renderConstraintsPanel();
  renderImageListPanel();
  updateHint();
  scheduleObservability();
}

function renderSolvePanel() {
  const el = $('solvepanel');
  const r = state.report;
  if (!r) {
    el.innerHTML = '<div class="empty">Not solved yet. Mark control lines and tie points, then press Solve.</div>';
    return;
  }
  const rows = [];
  if (r.ok) {
    rows.push(`<div class="row"><span class="k">reprojection RMS</span><span class="v">${r.rms_reprojection_px.toFixed(2)} px</span></div>`);
    rows.push(`<div class="row"><span class="k">control-line RMS</span><span class="v">${r.rms_line_px.toFixed(2)} px</span></div>`);
    rows.push(`<div class="row"><span class="k">cameras placed</span><span class="v">${r.n_images_positioned}</span></div>`);
    rows.push(`<div class="row"><span class="k">points</span><span class="v">${r.n_points_triangulated}</span></div>`);
    for (const [gid, f] of Object.entries(r.focals || {})) {
      rows.push(`<div class="row"><span class="k">focal (${gid})</span><span class="v">${f.toFixed(0)} px</span></div>`);
    }
    if (r.focal_sigma_pct != null) {
      const v = r.focal_sigma_pct;
      const cls = v < 2 ? 'v-good' : v < 8 ? 'v-weak' : 'v-poor';
      rows.push(`<div class="row"><span class="k">focal certainty</span>`
              + `<span class="v ${cls}">&plusmn;${v.toFixed(v < 10 ? 1 : 0)}%</span></div>`);
    }
    rows.push(`<div class="row"><span class="k">init conditioning</span><span class="v">${r.init_condition.toFixed(0)}</span></div>`);
    rows.push(`<div class="row"><span class="k">scale</span><span class="v">${r.scale_is_metric ? 'metric' : 'arbitrary'}</span></div>`);
  } else {
    rows.push(`<div class="note bad">stopped at: ${r.stage}</div>`);
  }
  for (const m of r.messages || []) rows.push(`<div class="msg">${escapeHtml(m)}</div>`);
  for (const w of r.warnings || []) rows.push(`<div class="note">${escapeHtml(w)}</div>`);
  el.innerHTML = rows.join('');
}

function renderImagePanel() {
  const im = currentImage();
  const el = $('imgpanel');
  if (!im) {
    $('imgcount').textContent = '';
    el.innerHTML = '<div class="empty">No photos yet. Drag some onto the canvas, '
                 + 'or use “+ add photos” in the strip on the left.</div>';
    return;
  }
  const meta = state.meta[im.id] || {};
  const rep = state.report && state.report.images ? state.report.images[im.id] : null;
  const g = groupOf(im);

  const counts = [0, 1, 2].map((a) => axisLinesOn(im, a).length);
  $('imgcount').textContent = meta.name || im.id;
  /* A pose moved by the edge alignment is not a pose the bundle adjustment ever
   * agreed to, and it outlives the panel that produced it. Saying so here --
   * where you look when you are reading this photo's residuals -- is the
   * difference between a recorded caveat and a visible one. */
  const refinedNote = im.edge_refined
    ? `<div class="note">This camera's pose was moved by the edge alignment, not by the solve. `
      + `Its residuals below are against that pose. `
      + `<button class="smallbtn" id="imgrefineundo">put the solved pose back</button></div>`
    : '';

  let html = `<div class="axiscounts">` + counts.map((c, a) =>
    `<div class="${['x','y','z'][a]} ${c < 2 ? 'short' : ''}">${AXIS_NAME[a]} ${c}</div>`
  ).join('') + `</div>`;

  const freeHere = im.axis_lines.filter((l) => l.direction_group).length;
  if (freeHere) {
    // Said beside the axis counts, never inside them: these do not help this
    // photo calibrate, and a count that implied they did would be a lie the
    // solve then has to contradict.
    html += `<div class="row" style="margin-top:7px"><span class="k">direction-group lines</span>`
          + `<span class="v">${freeHere}</span></div>`
          + `<div class="note">Not counted above: a group's direction is worked out by the `
          + `solve, so it cannot calibrate the camera that sees it. It constrains the bundle `
          + `once this photo is calibrated from two real axes.</div>`;
  }

  html += `<div class="row" style="margin-top:7px"><span class="k">tie points here</span>`
        + `<span class="v">${Object.keys(im.observations).length}</span></div>`;
  if (rep && rep.positioned) {
    html += `<div class="row"><span class="k">reproj RMS</span><span class="v">${rep.rms_px.toFixed(2)} px</span></div>`;
    html += `<div class="row"><span class="k">worst</span><span class="v">${rep.max_px.toFixed(2)} px`
          + (rep.worst_point ? ` (${rep.worst_point})` : '') + `</span></div>`;
    html += `<div class="row"><span class="k">line RMS</span><span class="v">${rep.line_rms_px.toFixed(2)} px</span></div>`;
  }
  for (const n of (rep && rep.notes) || []) html += `<div class="note">${escapeHtml(n)}</div>`;
  html += refinedNote;

  html += `<div class="row" style="margin-top:7px"><span class="k">camera group</span>`
        + `<input type="text" id="grpname" value="${escapeHtml(im.group)}" style="width:90px"></span></div>`;
  html += `<label class="toggle" style="margin-top:5px"><input type="checkbox" id="imgenabled"`
        + `${im.enabled ? ' checked' : ''}> include in solve</label>`;
  html += `<label class="toggle"><input type="checkbox" id="lockpp"`
        + `${g.lock_principal_point ? ' checked' : ''}> lock principal point to centre</label>`;
  html += `<label class="toggle"><input type="checkbox" id="lockk1"`
        + `${g.lock_k1 ? ' checked' : ''}> lock distortion (k1)</label>`;

  // focal length: show it, let it be typed or taken from EXIF, and lock it
  const mmEq = g.f ? pxTo35mm(g.f, im) : null;
  html += `<div class="focalbox">`
        + `<div class="row"><span class="k">focal length</span>`
        + `<span class="focalnow">`
        + (g.f
            ? `${g.f.toFixed(0)} px <span class="est">&asymp;${mmEq.toFixed(0)} mm eq</span>`
            : `<span class="est">from control lines</span>`)
        + `</span></div>`
        + `<div class="focalrow">`
        +   `<input type="number" id="focalval" step="any" placeholder="${g.f ? g.f.toFixed(0) : 'value'}">`
        +   `<select id="focalunit"><option value="px">px</option>`
        +   `<option value="mm35">mm eq</option></select>`
        +   `<button class="smallbtn" id="focalset">set &amp; lock</button>`
        + `</div>`
        + `<div id="exifrow" class="msg">EXIF: checking&hellip;</div>`
        + `<label class="toggle"><input type="checkbox" id="lockf"`
        + `${g.lock_focal ? ' checked' : ''}> lock focal length</label>`
        + (g.lock_focal
            ? `<button class="smallbtn" id="focalclear" style="margin-top:4px">unlock and re-estimate</button>`
            : '')
        + `</div>`;
  html += `<div style="display:flex;gap:5px;margin-top:7px">`
        + `<button class="smallbtn" id="clearlines">clear control lines</button>`
        + `<button class="smallbtn" id="removeimg">remove photo</button></div>`;
  el.innerHTML = html;

  $('grpname').onchange = (e) => { im.group = e.target.value.trim() || 'cam0'; groupOf(im); markDirty(); renderPanels(); };
  $('imgenabled').onchange = (e) => { im.enabled = e.target.checked; markDirty(); buildFilmstrip(); };
  $('lockpp').onchange = (e) => { g.lock_principal_point = e.target.checked; markDirty(); };
  $('lockk1').onchange = (e) => { g.lock_k1 = e.target.checked; markDirty(); };
  $('lockf').onchange = (e) => {
    if (e.target.checked && !g.f) {
      // nothing to lock yet: locking an unset focal would just freeze a guess
      e.target.checked = false;
      setStatus('no focal length to lock yet — type one, use EXIF, or solve first', 'warn');
      return;
    }
    g.lock_focal = e.target.checked;
    markDirty();
    renderPanels();
  };
  $('focalset').onclick = () => {
    const raw = parseFloat($('focalval').value);
    if (!(raw > 0)) { setStatus('enter a positive focal length', 'warn'); return; }
    const px = $('focalunit').value === 'mm35' ? pxFrom35mm(raw, im) : raw;
    applyFocal(px, $('focalunit').value === 'mm35' ? `${raw} mm equivalent` : 'a typed value');
  };
  const clearBtn = $('focalclear');
  if (clearBtn) clearBtn.onclick = () => {
    g.lock_focal = false;
    g.f = null;
    markDirty();
    renderPanels();
    setStatus('focal unlocked — it will be re-estimated from the control lines', 'ok');
  };
  $('clearlines').onclick = () => { im.axis_lines = []; markDirty(); renderPanels(); render(); };
  $('removeimg').onclick = () => removeImage(im.id);
  const undoOne = $('imgrefineundo');
  if (undoOne) undoOne.onclick = () => revertEdges([im.id]);
  scheduleExif();
}

/* --------------------------------------------------- direction groups panel */

function newDirGroup() {
  const used = new Set(dirGroups().map((g) => g.id));
  let i = 0;
  while (used.has(`d${i}`)) i++;
  const id = `d${i}`;
  dirGroups().push({ id, label: `group ${dirGroups().length + 1}`, d: null });
  markDirty();
  setDirGroup(id);
  renderPanels();
  setStatus(`drawing into ${groupLabel(id)} — mark the same edge in two or more photos; `
          + `either end, it makes no difference`, 'ok');
  return id;
}

function deleteDirGroup(gid) {
  const t = groupTally(gid);
  if (t.lines && !confirm(
        `Delete ${groupLabel(gid)} and the ${t.lines} control line(s) drawn into it?\n\n`
      + `The lines go too. They cannot become axis lines: nobody ever said which axis `
      + `they follow, and guessing would quietly feed the solve a false fact.`)) return;
  for (const im of state.project.images) {
    im.axis_lines = im.axis_lines.filter((l) => l.direction_group !== gid);
  }
  state.project.direction_groups = dirGroups().filter((g) => g.id !== gid);
  if (state.dirGroup === gid) setAxis(state.axis);
  markDirty();
  renderPanels();
  render();
}

function renderDirGroupPanel() {
  const el = $('dgpanel');
  const badge = $('dgcount');
  if (!el) return;
  const groups = dirGroups();
  const reports = {};
  for (const r of (state.report && state.report.direction_groups) || []) reports[r.id] = r;

  badge.textContent = groups.length ? `${groups.length}` : '';

  let html = '';
  if (!groups.length) {
    html += `<div class="empty">A direction group is a long straight edge whose world `
          + `direction nobody knows — a cable tray, a conduit run, a handrail. Mark the same `
          + `edge in two or more photos and the solve works out which way it goes, using it to `
          + `tie those cameras together. Three lines is where a group starts to help: it costs `
          + `two unknowns of its own.</div>`;
  }

  for (const g of groups) {
    const t = groupTally(g.id);
    const rep = reports[g.id];
    const active = state.dirGroup === g.id;
    const col = groupColor(g.id);
    const net = t.lines - 2;
    const grade = t.images < 2 ? 'bad' : net > 0 ? 'good' : 'warn';
    html += `<div class="dgrow${active ? ' sel' : ''}" data-gid="${escapeHtml(g.id)}">`
          + `<div class="dghead">`
          +   `<span class="dgswatch" style="background:${col}"></span>`
          +   `<input type="text" class="dglabel" data-gid="${escapeHtml(g.id)}" `
          +     `value="${escapeHtml(g.label || g.id)}">`
          +   `<button class="smallbtn dgdraw" data-gid="${escapeHtml(g.id)}">`
          +     `${active ? 'drawing' : 'draw'}</button>`
          +   `<button class="smallbtn dgdel" data-gid="${escapeHtml(g.id)}" title="Delete this group and its lines">&times;</button>`
          + `</div>`
          + `<div class="row"><span class="k">lines / photos</span>`
          +   `<span class="v">${t.lines} / ${t.images}</span></div>`
          + `<div class="row"><span class="k">net constraints</span>`
          +   `<span class="v ${grade}">${t.images < 2 ? '—' : Math.max(0, net)}</span></div>`;

    if (rep && rep.identifiable && rep.direction) {
      const d = rep.direction.map((v) => v.toFixed(3)).join(', ');
      html += `<div class="row"><span class="k">direction</span><span class="v">${d}</span></div>`
            + `<div class="row"><span class="k">line RMS</span>`
            +   `<span class="v">${rep.rms_px.toFixed(2)} px</span></div>`
            + `<div class="row"><span class="k">views cross at</span>`
            +   `<span class="v">${(Math.asin(Math.min(1, rep.conditioning)) * 180 / Math.PI).toFixed(0)}&deg;</span></div>`;
      if (rep.n_culled) {
        html += `<div class="note warn">${rep.n_culled} line(s) in this group were dropped as `
              + `outliers. They are still drawn — check each is on the edge you meant.</div>`;
      }
    } else if (rep) {
      html += `<div class="note bad">Not identifiable yet. ${escapeHtml(rep.note)}</div>`;
    } else {
      html += `<div class="note">${escapeHtml(groupAdvice(t))}. No solve yet, so no direction.</div>`;
    }
    if (rep && rep.identifiable) {
      html += `<div class="note">${escapeHtml(rep.note)}</div>`;
    }
    html += `</div>`;
  }

  html += `<button class="smallbtn" id="dgnew" style="margin-top:7px">+ new direction group</button>`;
  el.innerHTML = html;

  const mk = $('dgnew');
  if (mk) mk.onclick = () => newDirGroup();
  for (const b of el.querySelectorAll('.dgdraw')) {
    b.onclick = () => setDirGroup(b.dataset.gid);
  }
  for (const b of el.querySelectorAll('.dgdel')) {
    b.onclick = () => deleteDirGroup(b.dataset.gid);
  }
  for (const inp of el.querySelectorAll('.dglabel')) {
    inp.onchange = () => {
      const g = dirGroupOf(inp.dataset.gid);
      if (!g) return;
      g.label = inp.value.trim() || g.id;
      markDirty();
      syncAxisButtons();
      renderPanels();
    };
  }
}

/* ------------------------------------------------------- focal from EXIF */

/* 35mm equivalence is a diagonal field of view, so it converts through the
 * diagonal. Dividing by the 36 mm film width instead only agrees when the image
 * is 3:2 like the film frame -- a 4:3 photo reads 3.8% low and a 16:9 frame
 * 4.8% high. Mirrors focal_px_from_35mm / focal_px_to_35mm in exif.py. */
const FULL_FRAME_DIAGONAL_MM = Math.hypot(36, 24);
const diagPx = (im) => Math.hypot(im.width, im.height);
const pxFrom35mm = (mm, im) => mm * diagPx(im) / FULL_FRAME_DIAGONAL_MM;
const pxTo35mm = (px, im) => px * FULL_FRAME_DIAGONAL_MM / diagPx(im);

let exifCache = null;
let exifKey = null;
let exifTimer = null;

/** Applies to the whole camera group, since that is what shares a focal length. */
function applyFocal(px, why) {
  const im = currentImage();
  if (!im) return;
  const g = groupOf(im);
  g.f = px;
  g.lock_focal = true;
  markDirty();
  renderPanels();
  render();
  setStatus(`focal locked to ${px.toFixed(0)} px (${why}) for group ${im.group}`, 'ok');
}

function scheduleExif() {
  if (!state.project) return;
  // EXIF only changes when the photo set or the grouping does
  const key = JSON.stringify(state.project.images.map((i) => [i.path, i.group, i.width]));
  if (key === exifKey) { renderExifRow(); return; }
  exifKey = key;
  clearTimeout(exifTimer);
  exifTimer = setTimeout(async () => {
    try {
      exifCache = (await api('/api/exif', { project: projectPayload() })).groups || {};
    } catch (err) {
      exifCache = null;
    }
    renderExifRow();
  }, 250);
}

function renderExifRow() {
  const el = $('exifrow');
  const im = currentImage();
  if (!el || !im) return;
  if (exifCache === null) { el.className = 'msg'; el.textContent = 'EXIF: unavailable'; return; }
  const g = exifCache[im.group];
  if (!g) { el.className = 'msg'; el.textContent = 'EXIF: checking…'; return; }

  if (!g.focal_px) {
    el.className = 'msg';
    el.textContent = `EXIF: ${g.note || 'no focal length recorded'}`;
    return;
  }
  const mm = pxTo35mm(g.focal_px, im);
  const trusted = g.confidence !== 'none';
  el.className = 'msg';
  el.innerHTML =
    `EXIF: <b class="v-${g.confidence === 'good' ? 'good' : trusted ? 'weak' : 'poor'}">`
    + `${g.focal_px.toFixed(0)} px</b> <span class="est">&asymp;${mm.toFixed(0)} mm eq</span>`
    + ` · ${escapeHtml(g.source)} [${g.confidence}]`
    + (trusted ? `<button class="smallbtn" id="exifuse">use</button>` : '')
    + (g.note ? `<div class="${trusted ? 'msg' : 'note bad'}">${escapeHtml(g.note)}</div>` : '');
  const btn = $('exifuse');
  if (btn) btn.onclick = () => applyFocal(g.focal_px, `EXIF, ${g.source}`);
}

/* --------------------------------------------------- focal observability
 *
 * Answers "can these control lines determine a focal length at all?" while the
 * user is still drawing, instead of after a full markup and a solve. It needs
 * only four lines across two axes.
 */

let obsTimer = null;
let obsKey = null;

function scheduleObservability() {
  const im = currentImage();
  const key = im ? im.id + '|' + JSON.stringify(im.axis_lines) : null;
  if (key === obsKey) return;          // nothing about the lines changed
  obsKey = key;
  clearTimeout(obsTimer);
  obsTimer = setTimeout(refreshObservability, 300);
}

async function refreshObservability() {
  const im = currentImage();
  const panel = $('obspanel');
  const badge = $('obsverdict');
  if (!im) { panel.innerHTML = ''; badge.textContent = ''; badge.className = 'count'; return; }
  try {
    renderObservability(await api('/api/observability', {
      width: im.width,
      height: im.height,
      axis_lines: im.axis_lines,
      sigma_px: (state.project.settings && state.project.settings.sigma_line_px) || 2.0,
    }));
  } catch (err) {
    panel.innerHTML = `<div class="note bad">${escapeHtml(err.message)}</div>`;
  }
}

function renderObservability(d) {
  const panel = $('obspanel');
  const badge = $('obsverdict');
  const best = d.best;

  badge.textContent = best ? best.verdict : '—';
  badge.className = 'count ' + (best ? 'v-' + best.verdict : '');

  let html = '';
  if (best && best.relative_sigma != null) {
    const pct = best.relative_sigma * 100;
    // full bar at 0%, empty by 20% -- the range where the answer stops meaning anything
    const fill = Math.max(2, Math.min(100, 100 - pct * 5));
    html += `<div class="obs-head">${best.focal_px.toFixed(0)} px `
          + `<span class="v-${best.verdict}">&plusmn;${pct.toFixed(pct < 10 ? 1 : 0)}%</span></div>`
          + `<div class="obs-bar"><span class="${best.verdict}" style="width:${fill}%"></span></div>`
          + `<div class="msg">best pair ${best.name}`
          + (best.limiting ? `, limited by ${best.limiting}` : '') + `</div>`;
  } else {
    html += `<div class="msg">Focal length not determined yet.</div>`;
  }

  const axes = Object.values(d.axes || {});
  if (axes.length) {
    html += '<table class="obs"><tr><th>axis</th><th>lines</th><th>length</th>'
          + '<th>VP out</th><th>lever</th></tr>';
    for (const a of axes) {
      const lever = a.lever == null ? null : a.lever;
      const cls = lever == null ? '' : lever > 60 ? 'hi' : lever > 20 ? 'mid' : '';
      const vp = a.at_infinity ? '&infin;'
        : a.distance_px > 99999 ? `${(a.distance_px / 1000).toFixed(0)}k` : a.distance_px.toFixed(0);
      html += `<tr><td>${a.name}</td><td>${a.lines}</td>`
            + `<td>${a.median_length_px.toFixed(0)}px</td>`
            + `<td>${vp}</td>`
            + `<td class="lever ${cls}">${lever == null ? '—' : lever.toFixed(0) + '&times;'}</td></tr>`;
    }
    html += '</table>';
  }

  for (const p of d.pairs || []) {
    if (p.relative_sigma == null) html += `<div class="note">${p.name}: ${escapeHtml(p.note)}</div>`;
  }
  for (const pr of d.problems || []) html += `<div class="msg">${escapeHtml(pr)}</div>`;
  if (d.advice) html += `<div class="obs-advice">${escapeHtml(d.advice)}</div>`;

  panel.innerHTML = html;
}

/* ------------------------------------------------------------ nominals
 *
 * A nominal is what the drawing says, as opposed to what the photographs say.
 * Give three or more points their drawing coordinates and the reconstruction
 * stops floating in an arbitrary frame and lands in the part's own -- which is
 * the only frame in which "does the CAD model fit?" is a question with an
 * answer.
 *
 * Typed in the project's nominal units, not metres, because that is what is
 * written on the drawing. */

const nominalUnits = () => (state.project.nominal_units || 'mm');
const nominalScale = () => (state.project.nominal_scale || UNIT_SCALES[nominalUnits()] || 0.001);

const seenIn = (pid) => state.project.images.filter((i) => pid in i.observations).length;

function hasNominal(p) {
  return Array.isArray(p.nominal) && p.nominal.length === 3
      && p.nominal.some((v) => v !== null && v !== undefined);
}

/* How far this point ended up from where the drawing put it, in nominal units.
 * Read out of the constraint report rather than recomputed, so the number on
 * screen is the one the solver actually answered. */
function nominalMiss(pid) {
  const rep = state.report && state.report.constraints;
  if (!rep) return null;
  const rows = rep.filter((c) => c.source === 'nominal' && c.applied && c.points[0] === pid);
  if (rows.length !== 3) return null;
  const d = rows.map((c) => c.error / nominalScale());
  return { d, mag: Math.hypot(d[0], d[1], d[2]), sigmas: Math.max(...rows.map((c) => c.sigmas_off)) };
}

/* Count of nominal'd points, and whether they actually pin a frame. Mirrors
 * Project.in_part_space() -- three points in a straight line are three points
 * and still leave a rotation free. */
function partSpaceState() {
  const nom = state.project.points.filter(hasNominal);
  if (nom.length < 3) return { n: nom.length, ok: false };
  const P = nom.map((p) => p.nominal.map(Number));
  const c = [0, 1, 2].map((k) => P.reduce((s, q) => s + q[k], 0) / P.length);
  // rank-2 test by the largest cross product of centred positions
  const V = P.map((q) => [q[0] - c[0], q[1] - c[1], q[2] - c[2]]);
  let best = 0;
  for (let i = 0; i < V.length; i++) {
    for (let j = i + 1; j < V.length; j++) {
      const a = V[i], b = V[j];
      best = Math.max(best, Math.hypot(
        a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]));
    }
  }
  return { n: nom.length, ok: best > 1e-9 };
}

/* ---------------------------------------------------------- folding
 *
 * Remembered rather than reset each load: a project with four models is one
 * very tall sidebar, and re-collapsing the three you are not working on every
 * time you reopen it is the kind of small tax that makes a tool tiring. Keyed
 * by project name so two projects do not share one layout. */

const FOLD_KEY = () => `multispy.folded.${state.project.name || 'untitled'}`;
let foldedSet = null;

function foldState() {
  if (foldedSet === null) {
    try {
      foldedSet = new Set(JSON.parse(localStorage.getItem(FOLD_KEY()) || '[]'));
    } catch (e) {
      foldedSet = new Set();   // private window, blocked storage: fold nothing
    }
  }
  return foldedSet;
}

const isFolded = (key) => foldState().has(key);

function setFolded(key, folded) {
  const s = foldState();
  if (folded) s.add(key); else s.delete(key);
  try {
    localStorage.setItem(FOLD_KEY(), JSON.stringify([...s]));
  } catch (e) { /* nothing to do; the fold still applies for this session */ }
}

function toggleFold(key) {
  setFolded(key, !isFolded(key));
}

/* Section headers are in the static markup, so they are wired once and their
 * state re-applied rather than re-rendered. */
function initSectionFolding() {
  for (const sec of document.querySelectorAll('#panel .section')) {
    const h = sec.querySelector('h3');
    if (!h) continue;
    const key = 'sec:' + (sec.querySelector('.inner')?.id || h.textContent.trim());
    sec.classList.toggle('folded', isFolded(key));
    h.addEventListener('click', () => {
      toggleFold(key);
      sec.classList.toggle('folded', isFolded(key));
    });
  }
}

/* The origin's confidence, in the panel where the nominals that produce it
 * live. Deliberately verbose about where the number comes from: a coloured dot
 * invites being read as a verdict on the whole solve, when it is one quantity
 * -- how far the origin wanders under click noise -- and blind to every
 * systematic error in the project. */
function originPanelHtml() {
  const sph = originSphere();
  const st = state.project.settings || {};
  const u = nominalUnits();
  const samples = state.originSamples || 32;

  let head;
  if (!sph) {
    head = `<div class="note">The origin is extrapolated from the nominal'd points, so its `
         + `error is not the reprojection RMS. Measure it to find out how far it moves when `
         + `the marks are jittered.</div>`;
  } else {
    const col = GRADE_COLOR[sph.grade] || GRADE_COLOR.unknown;
    const aniso = sph.anisotropy >= 2
      ? ` The scatter is ${sph.anisotropy.toFixed(1)}:1, so the sphere is a loose summary of `
        + `an ellipsoid ${sph.sigma.map((v) => fmtRadius(v)).join(' / ')} ${sph.units} across.`
      : '';
    head = `<div class="row" style="align-items:center;gap:6px">`
         +   `<span style="width:10px;height:10px;border-radius:50%;background:${col};`
         +     `display:inline-block;flex:none"></span>`
         +   `<b>origin ±${fmtRadius(sph.radius)} ${escapeHtml(sph.units)}</b>`
         +   `<span style="color:var(--text-dim)">95%, ${sph.n} samples</span>`
         + `</div>`
         + `<div class="note">`
         +   (sph.stale
               ? 'Marks or nominals have changed since this was measured — it describes the old data. '
               : '')
         +   `Click noise only, at ±${sph.sigma_px || st.sigma_px || 1.5} px: a floor on the `
         +   `uncertainty, not a ceiling. A nominal that is simply wrong does not show up here.`
         +   aniso
         + `</div>`;
  }

  return `<div style="margin-bottom:7px">${head}`
       + `<div class="row"><span class="k">samples</span>`
       +   `<input type="number" id="originsamples" min="8" max="512" step="8" `
       +   `value="${samples}" style="width:62px" title="Each one is a full re-solve">`
       +   `<button class="smallbtn" id="measureorigin">measure</button></div>`
       + `<div class="row"><span class="k">green ≤</span>`
       +   `<input type="number" id="origingood" step="any" value="${st.origin_good ?? 5}" `
       +   `style="width:62px"><span class="k">amber ≤</span>`
       +   `<input type="number" id="originwarn" step="any" value="${st.origin_warn ?? 20}" `
       +   `style="width:62px"><span class="k">${escapeHtml(u)}</span></div>`
       + `</div>`;
}

/* Whether each nominal is on the feature it names.
 *
 * Kept out of the project file on purpose: it is a reading, not a setting, and
 * it is only true of the marks as they were when it ran. The sphere above can
 * afford to persist because it carries a fingerprint that marks it stale; this
 * one would need the same apparatus to earn the same privilege, and until it
 * has it, showing yesterday's verdict against today's links would be the exact
 * species of quietly-wrong number the check exists to catch. */
const SHAPE_COLOR = {
  consistent: 'good', suspect: 'warn', contradicted: 'bad', untestable: 'unknown',
};

function shapeCheckHtml() {
  const r = state.shapeCheck;
  if (!r) {
    return `<div class="note">Compares the shape the nominal'd points make with the shape `
         + `their coordinates describe — scale-free, and against a re-solve with every `
         + `nominal stripped out, so a wrong link cannot pass by having bent the solve to `
         + `meet itself. Needs three nominal'd points.</div>`;
  }
  const col = GRADE_COLOR[SHAPE_COLOR[r.verdict] || 'unknown'] || GRADE_COLOR.unknown;
  let html = `<div class="row" style="align-items:center;gap:6px">`
           +   `<span style="width:10px;height:10px;border-radius:50%;background:${col};`
           +     `display:inline-block;flex:none"></span>`
           +   `<b>${escapeHtml(r.verdict)}</b>`;
  if (r.ok) {
    html += `<span style="color:var(--text-dim)">${r.n_compared} point(s), `
          + `spread ${r.spread.toFixed(2)}×</span>`;
  }
  html += `</div>`;
  if (r.ok) {
    html += `<div class="note">Shape fit leaves ${r.rms.toFixed(1)} ${escapeHtml(r.units)} RMS, `
          + `worst ${r.worst_sigmas.toFixed(1)}× the tolerance in force. These photographs `
          + `resolve about ${(r.precision || 0).toFixed(1)} ${escapeHtml(r.units)}, so an RMS `
          + `near that is as close as the camera can testify. The nominal-free re-solve ran `
          + `at ${r.reproj_px.toFixed(2)} px — read that first.</div>`;
    if (r.suspect) {
      html += `<div class="note bad">Leaving <b>${escapeHtml(r.suspect)}</b> out makes the rest `
            + `consistent (${r.suspect_rms.toFixed(1)} ${escapeHtml(r.units)} RMS against `
            + `${r.rms.toFixed(1)}). Look there first, then confirm it against the photographs.`
            + `</div>`;
    }
    const rows = (r.points || []).filter((q) => q.positioned)
      .sort((a, b) => b.sigmas - a.sigmas);
    if (rows.length) {
      html += `<table class="obs"><tr><th>point</th><th>from</th><th>off</th><th>×tol</th></tr>`
            + ``
            + rows.map((q) =>
                `<tr><td>${escapeHtml(q.id)}</td>`
              + `<td>${escapeHtml(q.source || 'typed')}</td>`
              + `<td>${q.deviation.toFixed(1)}</td>`
              + `<td>${q.sigmas > 999 ? '&gt;999' : q.sigmas.toFixed(1)}`
              + `${q.floor_governs ? ' <span style="color:var(--text-dim)">(res)</span>'
                   : q.sigma_defaulted ? ' <span style="color:var(--text-dim)">(default)</span>' : ''}`
              + `</td></tr>`).join('')
            + `</table>`;
    }
  } else if (r.reason) {
    html += `<div class="note">${escapeHtml(r.reason)}</div>`;
  }
  for (const w of (r.warnings || [])) html += `<div class="note bad">${escapeHtml(w)}</div>`;
  return html;
}

/* Did the CAD bodies make the photographs fit worse?
 *
 * The one panel here that answers a question about something that already
 * happened rather than about the current state: it re-solves without the body
 * anchors and compares, per image, on tie points only. See multispy/cadguard.py
 * for why the comparison cannot be read off the per-image RMS already in the
 * report -- that figure has the anchor marks folded into it and so answers a
 * different question.
 *
 * Costs two full solves, so unlike the other readings it is never run on a
 * render. Cleared by any edit, same as the viewpoint reading: a verdict about
 * marks that have since changed is a verdict about nothing. */
const CAD_COLOR = {
  improves: 'good', harmless: 'good', worsens: 'bad', untestable: 'unknown',
};

/* The switch, above the verdict rather than below it.
 *
 * Above, because this is the control the user comes to this panel for most
 * often and it needs no verdict to justify it: "the fit was good before the
 * models went in" is a thing you notice, not a thing you measure, and on a cell
 * with twenty-eight bodies and five hundred marks, working out which three are
 * lying is a separate afternoon. The check is still here and still worth
 * running; it is no longer the only way through this panel.
 *
 * It re-solves on its own rather than leaving the project dirty, because the
 * state it names and the solution on screen have to agree -- the whole failure
 * this feature exists for is a camera solution that the bodies dictated sitting
 * under a report that does not mention them. */
function cadMuteSwitchHtml() {
  const s = state.project.settings || {};
  const off = s.anchors_in_solve === false;
  const bodies = instanceList().filter((o) => !o.is_part);
  const perBody = bodies.filter((o) => o.anchors_off).length;
  const marks = bodies.reduce((n, o) => n + anchorMarkCountFor(o.id), 0);
  const rep = state.report || {};

  let html = `<label class="toggle" title="Solve the cameras from tie points, control lines and `
           + `nominals alone. Every anchor stays marked; the bodies are placed against the `
           + `result instead of moving it."><input type="checkbox" id="cadmuteall"`
           + `${off ? ' checked' : ''}> keep every body's anchors out of the solve</label>`;
  if (off) {
    html += `<div class="note">The cameras, the points and the focal are what the tie points, `
          + `control lines and nominals make them. ${marks} anchor mark(s) are still on the `
          + `record and still drawn &mdash; they are simply not evidence about where the `
          + `cameras are, and this holds across every Solve until you untick it.`
          + `${rep.n_muted_instances ? ` Last solve placed `
            + `${Object.values(rep.instances || {}).filter((v) => v.placed).length} of `
            + `${rep.n_muted_instances} of them against the finished cameras.` : ''}</div>`
          + `<div class="note">What this buys and what it does not: on the rear-module cell it `
          + `took the fit from 9.37 px to 2.99 px and the culled marks from 15 to 2. It did `
          + `<b>not</b> improve the bodies' agreement with CAD &mdash; shape error 3.49 mm to `
          + `3.74 mm on one yardstick for both. It buys back the camera solution, not accuracy.`
          + `</div>`;
  } else if (perBody) {
    html += `<div class="note">${perBody} body/bodies have their anchors out of the solve `
          + `individually &mdash; see the objects panel. The rest are voting.</div>`;
  }
  return html;
}

function cadPanelHtml() {
  const r = state.cadCheck;
  if (!r) {
    return `<div class="note">Known objects are a much stronger claim than tie points &mdash; `
         + `a body says these features are in exactly this rigid arrangement, because a CAD `
         + `file says so. When that is right it is worth more than any number of marks. When `
         + `it is wrong it is a confident lie, and the bundle will bend the cameras to fit it. `
         + `This solves with the bodies and without them and compares each photograph's tie `
         + `points, so a body that cost you calibration says so instead of hiding.</div>`;
  }
  const col = GRADE_COLOR[CAD_COLOR[r.verdict] || 'unknown'] || GRADE_COLOR.unknown;
  let html = `<div class="row" style="align-items:center;gap:6px">`
           +   `<span style="width:10px;height:10px;border-radius:50%;background:${col};`
           +     `display:inline-block;flex:none"></span><b>${escapeHtml(r.verdict)}</b>`;
  if (r.ok) {
    html += `<span style="color:var(--text-dim)">${r.before_rms_px.toFixed(2)} &rarr; `
          + `${r.after_rms_px.toFixed(2)} px on tie points</span>`;
  }
  html += `</div>`;

  if (!r.ok) {
    html += `<div class="note">${escapeHtml(r.reason || 'nothing to compare')}</div>`;
    return html;
  }

  if (r.focal_before_px && r.focal_after_px) {
    const pct = 100 * Math.abs(r.focal_after_px - r.focal_before_px) / r.focal_before_px;
    html += `<div class="row"><span class="k">shared focal</span><span class="v">`
          + `${r.focal_before_px.toFixed(0)} &rarr; ${r.focal_after_px.toFixed(0)} px `
          + `<span style="color:var(--text-dim)">(${pct.toFixed(1)}%)</span></span></div>`;
  }

  const rows = r.images || [];
  if (rows.length) {
    html += `<table class="obs"><tr><th>image</th><th>pts</th><th>without</th><th>with</th>`
          + `<th>change</th><th>worst point</th></tr>`
          + rows.map((v) =>
              `<tr${v.worsened ? ' style="color:var(--bad)"' : ''}>`
            + `<td>${escapeHtml(v.image)}</td>`
            + `<td>${v.n_points}</td>`
            + `<td>${v.before_px.toFixed(2)}</td>`
            + `<td>${v.after_px.toFixed(2)}</td>`
            + `<td>${v.delta_px >= 0 ? '+' : ''}${v.delta_px.toFixed(2)}`
            + `${v.worsened ? ` <b>&gt;${v.allowed_px.toFixed(2)}</b>` : ''}</td>`
            + `<td>${escapeHtml(v.worst_point || '')}</td></tr>`).join('')
          + `</table>`;
    html += `<div class="note"><b>without</b> and <b>with</b> are that photograph's tie-point `
          + `RMS with the body anchors out of the solve and in it &mdash; tie points only, so `
          + `a body's own marks cannot flatter or damn it. A row is flagged when it worsens by `
          + `more than its own allowance, which is the larger of 0.25 px and a tenth of where `
          + `it started.</div>`;
  }

  for (const n of (r.notes || [])) {
    html += `<div class="note${r.verdict === 'worsens' ? ' bad' : ''}">${escapeHtml(n)}</div>`;
  }
  if (r.verdict === 'worsens') {
    html += `<div style="display:flex;gap:5px;margin-top:6px;flex-wrap:wrap">`
          + `<button class="smallbtn" id="cadmute">take their anchors out of the solve</button>`
          + `<button class="smallbtn" id="cadrevert">just put the cameras back</button>`
          + `</div>`
          // The two buttons do different things and the difference is the whole
          // point of this feature, so it is spelled out rather than left to the
          // labels. The first is a decision that survives a reload; the second
          // is a look at the old answer that the next Solve undoes.
          + `<div class="note"><b>Out of the solve</b> is the lasting one: it keeps every click `
          + `and every body, re-solves from tie points, control lines and nominals, and places `
          + `the bodies against that result. Press Solve as often as you like &mdash; it stays. `
          + `<b>Put the cameras back</b> only restores the poses and points from the body-free `
          + `solve and leaves the marks voting, so the next Solve re-makes exactly the damage `
          + `above. Use it to look, not to keep. To mute one body rather than all of them, use `
          + `the tickbox in the objects panel.</div>`;
  }
  return html;
}

function renderCadPanel() {
  const el = $('cadpanel');
  if (!el) return;
  const r = state.cadCheck;
  const badge = $('cadcount');
  if (badge) badge.textContent = r && r.ok ? r.verdict : '';
  el.innerHTML = `<div class="row"><span class="k">bodies against the marks</span>`
               +   `<button class="smallbtn" id="checkcad">check CAD</button></div>`
               + cadMuteSwitchHtml()
               + cadPanelHtml();
  wireCadPanel();
}

/* Set the switch and re-solve, in one act.
 *
 * One act because the two halves are not separable in any useful way: the
 * switch describes which marks are evidence, and leaving the old solution on
 * screen under it would show a camera solution those very marks produced. That
 * mismatch -- a good-looking report over a calibration the bodies bent -- is the
 * failure this whole feature exists to end, so it is not reproduced by the fix.
 *
 * Goes through its own route rather than the solve route because it writes the
 * project: /api/check_cad must stay read-only, and a switch that is set
 * client-side and only saved on the next autosave would be a decision that can
 * be lost. */
async function muteEverything() {
  pushUndoBefore('take the bodies’ anchors out of the solve');
  /* The switch is NOT set here, and that is not an oversight. The route reads
   * the payload to decide whether there is anything to do, so setting it
   * client-side first makes the server see a project that is already muted and
   * answer "nothing changed" -- without re-solving, leaving the old
   * body-dictated solution on screen under a panel saying the bodies are out of
   * it. Exactly the mismatch this feature exists to end, reintroduced by an
   * optimistic update. The server sets it, solves, and sends both back. */
  setStatus('taking the bodies’ anchors out of the solve and re-solving…');
  try {
    const data = await api('/api/mute_bodies', { project: projectPayload() });
    if (data.project) adopt(data);
    setStatus(data.applied || 'done', 'ok');
  } catch (err) {
    setStatus(`could not do it: ${err.message || err}`, 'warn');
  } finally {
    renderPanels();
    render();
    if (Viewer3D.isOn()) Viewer3D.render();
  }
}

function wireCadPanel() {
  const btn = $('checkcad');
  if (btn) {
    btn.onclick = async () => {
      btn.disabled = true;
      setStatus('checking the bodies — two solves, one with them and one without…');
      try {
        const data = await api('/api/check_cad', { project: projectPayload() });
        const r = data.cad;
        state.cadCheck = r;
        setStatus(
          r.ok
            ? `CAD ${r.verdict}: tie points ${r.before_rms_px.toFixed(2)} → `
              + `${r.after_rms_px.toFixed(2)} px, ${(r.worsened || []).length} image(s) worse`
            : `CAD check untestable: ${r.reason}`,
          r.ok && r.verdict === 'worsens' ? 'warn' : 'ok');
      } catch (err) {
        setStatus(`could not check the bodies: ${err.message || err}`, 'warn');
      } finally {
        btn.disabled = false;
        renderPanels();
        render();
      }
    };
  }
  const muteAll = $('cadmuteall');
  if (muteAll) {
    muteAll.onchange = (e) => {
      if (e.target.checked) { muteEverything(); return; }
      // Unmuting is the plain edit: the next Solve lets them back in, and
      // nothing needs re-solving behind the user's back to make that true.
      pushUndoBefore('let the bodies’ anchors back into the solve');
      state.project.settings.anchors_in_solve = true;
      state.cadCheck = null;
      markDirty();
      setStatus('the bodies’ anchors are back in the solve — press Enter to re-solve', 'ok');
      renderPanels();
    };
  }
  const mute = $('cadmute');
  if (mute) mute.onclick = () => muteEverything();
  const revert = $('cadrevert');
  if (revert) {
    revert.onclick = async () => {
      revert.disabled = true;
      pushUndoBefore('put the body-free cameras back');
      setStatus('restoring the solve from before the bodies…');
      try {
        const data = await api('/api/check_cad', { project: projectPayload(), apply: true });
        if (data.project) adopt(data);
        state.cadCheck = data.cad;
        setStatus(data.applied || 'restored', 'ok');
      } catch (err) {
        setStatus(`could not restore: ${err.message || err}`, 'warn');
      } finally {
        revert.disabled = false;
        renderPanels();
        render();
        if (Viewer3D.isOn()) Viewer3D.render();
      }
    };
  }
}

/* Where the information along the origin's worst direction comes from.
 *
 * The companion to the sphere above. That one says how big the uncertainty is;
 * this one says which photographs are holding it up, which photograph has the
 * most left to click, and what viewpoint is missing from the set entirely.
 *
 * A reading, never a setting. Nothing here changes a weight, and the panel says
 * so where the camera aims differ enough for the thought to occur: a camera
 * pointed down the weak direction is already discounted for it by the Jacobian,
 * and it is the best witness in the set for the two directions across it, so
 * down-weighting it by aim would suppress both. See multispy/viewpoint.py.
 *
 * Cleared by markDirty() and adopt(), because every figure is read off the poses
 * and marks as they stood when it ran. An edit makes it a true statement about a
 * project that no longer exists, which is the worst kind. */
const VIEWPOINT_COLOR = { 'viewpoint-limited': 'warn', 'evidence-limited': 'unknown' };

/* A direction as something you can stand in: nearest axis, and how far off it. */
function axisHint(v) {
  if (!v || v.length !== 3) return '';
  let k = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(v[i]) > Math.abs(v[k])) k = i;
  const deg = Math.acos(Math.min(1, Math.abs(v[k]))) * 180 / Math.PI;
  return `${v[k] < 0 ? '-' : '+'}${'XYZ'[k]}, ${deg.toFixed(0)}&deg; off it`;
}

function vec3(v, dp) {
  return `[${v.map((q) => q.toFixed(dp === undefined ? 3 : dp)).join(', ')}]`;
}

function viewpointPanelHtml() {
  const r = state.viewpoints;
  if (!r) {
    return `<div class="note">Which photographs supply the information along the direction `
         + `the origin is least sure of &mdash; and where a camera would supply more. No `
         + `re-solve: it reads the poses as they stand, so it is instant. Measure the origin `
         + `first, in <b>Tie points</b> above; without that it falls back to the direction `
         + `the <i>marks</i> are weakest in, which is a related but different question.</div>`;
  }
  if (!r.ok) {
    return `<div class="note">${escapeHtml(r.reason || 'nothing to report')}</div>`;
  }
  const col = GRADE_COLOR[VIEWPOINT_COLOR[r.verdict] || 'unknown'] || GRADE_COLOR.unknown;
  let html = `<div class="row" style="align-items:center;gap:6px">`
           +   `<span style="width:10px;height:10px;border-radius:50%;background:${col};`
           +     `display:inline-block;flex:none"></span>`
           +   `<b>${escapeHtml(r.verdict)}</b>`
           +   `<span style="color:var(--text-dim)">${r.anisotropy.toFixed(2)}&times; worst `
           +     `to best</span></div>`;

  html += `<div class="row"><span class="k">weak direction</span>`
        +   `<span class="v">${vec3(r.weak)}</span></div>`
        + `<div class="row"><span class="k"></span>`
        +   `<span class="v" style="color:var(--text-dim)">${axisHint(r.weak)}</span></div>`;
  if (r.weak_sigma > 0) {
    html += `<div class="row"><span class="k">its sigma</span>`
          +   `<span class="v">${r.weak_sigma.toFixed(2)} ${escapeHtml(r.units)}</span></div>`;
  }
  if (r.source === 'geometry') {
    html += `<div class="note">Read from the marks, not from a measured origin. Measure the `
          + `origin and look again.</div>`;
  }
  if (r.stale) {
    html += `<div class="note bad">The stored origin measurement was taken against different `
          + `marks, lines or nominals, so this direction may have moved. Re-measure before `
          + `standing anywhere on it.</div>`;
  }

  const cams = r.cameras || [];
  if (cams.length) {
    html += `<table class="obs"><tr><th>image</th><th>marks</th><th>|cos|</th><th>share</th>`
          + `<th>/mark</th><th>left</th><th>+if&nbsp;marked</th></tr>`
          + cams.map((c) => {
              const hot = c.image === r.best_per_mark;
              const left = c.image === r.most_left_to_click;
              return `<tr><td>${escapeHtml(c.image)}`
                   + `${hot ? ' <span style="color:var(--text-dim)">(best/mark)</span>' : ''}`
                   + `</td>`
                   + `<td>${c.marks}</td>`
                   + `<td>${c.cos_weak.toFixed(2)}</td>`
                   + `<td>${(100 * c.share).toFixed(1)}%</td>`
                   + `<td>${(100 * c.share_per_mark).toFixed(2)}%</td>`
                   + `<td>${c.unmarked_in_frame || ''}</td>`
                   + `<td${left ? ' style="font-weight:600"' : ''}>`
                   + `${c.gain_if_marked ? (100 * c.gain_if_marked).toFixed(1) + '%' : ''}`
                   + `</td></tr>`;
            }).join('')
          + `</table>`;
    html += `<div class="note"><b>|cos|</b> is how squarely each camera looks down the weak `
          + `direction &mdash; near 1 and it can say almost nothing about that direction, `
          + `near 0 and it is looking straight across it. <b>share</b> is how much of the `
          + `information along it that camera actually supplies, and <b>/mark</b> is what one `
          + `more mark there is worth. Distance counts as much as aim: information falls off `
          + `as its square, so a close photo beats a well-aimed far one.</div>`;
    const best = cams.find((c) => c.image === r.most_left_to_click);
    if (best && best.gain_if_marked > 0.02) {
      html += `<div class="note good">Cheapest gain available, and it needs no new photograph: `
            + `<b>${escapeHtml(best.image)}</b> has ${best.unmarked_in_frame} positioned `
            + `point(s) inside the frame that are not marked in it, worth about `
            + `${(100 * best.gain_if_marked).toFixed(0)}% more information along the weak `
            + `direction. Occlusion is not modelled, so some may be behind the part.</div>`;
    }
  }

  const s = r.suggestion;
  if (s) {
    html += `<div class="row" style="margin-top:6px"><span class="k">a missing vantage</span>`
          +   `<span class="v">${s.nearest_existing_deg.toFixed(0)}&deg; from the nearest `
          +     `line of sight you have</span></div>`
          + `<div class="row"><span class="k">face</span>`
          +   `<span class="v">${vec3(s.bearing)} <span style="color:var(--text-dim)">`
          +   `(${axisHint(s.bearing)})</span></span></div>`
          + `<div class="row"><span class="k">stand at</span>`
          +   `<span class="v">${vec3(s.position, 0)} ${escapeHtml(r.units)}</span></div>`
          + `<div class="row"><span class="k">or opposite</span>`
          +   `<span class="v" style="color:var(--text-dim)">${vec3(s.position_opposite, 0)}`
          +   `</span></div>`;
    let worth = `about ${(100 * s.gain).toFixed(0)}% more information along the weak `
              + `direction, assuming it marks ${s.marks_assumed} points`;
    if (r.weak_sigma > 0 && s.sigma_factor > 0) {
      worth += ` &mdash; which is ${r.weak_sigma.toFixed(1)} to `
             + `${(r.weak_sigma * s.sigma_factor).toFixed(1)} ${escapeHtml(r.units)} along that `
             + `one direction, indicative only`;
    } else {
      // Information is a ratio and says nothing about millimetres on its own.
      // Without a measured origin there is nothing to scale, so the percentage
      // has to be left as a percentage rather than dressed up as an improvement.
      worth += `, which cannot be put in ${escapeHtml(r.units)} until the origin has been `
             + `measured &mdash; a ratio of information is not a distance`;
    }
    html += `<div class="note">Both positions see the same slice: a camera and its opposite `
          + `pin a point identically, so take whichever is not a wall. Worth ${worth}. The `
          + `standoff is the closest one already in the set, because a distance one `
          + `photograph was taken from is a distance that can be stood at.</div>`;
    // A large gain and "no viewpoint will help much" are both true at once, and
    // read together they look like a contradiction. They are answers about
    // different things: the gain is information along one direction, and the
    // verdict is about the shape of the whole ellipsoid, which a single
    // direction barely moves when the three are already close to equal. Said
    // here because the percentage is the number someone would act on.
    if (r.verdict === 'evidence-limited' && s.gain > 0.25) {
      html += `<div class="note">That percentage is larger than it sounds. It is information `
            + `along one direction, and this error is nearly spherical &mdash; sharpening the `
            + `weakest of three near-equal axes moves the containment radius very little. `
            + `Worth taking if you are standing there anyway; not worth a trip.</div>`;
    }
  }

  for (const n of (r.notes || [])) html += `<div class="note">${escapeHtml(n)}</div>`;
  return html;
}

function renderViewpointPanel() {
  const el = $('vppanel');
  if (!el) return;
  const r = state.viewpoints;
  const badge = $('vpcount');
  if (badge) badge.textContent = r && r.ok ? r.verdict : '';
  el.innerHTML = `<div class="row"><span class="k">information along the weak axis</span>`
               +   `<button class="smallbtn" id="checkviewpoints">read viewpoints</button>`
               + `</div>`
               + viewpointPanelHtml();
  // Wired here and not from the init pass: this panel rebuilds its own button on
  // every render, so a handler attached once would be thrown away on the first
  // redraw after it.
  wireViewpointPanel();
}

function wireViewpointPanel() {
  const btn = $('checkviewpoints');
  if (!btn) return;
  btn.onclick = async () => {
    btn.disabled = true;
    setStatus('apportioning the information along the weak direction…');
    try {
      const data = await api('/api/viewpoints', { project: projectPayload() });
      const r = data.viewpoints;
      state.viewpoints = r;
      setStatus(
        r.ok
          ? `${r.verdict}: ${r.anisotropy.toFixed(2)}x worst to best, best per mark `
            + `${r.best_per_mark}, most left to click ${r.most_left_to_click}`
          : `viewpoints: ${r.reason}`,
        r.ok && r.verdict === 'viewpoint-limited' ? 'warn' : 'ok');
    } catch (err) {
      setStatus(`could not read the viewpoints: ${err.message || err}`, 'warn');
    } finally {
      btn.disabled = false;
      renderPanels();
      render();
    }
  };
}

function wireShapeCheck() {
  const shape = $('checkshape');
  if (!shape) return;
  shape.onclick = async () => {
    shape.disabled = true;
    setStatus('checking the nominal links — one re-solve without them…');
    try {
      const data = await api('/api/check_nominal_shape', { project: projectPayload() });
      const r = data.shape;
      state.shapeCheck = r;
      setStatus(
        r.ok
          ? `nominal links ${r.verdict}: ${r.n_compared} point(s), spread `
            + `${r.spread.toFixed(2)}×, worst ${r.worst_sigmas.toFixed(1)}× tolerance`
          : `nominal links untestable: ${r.reason}`,
        r.verdict === 'consistent' ? 'ok' : 'warn');
    } catch (err) {
      setStatus(`could not check the nominal links: ${err.message || err}`, 'warn');
    } finally {
      shape.disabled = false;
      renderPanels();
      render();
    }
  };
}

function wireOriginPanel() {
  const btn = $('measureorigin');
  if (!btn) return;
  const n = $('originsamples');
  n.onchange = () => { state.originSamples = Math.max(8, Math.min(512, parseInt(n.value, 10) || 32)); };
  for (const [id, key] of [['origingood', 'origin_good'], ['originwarn', 'origin_warn']]) {
    const el = $(id);
    if (!el) continue;
    el.onchange = () => {
      const v = parseFloat(el.value);
      if (!Number.isFinite(v) || v < 0) return;
      state.project.settings[key] = v;
      markDirty();
      renderPanels();
      render();
      if (Viewer3D.isOn()) Viewer3D.render();
    };
  }
  btn.onclick = async () => {
    const count = Math.max(8, Math.min(512, parseInt(n.value, 10) || 32));
    state.originSamples = count;
    btn.disabled = true;
    setStatus(`measuring the origin — ${count} re-solves…`);
    try {
      const data = await api('/api/origin_uncertainty', { project: projectPayload(), n: count });
      adopt(data);
      const got = originSphere();
      setStatus(got
        ? `origin ±${fmtRadius(got.radius)} ${got.units} (95%, ${got.n} samples)`
        : 'the origin could not be measured', got ? 'ok' : 'warn');
    } catch (err) {
      setStatus(`could not measure the origin: ${err.message || err}`, 'warn');
    } finally {
      btn.disabled = false;
      renderPanels();
      render();
      if (Viewer3D.isOn()) Viewer3D.render();
    }
  };
}

/* Nominals the calibration file already states, as a list to pick from.
 *
 * Forty-two numbers typed by hand is forty-two chances to transpose a digit, and
 * a transposed digit in a nominal does not read as a mistake -- it reads as a
 * part that is out of tolerance, which is the most expensive kind of wrong this
 * tool can produce. Picking a named feature instead carries its coordinate, its
 * own tolerance from the file's spec limits, and a record of where it came from.
 *
 * Withheld candidates are shown, disabled, with their reason. A feature whose
 * sensor was convicted is not a feature that does not exist, and hiding it would
 * only invite someone to type the number in by hand. */
/* The tolerance field above holds one number, and for a single-axis metrology
 * feature that is not what is in force. Saying so here is not decoration: a
 * field reading "2" while two of the three axes are held at 20 is a tolerance
 * that silently does not apply, which is the defect this whole per-axis form
 * exists to fix. Read-only, because editing it by hand is the scalar field's
 * job and typing there clears this. */
function perAxisTolHtml(ap, u) {
  const axes = ap.nominal_sigma_axes;
  if (!Array.isArray(axes) || axes.every((v) => v === null || v === undefined)) return '';
  const fallback = ap.nominal_sigma ?? (state.project.settings.sigma_nominal ?? 0.1);
  const cells = ['X', 'Y', 'Z'].map((ax, i) => {
    const v = axes[i] === null || axes[i] === undefined ? fallback : axes[i];
    const loose = axes[i] !== null && axes[i] !== undefined
                && Number(axes[i]) > Number(fallback) * 1.000001;
    return `${ax} ±${Number(v).toPrecision(3).replace(/\.?0+$/, '')}${loose ? ' (uninspected)' : ''}`;
  });
  return `<div class="msg">per axis, from the calibration file: `
       + `${escapeHtml(cells.join('  ·  '))} ${escapeHtml(u)}. The field above `
       + `applies only where this states nothing; typing in it replaces this.</div>`;
}

function calibPickerHtml(ap) {
  const all = state.project.calibration_nominals || [];
  if (!all.length) {
    return `<div class="note">No calibration file imported. One would offer its features `
         + `here as nominals to pick from, with their own tolerances, instead of numbers to `
         + `type. <button class="smallbtn" id="calibimport">import calibration…</button></div>`;
  }
  const usable = all.filter((c) => c.usable);
  const src = ap.nominal_source || '';
  const opts = [`<option value="">— type by hand —</option>`]
    .concat(all.map((c) => {
      const label = `${c.feature}  (${c.nominal.map((v) => Number(v).toFixed(1)).join(', ')})`;
      return `<option value="${escapeHtml(c.feature)}"${c.feature === src ? ' selected' : ''}`
           + `${c.usable ? '' : ' disabled'}>${escapeHtml(label)}`
           + `${c.usable ? '' : ' — withheld'}</option>`;
    }));
  const chosen = all.find((c) => c.feature === src);
  return `<div class="row" style="margin-top:4px"><span class="k">from file</span>`
       +   `<select id="calibpick" style="flex:1;min-width:0">${opts.join('')}</select></div>`
       + (chosen
            ? `<div class="msg">${escapeHtml(chosen.note)}</div>`
            : `<div class="note">${usable.length} of ${all.length} features offered; the rest `
              + `are withheld because the sensor that measures them disagrees with the nominal, `
              + `which says one of the two is wrong and not which. `
              + `<button class="smallbtn" id="calibimport">re-import…</button></div>`);
}

function wireCalibPicker() {
  const open = $('calibimport');
  if (open) open.onclick = () => $('calibinput').click();

  const pick = $('calibpick');
  if (!pick) return;
  pick.onchange = () => {
    const pt = state.project.points.find((p) => p.id === state.activePoint);
    if (!pt) return;
    const chosen = (state.project.calibration_nominals || [])
      .find((c) => c.feature === pick.value);
    if (!chosen) {
      // Back to typing by hand clears the provenance but keeps the numbers: the
      // coordinate may well be right, and silently blanking a nominal someone
      // has already solved against would be the surprising half of "undo".
      pt.nominal_source = '';
    } else {
      pt.nominal = chosen.nominal.map(Number);
      pt.nominal_source = chosen.feature;
      // Clear the tolerance when the file states no spec limit for this
      // feature, instead of leaving the previous pick's number sitting there.
      // A tolerance carried over from a different feature would be stamped
      // with this one's provenance and weighted as though the file had
      // vouched for it -- the one kind of wrong number that looks sourced.
      // Null falls back to Settings.sigma_nominal, which the tolerance field
      // shows as its placeholder, so what is actually being used stays on
      // screen.
      pt.nominal_sigma =
        chosen.sigma === null || chosen.sigma === undefined
          ? null
          : Number(chosen.sigma);
      // And the per-axis form beside it, which is the only thing that can say
      // "the cell inspects this on Y". Each entry is null where the file states
      // nothing for that axis, which falls back to the scalar and then to
      // Settings.sigma_nominal -- so a null here is not a missing tolerance,
      // it is the same tolerance the scalar already carries.
      pt.nominal_sigma_axes = Array.isArray(chosen.sigma_axes)
        ? chosen.sigma_axes.map((v) => (v === null || v === undefined ? null : Number(v)))
        : null;
    }
    markDirty();
    renderPanels();
    render();
    if (Viewer3D.isOn()) Viewer3D.render();
  };
}

async function importCalibration(file) {
  setStatus(`reading ${file.name}…`);
  try {
    const text = await file.text();
    const data = await api('/api/use_calibration_nominals',
                           { project: projectPayload(), text });
    adopt(data);
    for (const w of (data.warnings || [])) setStatus(w, 'warn');
    setStatus(
      `${data.n_usable} of ${data.n_candidates} features offered as nominals`,
      data.n_usable ? 'ok' : 'warn');
  } catch (err) {
    setStatus(`could not read the calibration file: ${err.message || err}`, 'warn');
  }
  renderPanels();
  render();
}

function renderPointsPanel() {
  const im = currentImage();
  const el = $('ptpanel');
  const pts = state.project.points;
  $('ptcount').textContent = `${pts.length}`;

  const ps = partSpaceState();
  let html = '';
  if (ps.n) {
    html += ps.ok
      ? `<div class="note good">In part space — ${ps.n} nominal'd point(s) fix the origin, `
        + `orientation and scale. A CAD model of the part belongs at the identity.</div>`
      : `<div class="note">${ps.n} nominal'd point(s). Three or more, not in a straight line, `
        + `put the solve into the part's own frame.</div>`;
    html += `<div class="row"><span class="k">nominal links</span>`
          +   `<button class="smallbtn" id="checkshape">check shape</button></div>`
          + shapeCheckHtml();
  }
  if (ps.ok) html += originPanelHtml();
  html += `<div style="display:flex;gap:5px;margin-bottom:6px">`
           + `<button class="smallbtn" id="newpt">new point (N)</button>`
           + `<button class="smallbtn" id="nextmissing">next missing (Tab)</button></div>`
           + `<label class="toggle" style="margin-bottom:6px"><input type="checkbox" id="autoadv"`
           + `${state.autoAdvance ? ' checked' : ''}> auto-advance after placing</label>`;

  if (!pts.length) {
    html += `<div class="empty">No tie points. Switch to Points mode and click a feature you can also find in another photo.</div>`;
  } else {
    html += '<div class="ptlist">';
    for (const p of pts) {
      const seen = state.project.images.filter((i) => p.id in i.observations).length;
      const rep = state.report && state.report.points ? state.report.points[p.id] : null;
      const here = im && p.id in im.observations;
      const err = rep && rep.rms_px ? `${rep.rms_px.toFixed(1)}px` : '';
      const warn = seen < 2 ? ' hi' : (rep && rep.rms_px > 4 ? ' hi' : '');
      const miss = nominalMiss(p.id);
      const badges = (hasNominal(p) ? `<span class="tag nom" title="has a nominal XYZ">nom</span>` : '')
                   + (p.uncertain ? `<span class="tag unc" title="flagged uncertain">?</span>` : '');
      // once in part space the deviation from nominal is the interesting
      // number, not the reprojection error -- it is the one in millimetres
      const right = miss ? `${miss.mag.toFixed(2)}${nominalUnits()}`
                         : (err || (seen < 2 ? 'needs 2' : ''));
      const rwarn = miss ? (miss.sigmas > 3 ? ' hi' : '') : warn;
      html += `<div class="pt${p.id === state.activePoint ? ' sel' : ''}${here ? ' here' : ''}" data-pt="${p.id}">`
            + `<span class="swatch"></span>`
            + `<span class="name">${escapeHtml(p.label || p.id)} ${badges}`
            +   `<span style="color:var(--text-dim)">${seen}img</span></span>`
            + `<span class="err${rwarn}">${right}</span></div>`;
    }
    html += '</div>';
    if (state.activePoint) {
      const ap = pts.find((p) => p.id === state.activePoint);
      if (ap) {
        html += `<div class="row" style="margin-top:7px"><span class="k">label</span>`
              + `<input type="text" id="ptlabel" value="${escapeHtml(ap.label)}" style="width:150px"></div>`;

        html += `<label class="toggle" style="margin-top:5px"><input type="checkbox" id="ptunc"`
              + `${ap.uncertain ? ' checked' : ''}> uncertain — I could not see this feature `
              + `clearly (U)</label>`;

        // ---- nominal XYZ
        const nom = hasNominal(ap) ? ap.nominal : [null, null, null];
        const u = nominalUnits();
        html += `<div class="nomhead">`
              +   `<span class="k">nominal (${escapeHtml(u)})</span>`
              +   `<select id="nomunit" title="Units the nominals are typed in">`
              +     Object.keys(UNIT_SCALES).map((k) =>
                      `<option value="${k}"${k === u ? ' selected' : ''}>${k}</option>`).join('')
              +   `</select>`
              + `</div>`
              + `<div class="nomrow">`
              +   ['X', 'Y', 'Z'].map((ax, i) =>
                    `<label class="nomcell"><span>${ax}</span>`
                  + `<input type="number" step="any" class="nomval" data-axis="${i}" `
                  + `value="${nom[i] === null || nom[i] === undefined ? '' : nom[i]}" `
                  + `placeholder="—"></label>`).join('')
              + `</div>`
              + `<div class="row"><span class="k">tolerance ±</span>`
              +   `<input type="number" step="any" id="nomtol" style="width:72px" `
              +   `value="${ap.nominal_sigma === null || ap.nominal_sigma === undefined
                            ? '' : ap.nominal_sigma}" `
              +   `placeholder="${(state.project.settings.sigma_nominal ?? 0.1)}">`
              +   `<span class="est">${escapeHtml(u)}</span></div>`
              + perAxisTolHtml(ap, u)
              + calibPickerHtml(ap);

        const miss = nominalMiss(ap.id);
        if (miss) {
          const cls = miss.sigmas > 3 ? 'note bad' : 'msg';
          html += `<div class="${cls}">solved ${miss.mag.toFixed(3)} ${escapeHtml(u)} from nominal `
                + `(X ${miss.d[0] >= 0 ? '+' : ''}${miss.d[0].toFixed(3)}, `
                + `Y ${miss.d[1] >= 0 ? '+' : ''}${miss.d[1].toFixed(3)}, `
                + `Z ${miss.d[2] >= 0 ? '+' : ''}${miss.d[2].toFixed(3)}) `
                + `— ${miss.sigmas.toFixed(1)}× the stated tolerance</div>`;
        } else if (hasNominal(ap)) {
          html += `<div class="msg">solve to see how far this landed from its nominal</div>`;
        }
        // Two rays cross in exactly one place, so a two-view point fits its
        // photos perfectly wherever the nominal puts it. The deviation readout
        // above will say 0.000 whatever the drawing claims, which looks like
        // agreement and is actually the absence of a test.
        if (hasNominal(ap) && seenIn(ap.id) < 3) {
          html += `<div class="note">marked in only ${seenIn(ap.id)} photo(s). A point seen `
                + `twice sits wherever its two rays cross and fits both exactly, so this `
                + `nominal steers the solve but nothing can check it. Mark it in a third `
                + `photo.</div>`;
        }

        html += `<div style="display:flex;gap:5px;margin-top:5px">`
              + (hasNominal(ap) ? `<button class="smallbtn" id="clearnom">clear nominal</button>` : '')
              + `<button class="smallbtn" id="delpt">delete point</button></div>`;
      }
    }
  }
  el.innerHTML = html;

  wireOriginPanel();
  wireShapeCheck();
  wireCalibPicker();
  $('newpt').onclick = () => { newPoint(); setMode('points'); renderPanels(); render(); };
  $('nextmissing').onclick = () => { if (state.activePoint) advanceToMissing(state.activePoint); };
  $('autoadv').onchange = (e) => { state.autoAdvance = e.target.checked; };
  for (const row of el.querySelectorAll('.pt')) {
    row.onclick = () => { state.activePoint = row.dataset.pt; setMode('points'); renderPanels(); render(); };
  }
  const lbl = $('ptlabel');
  if (lbl) lbl.onchange = (e) => {
    const ap = pts.find((p) => p.id === state.activePoint);
    if (ap) { ap.label = e.target.value; markDirty(); renderPanels(); render(); }
  };
  const del = $('delpt');
  if (del) del.onclick = () => { deletePoint(state.activePoint); renderPanels(); render(); };

  const unc = $('ptunc');
  if (unc) unc.onchange = (e) => {
    const ap = pts.find((p) => p.id === state.activePoint);
    if (ap) { ap.uncertain = e.target.checked; markDirty(); renderPanels(); render(); }
  };

  for (const inp of el.querySelectorAll('.nomval')) {
    inp.onchange = (e) => {
      const ap = pts.find((p) => p.id === state.activePoint);
      if (!ap) return;
      if (!Array.isArray(ap.nominal) || ap.nominal.length !== 3) ap.nominal = [null, null, null];
      const raw = e.target.value.trim();
      ap.nominal[+e.target.dataset.axis] = raw === '' ? null : Number(raw);
      // all three cleared means no nominal at all, not a nominal of nothing
      if (ap.nominal.every((v) => v === null)) ap.nominal = null;
      markDirty(); renderPanels(); render();
    };
  }

  const tol = $('nomtol');
  if (tol) tol.onchange = (e) => {
    const ap = pts.find((p) => p.id === state.activePoint);
    if (!ap) return;
    const raw = e.target.value.trim();
    ap.nominal_sigma = raw === '' ? null : Number(raw);
    // Typing one number here means one number, so any per-axis override the
    // calibration import left behind goes. Keeping it would leave a field
    // showing the figure that was typed while the axes it does not cover are
    // the only ones using it -- a tolerance that silently does not apply is
    // worse than one that is wrong.
    if (ap.nominal_sigma_axes) {
      ap.nominal_sigma_axes = null;
      setStatus('typed tolerance now governs all three axes, replacing the '
              + 'per-axis one the calibration file stated', 'warn');
    }
    markDirty(); renderPanels();
  };

  const nu = $('nomunit');
  if (nu) nu.onchange = (e) => {
    // the numbers stay as typed; what changes is what they mean. Saying so is
    // the difference between a unit switch and a silent 1000x on the datum.
    state.project.nominal_units = e.target.value;
    state.project.nominal_scale = UNIT_SCALES[e.target.value];
    setStatus(`nominals are now read as ${e.target.value} — the typed numbers did not change`, 'ok');
    markDirty(); renderPanels(); render();
  };

  const cn = $('clearnom');
  if (cn) cn.onclick = () => {
    const ap = pts.find((p) => p.id === state.activePoint);
    if (ap) {
      ap.nominal = null;
      ap.nominal_sigma = null;
      ap.nominal_sigma_axes = null;
      markDirty(); renderPanels(); render();
    }
  };
}

/* --------------------------------------------------------- known objects
 *
 * Two lists, because they are genuinely two things. A *model* is measured
 * geometry -- one CAD file, its units, and the anchor points marked on it. An
 * *object* is one placement of that model in the scene. Ten identical cameras in
 * an array are one model and ten objects, which is what makes measuring the body
 * once enough to place all ten.
 */

const AXIS_CHOICES = [
  ['+X', [1, 0, 0]], ['-X', [-1, 0, 0]],
  ['+Y', [0, 1, 0]], ['-Y', [0, -1, 0]],
  ['+Z', [0, 0, 1]], ['-Z', [0, 0, -1]],
];

const axisLabel = (v) => {
  const found = AXIS_CHOICES.find(([, a]) => a.every((x, i) => Math.abs(x - v[i]) < 1e-9));
  return found ? found[0] : 'custom';
};

function renderObjectsPanel() {
  const el = $('objpanel');
  const tpls = Object.entries(templates());
  const objs = instanceList();
  $('objcount').textContent = objs.filter((o) => !o.is_part).length
    ? `${objs.filter((o) => !o.is_part).length}` : '';

  let html = '';

  if (!tpls.length) {
    html += `<div class="empty">No models yet. Add a CAD model of an object whose real size you `
          + `know — its geometry fixes the scene's scale, and placing it is what a camera-array `
          + `layout actually is.</div>`;
  }

  for (const [tid, t] of tpls) {
    const mesh = state.meshes.get(tid);
    const sel = tid === state.activeTemplate;
    const anchors = t.anchors || [];
    const nInst = objs.filter((o) => o.template === tid).length;
    html += `<div class="tpl${sel ? ' sel' : ''}${isFolded('tpl:' + tid) ? ' folded' : ''}" `
          +   `data-tpl="${escapeHtml(tid)}">`
          // the header stays readable folded, so a collapsed card still says
          // what it is and how far along it is
          + `<div class="tplhead" data-fold="tpl:${escapeHtml(tid)}">`
          +   `<b>${escapeHtml(t.name || tid)}</b>`
          +   `<span class="est">${anchors.length} anchor(s)`
          +     `${nInst ? ` · ${nInst} object(s)` : ''}</span></div>`
          + `<div class="tplbody">`;

    if (mesh === null) {
      html += `<div class="msg">loading model…</div>`;
    } else if (mesh && mesh.error) {
      html += `<div class="note bad">${escapeHtml(mesh.error)}</div>`;
    } else if (mesh) {
      /* The model's real size, in the project's own units rather than metres.
       * "50 x 100 x 185 mm" is read at a glance where "0.050 x 0.100 x 0.185 m"
       * has to be decoded, it is the same unit as the size box on this row, and
       * the largest of the three IS that box -- so the row checks itself. */
      const real = mesh.size.map(
        (v) => +((v * t.unit_scale) / nominalScale()).toPrecision(3));
      const spanLocal = modelSpanLocal(t);
      const spanNom = spanLocal > 0 ? modelSpanNominal(t) : null;
      html += `<div class="msg">${mesh.n_faces} faces, ${mesh.n_feature_edges} of `
            +   `${mesh.n_all_edges || mesh.n_feature_edges} edges drawn</div>`
            // the angle that decides which edges those are. A filleted or finely
            // tessellated part loses its curves at the default, and nothing on
            // screen says so -- the count above is the only clue, and this is
            // the control that acts on it.
            + `<div class="unitrow">`
            +   `<span class="k">detail</span>`
            +   `<select class="tplcrease" data-tpl="${escapeHtml(tid)}">`
            +     CREASE_ANGLES.map(([deg, label]) =>
                    `<option value="${deg}"${Math.abs(deg - creaseOf(t)) < 0.01
                      ? ' selected' : ''}>${label}</option>`).join('')
            +   `</select>`
            +   `<span class="est">edges sharper than ${creaseOf(t)}\u00b0</span>`
            + `</div>`
            + `<div class="unitrow">`
            /* One row for how big the model is: the box is its longest side,
             * the dropdown sets that from a declared CAD unit. These were two
             * rows stating the same fact three ways -- a units picker, a
             * "1 model unit = 1.000 mm" ratio and a real-size readout -- which
             * is more arithmetic than anyone wants on a card they are only
             * glancing at. */
            +   `<span class="k">size</span>`
            +   (spanNom === null ? ''
                  : `<input class="tplspan" type="text" inputmode="decimal" `
                    + `data-tpl="${escapeHtml(tid)}" value="${spanNom.toPrecision(6)}" `
                    + `title="Longest side, in ${escapeHtml(nominalUnits())}. Type any size `
                    + `at all — every copy of the model keeps its own centre, so nothing `
                    + `jumps.">`
                    // named only when it is not the unit the dropdown beside it
                    // already shows, which is almost never
                    + (nominalUnits() === t.units
                        ? '' : `<span class="k">${escapeHtml(nominalUnits())}</span>`))
            +   `<select class="tplunit" data-tpl="${escapeHtml(tid)}" `
            +     `title="Set the scale from the units the CAD file is drawn in">`
            // a model resized by hand no longer sits on any named unit, and
            // showing it as "mm" when it is not would be a lie the real-size
            // readout beside it immediately contradicts
            +     (t.units in UNIT_SCALES ? ''
                    : `<option value="custom" selected>custom</option>`)
            +     Object.keys(UNIT_SCALES).map((u) =>
                    `<option value="${u}"${u === t.units ? ' selected' : ''}>${u}</option>`).join('')
            +   `</select>`
            +   `<span class="est">= ${real.join(' × ')} ${escapeHtml(nominalUnits())}</span>`
            + `</div>`;
      /* How far the file's own zero is from the body it describes. Silent
       * until it matters, and then worth one line: a tooling origin further
       * from the geometry than the geometry is wide is why a fit-all opens on
       * empty space and why the world triad is nowhere near the model. */
      const ob = mesh.bounds || [[0, 0, 0], [0, 0, 0]];
      const far = Math.max(...[0, 1, 2].map((kk) => Math.max(
        Math.abs(ob[0][kk]), Math.abs(ob[1][kk]))));
      if (spanLocal > 0 && far > spanLocal * 1.5) {
        html += `<div class="msg">origin sits `
              + `${fmtRadius((far * t.unit_scale) / nominalScale())} `
              + `${escapeHtml(nominalUnits())} off the geometry — use `
              + `<b>fit selection</b> in the 3D view, not <b>fit</b></div>`;
      }
      if (mesh.units_note) {
        html += `<div class="msg">${escapeHtml(mesh.units_note)}</div>`;
      }
      // what the model actually offers, so it is visible before opening the 3D
      // view -- a part with no round features and everything filleted is worth
      // knowing about early
      const pc = mesh.pick_counts || {};
      html += `<div class="msg">offers `
            + `<b style="color:#8ee06a">${pc.centre || 0}</b> round-feature centres, `
            + `<b style="color:#c89bff">${pc.virtual || 0}</b> filleted corners, `
            + `<b style="color:#8891a0">${pc.corner || 0}</b> sharp corners</div>`;
    } else {
      html += `<div class="msg">no model file — anchors can still be typed</div>`;
    }

    if (anchors.length) {
      html += '<div class="anchorlist">';
      for (const a of anchors) {
        const inst = activeInstance();
        const marks = inst ? anchorMarkCount(anchorKey(inst.id, a.name)) : 0;
        const isActive = sel && a.name === state.activeAnchor;
        html += `<div class="an${marks >= 2 ? ' marked' : ''}${isActive ? ' sel' : ''}" `
              + `data-tpl="${escapeHtml(tid)}" data-anchor="${escapeHtml(a.name)}">`
              + `<span class="dot"></span>`
              + `<span class="name">${escapeHtml(a.name)}</span>`
              + `<span class="n">${marks}img</span>`
              + `<button class="x-btn" data-delanchor="${escapeHtml(a.name)}" `
              + `data-tpl="${escapeHtml(tid)}" title="Remove this anchor">×</button></div>`;
      }
      html += '</div>';
      if (sel && state.activeAnchor && anchors.some((a) => a.name === state.activeAnchor)) {
        html += `<div class="unitrow"><span class="k">name</span>`
              + `<input type="text" id="anchorname" value="${escapeHtml(state.activeAnchor)}" style="flex:1"></div>`;
      }
    } else {
      html += `<div class="msg">No anchors. Open the 3D view (<b>D</b>), switch to <b>model</b>, `
            + `and click the corners you can also find in the photos.</div>`;
    }

    const o = t.optical;
    html += `<div class="row" style="margin-top:5px"><span class="k">optical frame</span>`
          + `<span class="v">${o ? `${axisLabel(o.forward)} fwd` : '<span class="est">none</span>'}</span></div>`;
    if (o) {
      html += `<div class="msg">lens at ${o.center.map((v) => v.toFixed(1)).join(', ')} ${escapeHtml(t.units)}`
            + `, up ${axisLabel(o.up)}${o.fov_deg ? `, ${o.fov_deg}° fov` : ''}</div>`;
    } else {
      html += `<div class="msg">Without one the solve places the body, not the camera inside it.</div>`;
    }
    html += `<div style="display:flex;gap:4px;margin-top:5px;flex-wrap:wrap">`
          + `<button class="smallbtn" data-autoanchor="${escapeHtml(tid)}" `
          +   `title="Take the best-spread features the model already offers, `
          +   `so posing can start without picking them by hand">use model features</button>`
          + `<button class="smallbtn" data-pick="${escapeHtml(tid)}">pick anchors in 3D</button>`
          + `<button class="smallbtn" data-typed="${escapeHtml(tid)}">type a position…</button>`
          + `<button class="smallbtn" data-optical="${escapeHtml(tid)}">${o ? 'edit' : 'set'} optical</button>`
          + `<button class="smallbtn" data-newobj="${escapeHtml(tid)}">+ object</button>`
          // The part is a different kind of thing from a placed object, so the
          // button says so rather than reading as one more placement.
          + (isPartTemplate(tid)
              ? `<button class="smallbtn" data-setpart="" title="Go back to treating this as an `
                + `ordinary placed object">is the part ✓</button>`
              : `<button class="smallbtn" data-setpart="${escapeHtml(tid)}" title="This is the `
                + `object being photographed, not something placed in the scene. In part space it `
                + `goes in at the origin with no anchors.">this is the part</button>`)
          + `<button class="smallbtn" data-deltpl="${escapeHtml(tid)}">remove model</button>`
          + `</div></div></div>`;
  }

  html += `<button class="smallbtn" id="addmodel" style="width:100%">+ add model (OBJ / STL / PLY)</button>`;

  // The part is listed in its own panel, not here: it has no anchors to count,
  // no pose to clear and no shape error to read, so every column in this table
  // would be blank or misleading for it.
  const placedObjs = objs.filter((o) => !o.is_part);
  if (placedObjs.length) {
    html += `<div class="objlist" style="margin-top:7px">`;
    for (const inst of placedObjs) {
      const rep = state.report && state.report.instances ? state.report.instances[inst.id] : null;
      const placed = isPlaced(inst);
      const shape = rep && rep.placed ? `${(rep.shape_rms * 1000).toFixed(1)}mm` : '';
      const bad = rep && rep.placed && (rep.shape_rms > 0.005 || Math.abs(rep.implied_scale - 1) > 0.05);
      html += `<div class="obj${inst.id === state.activeInstance ? ' sel' : ''}`
            + `${placed ? ' placed' : ''}${inst.locked ? ' locked' : ''}" data-obj="${escapeHtml(inst.id)}">`
            + `<span class="swatch"></span>`
            + `<span class="name">${escapeHtml(inst.label || inst.id)}`
            + `<span style="color:var(--text-dim)"> ${rep ? `${rep.n_anchors_used}/${rep.n_anchors_marked}` : ''}`
            // Said on the row, not only in the detail below it, because "why is
            // this body not holding the cameras" is a question asked while
            // scanning the list rather than after selecting one.
            + `${inst.anchors_off ? ' · not voting' : ''}</span></span>`
            + `<span class="err${bad ? ' hi' : ''}">${shape || (placed ? 'placed' : 'not placed')}</span></div>`;
    }
    html += `</div>`;

    const inst = activeInstance();
    if (inst) {
      const rep = state.report && state.report.instances ? state.report.instances[inst.id] : null;
      html += `<div class="row" style="margin-top:7px"><span class="k">label</span>`
            + `<input type="text" id="objlabel" value="${escapeHtml(inst.label)}" style="width:140px"></div>`;
      html += `<label class="toggle"><input type="checkbox" id="objlocked"${inst.locked ? ' checked' : ''}>`
            + ` lock — this object is the world frame</label>`;
      /* The per-body half of "take the CAD out of the solve". The project-wide
       * switch lives in the CAD panel beside the verdict that justifies it;
       * this one is for the single body you have just caught, which is the
       * granularity the failure actually comes in -- one anchor clicked onto
       * the wrong one of two similar corners, while the same model placed nine
       * other times is marked perfectly. */
      html += `<label class="toggle" title="Keeps every click and keeps the body in the scene, `
            + `but its anchors stop saying where the cameras are. It is still placed — against `
            + `the cameras the tie points give, rather than by moving them."`
            + `><input type="checkbox" id="objanchorsoff"${inst.anchors_off ? ' checked' : ''}>`
            + ` anchors out of the solve</label>`;
      if (inst.anchors_off) {
        html += `<div class="note">This body's marks are on the record but not in the solve: `
              + `they cannot move a camera, a point or the focal. It is placed afterwards by `
              + `triangulating them against the finished cameras, so its shape error below is a `
              + `measurement with nothing bent to meet it. Untick to let it vote again.</div>`;
      }
      if (!(rep && rep.placed)) {
        // a bare "not placed" reads as a failure when it usually means the
        // solve has not been run since the marks went in
        html += `<div class="note">${escapeHtml(placementAdvice(inst))}</div>`;
      }
      if (inst.manual) {
        html += `<div class="note">posed by hand${isPlaced(inst) ? '' : ' (no pose yet)'} — `
              + `an eye-fit, not a measurement. Snap its anchors and press Enter to `
              + `measure it.</div>`;
      }
      if (isPlaced(inst)) {
        /* The buttons carry the frame in their own labels. Putting it only in
         * the hint strip was not enough: this panel is where a body is turned,
         * and a button reading "90° X" beside a body sitting at a compound
         * angle is the same label for two different turns. "obj X" says which
         * one is about to happen without having to look anywhere else. */
        const sp = state.transformSpace === 'object' ? 'obj ' : '';
        html += `<div style="display:flex;gap:4px;margin-top:5px;flex-wrap:wrap">`
              + `<button class="smallbtn" id="posesnap">snap anchors here</button>`
              + `<button class="smallbtn" id="poseturn" data-ax="0">90° ${sp}X</button>`
              + `<button class="smallbtn" id="poseturny" data-ax="1">90° ${sp}Y</button>`
              + `<button class="smallbtn" id="poseturnz" data-ax="2">90° ${sp}Z</button>`
              + `<button class="smallbtn" id="poseclear">clear pose</button></div>`
              + `<div class="hintline" style="opacity:.75;margin-top:3px">turning in `
              + `<b>${spaceLabel()}</b> space — tap <b>S</b> to switch</div>`;
      }
      if (rep && rep.placed) {
        html += `<div class="row"><span class="k">anchor RMS</span><span class="v">${rep.rms_px.toFixed(2)} px</span></div>`;
        html += `<div class="row"><span class="k">shape error</span>`
              + `<span class="v${rep.shape_rms > 0.005 ? ' v-poor' : ''}">${(rep.shape_rms * 1000).toFixed(2)} mm</span></div>`;
        html += `<div class="row"><span class="k">solved size</span>`
              + `<span class="v${Math.abs(rep.implied_scale - 1) > 0.05 ? ' v-poor' : ''}">`
              + `${rep.implied_scale.toFixed(4)}×</span></div>`;
      }
      for (const n of (rep && rep.notes) || []) html += `<div class="note">${escapeHtml(n)}</div>`;
      html += `<div style="display:flex;gap:4px;margin-top:5px">`
            + `<button class="smallbtn" id="objclear">clear its marks here</button>`
            + `<button class="smallbtn" id="objdel">delete object</button></div>`;
    }
  }

  const rig = state.rig || (state.report && state.report.rig);
  if (rig && rig.pairs && rig.pairs.length) {
    const unit = rig.world_units || rig.units || 'm';
    html += `<div class="row" style="margin-top:8px"><span class="k">layout extent</span>`
          + `<span class="v">${rig.extent.toFixed(4)} ${unit}</span></div>`;
    html += '<table class="rigtable">';
    for (const p of rig.pairs.slice(0, 14)) {
      html += `<tr><td>${escapeHtml(p.a)} – ${escapeHtml(p.b)}</td>`
            + `<td>${p.distance.toFixed(4)}</td>`
            + `<td>${p.angle_deg != null ? p.angle_deg.toFixed(2) + '°' : '—'}</td></tr>`;
    }
    html += '</table>';
    if (rig.pairs.length > 14) {
      html += `<div class="msg">${rig.pairs.length - 14} more pair(s) — export rig.csv for all of them</div>`;
    }
    for (const n of rig.notes || []) html += `<div class="note">${escapeHtml(n)}</div>`;
  }

  el.innerHTML = html;
  wireObjectsPanel(el);
}

function wireObjectsPanel(el) {
  $('addmodel').onclick = () => $('modelinput').click();

  for (const btn of el.querySelectorAll('[data-setpart]')) {
    btn.onclick = (ev) => {
      ev.stopPropagation();
      designatePart(btn.dataset.setpart || null);
    };
  }

  // The header folds the card; clicking anywhere else in it still selects the
  // model, so folding never costs you the selection you were about to make.
  for (const head of el.querySelectorAll('.tplhead[data-fold]')) {
    head.onclick = (ev) => {
      ev.stopPropagation();
      state.activeTemplate = head.parentElement.dataset.tpl;
      toggleFold(head.dataset.fold);
      renderObjectsPanel();
      syncViewer();
    };
  }
  for (const card of el.querySelectorAll('.tpl[data-tpl]')) {
    card.onclick = (ev) => {
      if (ev.target.closest('button, select, input, .an, .tplhead')) return;
      state.activeTemplate = card.dataset.tpl;
      renderObjectsPanel();
      syncViewer();
    };
  }
  for (const sel of el.querySelectorAll('select.tplcrease')) {
    sel.onchange = (e) => {
      const t = templates()[sel.dataset.tpl];
      if (!t) return;
      t.crease_angle = parseFloat(e.target.value);
      markDirty();
      // the wireframe is built server-side at a fixed angle, so the cached one
      // is now the wrong picture; dropping it is what makes ensureMeshes refetch
      state.meshes.delete(sel.dataset.tpl);
      renderObjectsPanel();
      render();
      ensureMeshes();
    };
  }
  for (const box of el.querySelectorAll('input.tplspan')) {
    const commit = () => {
      const t = templates()[box.dataset.tpl];
      if (!t) return;
      const span = modelSpanLocal(t);
      const want = parseFloat(box.value);
      if (!(want > 0) || !(span > 0)) {
        setStatus('size must be a positive number', 'warn');
        renderObjectsPanel();
        return;
      }
      if (setTemplateScale(t, (want * nominalScale()) / span)) {
        setStatus(`${t.name || t.id} scaled to ${want} ${nominalUnits()} on its longest `
                  + `side — every copy kept its centre`, 'ok');
      }
      renderObjectsPanel();
      syncViewer();
      render();
      refreshRig();
    };
    box.onchange = commit;
    box.onkeydown = (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); box.blur(); } };
  }
  for (const sel of el.querySelectorAll('select.tplunit')) {
    sel.onchange = (e) => {
      const t = templates()[sel.dataset.tpl];
      if (!(e.target.value in UNIT_SCALES)) return;   // "custom" is a readout
      // through setTemplateScale, so switching units holds each copy's centre
      // the same way the size box and the drag handle do -- otherwise mm to m
      // would fling every body a thousand times its origin offset away
      if (!setTemplateScale(t, UNIT_SCALES[e.target.value])) {
        t.units = e.target.value;
        markDirty();
      }
      renderObjectsPanel();
      syncViewer();
      render();
      refreshRig();
    };
  }
  for (const row of el.querySelectorAll('.an[data-anchor]')) {
    row.onclick = (ev) => {
      if (ev.target.closest('button')) return;
      state.activeTemplate = row.dataset.tpl;
      state.activeAnchor = row.dataset.anchor;
      // selecting an anchor is only meaningful against an object of that model
      const inst = activeInstance();
      if (!inst || inst.template !== row.dataset.tpl) {
        const match = instanceList().find((o) => o.template === row.dataset.tpl);
        if (match) state.activeInstance = match.id;
      }
      setMode('objects');
      renderPanels();
      syncViewer();
      render();
    };
  }
  for (const btn of el.querySelectorAll('[data-delanchor]')) {
    btn.onclick = async (ev) => {
      ev.stopPropagation();
      const t = templates()[btn.dataset.tpl];
      const name = btn.dataset.delanchor;
      const marks = instanceList()
        .filter((o) => o.template === t.id)
        .reduce((sum, o) => sum + anchorMarkCount(anchorKey(o.id, name)), 0);
      if (marks && !(await askConfirm(
        `Remove anchor "${name}"?\n\nIt is marked in ${marks} photo(s), and those marks go with it.`,
        { ok: 'Remove', danger: true }))) return;
      deleteAnchorFromTemplate(t, name);
      renderPanels();
      syncViewer();
      render();
    };
  }
  for (const btn of el.querySelectorAll('[data-autoanchor]')) {
    btn.onclick = (ev) => {
      ev.stopPropagation();
      const tid = btn.dataset.autoanchor;
      state.activeTemplate = tid;
      if (!autoAnchors(tid)) {
        setStatus('this model offers no pickable features — try the 3D view, or '
                  + 'type a position from the drawing', 'warn');
      }
    };
  }
  for (const btn of el.querySelectorAll('[data-pick]')) {
    btn.onclick = (ev) => {
      ev.stopPropagation();
      state.activeTemplate = btn.dataset.pick;
      openTemplatePicker();
    };
  }
  for (const btn of el.querySelectorAll('[data-typed]')) {
    btn.onclick = (ev) => {
      ev.stopPropagation();
      state.activeTemplate = btn.dataset.typed;
      addAnchorByCoordinates();
    };
  }
  for (const btn of el.querySelectorAll('[data-optical]')) {
    btn.onclick = (ev) => { ev.stopPropagation(); editOptical(btn.dataset.optical); };
  }
  for (const btn of el.querySelectorAll('[data-newobj]')) {
    btn.onclick = (ev) => { ev.stopPropagation(); addInstance(btn.dataset.newobj); };
  }
  for (const btn of el.querySelectorAll('[data-deltpl]')) {
    btn.onclick = (ev) => { ev.stopPropagation(); removeTemplate(btn.dataset.deltpl); };
  }
  for (const row of el.querySelectorAll('.obj[data-obj]')) {
    row.onclick = () => {
      state.activeInstance = row.dataset.obj;
      const inst = activeInstance();
      if (inst) {
        state.activeTemplate = inst.template;
        const im = currentImage();
        const missing = im ? unmarkedAnchors(inst, im) : [];
        state.activeAnchor = missing.length ? missing[0] : state.activeAnchor;
      }
      setMode('objects');
      renderPanels();
      syncViewer();
      render();
    };
  }

  const aname = $('anchorname');
  if (aname) aname.onchange = (e) => renameAnchor(e.target.value);

  const label = $('objlabel');
  if (label) label.onchange = (e) => {
    const inst = activeInstance();
    if (inst) { inst.label = e.target.value; markDirty(); renderPanels(); render(); refreshRig(); }
  };
  const locked = $('objlocked');
  if (locked) locked.onchange = (e) => {
    const inst = activeInstance();
    if (!inst) return;
    if (e.target.checked && !isPlaced(inst)) {
      e.target.checked = false;
      setStatus('solve once first — locking an unplaced object would pin nothing', 'warn');
      return;
    }
    if (e.target.checked) {
      // only one object can define the frame
      for (const other of instanceList()) if (other !== inst) other.locked = false;
    }
    inst.locked = e.target.checked;
    markDirty();
    renderObjectsPanel();
  };
  const anchorsOff = $('objanchorsoff');
  if (anchorsOff) anchorsOff.onchange = (e) => {
    const inst = activeInstance();
    if (!inst) return;
    pushUndoBefore(e.target.checked
      ? 'take this body’s anchors out of the solve'
      : 'let this body’s anchors back into the solve');
    inst.anchors_off = e.target.checked;
    // The CAD verdict was measured against a different set of voters, so it is
    // void -- same rule as any other edit. Saying nothing here would leave a
    // stale "worsens" sitting above the very switch that answered it.
    state.cadCheck = null;
    markDirty();
    setStatus(e.target.checked
      ? `${inst.label || inst.id}: anchors out of the solve — press Enter to re-solve without `
        + `them. Nothing was deleted.`
      : `${inst.label || inst.id}: anchors back in the solve — press Enter to re-solve.`, 'ok');
    renderPanels();
  };
  const snap = $('posesnap');
  if (snap) snap.onclick = () => {
    const im = currentImage();
    if (im) snapAnchorsHere(im);
  };
  // quarter turns, because a CAD model almost never comes in facing the way the
  // real object does and nudging 90 degrees by hand is both slow and never
  // exactly 90. These follow the same frame the keyboard does -- a button that
  // kept turning about part X while X/Y/Z had been switched to the body's own
  // axes would be the one gesture in the panel that lied about its label.
  for (const [id, axis] of [['poseturn', 0], ['poseturny', 1], ['poseturnz', 2]]) {
    const btn = $(id);
    if (!btn) continue;
    btn.onclick = (e) => {
      const inst = activeInstance();
      if (!inst || !isPlaced(inst)) return;
      const dir = turnAxisWorld(inst, 'xyz'[axis]);
      rotateInPlace(inst, dir, (e.shiftKey ? -1 : 1) * Math.PI / 2);
      inst.manual = true;
      inst.pose_in_part_space = partSpaceState().ok;
      markDirty();
      renderPanels();
      render();
      setStatus(`turned ${e.shiftKey ? '-' : '+'}90\u00b0 about ${spaceLabel()} `
                + `${AXIS_NAME[axis]}`, 'ok');
    };
    btn.title = `Quarter turn about ${AXIS_NAME[axis]} in the current frame `
              + `(S switches part/object; shift for the other way)`;
  }
  const posedrop = $('poseclear');
  if (posedrop) posedrop.onclick = () => {
    const inst = activeInstance();
    if (inst) clearPose(inst);
  };
  const clear = $('objclear');
  if (clear) clear.onclick = () => {
    const inst = activeInstance();
    const im = currentImage();
    if (!inst || !im) return;
    for (const key of Object.keys(im.anchor_obs || {})) {
      if (splitAnchorKey(key)[0] === inst.id) delete im.anchor_obs[key];
    }
    markDirty();
    renderPanels();
    render();
  };
  const del = $('objdel');
  if (del) del.onclick = async () => {
    const inst = activeInstance();
    if (!inst) return;
    const marks = state.project.images.reduce(
      (sum, im) => sum + Object.keys(im.anchor_obs || {}).filter((k) => splitAnchorKey(k)[0] === inst.id).length, 0);
    let question = `Delete ${inst.label || inst.id}?`;
    if (marks) question += `\n\nIt carries ${marks} anchor mark(s), which will be lost.`;
    if (!(await askConfirm(question, { ok: 'Delete', danger: true }))) return;
    deleteInstance(inst.id);
    renderPanels();
    syncViewer();
    render();
  };
}

/** Rename an anchor, carrying its marks with it.
 *
 *  Marks are keyed by anchor name, so a rename has to rekey every one of them --
 *  otherwise the marks silently stop referring to anything and the solve just
 *  quietly has less to work with. */
function renameAnchor(raw) {
  const t = activeTemplate();
  const old = state.activeAnchor;
  if (!t || !old) return;
  // the separator cannot appear in a name or the observation key is ambiguous
  const name = String(raw || '').replace(/\//g, '-').trim().slice(0, 60);
  if (!name || name === old) { renderObjectsPanel(); return; }
  if ((t.anchors || []).some((a) => a.name === name)) {
    setStatus(`this model already has an anchor called “${name}”`, 'warn');
    renderObjectsPanel();
    return;
  }
  const anchor = (t.anchors || []).find((a) => a.name === old);
  if (!anchor) return;
  anchor.name = name;

  for (const im of state.project.images) {
    for (const key of Object.keys(im.anchor_obs || {})) {
      const [instId, anchorName] = splitAnchorKey(key);
      const inst = instanceList().find((o) => o.id === instId);
      if (anchorName === old && inst && inst.template === t.id) {
        im.anchor_obs[anchorKey(instId, name)] = im.anchor_obs[key];
        delete im.anchor_obs[key];
      }
    }
  }
  state.activeAnchor = name;
  markDirty();
  renderPanels();
  syncViewer();
  render();
}


/* ------------------------------------------------------------- the part panel
 *
 * Two features share this panel because they are two halves of one question:
 * "does the CAD model agree with the photographs?" The fit report answers it in
 * numbers, and the projected wireframe answers it by eye -- and the second is
 * the one that catches a body placed half a turn out, which no number does.
 */

const PART_DEV_WARN = 1.0;
/* Deviation, in nominal units, past which a row is coloured. Deliberately a
 * fixed figure and not derived from the tolerances: a nominal's tolerance is
 * what the solve was *told*, and colouring by it would mean a slack tolerance
 * painted a bad fit green. */

function renderPartPanel() {
  const el = $('partpanel');
  if (!el) return;
  const inst = partInstance();
  const t = partTemplate();
  const tpls = Object.entries(templates());
  const ps = partSpaceState();
  $('partcount').textContent = inst ? (ps.ok ? 'placed' : 'no frame') : '';

  let html = '';

  if (!inst) {
    html += `<div class="empty">No model is marked as the part. A placed object needs anchors `
          + `and a pose; <b>the part</b> needs neither — once the nominals put the solve in the `
          + `part's own frame it sits at the origin by definition, and its wireframe lands in `
          + `every photo with nothing marked.</div>`;
    if (!tpls.length) {
      html += `<div class="msg">Add the part's CAD model in <b>Known objects</b> below first.</div>`;
    } else {
      html += `<div style="display:flex;gap:4px;flex-wrap:wrap">`
            + tpls.map(([tid, tp]) =>
                `<button class="smallbtn" data-setpart="${escapeHtml(tid)}">`
                + `use “${escapeHtml(tp.name || tid)}”</button>`).join('')
            + `</div>`;
    }
    el.innerHTML = html;
    wirePartPanel(el);
    return;
  }

  html += `<div class="row"><span class="k">model</span>`
        + `<span class="v">${escapeHtml(t ? (t.name || t.id) : inst.template)}</span></div>`;

  // Units first and always, agreeing or not. A factor of 1000 between the two
  // declarations makes every number below it meaningless, and it is invisible
  // in all of them.
  const fit = partFit();
  const nu = nominalUnits();
  if (t) {
    const agree = Math.abs((t.unit_scale || 0) - nominalScale()) < 1e-12;
    html += `<div class="row"><span class="k">units</span>`
          + `<span class="v${agree ? '' : ' v-poor'}">model ${escapeHtml(t.units)} · `
          + `nominals ${escapeHtml(nu)}</span></div>`;
    if (!agree) {
      const ratio = (t.unit_scale || 0) / nominalScale();
      const factor = ratio > 1 ? ratio : 1 / ratio;
      html += `<div class="note bad">These disagree by a factor of ${factor.toPrecision(6)}×. `
            + `Nothing has been rescaled to hide it — if one declaration is wrong, the wireframe `
            + `lands that factor out in every photo and every deviation below is meaningless. `
            + `A model in inches against a millimetre drawing is legitimate; a model in metres `
            + `against one is a mistake.</div>`;
    }
  }

  if (!ps.ok) {
    html += `<div class="note bad">Not in part space — ${ps.n} nominal'd point(s), and it takes `
          + `three or more that are not in a straight line. Until then the part has no known `
          + `pose, so it is held at the origin but nothing confirms that is where it belongs. `
          + `Type the drawing's coordinates onto three points in <b>Tie points</b>.</div>`;
  }

  if (fit && fit.ok) {
    const bad = fit.rms_deviation > PART_DEV_WARN;
    html += `<div class="row"><span class="k">fit to the drawing</span>`
          + `<span class="v${bad ? ' v-poor' : ' v-good'}">`
          + `${fit.rms_deviation.toFixed(3)} ${escapeHtml(nu)} RMS</span></div>`
          + `<div class="row"><span class="k">off the model surface</span>`
          + `<span class="v${fit.rms_surface > PART_DEV_WARN ? ' v-poor' : ''}">`
          + `${fit.rms_surface.toFixed(3)} ${escapeHtml(nu)} RMS</span></div>`
          + `<div class="row"><span class="k">worst point</span>`
          + `<span class="v">${escapeHtml(fit.worst_point || '—')} `
          + `(${fit.max_deviation.toFixed(3)})</span></div>`
          + `<div class="row"><span class="k">checkable points</span>`
          + `<span class="v${fit.n_checkable < 3 ? ' v-weak' : ''}">`
          + `${fit.n_checkable} of ${fit.n_points}</span></div>`;

    html += `<table class="rigtable" style="margin-top:6px">`
          + `<tr><th>point</th><th>views</th><th>dev</th><th>surface</th></tr>`;
    for (const row of fit.points) {
      if (!row.solved || !row.solved.length) {
        html += `<tr><td>${escapeHtml(row.point)}</td><td>${row.n_views}</td>`
              + `<td colspan="2" class="est">not solved</td></tr>`;
        continue;
      }
      // A two-view point's deviation is structurally near zero whatever the
      // drawing says, so it is shown struck through rather than green.
      const cls = row.deviation_mag > PART_DEV_WARN ? ' class="hi"' : '';
      html += `<tr${row.checkable ? '' : ' style="opacity:.55"'}>`
            + `<td>${escapeHtml(row.point)}</td>`
            + `<td>${row.n_views}${row.checkable ? '' : '<span class="est">!</span>'}</td>`
            + `<td${cls}>${row.deviation_mag.toFixed(3)}</td>`
            + `<td>${Number.isFinite(row.surface_distance) ? row.surface_distance.toFixed(3) : '—'}</td>`
            + `</tr>`;
    }
    html += `</table>`;
    html += `<div class="msg">dev = how far the solve put the point from the drawing; `
          + `surface = how far it is from the model's own geometry. A point marked in fewer `
          + `than three photos (<b>!</b>) sits wherever its nominal puts it and checks nothing.</div>`;
  } else if (fit && fit.stage && fit.stage !== 'no part designated') {
    html += `<div class="msg">fit: ${escapeHtml(fit.stage)}</div>`;
  } else {
    html += `<div class="msg">Solve, or press <b>place from nominals</b>, to get the fit.</div>`;
  }

  for (const w of (fit && fit.warnings) || []) {
    html += `<div class="note bad">${escapeHtml(w)}</div>`;
  }

  // ---- stage one
  const pl = state.partPlacement;
  html += `<div style="display:flex;gap:4px;margin-top:6px;flex-wrap:wrap">`
        + `<button class="smallbtn" id="partplace" title="Pin the part at the origin and `
        +   `project its wireframe into every photo. Pure geometry: no anchors, no search.">`
        +   `place from nominals</button>`
        + `<button class="smallbtn" id="partclear">not the part</button></div>`;
  if (pl && pl.images && pl.images.length) {
    const off = pl.images.filter((i) => i.positioned && !i.n_inside).length;
    const none = pl.images.filter((i) => !i.positioned).length;
    html += `<div class="msg">stage one: drawn in `
          + `${pl.images.filter((i) => i.positioned && i.n_inside).length} photo(s)`
          + `${off ? `, off-frame in ${off}` : ''}${none ? `, ${none} with no pose` : ''}.</div>`;
  }

  // ---- stage two
  html += `<label class="toggle" title="Reads the photograph and slides the wireframe onto its `
        + `edges. Off by default, and per image."><input type="checkbox" id="edgerefine"`
        + `${state.project.settings.edge_refine ? ' checked' : ''}> align to image edges `
        + `(experimental)</label>`;
  if (state.project.settings.edge_refine) {
    html += `<div class="note">This is a <b>recovery</b> tool, not a refinement. Measured on a `
          + `scene with known truth: a pose a few pixels out is pulled back to about 1.5 px, but `
          + `a pose that is already good is made <i>worse</i> — the distance transform cannot `
          + `resolve better than half a working pixel and the bundle adjustment already does. `
          + `So this photo's own clicked marks decide, not the edge score, and a photo with no `
          + `marks is refused. Use it when the wireframe is visibly beside the part.</div>`;
    const im = currentImage();
    html += `<div style="display:flex;gap:4px;margin-top:4px;flex-wrap:wrap">`
          + (im ? `<button class="smallbtn" id="refinethis">align ${escapeHtml(im.id)}</button>` : '')
          + `<button class="smallbtn" id="refineall">align every photo</button>`
          + (refinedImages().length
              ? `<button class="smallbtn" id="refineundo">undo all `
                + `(${refinedImages().length})</button>` : '')
          + `</div>`;
    const rf = state.refine;
    if (rf && rf.results && rf.results.length) {
      html += `<table class="rigtable" style="margin-top:6px">`
            + `<tr><th>photo</th><th>wireframe px</th><th>marks px</th><th>outcome</th></tr>`;
      for (const r of rf.results) {
        const edge = Number.isFinite(r.median_before)
          ? `${r.median_before.toFixed(1)} → ${r.median_after.toFixed(1)}` : '—';
        const marks = r.mark_rms_before == null ? '—'
          : `${r.mark_rms_before.toFixed(2)} → `
            + `${(r.mark_rms_after == null ? r.mark_rms_before : r.mark_rms_after).toFixed(2)}`;
        html += `<tr><td>${escapeHtml(r.image)}</td><td>${edge}</td>`
              + `<td${r.accepted ? '' : ' class="hi"'}>${marks}</td>`
              + `<td>${r.accepted ? 'kept' : escapeHtml(r.stage)}</td></tr>`;
      }
      html += `</table>`;
      for (const r of rf.results) {
        if (r.reason) html += `<div class="msg">${escapeHtml(r.image)}: ${escapeHtml(r.reason)}</div>`;
      }
      for (const w of rf.warnings || []) html += `<div class="note bad">${escapeHtml(w)}</div>`;
    }
  }

  el.innerHTML = html;
  wirePartPanel(el);
}

function wirePartPanel(el) {
  for (const btn of el.querySelectorAll('[data-setpart]')) {
    btn.onclick = () => designatePart(btn.dataset.setpart);
  }
  const place = $('partplace');
  if (place) place.onclick = () => placePartNow();
  const clear = $('partclear');
  if (clear) clear.onclick = () => designatePart(null);
  const toggle = $('edgerefine');
  if (toggle) {
    toggle.onchange = (e) => {
      state.project.settings.edge_refine = e.target.checked;
      markDirty();
      renderPartPanel();
    };
  }
  const one = $('refinethis');
  if (one) one.onclick = () => refineEdges(currentImage() ? [currentImage().id] : []);
  const all = $('refineall');
  if (all) all.onclick = () => refineEdges(state.project.images.map((im) => im.id));
  const undo = $('refineundo');
  if (undo) undo.onclick = () => revertEdges(null);
}

/** Mark a template as the part, or (null) stop treating anything as one. */
async function designatePart(templateId) {
  if (templateId === null) {
    const inst = partInstance();
    if (!inst) return;
    if (!(await askConfirm(
      `Stop treating “${(partTemplate() || {}).name || inst.template}” as the part?`
      + `\n\nThe model stays in the project; it just goes back to being an object you would `
      + `have to place.`, { ok: 'Not the part' }))) return;
  }
  try {
    const data = await api('/api/set_part', {
      project: projectPayload(), template: templateId,
    });
    adopt(data);
    if (data.fit) state.partFit = data.fit;
    state.partPlacement = null;
    clearDirty();
    setStatus(templateId
      ? 'marked as the part — it sits at the origin, no anchors needed'
      : 'no longer treated as the part');
  } catch (err) {
    setStatus(`could not set the part: ${err.message}`, 'bad');
    return;
  }
  renderPanels();
  syncViewer();
  render();
}

/** Stage one: pin the part at the origin and see where it lands in each photo. */
async function placePartNow() {
  setStatus('placing the part from the nominals…');
  try {
    const data = await api('/api/place_part', { project: projectPayload() });
    adopt(data);
    state.partPlacement = data.placement;
    if (data.fit) state.partFit = data.fit;
    clearDirty();
    const pl = data.placement || {};
    if (!pl.ok) {
      setStatus((pl.warnings && pl.warnings[0]) || pl.stage || 'could not place the part', 'warn');
    } else {
      const drawn = (pl.images || []).filter((i) => i.positioned && i.n_inside).length;
      setStatus(`the part is projected into ${drawn} photo(s) from its known pose — `
                + `check the wireframe lands on it`);
    }
  } catch (err) {
    setStatus(`could not place the part: ${err.message}`, 'bad');
  }
  renderPanels();
  syncViewer();
  render();
}

async function refineEdges(imageIds) {
  if (!imageIds || !imageIds.length) return;
  setStatus(`aligning ${imageIds.length} photo(s) to their image edges…`);
  try {
    const data = await api('/api/edge_refine', {
      project: projectPayload(), images: imageIds,
    });
    adopt(data);
    state.refine = data.refine;
    if (data.fit) state.partFit = data.fit;
    clearDirty();
    const n = (data.refine && data.refine.n_accepted) || 0;
    const total = ((data.refine && data.refine.results) || []).length;
    setStatus(n
      ? `${n} of ${total} pose(s) accepted — the rest were left exactly as the solve had them`
      : `nothing accepted: all ${total} pose(s) kept, which is the usual answer when the solve `
        + `was already good`, n ? null : 'warn');
  } catch (err) {
    setStatus(`edge alignment failed: ${err.message}`, 'bad');
  }
  renderPanels();
  render();
}

async function revertEdges(imageIds) {
  try {
    const data = await api('/api/edge_revert', {
      project: projectPayload(), images: imageIds || [],
    });
    adopt(data);
    state.refine = null;
    clearDirty();
    setStatus(`put back ${(data.reverted || []).length} pose(s) as the solve left them`);
  } catch (err) {
    setStatus(`could not undo: ${err.message}`, 'bad');
  }
  renderPanels();
  render();
}

function addInstance(templateId) {
  const t = templates()[templateId];
  if (!t) return;
  const n = instanceList().filter((o) => o.template === templateId).length;
  const inst = {
    id: nextInstanceId(),
    template: templateId,
    label: `${t.name || templateId} ${n + 1}`,
    R: null,
    t: null,
    locked: false,
  };
  instanceList().push(inst);
  state.activeInstance = inst.id;
  state.activeTemplate = templateId;
  const im = currentImage();
  const missing = im ? unmarkedAnchors(inst, im) : [];
  state.activeAnchor = missing.length ? missing[0] : null;
  markDirty();
  setMode('objects');
  renderPanels();
  syncViewer();
  render();
  setStatus(
    (t.anchors || []).length
      ? `${inst.label}: click its anchors in the photos`
      : `${inst.label} added — now pick anchors on the model in the 3D view (D)`,
    'ok'
  );
}

async function removeTemplate(templateId) {
  const used = instanceList().filter((o) => o.template === templateId);
  let question = `Remove this model?`;
  if (used.length) question += `\n\n${used.length} object(s) use it, and they go too, with their marks.`;
  if (!(await askConfirm(question, { ok: 'Remove', danger: true }))) return;
  for (const inst of used) deleteInstance(inst.id);
  delete templates()[templateId];
  state.meshes.delete(templateId);
  if (state.activeTemplate === templateId) {
    const ids = Object.keys(templates());
    state.activeTemplate = ids.length ? ids[0] : null;
  }
  markDirty();
  renderPanels();
  syncViewer();
  render();
}

/** The optical frame: where the lens sits on the body and which way it faces.
 *
 *  Offered as axis choices because a CAD model of a camera almost always has its
 *  optical axis along a model axis; a lens pointing somewhere odd needs the
 *  numbers typed, and the prompt says so. */
async function editOptical(templateId) {
  const t = templates()[templateId];
  if (!t) return;
  const current = t.optical;
  const anchorNames = (t.anchors || []).map((a) => a.name);

  const centreRaw = await askText(
    `Lens centre on the model, in ${t.units}.\n\n`
    + `Type three numbers "x y z", or the name of an anchor to use its position`
    + (anchorNames.length ? ` (${anchorNames.slice(0, 6).join(', ')}${anchorNames.length > 6 ? '…' : ''})` : '')
    + `, or "centre" for the middle of the anchors.\n\nLeave empty to remove the optical frame.`,
    current ? current.center.join(' ') : 'centre'
  );
  if (centreRaw === null) return;
  if (!centreRaw.trim()) {
    t.optical = null;
    markDirty();
    renderObjectsPanel();
    syncViewer();
    refreshRig();
    return;
  }

  let center = null;
  const named = (t.anchors || []).find((a) => a.name === centreRaw.trim());
  if (named) {
    center = [...named.xyz];
  } else if (/^cent(re|er)$/i.test(centreRaw.trim())) {
    const pts = (t.anchors || []).map((a) => a.xyz);
    if (!pts.length) { setStatus('no anchors to average — type coordinates instead', 'warn'); return; }
    center = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  } else {
    const nums = centreRaw.trim().split(/[\s,]+/).map(Number);
    if (nums.length !== 3 || nums.some((v) => !Number.isFinite(v))) {
      setStatus('could not read that as three numbers or an anchor name', 'warn');
      return;
    }
    center = nums;
  }

  const fwdRaw = await askText(
    `Which model axis does the lens look along?\n\n`
    + `One of ${AXIS_CHOICES.map(([n]) => n).join(', ')}, or three numbers for a direction.`,
    current ? axisLabel(current.forward) : '+Z'
  );
  if (fwdRaw === null) return;
  const forward = parseDirection(fwdRaw);
  if (!forward) { setStatus('could not read that direction', 'warn'); return; }

  const upRaw = await askText(
    `Which model axis is the camera's up?\n\n`
    + `One of ${AXIS_CHOICES.map(([n]) => n).join(', ')}, or three numbers.\n`
    + `Only affects roll, and only in what gets drawn and exported.`,
    current ? axisLabel(current.up) : '-Y'
  );
  if (upRaw === null) return;
  const up = parseDirection(upRaw);
  if (!up) { setStatus('could not read that direction', 'warn'); return; }

  const fovRaw = await askText(
    `Horizontal field of view in degrees, if you know it.\n\n`
    + `Only sizes the frustum that gets drawn — nothing in the solve uses it, since `
    + `marking this body's shell can never reveal its lens. Leave empty to skip.`,
    current && current.fov_deg ? String(current.fov_deg) : ''
  );
  if (fovRaw === null) return;
  const fov = fovRaw.trim() ? Number(fovRaw) : null;

  t.optical = {
    center,
    forward,
    up,
    fov_deg: fov && Number.isFinite(fov) && fov > 0 && fov < 180 ? fov : null,
  };
  markDirty();
  renderObjectsPanel();
  syncViewer();
  render();
  refreshRig();
  setStatus(`optical frame set: lens at ${center.map((v) => v.toFixed(1)).join(', ')} ${t.units}`, 'ok');
}

function parseDirection(raw) {
  const text = String(raw || '').trim();
  const named = AXIS_CHOICES.find(([n]) => n.toLowerCase() === text.toLowerCase());
  if (named) return [...named[1]];
  const nums = text.split(/[\s,]+/).map(Number);
  if (nums.length !== 3 || nums.some((v) => !Number.isFinite(v))) return null;
  const len = Math.hypot(...nums);
  return len > 1e-9 ? nums.map((v) => v / len) : null;
}

/** Datum surfaces, typed in the units the drawing states them in.
 *
 *  These are the one thing in the project that is not evidence: nothing here
 *  enters the solve, and a plane has no residual to report, because it states
 *  where a datum *is* rather than claiming to have found one. It exists to be
 *  dimensioned against once the scene is open in CAD, which is why the only
 *  place it shows up is the 3D view and the STEP export.
 */
function renderPlanesPanel() {
  const el = $('plpanel');
  const planes = state.project.reference_planes || (state.project.reference_planes = []);
  const units = state.project.nominal_units || 'mm';
  $('plcount').textContent = planes.length ? `${planes.length}` : '';

  let html = '<div class="cons"><ul>';
  if (!planes.length) {
    html += `<li style="color:var(--text-dim);font-style:italic">None. Add the drawing's `
          + `datums and they come through the STEP export as real surfaces you can `
          + `dimension to.</li>`;
  }
  planes.forEach((pl, i) => {
    const n = pl.normal || [0, 0, 1];
    const axis = AXIS_NAME.findIndex((_, a) => Math.abs(Math.abs(n[a]) - 1) < 1e-9);
    const where = axis >= 0
      ? `${AXIS_NAME[axis]} = ${+(pl.offset * (n[axis] < 0 ? -1 : 1)).toFixed(4)} ${units}`
      : `n=(${n.map((v) => (+v).toFixed(2)).join(', ')}) at ${+(+pl.offset).toFixed(4)} ${units}`;
    const label = pl.label ? `${escapeHtml(pl.label)} — ` : '';
    html += `<li title="${escapeHtml(where)}"><span>${label}${where}</span>`
          + `<button class="x-btn" data-pli="${i}">×</button></li>`;
  });
  html += '</ul>';

  html += `<div style="display:flex;gap:4px;margin-top:7px;align-items:center">`
        + `<input type="text" id="pl-name" placeholder="name" style="flex:1;min-width:0">`
        + `<select id="pl-axis" style="width:52px">`
        + AXIS_NAME.map((nm, a) => `<option value="${a}"${a === 2 ? ' selected' : ''}>${nm}</option>`).join('')
        + `</select>`
        + `<input type="number" id="pl-off" value="0" step="any" style="width:68px" title="offset in ${units}">`
        + `<button class="smallbtn" id="pl-add">add</button></div>`;
  html += `<div class="msg">Offsets are in ${units}, in part space — the same numbers `
        + `you type for a point nominal.</div>`;
  html += '</div>';
  el.innerHTML = html;

  for (const b of el.querySelectorAll('.x-btn')) {
    b.onclick = () => {
      planes.splice(+b.dataset.pli, 1);
      markDirty(); renderPanels(); syncViewer();
    };
  }
  const add = $('pl-add');
  if (add) add.onclick = () => {
    const axis = +$('pl-axis').value;
    const normal = [0, 0, 0];
    normal[axis] = 1;
    const used = new Set(planes.map((p) => p.id));
    let k = 0;
    while (used.has(`pl${k}`)) k += 1;
    planes.push({
      id: `pl${k}`,
      label: $('pl-name').value.trim(),
      normal,
      offset: parseFloat($('pl-off').value) || 0,
      size: null,
    });
    markDirty(); renderPanels(); syncViewer();
  };
}

function renderConstraintsPanel() {
  const el = $('conspanel');
  const pts = state.project.points;
  const opts = (sel) => pts.map((p) =>
    `<option value="${p.id}"${p.id === sel ? ' selected' : ''}>${escapeHtml(p.label || p.id)}</option>`
  ).join('');

  // asked-versus-achieved, keyed back to the row the user typed
  const reports = {};
  if (state.report && state.report.constraints) {
    for (const r of state.report.constraints) reports[r.index] = r;
  }
  const fmt = (v) => (v === 0 ? '0' : Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(1));

  let html = '<div class="cons"><ul>';
  if (!state.project.constraints.length) {
    html += `<li style="color:var(--text-dim);font-style:italic">None. Scale stays arbitrary `
          + `until you add one distance, or place a known object.</li>`;
  }
  state.project.constraints.forEach((c, i) => {
    const text = c.type === 'axis_lock'
      ? `${c.point}.${AXIS_NAME[c.axis]} = ${c.value}`
      : `|${c.point_a} – ${c.point_b}| = ${c.value}`;
    const r = reports[i];
    let detail = '';
    let title = '';
    if (r && !r.applied) {
      detail = `<span class="err hi">no effect</span>`;
      title = r.note;
    } else if (r) {
      const bad = r.sigmas_off > 3;
      detail = `<span class="err${bad ? ' hi' : ''}">${r.error >= 0 ? '+' : ''}${fmt(r.error)}</span>`;
      title = `asked ${fmt(r.target)}, got ${fmt(r.achieved)} — off by ${fmt(r.error)}`
            + ` (${r.sigmas_off.toFixed(1)}x the ${r.sigma} tolerance)`
            + (r.note ? `\n\n${r.note}` : '');
    }
    html += `<li title="${escapeHtml(title)}"><span>${escapeHtml(text)}</span>${detail}`
          + `<button class="x-btn" data-ci="${i}">×</button></li>`;
  });
  html += '</ul>';

  if (state.report && state.report.datum_scale != null) {
    const scale = state.report.datum_scale;
    const off = (scale - 1) * 100;
    const bad = Math.abs(off) > 0.5;
    // only ever set when a known object fixed the scale, so this really is two
    // independent measurements of the same thing
    html += `<div class="row" style="margin-top:6px"><span class="k">stated vs object scale</span>`
          + `<span class="v${bad ? ' v-poor' : ''}">${scale.toFixed(5)}×</span></div>`
          + `<div class="${bad ? 'note' : 'msg'}">`
          + (bad
              ? `These figures and the object geometry disagree by ${off.toFixed(2)}%. `
                + `Check the model's declared units first — that is the usual cause, and it is `
                + `off by a round factor.`
              : `Agrees with the known object geometry to ${Math.abs(off).toFixed(2)}% — `
                + `two independent measurements, so this is a real check.`)
          + `</div>`;
  }

  if (pts.length) {
    html += `<div style="display:flex;gap:4px;margin-top:7px;align-items:center">`
          + `<select id="c-lp" style="flex:1">${opts(state.activePoint)}</select>`
          + `<select id="c-la" style="width:52px">${AXIS_NAME.map((n, a) => `<option value="${a}">${n}</option>`).join('')}</select>`
          + `<input type="number" id="c-lv" value="0" step="any" style="width:58px">`
          + `<button class="smallbtn" id="c-ladd">lock</button></div>`;
    html += `<div style="display:flex;gap:4px;margin-top:4px;align-items:center">`
          + `<select id="c-da" style="flex:1">${opts(state.activePoint)}</select>`
          + `<select id="c-db" style="flex:1">${opts(null)}</select>`
          + `<input type="number" id="c-dv" value="1" step="any" style="width:58px">`
          + `<button class="smallbtn" id="c-dadd">dist</button></div>`;
  }
  html += '</div>';
  el.innerHTML = html;

  for (const b of el.querySelectorAll('.x-btn')) {
    b.onclick = () => { state.project.constraints.splice(+b.dataset.ci, 1); markDirty(); renderPanels(); };
  }
  const ladd = $('c-ladd');
  if (ladd) ladd.onclick = () => {
    state.project.constraints.push({
      type: 'axis_lock', point: $('c-lp').value, axis: +$('c-la').value,
      value: parseFloat($('c-lv').value) || 0, sigma: 0.01,
    });
    markDirty(); renderPanels();
  };
  const dadd = $('c-dadd');
  if (dadd) dadd.onclick = () => {
    const a = $('c-da').value, b = $('c-db').value;
    if (a === b) { setStatus('a distance needs two different points', 'warn'); return; }
    const v = parseFloat($('c-dv').value);
    if (!(v > 0)) { setStatus('distance must be positive', 'warn'); return; }
    state.project.constraints.push({ type: 'distance', point_a: a, point_b: b, value: v, sigma: 0.01 });
    markDirty(); renderPanels();
  };
}

function renderImageListPanel() {
  const el = $('imglistpanel');
  const r = state.report;
  if (!r || !r.ok) { el.innerHTML = '<div class="empty">Solve to see per-image residuals.</div>'; return; }
  const rows = Object.values(r.images)
    .filter((x) => x.positioned)
    .sort((a, b) => b.rms_px - a.rms_px);
  let html = '<div class="imglist">';
  for (const row of rows) {
    const cls = row.rms_px < 2 ? '' : row.rms_px < 6 ? 'style="color:var(--warn)"' : 'style="color:var(--bad)"';
    html += `<div class="ir" data-img="${row.id}"><span>${row.id}</span>`
          + `<span class="n" ${cls}>${row.rms_px.toFixed(2)}px</span>`
          + `<span class="n" style="color:var(--text-dim)">${row.n_observations}</span></div>`;
  }
  const unplaced = Object.values(r.images).filter((x) => !x.positioned);
  for (const row of unplaced) {
    html += `<div class="ir" data-img="${row.id}"><span style="color:var(--text-dim)">${row.id}</span>`
          + `<span class="n" style="color:var(--warn)">not placed</span><span></span></div>`;
  }
  html += '</div>';
  el.innerHTML = html;
  for (const row of el.querySelectorAll('.ir')) {
    row.onclick = () => selectImage(row.dataset.img, true);
  }
}

/* --------------------------------------------------------------- 3D view */

function cameraCenter(im) {
  // centre = -R^T t, with R stored row-major
  const R = im.R;
  const t = im.t;
  return [
    -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
    -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
    -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
  ];
}

function instanceCenterWorld(inst, template) {
  const pts = Object.values(anchorPoints(template));
  if (!pts.length) return [...inst.t];
  const local = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
  const R = inst.R;
  return [
    R[0][0] * local[0] + R[0][1] * local[1] + R[0][2] * local[2] + inst.t[0],
    R[1][0] * local[0] + R[1][1] * local[1] + R[1][2] * local[2] + inst.t[1],
    R[2][0] * local[0] + R[2][1] * local[1] + R[2][2] * local[2] + inst.t[2],
  ];
}

/** Everything the 3D view draws, assembled from the project the client already
 *  holds. Only mesh geometry comes from the server. */
function buildScene() {
  if (!state.project) return null;
  const points = state.project.points.filter((p) => p.xyz).map((p) => p.xyz);
  const cameras = state.project.images
    .filter((im) => im.R && im.t)
    .map((im) => {
      const g = groupOf(im);
      const f = g.f || Math.max(im.width, im.height);
      return {
        id: im.id,
        label: (state.meta[im.id] || {}).name || im.id,
        center: cameraCenter(im),
        R: im.R,
        aspect: im.width / im.height,
        fovScale: im.height / 2 / f,
        selected: im.id === state.current,
      };
    });

  const objects = instanceList().map((inst) => {
    const template = templateOf(inst);
    const placed = isPlaced(inst);
    return {
      box: placed && template ? placedBoxWorld(inst, template) : null,
      id: inst.id,
      label: inst.label || inst.id,
      placed,
      locked: !!inst.locked,
      selected: inst.id === state.activeInstance,
      R: inst.R,
      t: inst.t,
      unitScale: template ? template.unit_scale : 1,
      mesh: template ? meshFor(inst.template) : null,
      anchors: template
        ? (template.anchors || []).map((a) => ({
            name: a.name,
            xyz: a.xyz,
            marked: anchorMarkCount(anchorKey(inst.id, a.name)),
          }))
        : [],
      optical:
        template && template.optical
          ? { center: template.optical.center, forward: template.optical.forward }
          : null,
      center: placed && template ? instanceCenterWorld(inst, template) : null,
    };
  });

  /* How big the scene is.
   *
   * This used to be measured from the tie points, the camera centres and each
   * object's *centre* -- never from how big the objects themselves are. The
   * difference is invisible while every object is small next to the camera
   * walk, and wrong the moment a fixture bigger than the walk is dropped in:
   * the grid step, the frustum size and the opening fit were then all derived
   * from a box the fixture hung a long way outside of, and the view opened
   * buried inside it. Each placed object now contributes its own corners. */
  const marks = [
    ...points,
    ...cameras.map((c) => c.center),
    ...objects.filter((o) => o.placed).map((o) => o.center || o.t),
  ];
  const all = [...marks, ...objects.flatMap((o) => o.box || [])];
  const whole = bboxOf(all);
  /* How far apart the things that merely MARK a position are -- the old
   * extent, before placed geometry started counting towards it.
   *
   * The viewer needs both numbers and they are not the same number. Framing
   * and the grid want the box that holds everything, geometry included.
   * Camera pyramids and optical-axis stubs are glyphs: they say where
   * something is, not how big it is, and sizing them from a box that a
   * fifty-metre fixture is inflating draws every camera several metres long. */
  const markBox = bboxOf(marks);
  const markScale = markBox
    ? Math.max(Math.hypot(...[0, 1, 2].map((k) => markBox[1][k] - markBox[0][k])), 1e-6)
    : 0;
  let center = [0, 0, 0];
  let extent = 1;
  if (whole) {
    const [lo, hi] = whole;
    center = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    extent = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), 1e-6);
  }
  /* What "fit selection" frames: the selected object if there is one, and
   * otherwise the marked points, which are the part when a part is what is
   * being positioned inside a fixture. Having this separate from ``extent`` is
   * the other half of the fixture problem -- the whole-scene box is the right
   * thing to size a grid from and the wrong thing to open the view on. */
  const selBox = (objects.find((o) => o.selected && o.box) || {}).box;
  const focus = selBox || bboxOf(points) || whole;
  /* The part's own zero, for the viewer to draw the way the photo overlay
   * does. Sent only when the nominals have actually pinned the frame, so the
   * viewer never has to decide whether the origin means anything -- and only
   * while the Origin toggle is on, so one checkbox governs both views. */
  const ps = partSpaceState();
  const origin = state.showOrigin && ps.ok ? partOriginAxis() : null;
  if (origin) {
    const sph = originSphere();
    // metres, like everything else the viewer is handed
    if (sph) origin.sphere = { r: sph.world, grade: sph.grade, stale: sph.stale,
                               label: `${fmtRadius(sph.radius)} ${sph.units}` };
  }

  /* Datum surfaces, converted out of the units they were typed in.
   *
   * The panel takes them in nominal units because that is what the drawing
   * states; the viewer works in world metres like everything else here. Doing
   * the conversion once, here, is what keeps a plane at "Z = 995" landing in
   * the same place as a point nominal of 995. */
  const nominalScale = state.project.nominal_scale || 0.001;
  const planes = (state.project.reference_planes || []).map((pl) => {
    const raw = pl.normal || [0, 0, 1];
    const len = Math.hypot(raw[0], raw[1], raw[2]) || 1;
    const n = raw.map((v) => v / len);
    return {
      id: pl.id,
      label: pl.label || '',
      normal: n,
      origin: n.map((v) => v * (pl.offset || 0) * nominalScale),
      size: pl.size != null ? pl.size * nominalScale : null,
    };
  });

  return { points, cameras, objects, planes, center, extent, markScale, focus, origin };
}

/** In model mode the viewer works in the model's own units, so the scene it
 *  needs for grid spacing and framing is the model's own bounding box. */
function templateScene() {
  const mesh = meshFor(state.activeTemplate);
  if (!mesh) return { points: [], cameras: [], objects: [], center: [0, 0, 0], extent: 1 };
  return {
    points: [],
    cameras: [],
    objects: [],
    center: mesh.centroid,
    extent: Math.max(Math.hypot(...mesh.size), 1e-6),
    // in model mode the geometry IS the selection, which is the useful thing
    // to frame when the file's own origin sits far off the body
    focus: mesh.bounds || null,
  };
}

function syncViewer() {
  const t = activeTemplate();
  Viewer3D.setTemplate(
    t
      ? {
          id: t.id,
          mesh: meshFor(state.activeTemplate),
          anchors: (t.anchors || []).map((a) => ({ name: a.name, xyz: a.xyz })),
          optical: t.optical
            ? { center: t.optical.center, forward: t.optical.forward }
            : null,
        }
      : null
  );
  if (Viewer3D.isOn()) Viewer3D.render();
  updateViewerBar();
}

function updateViewerBar() {
  const model = Viewer3D.mode() === 'template';
  const t = activeTemplate();
  $('view3dtitle').textContent = model ? `Model: ${t ? t.name || t.id : 'none'}` : 'Scene';
  $('v3-scene').classList.toggle('on', !model);
  $('v3-template').classList.toggle('on', model);
  const trackball = Viewer3D.orbitMode() === 'trackball';
  $('v3-orbit').textContent = trackball ? 'trackball' : 'turntable';
  $('v3-orbit').classList.toggle('on', trackball);
  // Only shown once it is non-zero. A roll of 0 is the state the view is in
  // almost all the time, and a permanent "roll 0" readout would be noise --
  // whereas a view that has been rolled and then left that way is exactly the
  // thing that later looks like a broken render, so it says so.
  /* The roll readout and the button that undoes it are both meaningless at
   * zero, which is where the view sits almost all the time -- so they are not
   * drawn at all until there is a roll to report. A view left rolled is
   * exactly the thing that later looks like a broken render, so when there IS
   * one it says so and offers the way out, right next to it. */
  const roll = Viewer3D.rollDegrees();
  const rolled = Math.abs(roll) >= 0.5;
  $('v3-roll').hidden = !rolled;
  $('v3-level').hidden = !rolled;
  if (rolled) $('v3-roll').textContent = `roll ${roll.toFixed(0)}°`;
}

function toggleViewer(on) {
  $('view3d').hidden = !on;
  $('btn-3d').classList.toggle('on', on);
  Viewer3D.setVisible(on);
  if (on) {
    Viewer3D.resize();
    Viewer3D.frameAll();
  }
  updateViewerBar();
}

function openTemplatePicker() {
  if (!meshFor(state.activeTemplate)) {
    setStatus('this model has no geometry to pick corners from', 'warn');
    return;
  }
  syncViewer();
  toggleViewer(true);
  Viewer3D.setMode('template');
  updateViewerBar();
  setStatus('click the model’s corners to add anchors, then mark them in the photos', 'ok');
}

/** Add an anchor at a point on the model, from whatever offered it. */
function addAnchor(xyz, label) {
  const t = activeTemplate();
  if (!t || !xyz) return null;
  const used = new Set((t.anchors || []).map((a) => a.name));
  let i = 1;
  while (used.has(`a${i}`)) i++;
  const name = `a${i}`;
  t.anchors = t.anchors || [];
  t.anchors.push({ name, xyz: xyz.map(Number) });
  state.activeAnchor = name;
  markDirty();
  renderObjectsPanel();
  syncViewer();
  render();
  const left = 3 - t.anchors.length;
  setStatus(
    `anchor ${name}${label ? ` (${label})` : ''} at `
      + `${xyz.map((v) => (+v).toFixed(1)).join(', ')} ${t.units} — `
      + (left > 0 ? `${left} more needed for a placement` : 'now mark it in the photos'),
    'ok'
  );
  return name;
}

/* Quality order for the points a model offers. A bored-hole or bolt-head
 * centre reads the same from every angle, so it is worth more than a filleted
 * corner, which is at least well defined; a sharp vertex is worth least,
 * because a part with fasteners on it has dozens of identical ones and no way
 * to say in a photograph which was clicked. */
const PICK_RANK = { centre: 0, virtual: 1, corner: 2 };
const PICK_WEIGHT = { centre: 1.0, virtual: 0.85, corner: 0.7 };

/** Adopt the model's own best features as anchors, spread over the body.
 *
 *  The per-photo loop -- align by eye, snap the anchors, move to the next
 *  photo, solve -- cannot start at all until a model has anchors, and picking
 *  them by hand in the 3D view is a detour taken before any of the value
 *  appears. The model has already offered candidates; this takes them.
 *
 *  Spread is the thing to optimise, not count. Anchors bunched together pin
 *  the body's position and barely its rotation, and three in a line leave it
 *  free to spin about that line however well they are marked. So this is
 *  farthest-point sampling, seeded with the best feature on the part and
 *  weighted so a good kind of feature wins a near tie with a distant poor one. */
function autoAnchors(tid, want = 8) {
  const t = templates()[tid];
  const mesh = meshFor(tid);
  const picks = (mesh && mesh.picks) || [];
  if (!t || !picks.length) return 0;
  t.anchors = t.anchors || [];
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

  /* Idempotent, because this button invites a second press and used to reward
   * one with a second copy of every anchor. The copies took fresh names, so
   * nothing on screen looked wrong -- the list simply grew, and every anchor
   * the user then snapped was recorded against each copy at the identical
   * pixel. Measured on a nine-photograph cell: a template pressed four times
   * reached 32 anchors at 8 distinct positions, so each click voted four times.
   * That is past what the (ref/N)^0.25 count normalisation softens, and out of
   * reach of the outlier cull, which can drop one copy of a bad click and
   * leave three. Worse, it is invisible in every readout -- the body's shape
   * error and implied scale are unchanged by duplication, because they are
   * properties of the marks. So a position the model has already given is not
   * offered twice. */
  const span = picks.length > 1
    ? Math.max(...picks.map((a) => Math.max(...picks.map((b) => dist(a.xyz, b.xyz)))))
    : 1;
  const same = Math.max(span * 1e-4, 1e-9);
  const held = t.anchors.map((a) => a.xyz);
  const pool = picks
    .filter((c) => !held.some((h) => dist(c.xyz, h) <= same))
    .sort((a, b) => (PICK_RANK[a.kind] ?? 9) - (PICK_RANK[b.kind] ?? 9) || b.score - a.score);

  /* Spread is measured against the anchors already on the template as well as
   * the ones being added, so topping a part back up after deleting a few puts
   * the new ones in the gaps rather than wherever the part is emptiest. */
  const chosen = [];
  while (held.length + chosen.length < want && chosen.length < pool.length) {
    const against = held.concat(chosen.map((c) => c.xyz));
    let best = null, bestScore = -Infinity;
    for (const c of pool) {
      if (chosen.includes(c)) continue;
      const near = against.length
        ? Math.min(...against.map((k) => dist(c.xyz, k)))
        : Infinity;
      const s = near * (PICK_WEIGHT[c.kind] ?? 0.7);
      if (s > bestScore) { bestScore = s; best = c; }
    }
    if (!best) break;
    chosen.push(best);
  }

  if (!chosen.length) {
    setStatus(
      held.length
        ? `this model's ${held.length} anchor(s) already are the points it offers — `
          + 'delete one to have it picked again, or add your own in the 3D view (D)'
        : 'this model offers no anchor points to take',
      'warn'
    );
    return 0;
  }

  const used = new Set(t.anchors.map((a) => a.name));
  let i = 1;
  for (const c of chosen) {
    while (used.has(`a${i}`)) i++;
    const name = `a${i}`;
    used.add(name);
    t.anchors.push({ name, xyz: c.xyz.map(Number) });
  }
  const kinds = chosen.reduce((m, c) => (m[c.kind] = (m[c.kind] || 0) + 1, m), {});
  markDirty();
  renderObjectsPanel();
  syncViewer();
  render();
  setStatus(
    `${chosen.length} anchor(s) taken from the model `
      + `(${Object.entries(kinds).map(([k, n]) => `${n} ${k}`).join(', ')}) — `
      + `pose it over a photo and snap them, or open the 3D view (D) to check them`,
    'ok'
  );
  return chosen.length;
}

/** One of the points the model offered, picked in the 3D view. */
function addAnchorFromPick(index) {
  const mesh = meshFor(state.activeTemplate);
  const pick = mesh && (mesh.picks || [])[index];
  if (!pick) return;
  addAnchor(pick.xyz, pick.kind === 'corner' ? null : pick.kind);
}

/** A point you measured rather than picked.
 *
 *  Needed because the best anchor is not always on the mesh: a drawing gives a
 *  datum the model does not carry, and a CAD export can round off exactly the
 *  corner you want. */
async function addAnchorByCoordinates() {
  const t = activeTemplate();
  if (!t) { setStatus('select a model first', 'warn'); return; }
  const raw = await askText(
    `Anchor position on the model, in ${t.units}.\n\n`
      + `Three numbers "x y z", in the model's own coordinates — the same frame `
      + `the picked anchors are listed in.`,
    '0 0 0'
  );
  if (raw === null) return;
  const nums = raw.trim().split(/[\s,]+/).map(Number);
  if (nums.length !== 3 || nums.some((v) => !Number.isFinite(v))) {
    setStatus('could not read that as three numbers', 'warn');
    return;
  }
  addAnchor(nums, 'typed');
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ------------------------------------------------------------------- startup */

for (const b of $('modes').children) b.onclick = () => setMode(b.dataset.mode);
for (const b of $('axes').children) {
  b.onclick = () => {
    if (b.dataset.axis !== 'group') { setAxis(+b.dataset.axis); return; }
    // the toolbar button picks up where the panel left off, or makes the first
    // group so that one click is enough to start marking
    const ids = dirGroups().map((g) => g.id);
    if (!ids.length) newDirGroup();
    else setDirGroup(state.dirGroup && ids.includes(state.dirGroup) ? state.dirGroup : ids[0]);
  };
}
$('btn-help').onclick = () => openDocs();
$('btn-save').onclick = saveNow;
$('btn-undo').onclick = doUndo;
$('btn-redo').onclick = doRedo;
$('showshaded').onchange = (e) => { state.showShaded = e.target.checked; render(); };
$('btn-solve').onclick = solveNow;
$('btn-export').onclick = exportNow;
$('btn-step').onclick = (ev) => exportStepNow(ev.shiftKey);
$('showvp').onchange = (e) => { state.showVP = e.target.checked; render(); };
$('showresid').onchange = (e) => { state.showResid = e.target.checked; updateHint(); render(); };
$('showexpected').onchange = (e) => { state.showExpected = e.target.checked; render(); };
$('showgrid').onchange = (e) => { state.showGrid = e.target.checked; render(); };
/* Ticked on by default, because once a project is in part space the origin is
 * the thing worth looking at -- and in a project with no nominals it draws
 * nothing, so it costs those projects nothing. Ticking it there is a fair
 * question to ask, though, and silence would look like a broken checkbox. */
$('showorigin').onchange = (e) => {
  state.showOrigin = e.target.checked;
  if (e.target.checked) {
    const ps = partSpaceState();
    if (!ps.ok) {
      setStatus(ps.n < 3
        ? `the origin needs three nominal'd points, not in a straight line; `
          + `this project has ${ps.n}`
        : `the ${ps.n} nominal'd points are in a straight line, which leaves a `
          + `rotation free, so there is no part origin to draw yet`, 'warn');
    }
  }
  render();
  // the photo canvas and the 3D view are two renderers; this toggle is the
  // only one that governs both, so it has to ask both to repaint
  if (Viewer3D.isOn()) Viewer3D.render();
};
$('showmodels').onchange = (e) => { state.showModels = e.target.checked; render(); };
$('btn-project').onclick = (ev) => { ev.stopPropagation(); toggleProjectMenu(); };
$('fileinput').onchange = (ev) => {
  if (ev.target.files.length) uploadFiles(ev.target.files);
  ev.target.value = ''; // so picking the same file twice still fires
};
$('modelinput').onchange = (ev) => {
  if (ev.target.files.length) uploadModel(ev.target.files[0]);
  ev.target.value = '';
};
$('calibinput').onchange = (ev) => {
  if (ev.target.files.length) importCalibration(ev.target.files[0]);
  ev.target.value = '';
};

/* ---- 3D view */
$('btn-3d').onclick = () => toggleViewer(!Viewer3D.isOn());
$('v3-close').onclick = () => toggleViewer(false);
$('v3-fit').onclick = () => Viewer3D.frameAll();
function fitSelection() {
  if (!Viewer3D.frameSelection()) {
    setStatus('nothing selected to frame — pick an object in Known objects', 'warn');
  }
}
$('v3-axis').onchange = (ev) => {
  // one picker for the two things that reframe without moving anything: what
  // to look at, and where to look from
  if (ev.target.value === 'fitsel') fitSelection();
  else Viewer3D.setAxisView(ev.target.value);
  ev.target.value = '';          // a momentary action, not a persistent state
  ev.target.blur();
};
$('v3-orbit').onclick = () => {
  Viewer3D.setOrbitMode(Viewer3D.orbitMode() === 'trackball' ? 'turntable' : 'trackball');
  updateViewerBar();
};
$('v3-level').onclick = () => Viewer3D.levelRoll();
$('v3-scene').onclick = () => { Viewer3D.setMode('scene'); updateViewerBar(); };
$('v3-template').onclick = () => openTemplatePicker();

Viewer3D.attach($('canvas3d'), {
  getScene: () => (Viewer3D.mode() === 'template' ? templateScene() : buildScene()),
  onPickCamera: (id) => {
    toggleViewer(false);
    selectImage(id, true);
    setStatus(`jumped to ${(state.meta[id] || {}).name || id}`, 'ok');
  },
  onPickObject: (id) => {
    state.activeInstance = id;
    const inst = activeInstance();
    if (inst) state.activeTemplate = inst.template;
    renderPanels();
    syncViewer();
  },
  onViewChange: () => updateViewerBar(),
  onPickCandidate: (index) => addAnchorFromPick(index),
  onPickAnchor: (name) => {
    state.activeAnchor = name;
    renderObjectsPanel();
    syncViewer();
    updateHint();
  },
});
document.addEventListener('click', (ev) => {
  const menu = $('projmenu');
  if (!menu.hidden && !$('projwrap').contains(ev.target)) menu.hidden = true;
  const show = $('showmenu');
  if (!show.hidden && !$('showwrap').contains(ev.target)) {
    show.hidden = true;
    $('btn-show').classList.remove('on');
  }
});

/** The overlay menu.
 *
 *  Its contents are the same checkboxes as before, with the same ids, so every
 *  shortcut that reaches them by `$('showvp').click()` and every read of
 *  `.checked` goes on working untouched -- a hidden checkbox is still a
 *  checkbox. Only where they are drawn has changed. */
$('btn-show').onclick = (ev) => {
  ev.stopPropagation();
  const menu = $('showmenu');
  menu.hidden = !menu.hidden;
  $('btn-show').classList.toggle('on', !menu.hidden);
  if (!menu.hidden) $('projmenu').hidden = true;
};

/* What is on, in the button's tooltip: the labels left the toolbar, so the
 * answer to "why can I not see the residuals" has to stay reachable without
 * opening anything. */
function updateShowButton() {
  const on = [...$('showmenu').querySelectorAll('input[type=checkbox]')]
    .filter((b) => b.checked)
    // the label's own text nodes only: the checkbox and the shortcut chip are
    // elements, so this cannot chew a letter out of the name the way stripping
    // the chip's text by value did ("VP rays V" lost its first V)
    .map((b) => [...b.parentElement.childNodes]
      .filter((n) => n.nodeType === Node.TEXT_NODE)
      .map((n) => n.textContent)
      .join(' ')
      .trim());
  $('btn-show').title = on.length
    ? `Drawing on the photo: ${on.join(', ')}. Click to change.`
    : 'Nothing is drawn on the photo. Click to change.';
}
$('showmenu').addEventListener('change', updateShowButton);
updateShowButton();
$('projectinput').onchange = (ev) => {
  if (ev.target.files.length) openProjectFile(ev.target.files[0]);
  ev.target.value = '';
};

window.addEventListener('resize', resizeCanvas);
window.addEventListener('beforeunload', (ev) => {
  // in static mode the photos live in this tab and nowhere else, so anything
  // marked at all is worth warning about, not only unsaved edits
  const atRisk = state.dirty
    || (Backend.isStatic() && state.project && state.project.images.length);
  if (atRisk) { ev.preventDefault(); ev.returnValue = ''; }
});

/** The launch screen, shown only on the static build.
 *
 *  Deliberately a button rather than booting on arrival: 20 MB should be the
 *  visitor's decision, and the pause wants explaining before it happens rather
 *  than after. */
function showLauncher() {
  return new Promise((resolve) => {
    const el = $('launcher');
    const btn = $('launchbtn');
    const prog = $('launchprog');
    const note = $('launchnote');
    el.hidden = false;
    btn.onclick = async () => {
      btn.disabled = true;
      try {
        await Backend.boot((msg, frac) => {
          note.textContent = msg;
          prog.value = Math.max(1, Math.round(frac * 100));
        });
        el.hidden = true;
        resolve();
      } catch (err) {
        note.innerHTML = `<span style="color:var(--bad)">${escapeHtml(err.message)}</span>`;
        prog.value = 0;
        btn.disabled = false;
      }
    };
  });
}

async function start() {
  setMode('lines');
  setAxis(0);
  resizeCanvas();
  if (Backend.isStatic()) await showLauncher();
  try {
    await loadState();
    initSectionFolding();
    // fit once the grid has actually laid the canvas out, otherwise the first
    // fitView measures a stale (usually too small) rect
    requestAnimationFrame(() => { resizeCanvas(); fitView(); render(); });
    setStatus(
      Backend.isStatic()
        ? 'ready — drop photos and a CAD model onto the canvas'
        : 'ready'
    );
  } catch (err) {
    setStatus(`could not load project: ${err.message}`, 'bad');
  }
}

start();
