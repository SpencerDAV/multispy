/* MultiSpy 3D viewer.
 *
 * A software renderer on a 2D canvas, deliberately. The whole app is dependency
 * free and the thing being drawn is a few thousand wireframe edges, a handful of
 * camera frustums and a point cloud -- none of which needs a GPU pipeline or a
 * 600 KB library. Edges are batched into one Path2D per colour, which is what
 * actually makes this fast; depth sorting is skipped for wireframes because an
 * unsorted wireframe reads correctly anyway, and only the solid-ish marks
 * (points, frustum caps) are drawn in depth order.
 *
 * It serves two jobs:
 *
 *   scene    -- orbit the solved reconstruction: cameras, tie points, and every
 *               placed object's CAD model, which is the one view that makes a
 *               camera-array layout legible at a glance.
 *   template -- orbit one model on its own to pick anchor points off it. Marking
 *               a CAD model in photos is impossible without first saying which
 *               3D point you mean, and that has to be said in 3D.
 */

const Viewer3D = (() => {
  const FOV_DEG = 45;
  const MAX_EDGES = 9000;
  /* Total wireframe edges drawn per frame, shared out between the placed
   * objects. A CAD model arrives already reduced to feature edges by the
   * server, but five of them at once still needs a ceiling to stay smooth;
   * past it, edges are subsampled with a stride, which thins the tessellation
   * evenly instead of lopping off whole regions of the model. */

  const COLOR = {
    grid: '#ffffff18',
    gridMajor: '#ffffff30',
    axis: ['#ff5a5a', '#4ddb7f', '#5a9cff'],
    point: '#b9c0cc',
    camera: '#ff8a5c',
    cameraSel: '#ffd24a',
    mesh: '#7fd4ff',
    meshDim: '#3f7f9f',
    meshLocked: '#9affc8',
    optical: '#ffd24a',
    anchor: '#4dd0e1',
    anchorPicked: '#ffd24a',
    corner: '#8891a0',
    centre: '#8ee06a',
    virtual: '#c89bff',
  };

  const state = {
    canvas: null,
    ctx: null,
    hooks: {},
    on: false,
    mode: 'scene',
    /* Camera orientation as three orthonormal world-space vectors -- the rows
     * of a world-to-camera rotation, the same convention a solved camera's R
     * uses here.
     *
     * This replaces the yaw/pitch pair the viewer started with. A turntable
     * cannot cover the sphere: its up vector has to be chosen, it degenerates
     * where the view direction meets that up vector, and the usual patch for
     * that -- clamping pitch just short of vertical -- removes exactly the
     * views an inverted fixture is read from, straight up and straight down
     * the Z axis. Carrying the basis itself has no poles to clamp and no up
     * vector to lose, so every orientation is reachable, roll included, and
     * the two orbit gestures below are then just a choice of which axis the
     * horizontal drag turns about. */
    cam: null,       // {right, down, forward}; set by resetCam()
    orbit: 'turntable',   // 'turntable' (world Z) | 'trackball' (camera up)
    dist: 4,
    target: [0, 0, 0],
    drag: null,
    hover: null,
    screen: [],      // projected pickables for hit testing
    template: null,  // {id, mesh, anchors, optical, unitScale}
  };

  /* ------------------------------------------------------------- maths */

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm = (a) => {
    const n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  };

  /** Turn ``v`` about ``axis`` by ``ang`` (Rodrigues). The one rotation
   *  primitive the camera needs, since every gesture here is "spin the rig
   *  about some world-space line through the target". */
  function spin(v, axis, ang) {
    if (!ang) return v.slice();
    const k = norm(axis);
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const kv = cross(k, v);
    const kd = dot(k, v) * (1 - c);
    return [
      v[0] * c + kv[0] * s + k[0] * kd,
      v[1] * c + kv[1] * s + k[1] * kd,
      v[2] * c + kv[2] * s + k[2] * kd,
    ];
  }

  /** Apply a stored 3x3 (row-major) rotation: out = M @ v. */
  function rot(M, v) {
    return [
      M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
      M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
      M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
    ];
  }

  /** Pull the basis back onto the rotation group.
   *
   *  Each drag composes small rotations onto the stored basis rather than
   *  rebuilding it from angles, so without this the rows would drift out of
   *  square over a long session. Forward is kept as the authority because it
   *  is what the user is aiming, and right is squared against it; down then
   *  follows from the other two, so the frame stays right-handed by
   *  construction instead of by luck. */
  function orthonormalize() {
    const forward = norm(state.cam.forward);
    let right = sub(state.cam.right, scale(forward, dot(state.cam.right, forward)));
    if (Math.hypot(right[0], right[1], right[2]) < 1e-9) {
      // degenerate only if the stored basis was already broken; any axis not
      // parallel to forward restores a usable frame
      right = Math.abs(forward[2]) < 0.9 ? cross(forward, [0, 0, 1]) : cross(forward, [0, 1, 0]);
    }
    right = norm(right);
    state.cam = { right, down: cross(forward, right), forward };
  }

  /** Spin the whole rig about a world-space axis through the target. */
  function rotateCam(axis, ang) {
    if (!ang) return;
    const c = state.cam;
    state.cam = {
      right: spin(c.right, axis, ang),
      down: spin(c.down, axis, ang),
      forward: spin(c.forward, axis, ang),
    };
    orthonormalize();
  }

  /** The basis the old yaw/pitch pair would have produced, plus roll.
   *
   *  Kept so the default view and the named axis views can be written as the
   *  angles they obviously are. The up vector is the analytic one for this
   *  spherical frame rather than world Z, which is what lets pitch be exactly
   *  +/- 90 degrees -- the top and bottom views -- instead of the almost-90
   *  the old clamp allowed. */
  function basisFromAngles(yaw, pitch, roll) {
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const sy = Math.sin(yaw);
    const cy = Math.cos(yaw);
    const forward = [-cp * sy, cp * cy, -sp];
    // the derivative of the eye direction with respect to pitch: the way the
    // top of the frame travels as the eye climbs. Getting its sign wrong is
    // invisible at small pitch, where it still spans the right plane and only
    // the normalisation differs, and mirrors the view past 45 degrees -- so
    // the top and bottom views came out rolled through 180 degrees.
    const up = [-sp * sy, sp * cy, cp];
    const right = norm(cross(forward, up));
    state.cam = { right, down: cross(forward, right), forward };
    orthonormalize();
    if (roll) rotateCam(state.cam.forward, roll);
  }

  function resetCam() {
    basisFromAngles(0.6, 0.45, 0);
  }

  /** World is Z-up here, matching the 2D grid overlay and the advice to draw
   *  vertical edges as Z -- but only as the default and as the turntable's
   *  axis, never as a limit on where the eye may go. */
  function eyePosition() {
    if (!state.cam) resetCam();
    return sub(state.target, scale(state.cam.forward, state.dist));
  }

  function viewBasis() {
    const eye = eyePosition();
    const { right, down, forward } = state.cam;
    return { eye, right, down, forward };
  }

  /** How far the view is tilted off level, for the readout and for levelling.
   *
   *  Zero when the horizon is horizontal and world +Z projects upwards. It is
   *  undefined looking straight along Z, where there is no horizon to be level
   *  with; there the sign of the dot product is all the orientation there is,
   *  so it reports 0 and levelling leaves the view alone. */
  function rollAngle() {
    const { right, down, forward } = state.cam;
    const up = [0, 0, 1];
    const flat = sub(up, scale(forward, dot(up, forward)));
    if (Math.hypot(flat[0], flat[1], flat[2]) < 1e-6) return 0;
    return Math.atan2(dot(flat, right), -dot(flat, down));
  }

  function projector() {
    const dpr = window.devicePixelRatio || 1;
    const w = state.canvas.width / dpr;
    const h = state.canvas.height / dpr;
    const f = 0.5 * Math.min(w, h) / Math.tan((FOV_DEG * Math.PI) / 360);
    const basis = viewBasis();
    const { eye, right, down, forward } = basis;
    return {
      basis,
      w,
      h,
      /** World point -> {x, y, z} with z the view depth, or null if behind. */
      project(X) {
        const v = sub(X, eye);
        const z = dot(v, forward);
        if (z <= 1e-6) return null;
        return { x: (f * dot(v, right)) / z + w / 2, y: (f * dot(v, down)) / z + h / 2, z };
      },
    };
  }

  /* ------------------------------------------------------------ drawing */

  function clear(ctx, w, h) {
    ctx.fillStyle = '#0b0d10';
    ctx.fillRect(0, 0, w, h);
  }

  /** A ground grid on the world XY plane, stepped to a round number.
   *
   *  The step comes from how far away the eye is, not from how big the scene
   *  is. Those agree while the scene is one part, and stop agreeing the moment
   *  a two-metre fixture is dropped in beside a hundred-millimetre part: a
   *  step sized for the fixture leaves the part sitting inside a single grid
   *  cell with nothing to judge it against, at every zoom level, because the
   *  scene's size does not change when you zoom into it. Tying the step to the
   *  view distance instead means the grid is always readable at whatever is
   *  currently on screen, which is the only thing it was ever for.
   *
   *  ``hasPartOrigin`` suppresses the plain gizmo: in part space the origin is
   *  drawn properly, with a ring and a dimension, and two triads sharing one
   *  point would only make the honest one look like a rendering artefact. */
  function drawGrid(ctx, P, hasPartOrigin) {
    const step = Math.pow(10, Math.round(Math.log10(Math.max(state.dist, 1e-6) / 6)));
    const reach = step * 12;
    const cx = Math.round(state.target[0] / step) * step;
    const cy = Math.round(state.target[1] / step) * step;
    const minor = new Path2D();
    const major = new Path2D();
    const line = (a, b, path) => {
      const pa = P.project(a);
      const pb = P.project(b);
      if (!pa || !pb) return;
      path.moveTo(pa.x, pa.y);
      path.lineTo(pb.x, pb.y);
    };
    for (let i = -12; i <= 12; i++) {
      const isMajor = i % 5 === 0;
      const path = isMajor ? major : minor;
      line([cx + i * step, cy - reach, 0], [cx + i * step, cy + reach, 0], path);
      line([cx - reach, cy + i * step, 0], [cx + reach, cy + i * step, 0], path);
    }
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLOR.grid;
    ctx.stroke(minor);
    ctx.strokeStyle = COLOR.gridMajor;
    ctx.stroke(major);

    // world axis gizmo at the origin
    if (hasPartOrigin) return step;
    for (let a = 0; a < 3; a++) {
      const end = [0, 0, 0];
      end[a] = step * 2;
      const p0 = P.project([0, 0, 0]);
      const p1 = P.project(end);
      if (!p0 || !p1) continue;
      ctx.strokeStyle = COLOR.axis[a];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
      ctx.fillStyle = COLOR.axis[a];
      ctx.font = '12px sans-serif';
      ctx.fillText('XYZ'[a], p1.x + 4, p1.y - 4);
    }
    return step;
  }

  /* The part's own zero, once the nominals have pinned it.
   *
   * The gizmo above marks the solved frame's origin, which without nominals is
   * a gauge choice -- somewhere in the middle of nothing in particular. With
   * them it is the drawing's zero, so it is drawn the way the photo overlay
   * draws it: longer axes, a ring on the origin, and the axis length written
   * in the drawing's own units. Seeing the triad sit on the datum corner of
   * the solved point cloud from any angle is the check the photographs can
   * only make one at a time. */
  const GRADE_COLOR = { good: '#43d17a', warn: '#e8c33a', bad: '#e2553d', unknown: '#9aa0a6' };

  /** The origin's confidence sphere, as three great circles.
   *
   *  Wireframe rather than a shaded ball because this renderer has no depth
   *  buffer: a filled sphere would have to be sorted against the point cloud
   *  it is meant to sit among, and would hide the very axes it is annotating.
   *  Three circles read as a sphere and leave everything inside visible. */
  function drawSphere(ctx, P, r, color, dashed) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    if (dashed) ctx.setLineDash([4, 4]);
    for (let plane = 0; plane < 3; plane++) {
      const a = plane, b = (plane + 1) % 3;
      let prev = null;
      ctx.beginPath();
      for (let i = 0; i <= 72; i++) {
        const th = (i / 72) * Math.PI * 2;
        const w = [0, 0, 0];
        w[a] = r * Math.cos(th);
        w[b] = r * Math.sin(th);
        const q = P.project(w);
        if (!q) { prev = null; continue; }
        if (prev) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y);
        prev = q;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  function drawPartOrigin(ctx, P, origin) {
    const len = origin.len;
    const o = P.project([0, 0, 0]);
    if (!o) return;

    if (origin.sphere && origin.sphere.r > 0) {
      const col = GRADE_COLOR[origin.sphere.grade] || GRADE_COLOR.unknown;
      drawSphere(ctx, P, origin.sphere.r, origin.sphere.stale ? col + '88' : col,
                 origin.sphere.stale);
    }

    for (let a = 0; a < 3; a++) {
      const end = [0, 0, 0];
      end[a] = len;
      const e = P.project(end);
      if (!e) continue;
      ctx.strokeStyle = COLOR.axis[a];
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(o.x, o.y);
      ctx.lineTo(e.x, e.y);
      ctx.stroke();
      ctx.fillStyle = COLOR.axis[a];
      ctx.font = '12px sans-serif';
      ctx.fillText('XYZ'[a], e.x + 4, e.y - 4);
    }

    // As on the photographs: true-to-scale sphere, grade on the marker, since
    // a millimetre sphere in a metre scene is smaller than the dot itself.
    const grade = origin.sphere && GRADE_COLOR[origin.sphere.grade];
    ctx.strokeStyle = grade || '#ffffff';
    ctx.lineWidth = grade ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = '11px sans-serif';
    const caption = origin.sphere
      ? `0,0,0 · ${origin.L} ${origin.units} axes · ±${origin.sphere.label}`
        + (origin.sphere.stale ? ' (stale)' : '')
      : `0,0,0 · ${origin.L} ${origin.units} axes`;
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#0b0d10cc';
    ctx.strokeText(caption, o.x + 9, o.y + 16);
    ctx.fillStyle = '#ffffffdd';
    ctx.fillText(caption, o.x + 9, o.y + 16);
  }

  /** One batched stroke per colour -- the thing that keeps this interactive. */
  function drawEdges(ctx, P, verts, edges, color, width, budget) {
    const path = new Path2D();
    const stride = budget && edges.length > budget ? Math.ceil(edges.length / budget) : 1;
    let drawn = 0;
    for (let i = 0; i < edges.length; i += stride) {
      const a = P.project(verts[edges[i][0]]);
      if (!a) continue;
      const b = P.project(verts[edges[i][1]]);
      if (!b) continue;
      path.moveTo(a.x, a.y);
      path.lineTo(b.x, b.y);
      drawn++;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke(path);
    return drawn;
  }

  function drawPoints(ctx, P, points, color, radius) {
    ctx.fillStyle = color;
    for (const p of points) {
      const s = P.project(p);
      if (!s) continue;
      ctx.beginPath();
      ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /** A camera as a little pyramid, the way every 3D app draws one. */
  function frustumEdges(center, R, aspect, size, fovScale) {
    // R is world-to-camera; its rows are the camera axes in world coordinates
    const right = [R[0][0], R[0][1], R[0][2]];
    const down = [R[1][0], R[1][1], R[1][2]];
    const forward = [R[2][0], R[2][1], R[2][2]];
    const h = size * fovScale;
    const w = h * aspect;
    const tip = center;
    const corners = [
      add(add(center, scale(forward, size)), add(scale(right, -w), scale(down, -h))),
      add(add(center, scale(forward, size)), add(scale(right, w), scale(down, -h))),
      add(add(center, scale(forward, size)), add(scale(right, w), scale(down, h))),
      add(add(center, scale(forward, size)), add(scale(right, -w), scale(down, h))),
    ];
    const verts = [tip, ...corners, add(add(center, scale(forward, size)), scale(down, -h * 1.5))];
    const edges = [
      [0, 1], [0, 2], [0, 3], [0, 4],
      [1, 2], [2, 3], [3, 4], [4, 1],
      [1, 5], [2, 5], // a little "up" tab, so roll is visible
    ];
    return { verts, edges };
  }

  /* --------------------------------------------------------------- scene */

  function render() {
    if (!state.on || !state.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = state.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const P = projector();
    clear(ctx, state.canvas.width, state.canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.screen = [];

    const scene = state.hooks.getScene ? state.hooks.getScene() : null;
    if (!scene) return;

    const extent = Math.max(scene.extent || 1, 1e-6);
    const partOrigin = state.mode === 'template' ? null : scene.origin;
    drawGrid(ctx, P, !!partOrigin);

    if (state.mode === 'template') {
      renderTemplate(ctx, P, extent);
    } else {
      renderScene(ctx, P, scene, extent);
      // after the scene, not before it: a reference the wireframe hides is
      // not a reference
      if (partOrigin) drawPartOrigin(ctx, P, partOrigin);
    }
    drawLegend(ctx, P, scene);
    // the bar carries the orbit mode and the roll, both of which a drag can
    // change without touching a control
    if (state.hooks.onViewChange) state.hooks.onViewChange();
  }

  function renderScene(ctx, P, scene, extent) {
    drawPoints(ctx, P, scene.points || [], COLOR.point, 2);

    /* Glyphs are sized from how far apart the marks are, not from the scene
     * box. The two used to be the same number; they stopped being the same
     * once placed geometry started counting towards the scene box, and a
     * fixture larger than the camera walk would otherwise draw every camera
     * pyramid metres long over the cameras it is meant to mark. */
    const markScale = Math.max(scene.markScale || extent, 1e-6);
    const camSize = markScale * 0.06;
    for (const cam of scene.cameras || []) {
      const { verts, edges } = frustumEdges(
        cam.center,
        cam.R,
        cam.aspect || 1.33,
        camSize,
        cam.fovScale || 0.6
      );
      drawEdges(ctx, P, verts, edges, cam.selected ? COLOR.cameraSel : COLOR.camera, cam.selected ? 2 : 1.2);
      const s = P.project(cam.center);
      if (s) {
        state.screen.push({ kind: 'camera', id: cam.id, x: s.x, y: s.y, z: s.z });
        if (cam.selected) {
          ctx.fillStyle = COLOR.cameraSel;
          ctx.font = '11px sans-serif';
          ctx.fillText(cam.label || cam.id, s.x + 7, s.y - 7);
        }
      }
    }

    const objects = (scene.objects || []).filter((o) => o.placed);
    const budget = objects.length ? Math.floor(MAX_EDGES / objects.length) : MAX_EDGES;
    for (const obj of objects) {
      const color = obj.selected ? COLOR.anchorPicked : obj.locked ? COLOR.meshLocked : COLOR.mesh;
      if (obj.mesh && obj.mesh.edges.length) {
        const verts = obj.mesh.vertices.map((v) =>
          add(rot(obj.R, scale(v, obj.unitScale)), obj.t)
        );
        drawEdges(ctx, P, verts, obj.mesh.edges, color, obj.selected ? 1.5 : 1, budget);
      }

      // anchors always drawn, mesh or not: they are what was actually marked
      for (const a of obj.anchors || []) {
        const world = add(rot(obj.R, scale(a.xyz, obj.unitScale)), obj.t);
        const s = P.project(world);
        if (!s) continue;
        const marked = a.marked > 0;
        ctx.beginPath();
        ctx.arc(s.x, s.y, marked ? 3.5 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = marked ? COLOR.anchor : COLOR.corner;
        ctx.fill();
      }

      if (obj.optical) {
        const eye = add(rot(obj.R, scale(obj.optical.center, obj.unitScale)), obj.t);
        const axis = rot(obj.R, obj.optical.forward);
        // as long as the body it comes out of, so a 100 mm camera body keeps a
        // 100 mm stub whatever else is in the scene
        const span = obj.box
          ? Math.hypot(obj.box[1][0] - obj.box[0][0],
                       obj.box[1][1] - obj.box[0][1],
                       obj.box[1][2] - obj.box[0][2])
          : 0;
        const tip = add(eye, scale(norm(axis), span > 0 ? span * 0.6 : markScale * 0.12));
        const a = P.project(eye);
        const b = P.project(tip);
        if (a && b) {
          ctx.strokeStyle = COLOR.optical;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
        if (a) {
          ctx.fillStyle = COLOR.optical;
          ctx.beginPath();
          ctx.arc(a.x, a.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const centerWorld = obj.center || obj.t;
      const s = P.project(centerWorld);
      if (s) {
        state.screen.push({ kind: 'object', id: obj.id, x: s.x, y: s.y, z: s.z });
        ctx.fillStyle = obj.selected ? COLOR.anchorPicked : '#cfd6e2';
        ctx.font = obj.selected ? 'bold 12px sans-serif' : '11px sans-serif';
        ctx.fillText(obj.label || obj.id, s.x + 8, s.y + 4);
      }
    }
  }

  /** One model on its own, with every pickable corner shown, for choosing anchors. */
  function renderTemplate(ctx, P, extent) {
    const tpl = state.template;
    if (!tpl || !tpl.mesh) return;
    const verts = tpl.mesh.vertices;
    drawEdges(ctx, P, verts, tpl.mesh.edges, COLOR.mesh, 1, MAX_EDGES);

    // a vertex already taken as an anchor is drawn as that anchor, not as a
    // candidate, so the two never sit on top of each other
    const pickedKeys = new Set(
      (tpl.anchors || []).map((a) => a.xyz.map((v) => v.toFixed(4)).join(','))
    );

    /* What the model offers as findable in a photograph, drawn so the three
     * kinds are told apart at a glance -- because they are not equally good and
     * the difference is not visible from the geometry alone:
     *   a dot      a sharp corner, an actual vertex
     *   a ring     the centre of a round feature; the best kind, and no vertex
     *   a cross    where two edges would meet but a fillet rounded them off */
    (tpl.mesh.picks || []).forEach((pick, index) => {
      const s = P.project(pick.xyz);
      if (!s) return;
      if (pickedKeys.has(pick.xyz.map((v) => v.toFixed(4)).join(','))) return;
      state.screen.push({ kind: 'pick', index, x: s.x, y: s.y, z: s.z });
      const hot = state.hover && state.hover.kind === 'pick' && state.hover.index === index;
      const color = hot ? '#fff' : COLOR[pick.kind] || COLOR.corner;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = hot ? 2 : 1.4;

      if (pick.kind === 'centre') {
        const r = hot ? 7 : 5;
        ctx.beginPath();
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(s.x, s.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
      } else if (pick.kind === 'virtual') {
        const r = hot ? 7 : 5;
        ctx.beginPath();
        ctx.moveTo(s.x - r, s.y); ctx.lineTo(s.x + r, s.y);
        ctx.moveTo(s.x, s.y - r); ctx.lineTo(s.x, s.y + r);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(s.x, s.y, hot ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    for (const a of tpl.anchors || []) {
      const s = P.project(a.xyz);
      if (!s) continue;
      state.screen.push({ kind: 'anchor', name: a.name, x: s.x, y: s.y, z: s.z });
      const hot = state.hover && state.hover.kind === 'anchor' && state.hover.name === a.name;
      ctx.beginPath();
      ctx.arc(s.x, s.y, hot ? 7 : 5, 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.anchorPicked;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = COLOR.anchorPicked;
      ctx.font = '11px sans-serif';
      ctx.fillText(a.name, s.x + 8, s.y - 6);
    }

    if (tpl.optical) {
      const eye = tpl.optical.center;
      const tip = add(eye, scale(norm(tpl.optical.forward), extent * 0.4));
      const a = P.project(eye);
      const b = P.project(tip);
      if (a && b) {
        ctx.strokeStyle = COLOR.optical;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLOR.optical;
        ctx.font = '11px sans-serif';
        ctx.fillText('optical axis', b.x + 5, b.y);
      }
    }
  }

  function drawLegend(ctx, P, scene) {
    if (state.mode === 'template') {
      const counts = (state.template && state.template.mesh && state.template.mesh.pick_counts) || {};
      const key = [
        ['centre', `ring  ${counts.centre || 0} round-feature centres — best`],
        ['virtual', `cross ${counts.virtual || 0} corners rounded off by a fillet`],
        ['corner', `dot   ${counts.corner || 0} sharp corners`],
      ];
      ctx.font = '11px ui-monospace, monospace';
      key.forEach(([kind, text], i) => {
        ctx.fillStyle = COLOR[kind];
        ctx.fillText(text, 10, P.h - 24 - (key.length - 1 - i) * 14);
      });
      ctx.fillStyle = '#8b93a1';
      ctx.font = '11px sans-serif';
      ctx.fillText('click one to add it as an anchor', 10, P.h - 8);

      // what is under the cursor, since the kinds are not equally useful and a
      // size is worth seeing before committing to a point
      const hovered =
        state.hover && state.hover.kind === 'pick' && state.template && state.template.mesh
          ? state.template.mesh.picks[state.hover.index]
          : null;
      if (hovered) {
        const label = hovered.note || hovered.kind;
        ctx.fillStyle = '#e4e7ec';
        ctx.fillText(label, state.hover.x + 12, state.hover.y - 10);
      }
      return;
    }
    const lines = [
      `${(scene.cameras || []).length} photo camera(s)`,
      `${(scene.objects || []).filter((o) => o.placed).length} placed object(s)`,
      ...(scene.origin ? ["0,0,0 is the drawing's own origin"] : []),
      ...(scene.origin && scene.origin.sphere
          ? ['the sphere is where the origin sits 95% of the time'] : []),
      'click a camera to jump to that photo',
    ];
    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#8b93a1';
    lines.forEach((text, i) => ctx.fillText(text, 10, P.h - 10 - (lines.length - 1 - i) * 14));
  }

  /* --------------------------------------------------------- interaction */

  function pick(ev) {
    const rect = state.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    let best = null;
    for (const s of state.screen) {
      const d = Math.hypot(s.x - x, s.y - y);
      if (d <= 11 && (!best || s.z < best.z)) best = { ...s, d };
    }
    return best;
  }

  function frameAll() {
    const scene = state.hooks.getScene ? state.hooks.getScene() : null;
    if (state.mode === 'template' && state.template && state.template.mesh) {
      const m = state.template.mesh;
      state.target = m.centroid || [0, 0, 0];
      state.dist = Math.max(Math.hypot(...(m.size || [1, 1, 1])) * 1.6, 1e-3);
    } else if (scene) {
      state.target = scene.center || [0, 0, 0];
      state.dist = Math.max((scene.extent || 1) * 1.8, 1e-3);
    }
    resetCam();
    render();
  }

  /* Frame one thing instead of everything.
   *
   * Fitting the whole scene is the wrong move as soon as the scene contains a
   * large fixture, or a model whose CAD origin sits a metre from its geometry:
   * the box that holds everything is then mostly empty space, and the part
   * actually being positioned ends up a few pixels across in the middle of it.
   * The scene supplies a focus box for whatever is selected, and this frames
   * that box and nothing else. The orientation is deliberately left alone --
   * it is a zoom, not a reset, and the view you were working from is usually
   * the view you still want. */
  function frameSelection() {
    const scene = state.hooks.getScene ? state.hooks.getScene() : null;
    const box = scene && scene.focus;
    if (!box) return false;
    const lo = box[0];
    const hi = box[1];
    state.target = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    // A box with no size -- one anchor and no mesh -- says where the thing is
    // and nothing about how big it is. Re-centre on it and leave the distance
    // alone, rather than diving to the 0.1 mm the clamp would otherwise pick.
    const span = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    if (span > 1e-9) state.dist = Math.max(span * 1.6, 1e-4);
    render();
    return true;
  }

  /** The six axis views, by where the eye is put. ``z`` looks straight down
   *  the -Z axis and ``-z`` straight up the +Z axis -- the pair the old pitch
   *  clamp could only ever approach. */
  const AXIS_VIEWS = {
    x: [Math.PI / 2, 0], '-x': [-Math.PI / 2, 0],
    y: [Math.PI, 0], '-y': [0, 0],
    z: [0, Math.PI / 2], '-z': [0, -Math.PI / 2],
  };

  function setAxisView(key) {
    const a = AXIS_VIEWS[key];
    if (!a) return false;
    basisFromAngles(a[0], a[1], 0);
    render();
    return true;
  }

  function attach(canvas, hooks) {
    state.canvas = canvas;
    state.ctx = canvas.getContext('2d');
    state.hooks = hooks || {};

    canvas.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      const hit = ev.button === 0 ? pick(ev) : null;
      if (hit) {
        if (hit.kind === 'camera' && state.hooks.onPickCamera) state.hooks.onPickCamera(hit.id);
        else if (hit.kind === 'object' && state.hooks.onPickObject) state.hooks.onPickObject(hit.id);
        else if (hit.kind === 'pick' && state.hooks.onPickCandidate) state.hooks.onPickCandidate(hit.index);
        else if (hit.kind === 'anchor' && state.hooks.onPickAnchor) state.hooks.onPickAnchor(hit.name);
        return;
      }
      state.drag = {
        x: ev.clientX,
        y: ev.clientY,
        cam: state.cam,
        target: [...state.target],
        // alt pans here because alt pans on the photograph, and one key
        // meaning "pan" on one canvas and "roll" on the other is the kind of
        // difference nobody discovers deliberately
        pan: ev.button === 1 || ev.shiftKey || ev.altKey,
        // roll gets a modifier of its own rather than a second axis on the
        // orbit drag: it is the one view angle you set once and then leave,
        // and having it ride along on every orbit would mean never being able
        // to orbit without disturbing it
        roll: !(ev.button === 1 || ev.shiftKey || ev.altKey) && (ev.ctrlKey || ev.metaKey),
      };
    });

    window.addEventListener('mousemove', (ev) => {
      if (!state.on) return;
      if (!state.drag) {
        if (state.mode !== 'template') return;
        const hit = pick(ev);
        const changed = JSON.stringify(hit) !== JSON.stringify(state.hover);
        state.hover = hit;
        state.canvas.style.cursor = hit ? 'pointer' : 'grab';
        if (changed) render();
        return;
      }
      const dx = ev.clientX - state.drag.x;
      const dy = ev.clientY - state.drag.y;
      if (state.drag.pan) {
        const { right, down } = viewBasis();
        const k = state.dist * 0.0018;
        state.target = add(
          state.drag.target,
          add(scale(right, -dx * k), scale(down, -dy * k))
        );
      } else {
        /* Replayed from the basis captured at press rather than accumulated
         * frame to frame, so the same hand position always gives the same
         * view: a drag can be walked back by moving the mouse back, which an
         * accumulating orbit cannot offer. */
        state.cam = state.drag.cam;
        if (state.drag.roll) {
          // right is clockwise, matching E; see rollBy for the negation
          rotateCam(state.cam.forward, -dx * 0.008);
        } else {
          // Horizontal turns about world Z (turntable) or about the camera's
          // own up (trackball). Vertical always turns about the camera's
          // right, taken AFTER the horizontal turn so the two compose the way
          // a hand expects. Neither is clamped: the view passes through
          // straight-down and straight-up and carries on, and at those poles
          // the horizontal drag spins the image, which is how an inverted
          // fixture gets squared up.
          const hAxis = state.orbit === 'trackball'
            ? scale(state.cam.down, -1)
            : [0, 0, 1];
          rotateCam(hAxis, dx * 0.008);
          rotateCam(state.cam.right, -dy * 0.008);
        }
      }
      render();
    });

    window.addEventListener('mouseup', () => {
      state.drag = null;
    });

    canvas.addEventListener(
      'wheel',
      (ev) => {
        ev.preventDefault();
        state.dist = Math.max(1e-4, state.dist * Math.exp(ev.deltaY * 0.0012));
        render();
      },
      { passive: false }
    );

    canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());
  }

  return {
    attach,
    render,
    frameAll,
    frameSelection,
    setAxisView,
    orbitMode: () => state.orbit,
    setOrbitMode(mode) {
      state.orbit = mode === 'trackball' ? 'trackball' : 'turntable';
      render();
    },
    /** Turn the view about the direction it is already looking.
     *
     *  Negated because spinning the rig one way tilts the world the other:
     *  ``rotateCam(forward, t)`` leaves ``rollAngle()`` reading ``-t``, so
     *  without this a roll of +15 degrees reported itself as -15 and
     *  levelling doubled the tilt instead of removing it. */
    rollBy(ang) {
      if (!state.cam) resetCam();
      rotateCam(state.cam.forward, -ang);
      render();
    },
    /** Put world +Z back towards screen-up without moving the eye. */
    levelRoll() {
      if (!state.cam) resetCam();
      rotateCam(state.cam.forward, rollAngle());
      render();
    },
    rollDegrees: () => (state.cam ? (rollAngle() * 180) / Math.PI : 0),
    /** The camera, read-only, for tests and for anything that needs to know
     *  where the eye actually is. Copies out, so nothing can reach in and
     *  leave the basis un-orthonormal behind the module's back. */
    view() {
      if (!state.cam) resetCam();
      return {
        dist: state.dist,
        target: [...state.target],
        eye: eyePosition(),
        right: [...state.cam.right],
        down: [...state.cam.down],
        forward: [...state.cam.forward],
        orbit: state.orbit,
        roll: (rollAngle() * 180) / Math.PI,
      };
    },
    isOn: () => state.on,
    setVisible(on) {
      state.on = !!on;
      if (on) render();
    },
    mode: () => state.mode,
    setMode(mode) {
      state.mode = mode;
      state.hover = null;
      frameAll();
    },
    setTemplate(template) {
      state.template = template;
      if (state.mode === 'template') render();
    },
    resize() {
      if (!state.canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = state.canvas.getBoundingClientRect();
      state.canvas.width = Math.max(1, Math.round(rect.width * dpr));
      state.canvas.height = Math.max(1, Math.round(rect.height * dpr));
      render();
    },
  };
})();
