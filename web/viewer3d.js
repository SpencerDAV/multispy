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
    yaw: 0.6,
    pitch: 0.5,
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

  /** Apply a stored 3x3 (row-major) rotation: out = M @ v. */
  function rot(M, v) {
    return [
      M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
      M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
      M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
    ];
  }

  /** World is Z-up here, matching the 2D grid overlay and the advice to draw
   *  vertical edges as Z. */
  function eyePosition() {
    const cp = Math.cos(state.pitch);
    const dir = [cp * Math.sin(state.yaw), -cp * Math.cos(state.yaw), Math.sin(state.pitch)];
    return add(state.target, scale(dir, state.dist));
  }

  function viewBasis() {
    const eye = eyePosition();
    const forward = norm(sub(state.target, eye));
    let up = [0, 0, 1];
    if (Math.abs(dot(forward, up)) > 0.999) up = [0, 1, 0];
    const right = norm(cross(forward, up));
    const down = cross(forward, right);
    return { eye, right, down, forward };
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

  /** A ground grid on the world XY plane, stepped to a round number. */
  function drawGrid(ctx, P, extent) {
    const step = Math.pow(10, Math.round(Math.log10(Math.max(extent, 1e-6) / 6)));
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
    drawGrid(ctx, P, extent);

    if (state.mode === 'template') {
      renderTemplate(ctx, P, extent);
    } else {
      renderScene(ctx, P, scene, extent);
    }
    drawLegend(ctx, P, scene);
  }

  function renderScene(ctx, P, scene, extent) {
    drawPoints(ctx, P, scene.points || [], COLOR.point, 2);

    const camSize = extent * 0.06;
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
        const tip = add(eye, scale(norm(axis), extent * 0.12));
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
    state.yaw = 0.6;
    state.pitch = 0.45;
    render();
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
        yaw: state.yaw,
        pitch: state.pitch,
        target: [...state.target],
        pan: ev.button === 1 || ev.shiftKey,
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
        state.yaw = state.drag.yaw + dx * 0.008;
        state.pitch = Math.max(-1.5, Math.min(1.5, state.drag.pitch + dy * 0.008));
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
