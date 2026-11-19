/* MultiSpy 3D viewer.
 *
 * Two layers, each doing what it is good at and neither pulling in a library.
 *
 * A 2D canvas draws the annotations -- camera frustums, tie points, anchors,
 * pick candidates, reference planes, the origin triad and every piece of text.
 * These are a few hundred glyphs that need to be legible rather than correctly
 * occluded, and a 2D canvas draws them in one batched Path2D per colour.
 *
 * A WebGL canvas underneath draws the geometry: the ground grid and every
 * triangle and every feature edge of every placed CAD body, depth-tested. That
 * layer is new, and it is new because the 2D canvas could not do it -- a real
 * part is a quarter of a million triangles, a canvas fill() is per triangle,
 * and there is no depth buffer, so a body could only be shown by discarding
 * most of it. See the `surfaces` section below for what that cost. Without a
 * GL context the 2D layer draws the thinned wireframe exactly as it used to,
 * so the view still works; it just cannot show a surface.
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
  const MAX_EDGES = 12000;
  /* Wireframe edges the *2D fallback* draws per frame, shared out between the
   * placed objects. It applies only when there is no GL context: the GL layer
   * draws every edge the server sent, which for a real part is a hundred
   * thousand of them and one draw call.
   *
   * Past the ceiling the list is truncated rather than strided. That is the
   * right way round now only because the server sorts feature edges
   * longest-first, so the first N are the N that carry the silhouette. Before
   * it did, a prefix would have been a fragment of one end of the model and a
   * stride was the lesser evil. */

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
    plane: '#d2aa50',
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
    /* Whether bodies are drawn as surfaces on the GPU. On by default: a
     * wireframe of a boxy part reads as either of two orientations, and the
     * whole reason to look at a solved scene in 3D is to see that a body was
     * placed the right way round. Turning it off falls back to the wireframe,
     * which is also what happens on its own when there is no GL context. */
    solid: true,
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

  /* ------------------------------------------------------------- surfaces
   *
   * The solid layer, on the GPU.
   *
   * Everything above this draws into a 2D canvas, and that is still the right
   * tool for what it draws: a few hundred glyphs, some text, and a hit test.
   * It is the wrong tool for a CAD body. A production part is a quarter of a
   * million triangles; a 2D canvas needs one fill() per triangle and has no
   * depth buffer, so the only way it could show one at all was to throw most
   * of it away -- 6000 of 329,000 feature edges and no surface whatsoever.
   * What reached the screen was a scatter of disconnected lines, which is why
   * a placed body looked like a few primitive lines rather than like the part.
   *
   * So the surfaces are drawn by WebGL, under the 2D canvas, sharing its
   * projection exactly. Still no library: a lambert shader and two buffers is
   * about two hundred lines, and pulling in three.js to draw untextured
   * triangles would be a 600 KB dependency in a page that has none.
   *
   * The division of labour, and why:
   *
   *   GL   the background, the ground grid, every triangle of every body and
   *        every feature edge -- all depth-tested, so a body occludes what is
   *        behind it and a hidden line is actually hidden. That last part is
   *        what makes the view readable: a wireframe of a boxy part is a
   *        Necker cube and reads as either orientation, which is how a body
   *        ends up fitted half a turn out behind a residual that raised no
   *        objection.
   *   2D   cameras, tie points, anchors, pick candidates, reference planes,
   *        the origin triad and all text, on top and deliberately not
   *        depth-tested. These are annotations: a label you cannot read
   *        because the part is in front of it has failed at its job, and a
   *        marked anchor is worth seeing through the body it sits on.
   *
   * If the context cannot be had -- an old browser, a blocklisted driver, a
   * headless run -- `gpu.ready()` stays false and the 2D path draws the
   * wireframe exactly as it did before. Nothing here is needed to see the
   * scene; it is needed to see the part.
   */

  const SOLID_VS = [
    'attribute vec3 a_pos;',
    'attribute vec3 a_nrm;',
    'uniform mat4 u_vp;',
    'uniform mat3 u_rot;',
    'uniform vec3 u_t;',
    'uniform float u_scale;',
    'varying vec3 v_nrm;',
    'varying vec3 v_world;',
    'void main() {',
    '  vec3 w = u_rot * (a_pos * u_scale) + u_t;',
    '  v_world = w;',
    '  v_nrm = u_rot * a_nrm;',
    '  gl_Position = u_vp * vec4(w, 1.0);',
    '}',
  ].join('\n');

  /* Two-sided, because an STL's winding is not to be trusted: a mesh that
   * came out of a converter with its triangles inverted would otherwise
   * render as a black hole, and nothing on screen would distinguish that from
   * a solve that put the body somewhere dark. Flipping the normal towards the
   * eye costs one dot product and removes the whole failure mode.
   *
   * Key light from over the viewer's shoulder, which is the direction the
   * photo overlay lights its fill from, so a body does not change shade when
   * you move between the two views. Then a weak sky term from +Z so downward
   * faces are dark rather than black, and a rim term so a silhouette stays
   * legible against another body of the same colour behind it.
   */
  const SOLID_FS = [
    'precision mediump float;',
    'uniform vec3 u_color;',
    'uniform vec3 u_eye;',
    'varying vec3 v_nrm;',
    'varying vec3 v_world;',
    'void main() {',
    '  vec3 toEye = normalize(u_eye - v_world);',
    '  vec3 n = normalize(v_nrm);',
    '  if (dot(n, toEye) < 0.0) n = -n;',
    '  vec3 key = normalize(vec3(0.35, -0.5, 0.79));',
    '  float lam = 0.17 + 0.58 * max(dot(n, key), 0.0)',
    '                   + 0.17 * (0.5 + 0.5 * n.z);',
    '  float rim = pow(1.0 - max(dot(n, toEye), 0.0), 3.0);',
    '  gl_FragColor = vec4(u_color * lam + vec3(rim * 0.20), 1.0);',
    '}',
  ].join('\n');

  const LINE_VS = [
    'attribute vec3 a_pos;',
    'uniform mat4 u_vp;',
    'uniform mat3 u_rot;',
    'uniform vec3 u_t;',
    'uniform float u_scale;',
    'void main() {',
    '  gl_Position = u_vp * vec4(u_rot * (a_pos * u_scale) + u_t, 1.0);',
    '}',
  ].join('\n');

  const LINE_FS = [
    'precision mediump float;',
    'uniform vec3 u_color;',
    'uniform float u_alpha;',
    'void main() { gl_FragColor = vec4(u_color, u_alpha); }',
  ].join('\n');

  /** Normals are averaged per vertex on the server, which is right across a
   *  curve and wrong across a crease -- a box would shade like a beach ball.
   *  This is where that is settled: a triangle corner keeps the smooth normal
   *  only while it agrees with its own triangle's flat normal to within this
   *  angle, and takes the flat one otherwise. Cylinders stay round, edges
   *  stay sharp, and it costs nothing per frame because the choice is baked
   *  into the buffer. */
  const SMOOTH_COS = Math.cos((33 * Math.PI) / 180);

  const IDENT3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const ZERO3 = [0, 0, 0];

  /** Body fills and edge colours, as the shader wants them: linear 0..1
   *  triples rather than the CSS hex the 2D layer uses. The hues are the same
   *  ones, so a body is the same colour whichever renderer drew it. */
  const RGB = {
    mesh: [0.42, 0.72, 0.90],
    sel: [1.00, 0.82, 0.29],
    locked: [0.55, 0.92, 0.72],
    part: [1.00, 0.72, 0.42],
    edge: [0.05, 0.07, 0.10],
    edgeSel: [0.18, 0.12, 0.02],
    grid: [1.00, 1.00, 1.00],
    axis: [[1.0, 0.35, 0.35], [0.30, 0.86, 0.50], [0.35, 0.61, 1.0]],
  };

  const gpu = {
    canvas: null,
    gl: null,
    solid: null,
    line: null,
    /* Buffers kept against the mesh payload object itself, which is the same
     * object for every instance of a template -- so twenty-eight placed
     * bodies off seven models upload seven times, not twenty-eight. A
     * WeakMap, so replacing a model's payload drops its buffers with it. */
    cache: new WeakMap(),
    failed: false,
    why: '',
    vp: null,
    eye: ZERO3,
    _scratch: null,

    ready() {
      return !!this.gl;
    },

    init(canvas) {
      if (!canvas || this.gl || this.failed) return this.ready();
      this.canvas = canvas;
      let gl = null;
      try {
        const opts = { antialias: true, alpha: false, depth: true };
        gl = canvas.getContext('webgl', opts)
          || canvas.getContext('experimental-webgl', opts);
      } catch (err) {
        gl = null;
      }
      if (!gl) {
        this.failed = true;
        this.why = 'this browser would not give out a WebGL context';
        return false;
      }
      try {
        this.solid = program(gl, SOLID_VS, SOLID_FS, ['a_pos', 'a_nrm']);
        this.line = program(gl, LINE_VS, LINE_FS, ['a_pos']);
      } catch (err) {
        this.failed = true;
        this.why = String((err && err.message) || err);
        return false;
      }
      this.gl = gl;
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.disable(gl.CULL_FACE);   // the shader is two-sided; see SOLID_FS
      return true;
    },

    buffersFor(mesh) {
      const gl = this.gl;
      let entry = this.cache.get(mesh);
      if (entry) return entry;

      const verts = mesh.vertices || [];
      const faces = mesh.faces || [];
      const smooth = mesh.normals || null;

      /* Triangles, un-indexed: one position and one normal per corner, so the
       * crease decision above can come out differently for two triangles
       * sharing a vertex. Three floats per corner more than an indexed draw,
       * and the only way to get a sharp edge out of per-vertex normals. */
      const tri = new Float32Array(faces.length * 18);
      let w = 0;
      for (let i = 0; i < faces.length; i++) {
        const f = faces[i];
        const A = verts[f[0]];
        const B = verts[f[1]];
        const C = verts[f[2]];
        if (!A || !B || !C) continue;
        const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2];
        const vx = C[0] - A[0], vy = C[1] - A[1], vz = C[2] - A[2];
        let fx = uy * vz - uz * vy;
        let fy = uz * vx - ux * vz;
        let fz = ux * vy - uy * vx;
        const fl = Math.hypot(fx, fy, fz);
        if (fl < 1e-20) continue;      // a degenerate sliver shades as noise
        fx /= fl; fy /= fl; fz /= fl;
        for (let k = 0; k < 3; k++) {
          const Pk = verts[f[k]];
          const s = smooth ? smooth[f[k]] : null;
          let nx = fx, ny = fy, nz = fz;
          if (s && s[0] * fx + s[1] * fy + s[2] * fz > SMOOTH_COS) {
            nx = s[0]; ny = s[1]; nz = s[2];
          }
          tri[w++] = Pk[0]; tri[w++] = Pk[1]; tri[w++] = Pk[2];
          tri[w++] = nx; tri[w++] = ny; tri[w++] = nz;
        }
      }

      /* Feature edges, also un-indexed. Indexed lines would halve this and
       * would need OES_element_index_uint the moment a model passed 65,535
       * vertices -- which is every real part -- so the extension would become
       * a hard requirement for the common case. A couple of megabytes is the
       * cheaper price. */
      const edges = mesh.edges || [];
      const edge = new Float32Array(edges.length * 6);
      let e = 0;
      for (let i = 0; i < edges.length; i++) {
        const A = verts[edges[i][0]];
        const B = verts[edges[i][1]];
        if (!A || !B) continue;
        edge[e++] = A[0]; edge[e++] = A[1]; edge[e++] = A[2];
        edge[e++] = B[0]; edge[e++] = B[1]; edge[e++] = B[2];
      }

      entry = {
        tri: upload(gl, tri.subarray(0, w)),
        triCount: w / 6,
        edge: upload(gl, edge.subarray(0, e)),
        edgeCount: e / 3,
      };
      this.cache.set(mesh, entry);
      return entry;
    },

    resize(width, height) {
      if (!this.canvas) return;
      this.canvas.width = width;
      this.canvas.height = height;
      if (this.gl) this.gl.viewport(0, 0, width, height);
    },

    begin(P, extent) {
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.043, 0.051, 0.063, 1);      // #0b0d10, the 2D background
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      this.vp = viewProjection(P, extent);
      this.eye = P.basis.eye;
    },

    /** One body: its triangles, then its feature edges over them.
     *
     *  The triangles are pushed a hair further from the eye with a polygon
     *  offset so that an edge lying exactly on a face wins the depth test
     *  instead of fighting it. Without that a wireframe on a curved surface
     *  stipples in and out as the view turns, which looks like a rendering
     *  fault and is really two coplanar primitives at the same depth. */
    body(mesh, R, t, unitScale, fill, edgeColor, edgeAlpha) {
      const gl = this.gl;
      const buf = this.buffersFor(mesh);

      if (buf.triCount) {
        const p = this.solid;
        gl.useProgram(p.prog);
        gl.uniformMatrix4fv(p.loc.u_vp, false, this.vp);
        gl.uniformMatrix3fv(p.loc.u_rot, false, glMat3(R));
        gl.uniform3fv(p.loc.u_t, t);
        gl.uniform1f(p.loc.u_scale, unitScale);
        gl.uniform3fv(p.loc.u_color, fill);
        gl.uniform3fv(p.loc.u_eye, this.eye);
        gl.bindBuffer(gl.ARRAY_BUFFER, buf.tri);
        gl.enableVertexAttribArray(p.loc.a_pos);
        gl.vertexAttribPointer(p.loc.a_pos, 3, gl.FLOAT, false, 24, 0);
        gl.enableVertexAttribArray(p.loc.a_nrm);
        gl.vertexAttribPointer(p.loc.a_nrm, 3, gl.FLOAT, false, 24, 12);
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(1.0, 1.0);
        gl.drawArrays(gl.TRIANGLES, 0, buf.triCount);
        gl.disable(gl.POLYGON_OFFSET_FILL);
      }

      if (buf.edgeCount && edgeAlpha > 0) {
        this.lines(buf.edge, buf.edgeCount, R, t, unitScale, edgeColor, edgeAlpha);
      }
      return buf;
    },

    lines(buffer, count, R, t, unitScale, color, alpha) {
      const gl = this.gl;
      const p = this.line;
      gl.useProgram(p.prog);
      gl.uniformMatrix4fv(p.loc.u_vp, false, this.vp);
      gl.uniformMatrix3fv(p.loc.u_rot, false, glMat3(R));
      gl.uniform3fv(p.loc.u_t, t);
      gl.uniform1f(p.loc.u_scale, unitScale);
      gl.uniform3fv(p.loc.u_color, color);
      gl.uniform1f(p.loc.u_alpha, alpha);
      if (alpha < 1) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(p.loc.a_pos);
      gl.vertexAttribPointer(p.loc.a_pos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, count);
      gl.disable(gl.BLEND);
    },

    /** A line set rebuilt every frame -- the ground grid and the axis triad,
     *  whose spacing follows the view distance. A few hundred floats a frame
     *  is not worth caching. */
    scratchLines(points, color, alpha) {
      if (!points.length) return;
      const gl = this.gl;
      if (!this._scratch) this._scratch = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this._scratch);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.DYNAMIC_DRAW);
      this.lines(this._scratch, points.length / 3, IDENT3, ZERO3, 1, color, alpha);
    },
  };

  function compile(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error('shader: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  function program(gl, vs, fs, attribs) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(prog));
    }
    const loc = {};
    for (const name of attribs) loc[name] = gl.getAttribLocation(prog, name);
    for (const name of ['u_vp', 'u_rot', 'u_t', 'u_scale', 'u_color', 'u_eye', 'u_alpha']) {
      loc[name] = gl.getUniformLocation(prog, name);
    }
    return { prog, loc };
  }

  function upload(gl, data) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buf;
  }

  /** A row-major 3x3 as the column-major 9 that `uniformMatrix3fv` wants.
   *
   *  The viewer stores rotations row-major because that is how a solved
   *  camera's R arrives from the server. GL reads columns. Getting this
   *  backwards transposes every body, which on screen is indistinguishable
   *  from a solve that placed them wrong. */
  function glMat3(R) {
    return new Float32Array([
      R[0][0], R[1][0], R[2][0],
      R[0][1], R[1][1], R[2][1],
      R[0][2], R[1][2], R[2][2],
    ]);
  }

  /** The 2D projector's own transform, as a 4x4 clip matrix.
   *
   *  Exactness matters here. The 2D canvas draws labels, anchors and pick
   *  targets on top of these triangles, and if the two projections disagree
   *  by even a pixel at the edge of the frame then an anchor marker sits off
   *  the corner it belongs to -- which is the single thing this view exists to
   *  let you check. So the terms come straight out of `projector()`:
   *
   *      x_px = f (v . right) / z + w/2   ->  ndc_x =  (2f/w) (v . right) / z
   *      y_px = f (v . down)  / z + h/2   ->  ndc_y = -(2f/h) (v . down)  / z
   *
   *  with w_clip = z, which buys the perspective divide for free. Only the
   *  depth row is new and it is the textbook one; near and far are tied to the
   *  orbit distance and the scene size so the depth buffer's precision is
   *  spent on what is actually on screen rather than on empty space behind
   *  it. */
  function viewProjection(P, extent) {
    const { eye, right, down, forward } = P.basis;
    const f = 0.5 * Math.min(P.w, P.h) / Math.tan((FOV_DEG * Math.PI) / 360);
    const sx = (2 * f) / P.w;
    const sy = -(2 * f) / P.h;
    const near = Math.max(state.dist * 0.002, 1e-7);
    const far = Math.max(near * 64, state.dist * 40 + (extent || 0) * 8);
    const A = (far + near) / (far - near);
    const B = (-2 * far * near) / (far - near);
    const ex = dot(eye, right);
    const ey = dot(eye, down);
    const ez = dot(eye, forward);
    // column-major: each group of four is one column
    return new Float32Array([
      sx * right[0], sy * down[0], A * forward[0], forward[0],
      sx * right[1], sy * down[1], A * forward[1], forward[1],
      sx * right[2], sy * down[2], A * forward[2], forward[2],
      -sx * ex, -sy * ey, B - A * ez, -ez,
    ]);
  }

  /** The ground grid and the world axis triad, in GL so that geometry
   *  standing on the grid hides the part of it that is underneath. Drawn from
   *  the same step the 2D layer labels, so the letters land on the ends of
   *  these lines. */
  function glGrid(step, hasPartOrigin) {
    const reach = step * 12;
    const cx = Math.round(state.target[0] / step) * step;
    const cy = Math.round(state.target[1] / step) * step;
    const minor = [];
    const major = [];
    for (let i = -12; i <= 12; i++) {
      const into = i % 5 === 0 ? major : minor;
      into.push(cx + i * step, cy - reach, 0, cx + i * step, cy + reach, 0);
      into.push(cx - reach, cy + i * step, 0, cx + reach, cy + i * step, 0);
    }
    gpu.scratchLines(minor, RGB.grid, 0.09);
    gpu.scratchLines(major, RGB.grid, 0.19);
    if (hasPartOrigin) return;
    for (let a = 0; a < 3; a++) {
      const end = [0, 0, 0];
      end[a] = step * 2;
      gpu.scratchLines([0, 0, 0, end[0], end[1], end[2]], RGB.axis[a], 1);
    }
  }

  /** Every placed body as a solid. One draw pair per instance, so twenty-eight
   *  bodies is fifty-six calls -- which is nothing, and is why this does not
   *  bother batching by template. */
  function renderSolids(scene) {
    if (state.mode === 'template') {
      const tpl = state.template;
      if (tpl && tpl.mesh) {
        gpu.body(tpl.mesh, IDENT3, ZERO3, 1, RGB.mesh, RGB.edge, 0.5);
      }
      return;
    }
    for (const obj of scene.objects || []) {
      if (!obj.placed || !obj.mesh || !(obj.mesh.faces || []).length) continue;
      const fill = obj.selected ? RGB.sel
        : obj.isPart ? RGB.part
        : obj.locked ? RGB.locked
        : RGB.mesh;
      gpu.body(obj.mesh, obj.R, obj.t, obj.unitScale, fill,
               obj.selected ? RGB.edgeSel : RGB.edge,
               obj.selected ? 0.75 : 0.45);
    }
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
  function drawGrid(ctx, P, hasPartOrigin, linesInGL) {
    const step = gridStep();
    if (linesInGL) {
      // GL drew the grid and the triad, depth-tested against the geometry.
      // The letters are still 2D, because GL has no text and because a label
      // is worth reading even when a body is in front of it.
      if (!hasPartOrigin) drawAxisLetters(ctx, P, step);
      return step;
    }
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

  /** The grid's spacing: a round number tied to how far the eye is, not to
   *  how big the scene is. Shared by both renderers so the axis letters land
   *  on the ends of whichever layer drew the axis lines. */
  function gridStep() {
    return Math.pow(10, Math.round(Math.log10(Math.max(state.dist, 1e-6) / 6)));
  }

  function drawAxisLetters(ctx, P, step) {
    for (let a = 0; a < 3; a++) {
      const end = [0, 0, 0];
      end[a] = step * 2;
      const p1 = P.project(end);
      if (!p1) continue;
      ctx.fillStyle = COLOR.axis[a];
      ctx.font = '12px sans-serif';
      ctx.fillText('XYZ'[a], p1.x + 4, p1.y - 4);
    }
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
    const limit = budget ? Math.min(edges.length, budget) : edges.length;
    let drawn = 0;
    for (let i = 0; i < limit; i++) {
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
    const solid = state.solid && gpu.ready();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const P = projector();
    /* With the GL layer behind it, the 2D canvas has to be see-through --
     * including its background, which GL now owns. Clearing rather than
     * filling is the whole of that change, and forgetting it would leave the
     * geometry hidden behind an opaque sheet of #0b0d10. */
    if (solid) ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    else clear(ctx, state.canvas.width, state.canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.screen = [];
    if (gpu.canvas) gpu.canvas.hidden = !solid;

    const scene = state.hooks.getScene ? state.hooks.getScene() : null;
    if (!scene) return;

    const extent = Math.max(scene.extent || 1, 1e-6);
    const partOrigin = state.mode === 'template' ? null : scene.origin;

    /* Geometry first and on its own layer, so that everything the 2D pass
     * draws below lands on top of it. The two share `P` exactly -- see
     * `viewProjection` for why that is not negotiable. */
    if (solid) {
      gpu.begin(P, extent);
      glGrid(gridStep(), !!partOrigin);
      renderSolids(scene);
    }
    drawGrid(ctx, P, !!partOrigin, solid);

    if (state.mode === 'template') {
      renderTemplate(ctx, P, extent, solid);
    } else {
      renderScene(ctx, P, scene, extent, solid);
      // after the scene, not before it: a reference the geometry hides is
      // not a reference
      if (partOrigin) drawPartOrigin(ctx, P, partOrigin);
    }
    drawLegend(ctx, P, scene);
    // the bar carries the orbit mode and the roll, both of which a drag can
    // change without touching a control
    if (state.hooks.onViewChange) state.hooks.onViewChange();
  }

  /* Reference planes: an outline, a cross through it and a stub along the
   * normal.
   *
   * The stub is there because an outline alone is ambiguous about which way a
   * plane faces, and in CAD the answer decides which side an offset lands on.
   * The cross is there because a square seen edge-on is a line, and without
   * something inside it you cannot tell a plane you are looking along from one
   * that failed to draw. */
  function drawPlanes(ctx, P, planes, extent) {
    for (const pl of planes || []) {
      const n = pl.normal;

      /* A drawn plane has four measured corners, so it is drawn as those
       * corners and not as a square about their middle. The difference is the
       * whole reason for drawing one: a fence is where its posts are, and a
       * tidy square centred on it would reach past both ends and stop short
       * of the top. The diagonals do the job the cross does below -- they
       * keep a plane seen edge-on from looking like a failure to draw. */
      if (pl.quad && pl.quad.length === 4) {
        const mid = [0, 1, 2].map(
          (i) => pl.quad.reduce((s, c) => s + c[i], 0) / 4);
        const span = Math.max(
          ...pl.quad.map((c) => Math.hypot(c[0] - mid[0], c[1] - mid[1], c[2] - mid[2])),
          1e-6);
        const verts = pl.quad.concat([
          mid, [0, 1, 2].map((i) => mid[i] + n[i] * span * 0.25),
        ]);
        drawEdges(ctx, P, verts,
          [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [4, 5]],
          COLOR.plane, 1.1);
        const sm = P.project(mid);
        if (sm && pl.label) {
          ctx.fillStyle = COLOR.plane;
          ctx.font = '11px sans-serif';
          ctx.fillText(pl.label, sm.x + 6, sm.y - 6);
        }
        continue;
      }

      const k = Math.abs(n[0]) <= Math.abs(n[1]) && Math.abs(n[0]) <= Math.abs(n[2])
        ? 0 : (Math.abs(n[1]) <= Math.abs(n[2]) ? 1 : 2);
      const seed = [0, 0, 0];
      seed[k] = 1;
      const d = seed[0] * n[0] + seed[1] * n[1] + seed[2] * n[2];
      let u = [seed[0] - n[0] * d, seed[1] - n[1] * d, seed[2] - n[2] * d];
      const ul = Math.hypot(u[0], u[1], u[2]) || 1;
      u = u.map((v) => v / ul);
      const v = [
        n[1] * u[2] - n[2] * u[1],
        n[2] * u[0] - n[0] * u[2],
        n[0] * u[1] - n[1] * u[0],
      ];
      const h = pl.size != null ? pl.size : Math.max(extent * 0.6, 1e-6);
      const at = (a, b) => [0, 1, 2].map((i) => pl.origin[i] + u[i] * a + v[i] * b);
      const verts = [at(-h, -h), at(h, -h), at(h, h), at(-h, h),
                     at(-h, 0), at(h, 0), at(0, -h), at(0, h),
                     pl.origin, [0, 1, 2].map((i) => pl.origin[i] + n[i] * h * 0.25)];
      drawEdges(ctx, P, verts,
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [6, 7], [8, 9]],
        COLOR.plane, 1.1);
      const s = P.project(pl.origin);
      if (s && pl.label) {
        ctx.fillStyle = COLOR.plane;
        ctx.font = '11px sans-serif';
        ctx.fillText(pl.label, s.x + 6, s.y - 6);
      }
    }
  }

  function renderScene(ctx, P, scene, extent, solid) {
    drawPlanes(ctx, P, scene.planes, extent);
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
      if (!solid && obj.mesh && obj.mesh.edges.length) {
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
  function renderTemplate(ctx, P, extent, solid) {
    const tpl = state.template;
    if (!tpl || !tpl.mesh) return;
    if (!solid) {
      drawEdges(ctx, P, tpl.mesh.vertices, tpl.mesh.edges, COLOR.mesh, 1, MAX_EDGES);
    }

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

  function attach(canvas, hooks, glCanvas) {
    state.canvas = canvas;
    state.ctx = canvas.getContext('2d');
    state.hooks = hooks || {};
    /* The GL canvas is optional on purpose. Passing nothing -- a test
     * harness, a page that has not been updated -- leaves `gpu.ready()`
     * false, and every draw falls back to the 2D wireframe. */
    gpu.init(glCanvas || null);

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
    /** Whether bodies are being drawn as surfaces, and whether they could be.
     *  `solidWhy()` carries the reason when they cannot, so the UI can say
     *  "no WebGL context" instead of leaving a dead checkbox. */
    solidOn: () => state.solid && gpu.ready(),
    solidWanted: () => state.solid,
    solidAvailable: () => gpu.ready(),
    solidWhy: () => gpu.why,
    setSolid(on) {
      state.solid = !!on;
      render();
    },
    /** What the GL layer actually drew for one mesh payload, for the status
     *  line and for tests: there is no other way to tell "every triangle
     *  arrived" from "the server thinned it". */
    meshCost(mesh) {
      if (!gpu.ready() || !mesh) return null;
      const buf = gpu.buffersFor(mesh);
      return { triangles: buf.triCount / 3, edges: buf.edgeCount / 2 };
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
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      state.canvas.width = w;
      state.canvas.height = h;
      // the same backing size, so the two layers stay registered to the pixel
      gpu.resize(w, h);
      render();
    },
  };
})();
