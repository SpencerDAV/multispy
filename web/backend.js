/* Where the UI's requests actually go.
 *
 * MultiSpy runs two ways from one codebase. Locally, a Python process serves
 * the UI and answers over HTTP. On a static host there is no process, so the
 * tab loads CPython itself -- numpy, scipy and Pillow included -- through
 * Pyodide and calls the very same `multispy` package, in a Web Worker beside
 * the page. See worker.js for why beside and not inside: a solve on the main
 * thread cannot paint while it runs, so the window freezes for its duration
 * with nothing on screen to say that it is working rather than hung.
 *
 * This is not a reimplementation of the solver. It is the same Python, running
 * a few milliseconds away instead of a few milliseconds away over a socket, and
 * it produces identical numbers. What differs is only what a browser lacks:
 *
 *   - no project directory, so saving is a download and opening is a file pick;
 *   - no folder to export into, so exports arrive as one zip;
 *   - nothing is uploaded anywhere, because there is nowhere to upload to --
 *     photos stay in the tab, which is the better property of the two.
 */

const Backend = (() => {
  const MODE = window.MULTISPY_MODE === 'pyodide' ? 'pyodide' : 'server';
  const PY_BASE = window.MULTISPY_PY_BASE || 'py/';
  const PYODIDE_URL =
    window.MULTISPY_PYODIDE_URL || 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/';

  /* The package, in dependency order. __main__ and server are deliberately
   * absent: one is a CLI and the other is the transport this replaces.
   *
   * The build states this list in config.js, because it is the build that
   * decides which files land in py/.
   *
   * There used to be a literal here as a fallback, and the comment above it
   * called it "a fallback with teeth" -- correctly, and that was the argument
   * for deleting it rather than maintaining it. It had rotted to ten names
   * against a package of fifteen, so the one circumstance it existed for, a
   * config.js too old to carry the variable, was also the circumstance in which
   * it would silently omit six modules: the page would boot, run, and then die
   * on the first import of one, while the local server -- which never reads
   * this -- carried on working perfectly. A stale cached config.js on a live
   * site is exactly that circumstance.
   *
   * So there is no fallback. Either the build told us what it shipped or we do
   * not know, and not knowing is said out loud before anything is loaded,
   * because a page that refuses to start is debuggable and a page that half
   * starts is not.
   *
   * Scoped to the static build, because only the static build loads Python into
   * the page. The local server's own config.js sets MULTISPY_MODE and nothing
   * else -- there is no module list because there is no virtual filesystem to
   * write one into -- so an unconditional check here takes the dev server down
   * at module scope, which is how this comment came to be two paragraphs long. */
  const MODULES = window.MULTISPY_PY_MODULES;
  if (MODE === 'pyodide' && (!Array.isArray(MODULES) || !MODULES.length)) {
    throw new Error(
      'config.js did not say which Python modules were shipped '
      + '(window.MULTISPY_PY_MODULES is missing or empty). This is a stale or missing '
      + 'config.js — a hard reload should fix it. Guessing the list here is what broke '
      + 'this build twice before, so it is not attempted.');
  }

  const WORKER_URL = window.MULTISPY_WORKER_URL || 'worker.js';

  let worker = null;
  let booted = false;
  let nextId = 1;
  const pending = new Map(); // id -> {resolve, reject}
  const urlCache = new Map(); // `${id}|${thumb}` -> object URL

  const isStatic = () => MODE === 'pyodide';
  const isReady = () => MODE === 'server' || booted;

  /* ----------------------------------------------------------- busy state
   *
   * Which operations are in flight, so the page can say that it is working.
   *
   * This exists because the thing it reports used to be unreportable. While
   * Python ran on the main thread nothing could be drawn during a call, so the
   * only honest indicator -- "still going, 12s so far" -- was the one piece of
   * information the architecture made impossible to show. Now that the work is
   * in a worker, the count is maintained here, once, around every request,
   * rather than remembered at each of the twenty-odd call sites.
   */

  let inflight = 0;
  let busyLabel = null;
  let busySince = 0;
  const busyWatchers = [];

  function announceBusy() {
    const state = {
      busy: inflight > 0,
      label: inflight > 0 ? busyLabel : null,
      since: inflight > 0 ? busySince : 0,
    };
    for (const fn of busyWatchers) {
      try { fn(state); } catch (err) { /* a watcher must not break a request */ }
    }
  }

  /** Be told when Python starts and stops working. */
  function onBusy(fn) {
    busyWatchers.push(fn);
    fn({ busy: inflight > 0, label: busyLabel, since: busySince });
  }

  function enter(label) {
    if (inflight === 0) { busyLabel = label; busySince = Date.now(); }
    inflight += 1;
    announceBusy();
  }

  function leave() {
    inflight = Math.max(0, inflight - 1);
    if (inflight === 0) { busyLabel = null; busySince = 0; }
    announceBusy();
  }

  /* --------------------------------------------------------------- boot */

  /** Ask the worker to do one thing, and wait for its answer.
   *
   *  `label` is what the page says it is doing while this is outstanding. */
  function send(kind, payload, transfer, label) {
    if (!worker) return Promise.reject(new Error('MultiSpy is not started'));
    const id = nextId++;
    enter(label || kind);
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, kind, ...payload }, transfer || []);
    }).finally(leave);
  }

  async function boot(onProgress) {
    if (booted) return;
    const step = (msg, frac) => onProgress && onProgress(msg, frac);

    if (!worker) {
      worker = new Worker(WORKER_URL);
      worker.onmessage = (ev) => {
        const { id, ok, result, error, kind, msg, frac } = ev.data || {};
        if (kind === 'progress') { step(msg, frac); return; }
        const slot = pending.get(id);
        if (!slot) return;
        pending.delete(id);
        if (ok) slot.resolve(result);
        else slot.reject(new Error(error));
      };
      /* A worker that dies takes every outstanding request with it, and a
       * promise that never settles is a spinner that never stops. Settle them
       * all with the reason, so the page can say what happened. */
      worker.onerror = (ev) => {
        const err = new Error(ev.message || 'the MultiSpy worker stopped');
        for (const [, slot] of pending) slot.reject(err);
        pending.clear();
      };
    }

    await send('boot', {
      pyodideUrl: PYODIDE_URL,
      // the worker resolves its own fetches against its own URL, so hand it
      // something absolute rather than a path that means two different things
      // depending on which thread reads it
      pyBase: new URL(PY_BASE, window.location.href).href,
      modules: MODULES,
    }, [], 'starting Python');
    booted = true;
    step('ready', 1);
  }


  /* ------------------------------------------------------------- calling */

  function unwrap(jsonText) {
    const data = JSON.parse(jsonText);
    if (data && data.__error__) throw new Error(data.__error__);
    return data;
  }

  /** Answer one UI request, whichever front end is behind it. */
  async function call(path, body) {
    if (!isStatic()) {
      const opts = body === undefined
        ? {}
        : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      return data;
    }

    if (!booted) throw new Error('MultiSpy is still starting up');
    // a GET in the server build; named apart so one dispatcher can serve both
    const route = body === undefined ? `GET ${path}` : path;
    return unwrap(await send('dispatch', {
      route,
      body: body === undefined ? '' : JSON.stringify(body),
    }, [], labelFor(path)));
  }

  /* What the page says it is doing, per route.
   *
   * Only the ones worth naming are named: a route that answers in a few
   * milliseconds is better left unmentioned than flashed on screen, and the
   * caller already sets its own status line. What this covers is the handful
   * that take long enough for the user to wonder, which is exactly the set
   * that used to freeze the tab. */
  const LABELS = {
    '/api/solve': 'solving',
    '/api/check_cad': 'comparing the CAD against the photographs',
    '/api/mute_bodies': 'muting bodies and re-solving',
    '/api/origin_uncertainty': 'sampling the origin (many re-solves)',
    '/api/edge_refine': 'aligning poses to the image edges',
    '/api/place_part': 'placing the part',
    '/api/set_part': 'fitting the part',
    '/api/check_nominal_shape': 'checking the nominal shape',
    '/api/viewpoints': 'working out where to shoot next',
    '/api/mesh': 'reading the model',
    '/api/rig': 'measuring the rig',
  };

  function labelFor(path) {
    return LABELS[path] || 'working';
  }

  /* --------------------------------------------------------- photo bytes */

  /** A URL the canvas and the filmstrip can use for one photo.
   *
   *  Static mode asks Python for the same downscaled JPEG the server would have
   *  sent, so EXIF orientation and display scaling are handled in exactly one
   *  place rather than two that can disagree. */
  async function photoSrc(imageId, thumb = false) {
    if (!isStatic()) {
      return `${thumb ? '/api/thumb' : '/api/photo'}?id=${encodeURIComponent(imageId)}`;
    }
    const key = `${imageId}|${thumb}`;
    if (urlCache.has(key)) return urlCache.get(key);
    const bytes = await send('photo', { imageId, thumb }, [], 'reading a photo');
    // the cache may have been filled while this was in the worker: two
    // filmstrip tiles asking for the same thumbnail used to be impossible,
    // since nothing else could run during the call
    if (urlCache.has(key)) return urlCache.get(key);
    const url = URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));
    urlCache.set(key, url);
    return url;
  }

  function forgetPhoto(imageId) {
    for (const thumb of [true, false]) {
      const key = `${imageId}|${thumb}`;
      const url = urlCache.get(key);
      if (url && url.startsWith('blob:')) URL.revokeObjectURL(url);
      urlCache.delete(key);
    }
  }

  function forgetAllPhotos() {
    for (const url of urlCache.values()) {
      if (url.startsWith('blob:')) URL.revokeObjectURL(url);
    }
    urlCache.clear();
  }

  /* ------------------------------------------------------------- uploads */

  /** Add one photo or CAD model. Returns whatever that endpoint returns. */
  async function addFileUpload(kind, file) {
    const endpoint = kind === 'photo' ? '/api/upload' : '/api/upload_model';
    if (!isStatic()) {
      const res = await fetch(`${endpoint}?name=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        headers: { 'Content-Type': file.type || 'application/octet-stream' },
        body: file,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      return data;
    }

    if (!booted) throw new Error('MultiSpy is still starting up');
    // the worker writes these into its own filesystem; the buffer is handed
    // over rather than copied, so a 40 MB model costs one allocation
    const buf = await file.arrayBuffer();
    return unwrap(await send(
      'addFile',
      { fileKind: kind, name: file.name, bytes: buf },
      [buf],
      kind === 'photo' ? 'reading the photo' : 'reading the model',
    ));
  }

  /* ------------------------------------------------- projects and exports */

  function download(filename, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  /** Static mode has nowhere to export to, so the same writers produce a zip.
   *
   *  `project` is required in server mode and ignored in static mode, where the
   *  workspace already holds the only copy. Leaving it out is how the server
   *  build's Export button came to answer `project` and write nothing: the
   *  route adopts the posted project, and a body without one is a KeyError
   *  dressed up as an error message.
   */
  async function exportAll(formats, project) {
    if (!isStatic()) return call('/api/export', { formats, project });
    if (!booted) throw new Error('MultiSpy is still starting up');
    const bytes = await send(
      'exportZip', { formats: formats ? JSON.stringify(formats) : '' }, [], 'writing the export',
    );
    download('multispy-export.zip', new Blob([bytes], { type: 'application/zip' }));
    return { written: { zip: 'multispy-export.zip' }, out_dir: 'your downloads folder' };
  }

  /** The STEP scene alone: a file next to the project, or a direct download. */
  async function exportStep(shape, project) {
    if (!isStatic()) {
      const data = await call('/api/export', {
        formats: ['step'],
        step_shape: shape,
        project,
      });
      return { ...data, file: 'scene.step' };
    }
    if (!booted) throw new Error('MultiSpy is still starting up');
    const bytes = await send(
      'exportStep', { shape: shape || 'none' }, [], 'writing the STEP scene',
    );
    download('scene.step', new Blob([bytes], { type: 'application/octet-stream' }));
    return { written: { step: 'scene.step' }, out_dir: 'your downloads folder' };
  }

  /** Download the whole workspace -- project, photos and models -- as one zip.
   *
   *  A tab has no disk, so this is what saving means here. Marks alone would
   *  reopen into a project whose photos are all missing. */
  async function saveBundle(projectName) {
    if (!isStatic()) throw new Error('not available in server mode');
    const bytes = await send('bundle', {}, [], 'packing the project');
    const safe = String(projectName || 'project').replace(/[^\w.-]+/g, '-');
    download(`${safe}.multispy.zip`, new Blob([bytes], { type: 'application/zip' }));
  }

  /** Replace the open project from a file the user picked: a zip with its
   *  photos, or a bare .multispy with just the marks. */
  async function openFile(file) {
    if (!isStatic()) throw new Error('not available in server mode');
    const isZip = /\.zip$/i.test(file.name);
    const buf = await file.arrayBuffer();
    const data = unwrap(await send(
      'openBundle', { bytes: buf, isZip }, [buf], 'opening the project',
    ));
    forgetAllPhotos();
    return data;
  }

  async function resetProject(name) {
    if (!isStatic()) throw new Error('not available in server mode');
    await send('reset', { name: name || 'untitled' }, [], 'starting a new project');
    forgetAllPhotos();
  }

  return {
    mode: () => MODE,
    isStatic,
    isReady,
    boot,
    call,
    onBusy,
    photoSrc,
    forgetPhoto,
    forgetAllPhotos,
    addFileUpload,
    exportAll,
    exportStep,
    saveBundle,
    openFile,
    resetProject,
    download,
  };
})();
