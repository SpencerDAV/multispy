/* Where the UI's requests actually go.
 *
 * MultiSpy runs two ways from one codebase. Locally, a Python process serves
 * the UI and answers over HTTP. On a static host there is no process, so the
 * page loads CPython itself -- numpy, scipy and Pillow included -- through
 * Pyodide and calls the very same `multispy` package in-process.
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
   * absent: one is a CLI and the other is the transport this replaces. */
  const MODULES = [
    '__init__', 'geometry', 'mesh', 'exif', 'project', 'solve', 'export', 'webapp',
  ];

  let pyodide = null;
  let dispatch = null;
  let addFile = null;
  let photoFn = null;
  let exportFn = null;
  let booted = false;
  const urlCache = new Map(); // `${id}|${thumb}` -> object URL

  const isStatic = () => MODE === 'pyodide';
  const isReady = () => MODE === 'server' || booted;

  /* --------------------------------------------------------------- boot */

  async function boot(onProgress) {
    if (booted) return;
    const step = (msg, frac) => onProgress && onProgress(msg, frac);

    step('loading the Python runtime…', 0.05);
    if (!window.loadPyodide) {
      await new Promise((resolve, reject) => {
        const tag = document.createElement('script');
        tag.src = `${PYODIDE_URL}pyodide.js`;
        tag.onload = resolve;
        tag.onerror = () => reject(new Error('could not load Pyodide from the CDN'));
        document.head.appendChild(tag);
      });
    }
    pyodide = await window.loadPyodide({ indexURL: PYODIDE_URL });

    step('loading numpy, scipy and pillow…', 0.35);
    await pyodide.loadPackage(['numpy', 'scipy', 'pillow']);

    step('loading MultiSpy…', 0.85);
    pyodide.FS.mkdirTree('/app/multispy');
    const sources = await Promise.all(
      MODULES.map(async (name) => {
        const res = await fetch(`${PY_BASE}multispy/${name}.py`);
        if (!res.ok) throw new Error(`missing ${name}.py (HTTP ${res.status})`);
        return [name, await res.text()];
      })
    );
    for (const [name, src] of sources) {
      pyodide.FS.writeFile(`/app/multispy/${name}.py`, src);
    }

    step('starting…', 0.95);
    pyodide.runPython(BRIDGE);
    dispatch = pyodide.globals.get('_ms_dispatch');
    addFile = pyodide.globals.get('_ms_add_file');
    photoFn = pyodide.globals.get('_ms_photo');
    exportFn = pyodide.globals.get('_ms_export_zip');
    booted = true;
    step('ready', 1);
  }

  /* The Python side of the bridge. Everything crosses as a JSON string, which
   * keeps the boundary identical in shape to the HTTP one -- the client cannot
   * tell which it is talking to, and neither can the Python. */
  const BRIDGE = `
import json, sys, traceback
sys.path.insert(0, "/app")

from multispy import webapp

_ws = webapp.blank_workspace("/workspace")

def _reply(fn):
    try:
        return json.dumps(fn())
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return json.dumps({"__error__": str(exc).strip("'")})
    except Exception as exc:
        traceback.print_exc()
        return json.dumps({"__error__": f"{type(exc).__name__}: {exc}"})

def _ms_dispatch(route, body_json):
    body = json.loads(body_json) if body_json else {}
    return _reply(lambda: _ws.handle(route, body))

def _ms_add_file(kind, name, tmp_path):
    data = open(tmp_path, "rb").read()
    import os
    os.unlink(tmp_path)
    if kind == "photo":
        return _reply(lambda: _ws.add_photo(name, data))
    return _reply(lambda: _ws.add_model(name, data))

def _ms_photo(image_id, thumb):
    # bytes, not JSON: this one is a JPEG
    return _ws.photo_bytes(image_id, thumb=bool(thumb))

def _ms_export_zip(formats_json):
    formats = json.loads(formats_json) if formats_json else None
    return webapp.export_zip(_ws.project, formats, work_dir="/export")

def _ms_reset(name):
    global _ws
    _ws = webapp.blank_workspace("/workspace", name or "untitled")
    return "{}"

def _ms_bundle():
    # bytes, not JSON: the whole workspace zipped, photos included
    return webapp.bundle_zip(_ws)

def _ms_open_bundle(tmp_path, is_zip):
    import os
    data = open(tmp_path, "rb").read()
    os.unlink(tmp_path)

    def go():
        if is_zip:
            webapp.open_bundle(_ws, data)
        else:
            # a bare .multispy restores the marks; its photos are not in it, and
            # the payload's per-image "exists" flags will say so
            from multispy.project import Project
            incoming = Project.from_dict(json.loads(data.decode("utf-8")))
            incoming.path = _ws.project.path
            _ws.project = incoming
            _ws.last_report = None
        return {"project": webapp.project_payload(_ws.project)}

    return _reply(go)
`;

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
    return unwrap(dispatch(route, body === undefined ? '' : JSON.stringify(body)));
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
    const bytes = photoFn(imageId, thumb);
    const blob = new Blob([bytes.toJs ? bytes.toJs() : bytes], { type: 'image/jpeg' });
    if (bytes.destroy) bytes.destroy();
    const url = URL.createObjectURL(blob);
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
    // hand the bytes over through the virtual filesystem rather than as a proxy
    // object: unambiguous, and it keeps the Python side reading a plain file
    const tmp = `/tmp/upload_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    pyodide.FS.writeFile(tmp, new Uint8Array(await file.arrayBuffer()));
    return unwrap(addFile(kind, file.name, tmp));
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

  /** Static mode has nowhere to export to, so the same writers produce a zip. */
  async function exportAll(formats) {
    if (!isStatic()) return call('/api/export', { formats });
    if (!booted) throw new Error('MultiSpy is still starting up');
    const bytes = exportFn(formats ? JSON.stringify(formats) : '');
    const blob = new Blob([bytes.toJs ? bytes.toJs() : bytes], { type: 'application/zip' });
    if (bytes.destroy) bytes.destroy();
    download('multispy-export.zip', blob);
    return { written: { zip: 'multispy-export.zip' }, out_dir: 'your downloads folder' };
  }

  /** Download the whole workspace -- project, photos and models -- as one zip.
   *
   *  A tab has no disk, so this is what saving means here. Marks alone would
   *  reopen into a project whose photos are all missing. */
  async function saveBundle(projectName) {
    if (!isStatic()) throw new Error('not available in server mode');
    const fn = pyodide.globals.get('_ms_bundle');
    const bytes = fn();
    fn.destroy();
    const blob = new Blob([bytes.toJs ? bytes.toJs() : bytes], { type: 'application/zip' });
    if (bytes.destroy) bytes.destroy();
    const safe = String(projectName || 'project').replace(/[^\w.-]+/g, '-');
    download(`${safe}.multispy.zip`, blob);
  }

  /** Replace the open project from a file the user picked: a zip with its
   *  photos, or a bare .multispy with just the marks. */
  async function openFile(file) {
    if (!isStatic()) throw new Error('not available in server mode');
    const isZip = /\.zip$/i.test(file.name);
    const tmp = `/tmp/open_${Date.now()}`;
    pyodide.FS.writeFile(tmp, new Uint8Array(await file.arrayBuffer()));
    const fn = pyodide.globals.get('_ms_open_bundle');
    const data = unwrap(fn(tmp, isZip));
    fn.destroy();
    forgetAllPhotos();
    return data;
  }

  async function resetProject(name) {
    if (!isStatic()) throw new Error('not available in server mode');
    const reset = pyodide.globals.get('_ms_reset');
    reset(name || 'untitled');
    reset.destroy();
    forgetAllPhotos();
  }

  return {
    mode: () => MODE,
    isStatic,
    isReady,
    boot,
    call,
    photoSrc,
    forgetPhoto,
    forgetAllPhotos,
    addFileUpload,
    exportAll,
    saveBundle,
    openFile,
    resetProject,
    download,
  };
})();
