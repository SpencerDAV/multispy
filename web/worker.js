/* The Python side of the static build, off the main thread.
 *
 * Pyodide used to be loaded into the page itself, and every call into it was a
 * plain synchronous function call. That worked and produced the right numbers,
 * and it had one fault that made the build look broken: a synchronous call
 * holds the main thread, so for the whole of a solve the tab cannot paint, the
 * status line cannot update, the spinner cannot turn, and no click can be
 * handled. A nine-second solve was nine seconds of a dead window with nothing
 * on screen to say so -- and since nothing distinguishes that from a hang, the
 * reasonable response is to close the tab, which on a page whose only
 * persistence is a download means losing the marks.
 *
 * So the runtime lives in here instead. The numbers are identical -- same
 * CPython, same numpy and scipy, same `multispy` package -- but the page stays
 * alive while they are computed, which is what lets a long operation say that
 * it is still running.
 *
 * Deliberately a classic worker rather than a module one: `importScripts` is
 * how Pyodide expects to be loaded in a worker, and a classic worker asks
 * nothing of the host's MIME types or CORS headers, which for a page served
 * off GitHub Pages is one fewer thing that can differ between local and live.
 */

/* global importScripts, loadPyodide */

let pyodide = null;
let fns = null;

/* The Python bridge, moved here verbatim from backend.js. Everything crosses
 * as a JSON string, which keeps this boundary identical in shape to the HTTP
 * one -- the client cannot tell which it is talking to, and neither can the
 * Python. */
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

def _ms_export_step(shape):
    # bytes, not JSON: one STEP file, and a zip holding one file is an extra
    # step between the user and what they asked for
    return webapp.step_bytes(_ws.project, shape or "none", work_dir="/export")

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

const NAMES = [
  '_ms_dispatch', '_ms_add_file', '_ms_photo', '_ms_export_zip',
  '_ms_export_step', '_ms_reset', '_ms_bundle', '_ms_open_bundle',
];

async function boot({ pyodideUrl, pyBase, modules }) {
  const step = (msg, frac) => self.postMessage({ kind: 'progress', msg, frac });

  step('loading the Python runtime...', 0.05);
  importScripts(`${pyodideUrl}pyodide.js`);
  pyodide = await loadPyodide({ indexURL: pyodideUrl });

  step('loading numpy, scipy and pillow...', 0.35);
  await pyodide.loadPackage(['numpy', 'scipy', 'pillow']);

  step('loading MultiSpy...', 0.85);
  pyodide.FS.mkdirTree('/app/multispy');
  const sources = await Promise.all(
    modules.map(async (name) => {
      const res = await fetch(`${pyBase}multispy/${name}.py`);
      if (!res.ok) throw new Error(`missing ${name}.py (HTTP ${res.status})`);
      return [name, await res.text()];
    })
  );
  for (const [name, src] of sources) {
    pyodide.FS.writeFile(`/app/multispy/${name}.py`, src);
  }

  step('starting...', 0.95);
  pyodide.runPython(BRIDGE);
  fns = {};
  for (const name of NAMES) fns[name] = pyodide.globals.get(name);
  step('ready', 1);
}

/** Copy whatever Python handed back into a buffer this worker can pass on.
 *
 *  Two reasons it is copied rather than forwarded: a Pyodide buffer is a view
 *  onto the WASM heap, which the next allocation may move out from under it,
 *  and only an owned ArrayBuffer can be transferred rather than cloned. */
function ownBytes(out) {
  const raw = out && out.toJs ? out.toJs() : out;
  const copy = new Uint8Array(raw.length);
  copy.set(raw);
  if (out && out.destroy) out.destroy();
  return copy;
}

/** Hand incoming bytes to Python through the virtual filesystem rather than as
 *  a proxy object: unambiguous, and it keeps the Python side reading a plain
 *  file, exactly as it did when this ran on the main thread. */
function stash(bytes) {
  const tmp = `/tmp/in_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  pyodide.FS.writeFile(tmp, new Uint8Array(bytes));
  return tmp;
}

self.onmessage = async (ev) => {
  const { id, kind } = ev.data || {};
  const reply = (result, transfer) =>
    self.postMessage({ id, ok: true, result }, transfer || []);
  const fail = (err) =>
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });

  try {
    if (kind === 'boot') {
      await boot(ev.data);
      reply(true);
      return;
    }
    if (!fns) throw new Error('MultiSpy is still starting up');

    switch (kind) {
      case 'dispatch':
        reply(fns._ms_dispatch(ev.data.route, ev.data.body || ''));
        return;
      case 'addFile':
        reply(fns._ms_add_file(ev.data.fileKind, ev.data.name, stash(ev.data.bytes)));
        return;
      case 'openBundle':
        reply(fns._ms_open_bundle(stash(ev.data.bytes), !!ev.data.isZip));
        return;
      case 'reset':
        reply(fns._ms_reset(ev.data.name || 'untitled'));
        return;
      case 'photo': {
        const bytes = ownBytes(fns._ms_photo(ev.data.imageId, !!ev.data.thumb));
        reply(bytes, [bytes.buffer]);
        return;
      }
      case 'exportZip': {
        const bytes = ownBytes(fns._ms_export_zip(ev.data.formats || ''));
        reply(bytes, [bytes.buffer]);
        return;
      }
      case 'exportStep': {
        const bytes = ownBytes(fns._ms_export_step(ev.data.shape || 'none'));
        reply(bytes, [bytes.buffer]);
        return;
      }
      case 'bundle': {
        const bytes = ownBytes(fns._ms_bundle());
        reply(bytes, [bytes.buffer]);
        return;
      }
      default:
        throw new Error(`unknown worker request: ${kind}`);
    }
  } catch (err) {
    fail(err);
  }
};
