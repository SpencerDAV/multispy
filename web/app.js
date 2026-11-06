/* MultiSpy marking UI.
 *
 * Everything is stored and edited in ORIGINAL image pixel coordinates. Photos
 * are served downscaled for display, so each loaded bitmap carries a scale
 * factor; drawing happens in image space via a canvas transform, which keeps
 * hit-testing and persistence in one coordinate system.
 */

const AXIS_COLOR = ['#ff5a5a', '#4ddb7f', '#5a9cff'];
const AXIS_NAME = ['X', 'Y', 'Z'];
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
  showVP: true,
  showResid: true,
  showGrid: false,
  showModels: true,
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

function markDirty() {
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
  return copy;
}

function adopt(data) {
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
    const data = await Backend.exportAll();
    setStatus(`exported to ${data.out_dir}`, 'ok');
  } catch (err) {
    setStatus(`export failed: ${err.message}`, 'bad');
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

/** Vanishing point of one axis from its control lines, by median pairwise
 *  intersection. Only an overlay -- the solver does this properly. */
function vanishingPoint(im, axis) {
  const lines = im.axis_lines.filter((l) => l.axis === axis);
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
  if (state.showVP) drawVanishingRays(im, z);
  if (state.showModels) drawObjectModels(im, z);
  drawControlLines(im, z);
  if (state.showResid) drawResiduals(im, z);
  drawObservations(im, z);
  drawAnchors(im, z);
  if (state.drag && state.drag.kind === 'newline') drawPendingLine(z);

  $('zoombadge').textContent = `${im.width}x${im.height}  ${(z * 100).toFixed(0)}%`;
}

function drawControlLines(im, z) {
  im.axis_lines.forEach((ln, idx) => {
    const isHover = state.hover && state.hover.kind === 'line' && state.hover.index === idx;
    ctx.strokeStyle = AXIS_COLOR[ln.axis];
    ctx.lineWidth = (isHover ? 3.5 : 2) / z;
    ctx.beginPath();
    ctx.moveTo(ln.p0[0], ln.p0[1]);
    ctx.lineTo(ln.p1[0], ln.p1[1]);
    ctx.stroke();

    // direction arrow: p0 -> p1 must point along the POSITIVE axis
    const dx = ln.p1[0] - ln.p0[0], dy = ln.p1[1] - ln.p0[1];
    const len = Math.hypot(dx, dy) || 1;
    const a = 13 / z;
    const ux = dx / len, uy = dy / len;
    ctx.beginPath();
    ctx.moveTo(ln.p1[0], ln.p1[1]);
    ctx.lineTo(ln.p1[0] - a * ux + a * 0.45 * uy, ln.p1[1] - a * uy - a * 0.45 * ux);
    ctx.lineTo(ln.p1[0] - a * ux - a * 0.45 * uy, ln.p1[1] - a * uy + a * 0.45 * ux);
    ctx.closePath();
    ctx.fillStyle = AXIS_COLOR[ln.axis];
    ctx.fill();

    for (const [hx, hy, which] of [[ln.p0[0], ln.p0[1], 0], [ln.p1[0], ln.p1[1], 1]]) {
      const hot = state.hover && state.hover.kind === 'handle'
        && state.hover.index === idx && state.hover.which === which;
      ctx.beginPath();
      ctx.arc(hx, hy, (hot ? 6 : 4) / z, 0, Math.PI * 2);
      ctx.fillStyle = hot ? '#fff' : '#0d0f12';
      ctx.fill();
      ctx.strokeStyle = AXIS_COLOR[ln.axis];
      ctx.lineWidth = 1.5 / z;
      ctx.stroke();
    }
  });
}

function drawVanishingRays(im, z) {
  for (let axis = 0; axis < 3; axis++) {
    const vp = vanishingPoint(im, axis);
    if (!vp) continue;
    const lines = im.axis_lines.filter((l) => l.axis === axis);
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
    ctx.beginPath();
    ctx.arc(uv[0], uv[1], r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = (active ? 2.6 : 1.8) / z;
    ctx.stroke();
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

/* Each placed object's model, projected through this camera.
 *
 * This is the check that makes a known-object solve trustworthy in a way no
 * number can: if the model's wireframe lands on the real object in the
 * photograph, the pose is right. A residual table cannot tell you that a body
 * was placed 180 degrees out, and this picture can. */
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
    const stride = mesh.edges.length > budget ? Math.ceil(mesh.edges.length / budget) : 1;

    ctx.beginPath();
    for (let i = 0; i < mesh.edges.length; i += stride) {
      const a = screen[mesh.edges[i][0]];
      const b = screen[mesh.edges[i][1]];
      if (!a || !b) continue; // the edge crosses behind the camera
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    const selected = inst.id === state.activeInstance;
    ctx.strokeStyle = selected ? '#ffd24a99' : inst.locked ? '#9affc866' : '#7fd4ff66';
    ctx.lineWidth = (selected ? 1.6 : 1.1) / z;
    ctx.stroke();

    const center = projectWorld(im, objectToWorld(inst, template, [0, 0, 0]));
    if (center && z > 0.12) {
      ctx.fillStyle = selected ? '#ffd24a' : '#7fd4ffaa';
      ctx.font = `${12 / z}px sans-serif`;
      ctx.fillText(inst.label || inst.id, center.x + 6 / z, center.y - 6 / z);
    }
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

function drawResiduals(im, z) {
  if (!state.report || !state.report.residual_lines) return;
  ctx.strokeStyle = '#ffa23f';
  ctx.lineWidth = 1.6 / z;
  for (const rl of state.report.residual_lines) {
    if (rl.image !== im.id) continue;
    const ox = rl.observed[0], oy = rl.observed[1];
    const dx = (rl.reprojected[0] - ox) * RESIDUAL_GAIN;
    const dy = (rl.reprojected[1] - oy) * RESIDUAL_GAIN;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(ox + dx, oy + dy);
    ctx.stroke();
  }
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

function drawPendingLine(z) {
  const d = state.drag;
  ctx.strokeStyle = AXIS_COLOR[state.axis];
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

/* ------------------------------------------------------- anchor marking */

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
      setStatus(`${inst.label || inst.id}: every anchor marked in every photo`, 'ok');
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
  const im = currentImage();
  if (!im) return;
  const p = toImage(ev);

  if (ev.button === 1 || state.spaceDown || ev.altKey) {
    state.drag = { kind: 'pan', sx: ev.clientX, sy: ev.clientY,
                   px: state.view.panX, py: state.view.panY };
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
  if (!d || !im) return;

  if (d.kind === 'newline') {
    const len = Math.hypot(d.to.x - d.from.x, d.to.y - d.from.y) * state.view.zoom;
    if (len > 12) {
      im.axis_lines.push({ axis: state.axis, p0: [d.from.x, d.from.y], p1: [d.to.x, d.to.y] });
      markDirty();
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

window.addEventListener('keydown', (ev) => {
  if (ev.target.matches('input, select, textarea')) return;
  if (dialogOpen()) return;
  const k = ev.key.toLowerCase();

  if (k === ' ') { state.spaceDown = true; canvas.style.cursor = 'grab'; ev.preventDefault(); return; }
  if (ev.ctrlKey && k === 's') { ev.preventDefault(); saveNow(); return; }
  if (k === 'enter') { ev.preventDefault(); solveNow(); return; }
  if (k === '?' || (k === '/' && ev.shiftKey)) { ev.preventDefault(); openDocs(); return; }

  if (k === 'escape' && Viewer3D.isOn()) { toggleViewer(false); return; }
  if (k === 'd') { toggleViewer(!Viewer3D.isOn()); return; }
  // while the 3D view is up it owns fit, since the 2D fit would be invisible
  if (k === 'f' && Viewer3D.isOn()) { Viewer3D.frameAll(); return; }

  if (k === '1' || k === '2' || k === '3') { setAxis(+k - 1); return; }
  if (k === 'l') { setMode('lines'); return; }
  if (k === 'p') { setMode('points'); return; }
  if (k === 'o') { setMode('objects'); return; }
  if (k === 's') { setMode('select'); return; }
  if (k === 'n') { newPoint(); renderPanels(); render(); return; }
  if (k === 'v') { $('showvp').click(); return; }
  if (k === 'r') { $('showresid').click(); return; }
  if (k === 'g') { $('showgrid').click(); return; }
  if (k === 'm') { $('showmodels').click(); return; }
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
  $('axes').style.opacity = mode === 'lines' ? '1' : '.4';
  // marking anchors means looking at the photo, so step out of the 3D view
  if (mode === 'objects' && Viewer3D.isOn() && Viewer3D.mode() === 'template') toggleViewer(false);
  updateHint();
  render();
}

function setAxis(axis) {
  state.axis = axis;
  for (const b of $('axes').children) b.classList.toggle('on', +b.dataset.axis === axis);
  if (state.mode !== 'lines') setMode('lines');
  updateHint();
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
        + `Three anchors in two photos each is the floor for a placement.`;

  const hints = {
    lines: `Drag along an edge parallel to the <b>${AXIS_NAME[state.axis]}</b> axis. `
         + `Draw in the <b>+${AXIS_NAME[state.axis]}</b> direction — the arrow shows which way. `
         + `Click a drawn line to reverse it, right-click to delete. `
         + `Two per axis minimum, three is better. Keys 1/2/3 switch axis.`,
    points: `Click the same physical feature in each photo it appears in. `
          + `<b>N</b> starts a new point, <b>Tab</b> jumps to the next photo missing it. `
          + `Two shared points per image pair is the minimum.`,
    objects: objectHint,
    select: `Drag markers or whole control lines; click a line to reverse it. `
          + `Right-click or Delete removes one.`,
  };
  $('hint').innerHTML = hints[state.mode]
    + ` &nbsp;·&nbsp; wheel zoom, space-drag pan, <b>F</b> fit`
    + (state.showResid ? ` &nbsp;·&nbsp; residuals ×${RESIDUAL_GAIN}` : '');
}

/* ----------------------------------------------------------------- filmstrip */

function imageStatus(im) {
  const rep = state.report && state.report.images ? state.report.images[im.id] : null;
  const axes = new Set(im.axis_lines.map((l) => l.axis));
  const ready = [...axes].filter((a) => im.axis_lines.filter((l) => l.axis === a).length >= 2);
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
      `<button class="rm" title="Remove from this project">&times;</button>` +
      `<div class="meta"><span>${st.text}</span>` +
      `<span class="dot" style="background:${st.color}"></span></div>`;
    // the thumbnail's source is resolved asynchronously, since in static mode it
    // is a blob the Python side has to render first
    Backend.photoSrc(im.id, true)
      .then((src) => { const img = div.querySelector('img'); if (img) img.src = src; })
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
  renderObjectsPanel();
  renderPointsPanel();
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

  const counts = [0, 1, 2].map((a) => im.axis_lines.filter((l) => l.axis === a).length);
  $('imgcount').textContent = meta.name || im.id;

  let html = `<div class="axiscounts">` + counts.map((c, a) =>
    `<div class="${['x','y','z'][a]} ${c < 2 ? 'short' : ''}">${AXIS_NAME[a]} ${c}</div>`
  ).join('') + `</div>`;

  html += `<div class="row" style="margin-top:7px"><span class="k">tie points here</span>`
        + `<span class="v">${Object.keys(im.observations).length}</span></div>`;
  if (rep && rep.positioned) {
    html += `<div class="row"><span class="k">reproj RMS</span><span class="v">${rep.rms_px.toFixed(2)} px</span></div>`;
    html += `<div class="row"><span class="k">worst</span><span class="v">${rep.max_px.toFixed(2)} px`
          + (rep.worst_point ? ` (${rep.worst_point})` : '') + `</span></div>`;
    html += `<div class="row"><span class="k">line RMS</span><span class="v">${rep.line_rms_px.toFixed(2)} px</span></div>`;
  }
  for (const n of (rep && rep.notes) || []) html += `<div class="note">${escapeHtml(n)}</div>`;

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
  scheduleExif();
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

function renderPointsPanel() {
  const im = currentImage();
  const el = $('ptpanel');
  const pts = state.project.points;
  $('ptcount').textContent = `${pts.length}`;

  let html = `<div style="display:flex;gap:5px;margin-bottom:6px">`
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
      html += `<div class="pt${p.id === state.activePoint ? ' sel' : ''}${here ? ' here' : ''}" data-pt="${p.id}">`
            + `<span class="swatch"></span>`
            + `<span class="name">${escapeHtml(p.label || p.id)} <span style="color:var(--text-dim)">${seen}img</span></span>`
            + `<span class="err${warn}">${err || (seen < 2 ? 'needs 2' : '')}</span></div>`;
    }
    html += '</div>';
    if (state.activePoint) {
      const ap = pts.find((p) => p.id === state.activePoint);
      if (ap) {
        html += `<div class="row" style="margin-top:7px"><span class="k">label</span>`
              + `<input type="text" id="ptlabel" value="${escapeHtml(ap.label)}" style="width:150px"></div>`;
        html += `<div style="display:flex;gap:5px;margin-top:5px">`
              + `<button class="smallbtn" id="delpt">delete point</button></div>`;
      }
    }
  }
  el.innerHTML = html;

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
  $('objcount').textContent = objs.length ? `${objs.length}` : '';

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
    html += `<div class="tpl${sel ? ' sel' : ''}" data-tpl="${escapeHtml(tid)}">`
          + `<div class="tplhead"><b>${escapeHtml(t.name || tid)}</b>`
          + `<span class="est">${anchors.length} anchor(s)</span></div>`;

    if (mesh === null) {
      html += `<div class="msg">loading model…</div>`;
    } else if (mesh && mesh.error) {
      html += `<div class="note bad">${escapeHtml(mesh.error)}</div>`;
    } else if (mesh) {
      const real = mesh.size.map((v) => (v * t.unit_scale).toFixed(3));
      html += `<div class="msg">${mesh.n_faces} faces, ${mesh.n_feature_edges} edges</div>`
            + `<div class="unitrow">`
            +   `<span class="k">units</span>`
            +   `<select class="tplunit" data-tpl="${escapeHtml(tid)}">`
            +     Object.keys(UNIT_SCALES).map((u) =>
                    `<option value="${u}"${u === t.units ? ' selected' : ''}>${u}</option>`).join('')
            +   `</select>`
            +   `<span class="est">= ${real.join(' × ')} m</span>`
            + `</div>`;
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
          + `<button class="smallbtn" data-pick="${escapeHtml(tid)}">pick anchors in 3D</button>`
          + `<button class="smallbtn" data-typed="${escapeHtml(tid)}">type a position…</button>`
          + `<button class="smallbtn" data-optical="${escapeHtml(tid)}">${o ? 'edit' : 'set'} optical</button>`
          + `<button class="smallbtn" data-newobj="${escapeHtml(tid)}">+ object</button>`
          + `<button class="smallbtn" data-deltpl="${escapeHtml(tid)}">remove model</button>`
          + `</div></div>`;
  }

  html += `<button class="smallbtn" id="addmodel" style="width:100%">+ add model (OBJ / STL / PLY)</button>`;

  if (objs.length) {
    html += `<div class="objlist" style="margin-top:7px">`;
    for (const inst of objs) {
      const rep = state.report && state.report.instances ? state.report.instances[inst.id] : null;
      const placed = isPlaced(inst);
      const shape = rep && rep.placed ? `${(rep.shape_rms * 1000).toFixed(1)}mm` : '';
      const bad = rep && rep.placed && (rep.shape_rms > 0.005 || Math.abs(rep.implied_scale - 1) > 0.05);
      html += `<div class="obj${inst.id === state.activeInstance ? ' sel' : ''}`
            + `${placed ? ' placed' : ''}${inst.locked ? ' locked' : ''}" data-obj="${escapeHtml(inst.id)}">`
            + `<span class="swatch"></span>`
            + `<span class="name">${escapeHtml(inst.label || inst.id)}`
            + `<span style="color:var(--text-dim)"> ${rep ? `${rep.n_anchors_used}/${rep.n_anchors_marked}` : ''}</span></span>`
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

  for (const card of el.querySelectorAll('.tpl[data-tpl]')) {
    card.onclick = (ev) => {
      if (ev.target.closest('button, select, input, .an')) return;
      state.activeTemplate = card.dataset.tpl;
      renderObjectsPanel();
      syncViewer();
    };
  }
  for (const sel of el.querySelectorAll('select.tplunit')) {
    sel.onchange = (e) => {
      const t = templates()[sel.dataset.tpl];
      t.units = e.target.value;
      t.unit_scale = UNIT_SCALES[e.target.value];
      markDirty();
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

  const all = [
    ...points,
    ...cameras.map((c) => c.center),
    ...objects.filter((o) => o.placed).map((o) => o.center || o.t),
  ];
  let center = [0, 0, 0];
  let extent = 1;
  if (all.length) {
    const lo = [0, 1, 2].map((k) => Math.min(...all.map((p) => p[k])));
    const hi = [0, 1, 2].map((k) => Math.max(...all.map((p) => p[k])));
    center = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
    extent = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), 1e-6);
  }
  return { points, cameras, objects, center, extent };
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
for (const b of $('axes').children) b.onclick = () => setAxis(+b.dataset.axis);
$('btn-help').onclick = () => openDocs();
$('btn-save').onclick = saveNow;
$('btn-solve').onclick = solveNow;
$('btn-export').onclick = exportNow;
$('showvp').onchange = (e) => { state.showVP = e.target.checked; render(); };
$('showresid').onchange = (e) => { state.showResid = e.target.checked; updateHint(); render(); };
$('showgrid').onchange = (e) => { state.showGrid = e.target.checked; render(); };
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

/* ---- 3D view */
$('btn-3d').onclick = () => toggleViewer(!Viewer3D.isOn());
$('v3-close').onclick = () => toggleViewer(false);
$('v3-fit').onclick = () => Viewer3D.frameAll();
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
});
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
