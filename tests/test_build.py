"""Does the static build ship every module the browser will import?

The browser build is not a second implementation — it is the same `multispy`
package, loaded into Pyodide and called in-process. Which files get copied is
decided by one explicit tuple, `tools.build_static.PY_MODULES`, and that tuple
is maintained by hand.

This file exists because the hand-maintained part has now failed twice, the same
way both times, and neither failure could be caught by running anything. The
tuple's own docstring records the first: adding `part` and `edgefit` shipped the
files while `backend.js` went on loading the old eight, so `webapp.py` could not
import them and the static build would not start — *with the local server
working perfectly*, because the local server does not read that list. The fix
then was to stop stating the list twice. The second failure was adding
`multispy/viewpoint.py` and a module-level `from . import viewpoint` in
`webapp.py` without adding it to the tuple. `tools/build_static.py` exited 0,
every one of eleven test files passed, and the bundle it produced could not
import its own entry point.

That is the shape of the problem worth defending against: **the break is
invisible to everything except loading the page.** The build only copies files,
so it cannot fail on a missing one; the tests run against the source tree, where
every module is present; and the local server imports from the source tree too.
Only the browser, which is the one place nobody runs in a terminal, sees it.

So what is checked here is the one invariant that makes the bundle loadable:
every module reachable by relative import from the shipped entry point is itself
shipped. Reachability rather than a flat list of `webapp.py`'s own imports,
because a shipped module importing an unshipped one breaks the bundle just as
completely, one level further down.

Two things are deliberately *not* asserted. The tuple is not required to be
minimal — shipping a module nothing imports costs a few kilobytes and breaks
nothing, so a test demanding minimality would fail on a deliberate inclusion and
teach people to delete modules the build wants. And the modules the build
deliberately omits (`__main__`, `server`, `docpage`) are named here rather than
inferred, so omitting one stays a decision somebody wrote down.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_build.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_static  # noqa: E402

PACKAGE = ROOT / "multispy"

ENTRY = "webapp"
"""What the page imports first. `backend.js` loads the package and calls into
`webapp`, so this is the root of everything the browser needs."""

NOT_SHIPPED = {"__main__", "server", "docpage"}
"""Modules the build omits on purpose.

`__main__` is a command line, `server` is the transport the static build
replaces, and `docpage` renders to HTML at build time rather than running in the
page. Listed so that a module reachable from `webapp` but missing from
`PY_MODULES` is either a real break or an explicit decision, never an oversight
that happens to look like one.
"""


def relative_imports(module: str) -> set[str]:
    """The package's own modules that `module` imports, by any relative form.

    Covers `from . import a, b`, `from .a import X` and `from .a.b import X`,
    which is every form the package actually uses. Absolute `import multispy.x`
    is deliberately not handled: the package does not use it, and quietly
    half-supporting a form nobody writes would make this test look more thorough
    than it is.
    """
    path = PACKAGE / f"{module}.py"
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        if node.module:
            # from .solve import x  ->  solve
            found.add(node.module.split(".")[0])
        else:
            # from . import solve, geometry
            found.update(alias.name for alias in node.names)
    return found


def reachable_from(entry: str) -> set[str]:
    """Every package module the entry point pulls in, transitively."""
    seen: set[str] = set()
    stack = [entry]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(relative_imports(name))
    return seen


def test_the_entry_point_is_shipped() -> None:
    assert ENTRY in build_static.PY_MODULES, (
        f"{ENTRY!r} is what the page imports first; without it there is no bundle"
    )


def test_every_module_the_page_reaches_is_shipped() -> None:
    """The invariant. This is the test that would have caught both breaks."""
    reachable = reachable_from(ENTRY)
    missing = sorted(reachable - set(build_static.PY_MODULES) - NOT_SHIPPED)
    assert not missing, (
        f"{', '.join(missing)} reachable from multispy/{ENTRY}.py by relative import but "
        f"not in tools/build_static.PY_MODULES. The static build copies files and will "
        f"exit 0 without them; the page then fails to import {ENTRY} and does not start. "
        f"Add them to the tuple."
    )


def test_a_shipped_module_that_is_missing_would_be_caught() -> None:
    """The test above asserts an absence, so prove it can see a presence.

    An assertion that something is empty passes just as well when the machinery
    that fills it is broken — a typo in the import walk, a wrong package path —
    and it would then go on passing through exactly the break it exists to
    catch. So the detector is run against a known-bad tuple.
    """
    real = build_static.PY_MODULES
    victim = ENTRY if len(real) < 2 else next(m for m in real if m != ENTRY)
    try:
        build_static.PY_MODULES = tuple(m for m in real if m != victim)
        reachable = reachable_from(ENTRY)
        missing = sorted(reachable - set(build_static.PY_MODULES) - NOT_SHIPPED)
        if victim in reachable:
            assert victim in missing, (
                f"removing {victim!r} from PY_MODULES was not noticed, so the check above "
                f"is not actually checking anything"
            )
    finally:
        build_static.PY_MODULES = real


def test_the_module_list_is_not_also_written_in_the_front_end() -> None:
    """The first break was duplication; it should stay un-duplicated.

    `backend.js` must take the list from the generated `config.js`, not carry its
    own copy. A second hand-maintained list is the exact failure the tuple's
    docstring describes.
    """
    js = (ROOT / "web" / "backend.js").read_text(encoding="utf-8")
    assert "MULTISPY_PY_MODULES" in js, (
        "backend.js should read the generated module list from config.js"
    )
    for name in ("nominalcheck", "sensorcheck", "edgefit", "viewpoint"):
        assert f"'{name}'" not in js and f'"{name}"' not in js, (
            f"backend.js names the module {name!r} directly. The list belongs in "
            f"tools/build_static.py alone — stating it twice is what broke the build the "
            f"first time."
        )


def test_the_worker_is_shipped_and_versioned() -> None:
    """The browser's Python runs in `worker.js`, which only the build can name.

    It is the one script the page loads that is not a `<script src>` in
    `index.html` -- `backend.js` constructs it -- so the index's content-hash
    rewrite cannot reach it and nothing else refers to it by name. Two ways that
    breaks without a word, both ending in a page that loads, draws its whole UI,
    and then does nothing at all when Solve is pressed: the file is never copied
    into the bundle, or it is copied unversioned and ten minutes of Pages cache
    pairs last build's worker with this build's `backend.js`.
    """
    assert (ROOT / "web" / "worker.js").is_file(), "web/worker.js is missing"
    assert "worker.js" in build_static.WEB_FILES, (
        "worker.js is not in WEB_FILES, so the build will not copy it and the static "
        "page will have no Python at all"
    )
    assert "MULTISPY_WORKER_URL" in build_static.CONFIG_JS, (
        "config.js must tell the page where the worker is, with a content hash on it"
    )
    js = (ROOT / "web" / "backend.js").read_text(encoding="utf-8")
    assert "MULTISPY_WORKER_URL" in js, (
        "backend.js should take the worker's versioned URL from config.js"
    )


def test_the_3d_view_has_both_of_its_layers() -> None:
    """The 3D view is a GL canvas under a 2D one, and both ends have to agree.

    Three separate pieces, in three files, none of which fails loudly on its
    own: `index.html` has to carry the second canvas, `app.js` has to hand it
    to `Viewer3D.attach` as its third argument, and `viewer3d.js` has to accept
    one. Drop any of them and the viewer keeps working -- it silently falls
    back to the 2D wireframe, which is the exact state this change exists to
    get out of, and which looks from the outside like a CAD model that renders
    as a few primitive lines.

    The stacking is checked too. The 2D canvas sits on top and had no z-index
    before; giving it one put the toolbar underneath it, where it was visible
    through the transparent layer and not clickable.
    """
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    viewer = (ROOT / "web" / "viewer3d.js").read_text(encoding="utf-8")

    assert 'id="canvas3dgl"' in html, (
        "index.html has no GL canvas, so the 3D view can only draw wireframes"
    )
    assert 'id="v3-solid"' in html, "the view bar has no control for the solid layer"
    assert "$('canvas3dgl')" in app, (
        "app.js never passes the GL canvas to Viewer3D.attach, so gpu.init gets "
        "nothing and every body falls back to the 2D wireframe"
    )
    assert "function attach(canvas, hooks, glCanvas)" in viewer, (
        "viewer3d.js does not take a GL canvas any more"
    )
    assert "z-index: 2" in html.split("#view3dbar {")[1].split("}")[0], (
        "#view3dbar has no z-index above the two canvases; it will be drawn "
        "under them and cannot be clicked"
    )


def test_the_viewer_can_live_without_a_gpu() -> None:
    """No WebGL context must mean a wireframe, never a blank view.

    A headless run, an old browser or a blocklisted driver all end in the same
    place, and the view still has to show the scene. The guard is that the
    shading path is reached through `gpu.ready()` rather than assumed.
    """
    viewer = (ROOT / "web" / "viewer3d.js").read_text(encoding="utf-8")
    assert "state.solid && gpu.ready()" in viewer, (
        "render() does not check for a GL context before taking the solid path"
    )
    assert "solidAvailable" in viewer and "solidWhy" in viewer, (
        "viewer3d.js does not expose whether shading is possible, so the UI "
        "cannot say why the control is dead"
    )


def test_the_payload_order_the_front_end_relies_on_is_stated() -> None:
    """The overlays take a prefix, so the server has to promise a sort.

    `app.js` stopped sorting edges for itself and now takes the first N of
    whatever arrives, on the understanding that `display_payload` sorted them
    longest-first. That understanding lives in two files and is enforced by
    neither, and getting it wrong is silent: the overlay would draw an
    arbitrary handful of edges and look merely sparse.
    """
    mesh = (ROOT / "multispy" / "mesh.py").read_text(encoding="utf-8")
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "np.argsort(-lengths)" in mesh and "np.argsort(-area)" in mesh, (
        "display_payload no longer sorts edges by length and faces by area, "
        "which is what makes the front end's prefix correct"
    )
    assert "edgesByLength" not in app, (
        "app.js still sorts edges for itself; the server's order is the one "
        "that counts and two orders will disagree"
    )


def test_python_does_not_run_on_the_main_thread() -> None:
    """The freeze this architecture exists to prevent, as an assertion.

    Pyodide used to be loaded into the page, and every call into it was an
    ordinary synchronous function call. The numbers were right and the window
    was dead for the duration: nothing could repaint during a solve, so a nine
    second solve showed nine seconds of a frozen UI with no indication it was
    working rather than hung -- which on a page whose only persistence is a
    download costs the user their marks when they close the tab.

    Nothing about that is visible from a terminal, and the obvious way to
    regress it is to reach for `pyodide` in `backend.js` again because one new
    call is easier to write synchronously. So: the runtime belongs to the
    worker, and the main thread may not speak to it directly.
    """
    js = (ROOT / "web" / "backend.js").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in js.splitlines() if not line.lstrip().startswith(("*", "/*", "//"))
    )
    for token in ("loadPyodide", "pyodide.FS", "pyodide.globals", "pyodide.runPython"):
        assert token not in code, (
            f"backend.js uses {token!r}, which means Python is being driven from the main "
            f"thread again. That is the freeze: a synchronous call cannot repaint, so the "
            f"page cannot say it is working. Route it through worker.js instead."
        )

    worker = (ROOT / "web" / "worker.js").read_text(encoding="utf-8")
    assert "_ms_dispatch" in worker, "worker.js should hold the Python bridge"
    assert "_ms_dispatch" not in code, (
        "the Python bridge is written in backend.js as well as worker.js. Stating it "
        "twice is what broke the module list the first time; one copy, in the worker."
    )


def test_the_omissions_are_real_modules() -> None:
    """A stale name in NOT_SHIPPED would silence a genuine miss.

    If `server.py` were renamed, its entry here would go on excusing whatever
    took its place in an import graph.
    """
    for name in sorted(NOT_SHIPPED):
        assert (PACKAGE / f"{name}.py").is_file(), (
            f"NOT_SHIPPED names {name!r} but multispy/{name}.py does not exist. Either the "
            f"module was renamed and this entry is now excusing something else, or the "
            f"entry can go."
        )


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    reachable = sorted(reachable_from(ENTRY))
    shipped = set(build_static.PY_MODULES)
    print(f"\n{len(reachable)} modules reachable from multispy/{ENTRY}.py:")
    for name in reachable:
        mark = "ship" if name in shipped else ("omit" if name in NOT_SHIPPED else "MISSING")
        print(f"  {mark:>7}  {name}")
    extra = sorted(shipped - set(reachable) - {"__init__"})
    if extra:
        print(f"\nshipped but not reached from {ENTRY} (allowed): {', '.join(extra)}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
