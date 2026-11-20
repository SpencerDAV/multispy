"""Building a ``.blend`` by driving Blender, without needing Blender.

CI has no Blender and never will -- the runners are a clean checkout and this
suite's whole premise is that it runs where nobody works. So the subprocess is
faked and what gets checked is everything around it: the command line Blender
is handed, that the script handed to it is real Python, and that a run which
produces no file is reported as a failure rather than passed off as success.

The one test that does use a real Blender finds it or says it did not, because
a test that silently becomes a no-op is worse than no test. Locally it is the
one that matters.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_blendfile.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy import blendfile as B  # noqa: E402
from multispy import export  # noqa: E402
from multispy.solve import solve  # noqa: E402
from test_synthetic import build_project  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="multispy_blend_"))

#: Run inside the built file by the one test that has a real Blender. Kept out
#: of that function because a script written as a pile of escaped newlines is
#: unreadable and, the first time this was done that way, unparseable.
PROBE_SCRIPT = """import bpy, json
obs = list(bpy.data.objects)
print("PROBE" + json.dumps(dict(
    cameras=sum(1 for o in obs if o.type == "CAMERA"),
    meshes=sum(1 for o in obs if o.type == "MESH"),
    images=len(bpy.data.images),
    packed=sum(1 for i in bpy.data.images if i.packed_file is not None),
)))
"""

_N = [0]
_SOLVED: list = []


def solved(photos: bool = False):
    """One solved synthetic project, built once -- the solve is the slow part."""
    if not _SOLVED:
        project, _ = build_project()
        report = solve(project)
        assert report.ok
        _SOLVED.append(project)
    project = _SOLVED[0]
    _N[0] += 1
    out = TMP / f"case{_N[0]}"
    out.mkdir(parents=True, exist_ok=True)
    project.path = out / "p.multispy"
    if photos:
        # Real files on disk, because packing is the point of the .blend and a
        # project whose photographs do not exist packs nothing and asserts
        # nothing. Flat colour, so a 4000x3000 JPEG costs almost nothing.
        from PIL import Image

        for im in project.images:
            path = out / im.path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (im.width, im.height), (70, 90, 120)).save(path, quality=40)
    return project, out


class FakeRun:
    """Stands in for ``subprocess.run``, recording argv.

    ``writes`` is what the pretend Blender leaves behind: ``None`` for nothing,
    which is the failure this has to notice.
    """

    def __init__(self, writes: str | None = "blend-bytes", returncode: int = 0,
                 stdout: str = "", stderr: str = ""):
        self.writes = writes
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.argv: list[str] = []
        self.calls = 0

    def __call__(self, argv, **kwargs):
        self.calls += 1
        self.argv = [str(a) for a in argv]
        if self.writes is not None:
            dest = Path(self.argv[self.argv.index("--save") + 1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(self.writes, encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def with_fake(fake, fn):
    real = B.subprocess.run
    B.subprocess.run = fake
    try:
        return fn()
    finally:
        B.subprocess.run = real


def a_real_file(name: str = "pretend-blender") -> Path:
    path = TMP / name
    path.write_text("not really an executable", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# finding Blender
# --------------------------------------------------------------------------


def test_an_explicit_path_wins():
    exe = a_real_file()
    assert B.find_blender(exe) == exe


def test_an_explicit_path_that_is_not_there_says_so():
    try:
        B.find_blender(TMP / "nowhere" / "blender.exe")
    except B.BlenderNotFound as exc:
        assert "blender.exe" in str(exc), exc
    else:
        raise AssertionError("a path that does not exist was accepted")


def test_the_environment_variable_is_honoured_and_named_when_wrong():
    import os

    exe = a_real_file("env-blender")
    old = os.environ.get(B.BLENDER_ENV)
    try:
        os.environ[B.BLENDER_ENV] = str(exe)
        assert B.find_blender() == exe
        os.environ[B.BLENDER_ENV] = str(TMP / "nope" / "blender")
        try:
            B.find_blender()
        except B.BlenderNotFound as exc:
            # the message has to name the variable, or the user has no idea
            # which of four lookups produced the complaint
            assert B.BLENDER_ENV in str(exc), exc
        else:
            raise AssertionError("a bad environment variable was accepted")
    finally:
        if old is None:
            os.environ.pop(B.BLENDER_ENV, None)
        else:
            os.environ[B.BLENDER_ENV] = old


def test_the_newest_install_wins_even_when_the_folder_name_says_otherwise():
    """The trap this was written for: a real machine with "Blender 43" beside
    "Blender 4.2" and "Blender 5.1". Sorted as text, 43 beats 5.1 and the
    oldest-but-one Blender gets picked; asked, 43 says it is 4.3.2."""
    root = TMP / "installs"
    made = {}
    for folder, version in (("Blender 4.2", (4, 2, 13)),
                            ("Blender 43", (4, 3, 2)),
                            ("Blender 5.1", (5, 1, 1))):
        exe = root / folder / "blender.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("", encoding="utf-8")
        made[exe.resolve()] = version

    real_globs = B._INSTALL_GLOBS.get(sys.platform)
    real_which = B.shutil.which
    real_version = B.blender_version
    try:
        B._INSTALL_GLOBS[sys.platform] = [str(root / "*" / "blender.exe")]
        B.shutil.which = lambda name: None          # nothing on PATH
        B.blender_version = lambda exe: made.get(Path(exe).resolve(), ())
        assert len(B._candidates()) == 3, B._candidates()
        assert B.find_blender().name == "blender.exe"
        assert B.find_blender().parent.name == "Blender 5.1", B.find_blender()
    finally:
        if real_globs is None:
            B._INSTALL_GLOBS.pop(sys.platform, None)
        else:
            B._INSTALL_GLOBS[sys.platform] = real_globs
        B.shutil.which = real_which
        B.blender_version = real_version


# --------------------------------------------------------------------------
# the command line, and what counts as having worked
# --------------------------------------------------------------------------


def test_blender_is_asked_for_a_headless_run_that_saves():
    project, out = solved()
    fake = FakeRun()
    exe = a_real_file("argv-blender")
    with_fake(fake, lambda: B.build_blend(project, out / "scene.blend", blender=exe))

    argv = fake.argv
    assert argv[0] == str(exe), argv
    for flag in ("--background", "--factory-startup", "--python"):
        assert flag in argv, f"{flag} missing from {argv}"
    # everything after the bare -- belongs to the script, not to Blender
    assert "--" in argv, argv
    tail = argv[argv.index("--") + 1 :]
    assert tail == ["--save", str(out / "scene.blend")], tail
    assert argv[argv.index("--python") + 1].endswith(".py")


def test_the_script_handed_over_is_real_python():
    """The template is built with ``str.format``, so one stray brace in it --
    or in anything interpolated into it -- produces a file that Blender will
    fail on, several seconds and one subprocess later."""
    project, out = solved()
    fake = FakeRun()
    exe = a_real_file("compile-blender")
    with_fake(fake, lambda: B.build_blend(project, out / "scene.blend", blender=exe))
    script = Path(fake.argv[fake.argv.index("--python") + 1])
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    assert "--save" in script.read_text(encoding="utf-8")


def test_a_run_that_writes_nothing_is_a_failure():
    project, out = solved()
    fake = FakeRun(writes=None, returncode=1, stderr="Error: could not open display\n")
    exe = a_real_file("silent-blender")
    try:
        with_fake(fake, lambda: B.build_blend(project, out / "scene.blend", blender=exe))
    except B.BlenderFailed as exc:
        assert "could not open display" in str(exc), exc
    else:
        raise AssertionError("a run that produced no file was reported as success")


def test_a_stale_file_left_from_last_time_is_not_mistaken_for_success():
    """``build_blend`` writes over an existing path, so "the file is there"
    proves nothing -- it was there before."""
    project, out = solved()
    dest = out / "scene.blend"
    dest.write_text("yesterday's file", encoding="utf-8")
    fake = FakeRun(writes=None, returncode=1)
    exe = a_real_file("stale-blender")
    try:
        with_fake(fake, lambda: B.build_blend(project, dest, blender=exe))
    except B.BlenderFailed:
        assert dest.read_text(encoding="utf-8") == "yesterday's file"
    else:
        raise AssertionError("yesterday's file was reported as today's export")


def test_an_unsolved_project_never_reaches_blender():
    from multispy.project import Project

    fake = FakeRun()
    exe = a_real_file("unused-blender")
    try:
        with_fake(
            fake,
            lambda: B.build_blend(Project(name="empty"), TMP / "none.blend", blender=exe),
        )
    except ValueError:
        assert fake.calls == 0, "Blender was started for a project with no solve"
    else:
        raise AssertionError("an unsolved project was exported")


def test_write_exports_routes_blend_through_the_builder():
    from multispy import webapp

    project, out = solved()
    fake = FakeRun()
    exe = a_real_file("route-blender")
    import os

    old = os.environ.get(B.BLENDER_ENV)
    os.environ[B.BLENDER_ENV] = str(exe)
    try:
        written = with_fake(
            fake, lambda: webapp.write_exports(project, out / "bundle", formats=("blend",))
        )
    finally:
        if old is None:
            os.environ.pop(B.BLENDER_ENV, None)
        else:
            os.environ[B.BLENDER_ENV] = old
    assert "blend" in written, written
    assert Path(written["blend"]).name == "scene.blend"
    # one script, shared with the "blender" format, rather than a second copy
    assert (out / "bundle" / "import_cameras.py").exists()


def test_blend_is_not_in_the_default_export_set():
    """Export must not start Blender. It is a separate button for a reason."""
    from multispy import webapp

    assert "blend" not in webapp.EXPORT_FORMATS, webapp.EXPORT_FORMATS


# --------------------------------------------------------------------------
# the real thing, when there is one
# --------------------------------------------------------------------------


def test_a_real_blender_builds_a_file_that_opens():
    try:
        exe = B.find_blender()
    except B.BlenderNotFound as exc:
        print(f"      no Blender here, so this checked nothing: {exc}")
        return
    version = B.blender_version(exe)
    project, out = solved(photos=True)
    dest = B.build_blend(project, out / "real.blend", blender=exe)
    assert dest.exists() and dest.stat().st_size > 1000, dest.stat().st_size
    # Opened rather than sniffed. A magic-byte check would have to know that
    # Blender 3.0 and later compress with Zstandard rather than gzip, which is
    # knowledge about file headers standing in for the only question worth
    # asking: does Blender read this back with the scene in it.
    probe = out / "probe.py"
    probe.write_text(
        PROBE_SCRIPT,
        encoding="utf-8",
    )
    spoken = subprocess.run(
        [str(exe), "--background", str(dest), "--python", str(probe)],
        capture_output=True, text=True, timeout=600, check=False,
    ).stdout or ""
    line = next((ln for ln in spoken.splitlines() if ln.startswith("PROBE")), None)
    assert line, "Blender could not reopen the file it wrote: " + spoken[-800:]
    got = json.loads(line[len("PROBE") :])
    expected = len([im for im in project.images if im.R is not None])
    assert got["cameras"] == expected, got
    assert got["meshes"] >= 1, got            # the tie-point cloud, at least
    # The photographs must be inside the file, or it is not portable and the
    # script it was built from would have been just as good. Checked against
    # the camera count as well, so a project that packed nothing because it
    # loaded nothing cannot pass this by having 0 == 0.
    assert got["images"] == expected, got
    assert got["packed"] == got["images"], got
    print(f"      {exe.name} {'.'.join(map(str, version))} -> "
          f"{dest.stat().st_size / 1e6:.1f} MB, reopened with "
          f"{got['cameras']} cameras, {got['meshes']} meshes, "
          f"{got['packed']}/{got['images']} photos packed")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    if failures:
        print(f"\nscratch files in {TMP}")
    else:
        shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
