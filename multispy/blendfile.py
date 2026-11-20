"""A ``.blend`` of the solved scene, built by driving Blender headless.

``export.export_blender`` writes a *script*, which is the right primitive and
the wrong deliverable. Someone who wants to look at the result has to find
Blender's Scripting tab, open the file and press Run, and the thing they end up
with still points at photographs and CAD models sitting elsewhere on disk --
move any of it and the scene is hollow. That is three steps and a trap between
a finished solve and seeing it.

So this runs the same script for them, in the same Blender they already have,
with ``--save``: the scene starts empty rather than on the startup cube, the
photographs are packed into the file, and the CAD arrives as real geometry.
What comes out is one file that opens anywhere.

**What this cannot do.** It needs Blender on the machine doing the work, so it
is server-only -- the browser build has no Blender and no way to reach one, and
the button is not offered there. It is also the one export that can fail for a
reason that has nothing to do with the project, which is why the errors below
say which of those happened rather than reporting "export failed".
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import export
from .project import Project

__all__ = [
    "BLENDER_ENV",
    "BlenderNotFound",
    "BlenderFailed",
    "find_blender",
    "blender_version",
    "build_blend",
]

#: Set this to the Blender executable to override everything below.
BLENDER_ENV = "MULTISPY_BLENDER"

#: Where Blender installs itself, per platform. Globs, newest match wins --
#: "newest" decided by asking each binary rather than by reading the directory
#: name, because this machine has a "Blender 43" next to a "Blender 4.2" and
#: sorting those as text puts 43 above 5.1.
_INSTALL_GLOBS = {
    "win32": [
        r"C:\Program Files\Blender Foundation\*\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\*\blender.exe",
        r"C:\Program Files\Blender\*\blender.exe",
    ],
    "darwin": [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
    ],
    "linux": [
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
        "/opt/blender*/blender",
    ],
}

_VERSION_RE = re.compile(r"Blender\s+(\d+)\.(\d+)(?:\.(\d+))?")


class BlenderNotFound(RuntimeError):
    """No Blender executable could be found to run the script with."""


class BlenderFailed(RuntimeError):
    """Blender ran and did not produce the file."""


def _candidates() -> list[Path]:
    globs = _INSTALL_GLOBS.get(sys.platform)
    if globs is None:  # an unlisted platform still gets PATH, above
        globs = _INSTALL_GLOBS["linux"]
    found: list[Path] = []
    for pattern in globs:
        if "*" not in pattern:
            base = Path(pattern)
            if base.exists():
                found.append(base)
            continue
        # Split at the first component holding a wildcard: everything before it
        # is a real directory to glob from. Splitting on the "*" character
        # instead gets `C:\...\Blender Foundation\*\blender.exe` wrong, because
        # the text before the star ends in a separator and its "parent" is a
        # level too high.
        parts = Path(pattern).parts
        idx = next(i for i, part in enumerate(parts) if "*" in part)
        anchor = Path(*parts[:idx]) if idx else Path(".")
        try:
            found.extend(sorted(anchor.glob(str(Path(*parts[idx:])))))
        except (OSError, ValueError, IndexError):
            continue
    return [p for p in found if p.is_file()]


def blender_version(exe: str | Path) -> tuple[int, ...]:
    """``(major, minor, patch)`` as the binary itself reports it, or ``()``.

    Asked rather than inferred: a directory name is not a version, and the one
    time it disagrees is the time this would pick the wrong Blender.
    """
    try:
        out = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    m = _VERSION_RE.search(out or "")
    if not m:
        return ()
    return tuple(int(g) for g in m.groups() if g is not None)


def find_blender(explicit: str | Path | None = None) -> Path:
    """The Blender to run, in order of how deliberately it was chosen.

    An explicit path, then ``MULTISPY_BLENDER``, then whatever is on ``PATH``,
    then the newest installed. The first three are somebody's decision and are
    taken as given -- including a version this script may not like, because
    overriding a stated choice to be helpful is how a tool becomes impossible
    to point at the build you meant.
    """
    for chosen, why in ((explicit, "the path given"), (os.environ.get(BLENDER_ENV), BLENDER_ENV)):
        if not chosen:
            continue
        path = Path(chosen)
        if path.is_file():
            return path
        found = shutil.which(str(chosen))
        if found:
            return Path(found)
        raise BlenderNotFound(f"{why} is not an executable: {chosen}")

    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)

    installed = _candidates()
    if not installed:
        raise BlenderNotFound(
            "no Blender found. Put it on PATH, or set "
            f"{BLENDER_ENV} to the executable."
        )
    best = max(installed, key=blender_version)
    if not blender_version(best):
        raise BlenderNotFound(
            f"found {len(installed)} candidate(s) but none reported a version; "
            f"set {BLENDER_ENV} to the executable to use"
        )
    return best


def build_blend(
    project: Project,
    out_path: str | Path,
    *,
    blender: str | Path | None = None,
    images_dir: str | Path | None = None,
    script_path: str | Path | None = None,
    timeout: float = 1800.0,
) -> Path:
    """Write ``out_path`` as a packed ``.blend`` of the solved scene.

    The script is written beside it (``script_path`` to put it elsewhere) and
    kept rather than cleaned up: when Blender does something unexpected the
    script is the only thing that says what it was asked to do.

    ``timeout`` is generous on purpose. The work is dominated by importing CAD
    and packing photographs, both of which scale with the project rather than
    with anything this can predict -- 28 bodies and 8 photographs take a few
    seconds, and there is no useful smaller bound to impose on a project nobody
    has seen.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not export._solved_images(project):
        raise ValueError("nothing solved yet")

    exe = find_blender(blender)
    script = Path(script_path) if script_path else out_path.with_suffix(".py")
    export.export_blender(project, script, images_dir)

    before = out_path.stat().st_mtime_ns if out_path.exists() else None
    proc = subprocess.run(
        [
            str(exe),
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            "--save",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    wrote = out_path.exists() and (before is None or out_path.stat().st_mtime_ns != before)
    if not wrote:
        raise BlenderFailed(
            f"{exe} ran but wrote no file.\n"
            f"exit {proc.returncode}\n{_tail(proc.stdout)}{_tail(proc.stderr)}"
        )
    return out_path


def _tail(text: str | None, lines: int = 12) -> str:
    """The end of Blender's output, which is where it says what went wrong.

    Blender is talkative on startup and the interesting part is always last, so
    an error that pasted the whole thing would bury the one line worth reading.
    """
    if not text:
        return ""
    kept = [ln for ln in text.strip().splitlines() if ln.strip()][-lines:]
    return "\n".join(kept) + "\n" if kept else ""
