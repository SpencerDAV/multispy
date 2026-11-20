"""Exports against photographs that are not stored the way they are shown.

Every mark in a project is in the frame the browser draws, which is the frame
*after* EXIF orientation has been applied. Nothing downstream applies it:
measured against COLMAP 3.13 and Blender 5.1.1, both of which read a photograph
stored 4032x3024 with orientation 6 as 4032x3024, and ``ns-process-data``
drives COLMAP. So for a phone photo shot in portrait the exported model and the
imagery it describes used to differ by a quarter turn, and the result looked
like a bad solve rather than a bad export.

The fix writes the pixels rather than adjusting the numbers, because a file
carrying no orientation tag reads the same whether or not the reader honours
one. These tests pin both halves of that: a rotated project gets copies that a
tag-ignoring reader sees correctly, and an upright project gets none, so
nothing pays for a problem it does not have.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_uprightexport.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy import export  # noqa: E402
from multispy.solve import solve  # noqa: E402
from test_synthetic import HEIGHT, WIDTH, build_project  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="multispy_upright_"))
_COUNTER = [0]

ORIENT_ROTATE_90_CW = 6  # "the stored pixels are a quarter turn from what you see"


def write_photo(path: Path, size, orientation: int | None, tint=(90, 110, 130)):
    """A JPEG of ``size`` *as stored*, carrying ``orientation`` if given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, tint)
    if orientation is None:
        img.save(path, quality=70)
        return path
    tags = Image.Exif()
    tags[274] = orientation
    tags[0x8769] = {37386: 6.86}  # focal length, in the Exif sub-IFD
    img.save(path, exif=tags, quality=70)
    return path


def solved_project(photos: str = "rotated"):
    """A solved synthetic project whose photographs are real files on disk.

    ``photos`` is ``"rotated"`` (stored a quarter turn from the project's own
    frame, as every portrait phone photo is), ``"upright"``, or ``"missing"``
    for the case where the files are simply not there.
    """
    _COUNTER[0] += 1
    out = TMP / f"case{_COUNTER[0]}"
    out.mkdir(parents=True, exist_ok=True)
    project, _ = build_project()
    report = solve(project)
    assert report.ok, "the fixture must solve before anything can be exported"
    project.path = out / "p.multispy"

    for im in project.images:
        if photos == "missing":
            continue
        path = out / im.path
        if photos == "rotated":
            # stored portrait, shown landscape: exactly the rearmod3 situation
            write_photo(path, (HEIGHT, WIDTH), ORIENT_ROTATE_90_CW)
            with Image.open(path) as raw:
                assert raw.size == (HEIGHT, WIDTH), "the fixture must be stored rotated"
                shown = ImageOps.exif_transpose(raw).size
            assert shown == (im.width, im.height), (
                f"fixture does not reproduce the bug: shown {shown}, "
                f"project {(im.width, im.height)}"
            )
        else:
            write_photo(path, (im.width, im.height), None)
    return project, out


def images_txt_names(sparse: Path) -> list[str]:
    names = []
    for line in (sparse / "images.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[9].lower().endswith(".jpg"):
            names.append(parts[9])
    return names


# --------------------------------------------------------------------------
# the bug itself
# --------------------------------------------------------------------------


def test_a_rotated_photograph_is_written_upright():
    project, out = solved_project("rotated")
    export.export_colmap(project, out / "sparse", out / "images")

    names = images_txt_names(out / "sparse")
    assert len(names) == len([im for im in project.images if im.R is not None])
    for im in project.images:
        if im.R is None:
            continue
        copy = out / "images" / Path(im.path).name
        assert copy.exists(), f"{copy.name} was named but not written"
        with Image.open(copy) as got:
            # the whole point: read with no regard for EXIF, get the frame the
            # marks are in
            assert got.size == (im.width, im.height), (
                f"{copy.name} is stored {got.size}, but the model says "
                f"{(im.width, im.height)} -- a reader ignoring EXIF still disagrees"
            )
            assert got.getexif().get(274) in (None, 1), (
                "the copy still carries an orientation tag, so a reader that "
                "does honour it would now be the one that is wrong"
            )
        assert copy.name in names


def test_an_upright_project_writes_no_copies():
    project, out = solved_project("upright")
    export.export_colmap(project, out / "sparse", out / "images")

    assert not (out / "images").exists(), (
        "a project with nothing to fix paid for the fix anyway"
    )
    expected = {Path(im.path).name for im in project.images if im.R is not None}
    assert set(images_txt_names(out / "sparse")) == expected


def test_a_missing_photograph_keeps_its_old_name():
    """A file that is not there cannot be rewritten, and must not be invented."""
    project, out = solved_project("missing")
    export.export_colmap(project, out / "sparse", out / "images")
    assert not (out / "images").exists()
    expected = {Path(im.path).name for im in project.images if im.R is not None}
    assert set(images_txt_names(out / "sparse")) == expected


# --------------------------------------------------------------------------
# the other two exporters that name photographs
# --------------------------------------------------------------------------


def test_transforms_json_points_at_the_copies():
    project, out = solved_project("rotated")
    export.export_transforms_json(project, out / "transforms.json", out / "images")
    doc = json.loads((out / "transforms.json").read_text(encoding="utf-8"))
    assert doc["frames"]
    for frame in doc["frames"]:
        # nerfstudio resolves file_path against the json's own directory
        target = (out / frame["file_path"]).resolve()
        assert target.exists(), f"{frame['file_path']} does not resolve to a file"
        with Image.open(target) as got:
            assert got.size == (frame["w"], frame["h"]), (
                f"{frame['file_path']} is stored {got.size}, but the frame "
                f"declares {(frame['w'], frame['h'])}"
            )


def test_blender_points_at_the_copies():
    """Asked of the filesystem, not of the spelling.

    The first version of this compared the path in the script against one built
    here, after juggling separators by hand. It passed on the machine it was
    written on and failed on the hosted Windows runner, whose temporary
    directory is the 8.3 short name ``C:\\Users\\RUNNER~1\\...``: ``resolve()``
    expands that and the script's own string does not, so two names for one
    file compared unequal. ``samefile`` asks the filesystem which file is
    meant, which is the question, and is indifferent to short names, case and
    separators alike.
    """
    project, out = solved_project("rotated")
    export.export_blender(project, out / "import_cameras.py", out / "images")
    # checked before samefile is asked anything, so that "no copies at all"
    # reads as a failed assertion rather than as FileNotFoundError from the
    # comparison below
    assert (out / "images").is_dir(), "no upright copies were written at all"
    text = (out / "import_cameras.py").read_text(encoding="utf-8")
    # the template embeds these as JSON, so let JSON unescape them
    quoted = re.findall(r'"image":\s*("(?:[^"\\]|\\.)*")', text)
    named = [Path(json.loads(q)) for q in quoted]
    live = [im for im in project.images if im.R is not None]
    assert len(named) == len(live), f"{len(named)} images named, {len(live)} solved"

    for im, used in zip(live, named):
        assert used.exists(), f"the script names {used}, which is not there"
        assert used.name == Path(im.path).name, (used, im.path)
        assert used.parent.samefile(out / "images"), (
            f"the script points at {used.parent}, not at the upright copies"
        )
        assert not used.parent.samefile((out / im.path).parent), (
            "the script still points at the rotated original"
        )


# --------------------------------------------------------------------------
# the things writing files into one directory can get wrong
# --------------------------------------------------------------------------


def test_photographs_that_share_a_name_do_not_overwrite_each_other():
    """Two folders, one basename. The old export named both the same thing; now
    they have to land in one directory, so a collision would destroy data."""
    project, out = solved_project("rotated")
    live = [im for im in project.images if im.R is not None]
    assert len(live) >= 2
    for i, im in enumerate(live[:2]):
        (out / im.path).unlink()
        im.path = f"shoot{i}/shot.jpg"
        write_photo(out / im.path, (HEIGHT, WIDTH), ORIENT_ROTATE_90_CW, tint=(i * 60, 20, 20))

    export.export_colmap(project, out / "sparse", out / "images")
    names = images_txt_names(out / "sparse")
    assert len(names) == len(set(names)), f"two images share a name in the model: {names}"
    written = sorted(p.name for p in (out / "images").iterdir())
    assert len(written) == len(live), f"a file was overwritten: {written}"
    assert sorted(names) == written


def test_the_copy_keeps_the_focal_length_in_its_metadata():
    """COLMAP's feature extractor seeds its intrinsics from EXIF, so a rotation
    that silently dropped the focal length would make the copies worse than the
    originals in a way nothing else here would notice."""
    project, out = solved_project("rotated")
    export.export_colmap(project, out / "sparse", out / "images")
    assert (out / "images").is_dir(), "no copies were written"
    copy = next((out / "images").iterdir())
    with Image.open(copy) as got:
        assert got.getexif().get_ifd(0x8769).get(37386) == 6.86, (
            "the upright copy lost the focal length the original carried"
        )


def test_re_export_does_not_re_encode():
    project, out = solved_project("rotated")
    export.export_colmap(project, out / "sparse", out / "images")
    assert (out / "images").is_dir(), "no copies were written, so there is nothing to re-encode"
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in (out / "images").iterdir()}
    assert before
    # the three exporters each ask for the copies; only the first may write them
    export.export_colmap(project, out / "sparse", out / "images")
    export.export_transforms_json(project, out / "transforms.json", out / "images")
    export.export_blender(project, out / "import_cameras.py", out / "images")
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in (out / "images").iterdir()}
    assert after == before, "the photographs were re-encoded on a repeat export"


def test_write_exports_reports_the_copies():
    """The front end has to be able to say why the download grew."""
    from multispy.webapp import write_exports

    project, out = solved_project("rotated")
    written = write_exports(project, out / "bundle", formats=("colmap", "transforms"))
    assert "images" in written, "write_exports wrote photographs without saying so"
    assert Path(written["images"]).is_dir()

    project2, out2 = solved_project("upright")
    quiet = write_exports(project2, out2 / "bundle", formats=("colmap", "transforms"))
    assert "images" not in quiet


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
