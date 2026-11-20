"""IMAGE_ID agreement between an exported model and COLMAP's own database.

``point_triangulator`` will not accept a model whose IMAGE_IDs differ from the
ones in the database it is handed, and it says so as "Check failed:
existing_frame.DataIds() == frame.DataIds()" -- a message about rig bookkeeping
that names nothing of the cause. The export used to guess the numbering by
sorting on filename, on the premise that COLMAP's extractor walks a directory
that way. It does not: measured against 4.1 on the eight photographs of
rearmod3, the extractor inserted ``IMG_6404`` as id 4 and ``IMG_6403`` as id 5,
because its reader is threaded.

So the numbering is read out of the database instead, and these tests pin that
it is read rather than assumed -- including the case the old code got right by
accident, where the database happens to agree with filename order.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_colmapids.py
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy import export  # noqa: E402
from multispy.solve import solve  # noqa: E402
from test_synthetic import build_project  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="multispy_colmapids_"))
_COUNTER = [0]

# enough of COLMAP's schema to be the thing being read; the real table carries
# camera_id and the pose-prior columns too, and nothing here looks at them
SCHEMA = """
CREATE TABLE images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL
)
"""


def case() -> Path:
    _COUNTER[0] += 1
    out = TMP / f"case{_COUNTER[0]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def solved():
    """A solved synthetic project. The photographs are deliberately not on
    disk: :func:`export.upright_images` then writes no copies and the model
    names the originals, which is the shortest fixture that still exercises
    the naming the ids are keyed on."""
    project, _ = build_project()
    report = solve(project)
    assert report.ok, "the fixture must solve before anything can be exported"
    return project


def names_in_order(project) -> list[str]:
    return sorted(
        (Path(im.path).name for im in project.images if im.R is not None),
        key=str.lower,
    )


def write_database(path: Path, ids: dict[str, int]) -> Path:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, 1)",
            [(i, n) for n, i in ids.items()],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def model_ids(sparse: Path) -> dict[str, int]:
    """``name -> IMAGE_ID`` as ``images.txt`` has it."""
    found = {}
    lines = [
        ln
        for ln in (sparse / "images.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    for header in lines[::2]:
        parts = header.split()
        found[parts[9]] = int(parts[0])
    return found


def tracks(sparse: Path) -> list[tuple[int, int, int]]:
    """``(point3D_id, image_id, point2D_idx)`` for every track entry."""
    out = []
    for ln in (sparse / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.split()
        pid = int(parts[0])
        rest = parts[8:]
        for i in range(0, len(rest) - 1, 2):
            out.append((pid, int(rest[i]), int(rest[i + 1])))
    return out


def observations(sparse: Path) -> dict[int, list[int]]:
    """``image_id -> POINT3D_ID per 2D point``, in the order written."""
    lines = [
        ln
        for ln in (sparse / "images.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    out = {}
    for header, points2d in zip(lines[::2], lines[1::2]):
        parts = points2d.split()
        out[int(header.split()[0])] = [int(parts[i]) for i in range(2, len(parts), 3)]
    return out


# --------------------------------------------------------------------------
# the bug itself
# --------------------------------------------------------------------------


def test_the_ids_come_from_the_database_not_from_the_filenames():
    """The rearmod3 failure, reproduced: a database that numbers two
    photographs the other way round from their names."""
    project = solved()
    out = case()
    names = names_in_order(project)
    assert len(names) >= 5

    scrambled = dict(zip(names, range(1, len(names) + 1)))
    scrambled[names[3]], scrambled[names[4]] = scrambled[names[4]], scrambled[names[3]]
    db = write_database(out / "database.db", scrambled)

    export.export_colmap(project, out / "sparse", out / "images", database=db)
    got = model_ids(out / "sparse")
    assert got == scrambled, (
        "the model numbered the photographs its own way; point_triangulator "
        f"would reject it.\n  database: {scrambled}\n  model:    {got}"
    )


def test_the_database_is_believed_even_when_it_skips_numbers():
    """COLMAP's ids are an autoincrement column, so a database that has had an
    image removed is sparsely numbered. Filename order cannot produce that."""
    project = solved()
    out = case()
    names = names_in_order(project)
    sparse_ids = {n: 10 * (i + 1) for i, n in enumerate(names)}
    db = write_database(out / "database.db", sparse_ids)

    export.export_colmap(project, out / "sparse", out / "images", database=db)
    assert model_ids(out / "sparse") == sparse_ids


def test_the_tracks_follow_the_renumbering():
    """A point3D's track names IMAGE_IDs too, and an id fixed in one file and
    not the other is worse than the bug -- the model would load and describe
    the wrong photographs."""
    project = solved()
    out = case()
    names = names_in_order(project)
    reversed_ids = dict(zip(names, range(len(names), 0, -1)))
    db = write_database(out / "database.db", reversed_ids)

    export.export_colmap(project, out / "sparse", out / "images", database=db)
    ids = set(model_ids(out / "sparse").values())
    obs = observations(out / "sparse")
    entries = tracks(out / "sparse")
    assert entries, "the fixture triangulated nothing, so nothing is being checked"

    for pid, image_id, idx in entries:
        assert image_id in ids, f"track names IMAGE_ID {image_id}, which the model has no line for"
        assert idx < len(obs[image_id]), (
            f"track names 2D point {idx} of image {image_id}, which has {len(obs[image_id])}"
        )
        assert obs[image_id][idx] == pid, (
            f"point {pid}'s track points at 2D point {idx} of image {image_id}, "
            f"which belongs to point {obs[image_id][idx]} -- the ids and the "
            f"observation lists disagree"
        )


def test_agreeing_with_filename_order_is_not_special():
    """The case the old code got right. It has to keep working, and for the
    right reason: the ids are still read, they just happen to match."""
    project = solved()
    out = case()
    names = names_in_order(project)
    in_order = dict(zip(names, range(1, len(names) + 1)))
    db = write_database(out / "database.db", in_order)

    export.export_colmap(project, out / "sparse", out / "images", database=db)
    assert model_ids(out / "sparse") == in_order


def test_without_a_database_the_old_guess_is_still_what_is_written():
    """There is no database until the photographs have been exported and fed
    to the extractor, so the first export has to write something. Filename
    order is that something, and it is pinned here so the fallback is a
    decision rather than an accident."""
    project = solved()
    out = case()
    names = names_in_order(project)

    export.export_colmap(project, out / "sparse", out / "images")
    assert model_ids(out / "sparse") == dict(zip(names, range(1, len(names) + 1)))


# --------------------------------------------------------------------------
# the ways it can be pointed at the wrong database
# --------------------------------------------------------------------------


def test_a_database_that_does_not_know_a_photograph_is_refused():
    """The common mistake: features extracted from the originals rather than
    from the upright copies, so the names do not line up. Numbering around the
    gap would be silently wrong, so it is an error instead."""
    project = solved()
    out = case()
    names = names_in_order(project)
    partial = dict(zip(names[:-1], range(1, len(names))))
    db = write_database(out / "database.db", partial)

    try:
        export.export_colmap(project, out / "sparse", out / "images", database=db)
    except ValueError as exc:
        assert names[-1] in str(exc), f"the error does not name the missing photograph: {exc}"
    else:
        raise AssertionError("a database missing a photograph was accepted")


def test_a_database_that_is_not_there_is_refused():
    """``sqlite3.connect`` would create an empty one and the next line would
    fail on a missing table, which reads as a corrupt database rather than a
    wrong path."""
    project = solved()
    out = case()
    try:
        export.export_colmap(
            project, out / "sparse", out / "images", database=out / "nothing.db"
        )
    except FileNotFoundError as exc:
        assert "nothing.db" in str(exc)
    else:
        raise AssertionError("a path with no database at it was accepted")
    assert not (out / "nothing.db").exists(), "a database was created by being read"


def test_a_database_can_hold_more_images_than_the_model():
    """Extraction runs over a directory; a project may have solved only some of
    what is in it, and the extras are simply not ours."""
    project = solved()
    out = case()
    names = names_in_order(project)
    extra = dict(zip(names, range(1, len(names) + 1)))
    extra["someone_elses_photo.jpg"] = len(names) + 1
    db = write_database(out / "database.db", extra)

    export.export_colmap(project, out / "sparse", out / "images", database=db)
    got = model_ids(out / "sparse")
    assert "someone_elses_photo.jpg" not in got
    assert got == {n: extra[n] for n in names}


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
