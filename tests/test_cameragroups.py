"""Putting each photograph in the camera group its EXIF says it belongs to.

Run directly::

    .venv\\Scripts\\python.exe tests\\test_cameragroups.py

A camera group is a promise that these photographs share intrinsics, and the
whole method leans on it: one focal length constrained by every image's control
lines beats N focals each constrained by four clicks. Every import path used to
write ``group="cam0"`` and leave it, so a set shot across a phone's wide, main
and telephoto lenses made that promise falsely -- and a bundle given a promise
it cannot keep puts the difference into camera positions and point depths,
where it reads as a solve that will not converge rather than as a grouping
mistake.

So the properties worth defending are in two halves. That a genuinely mixed set
is split (and that the solve says so when it meets one already made), and --
the half more likely to break something -- that nothing else moves: a
single-lens import still lands in one group called ``cam0``, a set with no EXIF
at all behaves exactly as it did, and an old project's ``cam0`` is adopted
rather than duplicated beside.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy import exif as X  # noqa: E402
from multispy.project import CameraGroup, ImageView, Project  # noqa: E402
from multispy.solve import SolveReport, _check_camera_groups  # noqa: E402
from multispy.webapp import add_images_from_folder  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="multispy_groups_"))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def photo(folder: Path, name: str, width: int, height: int, f35: float | None) -> Path:
    """A JPEG that claims a 35mm-equivalent focal length, or claims nothing."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    img = Image.new("RGB", (width, height), (90, 110, 130))
    if f35 is None:
        img.save(path, quality=60)
        return path
    data = Image.Exif()
    data[X.TAG_MAKE] = "TestCam"
    data[X.TAG_MODEL] = "Model X"
    data[X.TAG_EXIF_IFD] = {
        X.TAG_FOCAL_35MM: int(f35),
        X.TAG_EXIF_WIDTH: width,
        X.TAG_EXIF_HEIGHT: height,
    }
    img.save(path, exif=data, quality=60)
    return path


def project_in(folder: Path) -> Project:
    p = Project(name=folder.name)
    p.path = folder / "project.multispy"
    return p


def groups_of(project: Project) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for im in project.images:
        out.setdefault(im.group, []).append(im.id)
    return out


# --------------------------------------------------------------- the split

def test_mixed_lenses_split() -> None:
    """Three lenses in one folder become three groups, and nothing is lost."""
    folder = TMP / "mixed"
    for i, f35 in enumerate([13, 13, 24, 24, 24, 77]):
        photo(folder, f"img{i}.jpg", 4032, 3024, f35)
    p = project_in(folder)
    added = add_images_from_folder(p, folder)

    by_group = groups_of(p)
    check("mixed set: every photo imported", len(added) == 6, f"{len(added)} of 6")
    check("mixed set: three camera groups", len(by_group) == 3, str(sorted(by_group)))
    sizes = sorted(len(v) for v in by_group.values())
    check("mixed set: split 2/3/1 by lens", sizes == [1, 2, 3], str(sizes))
    check(
        "mixed set: every group is registered on the project",
        set(by_group) <= set(p.groups),
        f"{sorted(by_group)} vs {sorted(p.groups)}",
    )
    # the point of the whole exercise: no group is one the existing spread
    # check would turn round and reject as zoomed
    for gid in by_group:
        s = X.summarise_group(p, gid)
        check(
            f"mixed set: {gid} passes the zoom check",
            s.spread_pct is None or s.spread_pct <= 2.0,
            f"spread {s.spread_pct}",
        )


def test_telephoto_group_keeps_its_own_focal() -> None:
    """Each group's EXIF focal is its own, not the set's median."""
    folder = TMP / "focals"
    photo(folder, "wide.jpg", 4032, 3024, 13)
    photo(folder, "main.jpg", 4032, 3024, 24)
    p = project_in(folder)
    add_images_from_folder(p, folder)

    focals = {}
    for gid in sorted(groups_of(p)):
        focals[gid] = X.summarise_group(p, gid).focal_px
    values = sorted(v for v in focals.values() if v)
    ok = len(values) == 2 and values[1] / values[0] > 1.7
    check("two lenses keep two focals", ok, str([round(v) for v in values]))


# ------------------------------------------------- and nothing else moves

def test_single_lens_stays_one_group() -> None:
    """The ordinary case is untouched, and the group is still called cam0."""
    folder = TMP / "single"
    for i in range(5):
        photo(folder, f"img{i}.jpg", 4032, 3024, 24)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    by_group = groups_of(p)
    check("single lens: one group", len(by_group) == 1, str(sorted(by_group)))
    check("single lens: it is cam0", list(by_group) == ["cam0"], str(list(by_group)))


def test_no_exif_behaves_as_before() -> None:
    """Nothing to group by means the old default, not a group each."""
    folder = TMP / "bare"
    for i in range(4):
        photo(folder, f"img{i}.jpg", 3000, 2000, None)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    by_group = groups_of(p)
    check("no EXIF: still one group", len(by_group) == 1, str(sorted(by_group)))
    check("no EXIF: still cam0", list(by_group) == ["cam0"], str(list(by_group)))


def test_rotation_does_not_split() -> None:
    """Portrait and landscape off one lens are one camera.

    The diagonal is unchanged by a quarter turn, and the focal in pixels is
    defined on the diagonal, so this must not split -- it is the case that
    would quietly halve the evidence behind every focal length on a phone.
    """
    folder = TMP / "rotated"
    photo(folder, "land.jpg", 4032, 3024, 24)
    photo(folder, "port.jpg", 3024, 4032, 24)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    check("rotation does not split", len(groups_of(p)) == 1, str(sorted(groups_of(p))))


def test_resolution_does_split() -> None:
    """One lens at two resolutions is two groups, because `f` is in pixels.

    The same 24 mm equivalent on a 4032px frame and a 2016px one is two
    different focal lengths in pixels, and a group shares the pixel figure --
    so this is a real split rather than an over-eager one.
    """
    folder = TMP / "resolutions"
    photo(folder, "big.jpg", 4032, 3024, 24)
    photo(folder, "small.jpg", 2016, 1512, 24)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    check("two resolutions split", len(groups_of(p)) == 2, str(sorted(groups_of(p))))


# ------------------------------------------------------------- continuity

def test_second_import_joins_the_first() -> None:
    """A later import of the same lens lands in the group already made for it."""
    folder = TMP / "twobatches"
    photo(folder, "a0.jpg", 4032, 3024, 24)
    photo(folder, "a1.jpg", 4032, 3024, 13)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    first = dict(groups_of(p))
    before = set(p.groups)

    photo(folder, "b0.jpg", 4032, 3024, 24)
    photo(folder, "b1.jpg", 4032, 3024, 13)
    add_images_from_folder(p, folder)

    check("second import: no new groups", set(p.groups) == before, str(sorted(p.groups)))
    by_group = groups_of(p)
    check("second import: two groups of two", sorted(len(v) for v in by_group.values()) == [2, 2],
          str({k: len(v) for k, v in by_group.items()}))
    for gid, ids in first.items():
        check(f"second import: {gid} kept its photos", set(ids) <= set(by_group.get(gid, [])))


def test_old_project_cam0_is_adopted() -> None:
    """A keyless cam0 from before grouping existed is joined, not duplicated.

    The group carries a solved focal length and the user's locks. Minting
    ``cam1`` beside it for the very same lens would silently throw both away,
    so the key is derived from the photographs already in the group rather
    than imposed on it.
    """
    folder = TMP / "legacy"
    photo(folder, "old0.jpg", 4032, 3024, 24)
    photo(folder, "old1.jpg", 4032, 3024, 24)
    p = project_in(folder)
    # built by hand, the way a project written before grouping existed looks
    for i, name in enumerate(["old0.jpg", "old1.jpg"]):
        p.images.append(ImageView(id=f"img{i}", path=name, width=4032, height=3024))
    p.groups["cam0"] = CameraGroup(id="cam0", f=2795.7, lock_focal=True)

    photo(folder, "new.jpg", 4032, 3024, 24)
    add_images_from_folder(p, folder)

    by_group = groups_of(p)
    check("legacy cam0: still one group", len(by_group) == 1, str(sorted(by_group)))
    check("legacy cam0: the new photo joined it", len(by_group.get("cam0", [])) == 3,
          str({k: len(v) for k, v in by_group.items()}))
    check("legacy cam0: the solved focal survived", p.groups["cam0"].f == 2795.7)
    check("legacy cam0: the lock survived", p.groups["cam0"].lock_focal is True)


def test_already_mixed_cam0_is_not_joined() -> None:
    """A group that is already mixed does not adopt a key, so nothing joins it.

    Landing a new photograph beside the mess is recoverable; landing it inside
    one and calling the result a camera is not.
    """
    folder = TMP / "legacymixed"
    for i, (name, f35) in enumerate([("m0.jpg", 24), ("m1.jpg", 77)]):
        photo(folder, name, 4032, 3024, f35)
        p_img = ImageView(id=f"img{i}", path=name, width=4032, height=3024)
        if i == 0:
            p = project_in(folder)
            p.images = []
        p.images.append(p_img)
    p.groups["cam0"] = CameraGroup(id="cam0")

    photo(folder, "later.jpg", 4032, 3024, 24)
    add_images_from_folder(p, folder)
    check("mixed cam0 takes no key", p.groups["cam0"].exif_key is None)
    check("mixed cam0: the new photo went elsewhere",
          groups_of(p).get("cam0") == ["img0", "img1"], str(groups_of(p)))


def test_regrouping_an_existing_project() -> None:
    """The repair path: one big mixed cam0, regrouped in place.

    This is where every project that needs this feature actually is --
    auto-grouping on import only helps the next set of photographs. The ids
    should come out tidy rather than marching up from cam1 and leaving an
    empty cam0 behind, the biggest camera should keep the familiar name, and
    the settings on the group that turned out to hold three cameras must not
    be inherited by whichever of them lands on its id.
    """
    folder = TMP / "regroup"
    plan = [24, 24, 24, 24, 13, 13, 77]
    for i, f35 in enumerate(plan):
        photo(folder, f"g{i}.jpg", 4032, 3024, f35)
    p = project_in(folder)
    for i, name in enumerate(sorted(f"g{i}.jpg" for i in range(len(plan)))):
        p.images.append(ImageView(id=f"img{i}", path=name, width=4032, height=3024))
    # a mixed group carrying a focal nobody should inherit
    p.groups["cam0"] = CameraGroup(id="cam0", f=1234.0, lock_focal=True)

    moved = X.assign_camera_groups(p)
    by_group = groups_of(p)
    check("regroup: three groups", len(by_group) == 3, str(sorted(by_group)))
    check("regroup: no empty group left", set(p.groups) == set(by_group),
          f"{sorted(p.groups)} vs {sorted(by_group)}")
    check("regroup: ids start at cam0", sorted(by_group) == ["cam0", "cam1", "cam2"],
          str(sorted(by_group)))
    check("regroup: the biggest camera keeps cam0", len(by_group["cam0"]) == 4,
          str({k: len(v) for k, v in by_group.items()}))
    check("regroup: the mixed group's focal is not inherited",
          p.groups["cam0"].f is None and p.groups["cam0"].lock_focal is False,
          f"f={p.groups['cam0'].f} lock={p.groups['cam0'].lock_focal}")
    check("regroup: every group has a key", all(g.exif_key for g in p.groups.values()))
    check("regroup: it reports what it moved", len(moved) == 3, str(len(moved)))

    # and it is idempotent: running it again changes nothing
    again = X.assign_camera_groups(p)
    check("regroup: idempotent", not again, str(again))

    clean = SolveReport()
    _check_camera_groups(p, clean)
    check("regroup: the solve stops warning", not clean.warnings, str(clean.warnings[:1]))


def test_regrouping_a_solved_project() -> None:
    """A project that has been solved regroups too.

    Worth its own case because the images then carry numpy arrays, and any
    comparison of whole ImageViews -- `im in images` -- turns into an array
    truth test somewhere inside an import. Which is a ValueError out of the
    middle of adding a photograph, with nothing on screen to connect it to
    grouping at all.
    """
    import numpy as np

    folder = TMP / "solvedregroup"
    for i, f35 in enumerate([24, 24, 77]):
        photo(folder, f"s{i}.jpg", 4032, 3024, f35)
    p = project_in(folder)
    for i in range(3):
        im = ImageView(id=f"img{i}", path=f"s{i}.jpg", width=4032, height=3024)
        im.R = np.eye(3) * (i + 1)
        im.t = np.array([float(i), 0.0, 0.0])
        p.images.append(im)
    p.groups["cam0"] = CameraGroup(id="cam0")
    try:
        moved = X.assign_camera_groups(p)
        check("solved project: regroups without raising", len(moved) == 1, str(moved))
        check("solved project: two groups", len(groups_of(p)) == 2, str(sorted(groups_of(p))))
    except Exception as exc:  # noqa: BLE001 - the thing being guarded against
        check("solved project: regroups without raising", False, f"{type(exc).__name__}: {exc}")


def test_key_survives_a_round_trip() -> None:
    """The key is written to the file, so the next import still matches it."""
    folder = TMP / "roundtrip"
    photo(folder, "r0.jpg", 4032, 3024, 24)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    key = p.groups["cam0"].exif_key
    p.save()
    again = Project.load(p.path)
    check("key is written and read back", again.groups["cam0"].exif_key == key, str(key))


# ------------------------------------------------- naming an existing mess

def test_solve_warns_about_a_mixed_group() -> None:
    """The solve says what it found, and says nothing when there is nothing."""
    folder = TMP / "warn"
    for i, f35 in enumerate([24, 24, 24, 77]):
        photo(folder, f"w{i}.jpg", 4032, 3024, f35)
    p = project_in(folder)
    add_images_from_folder(p, folder)

    clean = SolveReport()
    _check_camera_groups(p, clean)
    check("correctly grouped: no warning", not clean.warnings, str(clean.warnings[:1]))

    # now undo the grouping, the way a project made before this existed looks
    for im in p.images:
        im.group = "cam0"
    mixed = SolveReport()
    _check_camera_groups(p, mixed)
    said = " ".join(mixed.warnings)
    check("mixed group: warns", len(mixed.warnings) == 1, str(len(mixed.warnings)))
    check("mixed group: names both cameras", "2 different cameras" in said, said[:90])
    check("mixed group: counts them", "3 at " in said and "1 at " in said, said[:140])


def test_cluster_group_ignores_the_unknowable() -> None:
    """A photo with no EXIF is not evidence of another camera."""
    folder = TMP / "cluster"
    photo(folder, "c0.jpg", 4032, 3024, 24)
    photo(folder, "c1.jpg", 4032, 3024, 24)
    photo(folder, "c2.jpg", 4032, 3024, None)
    p = project_in(folder)
    add_images_from_folder(p, folder)
    for im in p.images:
        im.group = "cam0"
    clusters = X.cluster_group(p, "cam0")
    check("unknowable EXIF makes no cluster", len(clusters) == 1, str(len(clusters)))
    check("unknowable EXIF is left out", len(clusters[0][1]) == 2, str(clusters[0][1]))


def main() -> int:
    for fn in [
        test_mixed_lenses_split,
        test_telephoto_group_keeps_its_own_focal,
        test_single_lens_stays_one_group,
        test_no_exif_behaves_as_before,
        test_rotation_does_not_split,
        test_resolution_does_split,
        test_second_import_joins_the_first,
        test_old_project_cam0_is_adopted,
        test_already_mixed_cam0_is_not_joined,
        test_regrouping_an_existing_project,
        test_regrouping_a_solved_project,
        test_key_survives_a_round_trip,
        test_solve_warns_about_a_mixed_group,
        test_cluster_group_ignores_the_unknowable,
    ]:
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all camera-group checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
