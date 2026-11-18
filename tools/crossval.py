"""Did that change help the solve, or only help the number?

Reprojection RMS cannot answer this and keeps being asked to.  It is measured
over the marks that survived the solve, so every variant is scored on a
different population, and the cheapest way to improve it is to delete the
evidence that disagreed.  :mod:`multispy.cadguard` was written after that
mistake cost a real answer -- "improves, 1.99 to 1.66 px" over 57 marks, where
the honest figure over the original 67 was 13.09 px -- and the same trap is
waiting behind every cull threshold, every deleted body and every discarded
photograph.

So this measures the other way round.  A slice of the anchor marks is held out
of the solve entirely -- not culled, never offered, the solver does not know
they exist -- each variant runs on what is left, and the held-out marks are
then reprojected through the cameras and body poses that came out.  Those marks
cannot be flattered by a cull, because no variant ever had the chance to drop
them.

The evaluation population is fixed once, up front, across every variant.  If
any variant deletes a body, a photograph or an anchor, that material is barred
from scoring in *all* of them, including the baseline.  Otherwise a run is
rewarded simply for no longer being asked the hard questions, which is the
original mistake wearing a different hat.

    .venv\\Scripts\\python.exe tools\\crossval.py projects/rearmod3
    .venv\\Scripts\\python.exe tools\\crossval.py projects/rearmod3 --cull-sweep
    .venv\\Scripts\\python.exe tools\\crossval.py projects/rearmod3 \\
        --drop-bodies obj6,obj27 --drop-images img0

Read the two columns together.  ``in-solve`` is what the app shows; ``held-out``
is what the solve is worth.  When they move apart, believe the second one.  On
the rig fixture, culling from nothing to a fifth of the marks took in-solve RMS
from 14.38 px to 5.30 px while held-out error stayed between 8.46 and 8.98 --
a 63% improvement in the number and none at all in the answer.

A caution the sweep taught, worth keeping in view: hand-picked drops bypass the
starvation guards in :func:`multispy.solve._cull_outliers`, which exist because
a track needs two views, an image two marks and a body three anchors.  Combining
several drops on that project produced the best in-solve figure of any run,
4.41 px, with a held-out error of 50 px -- a handful of bodies starved below
three anchors and their poses went anywhere, while the median mark looked fine.
Watch the ``>30px`` column for it; a mean alone will not show it to you.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multispy import solve as S  # noqa: E402
from multispy.geometry import Intrinsics, project as project_points  # noqa: E402
from multispy.project import Project  # noqa: E402

BLOWUP_PX = 30.0
"""Above this a held-out mark is not a worse fit, it is a different answer."""


# ---------------------------------------------------------------------------
# holding marks out
# ---------------------------------------------------------------------------


def holdout(project: Project, fold: int, folds: int, seed: int = 1234):
    """One fold of the anchor marks, chosen per (image, body) block.

    Blocks matter.  Taking a flat random sample would sometimes strip a body in
    one photograph down to nothing, and a starved block fits badly for reasons
    that have nothing to do with the variant being tested -- it would charge
    every row the same unfair penalty and hide the real differences underneath
    it.  So no block gives up more than a third of its marks, and none is left
    below three.
    """
    rng = random.Random(seed + fold)
    blocks: dict[tuple[str, str], list[str]] = {}
    for im in project.images:
        for key in im.anchor_obs:
            blocks.setdefault((im.id, key.split("/")[0]), []).append(key)

    held: list[tuple[str, str]] = []
    for (iid, _obj), keys in sorted(blocks.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        room = min(len(keys) // 3, max(len(keys) - 3, 0))
        held += [(iid, k) for i, k in enumerate(keys) if i % folds == fold][:room]
    return held


def apply_variant(project, held, *, bodies=(), images=(), anchors=(), cull=None):
    """Strip the held-out fold, then whatever this variant drops on top."""
    if cull is not None:
        project.settings.cull_sigma = float(cull)
    for iid, key in held:
        im = project.image(iid)
        if im is not None:
            im.anchor_obs.pop(key, None)
    for im in project.images:
        if im.id in images:
            im.anchor_obs.clear()
            im.observations.clear()
            im.enabled = False
            continue
        for key in [k for k in im.anchor_obs
                    if k.split("/")[0] in bodies or k.split("/", 1)[1] in anchors]:
            im.anchor_obs.pop(key, None)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def evaluate(project: Project, held, observed, scorable) -> list[float]:
    """Reproject the held-out marks through whatever the solve produced.

    Gauge-free by construction: the scene's origin, orientation and scale are
    only ever fixed by constraints, and may land differently from one variant
    to the next, but a camera and the bodies it sees move together -- so a
    reprojection in pixels means the same thing in every run.
    """
    errs: list[float] = []
    for iid, key in held:
        if not scorable(iid, key):
            continue
        obj, anchor = key.split("/", 1)
        im = project.image(iid)
        inst = project.instance(obj)
        if im is None or im.R is None or inst is None or inst.R is None:
            continue
        template = project.templates.get(inst.template)
        if template is None:
            continue
        xyz = next((a.xyz for a in template.anchors if a.name == anchor), None)
        if xyz is None:
            continue
        world = np.asarray(inst.R) @ (np.asarray(xyz) * template.unit_scale) + np.asarray(inst.t)
        g = project.groups[im.group]
        uv = project_points(
            np.asarray(im.R), np.asarray(im.t),
            Intrinsics(g.f, g.cx, g.cy, g.k1), world[None, :],
        )[0]
        if np.all(np.isfinite(uv)):
            errs.append(float(np.linalg.norm(uv - observed[(iid, key)])))
    return errs


def run(path: Path, variants, folds: int, seed: int):
    base = Project.load(str(path))
    observed = {
        (im.id, key): np.asarray(v, dtype=float)
        for im in base.images
        for key, v in im.anchor_obs.items()
    }
    if not observed:
        raise SystemExit("this project has no anchor marks -- nothing to hold out")

    # everything any variant removes, barred from scoring in all of them
    dead_bodies = {b for v in variants for b in v.get("bodies", ())}
    dead_images = {i for v in variants for i in v.get("images", ())}
    dead_anchors = {a for v in variants for a in v.get("anchors", ())}

    def scorable(iid: str, key: str) -> bool:
        obj, anchor = key.split("/", 1)
        return (obj not in dead_bodies and iid not in dead_images
                and anchor not in dead_anchors)

    folds_held = [holdout(base, f, folds, seed) for f in range(folds)]
    n_scored = sum(1 for held in folds_held for iid, k in held if scorable(iid, k))
    print(f"{path.name}: {len(observed)} anchor marks, {folds} folds, "
          f"{n_scored} held-out marks scored per variant")
    if dead_bodies or dead_images or dead_anchors:
        barred = ", ".join(sorted(dead_bodies | dead_images | dead_anchors))
        print(f"barred from scoring in every variant, because some variant drops it: {barred}")
    print()

    header = f"{'variant':<34}{'held-out rms':>13}{'median':>9}{'in-solve':>10}{'>30px':>7}  change"
    print(header)
    print("-" * len(header))

    reference = None
    for variant in variants:
        name = variant["name"]
        errs: list[float] = []
        in_solve: list[float] = []
        for held in folds_held:
            project = Project.load(str(path))
            apply_variant(
                project, held,
                bodies=variant.get("bodies", ()),
                images=variant.get("images", ()),
                anchors=variant.get("anchors", ()),
                cull=variant.get("cull"),
            )
            report = S.solve(project, run_bundle=True)
            in_solve.append(report.rms_reprojection_px)
            errs += evaluate(project, held, observed, scorable)

        e = np.array(errs) if errs else np.zeros(1)
        rms = math.sqrt(float((e ** 2).mean()))
        if reference is None:
            reference = rms
            change = "--"
        else:
            change = f"{100 * (rms / reference - 1):+.1f}%"
        print(f"{name:<34}{rms:>13.2f}{float(np.median(e)):>9.2f}"
              f"{float(np.mean(in_solve)):>10.2f}{int((e > BLOWUP_PX).sum()):>7}  {change}")

    print("\nthe first row is the reference; held-out rms is the column to believe")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("project", type=Path, help="project folder or .multispy file")
    ap.add_argument("--folds", type=int, default=4, help="hold-out folds (default: 4)")
    ap.add_argument("--seed", type=int, default=1234, help="fold seed (default: 1234)")
    ap.add_argument("--cull-sweep", action="store_true",
                    help="compare cull thresholds from none down to 2 sigma")
    ap.add_argument("--drop-bodies", default="",
                    help="comma-separated instance ids to test deleting")
    ap.add_argument("--drop-images", default="",
                    help="comma-separated image ids to test deleting")
    ap.add_argument("--drop-anchors", default="",
                    help="comma-separated anchor names to test deleting, every body")
    args = ap.parse_args()

    path = args.project
    if path.is_dir():
        found = sorted(path.glob("*.multispy"))
        if not found:
            raise SystemExit(f"no .multispy file in {path}")
        path = found[0]

    def split(s):
        return tuple(x.strip() for x in s.split(",") if x.strip())

    variants = [{"name": "as saved"}]
    if args.cull_sweep:
        variants = [
            {"name": "no culling at all", "cull": 1e9},
            {"name": "cull 5 sigma", "cull": 5.0},
            {"name": "cull 4 sigma", "cull": 4.0},
            {"name": "cull 3 sigma", "cull": 3.0},
            {"name": "cull 2.5 sigma", "cull": 2.5},
            {"name": "cull 2 sigma", "cull": 2.0},
        ]
    if args.drop_bodies:
        variants.append({"name": f"drop bodies {args.drop_bodies}",
                         "bodies": split(args.drop_bodies)})
    if args.drop_images:
        variants.append({"name": f"drop photos {args.drop_images}",
                         "images": split(args.drop_images)})
    if args.drop_anchors:
        variants.append({"name": f"drop anchors {args.drop_anchors}",
                         "anchors": split(args.drop_anchors)})
    if len(variants) == 1 and not args.cull_sweep:
        print("nothing to compare against -- add --cull-sweep or a --drop-* option\n")

    run(path, variants, max(args.folds, 2), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
