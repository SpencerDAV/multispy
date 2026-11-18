"""Did adding the CAD bodies make the photographs fit worse?

The failure this exists for is specific and it has happened repeatedly on this
project: the solve is good -- a couple of pixels, nothing culled -- then a model
is dropped in, its anchors are marked, Solve is pressed, and the calibration is
worse than it was. Nothing in the report says so, because the report describes
the solve it just did and has nothing to compare it against. The body is not
obviously wrong, the numbers are not obviously bad, and the good state is gone.

A known object is a much stronger claim than a tie point. A tie point says "this
feature is somewhere"; a body says "these N features are in exactly this rigid
arrangement, to the micron, because a CAD file says so". When the arrangement is
right that is worth more than any number of tie points. When it is wrong -- an
anchor on the wrong corner, a model at the wrong scale, a pose fitted to a
mirror-image reading of a symmetric part -- it is a lie told with enormous
confidence, and the bundle will bend the cameras to accommodate it. The focal
length is the usual casualty: a single mis-marked body has dragged a shared focal
from 2796 px to under 2000, which is why :mod:`solve` grew a focal wall. The wall
stops the collapse. It does not notice the damage below it.

So the rule this module enforces, in the user's words: the CAD solution should
only stand if it does not worsen the point solution **in each image**. Per image,
not overall, because an average lets a body ruin the two photographs it appears
in while twelve others carry the mean.

Measuring it honestly
---------------------
``ImageReport.rms_px`` looks like the number to compare and it is the wrong one,
though not for the reason first supposed. It is not a tie-point figure:
``_fill_statistics.record`` folds anchor marks into it deliberately, and says so
-- "Anchor marks go through here on equal terms with tie points: they are clicks
in that photo, so they belong in that photo's RMS." That is right for the panel
it feeds, and it answers a different question from this one. A body-free solve
has no anchor residuals at all, so the two ``rms_px`` figures are not even
measuring the same population.

The first version of this docstring claimed the mixed reading would be *less*
sensitive, on the grounds that a body's anchors are fitted tightly and would drag
the average down. Measured on the rig fixture, that is false for the cases that
matter most: a mis-marked body's own anchors fit *badly*, so they push the mixed
average up and make the damage look larger, not smaller -- on one spoiled
photograph the mixed reading moved +0.186 px where the tie-point-only reading
moved +0.071. The dilution argument holds only for a body whose anchors fit well
while sitting in the wrong place, which is a real failure but not one measured
here, so it is not claimed.

The reason to exclude anchors is therefore about meaning rather than sensitivity,
and it is the stronger reason. The mixed figure conflates "this body's own marks
are bad" with "this body damaged the point solution". Those call for opposite
responses -- re-click the anchor, versus stop letting the body move the cameras
-- and the user asked for the second. A guard that fires because a body's own
anchors are untidy, in a photograph whose tie points are untouched, is reporting
a fact about the body and calling it a fact about the calibration.

So the comparison is rebuilt from ``SolveReport.residual_lines``, filtered to
labels that are tie point ids, and restricted to the marks that survived the cull
in *both* solves. That last restriction matters for the same reason: if the
body-laden solve culls a mark the body-free one kept, the two RMS figures are
taken over different populations, and dropping the worst mark is an easy way to
look better while being worse.

And that restriction, on its own, was a hole big enough to drive the whole
failure through -- recorded here because it is the most useful thing this module
has learned about itself. Restricting to the intersection makes the *average*
fair and makes a cull **invisible**: a mark the bodies push out of the solve
simply leaves the comparison, instead of counting against them. Asked to judge
PRM1 with 260 restored anchor marks, the first version of this module answered
"improves, 1.99 to 1.66 px, no image worse". The real solve had gone from 2.03 px
to 13.09 px and had culled eleven tie-point marks, three of them on nominal'd
points. The 1.66 px was an average over 57 marks where the 1.99 px was over 67.
It had not improved the fit; it had discarded the evidence that disagreed and
reported the remainder.

So a cull is now damage in its own right. ``newly_culled`` convicts on sight, no
allowance applies to it, and the mark counts either side are reported next to the
pixels with instructions to read them first. A mark the solver has given up on
has not drifted within tolerance -- it is gone, it constrains nothing, and if it
carried a nominal the part space is a link short.

What this does and does not do
------------------------------
It measures, and it can act on the measurement in two ways: put the body-free
cameras back, or take the anchors out of the solve for good.

The second is the one that was missing, and its absence was the whole complaint.
Restoring the cameras is a *snapshot* -- the marks stay live, so the next press
of Solve re-makes the mess it just undid, and the user who has got a good fit
from tie points is back where they started. The persistent answer is
:data:`~multispy.project.Settings.anchors_in_solve` and
:attr:`~multispy.project.ObjectInstance.anchors_off`, which say "these marks are
not evidence about where the cameras are" and keep saying it after a reload.
:func:`mute_bodies` sets that and re-solves.

An earlier version of this docstring said a rejected body could not be placed
against locked cameras, because ``fit_instances`` runs inside the linear
initialisation and works from triangulated anchor positions rather than from a
finished camera solution. That is now :func:`~multispy.solve.place_muted_instances`,
which triangulates each anchor against the final cameras and fits the model to
the result. So a muted body is still placed, still drawn, and still reports its
shape error -- a better shape error than before, in fact, since nothing had the
chance to bend towards it.

What muting is and is not worth, measured
-----------------------------------------
On the rear-module cell (8 photographs, 33 tie points, 28 bodies, 582 anchor
marks) muting every body took the solve from 9.37 px to 2.99 px and from 15
culled marks to 2. That is the damage this module exists to detect, undone.

It did **not** improve the geometry: measured on one yardstick for both arms --
every body's anchors triangulated against that arm's own final cameras and the
model rigid-fitted -- the shape error median moved 3.49 mm to 3.74 mm, and muting
was better on 9 bodies of 23. The remaining error on that dataset is in the
marks, and no weighting or exclusion policy reaches it; see ``TODO.md``.

Both halves of that matter. Muting is the right answer to "the bodies are
wrecking a calibration I had" and the wrong answer to "the bodies disagree with
my CAD by 3 mm". It buys back the camera solution, not accuracy.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from .project import Project

TOLERANCE_PX = 0.25
"""How much worse one image's tie-point RMS may get before the bodies are judged
to have harmed it, as an absolute floor.

A judgement, not a measurement. Two solves of the same data do not land on
identical numbers -- the parameter vector is larger with bodies in it, the
starting point differs, and least squares is not deterministic to the last digit
-- so some allowance is needed or the guard fires on arithmetic. A quarter of a
pixel is well below anything a person could re-click and well above solver
jitter, which on this project runs in the thousandths."""

TOLERANCE_FRACTION = 0.10
"""And a relative allowance, for images that start out poorly.

An image already sitting at 8 px has nothing to gain from being held to a quarter
of a pixel; the absolute floor would make every body look harmful there. The
allowance is the larger of the two, so a sharp image is judged tightly and a
soft one proportionally."""

IMPROVEMENT_PX = 0.05
"""Below this, an overall change is called neither better nor worse.

Reported rather than hidden, because "the bodies changed nothing" is a useful
answer: it means the anchors are consistent with the photographs and are also not
contributing much, which is worth knowing before marking twenty more."""


@dataclass
class ImageVerdict:
    """One photograph's tie-point fit, with the bodies and without them."""

    image: str
    n_points: int = 0
    """Tie-point marks compared. Only those that survived the cull in *both*
    solves, so the two figures describe the same population."""
    before_px: float = 0.0
    """Tie-point-only RMS with no body anchors in the solve at all."""
    after_px: float = 0.0
    """Tie-point-only RMS with them."""
    delta_px: float = 0.0
    """``after - before``. Positive is worse."""
    allowed_px: float = 0.0
    """The allowance in force for this image -- see :data:`TOLERANCE_PX` and
    :data:`TOLERANCE_FRACTION`."""
    worsened: bool = False
    worst_point: str = ""
    """The tie point whose own error grew most in this image. Where to look
    first, and usually an anchor clicked on the wrong feature nearby."""
    worst_point_delta_px: float = 0.0
    newly_culled: list[str] = field(default_factory=list)
    """Tie points this photograph kept without the bodies and lost with them.

    Damage, and the worst kind, which is why it convicts on its own without
    reference to any RMS. A culled mark is a mark the solver gave up on: the
    feature stops constraining anything, and if it carried a nominal the part
    space loses a link. It also *flatters* every average it leaves, since the
    marks that drop out are the ones fitting worst -- which is how the first
    version of this module reported an improvement while ten marks fell out of
    the solve."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CadReport:
    """Whether the known objects earned their place in the solve."""

    ok: bool = False
    reason: str = ""
    verdict: str = ""
    """``"improves"``, ``"harmless"``, ``"worsens"`` or ``"untestable"``."""
    n_instances: int = 0
    n_anchor_marks: int = 0
    n_voting_anchor_marks: int = 0
    """Of those, the ones actually allowed to say where the cameras are.

    Lower than ``n_anchor_marks`` once anything is muted, and the two being
    different is the first thing a reader of this report needs to know: the
    verdict is about the voting ones only."""
    n_muted_anchor_marks: int = 0
    already_muted: bool = False
    """Nothing was voting, so there was no second state to compare against.

    Not a pass. See the ``untestable`` reason this sets."""
    before_rms_px: float = 0.0
    """Overall tie-point-only RMS without bodies. Read the per-image table
    rather than this: the whole point is that an average hides the damage."""
    after_rms_px: float = 0.0
    focal_before_px: float | None = None
    focal_after_px: float | None = None
    """The shared focal either side. Carried because it is the quantity a bad
    body attacks first, and a focal that moved several percent is a stronger
    signal than any single image's RMS."""
    n_marks_before: int = 0
    """Tie-point marks the body-free solve kept."""
    n_marks_after: int = 0
    """And how many survived with the bodies in. A drop is the first thing to
    read: the RMS figures below are averages, and an average taken over fewer
    marks is not comparable with one taken over more however the arithmetic is
    restricted."""
    n_newly_culled: int = 0
    """Tie-point marks lost to the bodies, across all images."""
    images: list[ImageVerdict] = field(default_factory=list)
    """Worst first."""
    worsened: list[str] = field(default_factory=list)
    """Images whose tie-point fit the bodies made worse past the allowance, or
    which lost a tie point to them."""
    worst_image: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["images"] = [v.to_dict() for v in self.images]
        return d


def _tie_ids(project: Project) -> set[str]:
    """Tie point ids, which is also what tells a point residual from an anchor's.

    Taken from the project rather than by looking for the anchor separator in the
    label, because a point id is whatever the user typed and could contain one.
    """
    return {p.id for p in project.points}


def _marks(
    report: Any, tie_ids: set[str]
) -> tuple[dict[str, dict[str, float]], dict[str, set[str]]]:
    """Tie-point marks this solve kept, and the ones it culled.

    Both, because the kept ones alone are not enough to judge a body -- see
    :func:`check_cad_bodies` on why a cull is damage rather than an absence.
    Anchors are excluded from both: that is the correction the module docstring
    describes.
    """
    kept: dict[str, dict[str, float]] = {}
    culled: dict[str, set[str]] = {}
    for row in getattr(report, "residual_lines", None) or []:
        label = row.get("point")
        if label not in tie_ids:
            continue
        iid = row.get("image", "")
        if row.get("culled"):
            culled.setdefault(iid, set()).add(label)
            continue
        err = row.get("error_px")
        if err is None:
            continue
        kept.setdefault(iid, {})[label] = float(err)
    return kept, culled


def _per_image(report: Any, tie_ids: set[str]) -> dict[str, dict[str, float]]:
    """Kept tie-point marks only. Retained for callers that want just those."""
    return _marks(report, tie_ids)[0]


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return float((sum(v * v for v in values) / len(values)) ** 0.5)


def _shared_focal(report: Any) -> float | None:
    focals = getattr(report, "focals", None) or {}
    values = [float(f) for f in focals.values() if f]
    return sum(values) / len(values) if values else None


def body_free_copy(project: Project) -> Project:
    """The project with the bodies' anchors muted rather than deleted.

    The state the user describes as "the good solution" before the models went
    in: the bodies are still in the scene, still drawn, every click still there,
    and they simply stop being evidence about where the cameras are.

    This used to clear ``anchor_obs`` outright, which gave the same tie-point
    residuals -- that is what the comparison reads -- but was not a state the
    user could ever be left in except by deleting their work. Now it sets the
    switch that :func:`mute_bodies` makes permanent, so the "without" column of
    the report is exactly, parameter for parameter, what pressing that button
    produces. A measurement of a state nobody can reach is a worse measurement,
    however equal its arithmetic.
    """
    trial = copy.deepcopy(project)
    trial.path = None          # a trial must never be saved over the project
    trial.settings.anchors_in_solve = False
    return trial


def check_cad_bodies(
    project: Project,
    *,
    tolerance_px: float = TOLERANCE_PX,
    tolerance_fraction: float = TOLERANCE_FRACTION,
) -> CadReport:
    """Solve with the bodies and without them, and compare per image.

    ``project`` is not modified: both solves run on deep copies.
    """
    from .solve import solve      # local, as in nominalcheck: solve imports widely

    rep = CadReport()
    instances = list(getattr(project, "instances", []) or [])
    rep.n_instances = len(instances)
    rep.n_anchor_marks = sum(len(im.anchor_obs or {}) for im in project.active_images())
    muted = project.muted_anchor_keys()
    rep.n_muted_anchor_marks = sum(
        sum(1 for key in im.anchor_obs if key in muted) for im in project.active_images()
    )
    rep.n_voting_anchor_marks = rep.n_anchor_marks - rep.n_muted_anchor_marks

    if not rep.n_anchor_marks:
        rep.verdict = "untestable"
        rep.reason = (
            "no body anchors are marked, so the bodies are not influencing the solve and "
            "there is nothing to compare. This check answers 'did the CAD make the "
            "photographs fit worse', and without anchor marks the answer is no by "
            "construction."
        )
        return rep

    # Already in the body-free state, by the switch rather than by deletion.
    # "harmless" would be the arithmetic's answer here and it would be a lie by
    # omission: both arms would be the same solve, and the user would read a
    # clean bill of health for bodies that are not in the solve to be judged.
    if not rep.n_voting_anchor_marks:
        rep.verdict = "untestable"
        rep.already_muted = True
        rep.reason = (
            f"all {rep.n_anchor_marks} anchor mark(s) are already out of the solve, so this "
            f"*is* the body-free state and there are not two solutions to compare. The marks "
            f"are not lost -- let the bodies vote again and run this to find out what they "
            f"would cost."
        )
        return rep

    tie_ids = _tie_ids(project)
    if not tie_ids:
        rep.verdict = "untestable"
        rep.reason = "no tie points, so there is no point solution for the bodies to worsen"
        return rep

    with_bodies = copy.deepcopy(project)
    with_bodies.path = None
    after_report = solve(with_bodies)
    before_report = solve(body_free_copy(project))

    if not getattr(after_report, "ok", False) or not getattr(before_report, "ok", False):
        rep.verdict = "untestable"
        which = "with the bodies" if not getattr(after_report, "ok", False) else "without them"
        rep.reason = (
            f"the solve {which} did not succeed, so there are not two solutions to compare. "
            f"Fix the solve first; this check cannot say anything about bodies in a project "
            f"that will not solve."
        )
        return rep

    before, culled_before = _marks(before_report, tie_ids)
    after, culled_after = _marks(after_report, tie_ids)
    rep.n_marks_before = sum(len(v) for v in before.values())
    rep.n_marks_after = sum(len(v) for v in after.values())
    rep.focal_before_px = _shared_focal(before_report)
    rep.focal_after_px = _shared_focal(after_report)

    all_before: list[float] = []
    all_after: list[float] = []
    for iid in sorted(set(before) | set(after) | set(culled_after)):
        b, a = before.get(iid, {}), after.get(iid, {})
        shared = sorted(set(b) & set(a))
        # Kept without the bodies, culled with them. Taken before the `shared`
        # guard below, because an image can lose every mark it had and would
        # otherwise drop out of the report entirely -- silence where the damage
        # is total.
        lost = sorted((culled_after.get(iid, set()) - culled_before.get(iid, set())) & set(b))
        if not shared and not lost:
            continue
        v = ImageVerdict(image=iid, n_points=len(shared), newly_culled=lost)
        if not shared:
            v.worsened = True
            rep.images.append(v)
            continue
        v.before_px = _rms([b[p] for p in shared])
        v.after_px = _rms([a[p] for p in shared])
        v.delta_px = v.after_px - v.before_px
        v.allowed_px = max(float(tolerance_px), float(tolerance_fraction) * v.before_px)
        # A lost mark convicts on its own. No allowance applies to it: the
        # tolerances above are for how much a fit may drift, and a mark the
        # solver has given up on has not drifted, it has gone.
        v.worsened = v.delta_px > v.allowed_px or bool(v.newly_culled)
        worst = max(shared, key=lambda p: a[p] - b[p])
        v.worst_point, v.worst_point_delta_px = worst, a[worst] - b[worst]
        rep.images.append(v)
        all_before.extend(b[p] for p in shared)
        all_after.extend(a[p] for p in shared)

    if not rep.images:
        rep.verdict = "untestable"
        rep.reason = (
            "no tie-point mark survived the cull in both solves, so the two have no common "
            "ground to be compared on"
        )
        return rep

    rep.before_rms_px = _rms(all_before)
    rep.after_rms_px = _rms(all_after)
    rep.images.sort(key=lambda v: (-len(v.newly_culled), -v.delta_px))
    rep.worsened = [v.image for v in rep.images if v.worsened]
    rep.n_newly_culled = sum(len(v.newly_culled) for v in rep.images)
    rep.worst_image = rep.images[0].image
    rep.ok = True

    if rep.n_newly_culled:
        rep.verdict = "worsens"
        lost = ", ".join(
            f"{v.image}:{'/'.join(v.newly_culled)}" for v in rep.images if v.newly_culled
        )
        rep.notes.append(
            f"the bodies cost the solve {rep.n_newly_culled} tie-point mark(s), dropped as "
            f"outliers with them in and kept without: {lost}. That is the clearest form of "
            f"damage there is -- a culled mark constrains nothing, and one on a nominal'd "
            f"point costs the part space a link. It also flatters every average it leaves "
            f"behind, since the marks that fall out are the worst-fitting ones: the RMS "
            f"figures below are over {rep.n_marks_after} marks against "
            f"{rep.n_marks_before} without the bodies, so read the counts before the "
            f"pixels."
        )
    elif rep.worsened:
        rep.verdict = "worsens"
        rep.notes.append(
            f"the bodies make {len(rep.worsened)} of {len(rep.images)} image(s) fit their own "
            f"tie points worse: {', '.join(rep.worsened)}. Worst is {rep.worst_image}, "
            f"{rep.images[0].before_px:.2f} to {rep.images[0].after_px:.2f} px, with "
            f"{rep.images[0].worst_point} alone moving "
            f"{rep.images[0].worst_point_delta_px:+.2f} px. Look at that point's "
            f"neighbourhood in that photo: an anchor clicked on the wrong one of two similar "
            f"corners is the usual cause, and a model at the wrong scale is the next."
        )
    elif rep.n_marks_after < rep.n_marks_before:
        # A belt to the per-image braces above: if the totals disagree and no
        # image could name which marks it lost, something has gone out of the
        # solve that this code cannot account for, and an improvement must not be
        # announced over it.
        rep.verdict = "worsens"
        rep.notes.append(
            f"the solve kept {rep.n_marks_after} tie-point marks with the bodies against "
            f"{rep.n_marks_before} without them, and the per-image tables cannot say which "
            f"went. Treat the comparison as void rather than favourable: an average over "
            f"fewer marks is not a better average."
        )
    elif rep.after_rms_px < rep.before_rms_px - IMPROVEMENT_PX:
        rep.verdict = "improves"
        rep.notes.append(
            f"the bodies improve the fit, {rep.before_rms_px:.2f} to {rep.after_rms_px:.2f} px "
            f"overall, and no image is worse than its allowance. This is what a correctly "
            f"marked known object looks like: it is a stronger claim than a tie point and it "
            f"is paying for itself."
        )
    else:
        rep.verdict = "harmless"
        rep.notes.append(
            f"the bodies change the tie-point fit very little, {rep.before_rms_px:.2f} to "
            f"{rep.after_rms_px:.2f} px, and no image is worse than its allowance. They are "
            f"not doing damage and they are not doing much work either -- worth knowing "
            f"before marking more anchors on them."
        )

    if rep.focal_before_px and rep.focal_after_px:
        drift = abs(rep.focal_after_px - rep.focal_before_px) / rep.focal_before_px
        if drift > 0.01:
            rep.notes.append(
                f"the shared focal moved {100 * drift:.1f}%, {rep.focal_before_px:.0f} to "
                f"{rep.focal_after_px:.0f} px. A body pulling the focal is the failure the "
                f"focal wall was added for; the wall stops a collapse but does not object to "
                f"a few percent, and a few percent is still the cameras being bent to suit "
                f"the model."
            )

    if rep.verdict == "worsens":
        rep.notes.append(
            "Two things to do about it, and they are not the same thing. Re-clicking the "
            "anchors above fixes the cause. Taking the anchors out of the solve stops the "
            "damage now and keeps every click: the bodies stay in the scene, stay drawn and "
            "are still placed -- against the cameras the tie points give, rather than by "
            "moving them -- and it holds across the next Solve, which putting the cameras "
            "back does not. What it does not buy is accuracy: measured on the rear-module "
            "cell, muting took the fit from 9.37 px to 2.99 px and left the bodies' shape "
            "error where it was, 3.49 mm to 3.74 mm. It buys back the camera solution."
        )
    if rep.n_muted_anchor_marks:
        rep.notes.append(
            f"{rep.n_muted_anchor_marks} of {rep.n_anchor_marks} anchor mark(s) were already "
            f"out of the solve and are in neither column: this verdict is about the "
            f"{rep.n_voting_anchor_marks} that are voting."
        )
    return rep


def mute_bodies(
    project: Project, *, instances: list[str] | None = None
) -> tuple[str, Any | None]:
    """Take the bodies' anchors out of the solve for good, and re-solve.

    The decision the verdict leads to, and the thing
    :func:`restore_without_bodies` could not do. That one puts a camera solution
    back and leaves the marks live, so Solve undoes it; this changes what the
    marks *are* -- clicks on the record rather than evidence about the cameras --
    and the change is in the project file.

    ``instances`` mutes those placements only, which is the right granularity
    when one body is at fault and the others are earning their keep. Omitted, it
    sets the project-wide switch, which is the right one when the answer to "was
    the fit better before the models went in" is yes and sorting out which of
    twenty-eight is lying is tomorrow's job.

    Nothing is deleted. Every anchor stays marked, every body stays in the
    scene, every body is still placed -- against the cameras this re-solve
    produces rather than by moving them. Unmuting is the same switch back.

    Returns ``(what it did, the new solve report or None)``. The report comes
    back rather than being thrown away because the project on disk now carries
    the solution it describes, and a caller that showed the old report beside
    the new project would be showing the damage this call just removed.
    """
    from .solve import solve

    if instances:
        wanted = set(instances)
        hit = [inst for inst in project.placed_object_instances() if inst.id in wanted]
        if not hit:
            return "none of those bodies are in this project, so nothing changed", None
        already = [inst for inst in hit if inst.anchors_off]
        for inst in hit:
            inst.anchors_off = True
        which = ", ".join(inst.label or inst.id for inst in hit)
        what = (
            f"took {len(hit)} body's anchors out of the solve ({which})"
            if len(already) < len(hit)
            else f"{which} already had their anchors out of the solve"
        )
    else:
        if not project.settings.anchors_in_solve:
            return (
                "the bodies' anchors are already out of the solve; nothing changed",
                None,
            )
        project.settings.anchors_in_solve = False
        what = (
            f"took every body's anchors out of the solve "
            f"({len(project.placed_object_instances())} bodies, "
            f"{project.n_muted_anchor_marks()} marks)"
        )

    fresh = solve(project)
    if not getattr(fresh, "ok", False):
        return (
            f"{what}, but the solve without them did not succeed, so the project is left with "
            f"the switch set and no new solution. Fix the tie-point solve first -- muting the "
            f"bodies cannot rescue a project that will not solve without them.",
            fresh,
        )
    placed = sum(1 for rep in fresh.instances.values() if rep.placed)
    return (
        f"{what} and re-solved: {fresh.rms_reprojection_px:.2f} px, {placed} body/bodies still "
        f"placed, every mark still where you clicked it. The cameras are now what the tie "
        f"points, control lines and nominals make them, and this survives the next Solve -- "
        f"which is what putting the cameras back did not.",
        fresh,
    )


def restore_without_bodies(project: Project, report: CadReport) -> str:
    """Put the body-free camera solution back, when the bodies made things worse.

    A snapshot, and that is now its whole job: it recovers the camera poses and
    point positions from a solve the bodies were not in, and leaves the marks
    voting. So the very next press of Solve re-makes the damage. That is not a
    flaw to be fixed here -- it is what "restore" means -- but it is why the
    decision belongs to :func:`mute_bodies` and this is for looking at the old
    answer, not for keeping it.

    Use it when you want to see the good solution again before deciding; use
    :func:`mute_bodies` when you have decided.

    Returns what it did, for the caller to show. Does nothing unless the verdict
    is ``"worsens"``, so calling it unconditionally is safe.
    """
    from .solve import solve

    if report.verdict != "worsens":
        return "nothing to undo: the bodies are not making any image worse"

    trial = body_free_copy(project)
    fresh = solve(trial)
    if not getattr(fresh, "ok", False):
        return (
            "the body-free solve did not succeed on its own, so there is no better state to "
            "go back to; the project is left as it is"
        )
    by_id = {im.id: im for im in trial.images}
    moved = 0
    for im in project.images:
        src = by_id.get(im.id)
        if src is None or src.R is None or src.t is None:
            continue
        im.R, im.t = src.R, src.t
        moved += 1
    positions = {p.id: p.xyz for p in trial.points if p.xyz is not None}
    for pt in project.points:
        if pt.id in positions:
            pt.xyz = positions[pt.id]
    return (
        f"restored the body-free solution: {moved} camera pose(s) and "
        f"{len(positions)} point(s) put back, anchors left marked and still voting. The next "
        f"Solve will put the bodies back in charge of the cameras -- to keep this state, take "
        f"their anchors out of the solve instead."
    )
