"""Control lines that share a direction nobody has to name.

An axis line says *this edge runs along world X*. That is a strong statement and
a cheap one: two of them in a photograph give a vanishing point, and two
vanishing points give the camera. But it can only be made about an edge whose
world direction is already known, and the kind of room this tool gets pointed at
is full of long straight features whose direction is not. Cable tray, conduit,
handrail, strut channel, a hose run, the diagonal of a safety cage. You cannot
say which way any of them goes. You can say that the edge in photo A is the
*same* edge -- or at least parallel to it -- in photos B and C.

A **direction group** is exactly that statement and nothing more: a named world
direction whose orientation is unknown and is solved for, with control lines in
any number of photographs declaring themselves parallel to it. Four lines marked
"Tray 1" across four photographs assert that one world direction explains all
four, and the four rotations then have to agree about what it is.

Deliberately a direction and not a line
---------------------------------------
A group has no position. Two photographs of the same physical tray constrain a
3D *line* -- 4 degrees of freedom, direction and offset -- and that is a bigger
and different idea, because it needs the user to promise the marks are on the
same edge rather than merely a parallel one. "Parallel to" is the weaker promise
and the one a person can actually keep at a glance across a cluttered cell, so
it is the one modelled here.

What a group costs and what it earns
------------------------------------
A direction is a unit vector: 2 unknowns, carried as an offset in the tangent
plane of the current estimate (see ``direction_at``). Each line in the group
contributes one residual. So the arithmetic is blunt:

* 2 lines -- two residuals against two unknowns. Exactly determined, and
  therefore worth **nothing**. The group fits its own lines perfectly and tells
  the cameras nothing at all.
* 3 lines -- one net constraint. This is where a group starts to earn its keep.
* n lines -- ``n - 2`` net constraints.

This is the one place the intuition behind the feature needs a caveat. "There
are lots of these and an edge beats a point" is true *per line*: a line pins a
direction over hundreds of pixels where a click pins one position. It is not
true per *group*. A scatter of two-line groups is free parameters wearing a
constraint's clothes, and ``contribution_note`` says so per group rather than
leaving someone to add twelve of them and believe they helped.

Why a group in one photograph is refused outright
-------------------------------------------------
Not because the direction is unrecoverable -- two lines in one photograph meet at
a vanishing point, which *is* the direction. Because it constrains the camera by
exactly zero, however many lines it holds, and the reason is an invariance
rather than a shortage of data.

Every residual in the group enters through ``m = R @ d``: the world direction
carried into that one camera's frame. With ``d`` free, ``m`` is a free unit
vector. Turn the camera by any small rotation and counter-turn ``d`` to leave
``m`` where it was, and not one residual in the group changes. The minimum is
the same for every rotation the camera could have, so the group cannot prefer
one -- it is only testing whether its own image lines happen to pass through a
common point, which is a fact about the lines and not about the pose.

A second photograph breaks the invariance, because one ``d`` now has to satisfy
two different ``R``s and cannot be counter-turned to suit both. So the rule is
*lines in two or more different images*, and it is a refusal rather than a
warning: a one-image group would otherwise sit in the solve adding two
parameters and a confident-looking residual to a problem it contributes nothing
to.

Why the drawn end does not matter, unlike an axis line
------------------------------------------------------
``AxisLine`` cares which way it was dragged: ``p0 -> p1`` points along the
*positive* axis, and that is how the sign ambiguity of a vanishing point gets
resolved. A group has no positive end to point at -- ``d`` and ``-d`` are the
same statement, and the residual is odd in ``d``, so squaring it makes both the
drawn direction and the group's own sign immaterial. One less thing to get
wrong, and the UI says so.

Why a group cannot bootstrap a calibration
------------------------------------------
``solve.calibrate_all`` gets each camera's rotation and focal from vanishing
points, which needs two axes with two lines each *whose world directions are
known*. A direction that is itself an output of the bundle is no use there:
there is nothing to measure the camera against. So free groups refine inside the
bundle only, and a photograph carrying nothing but free lines still will not
calibrate, no matter how many groups it takes part in. ``axis_segments``
therefore excludes grouped lines, and a group is never counted towards the
two-axis test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_PAIR_SINE = 0.05
"""How far from parallel two view planes must be before their intersection is
treated as a direction rather than as noise.

A drawn segment of length ``L`` with ``sigma`` pixels of endpoint error fixes its
image line's angle to roughly ``sigma * sqrt(2) / L`` radians -- about 0.4 of a
degree for two pixels over four hundred. Intersecting two planes that meet at an
angle ``alpha`` divides that error by ``sin(alpha)``, so at this floor (2.9
degrees) a well-drawn pair still starts the optimiser within about eight degrees
of the truth, which is the right basin. Below it the cross product is mostly the
noise, pointing somewhere arbitrary, and a bundle started there can settle into a
direction that fits its own lines and bends the cameras to do it.

A floor on the *starting point*, not on the answer: the bundle refines the
direction against every line in the group afterwards, and the figure the best
pair achieved is reported as ``DirectionGroupReport.conditioning``, so a badly
conditioned group is visible rather than merely survivable."""

MIN_IMAGES = 2
"""Distinct photographs a group needs before it constrains anything.

Not a quality threshold -- the line between a constraint and a
reparameterisation. See the invariance argument in the module docstring."""


@dataclass
class DirectionGroup:
    """A shared world direction, named by the user and measured by the solve."""

    id: str
    label: str = ""
    """What the user calls it -- "Tray 1", "handrail", "the cage diagonal". Free
    text, because the whole point is that this direction has no name in the
    part's coordinate system."""
    d: list[float] | None = None
    """Unit direction in world coordinates, as the last solve left it.

    An output, saved like ``ImageView.R`` so the panel can show it after a
    reload, and re-initialised from the lines themselves on every solve rather
    than used as a starting point -- a stored direction from a project whose
    lines have since been redrawn is a worse guess than the one those lines
    give. Sign is arbitrary: ``d`` and ``-d`` are the same group."""

    def vector(self) -> np.ndarray | None:
        """The stored direction as a unit vector, or ``None`` if unset or junk."""
        if self.d is None:
            return None
        v = np.asarray(self.d, dtype=float)
        n = float(np.linalg.norm(v))
        return None if not np.isfinite(n) or n < 1e-12 else v / n


@dataclass
class DirectionGroupReport:
    """What one group contributed, in the terms someone deciding whether to draw
    another line into it would want."""

    id: str
    label: str = ""
    n_lines: int = 0
    n_images: int = 0
    """Distinct photographs holding a line in this group. Two is the floor."""
    identifiable: bool = False
    """The group reached the solve. False means its lines were ignored entirely,
    and ``note`` says why."""
    direction: list[float] | None = None
    """Unit world direction as solved, or ``None`` if the group was refused."""
    net_constraints: int = 0
    """``n_lines - 2``, floored at zero: how much the group actually added after
    paying for its own two unknowns. Zero is the honest reading of a two-line
    group, and the reason this number is reported next to the line count rather
    than instead of it."""
    conditioning: float = 0.0
    """Sine of the angle between the two view planes that initialised the
    direction -- 1.0 is perpendicular and ideal, ``MIN_PAIR_SINE`` the floor.

    Low means every photograph in the group looks along nearly the same plane,
    so the direction is weakly observed however many lines are in it. This is
    the number that distinguishes "more lines" from "a better-placed photo"."""
    rms_px: float = 0.0
    """Endpoint slip of this group's lines against the solved direction, in
    pixels -- the same ``(L/2) sin(theta)`` measure an axis line reports, so the
    two are directly comparable in the same panel."""
    max_px: float = 0.0
    n_culled: int = 0
    """Lines in this group dropped as outliers before the final pass."""
    note: str = ""
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


def plane_normal(R: np.ndarray, K: np.ndarray, line: np.ndarray) -> np.ndarray | None:
    """World normal of the plane through the camera centre and a drawn image line.

    The direction a grouped line claims to follow must lie in this plane, and
    that is the whole geometric content of one such line. ``line`` is the
    homogeneous image line ``cross(p0, p1)``; ``K.T @ line`` carries it into the
    camera's normalised frame and ``R.T`` from there into the world.

    ``None`` when the segment is degenerate enough that its normal is
    numerically meaningless -- a zero-length drag, or a line whose homogeneous
    form has collapsed.
    """
    nrm = np.asarray(K, dtype=float).T @ np.asarray(line, dtype=float)
    n = float(np.linalg.norm(nrm))
    if not np.isfinite(n) or n < 1e-12:
        return None
    return np.asarray(R, dtype=float).T @ (nrm / n)


def initial_direction(
    normals: list[np.ndarray],
) -> tuple[np.ndarray | None, float, tuple[int, int] | None]:
    """Start the direction from the two view planes that cross most squarely.

    The direction lies in every plane its lines define, so any two of them pin it
    down: ``d = n1 x n2``. Which two matters a great deal. A nearly parallel pair
    -- two photographs taken a step apart, or two lines in one photograph running
    almost the same way on screen -- has a cross product dominated by whatever
    noise is in the endpoints, and hands the bundle a direction pointing
    somewhere arbitrary. So every pair is scored by ``|n1 x n2|``, the sine of the
    angle between the planes, and the best one wins.

    Pairs within a single photograph are allowed, and are often the best
    available: two lines in one image meet at the group's vanishing point, which
    is a perfectly good *starting* direction. That it is not enough to make the
    group worth solving is a separate question, settled by ``MIN_IMAGES``, and
    keeping the two apart is deliberate -- refusing a within-image pair here
    would leave a well-marked group starting from a worse guess than it had.

    Returns ``(direction, sine, (i, j))``. The direction is ``None`` when no pair
    reaches ``MIN_PAIR_SINE``, and the best sine found is returned regardless so
    the caller can report how close the group came.
    """
    best_sine = 0.0
    best_d: np.ndarray | None = None
    best_pair: tuple[int, int] | None = None
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            c = np.cross(normals[i], normals[j])
            s = float(np.linalg.norm(c))
            if s > best_sine:
                best_sine, best_d, best_pair = s, c / s, (i, j)
    if best_d is None or best_sine < MIN_PAIR_SINE:
        return None, best_sine, best_pair
    return best_d, best_sine, best_pair


def tangent_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to ``d``.

    This is what keeps the two unknowns well behaved. Parameterising a direction
    by two angles puts a coordinate singularity somewhere on the sphere and the
    optimiser will find it: near the pole the two parameters stop being
    independent, the Jacobian loses rank, and a direction that was merely
    uncertain becomes unsolvable. An offset in the tangent plane of the *current*
    estimate has no pole anywhere near the data by construction, because the
    basis is built around wherever the estimate already is.

    The seed axis is the world axis ``d`` is least aligned with, so the first
    cross product is never near zero and the basis itself cannot degenerate.
    """
    d = np.asarray(d, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(d)))] = 1.0
    u = np.cross(d, e)
    u = u / float(np.linalg.norm(u))
    v = np.cross(d, u)
    return u, v


def direction_at(
    d0: np.ndarray, u: np.ndarray, v: np.ndarray, a: float, b: float
) -> np.ndarray:
    """``d0`` turned by ``(a, b)`` radians within the tangent basis ``(u, v)``.

    A true rotation on the sphere rather than ``normalize(d0 + a*u + b*v)``: the
    result is a unit vector for every input, however large, so there is no
    excursion the solver can take that yields a non-unit direction or a division
    by something small. ``hypot(a, b)`` is then the angle moved, in radians,
    which also makes the two parameters readable in a diagnostic.
    """
    theta = math.hypot(float(a), float(b))
    d0 = np.asarray(d0, dtype=float)
    if theta < 1e-12:
        return d0.copy()
    axis = (float(a) * np.asarray(u, dtype=float) + float(b) * np.asarray(v, dtype=float)) / theta
    return math.cos(theta) * d0 + math.sin(theta) * axis


def contribution_note(n_lines: int, n_images: int, conditioning: float) -> str:
    """One sentence on what this group is doing for the solve, or why it is not.

    Written as a reading rather than a setting: it states what the counts imply
    and leaves the decision to draw another line to the person who can see the
    photograph.
    """
    if n_lines == 0:
        return "no lines drawn into this group yet"
    if n_lines < 2:
        return (
            "one line, which fixes a plane the direction lies in but not the direction. "
            "Draw the same edge in another photo."
        )
    if n_images < MIN_IMAGES:
        return (
            f"all {n_lines} lines are in one photo, so this group is ignored. A direction the "
            "solve works out can be turned to suit a single camera whatever that camera's "
            "rotation is, so one photo constrains nothing at all. Draw the same edge in a "
            "second photo."
        )
    surplus = n_lines - 2
    if surplus == 0:
        return (
            f"{n_lines} lines across {n_images} photos, which is break-even: the direction "
            "costs 2 unknowns and each line gives 1 residual, so this group fits itself exactly "
            "and constrains the cameras by nothing. A third line is where it starts to help."
        )
    weak = ""
    if conditioning < 0.3:
        degrees = math.degrees(math.asin(max(min(conditioning, 1.0), 0.0)))
        weak = (
            f" The views in this group cross at only {degrees:.0f} degrees, so the direction "
            "itself is weakly observed -- a photo looking across this edge would do more for it "
            "than another line from where you already stood."
        )
    return (
        f"{n_lines} lines across {n_images} photos: {surplus} net constraint(s) after paying "
        f"for the direction's own 2 unknowns.{weak}"
    )
