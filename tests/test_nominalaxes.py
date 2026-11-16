"""One tolerance per axis, because a single-axis feature has one.

A point nominal used to carry a single tolerance and expand into three axis
locks all holding it. That is right for a coordinate somebody typed off a
drawing and wrong for a metrology feature: a Perceptron ``Range`` or
``OneDimensionMinRange`` row measures exactly one axis and says so in its own
``Enabled``/``Disabled`` columns, so applying its half-millimetre tolerance to
the other two does not state the drawing -- it asserts two inspections that
nobody performed. On the PRM1 cell that was four axes across two points.

The dangerous fix is the obvious one, and it is why most of this file exists.
Releasing the uninspected axes -- a ``None`` coordinate, no lock at all -- is
measurably worse than holding them too tightly: on PRM1 it took the origin's
95% radius from 27.61 mm to 60.67 mm and swung the weak direction 54 degrees
while the reprojection RMS held at 2.0 px. The photographs there have no opinion
about the axis in question, because every camera that marks the feature is
looking down it. So the lock is weak evidence, and the only evidence that axis
has. What is wanted is a *looser* sigma, not no sigma, and these tests are
mostly here to stop a future reading of "fix the single-axis tolerance" from
deleting the constraint instead of widening it.

What is being defended here:

* **A per-axis sigma reaches its own lock and no other.** The whole feature is
  one number per axis arriving at one residual per axis; if the plumbing leaks,
  the uninspected axes go back to being pinned and nothing says so.

* **The scalar still works, untouched.** Most nominals are typed by hand and
  want one number for all three axes. The per-axis form is an override that
  must be invisible when it is absent -- including through a save and load,
  where a ``None`` entry has to come back as ``None`` and not as zero, which is
  the tightest tolerance this field can express.

* **A partial nominal does not crash the part-space test.** ``in_part_space``
  built a float array out of the nominals, and a ``None`` coordinate makes that
  an object array whose rank test raises ``LinAlgError: SVD did not converge``.
  A method whose entire contract is True or False must not raise, and this was
  the one place in the codebase where a partial nominal did.

* **A partial nominal does not count toward the frame, and the report says so.**
  A similarity fit needs three full coordinates per point, so partial nominals
  cannot build part space however many there are. Of PRM1's four links only two
  are fully specified, so a project is one careless edit from being out of part
  space with four points still labelled "nominal" in the panel. The failure that
  deserves a report line is the silent one.

* **Widening beats releasing, measured rather than asserted.** The same scene
  under the same noise, with a single-axis feature's uninspected axis held
  tight, widened and released: the widened origin stays nearer the tight one
  than the released origin does, while the reprojection RMS barely moves across
  all three -- which is the PRM1 observation in miniature, and the reason the
  factor is a factor and not infinity.

* **An uninspected axis is not held as though it had been measured**, and the
  asked-versus-achieved table says so. This one came back the opposite way round
  from how it was expected and is the more useful finding for it: a tight
  tolerance on an axis nobody inspected does not report a miss, it *wins*. Two
  such figures that contradict each other are satisfied to a third of a
  millimetre -- a clean datum table -- by bending the scene until the
  reprojection RMS roughly doubles. Widened, the same two figures come back as
  the millimetres of residual they are, in the table built to show that, and the
  photographs are left alone.

* **The import reads the file's flags and widens, rather than dropping or
  inventing.** Including the awkward case the cell actually contains: a
  single-axis feature with no stated spec limits anywhere in its row.

* **Nothing downstream quietly keeps reading the scalar.** ``nominalcheck``
  divides a deviation magnitude by one tolerance, which three different
  tolerances cannot supply, so it now takes the one in force along the direction
  the point actually missed in -- and the origin measurement's fingerprint
  covers the new field, or widening an axis would leave a stale radius looking
  current.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_nominalaxes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multispy import calibfile, nominalcheck, sensors, uncertainty
from multispy.geometry import similarity_fit
from multispy.project import Project, TiePoint
from multispy.sensors import UNINSPECTED_AXIS_FACTOR
from multispy.solve import solve

from test_synthetic import build_project, nominal_a_few

EXPORT = Path(__file__).resolve().parents[1] / "projects" / "prm1" / "Rear Module1.txt"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def one_point_project(
    nominal: list[float | None],
    sigma: float | None = None,
    sigma_axes: list[float | None] | None = None,
) -> Project:
    """The smallest project that can carry a nominal, for the plumbing tests.

    No images and no solve: ``nominal_constraints`` is pure bookkeeping over the
    point list, and testing it through a full reconstruction would hide a wrong
    sigma inside a converged residual.
    """
    project = Project(name="one point")
    project.nominal_units, project.nominal_scale = "mm", 0.001
    project.points.append(
        TiePoint(
            id="p0",
            nominal=nominal,
            nominal_sigma=sigma,
            nominal_sigma_axes=sigma_axes,
        )
    )
    return project


def locks_by_axis(project: Project) -> dict[int, float]:
    """``{axis: sigma}`` in nominal units, for the single point's locks."""
    scale = project.nominal_scale
    return {c.axis: round(c.sigma / scale, 9) for c in project.nominal_constraints()}


def claim(feature_name: str) -> sensors.FeatureClaim:
    """One named feature as the real PRM1 export states it."""
    imported = calibfile.read_export(EXPORT)
    for sensor in imported.sensors:
        for feature in sensor.features:
            if feature.name == feature_name:
                return feature
    raise AssertionError(f"{feature_name} is not in {EXPORT.name}")


def candidates(default_sigma: float | None = 0.1) -> dict[str, sensors.NominalCandidate]:
    imported = calibfile.read_export(EXPORT)
    filed = sensors.diagnose(imported.sensors, tol=1.0, warn=5.0, units="mm")
    return {
        c.feature: c
        for c in sensors.nominal_candidates(
            imported.sensors, filed, default_sigma=default_sigma
        )
    }


# --------------------------------------------------------------------------
# the plumbing: one number per axis, and the fallbacks behind it
# --------------------------------------------------------------------------


def test_a_per_axis_sigma_reaches_its_own_lock_and_only_that_one() -> None:
    project = one_point_project(
        [100.0, 200.0, 300.0], sigma=0.5, sigma_axes=[5.0, None, None]
    )
    got = locks_by_axis(project)
    assert got == {0: 5.0, 1: 0.5, 2: 0.5}, got


def test_a_scalar_sigma_still_governs_every_axis() -> None:
    """The common case, and the one that must not have changed at all."""
    project = one_point_project([100.0, 200.0, 300.0], sigma=0.5)
    got = locks_by_axis(project)
    assert got == {0: 0.5, 1: 0.5, 2: 0.5}, got


def test_with_no_tolerance_at_all_every_axis_takes_the_project_default() -> None:
    project = one_point_project([100.0, 200.0, 300.0])
    project.settings.sigma_nominal = 0.25
    got = locks_by_axis(project)
    assert got == {0: 0.25, 1: 0.25, 2: 0.25}, got


def test_a_per_axis_entry_beats_the_scalar_and_a_blank_one_falls_through() -> None:
    """The chain is per-axis, then scalar, then project default, in that order.

    Written as one test because the ordering *is* the contract: a per-axis entry
    that lost to the scalar would silently re-pin the axes this exists to free,
    and a blank entry that did not fall through would hand the solve a tolerance
    of zero.
    """
    project = one_point_project(
        [100.0, 200.0, 300.0], sigma=None, sigma_axes=[1.0, None, 3.0]
    )
    project.settings.sigma_nominal = 0.25
    got = locks_by_axis(project)
    assert got == {0: 1.0, 1: 0.25, 2: 3.0}, got


def test_a_per_axis_tolerance_survives_a_save_and_a_load() -> None:
    """Including the ``None`` entries, which must not come back as zero.

    Zero is not a missing value here, it is an infinitely tight tolerance -- the
    single most destructive number this field can hold, and the one a careless
    ``float(v or 0.0)`` would produce.
    """
    project = one_point_project(
        [100.0, None, 300.0], sigma=2.0, sigma_axes=[20.0, None, 20.0]
    )
    back = Project.from_dict(json.loads(json.dumps(project.to_dict())))
    pt = back.point("p0")
    assert pt.nominal == [100.0, None, 300.0], pt.nominal
    assert pt.nominal_sigma == 2.0, pt.nominal_sigma
    assert pt.nominal_sigma_axes == [20.0, None, 20.0], pt.nominal_sigma_axes

    got = locks_by_axis(back)
    assert got == {0: 20.0, 2: 20.0}, got


def test_a_project_without_per_axis_tolerances_round_trips_unchanged() -> None:
    """A hand-typed project must not grow a per-axis override by being loaded."""
    project = one_point_project([100.0, 200.0, 300.0], sigma=0.5)
    back = Project.from_dict(json.loads(json.dumps(project.to_dict())))
    assert back.point("p0").nominal_sigma_axes is None, back.point("p0").nominal_sigma_axes


# --------------------------------------------------------------------------
# a partial nominal: no longer a crash, and no longer a frame
# --------------------------------------------------------------------------


def test_a_partial_nominal_does_not_crash_the_part_space_test() -> None:
    """``np.array([None, 527.1, None], dtype=float)`` is the bug, in one line.

    It builds an object array, and ``matrix_rank`` on one raises
    ``LinAlgError: SVD did not converge`` -- out of a method that is only ever
    asked a yes-or-no question.
    """
    project = Project(name="partial")
    project.nominal_units, project.nominal_scale = "mm", 0.001
    project.points.append(TiePoint(id="p0", nominal=[None, 527.1, None]))
    project.points.append(TiePoint(id="p1", nominal=[0.0, 0.0, 0.0]))
    project.points.append(TiePoint(id="p2", nominal=[100.0, 0.0, 0.0]))
    project.points.append(TiePoint(id="p3", nominal=[0.0, 100.0, 0.0]))

    assert project.in_part_space() is True

    # And the lock it does generate is still there, on the one axis it states.
    p0 = [c for c in project.nominal_constraints() if c.point == "p0"]
    assert len(p0) == 1 and p0[0].axis == 1, p0


def test_a_partial_nominal_does_not_count_toward_the_frame() -> None:
    """Three points, one of them partial, is two points as far as a frame goes.

    A similarity fit needs three full coordinates per point. Counting a partial
    nominal here would promise a part space that ``align_to_nominals`` then
    declines to build, and the project would report coordinates that are not the
    part's while every panel said otherwise.
    """
    project = Project(name="two and a half")
    project.nominal_units, project.nominal_scale = "mm", 0.001
    project.points.append(TiePoint(id="p0", nominal=[0.0, 0.0, 0.0]))
    project.points.append(TiePoint(id="p1", nominal=[100.0, 0.0, 0.0]))
    project.points.append(TiePoint(id="p2", nominal=[None, 100.0, None]))
    assert project.in_part_space() is False

    project.point("p2").nominal = [50.0, 100.0, 0.0]
    assert project.in_part_space() is True


def test_three_nominals_in_a_line_still_do_not_make_a_frame() -> None:
    """The collinearity guard, re-checked because the loop around it was rewritten."""
    project = Project(name="collinear")
    project.nominal_units, project.nominal_scale = "mm", 0.001
    for i, x in enumerate((0.0, 50.0, 100.0)):
        project.points.append(TiePoint(id=f"p{i}", nominal=[x, 0.0, 0.0]))
    assert project.in_part_space() is False


def test_a_partial_nominal_is_told_apart_from_a_missing_one() -> None:
    """Three states, not two: all axes, some axes, no nominal at all."""
    assert TiePoint(id="a", nominal=[1.0, None, 3.0]).nominal_is_partial() is True
    assert TiePoint(id="b", nominal=[1.0, 2.0, 3.0]).nominal_is_partial() is False
    assert TiePoint(id="c", nominal=None).nominal_is_partial() is False
    assert TiePoint(id="d", nominal=[None, None, None]).nominal_is_partial() is False


def test_a_project_with_partial_nominals_says_so_in_its_report() -> None:
    """Two full nominals and two partial ones is not part space, and has to read
    as a sentence rather than as an absent flag.

    This is PRM1 one edit away: of its four links only p15 and p16 are fully
    specified, so making either single-axis feature partial would take the
    project out of part space with four points still carrying nominals.
    """
    project, truth = build_project()
    held = nominal_a_few(project, truth, which=(0, 8, 16, 24), tol_mm=30.0)
    for pid in held[2:]:
        pt = project.point(pid)
        pt.nominal = [None, pt.nominal[1], None]

    assert project.in_part_space() is False
    report = solve(project)
    said = [m for m in report.messages if "only some axes" in m]
    assert said, report.messages
    assert "not in part space" in said[0], said[0]
    assert report.part_space is False, report.part_space

    # And with the frame intact, the same fact is reported without the alarm.
    ok_project, ok_truth = build_project()
    ok_held = nominal_a_few(ok_project, ok_truth, which=(0, 8, 16, 24), tol_mm=30.0)
    pt = ok_project.point(ok_held[3])
    pt.nominal = [None, pt.nominal[1], None]
    assert ok_project.in_part_space() is True
    ok_report = solve(ok_project)
    said_ok = [m for m in ok_report.messages if "only some axes" in m]
    assert said_ok, ok_report.messages
    assert "not in part space" not in said_ok[0], said_ok[0]


def test_a_project_with_no_partial_nominals_says_nothing_about_them() -> None:
    """The report line has to be absent when there is nothing to report, or it
    becomes noise that nobody reads on the project where it matters."""
    project, truth = build_project()
    nominal_a_few(project, truth, which=(0, 8, 16, 24), tol_mm=30.0)
    report = solve(project)
    assert not [m for m in report.messages if "only some axes" in m], report.messages


# --------------------------------------------------------------------------
# the import: read the flags, widen, invent nothing
# --------------------------------------------------------------------------


def test_the_file_states_which_axes_it_inspects_and_the_parser_reads_them() -> None:
    assert claim("RH_CP_58").inspected == (False, True, False)
    assert claim("FM_RH_Inner").inspected == (False, True, False)
    assert claim("LH_SPARE_TIRE_TAB").inspected == (False, False, True)
    assert claim("LH_CP_103").inspected == (True, True, True)


def test_a_file_without_the_flags_says_nothing_rather_than_disabled() -> None:
    """``None``, not ``False``. A missing column is not a declaration.

    Defaulting to ``False`` would widen every axis of every feature in such a
    file on the strength of an absent header, which is the kind of change that
    looks like no change until an origin moves.
    """
    text = "\n".join(
        [
            "\t".join(("Serial Number", "Entity Name", "cfov(x)", "nominal(x)", "x(i)")),
            "\t".join(("S1", "F1", "100", "100", "1")),
        ]
    )
    imported = calibfile.read_export(text)
    feature = imported.sensors[0].features[0]
    assert feature.inspected == (None, None, None), feature.inspected
    assert sensors._axis_limits(feature, default_sigma=0.1) is None


def test_the_import_widens_the_axes_the_cell_does_not_inspect() -> None:
    """RH_CP_58 is the feature this was all found on.

    The file states a +0.50 / -3.50 band on Y and nothing on X or Z, and marks X
    and Z ``Disabled``. So Y gets the band's half-width and the other two get
    that widened -- not the half-width, and not nothing.
    """
    got = candidates()["RH_CP_58"]
    assert got.sigma == 2.0, got.sigma
    assert got.sigma_axes == [
        2.0 * UNINSPECTED_AXIS_FACTOR,
        2.0,
        2.0 * UNINSPECTED_AXIS_FACTOR,
    ], got.sigma_axes
    assert got.sigma_axes[1] < got.sigma_axes[0], got.sigma_axes


def test_a_feature_with_no_stated_limits_still_has_its_uninspected_axes_widened() -> None:
    """FM_RH_Inner: single-axis, and the file states no spec limit anywhere in
    its row. It is also one of the two PRM1 points whose uninspected axes were
    being pinned, so the awkward case is the case.

    The inspected axis comes back ``None`` -- the file really does say nothing
    about it, and the project default is the honest fallback -- while the two the
    file disclaimed are widened from that same default.
    """
    got = candidates(default_sigma=0.1)["FM_RH_Inner"]
    assert got.sigma is None, got.sigma
    assert got.sigma_axes == [
        0.1 * UNINSPECTED_AXIS_FACTOR,
        None,
        0.1 * UNINSPECTED_AXIS_FACTOR,
    ], got.sigma_axes


def test_without_a_default_no_tolerance_is_invented_for_an_unstated_axis() -> None:
    """A fabricated tolerance is weighted exactly like a measured one, so the
    import would rather offer none."""
    got = candidates(default_sigma=None)["FM_RH_Inner"]
    assert got.sigma_axes == [None, None, None], got.sigma_axes


def test_a_fully_inspected_feature_keeps_each_axis_its_own_stated_limit() -> None:
    """Nothing is widened here, but the per-axis form is still the better read:
    RH_CP_75 is held to 5 mm in X and Z and 4 mm in Y, and the scalar had to take
    the tightest of the three for all of them."""
    got = candidates()["RH_CP_75"]
    assert got.sigma == 4.0, got.sigma
    assert got.sigma_axes == [5.0, 4.0, 5.0], got.sigma_axes


def test_the_pick_list_says_which_axes_the_sensor_actually_inspected() -> None:
    """A tolerance that differs between axes for unstated reasons invites someone
    to overwrite it by hand."""
    note = candidates()["RH_CP_58"].note
    assert "inspects this on Y" in note, note
    assert "X, Z" in note, note
    assert "Disabled" in note, note
    assert f"{UNINSPECTED_AXIS_FACTOR:g}x" in note, note

    full = candidates()["LH_CP_103"].note
    assert "inspects this on X, Y, Z" in full, full
    assert "Disabled" not in full, full


def test_the_asymmetric_band_reads_the_same_per_axis_as_it_does_as_a_scalar() -> None:
    """+0.50 / -3.50 is a four-millimetre band, not a half-millimetre tolerance.

    That reading already cost one round of 6.5-sigma nonsense on RH_CP_58; the
    scalar and the per-axis form now share one implementation so the two cannot
    drift apart.
    """
    feature = claim("RH_CP_58")
    per_axis = [sensors._axis_half_width(feature, axis) for axis in range(3)]
    assert per_axis == [None, 2.0, None], per_axis
    assert sensors._tightest_limit(feature) == min(v for v in per_axis if v is not None)


# --------------------------------------------------------------------------
# the measurement: widening beats releasing
# --------------------------------------------------------------------------


def prm1_like(reading: str) -> tuple[Project, str]:
    """A scene with one single-axis feature, read three ways.

    Four points carry ordinary full nominals at a loose 30 mm and supply the
    frame. A fifth is the metrology feature: inspected on X and Z, and stating a
    Y the photographs disagree with by 30 mm -- which is the PRM1 figure, and
    PRM1's situation, because the cameras in this rig sit on an arc looking
    roughly along +Y and have least to say about exactly that axis.

    * ``tight``    -- the defect: one 0.5 mm sigma on all three axes, so the
      uninspected Y is held as though it had been inspected.
    * ``widened``  -- the fix: 0.5 mm on X and Z, ten times that on Y.
    * ``released`` -- the tempting wrong fix: no Y coordinate at all.
    """
    project, truth = build_project(seed=5)
    nominal_a_few(project, truth, which=(0, 8, 16, 24), tol_mm=30.0)

    seen = {
        p.id: sum(1 for im in project.images if p.id in im.observations)
        for p in project.points
        if not p.nominal
    }
    pid = max(seen, key=lambda k: seen[k])
    index = [p.id for p in project.points].index(pid)
    want = [float(v * 1000.0) for v in truth["points"][index]]

    pt = project.point(pid)
    pt.nominal_sigma = 0.5
    if reading == "tight":
        pt.nominal = [want[0], want[1] + 30.0, want[2]]
    elif reading == "widened":
        pt.nominal = [want[0], want[1] + 30.0, want[2]]
        pt.nominal_sigma_axes = [None, 0.5 * UNINSPECTED_AXIS_FACTOR, None]
    elif reading == "released":
        pt.nominal = [want[0], None, want[2]]
    else:
        raise AssertionError(reading)
    return project, pid


def origin_offset_mm(reference: Project, other: Project) -> float:
    """How far ``other``'s origin sits from ``reference``'s, in millimetres.

    Both are in part space, so both put their origin at (0,0,0) by construction;
    what differs is where the scene sits around it. So the scene is what is
    aligned -- scale free, exactly as :mod:`multispy.uncertainty` does it -- and
    the image of (0,0,0) under that fit is the answer.
    """
    ref = {p.id: np.asarray(p.xyz, float) for p in reference.points if p.xyz is not None}
    got = {p.id: np.asarray(p.xyz, float) for p in other.points if p.xyz is not None}
    common = sorted(set(ref) & set(got))
    assert len(common) >= 4, common
    fit = similarity_fit(
        np.array([got[i] for i in common]),
        np.array([ref[i] for i in common]),
        allow_scale=True,
    )
    return float(np.linalg.norm(fit.t)) * 1000.0


def measure_the_three_readings() -> dict[str, dict]:
    out: dict[str, dict] = {}
    solved: dict[str, Project] = {}
    for reading in ("tight", "widened", "released"):
        project, pid = prm1_like(reading)
        report = solve(project)
        assert report.ok, (reading, report.stage, report.messages)
        assert report.part_space, (reading, report.messages)
        solved[reading] = project
        out[reading] = {"rms_px": float(report.rms_reprojection_px), "point": pid}
    for reading in ("tight", "widened", "released"):
        out[reading]["origin_mm"] = origin_offset_mm(solved["tight"], solved[reading])
    return out


def test_widening_an_uninspected_axis_holds_the_origin_better_than_releasing_it() -> None:
    """The measurement that says the fix is a looser sigma and not no sigma.

    Ordered rather than thresholded, because the ordering is the claim: a widened
    lock keeps a graded fraction of its pull, so the widened origin sits between
    the tight one and the released one. A released axis keeps none, and an origin
    that has stopped answering to the drawing on that axis is free to wander as
    far as the photographs' indifference allows -- which on PRM1 was 27.61 mm to
    60.67 mm.

    "Closer to the tight origin" is not "closer to the truth", and must not be
    read as one: the tight reading is the defect, and on this scene its nominal is
    deliberately 30 mm wrong. What is shown is that the three readings are ordered
    by how much evidence they keep, with releasing at the end that keeps none.
    """
    got = measure_the_three_readings()
    widened, released = got["widened"]["origin_mm"], got["released"]["origin_mm"]

    assert widened < released, got
    assert widened > 0.5, got             # widening is not a no-op equal to tight
    assert released > 2.0 * widened, got  # and releasing is not a small change

    # The photographs are nearly indifferent to all of this, which is the point:
    # none of it is visible in the one number everybody reads.
    rms = [got[k]["rms_px"] for k in ("tight", "widened", "released")]
    assert max(rms) - min(rms) < 0.25, rms


def contradictory_pair(reading: str, lie_mm: float = 25.0) -> tuple[Project, list[str], int]:
    """Two single-axis features whose uninspected axis cannot be met at once.

    Their X nominals are each put ``lie_mm`` the wrong way, so the drawing
    demands an X separation the rigid scene cannot supply. One figure the solve
    can always satisfy by moving the whole frame; two that disagree with each
    other it cannot, and that is what makes the asked-versus-achieved table say
    anything at all -- the same reason
    :func:`test_synthetic.test_a_nominal_reports_asked_versus_achieved` uses six
    nominals rather than three.

    ``reading`` is ``"tight"`` (the defect: the uninspected axis held at the
    inspected axis's tolerance) or ``"widened"`` (the fix).
    """
    axis = 0
    project, truth = build_project(seed=5)
    nominal_a_few(project, truth, which=(0, 8, 16, 24), tol_mm=0.5)
    seen = {
        p.id: sum(1 for im in project.images if p.id in im.observations)
        for p in project.points
        if not p.nominal
    }
    pair = sorted(seen, key=lambda k: -seen[k])[:2]
    ids = [p.id for p in project.points]
    for pid, sign in zip(pair, (+1.0, -1.0)):
        want = [float(v * 1000.0) for v in truth["points"][ids.index(pid)]]
        if reading != "honest":
            want[axis] += sign * lie_mm
        pt = project.point(pid)
        pt.nominal = want
        pt.nominal_sigma = 0.5
        if reading == "widened":
            per: list[float | None] = [None, None, None]
            per[axis] = 0.5 * UNINSPECTED_AXIS_FACTOR
            pt.nominal_sigma_axes = per
    return project, pair, axis


def test_an_uninspected_axis_is_not_held_as_if_it_had_been_measured() -> None:
    """A tight tolerance on an axis nobody inspected does not report a miss. It
    wins, and charges the disagreement to the photographs.

    That is worse than it sounds, and it is the measured behaviour rather than
    the one expected when this test was written. With both features' uninspected
    X held at 0.5 mm, the solve bends the scene until the drawing is satisfied to
    a third of a millimetre -- a clean asked-versus-achieved table -- while the
    reprojection RMS roughly doubles. The reader is told the part met its
    nominals and is left to guess why the photographs got worse.

    Widened, the same two drawing figures come back as the millimetres of
    residual they are, in the one table built to show exactly that, and the
    photographs are left alone. The disagreement has not been hidden by a looser
    tolerance; it has stopped being hidden by a tighter one.
    """
    def locks(reading: str):
        project, pair, axis = contradictory_pair(reading)
        report = solve(project)
        assert report.ok, (reading, report.messages)
        found = [
            c
            for c in report.constraints
            if c.source == "nominal"
            and c.type == "axis_lock"
            and c.points[0] in pair
            and f".{'XYZ'[axis]}" in c.description
        ]
        assert len(found) == 2, (reading, [c.description for c in found])
        return report, found

    honest, _ = locks("honest")
    tight_report, tight = locks("tight")
    wide_report, wide = locks("widened")

    # The plumbing: the table grades each lock against the sigma actually in
    # force on that axis, which is the per-axis one.
    for a, b in zip(tight, wide):
        assert abs(b.sigma / a.sigma - UNINSPECTED_AXIS_FACTOR) < 1e-6, (a.sigma, b.sigma)

    # Tight: the drawing is reported as met, to a fraction of a millimetre.
    tight_miss = max(abs(c.error) for c in tight) * 1000.0
    assert tight_miss < 1.0, tight_miss
    assert all(c.sigmas_off < 3.0 for c in tight), [c.sigmas_off for c in tight]
    assert all(c.note == "" for c in tight), [c.note for c in tight]

    # Widened: the same disagreement, now visible as millimetres in the table.
    wide_miss = min(abs(c.error) for c in wide) * 1000.0
    assert wide_miss > 10.0, wide_miss
    assert max(c.sigmas_off for c in wide) > max(c.sigmas_off for c in tight), (
        [c.sigmas_off for c in tight],
        [c.sigmas_off for c in wide],
    )

    # And the cost of the tight reading, which is where it went instead: the
    # photographs. Widened, they are left close to where an honest drawing
    # leaves them.
    assert tight_report.rms_reprojection_px > 1.5 * honest.rms_reprojection_px, (
        honest.rms_reprojection_px,
        tight_report.rms_reprojection_px,
    )
    assert wide_report.rms_reprojection_px < 1.25 * honest.rms_reprojection_px, (
        honest.rms_reprojection_px,
        wide_report.rms_reprojection_px,
    )


def test_the_shape_check_grades_a_miss_against_the_axis_it_missed_along() -> None:
    """``nominalcheck`` divides a deviation *magnitude* by one tolerance, and
    with three different tolerances there is no such number to hand.

    Both obvious answers are wrong in opposite directions. The tightest axis
    convicts a point of a deviation lying along a loose one -- which for a
    single-axis feature means convicting it of disagreeing with an inspection
    nobody performed, the exact defect this change exists to remove. The loosest
    excuses the reverse. So the figure is the tolerance ellipsoid's own radius in
    the direction of the residual, and the sigma count stays the Mahalanobis
    distance it always was.
    """
    tight_on_y = [20.0, 2.0, 20.0]       # RH_CP_58, as the file states it

    along_y = nominalcheck._sigma_along(tight_on_y, np.array([0.0, 1.0, 0.0]))
    along_x = nominalcheck._sigma_along(tight_on_y, np.array([1.0, 0.0, 0.0]))
    assert abs(along_y - 2.0) < 1e-9, along_y
    assert abs(along_x - 20.0) < 1e-9, along_x

    # Off-axis sits between the two, and nearer the tight one -- the ellipsoid
    # is narrow, so a direction has to be close to the loose axis to be let off.
    diagonal = nominalcheck._sigma_along(tight_on_y, np.array([1.0, 1.0, 0.0]))
    assert 2.0 < diagonal < 20.0, diagonal
    assert diagonal < 4.0, diagonal

    # And it is the identity on the only case that existed before: three equal
    # tolerances, whatever direction the point missed in.
    for direction in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -2.0, 0.5]):
        got = nominalcheck._sigma_along([3.0, 3.0, 3.0], np.array(direction))
        assert abs(got - 3.0) < 1e-9, (direction, got)


def test_the_shape_check_calls_a_point_defaulted_only_when_nothing_stated_one() -> None:
    """A per-axis tolerance is a stated tolerance.

    ``sigma_defaulted`` marks the points that act as rigid anchors because
    nobody gave them a number. FM_RH_Inner states no scalar at all and would
    have been marked that way, while in fact the file's own ``Disabled`` flags
    put its uninspected axes at ten times the default.
    """
    stated = TiePoint(id="a", nominal=[0.0, 0.0, 0.0], nominal_sigma_axes=[1.0, None, 1.0])
    bare = TiePoint(id="b", nominal=[0.0, 0.0, 0.0])
    assert stated.nominal_sigma is None and bare.nominal_sigma is None
    assert stated.nominal_sigma_on(0, 0.1) == 1.0
    assert stated.nominal_sigma_on(1, 0.1) == 0.1
    assert bare.nominal_sigma_on(0, 0.1) == 0.1

    project = Project(name="defaulted")
    project.nominal_units, project.nominal_scale = "mm", 0.001
    project.points.extend([stated, bare])
    rows = {
        p.id: (
            p.nominal_sigma is None
            and not any(v is not None for v in (p.nominal_sigma_axes or []))
        )
        for p in project.points
    }
    assert rows == {"a": False, "b": True}, rows


def test_the_origin_measurement_goes_stale_when_an_axis_is_widened() -> None:
    """A stored radius that survived the edit would be a reading of a project
    that no longer exists -- and widening an axis is precisely an edit that moves
    the origin, which is the whole finding this change rests on."""
    project = one_point_project([100.0, 200.0, 300.0], sigma=0.5)
    before = uncertainty.fingerprint(project)
    project.point("p0").nominal_sigma_axes = [5.0, None, 5.0]
    assert uncertainty.fingerprint(project) != before


def test_the_widening_factor_is_a_stated_judgement() -> None:
    """Not a measurement, and the docstring has to keep saying so -- it is the one
    number in this change that the file does not supply, and a successor reading
    only the code would have no way to know that."""
    assert UNINSPECTED_AXIS_FACTOR > 1.0
    source = Path(sensors.__file__).read_text(encoding="utf-8")
    note = source.split("UNINSPECTED_AXIS_FACTOR = ", 1)[1].split('"""')[1]
    assert "judgement" in note.lower(), note
    assert "not a measurement" in note.lower(), note


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

    print("\nPRM1's two single-axis features, as the file itself states them")
    print(f"{'feature':<16}{'inspects':<10}{'scalar':>9}{'X':>11}{'Y':>11}{'Z':>11}")
    got = candidates()
    for name in ("RH_CP_58", "FM_RH_Inner", "RH_CP_75"):
        c = got[name]
        on = ", ".join(a for a, f in zip("XYZ", c.inspected or []) if f) or "-"
        cells = "".join(
            f"{'(default)' if v is None else format(v, '.2f'):>11}"
            for v in (c.sigma_axes or [None, None, None])
        )
        scalar = "-" if c.sigma is None else format(c.sigma, ".2f")
        print(f"{name:<16}{on:<10}{scalar:>9}{cells}")

    print("\none single-axis feature, its uninspected axis read three ways")
    m = measure_the_three_readings()
    print(f"{'reading':<10}{'origin moved':>15}{'reproj RMS':>14}")
    for k in ("tight", "widened", "released"):
        print(f"{k:<10}{m[k]['origin_mm']:12.2f} mm{m[k]['rms_px']:11.2f} px")
    print("\nReleasing the axis moves the origin furthest and costs almost nothing")
    print("in pixels, which is why the fix is a looser sigma and not no sigma.")

    print("\ntwo single-axis features whose uninspected axis contradicts the other")
    print(f"{'reading':<10}{'worst datum miss':>19}{'reproj RMS':>14}")
    for reading in ("honest", "tight", "widened"):
        project, pair, axis = contradictory_pair(reading)
        rep = solve(project)
        miss = max(
            abs(c.error)
            for c in rep.constraints
            if c.source == "nominal" and c.points[0] in pair
            and f".{'XYZ'[axis]}" in c.description
        ) * 1000.0
        print(f"{reading:<10}{miss:16.2f} mm{rep.rms_reprojection_px:11.2f} px")
    print("\nThe tight reading is the one with the clean datum table. It got there by")
    print("bending the photographs, which is the opposite of what the table is read for.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
