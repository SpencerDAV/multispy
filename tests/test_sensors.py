"""Can the calibration file be caught lying, and caught lying *by name*?

The fixture writes a calibration export in the real format -- tab separated, one
row per measured feature, direction cosines, declared standoff and field of view
-- and then injects faults into it deliberately, so the ground truth is known
exactly and a verdict can be checked against what was actually done rather than
against what the report happens to say. It runs through the real parser, so the
format and the judging are tested together and a column rename cannot pass.

Each fault from the shop floor gets its own test. So does each thing this module
*cannot* see, and that second half matters more:

* **A feature inside the field of view is not a correct transform.** The box is
  200-300 mm across and the tolerance that matters is a few millimetres. The
  tests pin that "consistent" says so in words, because reading it as "calibrated
  well" would be the most damaging misreading available.
* **A flip about the aim axis is invisible**, moving neither the sight line nor
  the field of view, so it legitimately passes -- and every verdict including the
  passing ones has to carry the fact that it was never tested.
* **A fault shared by a whole session cannot be told from a moved part.** The
  session test asserts the report names both readings rather than picking one.

Run directly for a readable report::

    .venv\\Scripts\\python.exe tests\\test_sensors.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multispy.calibfile import read_export
from multispy.sensors import (
    CONSISTENT,
    EDGE,
    EDGE_RATIO,
    EXPLAINED,
    MIN_CONSENSUS_SENSORS,
    SUSPECT,
    UNTESTABLE,
    SensorClaim,
    aim_consensus,
    diagnose,
    euler_matrix,
    fault_catalogue,
    infer_depth_axis,
    quaternion_matrix,
    quaternion_scalar_position,
    session_key,
)

TOL, WARN = 10.0, 50.0

FOV = {
    600.0: ((74.80, 102.85, -102.85), (117.30, 158.10, 160.65)),
    400.0: ((51.85, 71.40, -71.40), (80.75, 109.65, 110.50)),
}
"""Near and far half-extents, lifted from the real export.

The depth axis is first and is the *smallest* on both sensor types, which is what
``infer_depth_axis`` reads. The negative near-Z is in the real file too -- the
vendor stores a corner rather than an extent -- and keeping it here means the
importer is tested against the sign convention it will actually meet.
"""

COLUMNS = (
    "Serial Number\tEntity Name\tAlgorithm\tStandoff\tPart Number\tCA Sweep Face\t"
    "Calibration Date\tx(i)\tx(j)\tx(k)\ty(i)\ty(j)\ty(k)\tz(i)\tz(j)\tz(k)\t"
    "cfov(x)\tcfov(y)\tcfov(z)\tnominal(x)\tnominal(y)\tnominal(z)\t"
    "fov near X\tfov near Y\tfov near Z\tfov far X\tfov far Y\tfov far Z\t"
    "X Offset\tY Offset\tZ Offset\tX USL\tY USL\tZ USL\tX LSL\tY LSL\tZ LSL"
)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / float(np.linalg.norm(v))


def frame_aimed_along(aim) -> np.ndarray:
    """A direction-cosine block whose first row is ``aim``.

    First row, because local X is the range axis in this format -- the fixture
    commits to that and ``test_the_range_axis_is_read_off_the_optics`` checks the
    module works it out for itself rather than sharing the assumption.
    """
    x = _unit(aim)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(x @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    y = _unit(np.cross(up, x))
    z = np.cross(x, y)
    return np.array([x, y, z])


class Cell:
    """A cell being built up row by row, then rendered as an export."""

    def __init__(self) -> None:
        self.rows: list[str] = []

    def sensor(
        self,
        serial: str,
        cfov,
        *,
        features=None,
        standoff: float = 600.0,
        part: str = "917-4012",
        date: str = "10/13/2019 9:42:24 AM",
        aim=None,
        M=None,
        shift=None,
        offsets=None,
        sweep: str = "X-Y",
    ) -> "Cell":
        cfov = np.asarray(cfov, dtype=float)
        feats = [np.asarray(f, dtype=float) for f in (features or [cfov])]
        # A healthy sensor looks at the centroid of what it measures, from outside.
        if M is None:
            M = frame_aimed_along(aim if aim is not None else np.array([0.0, 0.0, -1.0]))
        placed = cfov + (np.zeros(3) if shift is None else np.asarray(shift, dtype=float))
        near, far = FOV[standoff]
        for feat in feats:
            off = offsets if offsets is not None else (None, None, None)
            cells = [
                serial,
                f"F{serial}_{len(self.rows)}",
                "Hole",
                f"{standoff:.0f}",
                part,
                sweep,
                date,
                *[f"{v:.5f}" for v in M.reshape(9)],
                *[f"{v:.5f}" for v in placed],
                *[f"{v:.5f}" for v in feat],
                *[f"{v:.2f}" for v in near],
                *[f"{v:.2f}" for v in far],
                *["" if v is None else f"{v:.2f}" for v in off],
                "5.00", "5.00", "5.00", "-5.00", "-5.00", "-5.00",
            ]
            self.rows.append("\t".join(cells))
        return self

    def text(self) -> str:
        return "\n".join([COLUMNS, *self.rows])

    def read(self):
        return read_export(self.text())


def healthy_cell(n: int = 6) -> Cell:
    """``n`` sensors down the length of a part, each watching its own feature.

    Spread along a car body rather than ringed about one volume, because that is
    the geometry the real file has and the geometry that broke the first design:
    the fields of view are 200 mm across and 500 mm apart, so there is no shared
    working volume to form a consensus from.
    """
    cell = Cell()
    for i in range(n):
        centre = np.array([4600.0 + 500.0 * i, -300.0 + 60.0 * (i % 3), 600.0])
        cell.sensor(
            f"S{i:05d}",
            centre,
            features=[centre],
            aim=np.array([0.1 * (i % 3 - 1), 0.2, -1.0]),
            date="10/13/2019 9:42:24 AM" if i % 2 else "2/1/2020 8:21:47 AM",
        )
    return cell


def verdict_for(report, serial: str):
    return next(v for v in report.verdicts if v.serial == serial)


# ------------------------------------------------- the format, and the grouping


def test_rows_are_features_and_sensors_are_serials():
    """The quiet trap: a sensor watching three features appears three times.

    An importer that took rows for sensors would report one transform three times,
    count one fault as three, and let a sensor with many features outvote one with
    a single feature in anything downstream.
    """
    centre = np.array([5000.0, 0.0, 600.0])
    cell = Cell().sensor(
        "100654",
        centre,
        features=[centre, centre + [0.0, 80.0, 0.0], centre + [0.0, -80.0, 0.0]],
    )
    cell.sensor("100653", np.array([5600.0, 0.0, 600.0]))
    imp = cell.read()
    assert imp.n_rows == 4, f"fixture wrote {imp.n_rows} rows"
    assert len(imp.sensors) == 2, f"grouped into {len(imp.sensors)} sensors, not 2"
    assert imp.n_features == 4
    by = {s.serial: s for s in imp.sensors}
    assert len(by["100654"].features) == 3 and len(by["100653"].features) == 1


def test_the_direction_cosines_come_back_as_a_rotation():
    imp = healthy_cell().read()
    assert imp.worst_orthonormality < 1e-4, (
        f"the parsed orientation blocks are {imp.worst_orthonormality:.2e} from orthonormal"
    )
    for s in imp.sensors:
        # 1e-4, not 1e-9: these files write direction cosines to five decimals, so
        # a parsed block is a rotation to about 1e-5 and no tighter. The real
        # export agrees -- 1.21e-05 across all 15 of its sensors. A test demanding
        # machine precision here would be testing the text format, not the pose.
        assert abs(float(np.linalg.det(s.M)) - 1.0) < 1e-4
        # R is the transpose, and that transpose is written exactly once.
        assert np.allclose(s.R, s.M.T)


def test_a_sensor_that_disagrees_with_itself_is_reported():
    """Two rows claiming the same serial with different transforms is a contradiction."""
    cell = Cell().sensor("100654", np.array([5000.0, 0.0, 600.0]))
    cell.sensor("100654", np.array([6000.0, 0.0, 600.0]))      # same serial, moved
    imp = cell.read()
    assert any("disagrees with itself" in w for w in imp.warnings), (
        f"no warning about a self-contradictory sensor: {imp.warnings}"
    )


def test_a_blank_offset_is_not_a_zero():
    """A blank means the axis was never measured; a zero means it measured clean.

    Collapsing them would invent perfect measurements on axes nobody looked at,
    and those inventions would pass every spec check that followed.
    """
    cell = Cell().sensor("1", np.array([5000.0, 0.0, 600.0]), offsets=(0.0, None, 3.0))
    s = cell.read().sensors[0]
    assert s.features[0].offsets == (0.0, None, 3.0), s.features[0].offsets


def test_a_wrong_path_raises_instead_of_parsing_as_content():
    """The first bug this module had, pinned.

    A mistyped path parsed cleanly as a one-line file -- a header with no rows --
    so the import reported "no data rows" about a file it had never opened. A
    wrong path has to look like a wrong path.
    """
    try:
        read_export("C:/nowhere/no-such-calibration-file.txt")
    except OSError:
        pass
    else:
        raise AssertionError("a non-existent path was accepted as file content")


def test_a_file_that_is_not_a_calibration_export_imports_nothing():
    imp = read_export("alpha\tbeta\tgamma\n1\t2\t3")
    assert not imp.sensors
    assert any("Serial Number" in w for w in imp.warnings), (
        f"the refusal does not name the missing column: {imp.warnings}"
    )


# ------------------------------------------------------------- the range axis


def test_the_range_axis_is_read_off_the_optics():
    """Each sensor votes from its own frustum shape -- no consensus, no reference data.

    A triangulation sensor trades depth for resolution, so its depth half-range is
    smaller than its lateral extents. The fixture never says which axis it used.
    """
    imp = healthy_cell().read()
    axis = infer_depth_axis(imp.sensors)
    assert axis.axis == 0 and axis.name == "X", f"inferred {axis.name}"
    assert axis.unanimous, f"not unanimous: {axis.votes}"


def test_a_square_frustum_abstains_rather_than_voting():
    """Nothing in a square cross-section says which axis is the range direction."""
    cell = Cell()
    for i in range(4):
        cell.sensor(f"Q{i}", np.array([5000.0 + 300.0 * i, 0.0, 600.0]))
    imp = cell.read()
    for s in imp.sensors:
        s.fov_near = np.array([100.0, 100.0, 100.0])
        s.fov_far = np.array([100.0, 100.0, 100.0])
    axis = infer_depth_axis(imp.sensors)
    assert not axis.unanimous
    assert "could not be read off the optics" in axis.reason, axis.reason


# ----------------------------------------------- the clean file, and its limits


def test_a_clean_file_has_every_sensor_consistent():
    report = diagnose(healthy_cell().read().sensors, tol=TOL, warn=WARN)
    assert report.ok, report.reason
    assert report.n_suspect == 0 and report.n_explained == 0, "; ".join(
        v.reason for v in report.verdicts if v.verdict != CONSISTENT
    )
    assert report.n_consistent == 6
    assert report.grade(TOL, WARN) == "good"


def test_a_healthy_sensor_with_several_features_is_not_penalised():
    """One field of view cannot centre on two features 160 mm apart.

    An earlier version graded on the correction vector of every sensor, including
    the passing ones, where it is just the centroid offset of that sensor's own
    features. A clean file came out amber for having more than one feature per
    sensor.
    """
    centre = np.array([5000.0, 0.0, 600.0])
    cell = healthy_cell(5)
    cell.sensor(
        "MANY",
        centre,
        features=[centre + [0.0, 80.0, 0.0], centre + [0.0, -80.0, 0.0]],
    )
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    v = verdict_for(report, "MANY")
    assert v.verdict == CONSISTENT, v.reason
    assert float(np.linalg.norm(v.correction)) < 1.0, (
        f"a sensor centred between its two features reported a {np.linalg.norm(v.correction):.0f} "
        f"correction"
    )
    assert report.grade(TOL, WARN) == "good", (
        "a clean file was marked down for having a sensor with two features"
    )


def test_consistent_says_what_it_is_narrower_than():
    """The most damaging available misreading, guarded in words."""
    report = diagnose(healthy_cell().read().sensors, tol=TOL, warn=WARN)
    v = report.verdicts[0]
    assert v.verdict == CONSISTENT
    assert "nothing finer" in v.reason and "millimetres" in v.reason, v.reason
    assert any("not grossly wrong" in m for m in report.messages), report.messages


# ----------------------------------------------- the faults, one at a time


def test_a_sensor_calibrated_against_the_wrong_nominal_is_caught():
    """Fault 1, the one in the real file: the transform is displaced, 412 mm in X."""
    cell = healthy_cell(5)
    centre = np.array([7000.0, 0.0, 600.0])
    cell.sensor("BADX", centre, features=[centre], shift=np.array([-412.0, 0.0, 0.0]),
                offsets=(412.0, 0.0, 0.0))
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    v = verdict_for(report, "BADX")
    assert v.verdict == SUSPECT, f"came back {v.verdict}: {v.reason}"
    assert v.ratio > 2.0, f"a 412 mm displacement showed as only {v.ratio:.2f} of the box"
    assert v.ratio_after <= 1.0, (
        f"a single translation left {v.ratio_after:.2f} of the box, so the report will not "
        f"call this displaced"
    )
    assert abs(float(np.linalg.norm(v.correction)) - 412.0) < 1.0
    assert "displaced rather than misoriented" in v.reason
    assert "LH" not in v.reason  # no borrowed detail from the real file
    assert v.out_of_spec, "a 412 mm reported deviation did not register as out of spec"
    assert report.grade(TOL, WARN) == "bad"
    assert sum(1 for x in report.verdicts if x.verdict != CONSISTENT) == 1, (
        "one bad sensor dragged others out with it, which cannot happen when each is "
        "measured against its own optics"
    )


def test_the_mirrored_feature_fault_is_explained_by_name():
    """Fault 2: calibrated against the mirror-image feature, so Y has the wrong sign."""
    cell = healthy_cell(5)
    centre = np.array([6000.0, 400.0, 600.0])
    cell.sensor("MIRROR", centre, features=[centre],
                shift=np.array([0.0, -800.0, 0.0]))     # places cfov at y = -400
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    v = verdict_for(report, "MIRROR")
    assert v.verdict == EXPLAINED, f"came back {v.verdict}: {v.reason}"
    assert v.fault in ("mirror_y", "reflect_y"), f"named the fault {v.fault!r}"
    assert v.fault_label and "mirror" in v.fault_label.lower()
    assert "accounts for it" in v.reason


def test_an_overrun_on_the_range_axis_is_called_a_depth_error():
    """The axis of the overrun is what makes the number actionable."""
    cell = healthy_cell(5)
    centre = np.array([6500.0, 0.0, 600.0])
    aim = np.array([0.0, 0.0, -1.0])
    cell.sensor("DEPTH", centre, features=[centre], aim=aim,
                shift=aim * -400.0)        # pushed along its own sight line
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    v = verdict_for(report, "DEPTH")
    assert v.verdict in (SUSPECT, EXPLAINED), v.reason
    assert v.overrun_axis == "X", f"blamed {v.overrun_axis}, not the range axis"
    assert "standoff or depth error" in v.reason, v.reason


def test_a_feature_just_outside_the_box_is_marginal_not_damning():
    cell = healthy_cell(5)
    centre = np.array([5200.0, 0.0, 600.0])
    # 1.1x of the lateral far extent: outside, but nowhere near a real fault.
    cell.sensor("EDGEY", centre, features=[centre + [0.0, FOV[600.0][1][1] * 1.1, 0.0]])
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    v = verdict_for(report, "EDGEY")
    assert v.verdict == EDGE, f"came back {v.verdict} at ratio {v.ratio:.2f}"
    assert 1.0 < v.ratio <= EDGE_RATIO
    assert "marginal rather than damning" in v.reason
    assert report.grade(TOL, WARN) == "warn", "an edge case graded as clean"


# --------------------------------------------------------- the blind spots


def test_a_flip_about_the_aim_axis_is_not_reported_as_verified():
    """Fault 3: wrong sweep face. Invisible here, and it must say so.

    The sensor legitimately comes back consistent -- its sight line and its field
    of view really are unchanged, and claiming otherwise would be a false
    positive. What must not happen is a bare pass.
    """
    cell = healthy_cell(5)
    centre = np.array([5800.0, 0.0, 600.0])
    aim = _unit(np.array([0.2, 0.1, -1.0]))
    M = frame_aimed_along(aim)
    turn = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # 180 about local X
    cell.sensor("FLIP", centre, features=[centre], M=turn.T @ M)
    imp = cell.read()
    flipped = next(s for s in imp.sensors if s.serial == "FLIP")
    assert np.allclose(flipped.aim(0), aim, atol=1e-5), (
        "the fixture's flip moved the sight line, so this is not testing the blind spot"
    )

    report = diagnose(imp.sensors, tol=TOL, warn=WARN)
    v = verdict_for(report, "FLIP")
    assert v.verdict == CONSISTENT
    assert v.fov_blind, "a passing verdict carried no record of what was never tested"
    assert any("sweep face" in t for t in v.fov_blind)
    assert any("photographs" in m for m in report.messages), (
        "the summary does not point at the only referee that can catch this"
    )


def test_every_verdict_carries_the_untested_fault_including_passing_ones():
    report = diagnose(healthy_cell().read().sensors, tol=TOL, warn=WARN)
    assert all(v.fov_blind for v in report.verdicts), (
        "some verdict omitted the untested fault, which is exactly where it would read as a "
        "clean bill of health"
    )


def test_a_sensor_with_no_features_is_untestable_not_consistent():
    """Nothing to contradict is not the same as nothing wrong."""
    s = SensorClaim(serial="LONE", cfov=np.array([5000.0, 0.0, 600.0]),
                    fov_near=np.array(FOV[600.0][0]), fov_far=np.array(FOV[600.0][1]))
    report = diagnose([s], tol=TOL, warn=WARN)
    v = report.verdicts[0]
    assert v.verdict == UNTESTABLE
    assert "no claim" in v.reason and "either way" in v.reason, v.reason
    assert report.grade(TOL, WARN) != "good"


# ---------------------------------------------------------- sessions


def test_session_key_groups_a_sitting_and_ignores_the_clock():
    a = SensorClaim(serial="a", calibrated="12/15/2015 11:13:11 AM")
    b = SensorClaim(serial="b", calibrated="12/15/2015 2:16:37 PM")
    c = SensorClaim(serial="c", calibrated="2/1/2020 8:21:47 AM")
    assert session_key(a) == session_key(b) == "12/15/2015"
    assert session_key(c) != session_key(a)


def test_a_whole_bad_session_is_reported_as_one_mistake():
    """The finding that makes this feature worth more than a list of bad sensors.

    Three sensors wrong by the same amount, all calibrated the same day, is one
    setup error -- and it is fixed in one action, where three separate verdicts
    are three things to go and check.
    """
    cell = healthy_cell(4)
    for i in range(3):
        centre = np.array([6200.0 + 220.0 * i, 100.0 * i, 600.0])
        cell.sensor(f"SESS{i}", centre, features=[centre],
                    shift=np.array([-412.0, 0.0, 0.0]),
                    date="12/15/2015 11:13:11 AM")
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    session = next(s for s in report.sessions if s.key == "12/15/2015")
    assert session.n_wrong == 3 and len(session.serials) == 3
    assert session.verdict == "session", session.reason
    assert abs(session.shared[0] - 412.0) < 1.0, f"shared correction {session.shared}"
    assert session.spread[0] < 1.0, f"the three should agree to a hair: {session.spread}"
    assert "one mistake in the setup" in session.reason
    # It must not pick between the two readings it cannot distinguish.
    assert "positioned wrongly" in session.reason and "nominals" in session.reason, (
        "the session verdict claims to know whether the sensors or the nominals moved"
    )


def test_a_session_with_one_bad_sensor_is_not_blamed_as_a_session():
    cell = healthy_cell(4)
    centre = np.array([6800.0, 0.0, 600.0])
    cell.sensor("ODD", centre, features=[centre], shift=np.array([-412.0, 0.0, 0.0]),
                date="10/13/2019 9:42:24 AM")
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    session = next(s for s in report.sessions if s.key == "10/13/2019")
    assert session.verdict == "mixed", session.reason
    assert "not the explanation" in session.reason


def test_a_session_wrong_by_different_amounts_is_not_called_one_mistake():
    """Three sensors all wrong, but not by one amount, is not one setup error.

    The shifts have to be big enough that all three are actually convicted. An
    earlier version used 150 mm, which is 0.95 of the lateral field of view and so
    correctly came back consistent -- leaving two bad sensors out of three and
    testing the "mixed" branch by accident instead of this one.
    """
    cell = healthy_cell(4)
    for i, dx in enumerate((-412.0, -900.0, -600.0)):
        centre = np.array([6200.0 + 220.0 * i, 0.0, 600.0])
        cell.sensor(f"VAR{i}", centre, features=[centre], shift=np.array([dx, 0.0, 0.0]),
                    date="12/15/2015 11:13:11 AM")
    report = diagnose(cell.read().sensors, tol=TOL, warn=WARN)
    session = next(s for s in report.sessions if s.key == "12/15/2015")
    assert "different amounts" in session.reason, session.reason
    assert "one mistake in the setup" not in session.reason


# -------------------------------------- the consensus, and where it refuses


def test_the_aim_consensus_refuses_on_a_cell_with_no_shared_volume():
    """The design error this module was restructured to fix.

    Sensors spread down a car body do not look at one working volume, so a
    consensus among their aim rays is an intersection of a pencil that is not one.
    It has to refuse rather than return a confident point.
    """
    imp = healthy_cell(8).read()
    got = aim_consensus(imp.sensors, depth_axis=0, tol=WARN)
    assert not got.ok
    assert "not all looking at one working volume" in got.reason, got.reason

    report = diagnose(imp.sensors, tol=TOL, warn=WARN)
    assert any("does not apply to this file" in m for m in report.messages), report.messages


def test_the_aim_consensus_still_works_where_it_applies():
    """A cell that really does ring one part keeps the old machinery."""
    volume = np.array([1500.0, 0.0, 1000.0])
    cell = Cell()
    for i in range(6):
        theta = 2.0 * math.pi * i / 6
        pos = volume + np.array([700.0 * math.cos(theta), 700.0 * math.sin(theta), 120.0 * i])
        cell.sensor(f"R{i}", volume, features=[volume], aim=volume - pos)
    sensors = cell.read().sensors
    got = aim_consensus(sensors, depth_axis=0, tol=WARN)
    assert got.ok, got.reason
    assert len(got.outliers) == 0, f"a clean ring produced outliers {got.outliers}"
    assert np.linalg.norm(np.array(got.point) - volume) < 5.0, got.point


def test_too_few_sensors_refuses_a_consensus_verdict():
    volume = np.array([1500.0, 0.0, 1000.0])
    cell = Cell()
    for i in range(MIN_CONSENSUS_SENSORS - 1):
        pos = volume + np.array([700.0 * math.cos(i), 700.0 * math.sin(i), 0.0])
        cell.sensor(f"T{i}", volume, features=[volume], aim=volume - pos)
    got = aim_consensus(cell.read().sensors, depth_axis=0, tol=WARN)
    assert not got.ok
    assert str(MIN_CONSENSUS_SENSORS) in got.reason, (
        "the refusal does not say how many sensors would be enough"
    )


# ------------------------------------------------------------- the units check


def test_the_span_notices_a_file_written_in_metres():
    """The 1000x mistake ``part.py`` exists to catch, in its calibration clothing."""
    mm = healthy_cell().read()
    assert 1000.0 < mm.span < 10000.0, f"a cell-sized span read as {mm.span:.1f}"
    assert not any("metres" in w for w in mm.warnings)

    cell = Cell()
    for i in range(4):
        centre = np.array([4.6 + 0.5 * i, -0.3, 0.6])
        cell.sensor(f"M{i}", centre, features=[centre])
    m = cell.read()
    assert m.span < 10.0
    assert any("metres" in w for w in m.warnings), (
        f"a file in metres drew no units warning: {m.warnings}"
    )
    assert any("Nothing was rescaled" in w for w in m.warnings), (
        "the warning does not say that nothing was silently converted"
    )


# ------------------------------------------- the conventions, which still bite


def test_the_euler_order_changes_the_answer():
    """Why the convention is an argument with no default worth having.

    Not needed by this format, which carries direction cosines, but kept for the
    files that do: the same three angles under two orders are not nearly the same
    rotation, and neither call fails.
    """
    angles = [20.0, 35.0, 50.0]
    a, b = euler_matrix(angles, "XYZ"), euler_matrix(angles, "ZYX")
    assert np.linalg.norm(a - b) > 0.3, "this fixture is not exercising the trap"
    for M in (a, b):
        assert abs(np.linalg.det(M) - 1.0) < 1e-12
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)


def test_intrinsic_and_extrinsic_are_the_same_product_reversed():
    angles = [11.0, -24.0, 63.0]
    assert np.allclose(
        euler_matrix(angles, "XYZ", intrinsic=True),
        euler_matrix(angles[::-1], "ZYX", intrinsic=False),
    )


def test_a_quaternion_round_trips_and_its_scalar_end_is_detectable():
    axis = _unit(np.array([0.3, -0.5, 0.81]))
    half = math.radians(6.0)
    first = [math.cos(half), *(axis * math.sin(half))]
    last = [*(axis * math.sin(half)), math.cos(half)]
    R = quaternion_matrix(first, scalar_first=True)
    assert np.allclose(R, quaternion_matrix(last, scalar_first=False))
    assert abs(math.degrees(math.acos((np.trace(R) - 1.0) / 2.0)) - 12.0) < 1e-6
    assert quaternion_scalar_position([first, first]) == "scalar_first"
    assert quaternion_scalar_position([last, last]) == "scalar_last"


def test_the_scalar_end_is_undetectable_for_large_rotations():
    """And the detector has to say so instead of inverting.

    Past about 51 degrees the scalar stops being the largest component, so a
    scalar-first file of large rotations has its maximum in the last slot --
    indistinguishable, by that test, from a scalar-last file of small ones. They
    are opposite answers. An earlier version returned ``scalar_last`` for a
    scalar-first input, confidently.
    """
    axis = _unit(np.array([0.3, -0.5, 0.81]))
    half = math.radians(85.0)
    big = [math.cos(half), *(axis * math.sin(half))]
    assert int(np.argmax(np.abs(big))) == 3, "the fixture's rotation is not large enough"
    assert quaternion_scalar_position([big, big]) is None, (
        "a scalar-first quaternion with a large rotation was reported as scalar-last"
    )
    small = [math.cos(math.radians(3.0)), *(axis * math.sin(math.radians(3.0)))]
    assert quaternion_scalar_position([small]) is None, (
        "one sample satisfies 'the same end always wins' trivially and is never evidence"
    )
    assert quaternion_scalar_position([small, small]) == "scalar_first"


def test_every_catalogued_fault_is_a_proper_rotation_and_leaves_the_claim_alone():
    """A reflection applied carelessly yields determinant -1 and no valid pose."""
    s = healthy_cell().read().sensors[2]
    M0, c0 = np.array(s.M), np.array(s.cfov)
    for fault in fault_catalogue():
        M, c = fault.apply(s.M, s.cfov)
        # The tolerance tracks the file's five decimal places, not machine
        # precision: a correction must not *degrade* a block that arrived 1e-5
        # from orthonormal, and cannot improve on it either.
        assert abs(float(np.linalg.det(M)) - 1.0) < 1e-4, (
            f"{fault.key} produced determinant {np.linalg.det(M):.6f}, which is not a rotation"
        )
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-4), f"{fault.key} is not orthonormal"
        assert np.allclose(s.M, M0) and np.allclose(s.cfov, c0), (
            f"{fault.key} mutated the claim it was given, so the file's own numbers are gone"
        )


def test_the_file_s_own_numbers_survive_the_import():
    """An import has to be reversible: the verdict's next question is always
    "what did the file actually say"."""
    imp = healthy_cell(4).read()
    for s in imp.sensors:
        assert s.raw, "the sensor kept no copy of its own row"
        assert s.raw["Serial Number"].strip() == s.serial
        assert float(s.raw["cfov(x)"]) == s.cfov[0]
        for feature in s.features:
            assert feature.raw and feature.raw["Entity Name"].strip() == feature.name


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
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  ERR  {fn.__name__}: {type(exc).__name__}: {exc}")

    cell = healthy_cell(4)
    for i in range(3):
        centre = np.array([6200.0 + 220.0 * i, 80.0 * i, 600.0])
        cell.sensor(f"SESS{i}", centre, features=[centre],
                    shift=np.array([-412.0, 0.0, 0.0]), offsets=(412.0, 0.0, 0.0),
                    date="12/15/2015 11:13:11 AM")
    imp = cell.read()
    report = diagnose(imp.sensors, tol=TOL, warn=WARN)
    print(f"\n{len(imp.sensors)} sensors / {imp.n_features} features, span {imp.span:.0f} mm, "
          f"range axis {report.depth_axis}, grade {report.grade(TOL, WARN)}")
    for v in sorted(report.verdicts, key=lambda v: -v.ratio):
        tag = {CONSISTENT: "ok  ", EDGE: "edge", EXPLAINED: "FIX ", SUSPECT: "BAD ",
               UNTESTABLE: "?   "}[v.verdict]
        print(f"    {tag} {v.serial:8s} ratio {v.ratio:6.2f}  after {v.ratio_after:5.2f}  "
              f"axis {v.overrun_axis}  correction "
              f"{[round(float(x)) for x in v.correction]}")
    for s in report.sessions:
        if s.n_wrong:
            print(f"    session {s.key}: {s.verdict}, {s.n_wrong}/{len(s.serials)} wrong, "
                  f"shared {[round(float(x)) for x in s.shared]}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
