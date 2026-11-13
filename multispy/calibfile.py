"""Reading a measurement cell's calibration export, without believing a word of it.

Separate from :mod:`multispy.sensors` on purpose: this module knows one vendor's
column names and nothing about whether the numbers are any good, and that one
does the judging and knows nothing about file formats. The boundary matters
because the next cell will arrive in a different export and only this file should
have to change.

**Rows are features, not sensors.** That is the trap this module exists to guard,
and it is a quiet one: a Perceptron-style export writes one row per measured
feature, and a sensor that watches four features appears four times with its
transform repeated verbatim. An importer that took rows for sensors would report
the same transform four times, count one calibration fault as four, and let a
sensor with many features outvote one with a single feature in anything that
followed. So rows are grouped by ``Serial Number``, which this module verifies
really is constant in every sensor-level column before trusting it -- on the file
this was built against, 22 rows are 15 sensors, and the sensor columns agree
exactly within every serial.

**What the format happens to settle, which is more than usual.** The orientation
arrives as direction cosines -- a 3x3 block, orthonormal to about 1e-5 with
determinant +1 -- so there is no Euler order to guess and no quaternion end to
pick, which removes the single most dangerous ambiguity in importing poses. The
rows of that block are the sensor's local axes expressed in part coordinates. The
standoff, the field of view and the calibration date are all declared outright.

**What it does not settle**, and is therefore left as an argument rather than
guessed here: whether the range axis points from the sensor at the part or the
other way. The file is self-consistent under either reading, so nothing in it can
decide, and a wrong choice puts the body an equal distance the wrong side of the
part -- obvious the moment it is drawn over a photograph, invisible in any
residual.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .sensors import FeatureClaim, SensorClaim

SENSOR_COLUMNS = (
    "Standoff",
    "Part Number",
    "CA Sweep Face",
    "Calibration Date",
    "cfov(x)",
    "cfov(y)",
    "cfov(z)",
    "x(i)",
    "y(i)",
    "z(i)",
)
"""Columns that describe the sensor rather than the feature.

Checked for constancy within a serial rather than assumed, because the whole
grouping rests on it. If a file ever disagrees with itself here, the right answer
is to say so and stop -- silently keeping the first row's transform would discard
a real contradiction.
"""

REQUIRED = ("Serial Number", "Entity Name", "cfov(x)", "nominal(x)", "x(i)")
"""The minimum to do anything at all. Reported by name when absent, since a
missing column otherwise surfaces as every pose coming back at the origin."""


@dataclass
class ImportReport:
    """What was read, and everything about it worth distrusting."""

    sensors: list[SensorClaim] = field(default_factory=list)
    n_rows: int = 0
    n_features: int = 0
    columns: int = 0
    span: float = 0.0
    """Largest distance between two field-of-view centres, in the file's units.

    The units check, before any judging happens. A measurement cell is metres
    across: a span of 2737 says millimetres, a span of 2.7 says the file is in
    metres and every tolerance downstream is about to be wrong by 1000. Nothing is
    ever rescaled to make it match -- ``part.py`` exists for the same reason, and a
    factor of 1000 applied quietly leaves every deviation reading as a clean zero.
    """
    worst_orthonormality: float = 0.0
    """Largest departure from a true rotation found in any direction-cosine block.

    Direction cosines are written to five decimal places, so about 1e-5 is
    expected and is nothing. Anything near 1e-2 means the block is not a rotation
    -- a transposed or mis-parsed column group, or a genuine scale factor hiding in
    it -- and the pose it describes is not a pose.
    """
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_sensors": len(self.sensors),
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "columns": self.columns,
            "span": self.span,
            "worst_orthonormality": self.worst_orthonormality,
            "warnings": list(self.warnings),
        }


def _number(row: dict[str, str], key: str) -> float | None:
    """A float, or ``None`` for a blank. Never 0.0 for a blank.

    The distinction carries real meaning in these files: a blank offset means the
    system did not measure that axis for that feature, where a zero means it
    measured it and found no deviation. Collapsing the two would invent perfect
    measurements on axes nobody looked at, and those inventions would then pass
    every spec check.
    """
    text = (row.get(key) or "").strip()
    if not text or text.upper() in ("N/A", "NA", "--"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _vector(row: dict[str, str], keys: Sequence[str]) -> np.ndarray:
    return np.array([_number(row, k) or 0.0 for k in keys], dtype=float)


def _matrix(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [[_number(row, f"{a}({c})") or 0.0 for c in "ijk"] for a in "xyz"], dtype=float
    )


def _lines(source: str | Path | Iterable[str]) -> list[str]:
    """Lines from a path, from the file's text, or from an iterable of lines.

    A single-line string is taken as a path and **raises** if it is not there,
    rather than falling back to treating it as content. That fallback was the
    first bug this module had: a mistyped path parsed cleanly as a one-line file,
    which is a header with no rows, and the import reported "no data rows" about a
    file that was never opened. A wrong path has to look like a wrong path.
    """
    if isinstance(source, Path):
        return Path(source).read_text(encoding="utf-8-sig").splitlines()
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source.lstrip("﻿").splitlines()
        return Path(source).read_text(encoding="utf-8-sig").splitlines()
    return list(source)


def read_export(
    source: str | Path | Iterable[str], *, delimiter: str = "\t"
) -> ImportReport:
    """Parse a feature-level calibration export into one claim per sensor.

    ``source`` is a path or any iterable of lines, so a browser build can hand
    over an uploaded file's text without touching a filesystem it does not have.
    """
    lines = _lines(source)

    rows = list(csv.DictReader(lines, delimiter=delimiter))
    report = ImportReport(n_rows=len(rows))
    if not rows:
        report.warnings.append("the file has a header but no data rows")
        return report
    report.columns = len(rows[0])

    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        report.warnings.append(
            f"this does not look like a calibration export: no {', '.join(missing)} "
            f"column(s). Nothing was imported, rather than importing poses at the origin"
        )
        return report

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        serial = (row.get("Serial Number") or "").strip()
        if not serial:
            report.warnings.append(
                f"a row for {(row.get('Entity Name') or '?').strip()!r} has no serial number, "
                f"so it cannot be attached to a sensor and was skipped"
            )
            continue
        grouped.setdefault(serial, []).append(row)

    worst = 0.0
    for serial, group in grouped.items():
        head = group[0]
        for column in SENSOR_COLUMNS:
            if column not in head:
                continue
            seen = {(r.get(column) or "").strip() for r in group}
            if len(seen) > 1:
                report.warnings.append(
                    f"sensor {serial} disagrees with itself in {column!r} across its "
                    f"{len(group)} feature rows ({sorted(seen)}). The first row was used, but "
                    f"this means the rows are not all the same sensor and the grouping is "
                    f"unsafe -- that contradiction is worth resolving before trusting any "
                    f"verdict about {serial}"
                )

        M = _matrix(head)
        err = float(np.abs(M @ M.T - np.eye(3)).max())
        worst = max(worst, err)
        if err > 1e-2:
            report.warnings.append(
                f"sensor {serial}'s direction cosines are {err:.2e} from orthonormal, so that "
                f"block is not a rotation. Either the columns are not what they are labelled "
                f"or a scale is hiding in them; the pose was kept verbatim and not repaired, "
                f"since silently orthonormalising it would hide the problem"
            )
        if float(np.linalg.det(M)) < 0:
            report.warnings.append(
                f"sensor {serial}'s orientation block has a negative determinant, which is a "
                f"reflection and not an orientation any body can be in"
            )

        features = []
        for row in group:
            features.append(
                FeatureClaim(
                    name=(row.get("Entity Name") or "").strip(),
                    nominal=_vector(row, ("nominal(x)", "nominal(y)", "nominal(z)")),
                    algorithm=(row.get("Algorithm") or "").strip(),
                    offsets=tuple(
                        _number(row, f"{a} Offset") for a in ("X", "Y", "Z")
                    ),  # type: ignore[arg-type]
                    limits=tuple(
                        _limits(row, a) for a in ("X", "Y", "Z")
                    ),  # type: ignore[arg-type]
                    raw=dict(row),
                )
            )

        report.sensors.append(
            SensorClaim(
                serial=serial,
                M=M,
                cfov=_vector(head, ("cfov(x)", "cfov(y)", "cfov(z)")),
                fov_near=_vector(head, ("fov near X", "fov near Y", "fov near Z")),
                fov_far=_vector(head, ("fov far X", "fov far Y", "fov far Z")),
                standoff=_number(head, "Standoff") or 0.0,
                part_number=(head.get("Part Number") or "").strip(),
                sweep_face=(head.get("CA Sweep Face") or "").strip(),
                calibrated=(head.get("Calibration Date") or "").strip(),
                features=features,
                raw=dict(head),
            )
        )

    report.n_features = sum(len(s.features) for s in report.sensors)
    report.worst_orthonormality = worst
    if report.sensors:
        C = np.array([s.cfov for s in report.sensors], dtype=float)
        report.span = float(np.linalg.norm(C.max(axis=0) - C.min(axis=0)))
        if report.span < 10.0:
            report.warnings.append(
                f"the fields of view span only {report.span:.3f} across the whole cell, which "
                f"for a measurement cell means the file is in metres while this project's "
                f"nominals are probably in millimetres. Nothing was rescaled: check "
                f"nominal_units before reading any figure below"
            )
    missing_fov = [s.serial for s in report.sensors if float(np.max(np.abs(s.fov_far))) <= 0]
    if missing_fov:
        report.warnings.append(
            f"{len(missing_fov)} sensor(s) declare no field of view "
            f"({', '.join(missing_fov[:6])}{'...' if len(missing_fov) > 6 else ''}), so the "
            f"containment test has nothing to measure them against and they come back "
            f"untestable rather than passing"
        )
    return report


def _limits(row: dict[str, str], axis: str) -> tuple[float | None, float | None]:
    """Spec limits for one axis, falling back to the point-tolerance columns.

    These exports carry two families of limit -- the feature's own and a
    point-tolerance set -- and which one is filled in varies by feature type. The
    fallback is deliberate rather than tidy: a feature with blank spec limits and
    populated point tolerances is not a feature without a tolerance.
    """
    usl, lsl = _number(row, f"{axis} USL"), _number(row, f"{axis} LSL")
    if usl is None and lsl is None:
        usl, lsl = _number(row, f"{axis} PTUSL"), _number(row, f"{axis} PTLSL")
    return usl, lsl
