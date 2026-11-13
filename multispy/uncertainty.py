"""How well is the part-space origin actually pinned down?

The origin is not a marked feature. It is wherever the nominal'd points say it
is, which makes it an *extrapolation* from the datum triangle rather than a
measurement, and extrapolations get worse with distance: a frame whose three
datum points span a metre, carrying an origin a metre and a half away, hands
any angular error in that triangle straight to the origin with a lever on it.
That is not visible in the reprojection RMS, and it is not visible in the
asked-versus-achieved nominal table either, because the origin sits at (0,0,0)
in its own frame *by construction* -- the fit puts it there however bad the
data is. A project can show a perfect datum table and still not know where its
origin is to a centimetre.

So the number has to come from somewhere else. This module re-solves the
project many times with every clicked mark and control-line endpoint jittered
by its stated sigma, and watches where the origin lands relative to the
*scene*: the ordinary tie points are the only reference two solves share, so
each sample is aligned to the reference solve through them and the origin is
carried along for the ride. The scatter that comes out is what click noise
alone does to the origin.

"Click noise alone" is the honest limit of it. It does not know about a nominal
that is simply wrong, about a datum point clicked on the wrong feature in every
photograph, or about nominals derived from the solve they are then used to
anchor -- that last one being self-consistent by construction, so it inflates
nothing here while validating nothing either. Read the sphere as a floor on the
uncertainty, never a ceiling.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from .geometry import similarity_fit
from .project import Project

DEFAULT_SAMPLES = 32
MIN_SAMPLES = 8
CONTAINMENT = 0.95
"""Fraction of the origin's probability mass the drawn sphere holds."""


@dataclass
class OriginUncertainty:
    """Where the part-space origin lands under click noise, in nominal units."""

    n: int = 0
    """Samples that converged. Below ``MIN_SAMPLES`` the figures are indicative
    at best -- a p95 from a handful of draws is itself very uncertain."""
    n_requested: int = 0
    radius: float = 0.0
    """The sphere drawn in the UI: the origin is inside it 95% of the time.

    Read off a large cheap draw from the *fitted* covariance rather than
    straight from the re-solves, because an empirical p95 taken from a few
    dozen draws is very nearly the largest of them and swings by tens of
    percent between runs, while the covariance it is computed from settles
    down much faster. Using the whole covariance and not just the worst sigma
    keeps the honest meaning -- a ball circumscribing the 95% ellipsoid would
    be much bigger than the 95% ball, because the scatter is a long way from
    spherical."""
    p95: float = 0.0
    """The empirical 95th percentile of sample distance, for comparison. If it
    is wildly under ``radius`` the scatter is very anisotropic; if it is over,
    the distribution has tails a covariance does not describe."""
    rms: float = 0.0
    sigma: list[float] = field(default_factory=list)
    """Per principal axis, largest first. Usually far from equal: the lever
    direction is much worse than the two across it, and a sphere drawn at
    ``radius`` is a ball around an ellipsoid -- deliberately generous in the
    good directions rather than flattering in the bad one."""
    axes: list[list[float]] = field(default_factory=list)
    """The principal directions in world axes, matching ``sigma``."""
    bias: list[float] = field(default_factory=list)
    """Mean offset of the samples. Should be near zero; a bias comparable to
    ``radius`` means the solve is being pulled somewhere, not just blurred."""
    anisotropy: float = 1.0
    """Largest sigma over smallest. 1 is a true sphere; large means the sphere
    is a poor summary of an ellipsoid and the axes are worth reading."""
    units: str = "mm"
    sigma_px: float = 0.0
    sigma_line_px: float = 0.0
    seed: int = 0
    fingerprint: str = ""
    """What the project looked like when this was measured, so a later edit can
    mark the figure stale instead of quietly showing it against new data."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "OriginUncertainty | None":
        if not d:
            return None
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    def grade(self, good: float, warn: float) -> str:
        """'good' / 'warn' / 'bad', against thresholds in nominal units."""
        if self.n < MIN_SAMPLES:
            return "unknown"
        if self.radius <= good:
            return "good"
        if self.radius <= warn:
            return "warn"
        return "bad"


def fingerprint(project: Project) -> str:
    """A digest of everything the estimate depends on.

    Marks, control lines, nominals and the sigmas that scale the jitter -- not
    the solved poses, which are an output. Edit any of those and the stored
    figure no longer describes the project you are looking at.
    """
    payload = {
        "images": [
            {
                "id": im.id,
                "enabled": im.enabled,
                "obs": {k: [round(float(v), 3) for v in uv]
                        for k, uv in sorted(im.observations.items())},
                "lines": [
                    [ln.axis,
                     [round(float(v), 3) for v in ln.p0],
                     [round(float(v), 3) for v in ln.p1],
                     bool(ln.uncertain)]
                    for ln in im.axis_lines
                ],
            }
            for im in sorted(project.images, key=lambda i: i.id)
        ],
        "nominals": {
            p.id: [p.nominal, p.nominal_sigma]
            for p in sorted(project.points, key=lambda p: p.id)
            if p.nominal
        },
        "scale": [project.nominal_units, project.nominal_scale],
        "sigmas": [
            project.settings.sigma_px,
            project.settings.sigma_line_px,
            project.settings.sigma_nominal,
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _containment_radius(cov: np.ndarray, draws: int = 200_000) -> float:
    """Radius holding ``CONTAINMENT`` of a zero-mean Gaussian with this covariance.

    There is no closed form for the radius of a ball under an anisotropic
    Gaussian -- it is a weighted sum of chi-squares -- so it is sampled. The
    draw is from the fitted covariance, not from the solver, so 200k of them
    cost milliseconds, and the fixed seed keeps the answer reproducible.
    """
    w, V = np.linalg.eigh(cov)
    sd = np.sqrt(np.maximum(w, 0.0))
    g = np.random.default_rng(0).normal(size=(draws, 3)) * sd
    return float(np.percentile(np.linalg.norm(g @ V.T, axis=1), CONTAINMENT * 100))


def origin_uncertainty(
    project: Project,
    n: int = DEFAULT_SAMPLES,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> OriginUncertainty:
    """Monte Carlo the origin's position against click noise.

    ``project`` must already be solved and in part space. It is not modified.
    """
    from .solve import solve   # local: solve imports geometry, and this imports solve

    if not project.in_part_space():
        raise ValueError(
            "no part space: three non-collinear nominals are needed before the origin exists"
        )

    ref = {p.id: np.array(p.xyz, dtype=float) for p in project.points if p.xyz is not None}
    if len(ref) < 4:
        raise ValueError("too few solved points to measure the origin against")

    n = max(int(n), 1)
    sig_px = float(project.settings.sigma_px)
    sig_line = float(project.settings.sigma_line_px)
    rng = np.random.default_rng(seed)
    scale = project.nominal_scale or 0.001

    offsets = []
    for k in range(n):
        trial = copy.deepcopy(project)
        trial.path = None                      # a trial must never be saved over the project
        for im in trial.images:
            for pid, uv in im.observations.items():
                im.observations[pid] = [uv[0] + rng.normal(0, sig_px),
                                        uv[1] + rng.normal(0, sig_px)]
            for ln in im.axis_lines:
                ln.p0 = [v + rng.normal(0, sig_line) for v in ln.p0]
                ln.p1 = [v + rng.normal(0, sig_line) for v in ln.p1]

        report = solve(trial)
        if progress is not None:
            progress(k + 1, n)
        if not report.ok:
            continue
        got = {p.id: np.array(p.xyz, dtype=float) for p in trial.points if p.xyz is not None}
        common = [i for i in ref if i in got]
        if len(common) < 4:
            continue
        # Align this sample to the reference through the scene, then read off
        # where its origin went. Scale is allowed to float: a solve that came
        # back a fraction larger has not moved its origin, it has resized, and
        # charging that to the origin would report a scale error as a position.
        fit = similarity_fit(np.array([got[i] for i in common]),
                             np.array([ref[i] for i in common]),
                             allow_scale=True)
        offsets.append(fit.t)                  # the image of (0,0,0) under the fit

    if not offsets:
        return OriginUncertainty(n=0, n_requested=n, units=project.nominal_units,
                                 sigma_px=sig_px, sigma_line_px=sig_line, seed=seed,
                                 fingerprint=fingerprint(project))

    A = np.array(offsets) / scale              # into nominal units
    mu = A.mean(axis=0)
    D = A - mu
    radii = np.linalg.norm(D, axis=1)
    cov = (D.T @ D) / max(len(A) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    sigma = np.sqrt(np.maximum(w[order], 0.0))

    return OriginUncertainty(
        n=len(A),
        n_requested=n,
        radius=_containment_radius(cov),
        p95=float(np.percentile(radii, 95)),
        rms=float(np.sqrt((radii ** 2).mean())),
        sigma=[float(v) for v in sigma],
        axes=[[float(c) for c in V[:, i]] for i in order],
        bias=[float(v) for v in mu],
        anisotropy=float(sigma[0] / sigma[-1]) if sigma[-1] > 1e-12 else float("inf"),
        units=project.nominal_units,
        sigma_px=sig_px,
        sigma_line_px=sig_line,
        seed=seed,
        fingerprint=fingerprint(project),
    )
