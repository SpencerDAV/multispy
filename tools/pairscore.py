"""Which image pairs actually share a view, scored against poses you already have.

COLMAP decides a pair is verified by fitting an essential matrix to the matches
and counting what agrees with it.  That is an in-sample test, and on a sparse
wide-baseline capture it is a generous one: hand it forty wrong
correspondences and RANSAC will usually find some model that eighteen of them
endorse, at which point the pair is reported verified and the dense pipeline
believes a connection that is not there.  Nothing in the number tells you
which happened.

A solved MultiSpy project can settle it, because it already knows where the
cameras are.  The poses imply an essential matrix per pair with nothing fitted
to the matches at all, so scoring the same correspondences against *that* is
out-of-sample with respect to the matcher.  A pair that passes COLMAP's test
and fails this one is a pair whose matches are self-consistent and wrong.

So two tables come out, on the same matches, at COLMAP's own thresholds
(4 px Sampson, 15 inliers):

  A.  Fitted to the matches -- what COLMAP reports, and what a matcher
      comparison usually quotes.
  B.  Against the project's poses -- what the matches are worth.

Read B.  A pair flagged ``verified and wrong`` in the per-pair listing passed A
and failed B, and the Sampson column says by how much; triple digits there is
not a slightly misplaced correspondence, it is a different part of the cell.

    .venv\\Scripts\\python.exe -m multispy export projects/rearmod3/project.multispy out
    colmap feature_extractor --database_path out/db --image_path out/images
    colmap exhaustive_matcher --database_path out/db
    .venv\\Scripts\\python.exe tools\\pairscore.py out out/db

Measured on `projects/rearmod3` with three matchers (see TODO.md): every one of
them put 0 of the 11 pairs more than 60 degrees apart into table B, and LoFTR
at 1280 px put 3 of them into table A.  The bands are reported separately
because that is where the answer lives -- a capture whose wide band is empty
cannot be densified by a better matcher, however much the total grows.

A caution about the reference.  Table B is only as good as the poses, and on a
survey-style capture those poses are themselves corroborated mostly by the
narrow pairs -- so B cannot, on its own, distinguish "no shared view" from
"poses wrong out here".  Read it beside A on a matcher that declines rather
than guesses: when a matcher returns no candidates at all for a pair, that is
evidence from the imagery and owes nothing to the poses.

The `--band` edges are a reporting choice, not a measurement.  Angles are
between the cameras' optical axes, which is what governs whether patch
matching can work at all; the baseline is printed beside them because a pair
can be close in angle and still useless for depth.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

MAX_IMAGE_ID = 2147483647
MAX_ERROR = 4.0      # COLMAP TwoViewGeometry.max_error default
MIN_INLIERS = 15     # COLMAP TwoViewGeometry.min_num_inliers default


# --------------------------------------------------------------------------
# the exported model


def read_cameras(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = dict(model=t[1], width=int(t[2]), height=int(t[3]),
                               params=[float(x) for x in t[4:]])
    return cams


def read_images(path: Path) -> dict:
    imgs = {}
    lines = [l for l in path.read_text().splitlines() if not l.startswith('#')]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        t = lines[i].split()
        imgs[int(t[0])] = dict(q=np.array([float(x) for x in t[1:5]]),
                               t=np.array([float(x) for x in t[5:8]]),
                               camera=int(t[8]), name=t[9])
        i += 2
    return imgs


def rotation(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def skew(v) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def normalise(xy: np.ndarray, cam: dict) -> np.ndarray:
    """Pixels to normalised camera coordinates, undoing the radial term."""
    model, P = cam['model'], cam['params']
    if model in ('PINHOLE',):
        fx, fy, cx, cy = P
        return np.stack([(xy[:, 0] - cx) / fx, (xy[:, 1] - cy) / fy], 1)
    if model == 'SIMPLE_PINHOLE':
        f, cx, cy = P
        return np.stack([(xy[:, 0] - cx) / f, (xy[:, 1] - cy) / f], 1)
    if model != 'SIMPLE_RADIAL':
        raise SystemExit('pairscore: unhandled camera model %s' % model)
    f, cx, cy, k = P
    u = (xy[:, 0] - cx) / f
    v = (xy[:, 1] - cy) / f
    if k == 0:
        return np.stack([u, v], 1)
    x, y = u.copy(), v.copy()
    for _ in range(20):                      # Newton on x_d = x (1 + k r^2)
        r2 = x * x + y * y
        d = 1 + k * r2
        rx, ry = x * d - u, y * d - v
        j11, j12 = d + 2 * k * x * x, 2 * k * x * y
        j21, j22 = 2 * k * x * y, d + 2 * k * y * y
        det = j11 * j22 - j12 * j21
        x -= (j22 * rx - j12 * ry) / det
        y -= (-j21 * rx + j11 * ry) / det
    return np.stack([x, y], 1)


# --------------------------------------------------------------------------
# the database


def pair_id(a: int, b: int) -> int:
    return min(a, b) * MAX_IMAGE_ID + max(a, b)


def keypoints(conn, image_id: int) -> np.ndarray:
    row = conn.execute('select rows, cols, data from keypoints where image_id=?',
                       (image_id,)).fetchone()
    if row is None or row[0] == 0 or row[2] is None:
        return np.zeros((0, 2), np.float32)
    rows, cols, data = row
    return np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2]


def index_pairs(conn, table: str, pid: int) -> np.ndarray:
    row = conn.execute('select rows, cols, data from %s where pair_id=?' % table,
                       (pid,)).fetchone()
    if row is None or row[0] == 0 or row[2] is None:
        return np.zeros((0, 2), np.uint32)
    rows, cols, data = row
    return np.frombuffer(data, np.uint32).reshape(rows, cols)


# --------------------------------------------------------------------------


def band_of(angle: float, edges) -> tuple:
    for lo, hi in zip(edges, list(edges[1:]) + [180.0]):
        if lo <= angle < hi:
            return (lo, hi)
    return (edges[-1], 180.0)


def print_table(title: str, counts: dict, bands, footer: str = '') -> None:
    print(title)
    print('| view angle apart | pairs | matched | median inliers | best |')
    print('|---|---|---|---|---|')
    for band in bands:
        v = counts[band]
        ok = [x for x in v if x > 0]
        label = ('%g-%g deg' % band) if band[1] < 180 else ('%g deg +' % band[0])
        print('| %s | %d | %d | %d | %d |' % (
            label, len(v), len(ok),
            int(np.median(ok)) if ok else 0, max(ok) if ok else 0))
    every = [x for v in counts.values() for x in v]
    ok = [x for x in every if x > 0]
    print('| all | %d | %d | %d | %d |' % (
        len(every), len(ok), int(np.median(ok)) if ok else 0, max(ok) if ok else 0))
    if footer:
        print(footer)
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('export_dir', help='an export directory, or its sparse/ model')
    ap.add_argument('database', help='a COLMAP database holding keypoints and matches')
    ap.add_argument('--band', default='0,20,40,60',
                    help='lower edges of the angle bands, degrees (default 0,20,40,60)')
    ap.add_argument('--max-error', type=float, default=MAX_ERROR,
                    help='Sampson threshold in px for table B (default %g)' % MAX_ERROR)
    ap.add_argument('--min-inliers', type=int, default=MIN_INLIERS,
                    help='matches a pair needs in table B (default %d)' % MIN_INLIERS)
    ap.add_argument('--quiet', action='store_true', help='tables only, no per-pair listing')
    a = ap.parse_args(argv)

    root = Path(a.export_dir)
    sparse = root if (root / 'images.txt').exists() else root / 'sparse'
    if not (sparse / 'images.txt').exists():
        print('pairscore: no images.txt under %s' % sparse, file=sys.stderr)
        return 2

    cams = read_cameras(sparse / 'cameras.txt')
    model = read_images(sparse / 'images.txt')
    by_name = {v['name']: k for k, v in model.items()}

    conn = sqlite3.connect(a.database)
    db_id = {name: i for i, name in conn.execute('select image_id, name from images')}
    missing = sorted(set(by_name) - set(db_id))
    if missing:
        print('pairscore: not in the database: %s' % ', '.join(missing), file=sys.stderr)
        return 2
    # Images are matched by name on purpose.  COLMAP's extractor does not
    # insert them in filename order, so the ids in the database and in the
    # exported model need not agree -- the mismatch that makes
    # point_triangulator die with `existing_frame.DataIds() == frame.DataIds()`.
    names = sorted(by_name)

    axis, centre = {}, {}
    for n in names:
        R = rotation(model[by_name[n]]['q'])
        axis[n] = R.T @ np.array([0.0, 0.0, 1.0])
        centre[n] = -R.T @ model[by_name[n]]['t']

    edges = tuple(float(x) for x in a.band.split(','))
    bands = [band_of(e, edges) for e in edges]
    counts_a = {b: [] for b in bands}
    counts_b = {b: [] for b in bands}
    samp_a, samp_b = [], []
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            cos = float(np.clip(axis[na] @ axis[nb], -1.0, 1.0))
            angle = math.degrees(math.acos(cos))
            baseline = float(np.linalg.norm(centre[na] - centre[nb]))

            # COLMAP stores a pair's indices in ascending image-id order
            lo, hi = sorted((db_id[na], db_id[nb]))
            name_lo = na if db_id[na] == lo else nb
            name_hi = nb if db_id[nb] == hi else na
            cam_lo = cams[model[by_name[name_lo]]['camera']]
            cam_hi = cams[model[by_name[name_hi]]['camera']]
            kp_lo, kp_hi = keypoints(conn, lo), keypoints(conn, hi)

            R1 = rotation(model[by_name[name_lo]]['q'])
            R2 = rotation(model[by_name[name_hi]]['q'])
            t1 = model[by_name[name_lo]]['t']
            t2 = model[by_name[name_hi]]['t']
            R = R2 @ R1.T
            E = skew(t2 - R @ t1) @ R
            focal = cam_lo['params'][0]

            def sampson(idx: np.ndarray) -> np.ndarray:
                p1 = normalise(kp_lo[idx[:, 0].astype(int)], cam_lo)
                p2 = normalise(kp_hi[idx[:, 1].astype(int)], cam_hi)
                x1 = np.concatenate([p1, np.ones((len(p1), 1))], 1)
                x2 = np.concatenate([p2, np.ones((len(p2), 1))], 1)
                l2 = (E @ x1.T).T
                l1 = (E.T @ x2.T).T
                num = np.sum(x2 * l2, 1) ** 2
                den = l2[:, 0] ** 2 + l2[:, 1] ** 2 + l1[:, 0] ** 2 + l1[:, 1] ** 2
                return np.sqrt(num / den) * focal

            raw = index_pairs(conn, 'matches', pair_id(lo, hi))
            inl = index_pairs(conn, 'two_view_geometries', pair_id(lo, hi))

            n_a, med_a = len(inl), float('nan')
            if n_a:
                s = sampson(inl)
                samp_a.append(s)
                med_a = float(np.median(s))

            n_b, med_b = 0, float('nan')
            if len(raw):
                s = sampson(raw)
                keep = s < a.max_error
                if int(keep.sum()) >= a.min_inliers:
                    n_b = int(keep.sum())
                    samp_b.append(s[keep])
                    med_b = float(np.median(s[keep]))

            band = band_of(angle, edges)
            counts_a[band].append(n_a)
            counts_b[band].append(n_b)
            rows.append((na, nb, angle, baseline, len(raw), n_a, med_a, n_b, med_b))

    total_kp = sum(len(keypoints(conn, db_id[n])) for n in names)
    print('%s' % a.database)
    print('%d photographs, %d pairs, %d keypoints' % (len(names), len(rows), total_kp))
    print()

    footer = ''
    if samp_a:
        s = np.concatenate(samp_a)
        footer = ('%d verified matches; against the project poses their Sampson error is '
                  'median %.2f px, 90th %.2f px, under 3 px %.0f%%'
                  % (len(s), np.median(s), np.percentile(s, 90), 100 * np.mean(s < 3)))
    print_table('A. essential matrix fitted to the matches (what COLMAP reports):',
                counts_a, bands, footer)

    footer = ''
    if samp_b:
        s = np.concatenate(samp_b)
        footer = ('%d matches within %.0f px; median %.2f px, 90th %.2f px'
                  % (len(s), a.max_error, np.median(s), np.percentile(s, 90)))
    print_table('B. against the project poses (<%.0f px Sampson, >=%d matches):'
                % (a.max_error, a.min_inliers), counts_b, bands, footer)

    wrong = [r for r in rows if r[5] >= a.min_inliers and r[7] == 0]
    if wrong:
        print('%d pair(s) passed A and failed B:' % len(wrong))
        for r in wrong:
            print('  %s %s  %.1f deg  %d COLMAP inliers at %.0f px median Sampson'
                  % (r[0], r[1], r[2], r[5], r[6]))
        print()

    if not a.quiet:
        print('per-pair, by angle.  A = COLMAP inliers and their median Sampson against')
        print('the project poses; B = raw matches within %.0f px of those poses.' % a.max_error)
        print('%-14s %-14s %9s %8s %7s %7s %9s %7s' % (
            'imageA', 'imageB', 'angle', 'baseline', 'raw', 'A', 'sampson', 'B'))
        for r in sorted(rows, key=lambda x: x[2]):
            flag = '   <- verified and wrong' if (r[5] >= a.min_inliers and r[7] == 0) else ''
            print('%-14s %-14s %5.1f deg %7.2fm %7d %7d %9.2f %7d%s' % (
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], flag))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
