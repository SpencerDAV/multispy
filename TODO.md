# TODO

Open work, newest first. Each item says what it is for and, where it has been
measured, what it is *not* worth — a proposal with a measured ceiling on it is
worth more than one without.

---

## Done: the rotation comes from the best-determined axis pair (Nov 2026)

`calibrate_image` built every rotation from `axes[0], axes[1]` — X and Y
whenever both were marked — and derived the third axis by cross product. The
focal in the same function has always been an inverse-variance weighted average
over *every* axis pair, "because a pair whose vanishing points sit near the
image is worth orders of magnitude more than one relying on a near-infinite
vanishing point". Only the rotation went on taking whichever pair came first.
`observability.best` was computed forty lines above it and never read.

It leaves no residual, which is why it survived. A pair whose vanishing points
sit far outside the frame fixes its two directions badly and the rotation built
on them is *confident*: the lines fit it to the pixel, the focal absorbs the
rest, and the orthogonality figure stays small because it is measured on those
same two directions. The only hint was the per-image `focal_sigma_pct`, which
is reported against the **best** pair — not the pair the rotation used.

Noticed 2026-11-20 during the third-axis-flip work and filed as *surprising but
cosmetic*, with the workaround "the lever is which axes you supply". Measured
2026-11-21. It is not cosmetic.

On `projects/alberto` — nine handheld phone photographs, mixed portrait and
landscape, mixed lens — the choice differs on four of the eight photographs and
moves those rotations by 0.48 to 1.19 deg:

| image | pair used / its focal sigma | best pair / sigma | rotations differ |
|---|---|---|---|
| img5 | X×Y / **51.5%** | X×Z / 2.8% | **1.19 deg** |
| img3 | X×Y / 16.4% | X×Z / 1.2% | 0.83 deg |
| img2 | X×Y / 2.8% | X×Z / 1.3% | 0.54 deg |
| img0 | X×Y / 2.9% | Y×Z / 2.8% | 0.48 deg |

| | before | after |
|---|---|---|
| group A (img3 4 5 7) | 0.510 / 1.132 px | 0.510 / 1.132 px |
| group C (img0 img2) | 0.335 / 1.036 px | 0.335 / 1.036 px |
| **all 8 photographs** | **34.749 / 32.352 px** | **2.222 / 3.026 px** |

Per image on the eight-photograph solve: img0 **117.95 → 0.27 px**, img2's
control-line RMS **84.67 → 1.19 px**. The cull note goes from "a further 15
mark(s) look just as bad but were kept" to "a further 1". Fifteen marks were
being blamed for a rotation error.

**Neutral on every other project** — artifex4, perceptron, prm1 and rearmod3 are
identical to the digit both ways. That is *not* because the choice never
differs, which was the first guess and is wrong. The pair changes on **21 of the
34 photographs** in `projects/`:

| project | images whose pair changed | solve moved? |
|---|---|---|
| **alberto** | **4 of 8** (img0 XY→YZ, img2/3/5 XY→XZ) | **yes, 15×** |
| prm1 | 7 of 9 | no |
| rearmod3 | 7 of 8 | no |
| perceptron | 3 of 7 | no |
| artifex4 | 0 of 2 | — |

So the change moves the *initialisation* on two thirds of every photograph in
the collection, and the converged answer on one project. That is a better
safety argument than neutrality would have been: the bundle demonstrably
recovers from the worse initialisation when the geometry lets it, and the one
place it cannot is the one place the old choice was doing damage.

**Why alberto and not the others — the honest version.** Which pair is well
determined depends on which world axis runs near the optical axis and on how
much of each axis the frame can see, and turning a phone on its side changes
the second by a factor of two (alberto's img8 has a ±40.5 deg vertical field of
view in portrait against img9's ±21.5 in landscape). So a mixed-orientation set
should be exactly where this bites. But alberto is **the only mixed-orientation
set in `projects/`** — every other is single-orientation and single-camera —
*and* the only one whose cameras sit within a hundredth of the scene of each
other. The two explanations are confounded in this collection and it cannot say
which is the discriminator. What it does say: on four single-orientation sets
with ordinary baselines the bundle absorbed the worse initialisation entirely,
and on the mixed-orientation near-panorama it did not.

`Calibration.rotation_axes` now reports the pair, because the old answer was
`axes_used[:2]` by construction and callers read it off there —
`tests/test_axissign.py` among them. A reversed line on the derived axis is
cosmetic while one on either chosen axis turns the camera half a turn, so
anything reasoning about that has to be able to ask.

The whole suite — all 25 files in `tests/` — passes with the change in, and was
confirmed passing before it, so nothing here is inherited. `test_axissign` was
the only existing test that needed touching: it read `axes_used[:2]` as the
rotation pair, which was true by construction and no longer is.

`tests/test_rotationpair.py` pins it on a geometry built from first principles:
a camera turned so world X runs nearly parallel to the image plane, which sends
X's vanishing point toward infinity while Y and Z stay well placed. Over 80
draws of the pixel noise the median rotation error falls 1.3× to 1.7× and the
90th percentile 1.2× to 1.8×, and both ratios grow as X is tipped further out of
view — the conditioning gap is the mechanism, so that trend is the real claim.

Two things deliberately **not** asserted. The worst draw: it reads well at the
shallow end (9.1 deg against 3.8) but at tip 0.10 the bad pair's worst, 3.31
deg, is *better* than the good pair's 3.79, and an extremum over 80 samples is
not a statistic. And per-draw superiority: the first version of the test
demanded it and failed at tip 0.02, seed 29 — 0.96 deg against 1.02. A
worse-conditioned pair is not reliably wrong on one draw, it is higher
variance, and it gets lucky on about a fifth of them. Both facts are pinned as
facts so the next reader does not "disprove" the change from a single
photograph.

### Still open: use all three measured directions, not two

Where all three axes are marked the third measured direction is used for the
right-handed check and then discarded. A weighted orthogonal Procrustes fit of
all three, each weighted by its own vanishing point's covariance, would use
everything that was clicked. Unmeasured, so no claim about what it is worth —
but note the ceiling: on alberto the two chosen directions are already good to
3.2% focal-equivalent, so this is a refinement of a solved problem, not another
15×.

## Done: IMAGE_ID is read out of COLMAP rather than guessed (Nov 2026)

`point_triangulator` refuses a model whose `IMAGE_ID`s disagree with the
database it is handed, and says so as `Check failed: existing_frame.DataIds() ==
frame.DataIds()` — a line about rig bookkeeping that names nothing of the cause.
The export used to guess that numbering by sorting on filename, on the premise
that COLMAP's extractor walks a directory in that order. **It does not.**
Measured on 4.1.0.dev0 over the eight photographs of `projects/rearmod3`, the
extractor inserted `IMG_6404` as id 4 and `IMG_6403` as id 5 — its reader is
threaded — and the check failed again, which is how the matcher investigation
below tripped over it.

So the numbering is read now. `export_colmap(..., database=...)`, exposed as
`multispy export --colmap-db PATH`, takes the ids straight out of a database
that already holds them, and the order of operations is export, extract, export
again:

```
multispy export project.multispy export\
colmap feature_extractor --database_path export\database.db --image_path export\images
multispy export project.multispy export\ --colmap-db export\database.db
```

The second export is cheap, because a repeat export re-encodes no photograph:
it rewrites the three text files and leaves `images/` alone.

| handed to `point_triangulator` | result |
|---|---|
| the export as it stood, no database | **refused**, `existing_frame.DataIds() == frame.DataIds()` |
| the same export with `--colmap-db` | 8 registered images, **61 points, 166 obs, 1.628 px** |

That is the same 61 / 166 / 1.63 px the matcher work reached with a scratch
renumbering script in the middle, so this buys the number back without the
script. The poses moved by 4e-11 over the run, which is a text round trip and
not a bundle adjustment.

Two things deliberately left alone. A database that does not know one of the
photographs is an **error**, not a gap to number around — the usual cause is
features extracted from the originals rather than from the exported upright
copies, and numbering around that would be silently wrong. And only the id
column moves: the lines stay in filename order, because COLMAP's incremental
triangulator walks the images in the order the file lists them and is greedy,
so reordering them changes the point count it arrives at — by one point in 61
here. Nobody's bug, and not worth inheriting.

Filename order survives as the fallback for the first export, because something
has to be on disk before a database can exist. It is a guess, it is now
written down as one, and a model numbered that way is still right for Blender,
`transforms.json` and the PLY. `point_triangulator` is the only reader that
counts the ids.

`tests/test_colmapids.py` pins the lot: a database that numbers two photographs
the other way round from their names, one that skips numbers, one that agrees
with filename order by coincidence, the tracks in `points3D.txt` following the
renumbering, and both ways of being handed the wrong database.

---

## Rejected: a learned matcher to rescue the rearmod3 capture (Nov 2026)

Eleven of the 28 image pairs in `projects/rearmod3` are more than 60 degrees
apart in viewing direction, and SIFT matched nothing in any of them. That is
the whole of why these eight photographs cannot feed a dense reconstruction, so
the obvious next move was a matcher that does not depend on a repeatable
detector. Measured, with MultiSpy's poses held fixed throughout:

- **LoFTR**, kornia 0.9.2, pretrained `outdoor`, confidence 0.5, at 640, 1024
  and 1280 px long side. 28 pairs in 4.7 s on a 4070 SUPER.
- **ALIKED-N16ROT + LightGlue**, which COLMAP 4.1 ships as ONNX and will run
  itself — `--FeatureExtraction.type ALIKED_N16ROT`,
  `--FeatureMatching.type ALIKED_LIGHTGLUE`.

Same upright exports, same `colmap geometric_verifier` at stock thresholds, so
it is like for like with the SIFT baseline. Verified pairs per band, of 28,
read as *COLMAP's own verification / scored against the poses*:

| matcher | 0-20 (6) | 20-40 (5) | 40-60 (6) | **60 deg + (11)** | total | points |
|---|---|---|---|---|---|---|
| SIFT default | 5 / 5 | 4 / 2 | 3 / 2 | **0 / 0** | 12 / 9 | 61 |
| LoFTR 640 | 6 / 5 | 4 / 2 | 2 / 2 | **0 / 0** | 12 / 9 | 220 |
| LoFTR 1024 | 6 / 5 | 4 / 3 | 3 / 2 | **0 / 0** | 14 / 10 | 423 |
| LoFTR 1280 | 6 / 5 | 4 / 3 | 3 / 2 | **3 / 0** | 16 / 10 | 613 |
| ALIKED + LightGlue | 6 / 5 | 5 / 4 | 3 / 3 | **0 / 0** | 14 / 12 | 513 |

**The wide band is dead in every run.** Three matchers, four resolutions, two
independent scorings: 0 of 11. The capture protocol is the lever here, not the
software.

What the matchers do buy is correspondence on the pairs that already worked.
693 verified matches becomes 3,560 (LoFTR 1024) or 3,873 (ALIKED+LightGlue),
and `point_triangulator` with the poses fixed goes from **61 points / 166 obs /
1.63 px** to **613 / 1700 / 1.71 px** at LoFTR's best. Seven times the points
at the same reprojection error, which is a real improvement to a sparse model
and may well be worth having for its own sake.

It is not worth having *for a dense reconstruction*, and the dense run says so
directly rather than by argument. Identical MVS on each model — undistort at
2000 px, patch match with geometric consistency, stereo fusion:

| | sparse points | photographs with observations | depth maps | fused points |
|---|---|---|---|---|
| SIFT default | 61 | 5 of 8 | 5 | **3,680** |
| LoFTR 1024 | 423 | 5 of 8 | 5 | **3,241** |
| ALIKED + LightGlue | 513 | 7 of 8 | 7 | **3,356** |

Every richer model produced a *smaller* dense cloud than the 61-point one, on
a 3.1 m cell. The sparse points only pick source views; patch matching works
from the poses and the pixels, and the pixels did not change. The three
photographs contributing nothing to the SIFT model — `IMG_6399`, `IMG_6401`,
`IMG_6407` — are the same three after LoFTR, and they get no depth map because
they have no source view to draw on.

The ALIKED row is the one that settles it, because that matcher did the thing
the whole exercise was hoping for: 12 of 28 pairs, seven of the eight
photographs in the model, and **two more depth maps actually computed**. They
fused to 324 points fewer than the baseline. `IMG_6401` and `IMG_6407` arrive
with **12 and 13 observations** — attachment, not coverage — and fusion wants a
depth value several views agree on, so a photograph that overlaps nothing
contributes a depth map nothing corroborates. Registering a camera and covering
a surface are different achievements, and only the second one makes a point
cloud.

### The number that would have fooled us

LoFTR at 1280 has COLMAP verify **three of the eleven wide pairs**. Read the
headline table alone and the dead band came alive. Scored against MultiSpy's
poses those three are wrong by **350, 582 and 1322 px** median Sampson — not
misplaced correspondences, different parts of the cell. Six pairs in that run
pass COLMAP's test and fail the poses'. The same trap, smaller, is in every run
including the baseline: SIFT 3, LoFTR 640 3, LoFTR 1024 4, LoFTR 1280 6,
ALIKED+LightGlue 2.

This is `harvest.py`'s lesson somewhere new. COLMAP calls a pair verified by
fitting an essential matrix to the matches and counting what agrees, which is
in-sample — hand RANSAC forty wrong correspondences on a wide pair and some
model will usually collect eighteen of them. A solved project can settle it,
because the poses imply an essential matrix with nothing fitted to the matches
at all.

So **`tools/pairscore.py`** is kept, and it is the only thing kept from this
investigation. It prints both tables over the same matches at COLMAP's own
thresholds and names the pairs that passed one and failed the other. It takes
an export directory and any COLMAP database, so it will score a matcher nobody
has thought of yet, and it answers the question that actually governs a dense
run — *which pairs share a view* — before anything is reconstructed.

LightGlue earns credit for the one thing neither other matcher does: on six of
the eleven wide pairs it returns **no candidates at all**. That is evidence
from the imagery and owes nothing to the poses, which matters, because the
second table's reference *is* the poses and those are corroborated mostly by
the narrow pairs. Read the two together, on a matcher that declines rather than
guesses. A detector-free matcher never declines — LoFTR returned 20 to 55
matches on every wide pair and all of them were wrong.

### Also found — since fixed, at the top of this file

`export.py` assigned `IMAGE_ID` in filename order on the premise that COLMAP's
extractor walks the directory in that order. **It does not on COLMAP 4.1**: the
extractor inserted `IMG_6404` as id 4 and `IMG_6403` as id 5 on this set, so
`point_triangulator` dies with the same
`Check failed: existing_frame.DataIds() == frame.DataIds()` the export fix
below was written to cure. Every measurement here matches images by name and so
is unaffected, and `pairscore.py` does the same deliberately; the triangulation
runs here needed the model renumbered from the database by a scratch script
first. The fix was to stop guessing the order — read the `images` table after
extraction and write the model to match — and `multispy export --colmap-db`
now does exactly that.

### What this does not show

Only `rearmod3`, only eight photographs. It says nothing about whether a
learned matcher helps a capture shot as a sweep, and the sparse improvement it
does measure is large enough to be worth trying on one. The claim here is the
narrow one that was asked: no matcher recovered a single pair the capture did
not already contain, and the dense verdict did not move.

---

## Done: a Blend button that hands over a file, not a script (Nov 2026)

`export_blender` wrote a script, which is the right primitive and the wrong
deliverable. Using it meant finding Blender's Scripting tab, opening the file
and pressing Run; and what came out still referenced photographs and CAD
elsewhere on disk, so moving any of it left the scene hollow. Three steps and a
trap between a finished solve and seeing it.

`blendfile.build_blend` runs the same script in the Blender already on the
machine, with a new `--save` the template understands: the scene starts empty
rather than on Blender's startup cube, and the photographs are packed into the
file at the end. One file that opens anywhere.

Measured on `projects/rearmod3`: **3.2 s**, 32.3 MB, 8 cameras, 28 bodies at
124,479 vertices, 8 photographs, **8 of 8 packed**. Built by 4.3.2 and reopened
in 5.1.1, where the tie points project through `MS_IMG_6405` onto the marks
they were clicked at to a median 2.8 px -- the solve's own residual, so the
round trip through Blender adds nothing.

Three things worth keeping:

- **Which Blender, decided by asking.** `--blender`, then `$MULTISPY_BLENDER`,
  then `PATH`, then the newest installed -- and "newest" is settled by running
  each candidate with `--version`, not by reading its directory name. The
  machine this was written on has a `Blender 43` that reports 4.3.2 sitting
  beside `Blender 4.2` and `Blender 5.1`; sorted as text, 43 wins and the tool
  picks the second-oldest Blender on the box. There is a test for exactly that
  arrangement.
- **"The file is there" proves nothing**, because it was there before. The
  build records the modification time first and treats an unchanged file as a
  failure, with the tail of Blender's own output attached. Without that, a
  Blender that refused to start looked like a successful re-export of
  yesterday's scene.
- **Not in `EXPORT_FORMATS`.** Export must not start a second program; this is
  asked for by name, by the button or by `multispy blend`. The browser build
  has no Blender to run, so the button is hidden there rather than present and
  always refusing.

Caught by `tests/test_build.py` while building this, which is what it is for:
`webapp.py` reaches `blendfile` by a relative import, so the module had to be
shipped into the browser bundle even though a tab can never use it. The
alternative was teaching that test which reachable imports do not count, and a
few unused kilobytes are cheaper than a weaker guard. The worker only *writes*
the shipped files to the Pyodide filesystem -- importing happens through
`webapp`, and this one is imported inside the function that needs it -- so it
costs nothing at boot. Verified by booting the bundle.

Also fixed in passing, because this feature needed it: **the elapsed badge
never appeared in the server build at all.** It was written for the static
build, where Python runs in the tab, and `call()` took the fetch path without
counting anything -- so a server-side solve was the same silent window the
worker was built to cure. Now counted in both. Calls finishing inside a second
never paint, so nothing flickers.

Not done: no progress *within* the Blender run. It is one subprocess and the
badge counts seconds at it. Fine at 3 s; a hundred-body project may want
Blender's own stdout relayed.

---

## Done: the exports describe the photographs that are actually there (Nov 2026)

Found while measuring whether a solve could feed a dense reconstruction, which
is the one thing the COLMAP export exists for and the one thing nobody had
tried. It could not, and the reason was not the solve.

Every mark in a project is in the frame the browser draws — the frame *after*
EXIF orientation is applied, which `webapp.image_dimensions` is explicit about.
Nothing downstream applies the tag. `projects/rearmod3`'s eight photographs are
stored 4032x3024 with orientation 6 while the model says 3024x4032, so handed
the export as it stood, `colmap point_triangulator` stopped at

    Check failed: existing_camera.width == camera.width (3024 vs. 4032)

That it stopped is luck: COLMAP 3.13 checks because the extractor was told
`--single_camera`. Without that check the poses and the imagery disagree by a
quarter turn and the output is nonsense that looks like a bad solve.
`ns-process-data` drives COLMAP and inherits it. So does Blender, checked
directly: `bpy.data.images.load` on `IMG_6399.JPG` reports 4032x3024 in Blender
5.1.1 against 3024x4032 for the upright copy, which means the background images
this tool wires to each camera — the thing offered as the quick way to see a
bad alignment — were a quarter turn out themselves, and had been all along.

Writing pixels rather than numbers is still the right call for a reason that
does not depend on any of those three: a file carrying no orientation tag reads
the same to a reader that honours one and a reader that does not, so the copies
cannot be wrong for a consumer nobody has tested.

The Blender end was then checked the whole way rather than at the file: the
exported script imports 8 cameras, 33 tie points and 28 bodies into 5.1.1, and
`world_to_camera_view` puts those tie points on the marks they were clicked at
to a median 2.8 px, max 7.3, on a 3024x4032 frame. That is the solve's own
residual on this image and nothing on top of it, which is the claim worth
making about an exporter -- it adds no error of its own.

`export.upright_images` writes the pixels rather than adjusting the numbers,
and the reason is worth keeping: rotating the exported geometry into the stored
frame costs nothing and would be correct for exactly those readers that ignore
the tag — and wrong for any that honour it, which is not a list we can
enumerate. A file carrying no orientation tag reads the same either way.
Originals are copied byte-for-byte, only rotated ones are re-encoded (keeping
their EXIF, since COLMAP's extractor seeds intrinsics from the focal length
there), and a repeat export re-encodes nothing.

A second defect sat behind the first and only surfaced once the images were
right: `IMAGE_ID` was assigned in the order photographs were added, and COLMAP
numbers them in the order its extractor walks the directory. The model was
rejected with `Check failed: existing_frame.DataIds() == frame.DataIds()`,
which names the rig bookkeeping and not the ids that caused it. Export order is
now the filename order.

| | before | after |
|---|---|---|
| `point_triangulator` on the export, untouched | **refused** | 8 images, 61 points, 166 obs |
| mean reprojection error | — | **1.63 px** |
| manual steps between Export and COLMAP | 2 (rotate, renumber) | **0** |

Only projects that need it pay for it: with every photograph already upright,
nothing is written and the export is byte-for-byte what it was. On rearmod3 it
is 27.5 MB of copies, which is now named in the status line rather than
appearing unexplained.

Worth saying plainly about what this does *not* buy. The same investigation
measured that these eight photographs cannot support a dense reconstruction at
all — 50.5 deg median angle between viewing directions, 13 of 28 pairs
verified, 90 points from 200k SIFT features with the poses handed over free.
This fix makes the handoff correct; it does not make the data sufficient. What
it did produce is the first check of the cameras against evidence nobody
clicked: 1477 verified SIFT matches sit at a **median 2.46 px** Sampson error
against these poses, which agrees with the muted-body RMS of 2.99 px below and
says the 9.37 px headline is body-fitting error, not camera error.

---

## Done: surfaces drawn on the photographs (Nov 2026)

A reference plane could only be typed: a name, an axis, an offset. That is the
right way to enter a datum off a print and no way at all to capture a fence, a
floor or the face of a tooling plate, which is most of what is actually in a
cell. Those have no stated direction and no stated distance — they are simply
*there*, in the photographs.

So a plane may now name four `corners` instead. They are ordinary tie points,
which is the whole design and the reason this added so little: marked with the
tool that already exists, triangulated by the solve that already runs. Click
four corners in order round the surface, then the same four in a second photo;
`solve.fit_drawn_planes` takes the least-squares plane through where they land
and projects the corners onto it, so the exported `FACE_SURFACE` is exactly
planar and bounded by the real corners rather than squared off around their
middle.

Two things worth keeping:

- **The corners are not evidence.** They are excluded from the bundle and
  triangulated against the finished cameras afterwards, exactly as a muted
  body's anchors are. Measured while building this: letting four
  roughly-clicked corners into the solve took a 7-photo project from
  **1.41 px to 82 px**. With the exclusion the RMS is unchanged to 1e-6, and
  the surface still fits. `ReferencePlane` always said a plane is what the
  measurement is reported *against*; this keeps that true for a kind of plane
  that is measured.
- **`flatness` is the plane's error bar.** Four marked corners are never
  coplanar, and projecting them makes the face exact — so the gap is reported
  instead of being quietly absorbed into the geometry. A carelessly marked
  fence says so in millimetres rather than arriving as a plausible plane.

Not done: no way to say "this corner is also that corner of the next panel", so
two abutting surfaces are marked independently and need not meet. Fine for a
fence and wrong for a tooling plate with pockets in it.

Rejected: fitting the plane from four corners in **one** photograph, via the
quad's two vanishing points. It is four clicks instead of eight and very much
in this tool's spirit, but it is only correct if the quad is a parallelogram in
space, and nothing about a fence promises that. Worth revisiting as an
*option*, never as the only way.

---

## Done: the panels stopped burying their own numbers (Nov 2026)

Twelve panels came to about 9,000 characters, 78% of it prose. The useful half
of that prose is the first sentence — nearly every message here is one
measurement followed by what it means — so the tail now folds, per message or
all at once, and nothing was rewritten to achieve it: the split is made at
render time out of the prose already there, by `explanationStart`, over the
live DOM with a `Range` so the dozen messages carrying markup survive it.

Measured on the 7-photo project: panel prose 3,287 → 1,419 characters, the hint
bar over the photograph 337 → 99, and **fold all** takes the whole panel to 202
— just the headings.

The thing worth recording is what it does *not* touch. On `rearmod3` the same
clamp only folds 29% of the prose, because that panel's length is not
explanation at all: it is twenty-eight bodies each stating four perfectly short
facts. Repetition and prose are different problems. **Fold all** is the answer
to the first one and the sentence clamp is the answer to the second; neither
substitutes for the other, which is why both are in that bar.

---

## Done: the browser build no longer freezes while it works (Nov 2026)

Reported as "the browser is frozen trying to solve without completing", on the
Pages build. It was not the solver: the project in question solves in 2.2s
natively and 3.6s in the tab, to the same RMS. It was that Pyodide ran *in the
page*, so every call into Python was a synchronous call on the main thread —
nothing could repaint for its duration, `setStatus('solving…')` never reached
the screen, and the window sat there looking hung. Measured on `rearmod3`:

| | before | after |
|---|---|---|
| solve | 8911 ms | 8881 ms |
| longest frame gap | **8977 ms** | **148 ms** |
| frames painted during the solve | **0** | **263** |
| reprojection RMS | 9.37 px | 9.37 px |

Same numbers, live window. Python now runs in `web/worker.js`; `backend.js`
keeps its public API and talks to it over postMessage, counts what is in flight,
and the status bar shows what is running and for how long.

Worth saying in that order: this made nothing faster. It made a slow thing
*legible*, which is the whole complaint — the user could not tell "working"
from "hung", and closed the tab, which on a page whose only persistence is a
download costs every mark since the last Save.

The elapsed counter also named a second offender nobody had measured: reading a
43 MB STL is **20+ seconds** in the browser, and it used to be 20 seconds of
dead window with no label on it. Still 20 seconds; now it says so.

Two guards in `tests/test_build.py`, since none of this is visible from a
terminal: `backend.js` may not touch `pyodide.*` or `loadPyodide`, and the
Python bridge may not be written in both files. CI also checks the bundle ships
a *versioned* `worker.js` — it is the one script `index.html` does not name, so
the content-hash rewrite cannot reach it, and a cached old worker beside a fresh
`backend.js` is a page that loads and then does nothing when Solve is pressed.

Not done: cancelling. Pyodide cannot be interrupted without a SharedArrayBuffer,
which needs COOP/COEP headers GitHub Pages does not send, so the only cancel
available is terminating the worker — which throws away the photos it holds.
Worth doing properly if anyone asks for it; the elapsed counter was the part
that mattered.

---

## Done: anchors can sit out the solve (Nov 2026)

`ObjectInstance.anchors_off` per body, `Settings.anchors_in_solve` for the lot.
A muted body keeps every click, stays in the scene, and is still placed — by
`solve.place_muted_instances`, which triangulates its anchors against the
*finished* cameras and rigid-fits the model, so it cannot move them. The CAD
guard's "worsens" verdict now leads somewhere that survives the next Solve;
`restore_without_bodies` was only ever a snapshot.

Measured on RearMod3, muting every body:

| | voting | muted |
|---|---|---|
| reprojection RMS | 9.37 px | **2.99 px** |
| culled marks | 15 | **2** |
| bodies placed | 26 of 28 | 26 of 28 |
| shape error, median¹ | 3.49 mm | 3.74 mm |

¹ one yardstick for both arms: every body's anchors triangulated against *that
arm's* final cameras, model rigid-fitted. Muting was better on 9 bodies of 23.

So it buys back the camera solution and **not** accuracy — the same ceiling the
table below reports, from the other side. Worth saying in that order whenever
this feature is described, because "9.37 → 2.99 px" on its own reads as an
accuracy claim and is not one.

Open question it raises, not scheduled: a muted body's `shape_rms` is measured
post-bundle while a voting body's is measured in `fit_instances`, pre-bundle —
see item 4 below, which this makes more worth doing, since the two columns now
mean different things in the same table.

---

## From the RearMod3 investigation (Nov 2026)

Dataset: `projects/rearmod3/` — 8 photos, 33 tie points, 28 bodies across 7 CAD
templates, 582 anchor marks, 3 nominal'd points. Launch it with the `rearmod3`
entry in `.claude/launch.json` (port 8806).

### Measured ceiling on all of this — read first

Six very different weighting policies were run against a held-out yardstick (the
17 bodies the solve does *not* flag as suspect, so a policy is not scored on the
data it reacted to). Their shape error against CAD:

| policy | culled | RMS | good bodies, shape |
|---|---|---|---|
| as shipped now | 15 | 9.37 px | **2.71 mm** |
| reweight off | 115 | 4.22 px | 2.79 mm |
| reweight off, no cull ceiling | 115 | 4.22 px | 2.79 mm |
| reweight on, no cull ceiling | 13 | 11.28 px | 2.71 mm |
| tighter cull (3 sigma) | 60 | 7.35 px | 2.75 mm |
| reweight off + 3 sigma | 151 | 3.82 px | 2.76 mm |

**0.08 mm of spread.** The remaining error on this dataset is in the marks, not
in the processing, and no weighting change reaches it. Nothing below will move
that number on RearMod3. They are all for finding out *sooner*, and for telling
the user what different data would be worth.

Note also what that table does to RMS as a yardstick: it swings 3.82 → 11.28 px
while the geometry moves 0.08 mm. RMS mostly measures how much data was
down-weighted or deleted. Prefer the held-out shape figure when judging a change.

### 1. Per-body pose uncertainty, decomposed

`uncertainty.py` already Monte Carlos the part origin against click noise, with
the right lever-arm reasoning in its docstring. Point the same machinery at every
placed body and report three numbers instead of one:

- **centroid sigma** — how well the average of its anchors is known
- **orientation sigma**
- **origin sigma** — the pose actually reported, which is the other two combined
  through the lever arm

Measured on RearMod3 (2 px click noise, 300 trials): centroid 0.64 mm median,
origin 2.45 mm median. For bodies whose model origin lies inside their anchor
cloud those two agree (0.62 → 0.67 mm); for the sensor templates, whose origin is
463–685 mm outside it, they do not (0.64 → 3.32 mm).

Replaces the two columns the UI shows today, which are worse than they look —
see item 4.

### 2. A design-time lever warning

`origin_sigma(anchors)` is closed form and needs no photographs: build the
rigid-fit Jacobian from the anchor coordinates, invert, read the translation
block. At template-definition time this can say

> 917-4007's origin is 463 mm outside its anchor cloud, whose thinnest axis is
> 70 mm. Expect roughly 3x the per-anchor noise at the reported pose.

Costs nothing and is knowable before anyone drives to the cell.

### 3. "Where should the next anchor go"

Same covariance, scanned over candidate positions. On RearMod3 it already gives
a clear answer for 917-4007 (predicted origin sigma 5.54 mm as marked):

| change | predicted | better by |
|---|---|---|
| one more anchor on the existing cloud | 5.52 mm | **0%** |
| one anchor at the best free position | 3.31 mm | 40% |
| one anchor at the model origin | 2.30 mm | 58% |
| seen in twice as many photos | 3.92 mm | 29% |

This is the feature that answers "we have lots of data, why is the fit not
better": it says when more of the same is worth nothing, and what is worth
something instead.

### 4. Recompute `shape_rms` and `implied_scale` after the bundle

Both are set in `fit_instances`, which runs *before* `bundle_adjust`, and nothing
recomputes them — so the per-object "shape" and "size" columns describe the
linear initialisation, not the answer. They came out byte-identical across seven
solver variants, which is how the bug shows itself.

Post-bundle they are materially different: shape median 4.29 → 3.48 mm, scale sd
5.26% → 3.40%, and individual bodies move a long way (obj0 3.44 → 0.35 mm,
obj8 27.5 → 4.8 mm, obj18 11.05 → 1.27 mm).

Smallest item here and the most misleading one while it stands.

### 5. A weight-share line in the report

One line: `tie points 84%, anchors 3%, lines 13%`. On the original code 582
anchor marks — 84% of every click in the project — carried 3.3% of the
photogrammetric weight, and nothing said so. A single line would have made that
visible immediately instead of after a day of digging.

### 6. Guard the invariant that failed

`_transform_scene` carried a comment asserting that non-manual poses are
"recomputed from `positions` after this runs". True at the two call sites inside
`fit_instances`, false at the one in `align_to_nominals`. Nothing checked it, and
the anchors started the bundle a median of 5.4 million pixels out.

Add the assertion the comment implies: after `align_to_nominals`, no anchor
should reproject further than some generous bound. Cheap, and it is the check
whose absence cost the most here.

### 7. Possibly: per-axis anchor sigma in the template frame

Measured anchor scatter on RearMod3 is anisotropic — 14.0 / 8.7 / 6.0 mm in X/Y/Z
of the template frame, a factor of 2.3 between axes — so one isotropic number is
not what the data says.

**Untested.** Uniform sigma changes were measured and do *not* help (shape 3.51 →
3.72 mm going from 2 px to 8 px), but a per-axis weighting is a different thing
and has not been tried. Treat as a question, not a plan.

---

## Known, not scheduled

- The three nominal'd points on RearMod3 form a sliver: 2122 mm long, 198 mm tall,
  against a 4148 mm scene, so 1 mm of error at p18 is ~21 mm across the scene.
  Each nominal is a 0.1 mm axis lock exempt from the robust kernel. A datum that
  strong deserves more than three points and more than two rays each.
- `p1` appears to be marked on different physical features in different photos
  (cone-pin tip in img1/img4, a post edge in img7). Worth re-clicking before
  trusting the frame.
