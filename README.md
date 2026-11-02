# MultiSpy

fSpy-style vanishing point calibration across a **whole photo set**, with
hand-marked corresponding features tying the images together, resolved in one
joint bundle adjustment.

Built for the case where there is real overlap between photos but not enough for
COLMAP or GLOMAP: wide baselines, few images, unknown camera, scenes with
man-made structure.

Objects whose shape you already know can be marked against their CAD model
instead of as free points — which makes the reconstruction metric without a tape
measure, and turns the layout of a camera array into something you can solve by
walking around it with a camera.

**New here?** [TUTORIAL.md](TUTORIAL.md) walks through both jobs start to finish
on scenes with known ground truth, so you can check your own work. This file is
the reference.

---

## Why this works where SfM does not

Two properties do the heavy lifting.

**Vanishing points give you global rotations for free.** Three orthogonal
vanishing points in one image hand you that camera's orientation in world axes
without reference to any other image. No match graph, no chaining, no drift
accumulation. That is normally the hardest part of structure-from-motion, and
here it falls out of control lines you draw by hand.

**With rotations known, the rest is linear.** Solving for camera translations
and 3D point positions with the rotations fixed is a linear least-squares
problem — no initial pair to choose, no incremental mapper to wander off, no
local minima. It needs roughly **two shared tie points per image pair**, against
the dozens of well-distributed matches an SfM pipeline wants.

And the thing that makes it beat fSpy on its own ground: **a single image's
focal length is determined by about four clicks with zero redundancy**, which is
why fSpy feels so twitchy — a distant vanishing point swings hundreds of pixels
for a two-pixel slip on a control line. Share one focal length across N images
and tie them together with corresponding points, and that estimate becomes
massively over-constrained.

Measured on the synthetic scene in `tests/`, with 1.0 px of click noise on tie
points and 1.5 px on control-line endpoints:

| | focal length error |
|---|---|
| Independent single-image solves (what fSpy gives you) | median **2.5%**, worst **81%** |
| Joint solve, shared focal + tie points | **0.05%** |

(Those single-image figures are already the *improved* ones. Each image's focal
estimate is an inverse-variance weighted blend of its axis pairs rather than a
plain median, which cut the median single-image error from 15.5% to 2.5% —
the weights come from the same error propagation that drives the Focal check
panel. The joint solve still beats it by 50x on the median and 1600x on the
worst case, which is the whole argument for marking a set rather than a photo.)

Rotations land within 0.2°, camera positions within 0.3% of the scene diagonal,
and reprojection RMS settles below the injected click noise.

---

## Install

```bat
cd "C:\Animation Studio\MultiSpy"
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy pillow
```

That is the whole dependency list. The web layer is stdlib only.

## Quick start

```bat
.venv\Scripts\python.exe -m multispy
```

No arguments needed. It reopens your most recent project, or starts an empty one
if there are none, and opens the UI. Then **drag photos straight onto the
canvas**.

---

## Or run it in a browser, with nothing installed

**[spencerdav.github.io/multispy](https://spencerdav.github.io/multispy/)**

The same solver, in a tab. Drop in photos and a CAD model, mark them, press
Solve, download the exports. There is no install and no sign-up.

It is not a demo or a cut-down version. The page loads CPython — numpy, scipy
and Pillow included — through [Pyodide](https://pyodide.org) and runs the very
same `multispy` package in-process, so it produces identical numbers. On the
synthetic rig scene, the browser and the command line agree to the decimal:
focal 3198.8 px, reprojection RMS 1.209 px, body shape error 0.55–0.72 mm,
baselines 225.2 / 450.0 / 675.0 / 900.3 mm. The solve itself takes about a
quarter of a second.

**Nothing is uploaded.** There is no server to upload to — the whole thing runs
in the tab, so photos of parts and tooling never leave the machine. That is the
main reason to hand someone this link rather than a hosted service.

Three things differ, all of them consequences of a browser having no disk:

- **First load is about 20 MB** and takes a few seconds, then it is cached. It
  waits behind a button rather than downloading on arrival.
- **Saving is a download.** The project comes back as a zip with its photos
  inside, because the marks alone would reopen pointing at files that are not
  there. Reopen it by dropping that zip back in.
- **Close the tab and it is gone.** Download the project to keep it.

Memory is the real limit: a 4000 × 3000 photo is about 36 MB decoded, so thirty
is comfortable and two hundred is not. For large sets, run it locally.

Build it yourself with:

```bat
.venv\Scripts\python.exe tools\build_static.py --serve 8800
```

which assembles `dist/` and serves it, so the static path can be exercised
without deploying anything.

---

## Projects and photos

Everything is managed in the app; no paths on the command line.

**Adding photos.** Drag them onto the canvas, or use **+ add photos** in the
strip on the left. They are copied into the project's own `photos/` folder, so
the project folder stays self-contained and portable. **link a folder…**
instead references photos where they already sit on disk, without copying —
better for a 200-frame set you do not want duplicated.

**Removing photos.** Hover a thumbnail and click its **×**, or use **remove
photo** in the This image panel. You are asked twice, on purpose: first to
confirm the removal (it warns you how many marks will be lost), then separately
whether to delete the file from disk. Cancelling the second keeps the file and
only unlinks it from the set. Photos that were *linked* rather than uploaded are
never deleted from disk — they are yours, sitting somewhere else on the machine,
and the app only ever forgets about them.

**Projects.** The name in the toolbar opens a switcher: every project under
`projects/`, most recent first, plus a field to create a new one. Each project
is a folder holding `project.multispy` and its `photos/`. Put them elsewhere
with `--projects "D:\somewhere"`.

Removing a photo can orphan tie points that were only visible in it. They are
not silently discarded — the points panel marks them **needs 2**, and the solve
report names them.

## Try it on the synthetic scene

```bat
.venv\Scripts\python.exe demo\make_demo.py
.venv\Scripts\python.exe -m multispy demo\project.multispy
```

This renders six synthetic photos of a 10 x 6 x 4 box, opens the UI with
everything already marked, and gives you a known-good solve to press **Solve**
on. Add `--blank` to `make_demo.py` for the same photos with no marks, which is
the one to practise the workflow on — the ground truth is known, so you can tell
how well you did (true focal 3200 px).

---

## The workflow

### 1. Control lines (Lines mode, keys 1 / 2 / 3)

Drag along edges in the photo that are **parallel in 3D**. Keys `1` `2` `3`
choose which world axis you are tracing: X, Y, Z.

- **Two per axis minimum, three is better.** Two lines fix a vanishing point
  exactly, which means nothing is left over to detect a mistake. Three gives the
  solver a disagreement to report.
- **Two axes minimum, three is better.** More axes means more focal estimates
  to cross-check.
- **Direction matters.** Draw from the lower coordinate toward the higher one,
  i.e. along the **positive** axis. The arrowhead shows which way the line
  points. This is what resolves the sign ambiguity — the same vanishing point
  serves both `+d` and `−d`, so only the direction you drew can tell them apart.
- **Longer is better and is weighted accordingly.** A 2 px slip on a 100 px
  segment says far less about a direction than the same slip on a 1000 px one,
  and the solver weights by segment length to match.
- Keep **VP rays** on. Each control line is extended to its vanishing point.
  If one ray does not converge with its neighbours, that line is on an edge that
  is not actually parallel to the others — the single most common mistake.

#### Watch the Focal check panel

As soon as you have two lines on each of two axes — four lines, long before a
full markup — the **Focal check** panel tells you whether the focal length is
determinable at all, and to what precision. It needs no solve.

```
FOCAL CHECK                              POOR
1884 px  ±57%
best pair XxZ, limited by Z

axis  lines  length   VP out   lever
X       3    1140px    15306     13x
Z       3     118px    13988    119x

The Z vanishing point is 13,988 px away while the Z control lines
are only 118 px long, a 119x lever — so one pixel of marking error
moves it about 119 px. Longer Z lines, further apart, are what help.
```

**Lever** is the number that matters: vanishing-point distance divided by
control-line length. It is the factor by which your marking error is magnified.
At 2x you can afford to be sloppy; at 119x you cannot mark accurately enough
with any amount of care, and no effort spent on tie points will rescue it.

The verdicts are **good** (±<2%), **weak** (±2–8%) and **poor** (±>8%), from a
first-order propagation of your assumed marking accuracy through the vanishing
points into the focal length. It is calibrated, not decorative — on the
synthetic scene it predicts ±1.0% and the solve lands within 0.05%; on a real
drone set it predicted ±11% and the solve came out 11.4% off.

If it says **poor**, you have three options: draw longer control lines on the
limiting axis, find a third axis with better convergence, or accept that this
viewpoint cannot determine focal length and lock the focal to a value you trust
from elsewhere.

#### Locking the focal length

The **focal length** box in the This image panel sets it for the whole camera
group, three ways:

- **From EXIF** — the row underneath reads the photos and offers a **use**
  button. Confidence is reported honestly, and there are two routes:
  *focal plane resolution* (`FocalLength` in mm plus the sensor-size tags) is
  exact and marked **good**; *35mm equivalent* is marked **approximate**,
  because cameras round it to a whole millimetre. On the demo set that rounding
  alone costs 0.7% — 28.8 mm stored as 29, read back as 3222 px instead of 3200.
- **Typed** — a number plus a unit, either raw **px** or **mm eq** (35mm
  equivalent, converted with the image width).
- **From a solve** — solve once with it free, then tick **lock focal length**
  to pin what came out, before refining anything else.

**Use EXIF as a fallback, not a default.** On a photo set where the Focal check
says *good*, the solve beats EXIF by a wide margin — on the demo it lands at
3201.6 px against a true 3200 (0.05%), where EXIF gives 3222 (0.7%), 14x worse.
EXIF earns its place exactly when the Focal check says *poor*.

Two readings are refused rather than offered:

- **Cropped images.** If the aspect ratio disagrees with the EXIF dimensions,
  the pixels no longer span the whole sensor and the conversion is void.
  Resizing is fine and handled; cropping is not.
- **A zoomed lens.** If the EXIF focal lengths across a group disagree by more
  than 2%, one shared focal is the wrong model and you are told to split the
  group rather than handed an average.

Check it without the UI:

```bat
.venv\Scripts\python.exe -m multispy exif demo\project.multispy
```

### 2. Tie points (Points mode)

Click the same physical feature in every photo it appears in.

- Press `N` for a new point, then click it in the first photo. With
  **auto-advance** on, you are moved straight to the next photo that does not
  have it yet. `Tab` does the same manually.
- **Two shared points per image pair** is the hard floor. Four to six
  well-spread points per pair is comfortable.
- Spread them in **depth**, not just across the frame. Points that are nearly
  coplanar as seen from the cameras make the linear stage ill-conditioned, and
  the solve panel will say so.
- Every image needs at least 2 tie points or its camera cannot be placed, and
  every point needs at least 2 images or it cannot be triangulated. Anything
  that fails these gets named in the report rather than silently dropped.

### 3. Constraints (optional)

- **Distance** between two points gives the solve real units. Without one,
  scale is arbitrary — everything is correct up to a similarity.
- **Axis lock** pins one world coordinate of a point. Three of them on one
  point sets the origin; the same value across several points states a plane
  ("these are all on the floor").

After a solve each constraint shows **what it got against what you asked for**,
in world units, with the tolerance it was given. Worth reading every time: a
constraint is the one figure you *assert* rather than click, so it is the only
place "did the reconstruction agree with me?" has a definite answer — and a datum
missed by a millimetre costs almost nothing in pixels, so it is invisible in
every other number here. A constraint naming a tie point that did not get solved
is reported as having had no effect, rather than silently having none.

Two things worth knowing before you lock several of them:

- **Locking several points imposes a scale**, whether or not you meant it to. Two
  points pinned 150 mm apart in X is a distance constraint wearing a different
  hat. Locking them to the *same* value states a plane instead, and says nothing
  about scale.
- **If a known object is also setting the scale**, the two are independent
  measurements of the same thing, and the solve reports how far apart they are.
  See below.

### 4. Solve, then read the residuals

The **Per-image residuals** panel sorts photos worst-first. Click a row to jump
there. That is the feedback loop fSpy cannot give you: with one image there is
nothing to compare against, but with a set, a bad control line shows up as one
photo whose residual stands out.

---

## Known objects, and camera arrays

If something in the scene has a shape you already know, say so and the solve gets
dramatically stronger. Give MultiSpy a CAD model, mark a few of its corners in
the photos, and it places that object as a rigid body.

**The arithmetic is the whole argument.** Twelve clicks on free features add
twelve unknown 3D positions — 36 unknowns the solve has to find. Twelve clicks on
a body whose geometry is known add **six**, because the only things not already
determined are where that body sits and which way it faces. Everything else was
settled by the measurement you already have.

And a known object carries its own size, so **one of them makes the whole
reconstruction metric** with no distance constraint and no tape measure.

### The case this was built for

Solving the layout of a **camera array**. Walk around the rig with a handheld
camera, mark each camera body against its CAD model, and the solve returns where
every camera in the array sits and which way it points.

```bat
.venv\Scripts\python.exe demo\make_rig_demo.py
.venv\Scripts\python.exe -m multispy demo_rig\project.multispy
```

Five identical 100 × 60 × 40 mm bodies on a 1.2 m frame, 225 mm apart, toed in
6° each, photographed six times. Add `--blank` to mark it yourself: the model,
its anchors, its optical frame and the five objects stay set up, and everything
you would have clicked is stripped. Note that control lines go too, and are still
required — known objects do not replace them, because the cameras' orientations
and the focal length still come from vanishing points. Tie points, on the other
hand, you can skip entirely once objects cover the scene.

Measured on that scene with 1 px of click noise:

| | |
|---|---|
| baseline error, 225–900 mm spacings | worst **1.4 mm** |
| body shape error against the model | **0.6 mm** on a 100 mm body |
| solved body size vs the model | within **0.3 %** |
| relative toe-in angle | within **1.2°** |

Note the gap between the position and angle figures, because it is a property of
the geometry and not a defect to be tuned away. A body's **orientation** is read
off a lever only as long as the body itself — 100 mm, seen from 1.3 m away — so
it inherits roughly *(distance ÷ body size)* times the error of a position.
Positions land near a millimetre; angles near a degree. Marking a **bigger**
object, or shooting **closer**, is what moves that number. More photos of the
same geometry does not.

### The workflow

**1. Add the model.** Drag an `.obj`, `.stl` or `.ply` onto the canvas, or use
**+ add model** in the Known objects panel. STEP, IGES and SLDPRT are parametric
formats needing a geometry kernel to read — convert once in Blender or FreeCAD
and the units come across.

**2. Check the units.** This is the one step that will quietly ruin everything if
you skip it. No mesh format records its units, so an STL exported in millimetres
and one in metres are byte-identical apart from the numbers. The panel shows what
the model's size works out to in metres and offers an opinion, but it never
guesses silently — get it wrong and the entire layout is scaled by 1000 while
every residual stays perfect.

**3. Pick anchors.** Press **D** for the 3D view, switch to **model**, and click
the points it offers. The only test that matters is whether you could find that
point again *in a photograph* — the middle of a flat face is unmarkable because
nothing in the picture locates it.

Three kinds are offered, drawn so you can tell them apart, because they are not
equally good and the difference is invisible in the geometry:

| | | |
|---|---|---|
| **ring** | the centre of a round or symmetric feature — a bored hole, a bolt head, a port | **the best kind.** Rotationally symmetric, so it reads identically from every angle, and you place it by eye at the centre of the visible ellipse. Not a vertex at all. |
| **cross** | where two straight edges would meet, if a fillet had not rounded them off | The sharp corner does not exist in the model, but it is well defined, and it is *steadier* than the rounded corner it replaces — whose apparent position slides with viewpoint and lighting. Mark it in the photo by extending the two visible edges by eye. |
| **dot** | a genuinely sharp vertex | Fine where they exist. Shell corners, square cut-outs. |

The panel reports the mix before you open the viewer, so a part with everything
filleted and no round features is obvious early.

**Do not mark the rim of a bolt head**, which is why those vertices are not
offered. Every one of them is a perfectly sharp corner and none is identifiable:
you cannot tell which of six hex corners you clicked, and in the next photo you
will click a different one. That mistake produces a plausible-looking solve whose
shape error never comes down. Its *centre*, by contrast, is one of the best
anchors on the part.

If the point you want is one you measured rather than one the model offers, **type
a position…** takes coordinates in the model's own frame. Worth having: a drawing
sometimes carries a datum the mesh does not, and a CAD export can round off
exactly the corner you were counting on.

**4. Add objects.** **+ object** per physical copy. Ten identical cameras are one
model and ten objects, which is the point: measuring the body once places all
ten, and every anchor marked in every photo refines the same shared geometry.

**5. Mark them.** Press **O**, pick a body, and click its anchors in the photos.
The next unmarked anchor on the same body follows automatically; `Tab` jumps to
the next photo. **Three anchors in two photos each** is the floor for a
placement — two leave the body free to spin about the line joining them, and the
panel says so rather than quietly returning one of the poses.

**6. Solve, then look at it.** Turn on **Models** and the model's wireframe is
drawn over the real object in the photo. That is the check no number gives you:
a residual table cannot tell you a body was placed 180° out, and this picture
can.

### The optical frame

Without one, the solve reports where a camera's *shell* is — which is not what
anyone wants to know. **edit optical** declares where the lens sits on the model
and which way it faces, and the layout then comes out as real camera extrinsics:
entrance pupil, optical axis, yaw and pitch.

Both are reported pairwise — baselines and relative angles — because that is how
a rig is specified, and because pairwise numbers do not depend on where the
arbitrary world origin ended up.

### Locking an object

Tick **lock** on one object and it stops being something to solve and becomes the
world frame: origin, orientation and scale all come from it. Re-solving after
adding photos then leaves the coordinates where they were, which matters once you
are iterating.

The catch is that the whole scene now hangs off that one body's orientation, and
orientation is the weak axis. On the demo, half a degree of frame rotation is
8 mm of error 900 mm away. Lock the best-determined body, and prefer a large one.

### What gets checked

Because the shape is known in advance, the solve can tell you things a free point
cloud cannot:

| Reading | Meaning |
|---|---|
| **shape error** | How far the marked anchors sit from the model's own geometry. A mis-clicked corner shows up here as a contradiction, instead of hiding inside an otherwise plausible reconstruction. |
| **solved size** | What this body came out as, relative to its model. Should be 1.000×. Anything else means an anchor is on the wrong feature, or this is not the object the model describes. |
| **scale** | Now reads `known object geometry (mm)` rather than `arbitrary`. |
| **stated vs object scale** | Only when you have *also* stated a real distance. Two independent measurements of the same thing, so their ratio is a genuine check rather than a restatement. 1.000× means they agree. |

### The units trap, and how it gets caught

A model declared in the wrong units is the one error that leaves every residual
looking perfect while the whole answer is out by a round factor. State **one**
real distance anywhere in the scene and the solve catches it:

```
the figures you stated and the known object geometry disagree about scale by a
factor of 10x (the stated figures want the scene smaller). The bundle adjustment
has split the difference, so neither is now exactly right. A near-round factor
almost always means the model's declared units are wrong.
```

That comparison is made at **initialisation**, before the refinement runs, which
is the only point it means anything. Afterwards the bundle adjustment has already
traded the two sources of scale against each other, so a disagreement shows up as
a little of each being wrong and nothing reads as the culprit.

One mis-marked body does **not** drag the scene's scale: the per-object sizes are
combined by median with outlier rejection, so a body that disagrees with the
others is left out of the scale estimate and named, rather than turning one local
mistake into a global one.

---

## Reading the diagnostics

| Reading | Meaning |
|---|---|
| **reprojection RMS** | How well tie points fit. Should land near your clicking accuracy — 1–2 px. Above ~5 px, something is mis-clicked. |
| **control-line RMS** | Perpendicular pixel slip of control lines. Similar scale. If this is large while reprojection RMS is small, a control line is on a non-parallel edge. |
| **focal certainty** | Predicted 1-sigma uncertainty on the focal length, combining every calibrated image. Available before the solve, from the Focal check panel. Above ±8% everything downstream — camera positions, point depths — inherits that error. |
| **init conditioning** | Eigenvalue ratio of the linear stage. Below 10 means some image pair hangs on too few tie points, or your points are too coplanar. |
| **focal spread warning** | Per-image focal estimates disagreeing wildly. Expected to some degree — that is the fragility this tool exists to average out — but above ~100% go and check the control lines. |
| **scale** | `arbitrary` until you add a distance constraint or place a known object. The solve panel names which one set it. |
| **constraint error** | Per constraint, what the solve delivered against what you asked for, in world units. The only reading here that compares against something you asserted rather than clicked — and a datum missed by a millimetre is invisible in every other number on this list. |

Turn on **Grid** after solving. It draws the world XY plane anchored to the
floor of the point cloud, plus an axis gizmo. If the grid lies flat along the
real ground in the photo and converges correctly, the solve is right. (Draw your
vertical edges as **Z** for this to land on the ground.)

**Residuals** draws observed → reprojected vectors at **×10**, since a good fit
is otherwise invisible.

---

## Exports

**Export** writes, next to the project:

| Output | Use |
|---|---|
| `export/sparse/` | COLMAP sparse model (text). Read by COLMAP, GLOMAP, Meshroom, `ns-process-data`. |
| `export/import_cameras.py` | Blender script. Builds the cameras with each photo wired up as that camera's background image, and imports every placed object's CAD model at its solved pose, with a camera object for each declared optical frame. Run via Scripting → Open → Run Script. |
| `export/transforms.json` | nerfstudio / instant-ngp. |
| `export/tiepoints.ply` | Tie points, camera centres and object anchors, each in its own colour, for a quick look in MeshLab or CloudCompare. |
| `export/rig.json` | Every placed object's pose and optical frame, plus the pairwise layout. Written only when something is placed. |
| `export/rig.csv` | The same layout as a flat table — position, optical axis, yaw and pitch per object — for a spreadsheet or a spec sheet. |

The COLMAP export is the useful one for this workspace: a hand-marked solve can
feed Meshroom's dense stage, or be handed to COLMAP's own bundle adjuster to
refine further.

Verified end to end — `colmap model_analyzer` on the demo export reports 6
registered images, 17 points, 70 observations, 0.487 px mean reprojection error.

---

## Keyboard

| Key | |
|---|---|
| `1` `2` `3` | X / Y / Z axis |
| `L` `P` `O` `S` | Lines / Points / Objects / Select mode |
| `N` | New tie point |
| `Tab` | Next photo missing the active point, or the active object's anchors |
| `←` `→` or `[` `]` | Previous / next photo |
| `D` | 3D view (orbit the scene, or a model to pick anchors off) |
| `F` | Fit view — the 3D view when it is open, otherwise the photo |
| `V` `R` `G` `M` | Toggle VP rays / residuals / grid / object models |
| `Enter` | Solve |
| `Ctrl+S` | Save |
| `?` | Open the tutorial (also the **Help** button) |
| `Delete` | Remove whatever is under the cursor |
| wheel | Zoom at cursor |
| space-drag, middle-drag, alt-drag | Pan |
| right-click | Delete a marker or line |

---

## Headless

```bat
.venv\Scripts\python.exe -m multispy solve demo\project.multispy
.venv\Scripts\python.exe -m multispy export demo\project.multispy out\
.venv\Scripts\python.exe -m multispy exif demo\project.multispy
.venv\Scripts\python.exe -m multispy rig demo_rig\project.multispy
```

`solve` prints the full per-image residual table, including each image's
predicted focal uncertainty, plus a per-object table and the layout. `exif`
prints what the photo files claim, per image and per camera group. `rig` prints
the solved object layout on its own, or the whole payload with `--json`.

## Tests

```bat
.venv\Scripts\python.exe tests\test_synthetic.py
.venv\Scripts\python.exe tests\test_rig.py
.venv\Scripts\python.exe tests\test_exif.py
```

`test_synthetic` builds a Manhattan-world scene with known ground truth,
generates only what a user could plausibly have clicked (control lines on
visible box edges, tie points on visible corners, with realistic noise), and
checks recovery of focal length, rotations, camera positions and metric scale.
It also checks that the focal-observability prediction brackets the error the
solver actually makes, in both directions — a diagnostic that is over-confident
is worse than none.

`test_rig` builds the camera-array scene — a frame with five toed-in camera
bodies — and checks the layout comes back: baselines, relative angles, metric
scale from object geometry alone, and the world frame a locked object defines.
It also checks the *reporting*: that swapping two anchors on one body surfaces as
a shape error on that body and does not contaminate the others, and that two
anchors are refused as a placement rather than silently resolved.

`test_exif` writes JPEGs carrying known EXIF tags and checks each extraction
route, including the cases that must be *refused*: a cropped frame and a group
whose lens was zoomed between shots.

All three are pytest-compatible and all three run standalone.

---

## Where this fits, and where it does not

Use MultiSpy when the scene has **man-made parallel structure** and too little
overlap for automatic SfM.

**Every image still needs control lines.** A known object fixes scale and places
itself, but the camera *rotations* still come from vanishing points, so each
photo needs two axes with two control lines each. Recovering a camera's rotation
from a placed object instead — ordinary pose estimation, once three anchors are
marked — would lift that requirement and is the obvious next thing to add; it is
not there yet. In practice the rig or frame an array is mounted on supplies the
parallel edges, which is why the demo scene has one.

It will not help with organic scenes — trees, terrain, rock — because there are
no parallel edges to trace. For those, the alternatives worth reaching for are
hloc with LightGlue or LoFTR (keeps your existing COLMAP pipeline, just replaces
the matcher), or a pose-free learned reconstructor like VGGT or MASt3R-SfM.

It also does not need to be all-or-nothing. A reasonable hybrid: solve the
cameras here, export to COLMAP format, then hand that to COLMAP's triangulator
and bundle adjuster to densify with automatic matches that now have a known pose
prior to work against.

## Layout

```
multispy/
  geometry.py   vanishing points, focal and rotation from control lines, the
                focal-observability error propagation, and the similarity fit
  mesh.py       OBJ / STL / PLY loading, welding, feature edges and corners
  exif.py       focal length from photo metadata, with confidence
  solve.py      calibration, linear initialisation, known-object placement,
                bundle adjustment
  project.py    data model and JSON persistence
  export.py     COLMAP, Blender, nerfstudio, PLY, rig layout
  webapp.py     every operation behind the UI, independent of transport --
                shared by the local server and the browser build
  server.py     local HTTP server: transport only, JSON in and out
  docpage.py    renders these docs as pages the app links to (/docs/...)
  __main__.py   CLI
web/
  app.js        marking UI (canvas, vanilla JS)
  viewer3d.js   3D view, a software renderer on a 2D canvas
  backend.js    sends the UI's requests to the local server, or to Pyodide
  config.js     which of the two this page is talking to
tools/
  build_static.py  assembles the browser build into dist/
tests/          synthetic ground-truth validation
demo/           synthetic scene generators (box scene, camera array)
projects/       your projects, one folder each (created on first run)
```

`webapp.py` is the reason the browser build is the same application rather than a
copy of it: both front ends call the same `Workspace`, and only the transport
differs.

A project folder holds `project.multispy`, its `photos/`, and `models/` for any
CAD models added to it, so it stays self-contained and portable.
