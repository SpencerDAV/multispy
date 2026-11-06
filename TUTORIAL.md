# MultiSpy — Tutorial

A start-to-finish walkthrough. The [README](README.md) is the reference: what
every number means and why the method works. This is the other thing — sit down
and do it, twice, on scenes where the right answer is known so you can tell
whether you got it.

- **Tutorial 1** solves a photo set: camera poses and focal length from scratch.
- **Tutorial 2** solves a camera array: where five camera bodies sit and which
  way they point, to the millimetre.

Both use generated scenes with known ground truth. Do them in order — the second
builds on the first.

---

## Before you start

### Launch it

```bat
cd "C:\Animation Studio\MultiSpy"
.venv\Scripts\python.exe -m multispy
```

No arguments. It opens your most recent project (or an empty one) and serves the
UI at `http://127.0.0.1:8777/`. Leave this terminal running; `Ctrl+C` stops it.

> **You can read this page inside the app.** The **Help** button in the toolbar
> (or the `?` key) opens it in a second tab, so you can keep it beside the
> window while you work through the steps. The **Reference** tab at the top of
> that page is the README.

If that fails, the install is one line:

```bat
.venv\Scripts\python.exe -m pip install numpy scipy pillow
```

### The window in sixty seconds

| Where | What |
|---|---|
| **Toolbar** | Project name (click to switch or create), mode buttons, axis buttons, overlay toggles, **3D**, **Save**, **Export**, **Solve**. |
| **Left strip** | Every photo. A coloured dot shows how that photo is doing; **×** removes it; **+ add photos** at the bottom. |
| **Centre** | The photo. Wheel zooms at the cursor, space-drag pans, **F** fits. |
| **Right panels** | Solve · This image · Focal check · Known objects · Tie points · Constraints · Per-image residuals. |
| **Status bar** | What just happened, and whether you have unsaved changes. |

### The one idea

You give MultiSpy two kinds of mark, and they do different jobs:

- **Control lines** say *which way things point*. Trace edges you know are
  parallel in 3D. These give each camera its orientation and the focal length,
  with no reference to any other photo.
- **Tie points** say *where things are*. Click the same physical feature in
  several photos. These tie the photos into one coordinate system.

Everything else is bookkeeping. If a solve goes wrong, it is nearly always one
of these two being wrong, and the panels tell you which.

---

## Tutorial 1 — Your first solve

### Set up

```bat
.venv\Scripts\python.exe demo\make_demo.py --blank --out demo_practice
.venv\Scripts\python.exe -m multispy demo_practice\project.multispy
```

Six photos of a box, no marks. The box is **10 × 6 × 4 units** and the true
focal length is **3200 px** — remember those two facts, they are how you grade
yourself at the end.

### Step 1 — Decide which axis is which

Look at the first photo and pick, once, for the whole project:

- **X** along the box's long horizontal edges
- **Y** along the short horizontal edges
- **Z** straight up

It does not matter which you choose, only that you stay consistent across every
photo. Write it down if you need to.

> **Make Z the vertical one.** The ground-plane grid overlay later assumes it,
> and so does most of the advice in these docs.

### Step 2 — Draw control lines

Press **L** for Lines mode, then **1** for the X axis. Drag along an edge of the
box that runs in the X direction. Then drag along a *different* edge that is
parallel to the first — a different face, or the top instead of the bottom.

Now **2** for Y, two more lines. Then **3** for Z, two more.

Four rules, in order of how often they are broken:

1. **Two lines per axis, minimum.** Two fix a vanishing point exactly, so
   nothing is left over to catch a mistake. Three is better.
2. **Different edges, not the same edge twice.** Two segments on one physical
   edge look like two lines but tell the solver one thing.
3. **Draw along the positive direction.** The arrowhead shows which way the line
   points; it must run the way the axis increases. Get this wrong and the solve
   comes out mirrored. **Click a drawn line to reverse it** — no need to delete
   and redraw. Watch for this especially when two axes in one photo are both
   backwards: that pair cancels into a legal rotation, so nothing warns you.
4. **Longer is much better.** A 2 px slip on a 1000 px line matters a tenth as
   much as on a 100 px one, and the solver weights it that way.

Keep **VP rays** on (the default). Each line is extended to where its family
converges. If one ray refuses to converge with its neighbours, that line is on an
edge that is not actually parallel — fix it now.

> **Checkpoint.** The **Focal check** panel should now read **GOOD**, with
> something like `3166 px ±1.0%`. If it says **weak** or **poor**, read the
> lever column: it is the vanishing-point distance divided by your line length,
> and it is the factor your clicking error gets multiplied by. Under 20× is
> comfortable. Over 60× means draw longer lines before going any further.

### Step 3 — Repeat on the other photos

Press **→** for the next photo and do the same. Do all six.

This is the step people skip, and it is where the accuracy comes from. One photo
gives a focal length pinned by four clicks with no redundancy. Six photos share
one focal length, and the estimate becomes massively over-constrained. The solve
report will tell you exactly how much it gained.

### Step 4 — Mark tie points

Press **P** for Points mode, then **N** for a new point. Click a distinctive
corner of the box. With auto-advance on, you are moved to the next photo that
does not have it — click the *same physical corner* there. Repeat round the set,
then **N** again for the next point.

- **Two shared points between any pair of photos is the hard floor.** Four to
  six per pair is comfortable. (For comparison, COLMAP wants dozens.)
- **Spread them in depth**, not just across the frame. Points that are nearly
  coplanar as seen from the cameras make the solve ill-conditioned, and it will
  say so.
- Pick corners, not the middles of flat faces. If you cannot say exactly which
  pixel it is, neither can the solve.

Aim for eight to twelve points. The Tie points panel shows how many photos each
appears in, and flags **needs 2** for any that cannot be triangulated.

### Step 5 — Solve

Press **Enter**.

> **Checkpoint — what good looks like.** On this scene, fully marked:
>
> | | |
> |---|---|
> | reprojection RMS | **under 2 px** (the demo's own marks give 0.77 px) |
> | focal | near **3200 px** |
> | cameras placed | **6** |
> | init conditioning | comfortably over 10 (the demo gives ~2580) |
> | scale | `arbitrary` — expected, we fix it in step 7 |

If RMS is high, open **Per-image residuals**. It sorts worst photo first; click
a row to jump there. One bad photo is a mis-drawn control line. One bad *point*
across many photos is a mis-clicked corner — the Tie points panel flags it.

### Step 6 — Check it with your eyes

Turn on **Grid** (**G**). A ground-plane grid is drawn using the solved camera.

This is the check no number gives you. If the grid lies flat along the box's base
and converges correctly into the distance, the solve is right. If it skews or
floats, something is wrong no matter how good the residuals look. A residual
table cannot tell you the whole frame is mirrored; this picture can.

**Residuals** (**R**) draws observed → reprojected error vectors at ×10, since a
good fit is otherwise invisible.

### Step 7 — Make it metric

Everything so far is correct up to scale. One real measurement fixes that.

In **Constraints**, pick two tie points whose real separation you know, type it,
and press **dist**. On this scene the box is 10 units along its long edge, so
mark two corners of that edge and enter `10`.

Re-solve. **scale** now reads `metric` instead of `arbitrary`, and every distance
in the project is in your units.

**Then check it was actually met.** Each constraint now shows what the solve
delivered against what you asked for — hover one for `asked 10.0000, got 9.9981`
and how that compares with its tolerance. This is worth looking at every time,
because a constraint is the one thing you *assert* rather than click, and a miss
of a millimetre costs almost nothing in pixels. It is invisible in the
reprojection RMS and visible here.

A constraint naming a tie point that did not get solved is marked **no effect**
rather than quietly doing nothing.

> You can also pin the origin: three **lock** constraints on one point (its X, Y
> and Z) put the world origin there.
>
> Mind that locking *several* points imposes a scale whether you meant it or
> not — two points pinned 150 mm apart in X is a distance constraint wearing a
> different hat.

### Step 8 — Export

Press **Export**. Next to the project you get:

| | |
|---|---|
| `export/sparse/` | COLMAP sparse model — feeds Meshroom, GLOMAP, `ns-process-data` |
| `export/import_cameras.py` | Blender script; run it via Scripting → Open → Run Script |
| `export/transforms.json` | nerfstudio / instant-ngp |
| `export/tiepoints.ply` | quick look in MeshLab or CloudCompare |

The Blender script wires each photo up as its camera's background image, so you
can eyeball the alignment immediately in 3D.

**You have now done the whole job.** Everything below is either a stronger way to
do it, or how to recover when it goes wrong.

---

## Tutorial 2 — Solving a camera array

If an object in the scene has a shape you already know, say so and the solve gets
dramatically stronger.

The arithmetic is the argument. Twelve clicks on free features add twelve unknown
3D positions — 36 unknowns. Twelve clicks on a body whose shape is known add
**six**: where it sits and which way it faces. Everything else was settled by the
measurement you already have. And because the object carries its own size, **one
known object makes the whole reconstruction metric** — no tape measure, no
distance constraint.

### Set up

```bat
.venv\Scripts\python.exe demo\make_rig_demo.py --blank --out demo_rig_practice
.venv\Scripts\python.exe -m multispy demo_rig_practice\project.multispy
```

Five identical **100 × 60 × 40 mm** camera bodies on a 1.2 m frame, **225 mm**
apart, each toed in **6°**, photographed six times.

`--blank` keeps the model, its eight anchors, its optical frame and the five
objects, and strips everything you would have clicked: the control lines, the tie
points and the anchor marks. So the setup is done and the marking is yours.

### Step 1 — Check the units

**Do this before anything else.** No mesh format records its units. An STL
exported in millimetres and one in metres are byte-identical apart from the
numbers.

The Known objects panel shows what the model's size works out to in metres and
offers an opinion. Confirm it.

> Get this wrong and the entire layout is scaled by 1000 **while every residual
> stays perfect**. There is no other symptom. This is the single most dangerous
> step in the tool.

(Adding your own model: drag an `.obj`, `.stl` or `.ply` onto the canvas, or use
**+ add model**. STEP, IGES and SLDPRT need a geometry kernel — convert once in
Blender or FreeCAD and the units come across.)

### Step 2 — Look at the anchors

Press **D** for the 3D view and switch to **model**. The anchor points are marked
on the mesh.

If you were setting up your own model you would pick these yourself, and the rule
is: **only pick features you can identify in a photograph.** The middle of a flat
face is unmarkable, because nothing in the picture locates it.

The viewer offers three kinds, drawn differently:

- a **ring** — the centre of a round or symmetric feature: a bored hole, a bolt
  head, a port. The best kind. It reads the same from every angle, and you place
  it in the photo at the centre of the visible ellipse.
- a **cross** — where two straight edges would meet if a fillet had not rounded
  them off. Mark it by extending the two visible edges by eye. Steadier than the
  rounded corner itself, whose apparent position moves as you walk around.
- a **dot** — a genuinely sharp vertex.

The demo part is a plain box, so you only see dots. A real machined part is
mostly rings and crosses, and the panel tells you the mix before you open the
viewer.

> **Never mark the rim of a bolt head** — which is why those vertices are not
> offered. Each one is perfectly sharp and none is identifiable: you cannot tell
> which of six hex corners you clicked, so in the next photo you will click a
> different one. The result is a plausible solve whose shape error will not come
> down however carefully you work. The bolt's *centre* is excellent.

If the point you want is measured rather than offered, **type a position…** takes
coordinates in the model's own frame.

Press **D** again to return to the photo.

### Step 3 — Draw control lines first

Known objects do **not** replace control lines. The solve still gets every
camera's orientation and the focal length from vanishing points, and without them
it stops at *"only 0 image(s) calibrated"* before it ever looks at your objects.

So do Tutorial 1's step 2 on each photo: two lines on each of two axes, drawn
along the positive direction. The rig frame and the bodies give you plenty of
parallel edges. Watch the Focal check panel reach **good** before moving on.

> **Tie points, on the other hand, you can skip.** Object anchors already say
> "this feature here is that feature there", so they do the tie points' job.
> Solving the demo rig with its tie points deleted still places all six cameras
> (RMS 1.25 px against 1.21 px with them). Add free tie points only for parts of
> the scene no object covers.

### Step 4 — Mark the anchors in the photos

Press **O** for Objects mode and pick the first body in the Known objects panel.

Click its anchors in the photo. The next unmarked anchor on the same body follows
automatically, so you work through them in order. **Tab** jumps to the next photo.

Then pick the second body and do the same, and so on through all five.

> **Three anchors in two photos each is the floor** for placing a body. Two
> anchors leave it free to spin about the line joining them, and the panel says
> so rather than quietly returning one of the possible poses.

Note what you are *not* doing: measuring the body five times. One model, five
objects. Measuring the shape once places all five, and every anchor you mark in
every photo refines the same shared geometry.

### Step 5 — Solve and look at it

Press **Enter**, then turn on **Models** (**M**).

Each model's wireframe is drawn over the real object in the photo. If a wireframe
sits on its object, that body is placed. If one is floating, inside-out or 180°
round, you can see it instantly — and no residual table would have told you.

> **Checkpoint.** The Known objects panel should show, per body:
>
> - **shape error** — how far your clicks sit from the model's own geometry. A
>   mis-clicked corner shows up here as a contradiction instead of hiding inside
>   a plausible-looking reconstruction.
> - **solved size** — should be **1.000×**. Anything else means an anchor is on
>   the wrong feature, or this is not the object the model describes.
> - **scale** now reads `known object geometry (mm)` rather than `arbitrary`.

One mis-marked body does not drag the scene's scale: per-object sizes are
combined by median with outlier rejection, so a body that disagrees is left out
and named, rather than turning a local mistake into a global one.

### Step 6 — Declare the optical frame

Without this, the solve reports where each camera's *shell* is, which is not what
anyone wants to know.

**set optical** (it reads **edit optical** once one exists — the demo model
already has one) declares where the lens sits on the model and which way it
faces. The layout then comes out as real camera extrinsics: entrance pupil,
optical axis, yaw and pitch.

### Step 7 — Read the layout

```bat
.venv\Scripts\python.exe -m multispy rig demo_rig_practice\project.multispy
```

```
5 object(s), extent 0.9003 m  (models in mm)
measured from the optical centre where the template declares one

object                    x          y          z   optical axis
cam0                0.77990    0.95038   -0.21576   +0.2038  +0.9765  -0.0702
cam1                1.00698    0.95051   -0.21569   +0.1071  +0.9918  -0.0696
...

pair                          distance     angle
cam0 - cam1                    0.22515    5.612d
cam1 - cam2                    0.22488    6.302d
cam0 - cam4                    0.90030   23.427d
```

Against ground truth of 225 mm and 6°: baselines land within **0.3 mm**, the full
900 mm span within **0.3 mm**, angles within about **1°**.

Results are reported **pairwise** — baselines and relative angles — because that
is how a rig is specified, and because pairwise numbers do not depend on where
the arbitrary world origin ended up.

`Export` also writes `rig.json` and `rig.csv`, the latter being a flat table per
object for a spreadsheet or a spec sheet.

### Step 8 — Understand the angle figure

Positions come out near a millimetre; angles near a degree. That gap is a
property of the geometry, not a defect to tune away.

A body's **orientation** is read off a lever only as long as the body itself —
100 mm, seen from 1.3 m away. So it inherits roughly *(distance ÷ body size)*
times the error of a position. 1.3 m ÷ 100 mm ≈ 13, and 1 mm × 13 ≈ 1°.

**What moves that number:** marking a bigger object, or shooting closer.
**What does not:** more photos of the same geometry. If you need tight angles,
change the shot, not the effort.

### Step 9 — Lock an object when you iterate

Tick **lock** on one object and it stops being something to solve and becomes the
world frame: origin, orientation and scale all come from it. Re-solving after
adding photos then leaves your coordinates where they were — which matters a lot
once you are working iteratively.

The catch: the whole scene now hangs off that one body's orientation, and
orientation is the weak axis. On this demo, half a degree of frame rotation is
8 mm of error 900 mm away. **Lock the best-determined body, and prefer a large
one.**

---

## Shooting your own photos

What the method needs, in rough order of importance:

- **Man-made structure with visible parallel edges.** Buildings, rooms,
  furniture, equipment. This is non-negotiable — the whole rotation stage comes
  from tracing edges you know are parallel. An organic scene (trees, terrain,
  rock) has nothing to trace, and no amount of care will fix that.
- **Move sideways between shots, not forwards.** Walking *toward* your subject
  gives almost no parallax and the triangulation degenerates. Walk *around* it.
- **Wide-ish lens beats telephoto.** A long lens compresses perspective and
  throws vanishing points so far off-frame that focal length stops being
  measurable. 24–50 mm equivalent is comfortable; 200 mm is hopeless.
- **Tall, long edges in frame.** Vanishing-point accuracy is set by the length of
  the lines you can draw. A scene whose verticals are 200 px tall will not give
  you a focal length, however carefully you click.
- **5–15 photos.** Enough to over-constrain the shared focal, few enough to mark
  in one sitting.
- **Same camera, same zoom.** That is what lets them share one focal length,
  which is where most of the accuracy comes from. Mixed cameras still work —
  put them in separate camera groups — but you lose that.
- **A known distance**, or a known object, if you want metric output.

Two failure modes worth recognising before you shoot:

- **Lens distortion.** `k1` is locked by default, and the control-line residual
  assumes straight lines. Noticeable barrel distortion bends the very edges you
  are tracing. Symptom: control-line RMS high while tie points fit fine. Fix:
  unlock `k1`.
- **A long lens with a shallow tilt.** Both the horizontal and vertical vanishing
  points end up effectively at infinity and the focal is simply not observable.
  The Focal check panel catches this before you invest in a markup — believe it.

---

## Troubleshooting by symptom

| Symptom | What it means | What to do |
|---|---|---|
| Focal check says **poor**, high lever | Your control lines are too short for how far their vanishing point is | Draw longer lines on the limiting axis, or lock the focal from EXIF / a known value |
| *"no vanishing point pair yields a real focal length"* | Axes are not perpendicular as marked | A control line is on the wrong axis, or on an edge that is not actually parallel |
| *"focal estimates from different axis pairs disagree"* | One axis family is wrong | Turn on **VP rays** and find the line that will not converge |
| Reprojection RMS high, control lines fine | A mis-clicked tie point | **Per-image residuals**, worst first; the Tie points panel flags the bad point |
| Control-line RMS high, tie points fine | Lens distortion, or a line on a non-parallel edge | Unlock `k1`, or re-check the lines |
| Some cameras **not placed** | Fewer than 2 tie points shared with the rest | The report names them; add points |
| *"not connected to the main group"* | No tie-point chain reaches those photos | Add points bridging the two groups |
| **init conditioning** below 10 | Too few tie points on some pair, or they are nearly coplanar | Add a couple of well-spread points, varied in depth |
| Everything looks fine but distances are 1000× off | Model units | Check the units field in Known objects. State one real distance and the solve catches this itself — *"disagree about scale by a factor of 10x"* |
| *"disagree about scale by a factor of …"* | Your stated figures and the object's model do not agree | A near-round factor is almost always the model's declared units; otherwise check which tie points the constraints name |
| A constraint shows **no effect** | It names a tie point that was not solved | Mark that point in a second photo, or delete the constraint |
| Constraint error large, everything else fine | Two constraints pulling against each other, or against the object geometry | Compare **stated vs object scale**; drop whichever figure you trust less |
| Solved size not 1.000× | An anchor is on the wrong feature | Check that body's shape error, re-mark it |
| Shape error high and will not come down | Anchors on features you cannot actually identify — bolt rims are the classic | Use ring (centre) and cross (filleted corner) anchors instead |
| The corners you want are not offered | They are filleted, so no vertex sits there | Look for the **cross** marks just outside each rounded corner; that is the corner, reconstructed |
| Nothing useful offered at all | A heavily filleted part with no round features | **type a position…** and enter coordinates from the drawing |
| Grid skewed or floating | Frame is wrong despite good residuals | Check control-line directions — a mirrored solve |
| *"hit the iteration limit"* | The solve did not converge | Treat the numbers as provisional; usually a symptom of one of the above |

---

## Quick reference

### Modes

| Key | Mode |
|---|---|
| **L** | Lines — control lines, axis chosen with **1** / **2** / **3** |
| **P** | Points — tie points |
| **O** | Objects — anchors on known objects |
| **S** | Select — move or delete existing marks |

### Keys

| Key | |
|---|---|
| `1` `2` `3` | X / Y / Z axis |
| `N` | New tie point |
| `Tab` | Next photo missing the active point or anchor |
| `←` `→` | Previous / next photo |
| `D` | 3D view |
| `F` | Fit view |
| `V` `R` `G` `M` | VP rays / residuals / grid / models |
| `Enter` | Solve |
| `Ctrl+S` | Save |
| `?` | Open this tutorial |
| `Delete`, right-click | Remove what is under the cursor |
| click a drawn line | Reverse its direction |
| wheel / space-drag | Zoom / pan |

### Minimums

| | |
|---|---|
| Control lines | 2 per axis, on 2 axes |
| Tie points | 2 shared per image pair |
| Object anchors | 3 anchors, in 2 photos each |

### Command line

```bat
.venv\Scripts\python.exe -m multispy                       open the UI
.venv\Scripts\python.exe -m multispy solve   <project>     solve, print the report
.venv\Scripts\python.exe -m multispy export  <project>     write the exports
.venv\Scripts\python.exe -m multispy rig     <project>     print the object layout
.venv\Scripts\python.exe -m multispy exif    <project>     what the photos claim
```

### Practice scenes

```bat
.venv\Scripts\python.exe demo\make_demo.py     --blank --out demo_practice
.venv\Scripts\python.exe demo\make_rig_demo.py --blank --out demo_rig_practice
```

Drop `--blank` for a fully-marked version to compare against. Ground truth:
box **10 × 6 × 4** at **3200 px** focal; rig **225 mm** spacing, **6°** toe-in.
