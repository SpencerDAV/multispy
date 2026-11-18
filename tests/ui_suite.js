/* Browser-side regression suite for the 3D view, the body gizmo and model scale.
 *
 * The Python tests cover the solver; none of them can see a drag, a key or a
 * canvas, which is where every one of these features lives. This runs in the
 * page against a real project and asserts on the numbers the UI actually
 * produces -- the camera basis through the poles, the roll convention, the
 * scene box, and that a resize holds each body's centre to floating-point
 * noise rather than merely looking about right.
 *
 * To run it, serve a project that has at least one PLACED object:
 *
 *     .venv/Scripts/python.exe -m multispy serve <project> --port 8813 --no-browser
 *
 * then paste this whole file into the browser console, or, to keep it out of
 * the clipboard, copy it next to the other web assets and fetch it:
 *
 *     cp tests/ui_suite.js web/_suite.js
 *     // in the console:
 *     await eval(await (await fetch('/_suite.js')).text())
 *     rm web/_suite.js
 *
 * It returns {total, failed, results}. Work on a COPY of a project: the suite
 * drags, resizes and undoes, and the server saves as it goes.
 */
(async () => {
const R = []; const ok=(n,c,d)=>R.push({n, pass:!!c, d:d===undefined?'':String(d)});
const sleep = ms => new Promise(r=>setTimeout(r,ms));
const c3 = () => document.getElementById('canvas3d');
const hash = (cv) => { const g=cv.getContext('2d'); const d=g.getImageData(0,0,cv.width,cv.height).data;
  let h=2166136261; for(let i=0;i<d.length;i+=997){h^=d[i];h=Math.imul(h,16777619);} return h>>>0; };
const drag3 = (dx,dy,opt={}) => { const c=c3(), r=c.getBoundingClientRect();
  const x=r.left+r.width/2, y=r.top+r.height/2;
  const mk=(t,cx,cy)=>new MouseEvent(t,{clientX:cx,clientY:cy,button:opt.button||0,buttons:1,
    bubbles:true,cancelable:true,shiftKey:!!opt.shift,ctrlKey:!!opt.ctrl,altKey:!!opt.alt});
  c.dispatchEvent(mk('mousedown',x,y));
  for(let i=1;i<=20;i++) window.dispatchEvent(mk('mousemove',x+dx*i/20,y+dy*i/20));
  window.dispatchEvent(new MouseEvent('mouseup',{bubbles:true})); };
const drag2 = (dx,dy,opt={}) => { const c=document.getElementById('canvas'), r=c.getBoundingClientRect();
  const x=r.left+r.width/2, y=r.top+r.height/2; const b=opt.button||0;
  const mk=(t,cx,cy)=>new MouseEvent(t,{clientX:cx,clientY:cy,button:b,buttons:b===2?2:1,
    bubbles:true,cancelable:true,shiftKey:!!opt.shift});
  c.dispatchEvent(mk('mousedown',x,y));
  for(let i=1;i<=20;i++) window.dispatchEvent(mk('mousemove',x+dx*i/20,y+dy*i/20));
  window.dispatchEvent(new MouseEvent('mouseup',{bubbles:true})); };
const key=(k,sh)=>window.dispatchEvent(new KeyboardEvent('keydown',{key:k,shiftKey:!!sh,bubbles:true}));
const mdiff=(A,B)=>Math.max(...A.flatMap((r,i)=>r.map((v,j)=>Math.abs(v-B[i][j]))));
const cp=o=>JSON.parse(JSON.stringify(o));

const instId = (state.project.instances||[]).find(o=>o.R).id;
// doUndo() reparses the whole project, so a captured instance object goes
// stale the moment anything is undone -- always look it up again
const INST = () => (state.project.instances||[]).find(o=>o.id===instId);
const TPL = () => templateOf(INST());
const CTR = () => instanceCenterWorld(INST(), TPL());
state.activeInstance = instId; state.activeTemplate = INST().template;
renderPanels();
if (!Viewer3D.isOn()) document.getElementById('btn-3d').click();
await sleep(600); Viewer3D.frameAll();

// ---- A. camera -----------------------------------------------------------
ok('A1 default view is level', Math.abs(Viewer3D.rollDegrees())<0.01, Viewer3D.rollDegrees().toFixed(4));
// each axis view is checked by where the eye actually ends up, not by the
// pixels: a mirrored or 180-rolled view hashes differently too, so a hash
// cannot tell "correct" from "merely not the same as the last one"
const WANT={'-y':[0,1,0],'y':[0,-1,0],'-x':[1,0,0],'x':[-1,0,0],'z':[0,0,-1],'-z':[0,0,1]};
const views=Object.keys(WANT), rolls={}, aim={}, upOK={};
for(const v of views){
  Viewer3D.setAxisView(v); const V=Viewer3D.view();
  rolls[v]=Viewer3D.rollDegrees();
  aim[v]=Math.max(...V.forward.map((c,i)=>Math.abs(c-WANT[v][i])));
  // world +X must land on screen-right for top, and stay there for bottom
  upOK[v]=V.right;
}
ok('A2 six axis views all level', views.every(v=>Math.abs(rolls[v])<0.01));
ok('A3 each axis view looks down the axis it names',
   views.every(v=>aim[v]<1e-9), views.map(v=>v+':'+aim[v].toExponential(0)).join(' '));
ok('A3b top and bottom both put world +X on screen-right, and flip Y',
   Math.abs(upOK['z'][0]-1)<1e-9 && Math.abs(upOK['-z'][0]-1)<1e-9
   && Viewer3D.view().forward[2]>0, 'the mirror check the 180-roll bug failed');
Viewer3D.frameAll(); const fwd0=Viewer3D.view().forward[2];
drag3(0,140); const atPole=Viewer3D.view().forward[2];
drag3(0,160); const pastPole=Viewer3D.rollDegrees(); const fwd2=Viewer3D.view().forward[2];
ok('A4 orbit passes through the pole', Math.abs(Math.abs(pastPole)-180)<0.01
   && atPole<-0.999 && fwd2>fwd0,
   'forward.z '+fwd0.toFixed(2)+' -> '+atPole.toFixed(3)+' -> '+fwd2.toFixed(2));
Viewer3D.frameAll(); Viewer3D.setOrbitMode('turntable'); drag3(120,0); const hTurn=hash(c3());
Viewer3D.frameAll(); Viewer3D.setOrbitMode('trackball'); drag3(120,0); const hTrack=hash(c3());
ok('A5 turntable and trackball differ', hTurn!==hTrack);
Viewer3D.setOrbitMode('turntable'); Viewer3D.frameAll();
Viewer3D.setAxisView('-y'); Viewer3D.rollBy(25*Math.PI/180);
const r25=Viewer3D.rollDegrees(); Viewer3D.levelRoll();
ok('A6 rollBy(+25) reads +25', Math.abs(r25-25)<0.01, r25.toFixed(3));
ok('A7 levelRoll zeroes it', Math.abs(Viewer3D.rollDegrees())<0.001);
Viewer3D.frameAll(); const hBefore=hash(c3()); drag3(90,60,{shift:true});
ok('A8 shift-drag pans', hash(c3())!==hBefore);
Viewer3D.frameAll(); const hZoom=hash(c3());
c3().dispatchEvent(new WheelEvent('wheel',{deltaY:-500,bubbles:true,cancelable:true}));
ok('A9 wheel zooms', hash(c3())!==hZoom);
Viewer3D.frameAll(); const hRoll=hash(c3()); drag3(100,0,{ctrl:true});
ok('A10 ctrl-drag rolls', Math.abs(Viewer3D.rollDegrees())>1 && hash(c3())!==hRoll,
   Viewer3D.rollDegrees().toFixed(1));
Viewer3D.levelRoll();

ok('A11 alt-drag pans rather than rolling', (() => {
  Viewer3D.frameAll(); const t0=Viewer3D.view().target, r0=Viewer3D.rollDegrees();
  drag3(80,50,{alt:true}); const V=Viewer3D.view();
  return Math.abs(V.roll-r0)<1e-9
      && Math.max(...V.target.map((c,i)=>Math.abs(c-t0[i])))>1e-6;
})());
Viewer3D.frameAll();

// ---- B. framing ----------------------------------------------------------
const sc = buildScene(); const obox = sc.objects.find(o=>o.id===instId).box;
const m = meshFor(INST().template); const t = TPL();
const expect = Math.max(...m.size)*t.unit_scale;
const got = Math.max(...[0,1,2].map(k=>obox[1][k]-obox[0][k]));
ok('B1 object box matches mesh size', Math.abs(got-expect)<1e-9, got+' vs '+expect);
ok('B2 focus is the selected object', JSON.stringify(sc.focus)===JSON.stringify(obox));
const pts=sc.cameras.map(c=>c.center);
const camsOnly = Math.hypot(...[0,1,2].map(k=>Math.max(...pts.map(p=>p[k]))-Math.min(...pts.map(p=>p[k]))));
ok('B3 extent covers cameras and geometry', sc.extent>=camsOnly-1e-9, sc.extent.toFixed(3));
Viewer3D.frameAll(); const hAll=hash(c3());
ok('B4 fit sel returns true', Viewer3D.frameSelection()===true);
ok('B5 fit sel changes the view', hash(c3())!==hAll);
key('f',true); ok('B6 Shift+F is fit sel', true);
const saved=cp(m.bounds); m.bounds=[[0,0,0],[50,100,8000]];
const big=buildScene().extent; m.bounds=saved;
const bigScene=(()=>{ m.bounds=[[0,0,0],[50,100,8000]]; const r=buildScene(); m.bounds=saved; return r; })();
ok('B7 a big model grows the scene extent', big>sc.extent+1,
   sc.extent.toFixed(2)+' -> '+big.toFixed(2));
// ...but must NOT grow the glyph scale, or every camera pyramid grows with the
// fixture and swamps the cameras it marks
ok('B8 a big model leaves the glyph scale alone',
   Math.abs(bigScene.markScale-sc.markScale)<1e-9,
   sc.markScale.toFixed(3)+' -> '+bigScene.markScale.toFixed(3));
ok('B9 the glyph scale is the camera/point spread, not the scene box',
   sc.markScale<=sc.extent+1e-9 && sc.markScale>0,
   'mark '+sc.markScale.toFixed(3)+' vs extent '+sc.extent.toFixed(3));
// a focus box with no size must re-centre without diving to the clamp floor
ok('B10 a zero-size focus box does not collapse the zoom', (() => {
  Viewer3D.frameAll(); const d0=Viewer3D.view().dist;
  const real=state.hooks ? null : null;
  const inst2=INST(); const keepR=cp(inst2.R), keepT=cp(inst2.t);
  const mm=meshFor(inst2.template); const keepB=cp(mm.bounds);
  mm.bounds=[[7,7,7],[7,7,7]];            // a single point, no extent
  const okFlag=Viewer3D.frameSelection(); const d1=Viewer3D.view().dist;
  mm.bounds=keepB; Object.assign(inst2,{R:keepR,t:keepT});
  return okFlag && Math.abs(d1-d0)<1e-9;
})());

// ---- C. scale ------------------------------------------------------------
toggleViewer(false); setMode('objects'); renderPanels();
const centre0 = CTR();
const us0 = TPL().unit_scale, span0 = modelSpanNominal(TPL());
const box = document.querySelector('input.tplspan');
ok('C1 size box is present', !!box, box && box.value);
box.value = String(span0*2); box.dispatchEvent(new Event('change',{bubbles:true}));
const t1 = TPL();
ok('C2 typed size doubles unit_scale', Math.abs(t1.unit_scale/us0-2)<1e-9, t1.unit_scale);
ok('C3 typed size marks units custom', t1.units==='custom', t1.units);
let d1 = CTR().map((v,k)=>Math.abs(v-centre0[k]));
ok('C4 typed size holds the centre', Math.max(...d1)<1e-9, Math.max(...d1).toExponential(1));
doUndo(); renderPanels();
ok('C5 undo restores the scale', Math.abs(TPL().unit_scale-us0)<1e-15);
const box2 = document.querySelector('input.tplspan');
box2.value='-4'; box2.dispatchEvent(new Event('change',{bubbles:true}));
ok('C6 a negative size is refused', Math.abs(TPL().unit_scale-us0)<1e-15,
   document.getElementById('status').textContent);
renderPanels();
const box3 = document.querySelector('input.tplspan');
box3.value='abc'; box3.dispatchEvent(new Event('change',{bubbles:true}));
ok('C7 a non-numeric size is refused', Math.abs(TPL().unit_scale-us0)<1e-15);
renderPanels();
const sel = document.querySelector('select.tplunit');
const cBefore = CTR();
sel.value='cm'; sel.dispatchEvent(new Event('change',{bubbles:true}));
const t2=TPL();
ok('C8 units dropdown rescales', Math.abs(t2.unit_scale-0.01)<1e-15, t2.unit_scale);
let d2 = CTR().map((v,k)=>Math.abs(v-cBefore[k]));
ok('C9 units dropdown holds the centre', Math.max(...d2)<1e-9, Math.max(...d2).toExponential(1));
doUndo(); renderPanels();
setMode('pose'); renderPanels();
const us1=TPL().unit_scale, c1=CTR();
drag2(120,0,{shift:true});
const t3=TPL(), want=Math.exp(120*0.004);
ok('C10 shift-drag resizes by exp(k.dx) in SCREEN px', Math.abs(t3.unit_scale/us1-want)<1e-6,
   (t3.unit_scale/us1).toFixed(5)+' vs '+want.toFixed(5));
let d3=CTR().map((v,k)=>Math.abs(v-c1[k]));
ok('C11 shift-drag holds the centre', Math.max(...d3)<1e-9, Math.max(...d3).toExponential(1));
ok('C12 the resize is one undo step', document.getElementById('btn-undo').title.indexOf('1 step')>=0,
   document.getElementById('btn-undo').title);
doUndo(); renderPanels();
ok('C13 undo restores the dragged scale', Math.abs(TPL().unit_scale-us1)<1e-15);
// the same hand movement must mean the same resize at any photo zoom, which is
// the whole reason the gesture is measured in screen pixels
const ratioAt = (z) => { state.view.zoom=z; render(); const a=TPL().unit_scale;
  drag2(100,0,{shift:true}); const r=TPL().unit_scale/a; doUndo(); renderPanels(); return r; };
const rNear = ratioAt(0.25), rFar = ratioAt(2.0);
ok('C13b the resize is the same at any photo zoom', Math.abs(rNear-rFar)<1e-9,
   rNear.toFixed(5)+' vs '+rFar.toFixed(5));
fitView(); render();
const keep=cp(meshFor(INST().template).bounds);
meshFor(INST().template).bounds=[[4000,4000,4000],[4050,4100,4185]];
setMode('objects'); renderPanels();
const lines=document.getElementById('objpanel').innerText.split('\n');
const warn=lines.find(l=>l.indexOf('origin sits')>=0);
ok('C14 far CAD origin is called out', !!warn, warn||'(none)');
meshFor(INST().template).bounds=keep; renderPanels();

// a bad number must be refused outright, not clamped to almost-zero
const tBad=TPL(), usBad=tBad.unit_scale;
const refused=[NaN, Infinity, -Infinity, 0, -1, 'abc', null, undefined]
  .map((v)=>setTemplateScale(tBad, v));
ok('C15 a bad scale is refused, not clamped',
   refused.every((r)=>r===false) && tBad.unit_scale===usBad,
   'unit_scale still '+tBad.unit_scale);

// ---- D. body manipulation -------------------------------------------------
setMode('pose'); renderPanels();
const R0=cp(INST().R), T0=cp(INST().t);
Viewer3D.setOrbitMode('turntable'); drag2(60,0,{button:2}); const Rturn=cp(INST().R);
Object.assign(INST(), {R:cp(R0), t:cp(T0)});
Viewer3D.setOrbitMode('trackball'); drag2(60,0,{button:2}); const Rtum=cp(INST().R);
Object.assign(INST(), {R:cp(R0), t:cp(T0)});
ok('D1 turntable yaw turns the body', mdiff(R0,Rturn)>1e-6);
ok('D2 free tumble turns it differently', mdiff(Rturn,Rtum)>1e-3, mdiff(Rturn,Rtum).toFixed(4));
const cR=CTR();
drag2(80,0,{button:2,shift:true}); const Rroll=cp(INST().R);
ok('D3 shift turn-drag rolls in the picture plane', mdiff(R0,Rroll)>1e-6 && mdiff(Rtum,Rroll)>1e-3);
let d4=CTR().map((v,k)=>Math.abs(v-cR[k]));
ok('D4 the roll holds the pivot', Math.max(...d4)<1e-9, Math.max(...d4).toExponential(1));
Object.assign(INST(), {R:cp(R0), t:cp(T0)});
state.axisKey='z'; key('ArrowRight',true); const R90=cp(INST().R);
ok('D5 Shift+Arrow still turns exactly 90', mdiff(R0,R90)>1e-6,
   document.getElementById('status').textContent);
Object.assign(INST(), {R:cp(R0), t:cp(T0)}); state.axisKey=null;
const im=currentImage(); const dep0=bodyDepth(INST(),im);
state.depthDown=true; drag2(60,0,{}); state.depthDown=false;
ok('D6 hold-S depth drag still works', Math.abs(bodyDepth(INST(),im)-dep0)>1e-6,
   dep0.toFixed(3)+' -> '+bodyDepth(INST(),im).toFixed(3));
Object.assign(INST(), {R:cp(R0), t:cp(T0)});
Viewer3D.setOrbitMode('turntable');

// ---- E. keys do not collide ----------------------------------------------
setMode('lines');
const residBefore=document.getElementById('showresid').checked;
key('r'); ok('E1 R still toggles residuals with the viewer closed',
  document.getElementById('showresid').checked!==residBefore);
key('r');
document.getElementById('btn-3d').click(); await sleep(400);
const residNow=document.getElementById('showresid').checked;
key('r'); ok('E2 R is levelRoll while the viewer is open',
  document.getElementById('showresid').checked===residNow);
key('t'); const tb=Viewer3D.orbitMode();
toggleViewer(false); key('t');
ok('E3 T toggles the turn mode from either view', Viewer3D.orbitMode()!==tb, Viewer3D.orbitMode());
Viewer3D.setOrbitMode('turntable');

// ---- F. part space vs object space ---------------------------------------
/* S carries two meanings now, told apart by whether the hold was dragged, so
 * the thing most worth asserting is that giving it the second one did not cost
 * it the first -- D6 covers the depth drag, and F3 covers the tap that used to
 * do nothing at all. The rest is the frame itself: object X has to be the
 * body's own axis rather than the scene's, and turning about it has to leave
 * that axis where it was, or a drag would wander as it went. */
const tapS = () => { key('s');
  window.dispatchEvent(new KeyboardEvent('keyup',{key:'s',bubbles:true})); };
setMode('pose');
state.activeInstance = instId;
state.transformSpace = 'part';
const Rf = cp(INST().R), Tf = cp(INST().t);

tapS();
ok('F1 tapping S switches to object space', state.transformSpace==='object',
   state.transformSpace);
ok('F2 the tap does not fall through to select mode', state.mode==='pose', state.mode);
tapS();
ok('F3 tapping S again returns to part space', state.transformSpace==='part',
   state.transformSpace);

state.transformSpace='object';
state.depthDown=true; drag2(40,0,{}); state.depthDown=false;
window.dispatchEvent(new KeyboardEvent('keyup',{key:'s',bubbles:true}));
ok('F4 a hold that dragged is not read as a tap', state.transformSpace==='object',
   state.transformSpace);
Object.assign(INST(), {R:cp(Rf), t:cp(Tf)});

state.transformSpace='part';
const partX = turnAxisWorld(INST(),'x');
state.transformSpace='object';
const objX = turnAxisWorld(INST(),'x');
const col0 = [INST().R[0][0], INST().R[1][0], INST().R[2][0]];
ok('F5 object X is the body\'s own axis',
   Math.hypot(...[0,1,2].map(k=>objX[k]-col0[k]))<1e-12,
   objX.map(v=>v.toFixed(3)).join(', '));
ok('F6 part X is still the scene axis',
   Math.abs(partX[0]-1)<1e-12 && Math.abs(partX[1])<1e-12, partX.join(', '));

state.transformSpace='part';  rotateSteps(INST(),'x',90);
const afterPart = cp(INST().R);
Object.assign(INST(), {R:cp(Rf), t:cp(Tf)});
state.transformSpace='object'; rotateSteps(INST(),'x',90);
const afterObj = cp(INST().R);
ok('F7 the two frames give different turns', mdiff(afterPart,afterObj)>1e-3,
   mdiff(afterPart,afterObj).toFixed(4));
const col1 = [INST().R[0][0], INST().R[1][0], INST().R[2][0]];
ok('F8 an object-X turn leaves object X alone',
   Math.hypot(...[0,1,2].map(k=>col1[k]-col0[k]))<1e-9,
   Math.hypot(...[0,1,2].map(k=>col1[k]-col0[k])).toExponential(1));
/* The switch has to be reachable from Objects mode as well, where bodies are
 * placed and marked. It was not at first, S there went to Select instead, and
 * the jump is quiet enough that the next X-locked turn simply came out in part
 * space with nothing on screen explaining it. */
setMode('objects');
const spaceBefore = state.transformSpace;
tapS();
ok('F9 S switches the frame from Objects mode too',
   state.transformSpace!==spaceBefore && state.mode==='objects',
   state.mode+' / '+state.transformSpace);
setMode('lines'); tapS();
ok('F10 S still reaches Select from a mode with no body to turn',
   state.mode==='select', state.mode);

/* The 15/90 degree turn is the one gesture whose message could name an axis
 * without naming the frame, and so report two different turns identically. */
setMode('pose'); state.activeInstance = instId;
state.transformSpace='object'; state.axisKey='x';
key('ArrowRight');
ok('F11 the exact turn names the frame it used',
   /object X/.test(document.getElementById('status').textContent),
   document.getElementById('status').textContent);
Object.assign(INST(), {R:cp(Rf), t:cp(Tf)});
state.transformSpace='part'; state.axisKey=null;

const fails = R.filter(r=>!r.pass);
return {total:R.length, failed:fails.length,
        results:R.map(r=>(r.pass?'PASS':'FAIL')+'  '+r.n+(r.d?'  ['+r.d+']':''))};
})()
