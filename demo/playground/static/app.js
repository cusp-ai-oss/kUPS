import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- connection
// Live from the server; when it is out of reach, a recorded tour plays through the same handlers.
let socket = null;
let config = null;
let busy = null;
let replay = null; // the Player, once live is out of reach
let opened = false;
let connects = 0; // socket openings; a tour restarts after a reconnect
let heard = 0; // when the socket last carried a message
let down = 0; // when the live socket went down
let retry = 1000; // reconnect delay; backs off while offline
let fallback = null; // starts the replay unless the socket (re)opens first
const bundled = document.querySelector("meta[name=kups-replay]")?.content; // a static bundle's recording: replay it, never look for a server
let recording = fetchRecording();
let nextScene = null; // only the latest structure is drawn, once per animation frame, so a slow page never lags behind
let resumeTour = false; // a tour carried over a reload
let liveTour = false; // whether the live page was touring when it fell back to replay
try {
  resumeTour = sessionStorage.getItem("tour") === "1";
  sessionStorage.removeItem("tour");
} catch {}

const dispatch = (msg) => ({ frame: onFrame, config: onConfig, status: onStatus, event: onEvent })[msg.type]?.(msg);
const isOpen = () => socket?.readyState === WebSocket.OPEN;

function connect() {
  try {
    const url = new URL("ws", document.baseURI);
    url.protocol = url.protocol.replace("http", "ws");
    socket = new WebSocket(url);
  } catch {
    return lost(); // no websocket from this page, e.g. an about: or file: URL
  }
  const handshake = socket;
  setTimeout(() => { // a stale tunnel can hold a handshake open for minutes
    if (socket === handshake && handshake.readyState === WebSocket.CONNECTING) { lost(); handshake.close(); }
  }, 8000);
  socket.onopen = () => {
    opened = true;
    connects++;
    heard = performance.now();
    down = 0;
    retry = 1000;
    clearTimeout(fallback);
    fallback = null;
    $("#conn").classList.add("on");
    document.body.classList.remove("offline");
    if (!replay) $("#conn-label").textContent = "";
  };
  socket.onclose = lost;
  socket.onmessage = (event) => {
    heard = performance.now();
    const msg = JSON.parse(event.data);
    if (!replay) dispatch(msg);
    else if (msg.type === "config") reload(); // the server is back and ready: go live
  };
}
function lost() {
  if (socket) Object.assign(socket, { onclose: null, onmessage: null });
  socket = null;
  $("#conn").classList.remove("on");
  document.body.classList.toggle("offline", !replay);
  if (opened) {
    down ||= performance.now();
    if (!replay) $("#conn-label").textContent = "Reconnecting…";
    fallback ??= setTimeout(fallBack, 5000);
  } else if (fallback) { // no server behind this page: replay now
    clearTimeout(fallback);
    fallBack();
  }
  setTimeout(connect, retry);
  retry = Math.min(2 * retry, opened ? 4000 : 30000); // a page that never had a server is on a static host
}
function reload() {
  try { sessionStorage.setItem("tour", touring && (!replay || liveTour) ? "1" : ""); } catch {}
  location.reload();
}
function send(msg) {
  if (replay) return replay.command(msg);
  if (!isOpen()) return toast("Reconnecting · try again in a moment", true);
  socket.send(JSON.stringify(msg));
  hush ||= Boolean(msg.quiet);
}

// A paused or compiling server is silent too, so silence only counts once /live stops answering.
setInterval(async () => {
  if (replay || !isOpen() || performance.now() - heard < 6000) return;
  const stalled = socket;
  // Messages that arrive during the probe, e.g. after a long main-thread stall, prove the socket alive.
  if (!(await serverUp()) && socket === stalled && performance.now() - heard >= 6000) {
    stalled.close();
    lost();
  }
}, 4000);

/** The recorded tour's bytes, fetched up front so a dropped tunnel still has them; parsed only to replay. */
async function fetchRecording() {
  for (const path of bundled ? [bundled] : ["recordings/tour.json.gz", "recordings/tour.json"]) {
    try {
      const response = await fetch(path, { signal: AbortSignal.timeout(30000) }); // a stalled tunnel moves on
      const bytes = new Uint8Array(response.ok ? await response.arrayBuffer() : []);
      if ((bytes[0] === 0x1f && bytes[1] === 0x8b) || bytes[0] === 0x7b) return bytes.buffer; // gzip or JSON, not a host's fallback page
    } catch {}
  }
  return null;
}
async function parseRecording(buffer) {
  const bytes = new Uint8Array(buffer);
  const gzipped = bytes[0] === 0x1f && bytes[1] === 0x8b; // else the host already unpacked it
  return new Response(gzipped ? new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")) : bytes).json();
}

/** Whether the kUPS server is behind this page; a host that answers every path does not count. */
const serverUp = () => fetch("live", { cache: "no-store", signal: AbortSignal.timeout(8000) })
  .then((r) => r.json()).then((body) => body?.build != null).catch(() => false);
/** Replay unless the live server still answers: a slow socket must not hide a reachable server, for a while. */
async function fallBack() {
  fallback = null;
  if (isOpen() || (opened && performance.now() - down < 15000 && (await serverUp()))) return;
  startReplay();
}

async function startReplay() {
  if (replay) return;
  const buffer = await recording;
  const rec = buffer && (await parseRecording(buffer).catch(() => null));
  if (replay || isOpen()) return;
  if (!rec) { // try again soon, from the network or the server
    recording = fetchRecording();
    $("#conn-label").textContent = "Offline";
    fallback ??= setTimeout(fallBack, 5000);
    return;
  }
  clearTimeout(fallback);
  replay = new Player(rec);
  document.body.classList.remove("offline");
  $("#conn").classList.add("replay");
  $("#conn-label").textContent = rec.device ? `Replay · ${rec.device}` : "Replay";
  liveTour = touring;
  runTour();
}

// ---------------------------------------------------------------- viewer
// The framework is muted and the CO₂ vivid: the guests are the story.
const FRAMEWORK = { Zr: ["#2bb39a", 0.72], O: ["#c9706b", 0.3], C: ["#8b929c", 0.28], H: ["#d5d9df", 0.18] };
const GUEST = { C: ["#ffb547", 0.58], O: ["#ff7a59", 0.54] };
const COVALENT = { H: 0.31, C: 0.76, O: 0.66, Zr: 1.75 };
const BOND = { framework: "#5d6677", guest: "#ffcf87" };
const look = (s, guest) => (guest ? GUEST : FRAMEWORK)[s] ?? FRAMEWORK.C;

const viewer = $("#viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
viewer.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.45;
scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x14171d, 1.3));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(-1, 2, 2);
const rim = new THREE.DirectionalLight(0x8fb4ff, 1.1);
rim.position.set(2, -1, -2);
camera.add(key, rim);
scene.add(camera);

const world = new THREE.Group();
scene.add(world);
const sphere = new THREE.SphereGeometry(1, 32, 20);
const cylinder = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);
const dummy = new THREE.Object3D();
const up = new THREE.Vector3(0, 1, 0);

/** Instanced meshes that grow their capacity as needed. */
class Instances {
  constructor(geometry, material, parent) {
    Object.assign(this, { geometry, material, parent, capacity: 0, mesh: null });
  }
  reserve(n) {
    if (n <= this.capacity) return;
    if (this.mesh) { this.parent.remove(this.mesh); this.mesh.dispose(); }
    this.capacity = Math.max(64, 2 * n);
    this.mesh = new THREE.InstancedMesh(this.geometry, this.material, this.capacity);
    this.parent.add(this.mesh);
  }
  commit(n) {
    if (!this.mesh) return;
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

// The simulation cell is periodic: draw it as a block of images so the MOF reads as a crystal.
const atomMaterial = new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0.08 });
const bondMaterial = new THREE.MeshStandardMaterial({ roughness: 0.5 });
/** A set of atoms and bonds drawn as ``images``³ periodic images, centred on the origin. */
class Crystal {
  constructor(parent, images) {
    this.group = new THREE.Group();
    parent.add(this.group);
    this.atoms = new Instances(sphere, atomMaterial, this.group);
    this.bonds = new Instances(cylinder, bondMaterial, this.group);
    this.images = images;
    this.frameworkBonds = null; // [i, j] pairs, found once per scene by minimum image
  }
}
const main = new Crystal(world, 2);
const v3 = (a) => new THREE.Vector3(...a);
let fitted = false;
let fitSpan = [0, 0, 0]; // [half width, above, below] the structure spans unzoomed at any spin angle, in half viewer heights

/** Cartesian ``d`` mapped to its shortest periodic image, and the image shift used. */
function minimumImage(d, cell, inverse) {
  const frac = d.clone().applyMatrix3(inverse);
  const shift = [-Math.round(frac.x), -Math.round(frac.y), -Math.round(frac.z)];
  frac.add(v3(shift));
  return [frac.applyMatrix3(cell), shift];
}

function findBonds(symbols, positions, cell, inverse) {
  const out = [];
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const limit = 1.2 * ((COVALENT[symbols[i]] ?? 0.8) + (COVALENT[symbols[j]] ?? 0.8));
      if (minimumImage(v3(positions[j]).sub(v3(positions[i])), cell, inverse)[0].lengthSq() < limit * limit) out.push([i, j]);
    }
  }
  return out;
}

/** Draw every atom and bond in each periodic image; atoms from ``guestFrom`` on are CO₂. */
function drawCrystal(view, symbols, positions, cellRows, guestFrom) {
  const { atoms, bonds, images: span } = view;
  // Row-vector convention: r = frac @ cell, so the matrices act on the right.
  const cell = new THREE.Matrix3().set(...cellRows.flat()).transpose();
  const inverse = cell.clone().invert();
  // Guests arrive wrapped atom by atom: keep each CO₂ whole around its carbon.
  const carbon = (i) => i - ((i - guestFrom) % 3);
  positions = positions.map((p, i) => (i < guestFrom || carbon(i) === i ? p
    : minimumImage(v3(p).sub(v3(positions[carbon(i)])), cell, inverse)[0].add(v3(positions[carbon(i)])).toArray()));
  view.frameworkBonds ??= findBonds(symbols.slice(0, guestFrom), positions.slice(0, guestFrom), cell, inverse);
  const pairs = [...view.frameworkBonds];
  for (let g = guestFrom; g + 2 < symbols.length; g += 3) pairs.push([g, g + 1], [g, g + 2]);

  const [a, b, c] = cellRows.map(v3);
  const centre = a.clone().add(b).add(c).multiplyScalar((span - 1) / 2);
  const images = [];
  for (let i = 0; i < span; i++) for (let j = 0; j < span; j++) for (let k = 0; k < span; k++) {
    images.push([i, j, k]);
  }
  const offsets = images.map(([i, j, k]) => a.clone().multiplyScalar(i).addScaledVector(b, j).addScaledVector(c, k).sub(centre));
  const inBlock = (image, shift) => image.every((x, k) => x + shift[k] >= 0 && x + shift[k] < span);
  const bonded = pairs.map(([i, j]) => [i, ...minimumImage(v3(positions[j]).sub(v3(positions[i])), cell, inverse)]);

  const color = new THREE.Color();
  atoms.reserve(offsets.length * symbols.length);
  bonds.reserve(offsets.length * pairs.length);
  let n = 0, m = 0;
  offsets.forEach((offset, image) => {
    symbols.forEach((s, i) => {
      const [hex, radius] = look(s, i >= guestFrom);
      dummy.position.set(...positions[i]).add(offset);
      dummy.quaternion.identity();
      dummy.scale.setScalar(radius);
      dummy.updateMatrix();
      atoms.mesh.setMatrixAt(n, dummy.matrix);
      atoms.mesh.setColorAt(n++, color.set(hex));
    });
    for (const [i, bond, shift] of bonded) {
      if (!inBlock(images[image], shift)) continue; // partner is in an image we don't draw
      const start = v3(positions[i]).add(offset);
      const d = bond.clone();
      const guest = i >= guestFrom;
      dummy.position.copy(start).addScaledVector(d, 0.5);
      dummy.scale.set(guest ? 0.16 : 0.14, d.length(), guest ? 0.16 : 0.14);
      dummy.quaternion.setFromUnitVectors(up, d.normalize());
      dummy.updateMatrix();
      bonds.mesh.setMatrixAt(m, dummy.matrix);
      bonds.mesh.setColorAt(m++, color.set(guest ? BOND.guest : BOND.framework));
    }
  });
  atoms.commit(n);
  bonds.commit(m);

  if (view === main && !fitted) {
    const size = a.clone().add(b).add(c).multiplyScalar(span).length();
    camera.position.set(0.9, 0.55, 1.1).normalize().multiplyScalar(size * 0.95);
    controls.target.set(0, 0, 0);
    controls.update();
    camera.updateMatrixWorld();
    fitSpan = orbitSpan(offsets.flatMap((o) => positions.map((p) => v3(p).add(o))));
    fitted = true;
    resize();
  }
}

/** The extent of ``points`` on screen over a full turn of the auto-rotation about the vertical. */
function orbitSpan(points) {
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const turn = new THREE.Matrix4(), p = new THREE.Vector3();
  const span = [0, 0, 0];
  for (let deg = 0; deg < 360; deg += 5) {
    turn.makeRotationY(THREE.MathUtils.degToRad(deg)).premultiply(camera.matrixWorldInverse);
    for (const q of points) {
      p.copy(q).applyMatrix4(turn);
      const pad = FRAMEWORK.Zr[1]; // the largest atom's radius
      [Math.abs(p.x), p.y, -p.y].forEach((d, k) => (span[k] = Math.max(span[k], (d + pad) / -p.z / tan)));
    }
  }
  return span;
}

let lastCell = null;
/** The current atoms: MD holds them all; GCMC adds its guests to the rigid framework. */
function drawScene(cfg, guests, cell = lastCell) {
  lastCell = cell;
  if (!cell) return;
  const framework = cfg.static_atoms;
  if (framework) {
    drawCrystal(main, [...framework.symbols, ...guests.symbols], [...framework.positions, ...guests.positions], cell, framework.symbols.length);
  } else {
    drawCrystal(main, guests.symbols, guests.positions, cell, cfg.n_framework);
  }
}

/** Redraw in place from the config, so a hand-over never blanks the view or moves the camera. */
function setScene(cfg) {
  main.frameworkBonds = null;
  const live = cfg.static_atoms ? cfg.guests : { symbols: cfg.symbols, positions: cfg.positions };
  drawScene(cfg, { symbols: live.symbols.map((s) => s.split("_")[0]), positions: live.positions });
}
const guestCount = (cfg, atomCount) => (atomCount - (cfg.static_atoms ? 0 : cfg.n_framework)) / 3;

/** Centre the structure in the band between the HUD and the legend, and shrink it to fit there at every spin angle. */
function resize() {
  const { clientWidth: w, clientHeight: h } = viewer;
  renderer.setSize(w, h, false);
  const hud = $("#hud"), legend = $("#legend");
  const top = hud.offsetTop + hud.offsetHeight, bottom = h - legend.offsetTop;
  viewer.style.setProperty("--hud-bottom", `${top}px`);
  camera.aspect = w / Math.max(h, 1);
  const [wide, above, below] = fitSpan;
  camera.zoom = Math.min(1, (2 * Math.max(h - top - bottom, 1)) / ((above + below) * h), Math.max(w, 1) / (wide * h));
  camera.setViewOffset(w, h, 0, (bottom - top) / 2 - ((above - below) * h * camera.zoom) / 4, w, h);
  camera.updateProjectionMatrix();
  activeCharts().forEach((c) => c.resize());
  markRowEnds();
}
function toggleSpin() {
  controls.autoRotate = !controls.autoRotate;
  $("#spin").textContent = controls.autoRotate ? "Pause rotation" : "Resume rotation";
}
$("#spin").addEventListener("click", toggleSpin);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.target.closest("input, button, textarea")) {
    e.preventDefault();
    toggleSpin();
  }
});

// ---------------------------------------------------------------- batch grid
// GCMC's batched isotherm holds one system per pressure in one state: show them side by side.
const grid = new THREE.Group();
grid.visible = false;
scene.add(grid);
const tileFramework = new Crystal(grid, 2);
const tileCamera = new THREE.PerspectiveCamera(36, 1, 0.1, 2000);
let tiles = []; // [{ crystal, el }] per pressure
let tileCentre = new THREE.Vector3(), tileRadius = 1;
let showBatch = false;

function setBatch(batch) {
  const { symbols, positions, cell } = batch.framework;
  tileFramework.frameworkBonds = null;
  drawCrystal(tileFramework, symbols, positions, cell, symbols.length);
  tileFramework.atoms.mesh.computeBoundingSphere();
  ({ center: tileCentre, radius: tileRadius } = tileFramework.atoms.mesh.boundingSphere);
  tiles.forEach((t) => grid.remove(t.crystal.group));
  tiles = batch.pressures.map((p) => {
    const el = document.createElement("div");
    el.innerHTML = `<span><b>–</b> CO₂</span><span>${fmtBar(p)} bar</span>`;
    return { crystal: new Crystal(grid, 2), el, cell, pressure: p };
  });
  $("#tiles").replaceChildren(...tiles.map((t) => t.el));
}

/** The entry of ``items`` whose pressure ``at(item)`` lies nearest the running one, on a log scale. */
const nearestPressure = (items, at) => items.reduce((a, b) => (
  Math.abs(Math.log(at(b) / config.pressure)) < Math.abs(Math.log(at(a) / config.pressure)) ? b : a), items[0]);

function drawBatch(guests) {
  const near = config?.pressure && nearestPressure(tiles, (t) => t.pressure);
  tiles.forEach((t, k) => {
    const positions = guests[k];
    drawCrystal(t.crystal, positions.map((_, i) => (i % 3 ? "O" : "C")), positions, t.cell, 0);
    t.el.querySelector("b").textContent = positions.length / 3;
    t.el.classList.toggle("current", t === near);
  });
}

function toggleBatch(on = !showBatch) {
  showBatch = on && tiles.length > 0;
  viewer.classList.toggle("batch", showBatch);
  $("#batch-toggle").textContent = showBatch ? "Show one" : `Show ×${tiles.length || 8} batch`;
}
$("#batch-toggle").addEventListener("click", () => toggleBatch());

/** One viewport per tile, all looking along the main camera's direction. */
function renderTiles() {
  const box = renderer.domElement.getBoundingClientRect();
  const look = camera.position.clone().sub(controls.target).normalize();
  tileCamera.quaternion.copy(camera.quaternion);
  world.visible = false;
  grid.visible = true;
  renderer.setScissorTest(true);
  renderer.clear();
  for (const t of tiles) {
    const r = t.el.getBoundingClientRect();
    const x = r.left - box.left, y = box.bottom - r.bottom;
    renderer.setViewport(x, y, r.width, r.height);
    renderer.setScissor(x, y, r.width, r.height);
    tileCamera.aspect = r.width / Math.max(r.height, 1);
    const fov = THREE.MathUtils.degToRad(tileCamera.fov) / 2;
    const half = Math.min(fov, Math.atan(Math.tan(fov) * tileCamera.aspect));
    tileCamera.position.copy(tileCentre).addScaledVector(look, tileRadius / Math.sin(half));
    tileCamera.updateProjectionMatrix();
    tiles.forEach((u) => (u.crystal.group.visible = u === t));
    tileFramework.group.visible = true;
    renderer.render(scene, tileCamera);
  }
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, box.width, box.height);
  world.visible = true;
  grid.visible = false;
}

(function drawLatest() {
  requestAnimationFrame(drawLatest);
  const s = nextScene;
  nextScene = null;
  if (s) {
    drawScene(config, s.guests, s.cell);
    if (s.batch) drawBatch(s.batch);
  }
})();
const resizer = new ResizeObserver(resize);
resizer.observe(viewer);
resizer.observe($("#hud"));
renderer.setAnimationLoop(() => {
  controls.update();
  if (showBatch) renderTiles();
  else renderer.render(scene, camera);
});

// ---------------------------------------------------------------- charts
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4"].map(css);

const CHART_POINTS = 600; // a line chart's window
class Chart {
  constructor(spec) {
    Object.assign(this, spec, { points: [], markers: [], pending: [], ref: null, xUnit: spec.x_unit ?? "" });
    this.kind = spec.kind ?? "line";
    this.el = document.createElement("figure");
    this.el.className = "chart";
    const keys = this.series.length > 1
      ? `<span class="keys">${this.series.map((s, k) => `<span><i style="--c:${SERIES[k]}"></i>${s}</span>`).join("")}</span>`
      : "";
    this.el.innerHTML = `<figcaption><span class="title">${this.title}</span>${keys}<span class="value"></span></figcaption><canvas></canvas><div class="tip" hidden></div>`;
    this.canvas = this.el.querySelector("canvas");
    this.value = this.el.querySelector(".value");
    if (this.kind === "line" && this.series.length > 1) { // the caption quotes the first series
      this.value.classList.add("keyed");
      this.value.style.setProperty("--c", SERIES[0]);
    }
    this.tip = this.el.querySelector(".tip");
    this.ctx = this.canvas.getContext("2d");
    this.hover = null;
    this.canvas.addEventListener("pointermove", (e) => { this.hover = e.offsetX; this.draw(); });
    this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    this.canvas.width = r.width * devicePixelRatio;
    this.canvas.height = r.height * devicePixelRatio;
    this.draw();
  }
  clear() {
    this.points = [];
    this.markers = [];
    this.pending = [];
    this.hover = null;
    this.tip.hidden = true;
  }
  /** Start over: while away, the shared simulation moved on under another method. */
  enter(from) {
    this.clear();
    this.pending = from ? [from] : [];
  }
  push(x, values, ref, xUnit) {
    this.ref = ref ?? null;
    if (this.kind === "line") this.xUnit = xUnit;
    if (this.kind === "scatter") {
      this.points = values; // [[x, y], …] sorted by pressure, already aggregated server-side
      const at = values.length ? nearestPressure(values, (p) => p[0]) : null;
      const ys = at?.slice(1).filter(Number.isFinite).map((y) => fmt(y, this.digits)).join(" / ");
      this.value.textContent = at ? `${ys} ${this.unit}` : "collecting…";
      return;
    }
    if (x < this.points.at(-1)?.[0]) this.clear(); // the shared clock was reset
    this.add([x, ...values]);
    this.markers.push(...this.pending.map((label) => [x, label]));
    this.pending = [];
    this.value.textContent = `${fmt(values[0], this.digits)} ${this.unit}`;
  }
  /** Break the line while this quantity is not sampled (Hybrid's other phase). */
  gap() {
    const last = this.points.at(-1);
    if (this.kind === "line" && last?.slice(1).some(Number.isFinite)) this.add([last[0], ...this.series.map(() => NaN)]);
  }
  add(point) {
    this.points.push(point);
    if (this.points.length > CHART_POINTS) this.points.shift();
  }
  /** A marker at the latest point, or at the next one when there is none yet. */
  mark(label) {
    if (this.kind !== "line") return;
    if (this.points.length) this.markers.push([this.points.at(-1)[0], label]);
    else this.pending.push(label);
  }
  draw() {
    const { ctx, canvas } = this;
    const dpr = devicePixelRatio, W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const scatter = this.kind === "scatter";
    const points = scatter ? this.points.map(([x, ...ys]) => [Math.log10(x), ...ys]) : this.points;
    if (points.length < (scatter ? 1 : 2)) {
      ctx.fillStyle = css("--muted"); ctx.font = `${12 * dpr}px system-ui, sans-serif`; ctx.textAlign = "center";
      ctx.fillText(scatter ? "Change the pressure to trace it" : "", W / 2, H / 2);
      return;
    }
    let x0 = points[0][0], x1 = points.at(-1)[0];
    if (scatter) { x0 = Math.min(x0, -2); x1 = Math.max(x1, 1); }
    this.markers = this.markers.filter(([x]) => x >= x0);
    let lo = Infinity, hi = -Infinity;
    for (const p of points) for (let s = 1; s < p.length; s++) if (Number.isFinite(p[s])) { lo = Math.min(lo, p[s]); hi = Math.max(hi, p[s]); }
    if (this.ref != null) { lo = Math.min(lo, this.ref); hi = Math.max(hi, this.ref); }
    if (scatter) lo = Math.min(lo, 0);
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const span = hi - lo; lo -= 0.08 * span; hi += 0.08 * span;
    ctx.font = `${11 * dpr}px system-ui, sans-serif`;
    const ticks = niceTicks(lo, hi, 4, 10 ** -this.digits).map((t) => [t, fmt(t, this.digits)]);
    const pad = {
      l: Math.max(24 * dpr, ...ticks.map(([, s]) => ctx.measureText(s).width)) + 10 * dpr,
      r: (scatter ? 22 : 10) * dpr,
      t: (this.markers.length ? 18 : 8) * dpr, // marker labels sit above the plot
      b: 18 * dpr,
    };
    const X = (x) => pad.l + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (W - pad.l - pad.r);
    const Y = (y) => pad.t + (1 - (y - lo) / (hi - lo)) * (H - pad.t - pad.b);

    ctx.lineWidth = dpr;
    for (const [t, s] of ticks) {
      ctx.strokeStyle = css("--grid");
      ctx.beginPath(); ctx.moveTo(pad.l, Y(t)); ctx.lineTo(W - pad.r, Y(t)); ctx.stroke();
      ctx.fillStyle = css("--muted"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(s, pad.l - 6 * dpr, Y(t));
    }
    ctx.textBaseline = "alphabetic"; ctx.fillStyle = css("--muted");
    if (scatter) {
      ctx.textAlign = "center";
      for (let e = Math.ceil(x0); e <= x1; e++) ctx.fillText(e === Math.floor(x1) ? `${10 ** e} bar` : `${10 ** e}`, X(e), H - 3 * dpr);
    } else {
      const xd = this.xUnit === "ps" ? (x1 - x0 < 1 ? 2 : 1) : 0;
      ctx.textAlign = "left"; ctx.fillText(fmtAxis(x0, xd), pad.l, H - 3 * dpr);
      ctx.textAlign = "right"; ctx.fillText(`${fmtAxis(x1, xd)} ${this.xUnit}`, W - pad.r, H - 3 * dpr);
    }

    if (this.ref != null) { // the thermostat's target
      ctx.strokeStyle = css("--muted");
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath(); ctx.moveTo(pad.l, Y(this.ref)); ctx.lineTo(W - pad.r, Y(this.ref)); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = css("--axis");
    for (const [x] of this.markers) {
      ctx.beginPath(); ctx.moveTo(X(x), pad.t); ctx.lineTo(X(x), H - pad.b); ctx.stroke();
    }
    this.series.forEach((_, k) => {
      ctx.strokeStyle = SERIES[k]; ctx.fillStyle = SERIES[k];
      ctx.lineWidth = 2 * dpr; ctx.lineJoin = ctx.lineCap = "round";
      ctx.beginPath();
      let pen = false; // lifted across a gap
      for (const p of points) {
        if (Number.isFinite(p[k + 1])) (pen ? ctx.lineTo : ctx.moveTo).call(ctx, X(p[0]), Y(p[k + 1]));
        pen = Number.isFinite(p[k + 1]);
      }
      ctx.stroke();
      if (scatter) for (const p of points.filter((p) => Number.isFinite(p[k + 1]))) {
        ctx.strokeStyle = css("--surface");
        ctx.beginPath(); ctx.arc(X(p[0]), Y(p[k + 1]), 5 * dpr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
    });
    let nextStart = Infinity; // a label that would overlap a newer one is dropped: the latest change matters most
    ctx.lineWidth = dpr; ctx.textAlign = "left";
    for (const [x, label] of this.markers.toReversed()) {
      const w = ctx.measureText(label).width;
      const lx = Math.max(pad.l, Math.min(X(x) + 4 * dpr, W - pad.r - w - 2 * dpr)), ly = pad.t - 5 * dpr;
      if (lx + w + 8 * dpr > nextStart) continue;
      nextStart = lx;
      ctx.globalAlpha = 0.85; ctx.fillStyle = css("--surface");
      ctx.fillRect(lx - 2 * dpr, ly - 10 * dpr, w + 4 * dpr, 13 * dpr);
      ctx.globalAlpha = 1; ctx.fillStyle = css("--muted");
      ctx.fillText(label, lx, ly);
    }

    if (this.hover != null) {
      const hx = this.hover * dpr;
      const p = points.reduce((best, q) => (Math.abs(X(q[0]) - hx) < Math.abs(X(best[0]) - hx) ? q : best));
      ctx.strokeStyle = css("--muted"); ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(X(p[0]), pad.t); ctx.lineTo(X(p[0]), H - pad.b); ctx.stroke();
      this.series.forEach((_, k) => {
        if (!Number.isFinite(p[k + 1])) return;
        ctx.fillStyle = SERIES[k]; ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2 * dpr;
        ctx.beginPath(); ctx.arc(X(p[0]), Y(p[k + 1]), 4 * dpr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      });
      const xText = scatter ? `${fmt(10 ** p[0], 2)} bar` : `${fmt(p[0], 2)} ${this.xUnit}`;
      this.tip.hidden = false;
      this.tip.innerHTML = `${xText}<br>` + this.series.map((s, k) => `${s}: ${fmt(p[k + 1], this.digits)} ${this.unit}`).join("<br>");
      const left = Math.min(X(p[0]) / dpr + 10, this.el.clientWidth - this.tip.offsetWidth - 4);
      this.tip.style.left = `${left}px`; this.tip.style.top = "26px";
    }
  }
}

function niceTicks(lo, hi, n, minStep) {
  const step0 = Math.max((hi - lo) / n, minStep);
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0 * (1 - 1e-9));
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(Math.abs(t) < step / 1e6 ? 0 : t);
  return ticks;
}
function fmt(v, digits) {
  if (!Number.isFinite(v)) return "–";
  const s = v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return (/^-0(\.0*)?$/.test(s) ? s.slice(1) : s).replace("-", "−");
}
const fmtAxis = (v, digits) => (Math.abs(v) >= 1e4 ? `${fmt(v / 1e3, 1)}k` : fmt(v, digits));

const chartSets = {}; // method -> [Chart]; each method keeps its own history
const activeCharts = () => chartSets[config?.method] ?? [];
function showCharts(cfg) {
  const charts = (chartSets[cfg.method] ??= cfg.charts.map((spec) => new Chart(spec)));
  $("#charts").replaceChildren(...charts.map((c) => c.el));
  requestAnimationFrame(() => charts.forEach((c) => c.resize()));
}
let chartsDirty = false;
setInterval(() => { if (chartsDirty) { activeCharts().forEach((c) => c.draw()); chartsDirty = false; } }, 100);

// ---------------------------------------------------------------- pipeline
const STEPS = {
  MomentumStep: ["kick", "B · p += F·Δt/2"],
  PositionStep: ["drift", "A · r += p/m·Δt"],
  PotentialAsPropagator: ["force", "F = −∇E"],
  CSVRStep: ["thermostat", "velocity rescaling"],
  StochasticStep: ["thermostat", "O · friction + noise"],
  StochasticCellRescalingStep: ["barostat", "cell rescaling"],
  CellMomentumKick: ["barostat", "cell kick"],
  CellPositionStep: ["barostat", "cell drift"],
  CellStochasticStep: ["barostat", "cell friction + noise"],
  CoupledMomentumStep: ["kick", "B · cell-coupled"],
  CoupledPositionStep: ["drift", "A · cell-coupled"],
  WrapStep: ["other", "wrap into cell"],
  TranslationMove: ["move"], RotationMove: ["move"], ReinsertionMove: ["move"], ExchangeMove: ["move"],
  LennardJones: ["force"], LJTailCorrection: ["force"], Ewald: ["force"],
  GCMC: ["move"], MD: ["kick"],
};
let lastPipeline = { method: null, names: [], potential: null };

function renderPipeline(cfg) {
  const names = cfg.pipeline.map((p) => p.name);
  const potential = cfg.blocks?.potential;
  const same = lastPipeline.method === cfg.method;
  const before = {};
  if (same) lastPipeline.names.forEach((n) => (before[n] = (before[n] ?? 0) + 1));
  const seen = {};
  $("#pipeline-head").textContent = cfg.pipeline_head;
  $("#pipeline").innerHTML = cfg.pipeline.map(({ name, detail, active }) => {
    seen[name] = (seen[name] ?? 0) + 1;
    const [role, sub] = STEPS[name] ?? ["other", ""];
    const fresh = same && (seen[name] > (before[name] ?? 0) || (role === "force" && potential !== lastPipeline.potential));
    const text = detail ?? (role === "force" && potential ? `${sub} · ${cfg.potentials[potential].label}` : sub);
    return `<li class="${fresh ? "new" : ""} ${active ? "active" : ""}" style="--role:var(--role-${role})">${name}<small>${text}</small></li>`;
  }).join("");
  lastPipeline = { method: cfg.method, names, potential };
  markRowEnds();
}
/** No arrow after the last step of each row. */
function markRowEnds() {
  const steps = [...$$("#pipeline li")];
  steps.forEach((li, k) => li.classList.toggle("end", steps[k + 1]?.offsetTop !== li.offsetTop));
}

// ---------------------------------------------------------------- code panel
const KEYWORDS = /\b(import|from|def|class|return|while|for|in|if|else|self|super|True|False|None)\b/g;
function highlight(line) {
  const esc = line.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const [code, ...comment] = esc.split("#");
  const body = code
    .replace(/("[^"]*"|'[^']*')/g, "\u0000$1\u0001")
    .split(/(\u0000[^\u0001]*\u0001)/)
    .map((part) => part.startsWith("\u0000")
      ? `<span class="str">${part.slice(1, -1)}</span>`
      : part.replace(KEYWORDS, '<span class="kw">$1</span>').replace(/\b(\d+\.?\d*)\b/g, '<span class="num">$1</span>'))
    .join("");
  return comment.length ? `${body}<span class="com">#${comment.join("#")}</span>` : body;
}
let lastListing = { method: null, lines: [] };
function renderListing(cfg) {
  const previous = lastListing.method === cfg.method ? new Set(lastListing.lines) : null;
  $("#code-run").innerHTML = cfg.listing
    .map((line) => `<span class="ln ${previous && line && !previous.has(line) ? "changed" : ""}">${highlight(line) || " "}</span>`)
    .join("");
  lastListing = { method: cfg.method, lines: cfg.listing };
}

// ---------------------------------------------------------------- controls
function card(key, label, tag, { active, disabled, title }) {
  return `<button class="card ${active ? "active" : ""}" data-key="${key}" ${disabled ? "disabled" : ""} title="${title ?? ""}">
    <span>${label}</span>${tag ? `<span class="tag">${tag}</span>` : ""}</button>`;
}

const SLIDERS = {
  md: { temperature: [10, 600, 5] },
  gcmc: { temperature: [200, 450, 5] },
  hybrid: { temperature: [200, 450, 5] },
};

/** Whether the MD combination ``b`` still compiles in the background. */
const cold = (cfg, b) => cfg.warming_blocks?.some((w) => w.potential === b.potential && w.thermostat === b.thermostat && w.barostat === b.barostat);
const isPaused = (cfg) => (replay ? replay.pausedAt != null : cfg.paused);

function renderControls(cfg) {
  document.body.dataset.method = cfg.method;
  const warming = new Set(cfg.warming);
  $$("#method button").forEach((b) => {
    b.classList.toggle("active", b.dataset.v === cfg.method);
    b.disabled = warming.has(b.dataset.v) || (b.dataset.v === "hybrid" && warming.has("gcmc"));
    b.title = b.disabled ? "Warming up…" : "";
  });
  $("#pause").textContent = isPaused(cfg) ? "Resume" : "Pause";
  const slider = $("#temperature");
  const [min, max, step] = SLIDERS[cfg.method].temperature;
  Object.assign(slider, { min, max, step });
  if (document.activeElement !== slider) slider.value = cfg.temperature;

  if (cfg.moves) {
    $$("#moves button").forEach((b) => b.classList.toggle("on", cfg.moves[b.dataset.move]));
    const p = $("#pressure");
    if (document.activeElement !== p) p.value = Math.log10(cfg.pressure);
    $("#pressure-out").textContent = `${fmtBar(cfg.pressure)} bar`;
    slider.disabled = false;
    $("#temperature-out").textContent = `${Math.round(cfg.temperature)} K`;
  }
  if (!cfg.blocks) return;
  const { blocks, potentials } = cfg;
  $("#potentials").innerHTML = Object.entries(potentials).map(([k, p]) => {
    const warm = !warming.has(k) && !cold(cfg, { ...blocks, potential: k });
    const reason = p.unavailable ? `Unavailable: ${p.unavailable}` : warm ? null : "Warming up…";
    const tag = warm ? p.tag : "warming…";
    return card(k, p.label, tag, { active: k === blocks.potential, disabled: reason, title: reason });
  }).join("");
  $$("#thermostat button").forEach((b) => {
    b.classList.toggle("active", b.dataset.v === blocks.thermostat);
    const invalid = b.dataset.v === "none" && (blocks.barostat || cfg.method === "hybrid");
    b.disabled = invalid || cold(cfg, { ...blocks, thermostat: b.dataset.v });
    b.title = b.disabled && !invalid ? "Warming up…" : "";
  });
  $$("#barostat button").forEach((b) => {
    b.classList.toggle("active", b.dataset.v === String(blocks.barostat));
    const invalid = b.dataset.v === "true" && blocks.thermostat === "none";
    b.disabled = invalid || cold(cfg, { ...blocks, barostat: b.dataset.v === "true" });
    b.title = b.disabled && !invalid ? "Warming up…" : "";
  });
  slider.disabled = blocks.thermostat === "none";
  $("#temperature-out").textContent = blocks.thermostat === "none" ? "NVE" : `${Math.round(cfg.temperature)} K`;
}
const fmtBar = (p) => String(Number(p.toPrecision(2)));

const pick = (sel, attr) => (e) => e.target.closest(`${sel}:not(:disabled)`)?.dataset[attr];
$("#method").addEventListener("click", (e) => { const v = pick("button", "v")(e); if (v) send({ type: "method", value: v }); });
$("#potentials").addEventListener("click", (e) => { const key = pick(".card", "key")(e); if (key) send({ type: "blocks", potential: key }); });
$("#thermostat").addEventListener("click", (e) => { const v = pick("button", "v")(e); if (v) send({ type: "blocks", thermostat: v }); });
$("#barostat").addEventListener("click", (e) => { const v = pick("button", "v")(e); if (v) send({ type: "blocks", barostat: v === "true" }); });
$("#moves").addEventListener("click", (e) => {
  const move = pick("button", "move")(e);
  if (move && config) send({ type: "moves", move, value: !config.moves[move] });
});

function debounced(fn, ms = 150) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}
const sendTemperature = debounced((v) => send({ type: "temperature", value: v }));
$("#temperature").addEventListener("input", (e) => {
  $("#temperature-out").textContent = `${e.target.value} K`;
  sendTemperature(Number(e.target.value));
});
const sendPressure = debounced((v) => send({ type: "pressure", value: v }), 300);
$("#pressure").addEventListener("input", (e) => {
  const bar = Number(fmtBar(10 ** Number(e.target.value)));
  $("#pressure-out").textContent = `${fmtBar(bar)} bar`;
  sendPressure(bar);
});
$("#pause").addEventListener("click", () => config && send({ type: "pause", value: !config.paused, tour: touring }));
$("#reset").addEventListener("click", () => send({ type: "reset" }));

// ---------------------------------------------------------------- messages
function onConfig(cfg) {
  if (!replay && config && cfg.build !== config.build) return reload(); // server restarted with new code
  const old = config;
  const methodChanged = old?.method !== cfg.method;
  const sceneChanged = methodChanged || old.phase !== cfg.phase || (cfg.symbols && old.symbols?.length !== cfg.symbols.length);
  config = cfg;
  if (sceneChanged) {
    nextScene = null; // drawn for the old scene
    setScene(cfg);
  }
  if (cfg.batch && !tiles.length) setBatch(cfg.batch);
  if (cfg.method === "md") toggleBatch(false);
  if (methodChanged) {
    showCharts(cfg);
    const from = old && `from ${old.method === "hybrid" ? "Hybrid" : old.method.toUpperCase()}`;
    activeCharts().forEach((c) => c.kind === "line" && c.enter(from));
  } else if (old.phase !== cfg.phase) activeCharts().forEach((c) => c.gap()); // a skipped phase must not be bridged
  renderControls(cfg);
  renderPipeline(cfg);
  renderListing(cfg);
  renderHud(cfg);
  chartsDirty = true;
  if (resumeTour && !replay) {
    resumeTour = false;
    runTour();
  }
}

/** One line a passer-by can read: what is running, under which conditions. */
function renderHud(cfg) {
  const ensemble = cfg.blocks?.barostat ? (cfg.method === "hybrid" ? "NPT" : "NPT · 1\u00a0bar") : "NVT";
  const kelvin = `${Math.round(cfg.temperature)}\u00a0K`;
  const parts = cfg.method === "hybrid"
    ? [
      `Hybrid · now ${cfg.phase.toUpperCase()}`,
      cfg.potentials[cfg.blocks.potential].label,
      ensemble,
      `${fmtBar(cfg.pressure)}\u00a0bar`,
      kelvin,
    ]
    : cfg.method === "md"
    ? [
      "Molecular dynamics",
      cfg.potentials[cfg.blocks.potential].label,
      cfg.blocks.thermostat === "none" ? "NVE" : ensemble,
      cfg.blocks.thermostat === "none" ? null : kelvin,
    ]
    : ["Grand-canonical Monte Carlo", "LJ + Ewald", `${fmtBar(cfg.pressure)}\u00a0bar`, kelvin];
  const paused = isPaused(cfg);
  $("#hud-sub").textContent = [paused ? "Paused" : null, ...parts].filter(Boolean).join(" · ");
  if (paused) $("#rate").textContent = "0";
}

function onFrame(f) {
  if (!config || f.method !== config.method) return;
  const symbols = (f.symbols ?? config.symbols).map((s) => s.split("_")[0]);
  if (symbols.length !== f.positions.length) return;
  nextScene = { guests: { symbols, positions: f.positions }, cell: f.cell, batch: f.batch };
  $("#hud-n").textContent = guestCount(config, symbols.length);
  for (const chart of activeCharts()) {
    const values = f.charts[chart.key];
    if (values == null) { chart.gap(); continue; } // not this phase's quantity (hybrid)
    const [x, unit] = f.chart_x?.[chart.key] ?? [f.x, f.x_unit];
    chart.push(x, values, f.reference?.[chart.key], unit);
  }
  chartsDirty = true;
  $("#rate").textContent = fmt(f.rate, 0);
  $("#rate-unit").textContent = f.rate_unit;
  $("#xval").textContent = fmt(f.x, f.x_unit === "ps" ? 1 : 0);
  $("#x-unit").textContent = f.x_unit;
}

function onStatus(msg) {
  if ("busy" in msg) {
    busy = msg.busy;
    $("#busy").hidden = !busy;
    $("#busy span").textContent = busy ?? "";
  }
  if ("background" in msg) {
    $("#warm").hidden = !msg.background;
    $("#warm span:last-child").textContent = msg.background ?? "";
    $("#warm").title = msg.background ?? "";
  }
}

let hush = false; // a quiet command's hand-over toast is dropped: the reset that follows undoes it
function onEvent(msg) {
  if (msg.kind === "takeover") return stopTour();
  if (msg.kind === "swap") activeCharts().forEach((c) => c.mark(msg.text));
  const quiet = msg.kind === "reset" || (msg.kind === "method" && hush);
  if (msg.kind === "reset") activeCharts().forEach((c) => c.clear());
  if (quiet) hush = false;
  else if (!msg.late) toast(msg.text, msg.kind === "error");
}

function toast(text, error = false) {
  const el = document.createElement("div");
  el.className = `toast ${error ? "error" : ""}`;
  el.textContent = text;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), 5000);
}

// ---------------------------------------------------------------- guided tour
/** Whether ``method`` with the MD ``blocks`` has compiled. */
const warm = (method, blocks) => !config.warming.includes(blocks.potential) && !cold(config, blocks)
  && (method === "md" || !config.warming.includes("gcmc"));
/** The tour for the server's potentials: MD on the default one, later a swap to the next. */
const MOVES = ["translation", "rotation", "reinsertion", "exchange"];
function tourSteps({ potentials }) {
  const [home, other] = Object.keys(potentials);
  return [
    { do: [{ type: "method", value: "md", quiet: true }, { type: "reset" }, { type: "blocks", potential: home, thermostat: "none", barostat: false },
      { type: "temperature", value: 300 }],
      say: `MD with ${potentials[home].say} · energy conserved`, wait: 10 },
    { do: [{ type: "blocks", thermostat: "langevin" }],
      say: "+ Langevin thermostat · one block added", wait: 8 },
    { do: [{ type: "method", value: "gcmc" }, { type: "temperature", value: 298 }, { type: "pressure", value: 1 },
      ...MOVES.map((move) => ({ type: "moves", move, value: true }))],
      say: "Switch to GCMC · same structure, 1 bar", wait: 14 },
    { do: [{ type: "pressure", value: 10 }],
      say: "10 bar · no recompile, pores fill", wait: 14 },
    { do: [{ type: "method", value: "hybrid" }, { type: "blocks", barostat: true }],
      say: "Hybrid · GCMC and MD take turns", wait: 40 },
    { do: [{ type: "method", value: "md" }, { type: "blocks", potential: other }], if: () => other && !potentials[other].unavailable,
      say: `${potentials[other]?.say} · same run`, wait: 14 },
    { do: [{ type: "blocks", potential: home, barostat: false }], if: () => config.blocks?.potential !== home,
      say: `Back to ${potentials[home].say} · mid-trajectory`, wait: 10 },
  ];
}
let touring = false;
let tourRun = 0;

async function runTour() {
  if (!config && !replay) return;
  touring = true;
  $("#tour").classList.add("active");
  $("#tour").textContent = "Stop tour";
  const run = ++tourRun; // a stopped run must not resume when a new one starts
  if (replay) return replay.seek(0, true); // the recording carries the captions
  const live = () => touring && run === tourRun;
  const steps = tourSteps(config);
  let method = config.method, blocks = { ...config.blocks };
  for (let k = 0; live(); k = (k + 1) % steps.length) {
    const step = steps[k];
    if (step.if && !step.if()) continue;
    let to = method;
    const next = { ...blocks };
    for (const { type, value, ...fields } of step.do) {
      if (type === "method") to = value;
      if (type === "blocks") Object.assign(next, fields);
    }
    if (to === "hybrid" && next.thermostat === "none") next.thermostat = "langevin";
    while (live() && !(isOpen() && warm(to, next) && !isPaused(config))) await sleep(250); // wait out compiles, reconnects and pauses
    if (!live()) break;
    const epoch = connects;
    [method, blocks] = [to, next];
    step.do.forEach((cmd, i) => send({ ...cmd, tour: true, say: i ? undefined : step.say }));
    caption(step.say);
    await sleep(400);
    while (touring && busy) await sleep(250);
    for (let left = step.wait * 1000; live() && left > 0; left -= isPaused(config) ? 0 : 250) await sleep(250);
    if (connects !== epoch) k--; // the commands may have died with the socket: repeat this step
  }
}
function stopTour() {
  touring = false;
  $("#tour").classList.remove("active");
  $("#tour").textContent = "Play tour";
  caption(null);
}
function caption(text) {
  $("#caption").textContent = text ?? "";
  $("#caption").hidden = !text;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
$("#tour").addEventListener("click", (e) => { e.stopPropagation(); touring ? stopTour() : runTour(); });

// Using a control takes over from the tour; pausing or looking around the structure does not.
$(".build").addEventListener("pointerdown", (e) => {
  if (touring && e.target.closest("button, input") && !e.target.closest("#tour, #pause")) stopTour();
}, { capture: true });

// ---------------------------------------------------------------- replay
/** Plays a recorded tour through the live handlers, in a loop; ``events`` are ``[t, "msg" | "cmd", body]``. */
class Player {
  constructor({ duration, events }) {
    Object.assign(this, { duration, events, i: 0, origin: 0, timer: null, pausedAt: null });
  }
  now() { return (performance.now() - this.origin) / 1000; }
  /** Continue from event ``i`` with the scene, phase, status, numbers and caption it had then; a pause holds unless ``play``. */
  seek(i, play = false) {
    clearTimeout(this.timer);
    for (const method in chartSets) delete chartSets[method];
    config = null;
    let cfg = null, frame = null, say = null;
    const status = { type: "status" };
    for (const [, kind, body] of this.events.slice(0, i)) {
      if (kind === "cmd") say = body.say ?? say;
      else if (body.type === "config") cfg = body;
      else if (body.type === "frame") frame = body;
      else if (body.type === "status") Object.assign(status, body);
    }
    if (frame?.method !== cfg?.method) { // at a chapter start: show its first frame
      frame = this.events.slice(i).find(([, kind, body]) => kind === "msg" && body.type === "frame" && body.method === cfg?.method)?.[2];
    }
    this.i = i;
    if (!play && this.pausedAt != null) this.pausedAt = this.events[i][0];
    for (const msg of [cfg, status, frame]) if (msg) dispatch(msg);
    caption(touring ? say : null);
    if (play || this.pausedAt == null) this.resume(this.events[i][0]);
    else this.show();
  }
  pause() {
    clearTimeout(this.timer);
    this.pausedAt = this.now();
    this.show();
  }
  resume(t = this.pausedAt) {
    this.pausedAt = null;
    this.origin = performance.now() - 1000 * t;
    this.show();
    this.tick();
  }
  show() {
    if (config) { renderControls(config); renderHud(config); }
  }
  /** Play every event that is due; after a long stall (a hidden tab) only the frames the charts can still show. */
  tick() {
    let end = this.i;
    const frames = [];
    for (; end < this.events.length && this.events[end][0] <= this.now(); end++) {
      if (this.events[end][2].type === "frame") frames.push(end);
    }
    const first = frames.at(-CHART_POINTS) ?? 0;
    for (; this.i < end; this.i++) {
      const [, kind, body] = this.events[this.i];
      if (kind === "cmd") {
        if (touring && body.say) caption(body.say);
        hush ||= Boolean(body.quiet);
      } else if (body.type !== "frame" || this.i >= first) {
        dispatch(this.now() - this.events[this.i][0] > 2 ? { ...body, late: true } : body); // no stale toasts after a stall
      }
    }
    const next = this.events[this.i]?.[0] ?? this.duration;
    this.timer = setTimeout(() => (this.i < this.events.length ? this.tick() : this.seek(0)), 1000 * (next - this.now()));
  }
  /** Just past the first config, or the first config of ``method``; 0 when there is none. */
  start(method) {
    return 1 + this.events.findIndex(([, kind, body]) => kind === "msg" && body.type === "config" && (!method || body.method === method));
  }
  /** Method buttons jump to that method's chapter, Reset to the start; the rest needs the live server. */
  command(msg) {
    if (msg.type === "method" && this.start(msg.value)) this.seek(this.start(msg.value));
    else if (msg.type === "reset") this.seek(this.start());
    else if (msg.type === "pause") this.pausedAt == null ? this.pause() : this.resume();
    else {
      toast("Replay · controls need the live server");
      document.activeElement?.blur(); // snap the controls back to what the recording runs
      if (config) renderControls(config);
    }
  }
}

if (bundled || new URLSearchParams(location.search).has("replay")) startReplay();
else {
  fallback = setTimeout(fallBack, 4000);
  connect();
}
