/**
 * main.js — application controller.
 *
 * Screens are plain sections toggled by class; all state lives in the small
 * `state` object below. The camera runs continuously while a capture screen is
 * active and is stopped the moment it is not, because holding a camera open
 * drains a phone battery fast.
 *
 * Accuracy note: detection runs on a downscaled copy of the frame for speed, but
 * the QR corners that define the metric scale are taken at the highest
 * resolution the browser will give us. Corner precision is the dominant error
 * term in every measurement, so it is the one thing worth spending pixels on.
 */

import { detectMarkers } from './lib/aruco.js';
import { ARUCO_DICTS } from './aruco-dict.js';
import {
  makeBoardModel, detectCharucoCorners, charucoViewToCalibration, isViewUsable,
} from './lib/charuco.js';
import { calibrateCamera, horizontalFovDeg } from './lib/calibrate.js';
import {
  laplacianVariance, makeSharpnessGate, makeCoverageTracker, recordView,
  coverageFraction, suggestNextRegion, noveltyScore, assessFrame,
} from './lib/capture.js';
import { detectQrCodes, qrIsUsable, qrApparentSizePx } from './lib/qr.js';
import { buildSurfaces, tapToPoint, segmentLengths, polylineLength, surfaceDistanceMm, closedLoopDimensions } from './lib/measure.js';
import { snapToEdge } from './lib/subpix.js';
import { probeCapabilities, chooseScaleSource, describeLadder } from './lib/capabilities.js';
import { distance3D } from './lib/camera.js';
import * as store from './lib/store.js';

const MAX_PROC_WIDTH = 1280;

const state = {
  caps: null,
  calibration: null,
  boardModel: null,
  stream: null,
  activeScreen: 'home',
  // calibration capture
  calViews: [],
  calResult: null,
  // measurement
  surfaces: [],
  points: [],
  qrDetections: [],
  frozen: false,
  frozenFrame: null,
  lastFrame: null,
  dragging: null,
  procSize: { width: 0, height: 0 },
  displayRect: null,
  busy: false,
};

const $ = (id) => document.getElementById(id);

function showError(msg) {
  const bar = $('global-error');
  bar.textContent = msg;
  bar.hidden = false;
  clearTimeout(showError._t);
  showError._t = setTimeout(() => { bar.hidden = true; }, 6000);
}

function toast(el, msg, ms = 1800) {
  if (!el) return;
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, ms);
}

/* ------------------------------------------------------------------ camera */

/**
 * Report everything a browser checks before it will hand over a camera.
 *
 * Camera access fails for several unrelated reasons that all surface as one
 * unhelpful message, and the fixes are completely different:
 *
 *   not a secure origin   -> serve over HTTPS
 *   inside a sandboxed
 *   iframe                -> opaque origin; "invalid security origin".
 *                            The host must serve the page top-level.
 *   permissions policy    -> the embedding page must allow="camera"
 *   user denied           -> browser setting
 *
 * Printing the actual values turns a guessing game into a one-line answer.
 */
function cameraEnvironmentReport() {
  const md = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  let framed = 'no';
  try {
    if (window.top !== window.self) framed = 'YES';
  } catch {
    framed = 'YES (cross-origin parent)';
  }

  let origin = 'unknown';
  try {
    origin = window.origin || '(opaque)';
  } catch {
    origin = '(opaque)';
  }

  return [
    `origin=${origin}`,
    `secure=${typeof window.isSecureContext === 'boolean' ? window.isSecureContext : '?'}`,
    `framed=${framed}`,
    `mediaDevices=${md}`,
  ].join('  ');
}

/** True when the page is embedded, which is the one case we cannot fix in code. */
function isFramed() {
  try {
    return window.top !== window.self;
  } catch {
    return true;
  }
}

async function startCamera(videoEl) {
  stopCamera();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(`This browser exposes no camera API. ${cameraEnvironmentReport()}`);
  }
  // Ask for the largest frame the device will negotiate: resolution is the
  // single biggest lever on measurement precision.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 4096 },
      height: { ideal: 2160 },
    },
    audio: false,
  });
  state.stream = stream;
  videoEl.srcObject = stream;
  await videoEl.play().catch(() => {});
  await new Promise((resolve) => {
    if (videoEl.videoWidth) { resolve(); return; }
    videoEl.onloadedmetadata = () => resolve();
    setTimeout(resolve, 2500);
  });
  return stream;
}

/** Turn a raw getUserMedia failure into a message that names the actual cause. */
function explainCameraError(e) {
  const raw = (e && e.message) || String(e);
  const env = cameraEnvironmentReport();

  if (/security origin/i.test(raw) && isFramed()) {
    return [
      'The camera is blocked because this page is embedded in a frame, which',
      'gives it an opaque origin. Hosts that preview sites in a sandboxed iframe',
      'cannot grant camera access.',
      '',
      'Fix: open the site at its own top-level URL, or host it somewhere that',
      'serves the files directly (e.g. GitLab Pages).',
      '',
      env,
    ].join(' ');
  }
  if (/security origin/i.test(raw) && !window.isSecureContext) {
    return `The page is not on a secure origin, so the camera is blocked. ${env}`;
  }
  if (/permission/i.test(raw)) {
    return `Camera permission was refused. Allow camera access for this site in the browser settings, then reload. ${env}`;
  }
  if (/not found|readable|track/i.test(raw)) {
    return `No usable camera track. Another app may be holding the camera. ${env}`;
  }
  return `${raw} — ${env}`;
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
}

/** Grab the current frame as a grey buffer, downscaled for processing. */
function grabGrey(videoEl) {
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  if (!vw || !vh) return null;

  const scale = Math.min(1, MAX_PROC_WIDTH / vw);
  const w = Math.max(1, Math.round(vw * scale));
  const h = Math.max(1, Math.round(vh * scale));

  const c = grabGrey._canvas || (grabGrey._canvas = document.createElement('canvas'));
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, w, h);

  const img = ctx.getImageData(0, 0, w, h);
  const grey = new Uint8ClampedArray(w * h);
  const d = img.data;
  for (let i = 0, p = 0; i < grey.length; i++, p += 4) {
    grey[i] = (d[p] * 299 + d[p + 1] * 587 + d[p + 2] * 114) / 1000;
  }
  state.procSize = { width: w, height: h };
  return { data: grey, width: w, height: h };
}

/**
 * Map a pointer position to processing-image coordinates.
 *
 * The video is letterboxed inside its container by object-fit: contain, so the
 * displayed picture does not fill the element. Treating the element as the image
 * would skew every tap, and the error grows toward the edges.
 */
function pointerToImage(clientX, clientY, el, imgW, imgH) {
  const rect = el.getBoundingClientRect();
  const scale = Math.min(rect.width / imgW, rect.height / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;
  const offX = rect.left + (rect.width - drawW) / 2;
  const offY = rect.top + (rect.height - drawH) / 2;

  const x = (clientX - offX) / scale;
  const y = (clientY - offY) / scale;
  return [Math.max(0, Math.min(imgW, x)), Math.max(0, Math.min(imgH, y))];
}

/** Draw an image-space point on the overlay canvas. */
function imageToCanvas(p, canvas, imgW, imgH) {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / imgW, rect.height / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;
  const offX = (rect.width - drawW) / 2;
  const offY = (rect.height - drawH) / 2;
  return [offX + p[0] * scale, offY + p[1] * scale];
}

function fitOverlay(canvas, imgW, imgH) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height, imgW, imgH };
}

function clearOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/* --------------------------------------------------------------- routing */

const SCREENS = ['home', 'calibrate', 'measure', 'diagnostics'];

async function go(name) {
  state.activeScreen = name;
  for (const s of SCREENS) {
    const el = $(`screen-${s}`);
    if (el) el.classList.toggle('active', s === name);
  }
  $('btn-back').hidden = name === 'home';
  $('title').textContent = {
    home: 'Parcel Measure',
    calibrate: 'Calibrate',
    measure: 'Measure',
    diagnostics: 'Diagnostics',
  }[name] || 'Parcel Measure';

  if (name !== 'calibrate' && name !== 'measure') stopCamera();
  if (name !== 'calibrate') stopCalLoop();

  if (name === 'calibrate') await enterCalibrate();
  if (name === 'measure') await enterMeasure();
  if (name === 'diagnostics') await renderDiagnostics();
  if (name === 'home') await renderHome();
}

/* ------------------------------------------------------------------- home */

async function renderHome() {
  state.calibration = await store.loadLatestCalibration();

  const status = $('cal-status');
  const summary = $('cal-summary');
  const badge = $('badge');

  if (!state.calibration) {
    status.textContent = 'No calibration yet. You need one before measuring.';
    summary.hidden = true;
    badge.hidden = false;
    badge.textContent = 'uncalibrated';
    badge.className = 'badge bad';
    $('btn-export-cal').hidden = true;
    $('btn-go-measure').disabled = true;
    $('measure-blocked').hidden = false;
  } else {
    const c = state.calibration;
    status.textContent = `Calibrated ${String(c.createdAt).slice(0, 10)} from ${c.viewCount} views.`;
    summary.hidden = false;
    summary.innerHTML = `
      <dt>RMS error</dt><dd>${Number(c.rms).toFixed(3)} px</dd>
      <dt>Resolution</dt><dd>${c.imageSize[0]} × ${c.imageSize[1]}</dd>
      <dt>Focal length</dt><dd>${Math.round(c.K[0][0])} px</dd>
      <dt>Field of view</dt><dd>${horizontalFovDeg({
        K: c.K, imageSize: c.imageSize,
      }).toFixed(1)}°</dd>
      <dt>Device</dt><dd>${c.deviceLabel || 'unknown'}</dd>`;
    badge.hidden = false;
    badge.textContent = Number(c.rms) < 0.5 ? 'calibrated' : 'check RMS';
    badge.className = `badge ${Number(c.rms) < 0.5 ? 'ok' : 'bad'}`;
    $('btn-export-cal').hidden = false;
    $('btn-go-measure').disabled = false;
    $('measure-blocked').hidden = true;
  }

  // Rebuild the board model whenever the calibration changes.
  state.boardModel = null;

  await renderPrintables();
}

async function renderPrintables() {
  const list = $('printables-list');
  const known = [
    ['printables/charuco_A4_8x6_23.5mm.pdf', 'ChArUco board (A4, 23.5 mm squares)'],
    ['printables/charuco_A4_8x6_40.0mm.pdf', 'ChArUco board (A4 tiled, 40 mm squares)'],
    ['printables/qr_sheet_A4_30-50-80mm.pdf', 'QR stickers (30 / 50 / 80 mm)'],
  ];
  list.innerHTML = '';
  for (const [href, label] of known) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = href;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = label;
    li.appendChild(a);
    const span = document.createElement('span');
    span.className = 'muted';
    span.textContent = 'PDF';
    li.appendChild(span);
    list.appendChild(li);
  }
}

/* -------------------------------------------------------------- calibrate */

const CAL_GRID = 4;
const CAL_TARGET_VIEWS = 16;

let calTimer = null;
const calState = {
  tracker: null,
  sharpness: makeSharpnessGate(),
  autoCapture: true,
  lastGuidance: '',
  busy: false,
  cooldownUntil: 0,
};

async function ensureBoardModel() {
  if (state.boardModel) return;
  try {
    const res = await fetch('printables/charuco_A4_8x6_23.5mm.board.json');
    const spec = await res.json();
    state.boardModel = makeBoardModel(spec);
    state.boardSpec = spec;
  } catch {
    state.boardModel = makeBoardModel({
      squaresX: 8, squaresY: 6, squareLengthMm: 23.5, markerLengthMm: 11.75,
    });
    state.boardSpec = { dictionary: 'DICT_4X4_50' };
  }
}

function buildCoverageGrid(n) {
  const el = $('cal-coverage');
  el.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  el.style.gridTemplateRows = `repeat(${n}, 1fr)`;
  el.innerHTML = '';
  for (let i = 0; i < n * n; i++) {
    const d = document.createElement('div');
    d.className = 'cell';
    el.appendChild(d);
  }
}

function renderCoverage() {
  const t = calState.tracker;
  if (!t) return;
  const cells = $('cal-coverage').children;
  const next = suggestNextRegion(t);

  for (let i = 0; i < cells.length; i++) {
    const gy = Math.floor(i / t.gridN);
    const gx = i % t.gridN;
    cells[i].classList.toggle('hit', t.cells[i] > 0);
    cells[i].classList.toggle('next', !!next && next.gx === gx && next.gy === gy);
  }
  $('cal-cov').textContent = `coverage ${Math.round(coverageFraction(t) * 100)}%`;
}

function setGuidance(text, kind = '') {
  if (text === calState.lastGuidance) return;
  calState.lastGuidance = text;
  const el = $('cal-guidance');
  el.textContent = text;
  el.className = `guidance ${kind}`;
}

function updateCalUi() {
  const n = state.calViews.length;
  $('cal-count').textContent = `${n} view${n === 1 ? '' : 's'}`;
  $('btn-cal-solve').disabled = n < 6;
}

async function enterCalibrate() {
  try {
    await startCamera($('cal-video'));
  } catch (e) {
    showError(explainCameraError(e));
    return;
  }
  await ensureBoardModel();
  calState.tracker = null; // created on the first frame, once the size is known
  calState.lastGuidance = '';
  buildCoverageGrid(CAL_GRID);
  updateCalUi();
  startCalLoop();
}

function startCalLoop() {
  stopCalLoop();
  // ~4 fps: fast enough to feel live, slow enough to leave the phone's CPU for
  // the detection itself, which is the expensive part.
  calTimer = setInterval(calTick, 240);
  calTick();
}

function stopCalLoop() {
  if (calTimer) { clearInterval(calTimer); calTimer = null; }
}

async function calTick() {
  if (state.activeScreen !== 'calibrate' || calState.busy) return;
  calState.busy = true;

  try {
    const frame = grabGrey($('cal-video'));
    if (!frame) return;

    if (!calState.tracker) {
      calState.tracker = makeCoverageTracker([frame.width, frame.height], CAL_GRID);
    }

    const dict = ARUCO_DICTS[state.boardSpec?.dictionary || 'DICT_4X4_50'];
    if (!dict) { showError(`Dictionary ${state.boardSpec?.dictionary} is not bundled.`); return; }

    const sharp = calState.sharpness.test(
      laplacianVariance(frame.data, frame.width, frame.height)
    );
    $('cal-sharp').textContent = `sharp ${Math.round(sharp.value)}`;

    const markers = detectMarkers(frame.data, frame.width, frame.height, dict);
    if (markers.length < 4) {
      setGuidance('Show the whole board', '');
      return;
    }

    const corners = detectCharucoCorners(frame.data, frame.width, frame.height, markers, state.boardModel);
    const usable = corners.filter((c) => c.imagePoint).map((c) => c.imagePoint);

    if (!isViewUsable(corners, state.boardModel)) {
      setGuidance(`${usable.length} corners — get the whole board in frame, or hold it flatter`, 'warn');
      return;
    }

    const novelty = noveltyScore(calState.tracker, usable);
    const verdict = assessFrame({
      cornerCount: usable.length,
      sharpness: sharp,
      novelty,
    });

    const enough = state.calViews.length >= CAL_TARGET_VIEWS;

    if (verdict.accept && !enough && calState.autoCapture && Date.now() > calState.cooldownUntil) {
      commitCalView(corners, usable);
      setGuidance(`Captured — ${state.calViews.length}/${CAL_TARGET_VIEWS}`, 'ok');
      calState.cooldownUntil = Date.now() + 700;
      return;
    }

    if (enough) {
      setGuidance('Enough views — press Solve', 'ok');
      return;
    }

    if (!verdict.accept) {
      setGuidance(verdict.guidance, 'warn');
      return;
    }

    const next = suggestNextRegion(calState.tracker);
    setGuidance(
      next ? `${next.instruction}${calState.autoCapture ? '' : ' — then tap Capture'}` : 'Good — keep sweeping',
      'ok'
    );
  } catch (e) {
    setGuidance(`Detection error: ${e.message}`, 'warn');
  } finally {
    calState.busy = false;
  }
}

function commitCalView(corners, usablePixels) {
  state.calViews.push(charucoViewToCalibration(corners, state.boardModel));
  recordView(calState.tracker, usablePixels);
  renderCoverage();
  updateCalUi();
}

/** Manual capture: bypasses the novelty and sharpness gates on purpose. */
function captureCalibrationView() {
  const video = $('cal-video');
  const frame = grabGrey(video);
  if (!frame) { toast($('cal-toast'), 'No frame yet'); return; }

  const dict = ARUCO_DICTS[state.boardSpec?.dictionary || 'DICT_4X4_50'];
  if (!dict) { showError(`Dictionary ${state.boardSpec?.dictionary} is not bundled.`); return; }

  const markers = detectMarkers(frame.data, frame.width, frame.height, dict);
  if (markers.length < 4) {
    toast($('cal-toast'), `Only ${markers.length} markers — get the whole board in frame`);
    return;
  }
  const corners = detectCharucoCorners(frame.data, frame.width, frame.height, markers, state.boardModel);
  if (!isViewUsable(corners, state.boardModel)) {
    toast($('cal-toast'), `Only ${corners.length} corners — too few or too edge-on`);
    return;
  }
  if (!calState.tracker) {
    calState.tracker = makeCoverageTracker([frame.width, frame.height], CAL_GRID);
  }
  const usable = corners.filter((c) => c.imagePoint).map((c) => c.imagePoint);
  commitCalView(corners, usable);
  toast($('cal-toast'), `Captured ${corners.length} corners`);
}

function solveCalibration() {
  const video = $('cal-video');
  const size = [video.videoWidth, video.videoHeight];
  if (!size[0]) { showError('No frame size available.'); return; }

  try {
    const result = calibrateCamera(state.calViews, size, { maxIter: 80 });
    state.calResult = result;

    const kv = $('cal-result-kv');
    kv.innerHTML = `
      <dt>RMS error</dt><dd>${result.rms.toFixed(3)} px</dd>
      <dt>Views used</dt><dd>${result.viewCount}</dd>
      <dt>Points</dt><dd>${result.totalPoints}</dd>
      <dt>fx / fy</dt><dd>${result.K[0][0].toFixed(0)} / ${result.K[1][1].toFixed(0)} px</dd>
      <dt>Principal point</dt><dd>${result.K[0][2].toFixed(0)}, ${result.K[1][2].toFixed(0)}</dd>
      <dt>k1 / k2</dt><dd>${result.dist[0].toFixed(4)} / ${result.dist[1].toFixed(4)}</dd>
      <dt>Field of view</dt><dd>${horizontalFovDeg(result).toFixed(1)}°</dd>`;

    const warnBox = $('cal-warnings');
    if (result.warnings.length) {
      warnBox.hidden = false;
      warnBox.innerHTML = `<ul>${result.warnings.map((w) => `<li>${w}</li>`).join('')}</ul>`;
    } else {
      warnBox.hidden = true;
    }

    $('cal-result').hidden = false;
    $('cal-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    showError(`Calibration failed: ${e.message}`);
  }
}

async function saveCalibration() {
  if (!state.calResult) return;
  try {
    const video = $('cal-video');
    const record = await store.saveCalibration({
      name: 'phone',
      deviceLabel: state.caps?.deviceLabel || null,
      imageSize: state.calResult.imageSize,
      K: state.calResult.K,
      dist: state.calResult.dist,
      rms: state.calResult.rms,
      viewCount: state.calResult.viewCount,
      totalPoints: state.calResult.totalPoints,
      warnings: state.calResult.warnings,
    });
    void video;
    await store.requestPersistence();
    toast($('cal-toast'), 'Saved');
    state.calViews = [];
    state.calResult = null;
    $('cal-result').hidden = true;
    updateCalUi();
    await go('home');
    void record;
  } catch (e) {
    showError(`Could not save: ${e.message}`);
  }
}

/* ---------------------------------------------------------------- measure */

async function enterMeasure() {
  if (!state.calibration) { showError('Calibrate first.'); return; }
  try {
    await startCamera($('me-video'));
  } catch (e) {
    showError(explainCameraError(e));
    return;
  }
  state.points = [];
  state.surfaces = [];
  state.frozen = false;
  state.frozenFrame = null;
  $('btn-me-lock').textContent = 'Freeze';
  startMeasureLoop();
}

let measureTimer = null;

function startMeasureLoop() {
  stopMeasureLoop();
  measureTimer = setInterval(measureTick, 320);
  measureTick();
}

function stopMeasureLoop() {
  if (measureTimer) { clearInterval(measureTimer); measureTimer = null; }
}

async function measureTick() {
  if (state.activeScreen !== 'measure' || state.busy) return;
  state.busy = true;
  try {
    const video = $('me-video');
    const frame = state.frozen && state.frozenFrame
      ? state.frozenFrame
      : grabGrey(video);
    if (!frame) return;
    if (state.frozen && !state.frozenFrame) state.frozenFrame = frame;
    // Keep the latest frame so a tap can snap to the object's edges without
    // grabbing a fresh one mid-gesture.
    state.lastFrame = frame;

    const sizeMap = await store.getQrSizeMap();
    const defaultSizeMm = Number(await store.getMeta('defaultQrSizeMm', 0)) || null;
    const qrs = await detectQrCodes(frame.data, frame.width, frame.height, {
      sizeRegistry: sizeMap,
      defaultSizeMm,
    });

    state.qrDetections = qrs;
    const usable = qrs.filter((q) => qrIsUsable(q));

    if (usable.length && state.calibration) {
      const cal = state.calibration;
      state.surfaces = buildSurfaces(usable, cal.K, cal.dist);
    } else {
      state.surfaces = [];
    }

    updateMeasureHud(usable);
    redrawMeasureOverlay();
  } catch (e) {
    showError(`Detection error: ${e.message}`);
  } finally {
    state.busy = false;
  }
}

function updateMeasureHud(usable) {
  const scalePill = $('me-scale');
  const precPill = $('me-precision');

  if (state.surfaces.length) {
    const merged = state.surfaces.filter((s) => s.mergedCount > 1).length;
    scalePill.textContent = merged
      ? `${state.surfaces.length} surface(s), ${merged} merged`
      : `${state.surfaces.length} surface(s)`;
    scalePill.className = 'pill ok';

    const s = state.surfaces[0];
    const dist = surfaceDistanceMm(s);
    const px = qrApparentSizePx({ corners: s.qr.corners });
    const mmPerPx = s.qr.sizeMm / px;
    // Three pixels is a realistic fingertip placement error once the loupe is
    // in use; the number tells the user whether to move closer.
    precPill.textContent = `±${(3 * mmPerPx).toFixed(1)} mm @ ${Math.round(dist)} mm`;
  } else if (state.qrDetections.length) {
    scalePill.textContent = 'code seen, size unknown';
    scalePill.className = 'pill warn';
    precPill.textContent = 'set its size in Diagnostics';
  } else {
    scalePill.textContent = 'no scale';
    scalePill.className = 'pill';
    precPill.textContent = 'point at a QR sticker';
  }
  void usable;
}

function redrawMeasureOverlay() {
  const canvas = $('me-overlay');
  const { width: iw, height: ih } = state.procSize;
  if (!iw) return;
  const { ctx, w, h } = fitOverlay(canvas, iw, ih);

  const toC = (p) => imageToCanvas(p, canvas, iw, ih);

  // QR quads
  for (const qr of state.qrDetections) {
    const usable = qrIsUsable(qr);
    ctx.strokeStyle = usable ? 'rgba(62,207,142,.95)' : 'rgba(245,166,35,.95)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    qr.corners.forEach((p, i) => {
      const [x, y] = toC(p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    const [cx, cy] = toC(qr.corners.reduce(
      (a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]
    ));
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.fillText(qr.id ? `${qr.id}${usable ? ` · ${qr.sizeMm}mm` : ''}` : 'unreadable', cx + 6, cy - 6);
  }

  // Surface plane normals, drawn from the sticker centre so the user can see
  // which faces the app has locked on to.
  ctx.strokeStyle = 'rgba(238,77,45,.85)';
  ctx.lineWidth = 1.5;
  for (const s of state.surfaces) {
    const t = s.pose.t;
    const n = s.plane.normal;
    const c0 = toC(projectToImage(s, t));
    const c1 = toC(projectToImage(s, [
      t[0] + n[0] * 60, t[1] + n[1] * 60, t[2] + n[2] * 60,
    ]));
    ctx.beginPath();
    ctx.moveTo(c0[0], c0[1]);
    ctx.lineTo(c1[0], c1[1]);
    ctx.stroke();
  }

  // Measurement points and the outline through them. With three or more points
  // the loop is closed and shaded, so it reads as a face rather than a path.
  if (state.points.length) {
    const closed = state.points.length >= 3;

    ctx.beginPath();
    state.points.forEach((p, i) => {
      const [x, y] = toC(p.pixel);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    if (closed) {
      const [x, y] = toC(state.points[0].pixel);
      ctx.lineTo(x, y);
      ctx.closePath();
      ctx.fillStyle = 'rgba(238,77,45,.18)';
      ctx.fill();
    }
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Edge lengths, drawn at each edge midpoint so the outline carries its own
    // dimensions without needing the readout panel.
    if (closed) {
      const pts3D = state.points.filter((p) => p.point3D).map((p) => p.point3D);
      if (pts3D.length === state.points.length && pts3D.length >= 3) {
        const dims = closedLoopDimensions(pts3D);
        ctx.font = '600 11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        for (let i = 0; i < state.points.length; i++) {
          const a = toC(state.points[i].pixel);
          const b = toC(state.points[(i + 1) % state.points.length].pixel);
          const mx = (a[0] + b[0]) / 2;
          const my = (a[1] + b[1]) / 2;
          const label = `${dims.edges[i].toFixed(0)}`;
          const wpx = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0,0,0,.7)';
          ctx.fillRect(mx - wpx / 2 - 3, my - 8, wpx + 6, 15);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, mx, my + 3);
        }
        ctx.textAlign = 'left';
      }
    }

    state.points.forEach((p, i) => {
      const [x, y] = toC(p.pixel);
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = p.point3D ? 'rgba(238,77,45,.95)' : 'rgba(255,92,92,.95)';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      // A ring marks a point that was pulled onto an edge, so the user can tell
      // at a glance which placements the app adjusted.
      if (p.snapped) {
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(62,207,142,.95)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.fillStyle = '#fff';
      ctx.font = '700 12px system-ui, sans-serif';
      ctx.fillText(String(i + 1), x + 11, y + 4);
    });
  }
  void w; void h;
}

/** Reproject a camera-space point back to image pixels, for drawing. */
function projectToImage(surface, camPoint) {
  const cal = state.calibration;
  const R = surface.pose.R;
  const t = surface.pose.t;
  // camera -> surface frame
  const d = [camPoint[0] - t[0], camPoint[1] - t[1], camPoint[2] - t[2]];
  const local = [
    R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2],
    R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2],
    R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2],
  ];
  const z = local[2] || 1e-6;
  const x = local[0] / z;
  const y = local[1] / z;
  void cal;
  // Approximate: scale by the frame's focal length in processing pixels.
  const fx = state.calibration.K[0][0] * (state.procSize.width / state.calibration.imageSize[0]);
  const fy = state.calibration.K[1][1] * (state.procSize.height / state.calibration.imageSize[1]);
  const cx = state.calibration.K[0][2] * (state.procSize.width / state.calibration.imageSize[0]);
  const cy = state.calibration.K[1][2] * (state.procSize.height / state.calibration.imageSize[1]);
  return [fx * x + cx, fy * y + cy];
}

function commitPoint(pixel) {
  if (!state.surfaces.length) {
    showError('No QR sticker in view — nothing establishes a real-world length.');
    return;
  }
  const cal = state.calibration;

  // Pull the tap onto the object's own edge before measuring. A fingertip lands
  // several pixels out, and on a parcel that error IS the measurement.
  let placed = pixel;
  let snapped = false;
  const frame = state.frozen && state.frozenFrame ? state.frozenFrame : state.lastFrame;
  if (frame) {
    const snap = snapToEdge(frame.data, frame.width, frame.height, pixel, 9);
    if (snap.snapped) { placed = snap.point; snapped = true; }
  }

  const hit = tapToPoint(placed, state.surfaces, cal.K, cal.dist);
  if (!hit) {
    showError('That tap does not land on a known surface.');
    return;
  }
  if (!hit.onPlane) {
    toast($('me-toast'), 'Far from the sticker — result may be unreliable');
  }
  state.points.push({ pixel: placed, point3D: hit.point, surface: hit.surface, snapped });
  updateMeasurementReadout();
  redrawMeasureOverlay();
}

function updateMeasurementReadout() {
  const pts = state.points.filter((p) => p.point3D);
  const valueEl = document.querySelector('#me-readout .readout-value');
  const segEl = $('me-segments');

  if (pts.length < 2) {
    valueEl.textContent = '—';
    segEl.textContent = pts.length === 1
      ? 'Tap a second point — or three more to outline a face'
      : 'Tap two points to measure, or four corners to outline a face';
    return;
  }

  const pts3D = pts.map((p) => p.point3D);

  // Three or more points on one surface reads as an attempt to outline a face,
  // so report the shape rather than just the last segment.
  if (pts.length >= 3) {
    const dims = closedLoopDimensions(pts3D);
    if (dims) {
      valueEl.textContent = dims.perimeter.toFixed(0);
      const parts = dims.edges.map((e, i) => `${i + 1}: ${e.toFixed(1)}`).join('   ');
      let note = `perimeter ${dims.perimeter.toFixed(1)} mm   area ${(dims.area / 100).toFixed(1)} cm²`;
      if (dims.rectangularity !== null && dims.rectangularity < 0.9) {
        note += `   ⚠ opposite edges differ by ${Math.round((1 - dims.rectangularity) * 100)}% — re-check the corners`;
      }
      segEl.textContent = `${parts}\n${note}`;
      return;
    }
  }

  const segs = segmentLengths(pts3D);
  const total = polylineLength(pts3D);
  valueEl.textContent = total.toFixed(1);
  segEl.textContent = segs.length > 1
    ? segs.map((s, i) => `${i + 1}: ${s.toFixed(1)}`).join('   ') + `   total ${total.toFixed(1)} mm`
    : `${distance3D(pts3D[0], pts3D[1]).toFixed(1)} mm`;
}

/* --------------------------------------------------------- loupe dragging */

function showLoupe(frame, pixel, canvas, loupe) {
  // The canvas has a CSS size but its backing store defaults to 300x150, so
  // drawing at "size" without setting width/height squashes the magnification
  // into a 132 px box and makes the loupe useless for aiming.
  const rect = loupe.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssSize = Math.round(rect.width) || 132;
  if (loupe.width !== Math.round(cssSize * dpr)) {
    loupe.width = Math.round(cssSize * dpr);
    loupe.height = Math.round(cssSize * dpr);
  }

  const size = loupe.width;
  const zoom = 6;
  const ctx = loupe.getContext('2d');
  const half = size / (2 * zoom);

  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);

  // Draw from the video element directly so the loupe shows full-resolution
  // detail rather than the downscaled processing copy.
  const video = $('me-video');
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sx = (pixel[0] / frame.width) * vw;
  const sy = (pixel[1] / frame.height) * vh;
  const sw = (2 * half / frame.width) * vw;
  const sh = (2 * half / frame.height) * vh;

  ctx.drawImage(video, sx - sw / 2, sy - sh / 2, sw, sh, 0, 0, size, size);

  // Crosshair
  ctx.strokeStyle = 'rgba(238,77,45,.95)';
  ctx.lineWidth = Math.max(1, dpr);
  ctx.beginPath();
  ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size);
  ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2);
  ctx.stroke();

  loupe.hidden = false;
  void canvas;
}

function onPointerDown(ev) {
  if (state.activeScreen !== 'measure') return;
  const video = $('me-video');
  const frame = state.frozen && state.frozenFrame ? state.frozenFrame : grabGrey(video);
  if (!frame) return;
  const pixel = pointerToImage(ev.clientX, ev.clientY, video, frame.width, frame.height);
  state.dragging = { pixel, frame };
  showLoupe(frame, pixel, $('me-overlay'), $('me-loupe'));
  ev.preventDefault();
}

function onPointerMove(ev) {
  if (!state.dragging || state.activeScreen !== 'measure') return;
  const video = $('me-video');
  const pixel = pointerToImage(ev.clientX, ev.clientY, video, state.dragging.frame.width, state.dragging.frame.height);
  state.dragging.pixel = pixel;
  showLoupe(state.dragging.frame, pixel, $('me-overlay'), $('me-loupe'));
  ev.preventDefault();
}

function onPointerUp(ev) {
  if (!state.dragging) return;
  const { pixel } = state.dragging;
  state.dragging = null;
  $('me-loupe').hidden = true;
  commitPoint(pixel);
  ev.preventDefault();
}

/* ------------------------------------------------------------ diagnostics */

async function renderDiagnostics() {
  const caps = state.caps || await probeCapabilities();
  state.caps = caps;

  const ladder = $('ladder');
  ladder.innerHTML = '';
  for (const t of describeLadder(caps)) {
    const li = document.createElement('li');
    li.innerHTML =
      `<span class="tier-name ${t.available ? '' : 'tier-off'}">` +
      `${t.tier}. ${t.name}${t.available ? '' : ' — unavailable'}</span>` +
      `<span class="tier-detail">${t.detail}</span>`;
    ladder.appendChild(li);
  }

  const kv = $('caps-kv');
  const res = caps.maxCameraResolution;
  let framedNote = 'no';
  try {
    framedNote = window.top === window.self ? 'no' : 'YES — camera will be blocked';
  } catch {
    framedNote = 'YES (cross-origin parent) — camera will be blocked';
  }
  let originNote = 'unknown';
  try { originNote = window.origin || '(opaque)'; } catch { originNote = '(opaque)'; }

  kv.innerHTML = `
    <dt>Origin</dt><dd>${originNote}</dd>
    <dt>Secure context</dt><dd>${caps.secureContext ? 'yes' : 'NO — camera will not work'}</dd>
    <dt>Inside a frame</dt><dd>${framedNote}</dd>
    <dt>Camera</dt><dd>${caps.deviceLabel || 'unknown'}</dd>
    <dt>Max resolution</dt><dd>${res ? `${res.width} × ${res.height}` : 'unknown'}</dd>
    <dt>BarcodeDetector</dt><dd>${caps.barcodeDetector ? 'yes (fast path)' : 'no (using jsQR fallback)'}</dd>
    <dt>Focus distance</dt><dd>${caps.focusDistance ? 'exposed' : 'not exposed'}</dd>
    <dt>Orientation sensor</dt><dd>${caps.deviceOrientation ? 'yes' : 'no'}</dd>
    <dt>Torch</dt><dd>${caps.torch ? 'yes' : 'no'}</dd>`;

  const sizeList = $('qr-size-list');
  const sizes = await store.getQrSizeMap();
  sizeList.innerHTML = sizes.size
    ? [...sizes].map(([id, mm]) => `<li><span>${id}</span><span>${mm} mm</span></li>`).join('')
    : '<li class="muted">None recorded.</li>';

  const defaultMm = Number(await store.getMeta('defaultQrSizeMm', 0)) || 0;
  $('qr-default-mm').value = defaultMm || '';
  $('qr-default-status').textContent = defaultMm
    ? `Any QR without its own size is treated as ${defaultMm} mm.`
    : 'No default set — a plain QR code cannot be used as a ruler yet.';

  const calList = $('cal-list');
  const cals = await store.listCalibrations();
  calList.innerHTML = cals.length
    ? cals.map((c) => `<li><span>${String(c.createdAt).slice(0, 19).replace('T', ' ')}</span>
        <span>${Number(c.rms).toFixed(2)} px · ${c.viewCount} views</span></li>`).join('')
    : '<li class="muted">None.</li>';

  if (caps.errors && caps.errors.length) {
    showError(`Probe warnings: ${caps.errors.join('; ')}`);
  }
}

/* ------------------------------------------------------------------- wire */

function wire() {
  $('btn-back').addEventListener('click', () => go('home'));
  $('btn-go-calibrate').addEventListener('click', () => go('calibrate'));
  $('btn-go-measure').addEventListener('click', () => go('measure'));
  $('btn-go-diagnostics').addEventListener('click', () => go('diagnostics'));

  $('btn-export-cal').addEventListener('click', async () => {
    const cal = await store.loadLatestCalibration();
    if (!cal) return;
    const sizes = await store.getQrSizeMap();
    store.downloadCalibration(cal, Object.fromEntries(sizes));
  });

  $('btn-cal-capture').addEventListener('click', captureCalibrationView);
  $('btn-cal-auto').addEventListener('click', () => {
    calState.autoCapture = !calState.autoCapture;
    const b = $('btn-cal-auto');
    b.textContent = `Auto-capture: ${calState.autoCapture ? 'ON' : 'OFF'}`;
    b.classList.toggle('primary', calState.autoCapture);
  });
  $('btn-cal-solve').addEventListener('click', solveCalibration);
  $('btn-cal-reset').addEventListener('click', () => {
    state.calViews = [];
    state.calResult = null;
    calState.tracker = null;
    calState.sharpness = makeSharpnessGate();
    $('cal-result').hidden = true;
    buildCoverageGrid(CAL_GRID);
    renderCoverage();
    updateCalUi();
    setGuidance('Sweep the board around the frame');
  });
  $('btn-cal-save').addEventListener('click', saveCalibration);
  $('btn-cal-discard').addEventListener('click', () => {
    state.calResult = null;
    $('cal-result').hidden = true;
  });

  const overlay = $('me-overlay');
  overlay.addEventListener('pointerdown', onPointerDown);
  overlay.addEventListener('pointermove', onPointerMove);
  overlay.addEventListener('pointerup', onPointerUp);
  overlay.addEventListener('pointercancel', onPointerUp);

  $('btn-me-undo').addEventListener('click', () => {
    state.points.pop();
    updateMeasurementReadout();
    redrawMeasureOverlay();
  });
  $('btn-me-clear').addEventListener('click', () => {
    state.points = [];
    updateMeasurementReadout();
    redrawMeasureOverlay();
  });
  $('btn-me-lock').addEventListener('click', () => {
    state.frozen = !state.frozen;
    if (state.frozen) {
      state.frozenFrame = grabGrey($('me-video'));
      $('btn-me-lock').textContent = 'Live';
      toast($('me-toast'), 'Frozen — points stay put');
    } else {
      state.frozenFrame = null;
      $('btn-me-lock').textContent = 'Freeze';
    }
  });

  $('btn-probe').addEventListener('click', async () => {
    state.caps = await probeCapabilities();
    await renderDiagnostics();
  });

  $('btn-add-qr-size').addEventListener('click', async () => {
    const id = $('qr-size-id').value.trim();
    const mm = Number($('qr-size-mm').value);
    try {
      await store.setQrSize(id, mm);
      $('qr-size-id').value = '';
      $('qr-size-mm').value = '';
      await renderDiagnostics();
    } catch (e) {
      showError(e.message);
    }
  });

  $('btn-set-qr-default').addEventListener('click', async () => {
    const mm = Number($('qr-default-mm').value);
    if (!Number.isFinite(mm) || mm <= 0) {
      showError('Enter a positive size in millimetres.');
      return;
    }
    await store.setMeta('defaultQrSizeMm', mm);
    await renderDiagnostics();
  });

  $('btn-export-all').addEventListener('click', async () => {
    const cal = await store.loadLatestCalibration();
    if (!cal) { showError('Nothing to export — no calibration stored.'); return; }
    const sizes = await store.getQrSizeMap();
    store.downloadCalibration(cal, Object.fromEntries(sizes));
  });

  $('btn-import').addEventListener('click', () => $('file-import').click());
  $('file-import').addEventListener('change', async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const res = await store.importCalibrationJson(text);
      $('import-msg').textContent =
        `Imported calibration (RMS ${Number(res.calibration.rms).toFixed(2)} px) and ${res.qrSizeCount} sticker size(s).`;
      await renderDiagnostics();
    } catch (e) {
      $('import-msg').textContent = '';
      showError(`Import failed: ${e.message}`);
    } finally {
      ev.target.value = '';
    }
  });

  $('btn-clear-all').addEventListener('click', async () => {
    if (!window.confirm('Erase all calibrations and sticker sizes on this device?')) return;
    await store.clearAll();
    await renderDiagnostics();
    await renderHome();
  });

  window.addEventListener('beforeunload', stopCamera);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopCamera();
    else if (state.activeScreen === 'calibrate' || state.activeScreen === 'measure') {
      startCamera(state.activeScreen === 'calibrate' ? $('cal-video') : $('me-video'))
        .catch((e) => showError(explainCameraError(e)));
    }
  });
}

async function boot() {
  wire();
  try {
    state.caps = await probeCapabilities();
    if (!state.caps.secureContext) {
      showError('This page is not on a secure origin. The camera will not open — serve it over HTTPS or localhost.');
    } else if (isFramed()) {
      // Say this up front rather than letting the user discover it by tapping
      // Calibrate and getting an opaque browser error.
      showError(
        'This page is running inside a frame, which blocks the camera. ' +
        'Open the site at its own top-level URL instead.'
      );
    }
  } catch (e) {
    showError(`Capability probe failed: ${e.message}`);
  }
  await renderHome();
  await go('home');

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
}

boot();
