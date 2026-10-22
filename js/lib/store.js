/**
 * store.js — persistence for calibrations, sticker sizes and settings.
 *
 * Two layers, deliberately:
 *
 *   IndexedDB is the working copy. It is fast and always available, but a
 *   browser may evict it under storage pressure, so it is a cache.
 *
 *   A downloadable JSON file is the durable copy. Losing a calibration means
 *   re-shooting a dozen board views, so the file is what actually protects the
 *   user's work — and it is also how a calibration moves between phones.
 *
 * The schema is versioned and validated on import, because a silently corrupt
 * calibration produces measurements that are wrong but self-consistent, which is
 * the worst failure mode this app has.
 */

const DB_NAME = 'spx-parcel-measure';
const DB_VERSION = 1;
const STORE_CAL = 'calibrations';
const STORE_QR = 'qrSizes';
const STORE_META = 'meta';

export const EXPORT_FORMAT = 'spx-parcel-measure/calibration';
export const EXPORT_SCHEMA = 1;
export const APP_VERSION = '1.0.0';

/* ------------------------------------------------------------ IndexedDB */

function hasIndexedDb() {
  return typeof indexedDB !== 'undefined';
}

function openDb() {
  return new Promise((resolve, reject) => {
    if (!hasIndexedDb()) { reject(new Error('IndexedDB unavailable')); return; }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_CAL)) {
        db.createObjectStore(STORE_CAL, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_QR)) {
        db.createObjectStore(STORE_QR, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_META)) {
        db.createObjectStore(STORE_META, { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const s = t.objectStore(store);
    let result;
    try {
      result = fn(s);
    } catch (e) {
      reject(e);
      return;
    }
    t.oncomplete = () => resolve(result && result.result !== undefined ? result.result : result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

/** In-memory fallback so the module is usable (and testable) without IndexedDB. */
const memory = {
  calibrations: new Map(),
  qrSizes: new Map(),
  meta: new Map(),
};

/* ------------------------------------------------------- calibration IO */

/**
 * Validate a calibration record. Throws with a specific reason rather than
 * letting a malformed one reach the measurement code.
 */
export function validateCalibration(cal) {
  if (!cal || typeof cal !== 'object') throw new Error('calibration is not an object');

  const K = cal.K;
  if (!Array.isArray(K) || K.length !== 3) throw new Error('K must be a 3x3 matrix');
  for (const row of K) {
    if (!Array.isArray(row) || row.length !== 3) throw new Error('K must be a 3x3 matrix');
    for (const v of row) if (!Number.isFinite(v)) throw new Error('K contains a non-finite value');
  }
  if (!(K[0][0] > 0) || !(K[1][1] > 0)) throw new Error('focal length must be positive');

  if (!Array.isArray(cal.dist) || cal.dist.length < 4) {
    throw new Error('dist must have at least 4 coefficients');
  }
  for (const v of cal.dist) if (!Number.isFinite(v)) throw new Error('dist contains a non-finite value');

  if (!Array.isArray(cal.imageSize) || cal.imageSize.length !== 2) {
    throw new Error('imageSize must be [width, height]');
  }
  if (!(cal.imageSize[0] > 0) || !(cal.imageSize[1] > 0)) throw new Error('imageSize must be positive');

  return true;
}

/** Save a calibration. Returns the stored record. */
export async function saveCalibration(cal) {
  validateCalibration(cal);
  const record = {
    ...cal,
    id: cal.id || `cal-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: cal.createdAt || new Date().toISOString(),
    appVersion: APP_VERSION,
  };

  if (!hasIndexedDb()) {
    memory.calibrations.set(record.id, record);
    return record;
  }
  const db = await openDb();
  await tx(db, STORE_CAL, 'readwrite', (s) => s.put(record));
  db.close();
  return record;
}

/** Load the most recent calibration, or null. */
export async function loadLatestCalibration() {
  const all = await listCalibrations();
  if (!all.length) return null;
  all.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  return all[0];
}

/** All stored calibrations, newest first. */
export async function listCalibrations() {
  if (!hasIndexedDb()) return [...memory.calibrations.values()];
  const db = await openDb();
  const all = await new Promise((resolve, reject) => {
    const t = db.transaction(STORE_CAL, 'readonly');
    const req = t.objectStore(STORE_CAL).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
  db.close();
  all.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  return all;
}

/** Delete a calibration by id. */
export async function deleteCalibration(id) {
  if (!hasIndexedDb()) { memory.calibrations.delete(id); return; }
  const db = await openDb();
  await tx(db, STORE_CAL, 'readwrite', (s) => s.delete(id));
  db.close();
}

/* ------------------------------------------------------ QR size registry */

/**
 * Remember the printed size of a sticker whose payload did not carry one.
 * Keyed by the sticker's id, so the user types it once per sticker design.
 */
export async function setQrSize(id, sizeMm) {
  if (!id) throw new Error('setQrSize needs an id');
  if (!Number.isFinite(sizeMm) || sizeMm <= 0) throw new Error('sizeMm must be positive');
  const rec = { id, sizeMm };

  if (!hasIndexedDb()) { memory.qrSizes.set(id, rec); return rec; }
  const db = await openDb();
  await tx(db, STORE_QR, 'readwrite', (s) => s.put(rec));
  db.close();
  return rec;
}

/** All known sticker sizes as a Map of id -> sizeMm. */
export async function getQrSizeMap() {
  let rows;
  if (!hasIndexedDb()) {
    rows = [...memory.qrSizes.values()];
  } else {
    const db = await openDb();
    rows = await new Promise((resolve, reject) => {
      const t = db.transaction(STORE_QR, 'readonly');
      const req = t.objectStore(STORE_QR).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
    db.close();
  }
  const m = new Map();
  for (const r of rows) m.set(r.id, r.sizeMm);
  return m;
}

export async function deleteQrSize(id) {
  if (!hasIndexedDb()) { memory.qrSizes.delete(id); return; }
  const db = await openDb();
  await tx(db, STORE_QR, 'readwrite', (s) => s.delete(id));
  db.close();
}

/* ------------------------------------------------------------- settings */

export async function setMeta(key, value) {
  const rec = { key, value };
  if (!hasIndexedDb()) { memory.meta.set(key, rec); return rec; }
  const db = await openDb();
  await tx(db, STORE_META, 'readwrite', (s) => s.put(rec));
  db.close();
  return rec;
}

export async function getMeta(key, fallback = null) {
  if (!hasIndexedDb()) {
    const r = memory.meta.get(key);
    return r ? r.value : fallback;
  }
  const db = await openDb();
  const rec = await new Promise((resolve, reject) => {
    const t = db.transaction(STORE_META, 'readonly');
    const req = t.objectStore(STORE_META).get(key);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return rec ? rec.value : fallback;
}

/**
 * Ask the browser to keep our data. Without this, IndexedDB is best-effort and
 * can be cleared when the device runs low on space.
 */
export async function requestPersistence() {
  if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.persist) {
    return { supported: false, persisted: false };
  }
  try {
    const already = navigator.storage.persisted ? await navigator.storage.persisted() : false;
    const persisted = already || await navigator.storage.persist();
    return { supported: true, persisted };
  } catch (e) {
    return { supported: true, persisted: false, error: e.message };
  }
}

/* --------------------------------------------------------- export/import */

/** Build the portable JSON document for a calibration. */
export function calibrationToJson(cal, extra = {}) {
  validateCalibration(cal);
  return {
    format: EXPORT_FORMAT,
    schema: EXPORT_SCHEMA,
    appVersion: APP_VERSION,
    exportedUtc: new Date().toISOString(),
    calibration: {
      name: cal.name || 'calibration',
      deviceLabel: cal.deviceLabel || null,
      imageSize: cal.imageSize,
      K: cal.K,
      dist: cal.dist,
      rms: cal.rms ?? null,
      viewCount: cal.viewCount ?? null,
      totalPoints: cal.totalPoints ?? null,
      createdAt: cal.createdAt || new Date().toISOString(),
      warnings: cal.warnings || [],
    },
    qrSizes: extra.qrSizes || {},
  };
}

/**
 * Parse and validate an imported document.
 * Throws a specific message so the UI can tell the user what is wrong with the
 * file rather than "import failed".
 */
export function calibrationFromJson(text) {
  let doc;
  try {
    doc = typeof text === 'string' ? JSON.parse(text) : text;
  } catch (e) {
    throw new Error(`not valid JSON: ${e.message}`);
  }

  if (!doc || typeof doc !== 'object') throw new Error('file is empty or not an object');
  if (doc.format !== EXPORT_FORMAT) {
    throw new Error(`unrecognised file format "${doc.format ?? 'missing'}" — expected ${EXPORT_FORMAT}`);
  }
  if (typeof doc.schema !== 'number') throw new Error('missing schema version');
  if (doc.schema > EXPORT_SCHEMA) {
    throw new Error(
      `file was written by a newer version (schema ${doc.schema}, this app supports ${EXPORT_SCHEMA})`
    );
  }

  const cal = doc.calibration;
  if (!cal) throw new Error('no calibration section in the file');
  validateCalibration(cal);

  return {
    calibration: cal,
    qrSizes: doc.qrSizes && typeof doc.qrSizes === 'object' ? doc.qrSizes : {},
    exportedUtc: doc.exportedUtc || null,
    appVersion: doc.appVersion || null,
  };
}

/** Import a document, storing the calibration and any sticker sizes it carries. */
export async function importCalibrationJson(text) {
  const parsed = calibrationFromJson(text);
  const saved = await saveCalibration({
    ...parsed.calibration,
    id: undefined, // always mint a fresh id so imports never overwrite silently
  });
  for (const [id, sizeMm] of Object.entries(parsed.qrSizes)) {
    if (Number.isFinite(sizeMm) && sizeMm > 0) await setQrSize(id, sizeMm);
  }
  return { calibration: saved, qrSizeCount: Object.keys(parsed.qrSizes).length };
}

/** Filename for a downloaded calibration, including the device and date. */
export function calibrationFilename(cal) {
  const dev = (cal.deviceLabel || 'device').replace(/[^A-Za-z0-9]+/g, '-').slice(0, 24);
  const stamp = (cal.createdAt || new Date().toISOString()).slice(0, 10);
  return `spxm-calibration-${dev}-${stamp}.json`;
}

/** Trigger a browser download of the calibration document. */
export function downloadCalibration(cal, qrSizes = {}) {
  const doc = calibrationToJson(cal, { qrSizes });
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = calibrationFilename(cal);
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  return doc;
}

/** Clear everything. Used by the diagnostics screen. */
export async function clearAll() {
  if (!hasIndexedDb()) {
    memory.calibrations.clear();
    memory.qrSizes.clear();
    memory.meta.clear();
    return;
  }
  const db = await openDb();
  for (const store of [STORE_CAL, STORE_QR, STORE_META]) {
    await tx(db, store, 'readwrite', (s) => s.clear());
  }
  db.close();
}

export const _internal = { memory, DB_NAME, STORE_CAL, STORE_QR, STORE_META };
