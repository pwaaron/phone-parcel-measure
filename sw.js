/* sw.js — service worker.
 *
 * Caches the whole app shell so it runs offline on a hub floor, where the wifi
 * is unreliable and the camera must work regardless.
 *
 * Strategy:
 *   - App shell (HTML/CSS/JS/vendor): cache-first, refreshed in the background.
 *   - Everything else: network, falling back to cache.
 *
 * The cache name carries a version; bumping it retires the old cache on the
 * next activation, so a deploy never leaves a stale module in place.
 */

// Bump this on every deploy that changes a cached file. The strategy is
// cache-first, so without a bump the phone keeps serving the previous version
// and a fix appears not to have worked.
const VERSION = 'v3';
const CACHE = `spx-parcel-measure-${VERSION}`;

const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './css/app.css',
  './js/main.js',
  './js/aruco-dict.js',
  './js/lib/linalg.js',
  './js/lib/camera.js',
  './js/lib/calibrate.js',
  './js/lib/charuco.js',
  './js/lib/contour.js',
  './js/lib/image.js',
  './js/lib/aruco.js',
  './js/lib/subpix.js',
  './js/lib/pnp.js',
  './js/lib/measure.js',
  './js/lib/qr.js',
  './js/lib/store.js',
  './js/lib/capabilities.js',
  './vendor/jsqr.js',
  './icons/icon.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Never cache the printables: they are large PDFs, and a stale board or
  // sticker sheet is worse than a slow one.
  if (url.pathname.includes('/printables/')) return;

  event.respondWith(
    caches.match(req).then((cached) => {
      const network = fetch(req)
        .then((res) => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
          }
          return res;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
