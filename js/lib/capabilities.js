/**
 * capabilities.js — probe what this device and browser actually expose, then
 * rank the available scale sources.
 *
 * The design principle: use the best scale source the device can reach, and
 * degrade gracefully rather than failing. Every probe here is advisory — if a
 * capability is missing the app falls back down the ladder instead of erroring.
 *
 * The ladder, best accuracy first:
 *
 *   1. QR_STICKER       known printed size + calibrated camera   ~0.2% of size
 *   2. MULTI_VIEW       deliberate device motion, triangulated   ~0.1% of size
 *   3. FOCUS_DISTANCE   VCM lens position (coarse, fallback)     ~1-3%
 *   4. GRAVITY_HEIGHT   gravity + assumed camera height          coarse
 *
 * Deliberately NOT implemented, because the browser cannot reach the hardware:
 *   - Synchronised flash/ambient bracketing for shadow triangulation
 *   - Simultaneous multi-camera capture (getUserMedia holds one camera only)
 *   - Photometric inverse-square falloff (needs manual LED flux control)
 *   - WebXR depth (plane detection is dev-trial; an XR session takes the camera
 *     exclusively, so the full-resolution frames the detectors need are gone)
 *
 * These are recorded in the capability report as known-unavailable so the UI can
 * explain the ceiling honestly rather than silently underperforming.
 */

export const ScaleSource = {
  QR_STICKER: 'qr-sticker',
  MULTI_VIEW: 'multi-view',
  FOCUS_DISTANCE: 'focus-distance',
  GRAVITY_HEIGHT: 'gravity-height',
  NONE: 'none',
};

/** Probe the environment. Safe to call anywhere; never throws. */
export async function probeCapabilities() {
  const caps = {
    secureContext: typeof window !== 'undefined' ? window.isSecureContext : false,
    getUserMedia: false,
    barcodeDetector: false,
    barcodeFormats: [],
    imageCapture: false,
    focusDistance: false,
    torch: false,
    deviceOrientation: false,
    deviceMotion: false,
    maxCameraResolution: null,
    deviceLabel: null,
    errors: [],
  };

  if (typeof navigator === 'undefined') return caps;

  caps.getUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  caps.imageCapture = typeof window !== 'undefined' && 'ImageCapture' in window;

  if (typeof window !== 'undefined' && 'BarcodeDetector' in window) {
    caps.barcodeDetector = true;
    try {
      caps.barcodeFormats = await window.BarcodeDetector.getSupportedFormats();
    } catch (e) {
      caps.errors.push(`BarcodeDetector formats: ${e.message}`);
    }
  }

  if (typeof window !== 'undefined' && 'DeviceOrientationEvent' in window) {
    caps.deviceOrientation = true;
  }
  if (typeof window !== 'undefined' && 'DeviceMotionEvent' in window) {
    caps.deviceMotion = true;
  }

  // Ask the camera what it can actually do. Requesting a large size and reading
  // back the negotiated settings is the only reliable way to learn the real
  // resolution ceiling — the spec does not expose it directly.
  if (caps.getUserMedia && caps.secureContext) {
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 4096 } },
        audio: false,
      });
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings ? track.getSettings() : {};
      caps.maxCameraResolution = {
        width: settings.width || null,
        height: settings.height || null,
      };
      caps.deviceLabel = track.label || null;

      if (caps.imageCapture) {
        try {
          const ic = new window.ImageCapture(track);
          const cap = await ic.getPhotoCapabilities();
          // Presence of focusDistance in the capability set is the signal that
          // the VCM ladder rung is worth offering.
          caps.focusDistance = !!(cap && cap.focusDistance);
          caps.torch = !!(cap && cap.fillLightMode && cap.fillLightMode.includes('flash'));
        } catch (e) {
          caps.errors.push(`ImageCapture capabilities: ${e.message}`);
        }
      }
    } catch (e) {
      caps.errors.push(`camera probe: ${e.message}`);
    } finally {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    }
  }

  caps.knownUnavailable = {
    simultaneousMultiCamera: 'getUserMedia holds one camera at a time; no browser API for concurrent capture.',
    flashBracketing: 'Web pages cannot drive the LED flash in sync with capture.',
    webxrDepth: 'Plane detection is dev-trial; an XR session also takes the camera exclusively.',
    highRateImu: 'DeviceMotion is typically ~60 Hz, far below the 400+ Hz VIO needs.',
  };

  return caps;
}

/**
 * Choose the best scale source available given a live scene.
 *
 * `scene` is what the detector currently sees:
 *   { decodedQrCount, usableQrCount, hasFocusDistance, hasGravityPlane }
 */
export function chooseScaleSource(caps, scene) {
  if (scene.usableQrCount > 0) {
    return {
      source: ScaleSource.QR_STICKER,
      label: 'QR sticker',
      expectedRelativeError: 0.002,
      note: `${scene.usableQrCount} code(s) decoded with a known printed size.`,
    };
  }

  if (scene.decodedQrCount > 0) {
    return {
      source: ScaleSource.NONE,
      label: 'QR seen but size unknown',
      expectedRelativeError: null,
      note: 'A code was detected but its printed size is not known. Set it once and the app will remember.',
    };
  }

  if (scene.multiViewReady) {
    return {
      source: ScaleSource.MULTI_VIEW,
      label: 'Multi-view',
      expectedRelativeError: 0.001,
      note: 'Several views from different positions are available for triangulation.',
    };
  }

  if (caps.focusDistance) {
    return {
      source: ScaleSource.FOCUS_DISTANCE,
      label: 'Lens focus distance',
      expectedRelativeError: 0.02,
      note: 'Coarse: the lens reports the distance to whatever autofocus locked onto, not to your target.',
    };
  }

  if (scene.hasGravityPlane) {
    return {
      source: ScaleSource.GRAVITY_HEIGHT,
      label: 'Gravity + camera height',
      expectedRelativeError: 0.05,
      note: 'Rough. Needs a known camera height above a detected surface.',
    };
  }

  return {
    source: ScaleSource.NONE,
    label: 'No scale reference',
    expectedRelativeError: null,
    note: 'Nothing in view establishes a real-world length. Add a QR sticker, or move the phone to enable multi-view.',
  };
}

/** Human-readable summary of the ladder for the diagnostics screen. */
export function describeLadder(caps) {
  return [
    {
      tier: 1,
      name: 'QR sticker',
      available: true,
      detail: 'Known printed size read from the code itself. The primary path.',
    },
    {
      tier: 2,
      name: 'Multi-view triangulation',
      available: true,
      detail: 'Requires the user to move the phone between shots. Best accuracy.',
    },
    {
      tier: 3,
      name: 'Lens focus distance',
      available: !!caps.focusDistance,
      detail: caps.focusDistance
        ? 'Exposed by this device. Coarse fallback.'
        : 'Not exposed by this browser/device.',
    },
    {
      tier: 4,
      name: 'Gravity + camera height',
      available: !!caps.deviceOrientation,
      detail: caps.deviceOrientation
        ? 'Orientation sensor present. Rough fallback.'
        : 'No orientation sensor.',
    },
  ];
}
