const video = document.querySelector("#camera");
const canvas = document.querySelector("#matrix");
const ctx = canvas.getContext("2d", { alpha: true });

const startPanel = document.querySelector("#startPanel");
const startButton = document.querySelector("#startCamera");
const statusText = document.querySelector("#statusText");
const styleButtons = [...document.querySelectorAll("[data-style]")];
const resetButton = document.querySelector("#resetButton");
const undoButton = document.querySelector("#undoButton");
const clearButton = document.querySelector("#clearButton");

const performanceConfig = {
  detectIntervalMs: 33,
  overlayIntervalMs: 33,
  sampleWidth: 160,
  maxEmojiCells: 620,
  minCellSize: 12,
  maxCellSize: 26,
};

const stylePalettes = {
  moon: ["🌑", "🌘", "🌗", "🌖", "🌕"],
  heart: ["🖤", "🤎", "💜", "💗"],
  flower: ["🥀", "🌹", "🌷", "🌸", "💮"],
};

const glyphCache = new Map();
const sampleCanvas = document.createElement("canvas");
const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });

let activeStyle = "heart";
let handLandmarker = null;
let vision = null;
let lastDetectTime = 0;
let lastOverlayTime = 0;
let lastVideoTime = -1;
let latestResult = null;
let currentQuad = null;
let smoothedQuad = null;
let pinnedCanvas = null;
let isPinned = false;
let cameraReady = false;
let modelReady = false;
let loadingModel = false;
let fallbackClock = 0;
let pinRequested = false;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const lerp = (a, b, t) => a + (b - a) * t;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function viewportPoint(landmark) {
  return {
    x: (1 - landmark.x) * canvas.width,
    y: landmark.y * canvas.height,
  };
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(window.innerWidth * ratio);
  const height = Math.round(window.innerHeight * ratio);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    cameraReady = true;
    startPanel?.classList.add("is-hidden");
    updateStatus("ACTIVE");
    loadModel();
  } catch (error) {
    updateStatus("CAMERA BLOCKED");
    if (statusText) {
      statusText.textContent = "需要浏览器摄像头权限";
    }
  }
}

async function loadModel() {
  if (loadingModel || modelReady) return;
  loadingModel = true;

  try {
    const { FilesetResolver, HandLandmarker } = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18"
    );

    vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    });

    modelReady = true;
    updateStatus("ACTIVE");
  } catch (error) {
    modelReady = false;
    updateStatus("PREVIEW");
    if (statusText) {
      statusText.textContent = "手势模型加载失败，正在使用预览形态";
    }
  } finally {
    loadingModel = false;
  }
}

function updateStatus(text) {
  const label = document.querySelector("#activeLabel");
  if (label) label.textContent = text;
}

function analyzeHand(landmarks) {
  const thumb = viewportPoint(landmarks[4]);
  const index = viewportPoint(landmarks[8]);
  const middle = viewportPoint(landmarks[12]);
  const ring = viewportPoint(landmarks[16]);
  const pinky = viewportPoint(landmarks[20]);
  const wrist = viewportPoint(landmarks[0]);
  const fingertips = [thumb, index, middle, ring, pinky];
  const spread = Math.max(...fingertips.map((point) => dist(point, wrist)));
  const pinchDistance = dist(thumb, index);

  return {
    thumb,
    index,
    middle,
    ring,
    pinky,
    fingertips,
    wrist,
    spread,
    pinched: pinchDistance < Math.max(24, spread * 0.18),
    pinchPoint: {
      x: (thumb.x + index.x) / 2,
      y: (thumb.y + index.y) / 2,
    },
  };
}

function controlPair(hand) {
  const preferred = [hand.thumb, hand.index];
  if (dist(preferred[0], preferred[1]) > Math.max(42, hand.spread * 0.26)) {
    return preferred;
  }

  let bestPair = [hand.index, hand.pinky];
  let bestScore = -Infinity;

  for (let a = 0; a < hand.fingertips.length; a += 1) {
    for (let b = a + 1; b < hand.fingertips.length; b += 1) {
      const one = hand.fingertips[a];
      const two = hand.fingertips[b];
      const score = dist(one, two) + Math.abs(one.y - two.y) * 0.35;
      if (score > bestScore) {
        bestScore = score;
        bestPair = [one, two];
      }
    }
  }

  return bestPair;
}

function sortPairVertical(pair) {
  return pair[0].y <= pair[1].y ? pair : [pair[1], pair[0]];
}

function quadFromHands(result) {
  const hands = result?.landmarks?.map(analyzeHand) ?? [];

  if (hands.length >= 2) {
    const sorted = hands.sort((a, b) => a.pinchPoint.x - b.pinchPoint.x);
    const [leftTop, leftBottom] = sortPairVertical(controlPair(sorted[0]));
    const [rightTop, rightBottom] = sortPairVertical(controlPair(sorted[1]));
    const rawQuad = [leftTop, rightTop, rightBottom, leftBottom];

    if (quadArea(rawQuad) > canvas.width * canvas.height * 0.015) {
      return smoothQuad(rawQuad, 0.42);
    }
  }

  return null;
}

function smoothQuad(rawQuad, follow) {
  if (!smoothedQuad) {
    smoothedQuad = rawQuad.map((point) => ({ ...point }));
    return smoothedQuad;
  }

  smoothedQuad = rawQuad.map((point, index) => ({
    x: lerp(smoothedQuad[index].x, point.x, follow),
    y: lerp(smoothedQuad[index].y, point.y, follow),
  }));

  return smoothedQuad;
}

function quadArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const next = points[(i + 1) % points.length];
    area += points[i].x * next.y - next.x * points[i].y;
  }
  return Math.abs(area / 2);
}

function fallbackQuad(time) {
  const width = canvas.width;
  const height = canvas.height;
  const cx = width * 0.5;
  const cy = height * 0.5;
  const w = width * 0.5;
  const h = height * 0.23;
  const angle = Math.sin(time * 0.0012) * 0.15;
  const skew = Math.sin(time * 0.0018) * width * 0.035;
  const points = [
    { x: cx - w * 0.52 - skew, y: cy - h * 0.55 },
    { x: cx + w * 0.5 + skew, y: cy - h * 0.42 },
    { x: cx + w * 0.54 - skew, y: cy + h * 0.48 },
    { x: cx - w * 0.48 + skew, y: cy + h * 0.58 },
  ];

  return rotatePoints(points, { x: cx, y: cy }, angle);
}

function rotatePoints(points, center, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return points.map((point) => {
    const x = point.x - center.x;
    const y = point.y - center.y;
    return {
      x: center.x + x * cos - y * sin,
      y: center.y + x * sin + y * cos,
    };
  });
}

function prepareVideoSample() {
  if (!cameraReady || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  const width = performanceConfig.sampleWidth;
  const height = Math.max(
    1,
    Math.round(width * (canvas.height / Math.max(1, canvas.width)))
  );

  if (sampleCanvas.width !== width || sampleCanvas.height !== height) {
    sampleCanvas.width = width;
    sampleCanvas.height = height;
  }

  const videoRatio = video.videoWidth / video.videoHeight;
  const viewRatio = canvas.width / canvas.height;
  let sx = 0;
  let sy = 0;
  let sw = video.videoWidth;
  let sh = video.videoHeight;

  if (videoRatio > viewRatio) {
    sw = video.videoHeight * viewRatio;
    sx = (video.videoWidth - sw) / 2;
  } else {
    sh = video.videoWidth / viewRatio;
    sy = (video.videoHeight - sh) / 2;
  }

  sampleCtx.save();
  sampleCtx.clearRect(0, 0, width, height);
  sampleCtx.translate(width, 0);
  sampleCtx.scale(-1, 1);
  sampleCtx.drawImage(video, sx, sy, sw, sh, 0, 0, width, height);
  sampleCtx.restore();

  return sampleCtx.getImageData(0, 0, width, height);
}

function pointInQuad(point, quad) {
  let inside = false;
  for (let i = 0, j = quad.length - 1; i < quad.length; j = i, i += 1) {
    const xi = quad[i].x;
    const yi = quad[i].y;
    const xj = quad[j].x;
    const yj = quad[j].y;
    const intersects =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 0.0001) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function boundsOfQuad(quad) {
  const xs = quad.map((point) => point.x);
  const ys = quad.map((point) => point.y);
  return {
    left: clamp(Math.min(...xs), 0, canvas.width),
    right: clamp(Math.max(...xs), 0, canvas.width),
    top: clamp(Math.min(...ys), 0, canvas.height),
    bottom: clamp(Math.max(...ys), 0, canvas.height),
  };
}

function chooseEmoji(rgba) {
  const [r, g, b] = rgba;
  const lum = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 255;
  const palette = stylePalettes[activeStyle] ?? stylePalettes.heart;
  return palette[
    clamp(Math.floor(lum * palette.length), 0, palette.length - 1)
  ];
}

function sampleColor(imageData, x, y) {
  if (!imageData) {
    const wave = (Math.sin((x + y + fallbackClock) * 0.01) + 1) * 0.5;
    return [wave * 255, 80, 180, 255];
  }

  const sx = clamp(
    Math.floor((x / canvas.width) * imageData.width),
    0,
    imageData.width - 1
  );
  const sy = clamp(
    Math.floor((y / canvas.height) * imageData.height),
    0,
    imageData.height - 1
  );
  const index = (sy * imageData.width + sx) * 4;
  return [
    imageData.data[index],
    imageData.data[index + 1],
    imageData.data[index + 2],
    imageData.data[index + 3],
  ];
}

function getGlyph(emoji, size) {
  const roundedSize = Math.round(size);
  const key = `${emoji}:${roundedSize}`;
  if (glyphCache.has(key)) return glyphCache.get(key);

  const glyph = document.createElement("canvas");
  const pad = Math.ceil(roundedSize * 0.18);
  glyph.width = roundedSize + pad * 2;
  glyph.height = roundedSize + pad * 2;
  const glyphCtx = glyph.getContext("2d");
  glyphCtx.font = `${roundedSize}px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
  glyphCtx.textAlign = "center";
  glyphCtx.textBaseline = "middle";
  glyphCtx.fillText(emoji, glyph.width / 2, glyph.height / 2);
  glyphCache.set(key, glyph);
  return glyph;
}

function drawMatrix(quad) {
  if (!quad) return;

  const imageData = prepareVideoSample();
  const bounds = boundsOfQuad(quad);
  const area = Math.max(1, quadArea(quad));
  const cell = clamp(
    Math.sqrt(area / performanceConfig.maxEmojiCells),
    performanceConfig.minCellSize,
    performanceConfig.maxCellSize
  );
  const stepX = cell * 0.86;
  const stepY = cell * 0.72;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(quad[0].x, quad[0].y);
  for (let i = 1; i < quad.length; i += 1) {
    ctx.lineTo(quad[i].x, quad[i].y);
  }
  ctx.closePath();
  ctx.clip();

  ctx.fillStyle = "#030407";
  ctx.fillRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);

  for (let y = bounds.top - stepY; y <= bounds.bottom + stepY; y += stepY) {
    const rowOffset = Math.round(y / stepY) % 2 ? stepX * 0.5 : 0;
    for (let x = bounds.left - stepX; x <= bounds.right + stepX; x += stepX) {
      const point = { x: x + rowOffset, y };
      if (!pointInQuad(point, quad)) continue;

      const rgba = sampleColor(imageData, point.x, point.y);
      const emoji = chooseEmoji(rgba);
      const glyph = getGlyph(emoji, cell);
      ctx.drawImage(glyph, point.x - glyph.width / 2, point.y - glyph.height / 2);
    }
  }

  ctx.restore();
  drawGlow(quad);
}

function drawGlow(quad) {
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowBlur = 16;
  ctx.shadowColor = "rgba(41, 220, 255, 0.72)";
  ctx.strokeStyle = "rgba(68, 220, 255, 0.95)";
  ctx.lineWidth = Math.max(2, canvas.width * 0.0012);
  ctx.beginPath();
  ctx.moveTo(quad[0].x, quad[0].y);
  for (let i = 1; i < quad.length; i += 1) ctx.lineTo(quad[i].x, quad[i].y);
  ctx.closePath();
  ctx.stroke();

  ctx.shadowColor = "rgba(255, 65, 180, 0.95)";
  ctx.fillStyle = "rgba(255, 76, 183, 0.52)";
  quad.forEach((point) => {
    const radius = Math.max(10, canvas.width * 0.006);
    const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius * 1.8);
    gradient.addColorStop(0, "rgba(255, 76, 183, 0.82)");
    gradient.addColorStop(1, "rgba(255, 76, 183, 0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius * 1.8, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function pinCurrentFrame() {
  if (!currentQuad || isPinned) return;

  pinnedCanvas = document.createElement("canvas");
  pinnedCanvas.width = canvas.width;
  pinnedCanvas.height = canvas.height;
  pinnedCanvas.getContext("2d").drawImage(canvas, 0, 0);
  isPinned = true;
  currentQuad = null;
  smoothedQuad = null;
}

function maybePin(result) {
  const hands = result?.landmarks?.map(analyzeHand) ?? [];
  if (hands.length < 2 || !currentQuad) return;
  if (hands[0].pinched && hands[1].pinched) {
    pinRequested = true;
  }
}

function clearPinned() {
  pinnedCanvas = null;
  isPinned = false;
  pinRequested = false;
  currentQuad = null;
  smoothedQuad = null;
}

function render(time) {
  resizeCanvas();

  if (time - lastOverlayTime < performanceConfig.overlayIntervalMs) {
    requestAnimationFrame(render);
    return;
  }
  lastOverlayTime = time;
  fallbackClock = time;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (pinnedCanvas) {
    ctx.drawImage(pinnedCanvas, 0, 0);
    requestAnimationFrame(render);
    return;
  }

  if (
    cameraReady &&
    modelReady &&
    handLandmarker &&
    video.currentTime !== lastVideoTime &&
    time - lastDetectTime >= performanceConfig.detectIntervalMs
  ) {
    lastDetectTime = time;
    lastVideoTime = video.currentTime;
    latestResult = handLandmarker.detectForVideo(video, performance.now());
    currentQuad = quadFromHands(latestResult);
    maybePin(latestResult);
  }

  if (!cameraReady || !modelReady) {
    currentQuad = fallbackQuad(time);
  }

  drawMatrix(currentQuad);
  if (pinRequested) {
    pinRequested = false;
    pinCurrentFrame();
  }
  requestAnimationFrame(render);
}

styleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeStyle = button.dataset.style;
    styleButtons.forEach((item) => {
      item.classList.toggle("is-active", item === button);
    });
    glyphCache.clear();
  });
});

startButton?.addEventListener("click", startCamera);
resetButton?.addEventListener("click", clearPinned);
undoButton?.addEventListener("click", clearPinned);
clearButton?.addEventListener("click", clearPinned);
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
requestAnimationFrame(render);

if (navigator.mediaDevices?.getUserMedia) {
  startCamera();
} else if (statusText) {
  statusText.textContent = "当前浏览器不支持摄像头";
}
