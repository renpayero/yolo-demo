// Mapa de calor de permanencia: segundos-persona por celda, pintados sobre la imagen actual de la cámara.
// Un solo tono (secuencial): transparente donde no hubo nadie, naranja intenso donde más tiempo estuvo la gente.

import { formatDuration } from './format.js';

const HEAT_RGB = [217, 89, 38];   // #d95926: slot 2 de la paleta validada (se distingue sobre el gris del piso)
const MIN_ALPHA = 0.25;           // cualquier lugar por donde pasó alguien se ve...
const MAX_ALPHA = 0.9;            // ...y el más concurrido, casi opaco
const BACKGROUND_ALPHA = 0.45;    // la imagen de la cámara apagada, para que el calor resalte
const BLUR_PX = 3;
const TOP_ROWS = 10;

function heatLayer(data) {
  // Una celda = un píxel; al escalar con suavizado quedan manchas continuas en vez de cuadraditos.
  const layer = document.createElement('canvas');
  layer.width = data.width;
  layer.height = data.height;
  const ctx = layer.getContext('2d');
  const pixels = ctx.createImageData(data.width, data.height);
  data.grid.forEach((row, y) =>
    row.forEach((value, x) => {
      if (!value) return;
      const i = (y * data.width + x) * 4;
      const alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * Math.sqrt(value / data.max_s);
      pixels.data.set([...HEAT_RGB, Math.round(255 * alpha)], i);
    }),
  );
  ctx.putImageData(pixels, 0, 0);
  return layer;
}

// Dónde estaba el mouse sobre cada mapa: el refresco (cada 5 s) reemplaza el canvas y hay que reponer el tooltip.
const lastPointer = new WeakMap();

export function renderHeatmap(container, data, background) {
  const previous = lastPointer.get(container);
  const width = Math.max(320, container.clientWidth);
  const aspect = background?.naturalWidth ? background.naturalHeight / background.naturalWidth : 9 / 16;
  const height = Math.round(width * aspect);
  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Mapa de calor de permanencia; los valores están en «Ver como tabla»');
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  if (background?.naturalWidth) {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);
    ctx.globalAlpha = BACKGROUND_ALPHA;
    ctx.drawImage(background, 0, 0, width, height);
    ctx.globalAlpha = 1;
  } else {
    ctx.fillStyle = '#20252c';
    ctx.fillRect(0, 0, width, height);
  }
  if (data.max_s > 0) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.filter = `blur(${BLUR_PX}px)`;
    ctx.drawImage(heatLayer(data), 0, 0, width, height);
    ctx.filter = 'none';
  }

  const tooltip = document.createElement('div');
  tooltip.className = 'chart-tooltip';
  tooltip.hidden = true;
  const show = (x, y) => {
    const cx = Math.min(data.width - 1, Math.max(0, Math.floor((x / width) * data.width)));
    const cy = Math.min(data.height - 1, Math.max(0, Math.floor((y / height) * data.height)));
    const value = data.grid[cy]?.[cx] ?? 0;
    const strong = document.createElement('strong');
    strong.textContent = value ? formatDuration(value) : 'nadie';
    const label = document.createElement('span');
    label.textContent = value ? ' de permanencia en este punto' : ' estuvo en este punto';
    tooltip.replaceChildren(strong, label);
    tooltip.hidden = false;
    tooltip.style.left = `${x + 12 + 220 > width ? x - 12 - 220 : x + 12}px`;
    tooltip.style.top = `${y + 12}px`;
    lastPointer.set(container, { x, y });
  };
  const hide = () => {
    tooltip.hidden = true;
    if (canvas.isConnected) lastPointer.delete(container);  // al reemplazarlo por un refresco no cuenta como salir
  };
  canvas.addEventListener('pointermove', (event) => {
    const box = canvas.getBoundingClientRect();
    show(event.clientX - box.left, event.clientY - box.top);
  });
  canvas.addEventListener('pointerleave', hide);
  container.replaceChildren(canvas, tooltip);

  if (previous) {
    // El mouse sigue donde estaba (el canvas nuevo no recibe eventos hasta que se mueva): se repone el tooltip
    // y se oculta recién cuando el mouse se mueva fuera del mapa.
    show(previous.x, previous.y);
    const leaveCheck = (event) => {
      document.removeEventListener('pointermove', leaveCheck, true);
      if (!container.contains(event.target)) hide();
    };
    document.addEventListener('pointermove', leaveCheck, true);
  }
}

/** Vista de tabla: los puntos con más permanencia (posición en % del cuadro). */
export function heatmapRows(data) {
  const cells = [];
  data.grid.forEach((row, y) => row.forEach((value, x) => value && cells.push({ x, y, value })));
  return cells
    .sort((a, b) => b.value - a.value)
    .slice(0, TOP_ROWS)
    .map(({ x, y, value }) => [
      `${Math.round(((x + 0.5) / data.width) * 100)} % · ${Math.round(((y + 0.5) / data.height) * 100)} %`,
      formatDuration(value),
    ]);
}
