// Gráficos SVG del dashboard: línea de tiempo (Gantt) y personas en zonas (escalonado).
// Comparten el eje de tiempo y el margen izquierdo, así el crosshair cae en el mismo instante en los dos.
// Los textos que vienen de datos (nombres de zonas) se insertan siempre con textContent.

const SVG_NS = 'http://www.w3.org/2000/svg';
export const MARGIN = { left: 136, right: 16, top: 12, bottom: 26 };
const ROW_H = 30;
const BAR_H = 14;
const HEADCOUNT_H = 150;
const TICK_STEPS_S = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];

function svgEl(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.append(node);
  return node;
}

export const clock = (ts, withSeconds = false) =>
  new Date(ts * 1000).toLocaleTimeString('es-AR', {
    hour: '2-digit', minute: '2-digit', ...(withSeconds ? { second: '2-digit' } : {}), hour12: false,
  });

const dayClock = (ts) =>
  new Date(ts * 1000).toLocaleString('es-AR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });

/** Ticks en horas "redondas" de la hora local, ~1 cada 110 px. */
function timeTicks(from, to, width) {
  const wanted = Math.max(2, Math.floor(width / 110));
  const step = TICK_STEPS_S.find((s) => (to - from) / s <= wanted) ?? TICK_STEPS_S.at(-1);
  const offset = -new Date(from * 1000).getTimezoneOffset() * 60;
  const ticks = [];
  for (let t = Math.ceil((from + offset) / step) * step - offset; t <= to; t += step) ticks.push(t);
  const label = to - from > 86400 ? dayClock : (t) => clock(t);
  return ticks.map((t) => ({ t, label: label(t) }));
}

function frame(container, height, data) {
  const width = Math.max(320, container.clientWidth);
  const { from, to } = data.range;
  const plotW = width - MARGIN.left - MARGIN.right;
  const x = (t) => MARGIN.left + ((t - from) / (to - from)) * plotW;
  const tAt = (px) => from + ((px - MARGIN.left) / plotW) * (to - from);
  const svg = svgEl('svg', { width, height, viewBox: `0 0 ${width} ${height}`, class: 'chart-svg' });
  return { svg, width, height, x, tAt, from, to };
}

function drawAxis(f, plotTop, plotBottom) {
  const grid = svgEl('g', { class: 'grid' }, f.svg);
  for (const tick of timeTicks(f.from, f.to, f.width - MARGIN.left - MARGIN.right)) {
    const px = f.x(tick.t);
    svgEl('line', { x1: px, x2: px, y1: plotTop, y2: plotBottom }, grid);
    const text = svgEl('text', { x: px, y: plotBottom + 17, 'text-anchor': 'middle', class: 'axis-label' }, grid);
    text.textContent = tick.label;
  }
  svgEl('line', { x1: MARGIN.left, x2: f.width - MARGIN.right, y1: plotBottom, y2: plotBottom, class: 'baseline' }, f.svg);
}

function drawUnmonitored(f, bands, top, bottom) {
  const g = svgEl('g', { class: 'unmonitored' }, f.svg);
  for (const [start, end] of bands) {
    svgEl('rect', { x: f.x(start), y: top, width: Math.max(1, f.x(end) - f.x(start)), height: bottom - top }, g);
  }
}

/**
 * Capa de interacción: crosshair, tooltip y teclado. `describe(t)` arma el contenido del tooltip.
 * Todo vive en un envoltorio nuevo por render: al refrescar se reemplaza entero (sin listeners acumulados).
 */
function interactive(container, key, f, plotTop, plotBottom, describe, sync, ariaLabel) {
  const wrapper = document.createElement('div');
  wrapper.className = 'chart-inner';
  wrapper.tabIndex = 0;
  wrapper.setAttribute('role', 'img');
  wrapper.setAttribute('aria-label', ariaLabel);
  wrapper.append(f.svg);
  const line = svgEl('line', { class: 'crosshair', y1: plotTop, y2: plotBottom, visibility: 'hidden' }, f.svg);
  const hit = svgEl('rect', {
    x: MARGIN.left, y: plotTop, width: f.width - MARGIN.left - MARGIN.right, height: plotBottom - plotTop,
    fill: 'transparent',
  }, f.svg);
  const tooltip = document.createElement('div');
  tooltip.className = 'chart-tooltip';
  tooltip.hidden = true;
  let current = null;

  const showLine = (t) => {
    const px = f.x(t);
    line.setAttribute('x1', px);
    line.setAttribute('x2', px);
    line.setAttribute('visibility', 'visible');
  };
  const show = (t) => {
    current = Math.min(f.to, Math.max(f.from, t));
    showLine(current);
    tooltip.replaceChildren(...describe(current));
    tooltip.hidden = false;
    const px = f.x(current);
    const left = px + 12 + tooltip.offsetWidth > f.width ? px - 12 - tooltip.offsetWidth : px + 12;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${plotTop}px`;
    sync.move(current, api);
  };
  const hide = () => {
    current = null;
    line.setAttribute('visibility', 'hidden');
    tooltip.hidden = true;
    sync.leave(api);
  };

  hit.addEventListener('pointermove', (event) => {
    const box = f.svg.getBoundingClientRect();
    show(f.tAt(event.clientX - box.left));
  });
  hit.addEventListener('pointerleave', hide);
  wrapper.addEventListener('focus', () => show(current ?? f.to));
  wrapper.addEventListener('blur', hide);
  wrapper.addEventListener('keydown', (event) => {
    const step = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
    if (!step) return;
    event.preventDefault();
    show((current ?? f.to) + (step * (f.to - f.from)) / 60);
  });

  const api = {
    key,
    // Lo llama el otro gráfico: acá solo se mueve la línea, sin tooltip.
    follow(t) { showLine(t); },
    unfollow() { if (tooltip.hidden) line.setAttribute('visibility', 'hidden'); },
    // Después de un refresco el gráfico es un elemento nuevo: el navegador no le marca :hover hasta que el mouse
    // se mueva. Se restaura donde estaba y se oculta recién cuando el mouse se mueve fuera del gráfico.
    restore(t, focused) {
      if (focused) wrapper.focus({ preventScroll: true });
      show(t);
      const leaveCheck = (event) => {
        document.removeEventListener('pointermove', leaveCheck, true);
        if (!wrapper.contains(event.target) && document.activeElement !== wrapper) hide();
      };
      document.addEventListener('pointermove', leaveCheck, true);
    },
    get current() { return current; },
    get focused() { return document.activeElement === wrapper; },
  };
  sync.register(api);
  wrapper.append(tooltip);
  container.replaceChildren(wrapper);
}

/**
 * Coordina el crosshair entre gráficos que comparten el eje de tiempo y lo conserva entre refrescos:
 * `beginRender()` antes de redibujar, `endRender()` después.
 */
export function createCrosshairSync() {
  let charts = [];
  let pending = null;   // { key, t } del gráfico que el usuario estaba leyendo
  return {
    beginRender() {
      const reading = charts.find((c) => c.current != null);
      pending = reading ? { key: reading.key, t: reading.current, focused: reading.focused } : null;
      charts = [];
    },
    endRender() {
      if (pending) charts.find((c) => c.key === pending.key)?.restore(pending.t, pending.focused);
      pending = null;
    },
    register(chart) { charts.push(chart); },
    move(t, source) { for (const c of charts) if (c !== source) c.follow(t); },
    leave(source) { for (const c of charts) if (c !== source) c.unfollow(); },
  };
}

function tooltipHeader(text) {
  const node = document.createElement('p');
  node.className = 'tt-header';
  node.textContent = text;
  return node;
}

function tooltipRow(color, value, label) {
  const row = document.createElement('p');
  row.className = 'tt-row';
  const key = document.createElement('span');
  key.className = 'tt-key';
  key.style.background = color;
  const strong = document.createElement('strong');
  strong.textContent = value;
  const name = document.createElement('span');
  name.textContent = label;
  row.append(key, strong, name);
  return row;
}

const inside = (intervals, t) => intervals.some(([s, e]) => s <= t && t < e);

// ---------- Línea de tiempo (Gantt) ----------

export function renderTimeline(container, data, sync) {
  const rows = data.timeline;
  const plotTop = MARGIN.top;
  const plotBottom = plotTop + Math.max(1, rows.length) * ROW_H;
  const f = frame(container, plotBottom + MARGIN.bottom, data);
  drawUnmonitored(f, data.unmonitored, plotTop, plotBottom);
  drawAxis(f, plotTop, plotBottom);

  rows.forEach((row, i) => {
    const y = plotTop + i * ROW_H;
    const g = svgEl('g', {}, f.svg);
    svgEl('rect', { x: 8, y: y + ROW_H / 2 - 5, width: 10, height: 10, rx: 2, fill: row.color }, g);
    const label = svgEl('text', { x: 24, y: y + ROW_H / 2 + 4, class: 'row-label' }, g);
    label.textContent = row.name.length > 16 ? `${row.name.slice(0, 15)}…` : row.name;
    if (i > 0) svgEl('line', { x1: MARGIN.left, x2: f.width - MARGIN.right, y1: y, y2: y, class: 'row-sep' }, g);
    for (const [start, end] of row.intervals) {
      svgEl('rect', {
        x: f.x(start), y: y + (ROW_H - BAR_H) / 2, width: Math.max(2, f.x(end) - f.x(start)), height: BAR_H,
        rx: 3, fill: row.color,
      }, g);
    }
    for (const alert of row.alerts) {
      const ax = f.x(alert.ts);
      svgEl('path', { d: `M${ax - 5},${y + 1} h10 l-5,7 z`, class: 'alert-mark' }, g);
    }
  });

  const tolerance = ((f.to - f.from) / (f.width - MARGIN.left - MARGIN.right)) * 6;  // 6 px en segundos
  interactive(container, 'timeline', f, plotTop, plotBottom, (t) => {
    const monitored = !inside(data.unmonitored, t);
    const nodes = [tooltipHeader(clock(t, true))];
    for (const row of rows) {
      const state = !monitored ? 'sin video' : inside(row.intervals, t) ? 'ocupada' : 'vacía';
      nodes.push(tooltipRow(row.color, state, row.name));
      for (const alert of row.alerts.filter((a) => Math.abs(a.ts - t) <= tolerance)) {
        nodes.push(tooltipRow('var(--danger)', `alerta ${clock(alert.ts, true)}`, row.name));
      }
    }
    return nodes;
  }, sync, 'Línea de tiempo de ocupación por zona; los datos están en «Ver como tabla»');
}

// ---------- Personas en zonas (escalonado) ----------

export function renderHeadcount(container, data, sync) {
  const points = data.headcount;
  const plotTop = MARGIN.top;
  const plotBottom = plotTop + HEADCOUNT_H;
  const f = frame(container, plotBottom + MARGIN.bottom, data);
  const peak = Math.max(1, ...points.map((p) => p.total));
  const yStep = Math.max(1, Math.ceil(peak / 4));
  const yMax = Math.ceil(peak / yStep) * yStep;
  const y = (v) => plotBottom - (v / yMax) * HEADCOUNT_H;

  drawUnmonitored(f, data.unmonitored, plotTop, plotBottom);
  drawAxis(f, plotTop, plotBottom);
  const grid = svgEl('g', { class: 'grid' }, f.svg);
  for (let v = yStep; v <= yMax; v += yStep) {
    svgEl('line', { x1: MARGIN.left, x2: f.width - MARGIN.right, y1: y(v), y2: y(v) }, grid);
  }
  for (let v = 0; v <= yMax; v += yStep) {
    const text = svgEl('text', { x: MARGIN.left - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'axis-label' }, grid);
    text.textContent = v;
  }

  if (points.length) {
    let d = `M${f.x(points[0].t)},${y(points[0].total)}`;
    for (const p of points.slice(1)) d += ` H${f.x(p.t)} V${y(p.total)}`;
    d += ` H${f.x(f.to)}`;
    svgEl('path', { d: `${d} V${plotBottom} H${f.x(points[0].t)} Z`, class: 'area' }, f.svg);
    svgEl('path', { d, class: 'step-line' }, f.svg);
  }

  const zones = Object.fromEntries(data.timeline.map((z) => [z.id, z]));
  interactive(container, 'headcount', f, plotTop, plotBottom, (t) => {
    const point = points.filter((p) => p.t <= t).at(-1) ?? points[0];
    const nodes = [tooltipHeader(clock(t, true)), tooltipRow('var(--series-total)', String(point?.total ?? 0), 'en total')];
    for (const [zoneId, count] of Object.entries(point?.zones ?? {})) {
      const zone = zones[zoneId];
      if (zone) nodes.push(tooltipRow(zone.color, String(count), zone.name));
    }
    return nodes;
  }, sync, `Personas en zonas a lo largo del período, máximo ${peak}; datos en «Ver como tabla»`);
}
