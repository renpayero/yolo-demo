// Pestaña "Métricas": un rango horario arriba que filtra todo lo de abajo; se refresca sola si el rango
// termina "ahora". Mientras recarga por un cambio del usuario, se mantiene lo anterior atenuado (sin parpadeo).

import { clock, createCrosshairSync, renderHeadcount, renderTimeline } from './charts.js';
import { formatDuration, humanDuration } from './format.js';

const REFRESH_MS = 5000;
const MAX_TABLE_ROWS = 200;
const $ = (id) => document.getElementById(id);
const ALERT_KIND = { dwell: 'Permanencia', empty: 'Estación vacía', restricted: 'Zona restringida' };

const dateTime = (ts) =>
  new Date(ts * 1000).toLocaleString('es-AR', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
const isToday = (ts) => new Date(ts * 1000).toDateString() === new Date().toDateString();
const when = (ts) => (isToday(ts) ? clock(ts, true) : dateTime(ts));
const orDash = (value, fmt) => (value == null ? '—' : fmt(value));

/** Convierte epoch a valor de <input type="datetime-local"> en hora local. */
function toLocalInput(ts) {
  const d = new Date(ts * 1000);
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
}

function cell(text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function zoneCell(name, color) {
  const td = document.createElement('td');
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = color;
  const label = document.createElement('span');
  label.textContent = name;
  td.className = 'zone-cell';
  td.append(swatch, label);
  return td;
}

function meterCell(pct, color) {
  const td = document.createElement('td');
  td.className = 'meter-cell';
  const meter = document.createElement('span');
  meter.className = 'meter';
  meter.style.setProperty('--meter-color', color);
  const fill = document.createElement('span');
  fill.style.width = `${pct ?? 0}%`;
  meter.append(fill);
  const value = document.createElement('span');
  value.className = 'meter-value';
  value.textContent = pct == null ? 'sin video' : `${Math.round(pct)} %`;
  td.append(meter, value);
  return td;
}

function fillTable(tableId, rows, emptyId) {
  $(tableId).tBodies[0].replaceChildren(...rows);
  if (emptyId) $(emptyId).hidden = rows.length > 0;
}

// ---------- Secciones ----------

function renderKpis({ summary }) {
  $('kpi-monitored').textContent = humanDuration(summary.monitored_s);
  $('kpi-visits').textContent = summary.visits.toLocaleString('es-AR');
  $('kpi-alerts').textContent = summary.alerts.toLocaleString('es-AR');
  $('kpi-alerts-sub').textContent = summary.active_alerts ? `${summary.active_alerts} sin resolver` : '';
  $('kpi-peak').textContent = summary.peak_people;
}

function renderZoneTable({ zones }) {
  fillTable('zone-kpis', zones.map((z) => {
    const tr = document.createElement('tr');
    tr.append(
      zoneCell(z.kind === 'restricted' ? `⚠ ${z.name}` : z.name, z.color),
      meterCell(z.occupancy_pct, z.color),
      cell(formatDuration(z.occupied_s), 'num'),
      cell(formatDuration(z.empty_s), 'num'),
      cell(String(z.visits), 'num'),
      cell(orDash(z.avg_visit_s, formatDuration), 'num'),
      cell(orDash(z.max_visit_s, formatDuration), 'num'),
      cell(formatDuration(z.person_s), 'num'),
      cell(String(z.alerts), 'num'),
    );
    return tr;
  }), 'zone-kpis-empty');
}

function renderTimelineTable({ timeline }) {
  const rows = timeline
    .flatMap((zone) => zone.intervals.map(([start, end]) => ({ zone, start, end })))
    .sort((a, b) => b.start - a.start)
    .slice(0, MAX_TABLE_ROWS)
    .map(({ zone, start, end }) => {
      const tr = document.createElement('tr');
      tr.append(zoneCell(zone.name, zone.color), cell(when(start)), cell(when(end)), cell(formatDuration(end - start), 'num'));
      return tr;
    });
  fillTable('timeline-table', rows);
}

function renderHeadcountTable({ headcount }) {
  const rows = headcount
    .filter((p, i) => i === 0 || p.total !== headcount[i - 1].total)
    .reverse()
    .slice(0, MAX_TABLE_ROWS)
    .map((p) => {
      const tr = document.createElement('tr');
      tr.append(cell(when(p.t)), cell(String(p.total), 'num'));
      return tr;
    });
  fillTable('headcount-table', rows);
}

function renderPersonsTable({ persons }) {
  const rows = persons.flatMap((person) =>
    person.zones.map((zone, i) => {
      const tr = document.createElement('tr');
      if (i === 0) {
        const name = cell(person.name, 'person-cell');
        name.rowSpan = person.zones.length;
        tr.append(name);
      }
      tr.append(
        zoneCell(zone.zone_name, zone.color),
        cell(String(zone.visits), 'num'),
        cell(formatDuration(zone.time_s), 'num'),
        cell(orDash(zone.avg_visit_s, formatDuration), 'num'),
      );
      return tr;
    }),
  );
  fillTable('persons-table', rows, 'persons-table-empty');
  $('persons-table').hidden = rows.length === 0;
}

function renderAlertsTable({ alerts }) {
  fillTable('alerts-table', alerts.slice(0, MAX_TABLE_ROWS).map((a) => {
    const tr = document.createElement('tr');
    const status = cell(a.resolved_ts ? formatDuration(a.resolved_ts - a.ts) : 'activa', 'num');
    if (!a.resolved_ts) status.classList.add('status-active');
    tr.append(cell(when(a.ts)), cell(ALERT_KIND[a.kind] ?? a.kind), cell(a.zone_name ?? '—'), cell(a.message), status);
    return tr;
  }), 'alerts-table-empty');
}

// ---------- Controlador ----------

export function createDashboard() {
  const sync = createCrosshairSync();
  let preset = '15m';  // en una demo recién arrancada, la última hora es casi toda "sin video"
  let custom = null;
  let active = false;
  let timer = null;
  let inflight = false;
  let lastData = null;

  function currentRange() {
    const now = Date.now() / 1000;
    if (preset === '15m') return { from: now - 15 * 60, to: now, live: true };
    if (preset === '8h') return { from: now - 8 * 3600, to: now, live: true };
    if (preset === 'today') {
      const midnight = new Date();
      midnight.setHours(0, 0, 0, 0);
      return { from: midnight.getTime() / 1000, to: now, live: true };
    }
    if (preset === 'custom' && custom) return { ...custom, live: false };
    return { from: now - 3600, to: now, live: true };
  }

  function renderCharts(data) {
    sync.beginRender();
    renderTimeline($('timeline-chart'), data, sync);
    renderHeadcount($('headcount-chart'), data, sync);
    sync.endRender();
  }

  function render(data) {
    renderKpis(data);
    renderZoneTable(data);
    renderCharts(data);
    renderTimelineTable(data);
    renderHeadcountTable(data);
    renderPersonsTable(data);
    renderAlertsTable(data);
  }

  async function load({ userInitiated = false } = {}) {
    if (inflight) return;
    inflight = true;
    const range = currentRange();
    if (userInitiated) $('metrics-body').classList.add('loading');
    try {
      const res = await fetch(`/api/dashboard?from=${range.from}&to=${range.to}`);
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Error ${res.status}`);
      lastData = data;
      render(data);
      $('range-status').textContent = range.live
        ? `Actualizado ${clock(Date.now() / 1000, true)} · se actualiza cada ${REFRESH_MS / 1000} s`
        : `Del ${dateTime(range.from)} al ${dateTime(range.to)}`;
    } catch (err) {
      $('range-status').textContent = `No se pudieron cargar las métricas: ${err.message}`;
    } finally {
      inflight = false;
      $('metrics-body').classList.remove('loading');
    }
  }

  function schedule() {
    clearInterval(timer);
    timer = active && currentRange().live ? setInterval(load, REFRESH_MS) : null;
  }

  function selectPreset(button) {
    preset = button.dataset.range;
    for (const b of $('range-presets').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b === button));
    $('custom-range').hidden = preset !== 'custom';
    if (preset === 'custom') {
      const range = custom ?? { from: Date.now() / 1000 - 3600, to: Date.now() / 1000 };
      $('range-from').value = toLocalInput(range.from);
      $('range-to').value = toLocalInput(range.to);
      $('range-from').focus();
      return;  // se carga al tocar "Aplicar"
    }
    load({ userInitiated: true });
    schedule();
  }

  for (const button of $('range-presets').querySelectorAll('button')) {
    button.addEventListener('click', () => selectPreset(button));
  }
  $('custom-range').addEventListener('submit', (event) => {
    event.preventDefault();
    const from = new Date($('range-from').value).getTime() / 1000;
    const to = new Date($('range-to').value).getTime() / 1000;
    if (!(from < to)) {
      $('range-status').textContent = 'El inicio tiene que ser anterior al fin.';
      return;
    }
    custom = { from, to };
    load({ userInitiated: true });
    schedule();
  });

  let resizeFrame = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => active && lastData && renderCharts(lastData));
  }).observe($('timeline-chart'));

  return {
    setActive(value) {
      active = value;
      if (value) load({ userInitiated: !lastData });
      schedule();
    },
  };
}
