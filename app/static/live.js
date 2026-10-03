// Pestaña "En vivo": video MJPEG y panel de estado.
import { formatDuration, plural } from './format.js';

const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';
const $ = (id) => document.getElementById(id);

const PHASE_LABEL = {
  starting: 'Iniciando',
  running: 'En vivo',
  no_signal: 'Sin señal de la cámara',
  error: 'Error',
};
const PHASE_LEVEL = { starting: 'idle', running: 'ok', no_signal: 'warn', error: 'danger' };

export function createLivePanel() {
  const video = $('live-video');
  let active = true;     // la pestaña está a la vista
  let playing = false;

  // El stream ocupa una de las 6 conexiones por servidor que permite el navegador:
  // se corta al cambiar de pestaña o al ocultar la página.
  const sync = () => {
    const shouldPlay = active && document.visibilityState === 'visible';
    if (shouldPlay && !playing) {
      video.src = `/stream.mjpg?t=${Date.now()}`;
      playing = true;
    } else if (!shouldPlay && playing) {
      video.src = BLANK;
      playing = false;
    }
  };
  document.addEventListener('visibilitychange', sync);
  sync();

  return {
    setActive(value) {
      active = value;
      sync();
    },
    // Tras una caída del servidor el MJPEG queda congelado: se reabre al volver la conexión.
    restartStream() {
      playing = false;
      sync();
    },
    render(state) {
      const cam = state.camera;
      $('st-people').textContent = state.phase === 'running' ? state.people : '—';
      $('st-camera').textContent = cam.name;
      $('st-source').textContent = cam.source;
      $('st-resolution').textContent = cam.width ? `${cam.width} × ${cam.height}` : '—';
      $('st-fps').textContent = state.phase === 'running' ? `${state.fps.toFixed(1)} fps` : '—';
      $('st-device').textContent = state.device ? (state.device.startsWith('cuda') ? 'GPU (CUDA)' : 'CPU') : '—';
      $('st-model').textContent = state.model ?? '—';
      $('st-reconnects').textContent = cam.reconnects;
      renderZones(state.zones);
    },
  };
}

function renderZones(zones) {
  $('live-zones-empty').hidden = zones.length > 0;
  $('live-zones').replaceChildren(
    ...zones.map((zone) => {
      const item = document.createElement('li');
      item.className = 'live-zone';
      item.dataset.occupied = String(zone.count > 0);
      item.innerHTML =
        '<span class="swatch" aria-hidden="true"></span><span class="zone-name"></span>' +
        '<span class="zone-status"></span><span class="zone-occupants"></span>';
      item.querySelector('.swatch').style.background = zone.color;
      item.querySelector('.zone-name').textContent = zone.kind === 'restricted' ? `⚠ ${zone.name}` : zone.name;
      item.querySelector('.zone-status').textContent = zone.count
        ? plural(zone.count, 'persona', 'personas')
        : `vacía ${zone.empty_for_s != null ? formatDuration(zone.empty_for_s) : ''}`;
      item.querySelector('.zone-occupants').textContent = zone.occupants
        .map((o) => `ID ${o.track_id} · ${formatDuration(o.dwell_s)}`)
        .join('   ');
      return item;
    }),
  );
}

export function renderConnectionPill(pill, { serverConnected, state }) {
  let level = 'danger';
  let text = 'Sin conexión con el servidor';
  if (serverConnected && state) {
    level = PHASE_LEVEL[state.phase] ?? 'idle';
    text = PHASE_LABEL[state.phase] ?? state.phase;
    if (state.phase === 'error' && state.error) text = `Error: ${state.error}`;
  } else if (serverConnected) {
    level = 'idle';
    text = 'Conectando…';
  }
  pill.dataset.level = level;
  pill.querySelector('.pill-text').textContent = text;
}
