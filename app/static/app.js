import { connectState } from './state.js';
import { createLivePanel, renderConnectionPill } from './live.js';
import { createZonesEditor } from './zones-editor.js';

const live = createLivePanel();
const panels = { 'panel-live': live, 'panel-zones': createZonesEditor() };
const pill = document.getElementById('conn-pill');

// ---------- Pestañas ----------
const tabs = [...document.querySelectorAll('[role="tab"]')];

function selectTab(tab) {
  for (const t of tabs) {
    const selected = t === tab;
    t.setAttribute('aria-selected', String(selected));
    t.tabIndex = selected ? 0 : -1;
    document.getElementById(t.dataset.panel).hidden = !selected;
    panels[t.dataset.panel]?.setActive(selected);
  }
}

for (const tab of tabs) {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    const next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
    next.focus();
    selectTab(next);
  });
}
selectTab(tabs[0]);

// ---------- Estado en vivo ----------
let serverConnected = false;
let lostConnection = false;
let lastState = null;

connectState({
  onConnection(connected) {
    if (connected && lostConnection) live.restartStream();
    lostConnection = !connected;
    serverConnected = connected;
    if (!connected) lastState = null;
    renderConnectionPill(pill, { serverConnected, state: lastState });
  },
  onState(state) {
    lastState = state;
    live.render(state);
    renderConnectionPill(pill, { serverConnected, state });
    // Contador de alertas en la pestaña: se ve aunque estés editando zonas.
    const count = state.alerts.active.length;
    const badge = document.getElementById('alert-badge');
    badge.hidden = count === 0;
    badge.textContent = `⚠ ${count}`;
    badge.title = `${count} alerta(s) activa(s)`;
  },
});
