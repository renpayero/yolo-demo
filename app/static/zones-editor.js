// Pestaña "Zonas": editor de polígonos sobre una imagen congelada de la cámara.
// Las coordenadas se guardan normalizadas (0-1), así la zona sirve con cualquier resolución.

const $ = (id) => document.getElementById(id);

const DRAFT_COLOR = '#f2a33a';
const HANDLE_RADIUS = 6;
const CLOSE_DISTANCE = 12;      // px: click cerca del primer vértice cierra el polígono
const DUPLICATE_DISTANCE = 4;   // px: el doble click genera clicks repetidos en el mismo punto
const MIN_VERTICES = 3;

const HINTS = {
  empty: 'Tocá «Nueva zona» para empezar.',
  idle: 'Hacé click en una zona para editarla, o tocá «Nueva zona».',
  drawing: 'Click: agregar vértice · Doble click, Enter o click en el primer vértice: cerrar · Retroceso o Z: deshacer · Esc: cancelar',
  form: 'Arrastrá los vértices para ajustar la forma. Poné un nombre y guardá.',
};

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errorMessage(data, res.status));
  return data;
}

function errorMessage(data, status) {
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail) && data.detail.length) {
    return data.detail.map((d) => d.msg.replace(/^Value error, /, '')).join(' · ');
  }
  return `Error ${status}`;
}

function pointInPolygon([x, y], polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function createZonesEditor() {
  const canvas = $('zone-canvas');
  const ctx = canvas.getContext('2d');
  const image = new Image();
  const form = $('zone-form');

  let zones = [];
  let mode = 'idle';           // idle | drawing | form
  let editing = null;          // { id|null, points: [[x,y]...] } la zona que se dibuja o edita
  let dragIndex = null;
  let hover = null;            // posición del mouse en px, para la línea guía al dibujar
  let imageOk = false;
  let rect = { x: 0, y: 0, w: 1, h: 1 };  // dónde se dibuja la imagen dentro del canvas (px CSS)
  let active = false;

  // ---------- Geometría de pantalla ----------

  function layout() {
    const box = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(box.width * dpr);
    canvas.height = Math.round(box.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const iw = imageOk ? image.naturalWidth : 16;
    const ih = imageOk ? image.naturalHeight : 9;
    const scale = Math.min(box.width / iw, box.height / ih);
    rect = { w: iw * scale, h: ih * scale, x: (box.width - iw * scale) / 2, y: (box.height - ih * scale) / 2 };
  }

  const toPx = ([nx, ny]) => [rect.x + nx * rect.w, rect.y + ny * rect.h];
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function eventPoint(event) {
    const box = canvas.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  const toNorm = ([px, py]) => [clamp01((px - rect.x) / rect.w), clamp01((py - rect.y) / rect.h)];
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  // ---------- Dibujo ----------

  function tracePolygon(points) {
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = toPx(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
  }

  function drawZoneShape(points, color, { fill, label, width = 2 }) {
    tracePolygon(points);
    ctx.closePath();
    if (fill) {
      ctx.globalAlpha = 0.22;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.stroke();
    if (label) drawLabel(points, label, color);
  }

  function drawLabel(points, text, color) {
    const top = points.reduce((a, b) => (b[1] < a[1] ? b : a));
    const [x, y] = toPx(top);
    ctx.font = '600 13px system-ui, sans-serif';
    const w = ctx.measureText(text).width + 12;
    const h = 20;
    const ly = y - h >= 0 ? y - h : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, w, h);
    ctx.fillStyle = '#000';
    ctx.fillText(text, x + 6, ly + 14);
  }

  function drawHandles(points) {
    points.forEach((p, i) => {
      const [x, y] = toPx(p);
      ctx.beginPath();
      ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 && mode === 'drawing' ? DRAFT_COLOR : '#fff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#000';
      ctx.stroke();
    });
  }

  function draw() {
    const box = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, box.width, box.height);
    if (imageOk) {
      ctx.drawImage(image, rect.x, rect.y, rect.w, rect.h);
    } else {
      ctx.fillStyle = '#20252c';
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = '#9aa4b2';
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No hay imagen de la cámara. ¿Está corriendo webcam.bat?', rect.x + rect.w / 2, rect.y + rect.h / 2);
      ctx.textAlign = 'start';
    }

    for (const zone of zones) {
      if (editing && editing.id === zone.id) continue;
      drawZoneShape(zone.polygon, zone.color, { fill: true, label: zone.name });
    }
    if (!editing) return;

    const color = editing.id ? zones.find((z) => z.id === editing.id)?.color ?? DRAFT_COLOR : DRAFT_COLOR;
    if (mode === 'drawing') {
      tracePolygon(editing.points);
      if (hover && editing.points.length) ctx.lineTo(...hover);
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = DRAFT_COLOR;
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      drawZoneShape(editing.points, color, { fill: true, width: 3 });
    }
    drawHandles(editing.points);
  }

  // ---------- Estado de la UI ----------

  function setHint(extra = '') {
    let text = HINTS[mode === 'idle' ? (zones.length ? 'idle' : 'empty') : mode];
    if (mode === 'drawing') text = `${text} — ${editing.points.length} vértice(s)`;
    $('zone-hint').textContent = extra ? `${text} ${extra}` : text;
  }

  function renderList() {
    const list = $('zone-list');
    list.replaceChildren(
      ...zones.map((zone) => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'zone-item';
        button.setAttribute('aria-pressed', String(editing?.id === zone.id));
        button.innerHTML = '<span class="swatch" aria-hidden="true"></span><span class="zone-name"></span><span class="badge"></span>';
        button.querySelector('.swatch').style.background = zone.color;
        button.querySelector('.zone-name').textContent = zone.name;
        button.querySelector('.badge').textContent = zone.kind === 'restricted' ? 'Restringida' : 'Trabajo';
        button.addEventListener('click', () => select(zone));
        item.append(button);
        return item;
      }),
    );
    $('zone-list-empty').hidden = zones.length > 0;
  }

  function showForm(zone) {
    $('zone-form-card').hidden = false;
    $('zone-form-title').textContent = zone ? 'Editar zona' : 'Nueva zona';
    $('zone-name').value = zone?.name ?? '';
    form.elements['zone-kind'].value = zone?.kind ?? 'work';
    $('zone-delete').hidden = !zone;
    resetDeleteButton();
    $('zone-error').textContent = '';
    $('zone-name').focus();
  }

  function hideForm() {
    $('zone-form-card').hidden = true;
  }

  function refresh() {
    renderList();
    setHint();
    draw();
  }

  // ---------- Acciones ----------

  async function loadZones() {
    zones = await api('GET', '/api/zones');
    refresh();
  }

  function loadImage() {
    image.src = `/api/snapshot.jpg?t=${Date.now()}`;
  }

  function startDrawing() {
    editing = { id: null, points: [] };
    mode = 'drawing';
    hideForm();
    refresh();
  }

  function finishDrawing() {
    if (editing.points.length < MIN_VERTICES) {
      setHint(`(hacen falta al menos ${MIN_VERTICES} vértices)`);
      return;
    }
    mode = 'form';
    showForm(null);
    refresh();
  }

  function select(zone) {
    editing = { id: zone.id, points: zone.polygon.map((p) => [...p]) };
    mode = 'form';
    showForm(zone);
    refresh();
  }

  function cancel() {
    editing = null;
    mode = 'idle';
    dragIndex = null;
    hideForm();
    refresh();
  }

  async function save(event) {
    event.preventDefault();
    const body = {
      name: $('zone-name').value,
      kind: form.elements['zone-kind'].value,
      polygon: editing.points,
    };
    try {
      if (editing.id) await api('PUT', `/api/zones/${editing.id}`, body);
      else await api('POST', '/api/zones', body);
    } catch (err) {
      $('zone-error').textContent = err.message;
      return;
    }
    cancel();
    await loadZones();
  }

  // Borrado en dos pasos, sin diálogos modales. El botón no cambia de lugar entre el primer y el segundo click.
  let deleteArmed = null;
  function resetDeleteButton() {
    clearTimeout(deleteArmed);
    deleteArmed = null;
    $('zone-delete').textContent = 'Eliminar';
    $('zone-delete').classList.remove('armed');
  }

  async function remove() {
    if (!deleteArmed) {
      $('zone-delete').textContent = 'Confirmar';
      $('zone-delete').classList.add('armed');
      deleteArmed = setTimeout(resetDeleteButton, 4000);
      return;
    }
    resetDeleteButton();
    try {
      await api('DELETE', `/api/zones/${editing.id}`);
    } catch (err) {
      $('zone-error').textContent = err.message;
      return;
    }
    cancel();
    await loadZones();
  }

  async function importFile(file) {
    try {
      const data = JSON.parse(await file.text());
      await api('POST', '/api/zones/import', data);
      cancel();
      await loadZones();
      setHint(`Se importaron ${zones.length} zona(s).`);
    } catch (err) {
      setHint(`No se pudo importar: ${err.message}`);
    }
  }

  // ---------- Mouse y teclado ----------

  function onPointerDown(event) {
    // Sin esto, el mousedown posterior le saca el foco al campo de nombre cuando el click cierra el polígono.
    event.preventDefault();
    if (!imageOk && mode === 'idle') return;
    const p = eventPoint(event);

    if (mode === 'drawing') {
      const pts = editing.points;
      if (pts.length >= MIN_VERTICES && distance(p, toPx(pts[0])) <= CLOSE_DISTANCE) {
        finishDrawing();
      } else if (!pts.length || distance(p, toPx(pts[pts.length - 1])) > DUPLICATE_DISTANCE) {
        pts.push(toNorm(p));
        refresh();
      }
      return;
    }

    if (mode === 'form') {
      const index = editing.points.findIndex((q) => distance(p, toPx(q)) <= HANDLE_RADIUS + 3);
      if (index >= 0) {
        dragIndex = index;
        canvas.setPointerCapture(event.pointerId);
        return;
      }
    }

    // Click sobre una zona: seleccionarla (la última dibujada queda arriba).
    const hit = [...zones].reverse().find((z) => pointInPolygon(toNorm(p), z.polygon));
    if (hit && hit.id !== editing?.id) select(hit);
  }

  function onPointerMove(event) {
    const p = eventPoint(event);
    if (dragIndex !== null) {
      editing.points[dragIndex] = toNorm(p);
      draw();
    } else if (mode === 'drawing') {
      hover = p;
      draw();
    }
    const overHandle = mode === 'form' && editing.points.some((q) => distance(p, toPx(q)) <= HANDLE_RADIUS + 3);
    canvas.style.cursor = dragIndex !== null || overHandle ? 'grabbing' : mode === 'drawing' ? 'crosshair' : 'default';
  }

  function onPointerUp() {
    dragIndex = null;
  }

  function onKeyDown(event) {
    if (!active || event.target.closest('input, textarea, select')) return;
    if (mode === 'drawing') {
      if (event.key === 'Enter') finishDrawing();
      else if (event.key === 'Backspace' || event.key.toLowerCase() === 'z') {
        editing.points.pop();
        refresh();
      } else if (event.key === 'Escape') cancel();
      else return;
      event.preventDefault();
    } else if (mode === 'form' && event.key === 'Escape') {
      cancel();
    }
  }

  // ---------- Inicialización ----------

  image.addEventListener('load', () => {
    imageOk = true;
    layout();
    draw();
  });
  image.addEventListener('error', () => {
    imageOk = false;
    layout();
    draw();
  });

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', () => {
    hover = null;
    if (mode === 'drawing') draw();
  });
  canvas.addEventListener('dblclick', () => mode === 'drawing' && finishDrawing());
  document.addEventListener('keydown', onKeyDown);
  new ResizeObserver(() => {
    layout();
    draw();
  }).observe(canvas);

  $('zone-new').addEventListener('click', startDrawing);
  $('zone-refresh').addEventListener('click', loadImage);
  $('zone-cancel').addEventListener('click', cancel);
  $('zone-delete').addEventListener('click', remove);
  form.addEventListener('submit', save);
  $('zone-import').addEventListener('change', (event) => {
    const [file] = event.target.files;
    if (file) importFile(file);
    event.target.value = '';
  });

  return {
    setActive(value) {
      active = value;
      if (value) {
        loadImage();
        loadZones().catch((err) => setHint(`No se pudieron cargar las zonas: ${err.message}`));
      }
    },
  };
}
