// Pestaña "Personas": interruptor de identificación, registro con consentimiento y lista de registrados.
// Los nombres vienen de datos: se insertan siempre con textContent.

const $ = (id) => document.getElementById(id);

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg.replace(/^Value error, /, '')).join(' · ') : data.detail;
    throw new Error(detail || `Error ${res.status}`);
  }
  return data;
}

const date = (ts) =>
  new Date(ts * 1000).toLocaleString('es-AR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });

export function createPersonsPanel() {
  let enabled = false;
  let busy = false;   // hay una captura en curso (una sola a la vez)

  function renderSwitch(value) {
    enabled = value;
    $('ident-switch').setAttribute('aria-checked', String(value));
    $('ident-switch-label').textContent = value ? 'Activa' : 'Apagada';
  }

  function setBusy(value, message = '') {
    busy = value;
    $('enroll-submit').disabled = value;
    for (const b of $('person-list').querySelectorAll('button')) b.disabled = value;
    $('enroll-status').textContent = message;
  }

  function personItem(person) {
    const item = document.createElement('li');
    item.className = 'person-item';
    const name = document.createElement('p');
    name.className = 'person-name';
    name.textContent = person.name;
    const meta = document.createElement('p');
    meta.className = 'person-meta';
    meta.textContent = `${person.samples} muestra(s) · consentimiento ${date(person.consent_ts)}`;
    const actions = document.createElement('div');
    actions.className = 'form-actions';
    const more = document.createElement('button');
    more.type = 'button';
    more.className = 'btn';
    more.textContent = 'Más muestras';
    more.title = 'Capturar otra vez (otra luz u otro ángulo) mejora el reconocimiento';
    more.addEventListener('click', () => addSamples(person));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'btn btn-danger';
    remove.textContent = 'Eliminar';
    let armed = null;
    remove.addEventListener('click', async () => {
      if (!armed) {  // dos pasos, sin diálogos: el botón no cambia de lugar
        remove.textContent = 'Confirmar';
        remove.classList.add('armed');
        armed = setTimeout(() => { remove.textContent = 'Eliminar'; remove.classList.remove('armed'); armed = null; }, 4000);
        return;
      }
      clearTimeout(armed);
      try {
        const report = await api('DELETE', `/api/persons/${person.id}`);
        $('enroll-status').textContent =
          `Se eliminó a ${person.name} y su huella. Quedaron anónimas ${report.visits_anonymized} visita(s) de su historial.`;
      } catch (err) {
        $('enroll-error').textContent = err.message;
      }
      await loadPersons();
    });
    actions.append(more, remove);
    item.append(name, meta, actions);
    return item;
  }

  async function loadPersons() {
    const persons = await api('GET', '/api/persons');
    $('person-list').replaceChildren(...persons.map(personItem));
    $('person-list-empty').hidden = persons.length > 0;
    for (const b of $('person-list').querySelectorAll('button')) b.disabled = busy;
  }

  async function capture(run, doneMessage) {
    $('enroll-error').textContent = '';
    setBusy(true, 'Capturando… mirá de frente a la cámara.');
    try {
      const person = await run();
      setBusy(false, doneMessage(person));
    } catch (err) {
      setBusy(false);
      $('enroll-error').textContent = err.message;
    }
    await loadPersons();
  }

  function addSamples(person) {
    if (busy) return;
    capture(
      () => api('POST', `/api/persons/${person.id}/samples`),
      (p) => `Listo: ${p.name} ahora tiene ${p.samples} muestras.`,
    );
  }

  $('enroll-form').addEventListener('submit', (event) => {
    event.preventDefault();
    if (busy) return;
    const name = $('enroll-name').value.trim();
    if (!name) {
      $('enroll-error').textContent = 'Poné el nombre de la persona.';
      $('enroll-name').focus();
      return;
    }
    if (!$('enroll-consent').checked) {
      $('enroll-error').textContent = 'Sin el consentimiento de la persona no se la puede registrar.';
      $('enroll-consent').focus();
      return;
    }
    capture(
      () => api('POST', '/api/persons', { name, consent: true }),
      (p) => {
        $('enroll-name').value = '';
        $('enroll-consent').checked = false;
        return `Registrada: ${p.name} (${p.samples} muestras).${enabled ? '' : ' Prendé la identificación para que el sistema la reconozca.'}`;
      },
    );
  });

  $('ident-switch').addEventListener('click', async () => {
    try {
      renderSwitch((await api('PUT', '/api/identification', { enabled: !enabled })).enabled);
    } catch (err) {
      $('enroll-error').textContent = `No se pudo cambiar la identificación: ${err.message}`;
    }
  });

  return {
    setActive(value) {
      if (value) loadPersons().catch((err) => { $('enroll-error').textContent = err.message; });
    },
    // El estado en vivo trae si la identificación está activa (por si se cambió desde otra ventana).
    sync(state) {
      if (state.identification && state.identification.enabled !== enabled) renderSwitch(state.identification.enabled);
    },
  };
}
