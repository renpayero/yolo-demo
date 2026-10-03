// Formatos compartidos por las pestañas.

/** 12 → "0:12" · 754 → "12:34" · 3754 → "1:02:34" (igual que en el video). */
export function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

/** Para indicadores: 45 → "45 s" · 754 → "12 min" · 3754 → "1 h 02 min". */
export function humanDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total} s`;
  if (total < 3600) return `${Math.floor(total / 60)} min`;
  const h = Math.floor(total / 3600);
  return `${h} h ${String(Math.floor((total % 3600) / 60)).padStart(2, '0')} min`;
}

export function plural(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}
