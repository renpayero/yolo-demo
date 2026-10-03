// Estado en vivo del servidor por WebSocket, con reconexión automática.

export function connectState({ onState, onConnection }) {
  let retryMs = 1000;

  const open = () => {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${scheme}://${location.host}/ws`);
    ws.addEventListener('open', () => {
      retryMs = 1000;
      onConnection(true);
    });
    ws.addEventListener('message', (event) => onState(JSON.parse(event.data)));
    ws.addEventListener('close', () => {
      onConnection(false);
      setTimeout(open, retryMs);
      retryMs = Math.min(retryMs * 2, 10000);
    });
  };

  open();
}
