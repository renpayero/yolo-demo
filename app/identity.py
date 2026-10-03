"""Identificación de personas registradas. Lógica pura (numpy, sin OpenCV ni cámara).

- `Gallery`: huellas faciales (vectores de 128 valores, normalizados) de quienes dieron su consentimiento.
  Una cara se asigna a la persona más parecida solo si supera el umbral Y le saca margen a la segunda: así
  una cara ambigua queda como "no identificada" en vez de arriesgar un nombre equivocado.
- `IdentityVoter`: un solo parecido no alcanza; un track recibe un nombre cuando varias observaciones
  recientes coinciden. Después se lo re-verifica de vez en cuando: si la cara deja de coincidir (el tracker
  le pasó el ID a otra persona), el nombre se retira.
"""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    person_id: int | None   # None: no supera el umbral o es ambigua
    score: float            # similitud coseno con la persona más parecida
    second: float           # similitud con la segunda más parecida


class Gallery:
    def __init__(self, embeddings: Mapping[int, np.ndarray]) -> None:
        self._embeddings = {pid: np.atleast_2d(vectors) for pid, vectors in embeddings.items() if len(vectors)}

    def __len__(self) -> int:
        return len(self._embeddings)

    def match(self, feature: np.ndarray, threshold: float, min_margin: float) -> Match:
        if not self._embeddings:
            return Match(None, 0.0, 0.0)
        scores = sorted(
            ((float(np.max(vectors @ feature)), pid) for pid, vectors in self._embeddings.items()), reverse=True
        )
        best, person_id = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        accepted = best >= threshold and best - second >= min_margin
        return Match(person_id if accepted else None, best, second)


class IdentityVoter:
    def __init__(self, votes_needed: int = 3, window: int = 5, contradictions_to_revoke: int = 2) -> None:
        if not 1 <= votes_needed <= window:
            raise ValueError("votes_needed tiene que estar entre 1 y window")
        self._votes_needed = votes_needed
        self._window = window
        self._contradictions_to_revoke = contradictions_to_revoke
        self._votes: dict[int, deque[int | None]] = {}
        self._assigned: dict[int, int] = {}        # track → persona
        self._contradictions: dict[int, int] = {}  # track identificado → chequeos seguidos que no coinciden

    def verify(self, track_id: int, observed: int | None) -> bool:
        """Re-verificación de un track ya identificado con una cara vista ahora (None: no coincide con nadie).

        Si el tracker le pasó el ID a otra persona, la cara deja de coincidir: tras varias contradicciones
        seguidas se retira el nombre (devuelve True) y el track vuelve a votación. Una coincidencia las reinicia.
        """
        assigned = self._assigned.get(track_id)
        if assigned is None:
            return False
        if observed == assigned:
            self._contradictions.pop(track_id, None)
            return False
        count = self._contradictions.get(track_id, 0) + 1
        if count >= self._contradictions_to_revoke:
            self.forget(track_id)
            return True
        self._contradictions[track_id] = count
        return False

    def observe(self, track_id: int, person_id: int | None) -> int | None:
        """Registra una observación. Devuelve la persona si con esta el track quedó identificado."""
        if track_id in self._assigned:
            return None
        votes = self._votes.setdefault(track_id, deque(maxlen=self._window))
        votes.append(person_id)
        counts = Counter(v for v in votes if v is not None)
        if not counts:
            return None
        winner, count = counts.most_common(1)[0]
        if count < self._votes_needed:
            return None
        # Una persona no puede estar en dos tracks: si ya tenía otro (ID viejo del tracker), pasa a este.
        for other in [t for t, p in self._assigned.items() if p == winner]:
            del self._assigned[other]
        self._assigned[track_id] = winner
        self._votes.pop(track_id, None)
        return winner

    def identity(self, track_id: int) -> int | None:
        return self._assigned.get(track_id)

    def assignments(self) -> dict[int, int]:
        return dict(self._assigned)

    def transfer(self, old_track: int, new_track: int) -> None:
        """El tracker le dio otro ID a la misma persona (reasignación en el motor de ocupación)."""
        if old_track in self._assigned:
            self._assigned[new_track] = self._assigned.pop(old_track)
            self._votes.pop(new_track, None)
            self._contradictions.pop(old_track, None)

    def forget(self, track_id: int) -> None:
        self._assigned.pop(track_id, None)
        self._votes.pop(track_id, None)
        self._contradictions.pop(track_id, None)

    def forget_person(self, person_id: int) -> None:
        for track_id in [t for t, p in self._assigned.items() if p == person_id]:
            del self._assigned[track_id]
        for votes in self._votes.values():
            for i, vote in enumerate(votes):
                if vote == person_id:
                    votes[i] = None

    def clear(self) -> None:
        self._votes.clear()
        self._assigned.clear()
        self._contradictions.clear()
