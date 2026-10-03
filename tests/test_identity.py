"""Identificación: galería de huellas faciales y votación por track (sin modelos ni cámara)."""
import numpy as np
import pytest

from app.identity import Gallery, IdentityVoter

RNG = np.random.default_rng(7)


def unit(v):
    return v / np.linalg.norm(v)


def around(center, noise=0.15, n=1):
    """Vectores cerca de `center` (la misma cara con otra luz/ángulo)."""
    return np.array([unit(center + RNG.normal(0, noise / np.sqrt(128), 128)) for _ in range(n)])


ANA = unit(RNG.normal(size=128))
BETO = unit(RNG.normal(size=128))
GALLERY = Gallery({1: around(ANA, n=5), 2: around(BETO, n=5)})


# ---------- Galería ----------

def test_matches_the_enrolled_person():
    match = GALLERY.match(around(ANA)[0], threshold=0.4, min_margin=0.08)
    assert match.person_id == 1 and match.score > 0.8 and match.second < 0.4


def test_stranger_is_not_matched():
    stranger = unit(RNG.normal(size=128))
    match = GALLERY.match(stranger, threshold=0.4, min_margin=0.08)
    assert match.person_id is None and match.score < 0.4


def test_ambiguous_match_is_rejected():
    """Una cara que se parece por igual a dos personas no se asigna a ninguna."""
    halfway = unit(ANA + BETO)
    match = GALLERY.match(halfway, threshold=0.4, min_margin=0.08)
    assert match.score >= 0.4 and match.score - match.second < 0.08
    assert match.person_id is None


def test_empty_gallery():
    match = Gallery({}).match(ANA, threshold=0.4, min_margin=0.08)
    assert match.person_id is None and match.score == 0.0
    assert len(Gallery({})) == 0 and len(GALLERY) == 2


# ---------- Votación ----------

def test_identity_needs_several_agreeing_observations():
    voter = IdentityVoter(votes_needed=3, window=5)
    assert voter.observe(7, 1) is None
    assert voter.observe(7, None) is None    # una observación dudosa no suma ni resta
    assert voter.observe(7, 1) is None
    assert voter.observe(7, 1) == 1          # tercera coincidencia: se asigna
    assert voter.identity(7) == 1


def test_mixed_votes_do_not_assign():
    voter = IdentityVoter(votes_needed=3, window=5)
    for person in (1, 2, 1, 2, None):
        assert voter.observe(7, person) is None
    assert voter.identity(7) is None


def test_old_votes_leave_the_window():
    voter = IdentityVoter(votes_needed=3, window=3)
    for person in (1, 1, None, None):
        voter.observe(7, person)
    assert voter.observe(7, 1) is None       # los dos primeros "1" ya salieron de la ventana


def test_identity_is_sticky_for_the_track():
    voter = IdentityVoter(votes_needed=2, window=5)
    voter.observe(7, 1)
    voter.observe(7, 1)
    for _ in range(5):
        assert voter.observe(7, 2) is None
    assert voter.identity(7) == 1


def test_a_person_is_only_on_one_track():
    """Si el tracker le da otro ID a la misma persona, la identidad pasa al track nuevo."""
    voter = IdentityVoter(votes_needed=2, window=5)
    voter.observe(7, 1)
    voter.observe(7, 1)
    voter.observe(9, 1)
    assert voter.observe(9, 1) == 1
    assert voter.identity(9) == 1 and voter.identity(7) is None


def test_transfer_forget_and_forget_person():
    voter = IdentityVoter(votes_needed=1, window=5)
    voter.observe(7, 1)
    voter.transfer(7, 8)                     # reasignación de ID del tracker
    assert voter.identity(8) == 1 and voter.identity(7) is None
    voter.forget(8)
    assert voter.identity(8) is None

    voter.observe(3, 2)
    voter.forget_person(2)                   # persona borrada
    assert voter.identity(3) is None
    assert voter.assignments() == {}


def test_verification_revokes_after_repeated_contradictions():
    """Si el tracker le pasa el ID a otra persona, la cara deja de coincidir y el nombre se retira."""
    voter = IdentityVoter(votes_needed=1, window=5, contradictions_to_revoke=2)
    voter.observe(7, 1)
    assert voter.verify(7, 1) is False           # sigue siendo Ana
    assert voter.verify(7, 2) is False           # una contradicción sola no alcanza
    assert voter.verify(7, 1) is False           # y una coincidencia la borra
    assert voter.verify(7, None) is False
    assert voter.verify(7, 2) is True            # dos seguidas: se retira
    assert voter.identity(7) is None
    assert voter.observe(7, 2) == 2              # vuelve a votar desde cero


@pytest.mark.parametrize("bad", [0, 6])
def test_votes_needed_must_fit_the_window(bad):
    with pytest.raises(ValueError):
        IdentityVoter(votes_needed=bad, window=5)
