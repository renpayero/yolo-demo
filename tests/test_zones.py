import pytest

from app.zones import RESTRICTED_COLOR, Zone, anchor_point, point_in_polygon

SQUARE = ((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6))
# Forma de L (cóncava): el hueco de arriba a la derecha queda afuera.
L_SHAPE = ((0.0, 0.0), (0.4, 0.0), (0.4, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))


@pytest.mark.parametrize(
    ("point", "inside"),
    [
        ((0.4, 0.4), True),
        ((0.1, 0.4), False),
        ((0.4, 0.7), False),
        ((0.59, 0.21), True),
    ],
)
def test_point_in_square(point, inside):
    assert point_in_polygon(point, SQUARE) is inside


@pytest.mark.parametrize(
    ("point", "inside"),
    [
        ((0.2, 0.3), True),    # brazo vertical de la L
        ((0.8, 0.8), True),    # brazo horizontal
        ((0.8, 0.3), False),   # el hueco
    ],
)
def test_point_in_concave_polygon(point, inside):
    assert point_in_polygon(point, L_SHAPE) is inside


def test_anchor_points_are_normalized():
    bbox = (100.0, 50.0, 300.0, 450.0)  # persona de 200x400 px en un frame de 1000x500
    assert anchor_point(bbox, 1000, 500, "center") == pytest.approx((0.2, 0.5))
    assert anchor_point(bbox, 1000, 500, "feet") == pytest.approx((0.2, 0.9))
    assert anchor_point(bbox, 1000, 500, "torso") == pytest.approx((0.2, (50 + 0.35 * 400) / 500))


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError):
        anchor_point((0, 0, 10, 10), 100, 100, "cabeza")


def test_zone_colors():
    work = Zone(id=1, name="Lijado", kind="work", polygon=SQUARE)
    other = Zone(id=2, name="Embalaje", kind="work", polygon=SQUARE)
    restricted = Zone(id=3, name="Prohibido", kind="restricted", polygon=SQUARE)
    assert work.color != other.color
    assert restricted.color == RESTRICTED_COLOR
