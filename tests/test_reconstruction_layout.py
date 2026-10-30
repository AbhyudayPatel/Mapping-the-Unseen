"""Regression checks for production-diagram UI at print and preview resolutions."""
import pytest

from paper.bench.make_v4_recon import (
    BODY_SIZE, DIAGRAMS, FOOT_SIZE, PAD, TEXT_GAP, Link, create_diagram,
)


@pytest.mark.parametrize("diagram", DIAGRAMS, ids=lambda d: d.name)
@pytest.mark.parametrize("dpi", [72, 200])
def test_all_reconstruction_text_and_routes_fit(diagram, dpi):
    scene = create_diagram(diagram)
    try:
        scene.fig.set_dpi(dpi)
        qa = scene.validate()
        assert qa["cards"] == 10 and qa["links"] == 6
        assert qa["style"] == "v2 Excalidraw-inspired"
        assert qa["min_text_gap_pt"] >= TEXT_GAP - 0.1
        assert qa["min_inner_padding_pt"] >= PAD - 0.1
        # No auto-shrinking hides problems by making the text unreadable.
        assert {c.texts[1].get_fontsize() for c in scene.cards.values()} == {BODY_SIZE}
        assert {c.texts[2].get_fontsize() for c in scene.cards.values()} == {FOOT_SIZE}
        # The boundary is not a computational stage: do not hang unrelated cards off it.
        assert all("boundary" not in (e.source, e.target) for e in scene.links)
    finally:
        scene.close()


def test_validation_catches_footer_collision():
    scene = create_diagram(DIAGRAMS[2])
    try:
        body, foot = scene.cards["center0"].texts[1:]
        foot.set_position(body.get_position())
        with pytest.raises(AssertionError, match="text blocks collide"):
            scene.validate()
    finally:
        scene.close()


def test_validation_catches_connector_inside_card():
    scene = create_diagram(DIAGRAMS[0])
    try:
        old = scene.links[0]
        a, _, _, b = old.points
        # Regression: the old elbow ran vertically INSIDE the central card edge.
        scene.links[0] = Link(old.source, old.target,
                             [a, (b[0] + 2, a[1]), (b[0] + 2, b[1]), b], old.kind)
        with pytest.raises(AssertionError, match="crosses"):
            scene.validate()
    finally:
        scene.close()


def test_validation_catches_floating_connector():
    scene = create_diagram(DIAGRAMS[1])
    try:
        scene.links[0].points[0] = (0, 0)
        with pytest.raises(AssertionError, match="Floating source"):
            scene.validate()
    finally:
        scene.close()


def test_wrapping_measures_actual_glyph_width():
    scene = create_diagram(DIAGRAMS[0])
    try:
        narrow = scene.wrap("iiii iiii iiii", 35, BODY_SIZE)
        wide = scene.wrap("WWWW WWWW WWWW", 35, BODY_SIZE)
        assert narrow.count("\n") < wide.count("\n")
        for line in wide.splitlines():
            assert scene.measure(line, BODY_SIZE)[0] <= 35
    finally:
        scene.close()
