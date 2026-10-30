"""V2/Excalidraw-inspired reconstructions with checked typography and routing.

All coordinates are typographic points at the paper's full text width. Card heights
come from actual multiline text extents, not character counts or arbitrary shrinking.
Arrows denote conceptual functional order; other links are evidence associations,
NOT claims of measured physical wiring. The quantitative boundary is not a stage.

Run: PYTHONIOENCODING=utf-8 python paper/bench/make_v4_recon.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
QA = ROOT / "paper" / "bench" / "out" / "reconstruction_layout_qa.json"

# A4 minus the paper's two 23 mm margins. Full-width inclusion preserves font sizes.
WIDTH = 164 / 25.4 * 72
MARGIN, GUTTER, CENTER_WIDTH = 8.0, 20.0, 160.0
SIDE_WIDTH = (WIDTH - 2 * MARGIN - 2 * GUTTER - CENTER_WIDTH) / 2
PAD, TEXT_GAP, ROW_GAP = 9.0, 6.0, 17.0
LAYOUT_GAP = TEXT_GAP + 1.5  # Reserve for raster/PDF font-metric rounding differences.
TITLE_SIZE, BODY_SIZE, FOOT_SIZE = 9.1, 8.3, 7.3
INK, SOFT = "#25231F", "#777167"
FILLS = {"measured": "#E9F0EB", "mechanism": "#F5EDDF",
         "prior": "#F1F0EC", "U": "#F1F0EC", "aux": "#E8EDF2"}
EDGES = {"measured": "#6D9A80", "mechanism": "#C49B5E",
         "prior": "#A9A499", "U": "#A9A499", "aux": "#7C93AB"}
AVAILABLE_FONTS = {f.name for f in font_manager.fontManager.ttflist}
HAND_FONT = "Segoe Print" if "Segoe Print" in AVAILABLE_FONTS else "DejaVu Sans"


def sketch(artist):
    # Same subtle outline wobble as v2; glyphs and measured geometry stay unjittered.
    artist.set_sketch_params(scale=0.45, length=100, randomness=1.25)
    return artist


@dataclass(frozen=True)
class Spec:
    title: str
    lines: tuple[str, ...]
    grade: str
    foot: str


def spec(title, lines, grade, foot):
    return Spec(title, tuple(lines), grade, foot)


@dataclass(frozen=True)
class Diagram:
    name: str
    title: str
    center: tuple[Spec, ...]
    left: tuple[Spec, ...]
    right: tuple[Spec, ...]
    # Index of the central card each left/right evidence card annotates.
    left_targets: tuple[int, ...]
    right_targets: tuple[int, ...]
    boundary: Spec


DIAGRAMS = (
    Diagram(
        "gpt", "GPT-4o-mini | functional reconstruction",
        center=(
            spec("Tokenization", ["o200k count family", "base = harmony tie", "18/18 exact matches"],
                 "measured", "L2 · P02"),
            spec("Generation path", ["Autoregressive decoder", "Output-coupled latency"],
                 "measured", "L2 · P19"),
            spec("Output / readout", ["Generative LM head"], "measured", "L2 · P19"),
        ),
        left=(
            spec("Attention / state", ["Full-range recall to 8.2k tokens", "Kernel unseparated"],
                 "mechanism", "L4 · P06/P07 + follow-up"),
            spec("KV / position", ["GQA/MQA and RoPE", "Unchanged priors"],
                 "prior", "U · prior-only"),
            spec("Serving evidence", ["Repeat stability", "Cache/drafting signatures tested"],
                 "measured", "P08–P10 / P16"),
        ),
        right=(
            spec("Feed-forward", ["MoE 0.61 / dense 0.39", "Ambiguous; detector accuracy 88%"],
                 "mechanism", "U/L4 · P11 (weak signal)"),
            spec("Numeric regime", ["bf16-consistent wire grid: 0.76", "Compute dtype unresolved"],
                 "mechanism", "L4 wire · P04"),
            spec("Interface", ["16,393-token input cap", "Deployment fact, not model size"],
                 "measured", "L1 · P01/P05"),
        ),
        left_targets=(1, 1, 2), right_targets=(1, 1, 2),
        boundary=spec("Quantitative boundary - not an execution stage",
                      ["Width unresolved by top-5 access; depth unresolved by serving confounds."],
                      "U", "U · P03/P07 · next experiments in the boundary ledger"),
    ),
    Diagram(
        "deepseek", "DeepSeek-V3.2 | functional reconstruction",
        center=(
            spec("Tokenization", ["BBPE count family", "19/19 matches", "465-entry vocabulary gap"],
                 "measured", "L2 · P02"),
            spec("Generation path", ["Autoregressive decoder"], "measured", "L2 · P19"),
            spec("Output / readout", ["Generative LM head"], "measured", "L2 · P19"),
        ),
        left=(
            spec("Attention / state", ["Long-range; fresh prefill", "Exponent 0.57",
                                        "95% CI [0.41, 0.74]", "Kernel unseparated"],
                 "mechanism", "L4 · P07 + follow-up"),
            spec("KV / position", ["GQA/MQA, RoPE and MLA", "Unchanged priors"],
                 "prior", "U · prior-only"),
            spec("Serving evidence", ["Prefix-cache contrast 5–10×", "Repeated vs fresh prefix"],
                 "measured", "P09 + follow-up"),
        ),
        right=(
            spec("Feed-forward", ["MoE 0.82 / dense 0.18", "Detector accuracy 88%", "Published design: anchor"],
                 "mechanism", "L4 · P11 + control null"),
            spec("Numeric regime", ["bf16-consistent wire: 0.76", "Compute dtype unresolved"],
                 "mechanism", "L4 wire · P04"),
            spec("Interface", ["131,072 context (stated)", "Buffered streams; no ITL"],
                 "measured", "L1 · P01/P05"),
        ),
        left_targets=(1, 1, 2), right_targets=(1, 1, 2),
        boundary=spec("Quantitative boundary - not an execution stage",
                      ["Width, depth and expert count remain unresolved (U)."],
                      "U", "P03/P07 · access and confounds; next experiments in the boundary ledger"),
    ),
    Diagram(
        "jev", "JEV | decision-architecture reconstruction",
        center=(
            spec("State accounting", ["u(q,s) = 259", "+ state_cost(s) + 12.0q", "Affine ledger; R² = 1.00"],
                 "measured", "L1/L2 · P18/P22"),
            spec("Core candidate", ["Decoder-compatible: 0.76", "Alternatives retained"],
                 "mechanism", "L4 · P19 / knowledge"),
            spec("Decision readout", ["Single-pass-compatible heads: 0.94", "Fixed 38-token output"],
                 "mechanism", "L4 · P18/P22 controls"),
        ),
        left=(
            spec("Question branches", ["State effect +0.98", "Sibling effect 0.00", "n = 10; p = 2 × 10⁻⁴"],
                 "measured", "L2 · P18 powered"),
            spec("Tokenizer / serving", ["Reference unmatched: 58.3%", "Cache CI crosses zero"],
                 "U", "U · P02/P09"),
            spec("Knowledge probe", ["17 MCQs / 15 facts answered", "Broad core plausible"],
                 "mechanism", "L4 · P12"),
        ),
        right=(
            spec("Choice organization", ["Semantic options", "Jointly normalized", "Permutation-invariant"],
                 "measured", "L2 · P18"),
            spec("Sparsity / size", ["Dense/MoE 0.50/0.50 prior", "No routing signal"],
                 "prior", "U · P11 (prior-only)"),
            spec("Physical prefill", ["Shared-state accounting measured", "Physical sharing unresolved"],
                 "U", "U · attestation needed"),
        ),
        left_targets=(1, 1, 2), right_targets=(1, 1, 2),
        boundary=spec("Quantitative boundary - not an execution stage",
                      ["Width, depth and expert count remain unresolved (U)."],
                      "U", "Functional organization above does not establish physical weight sharing."),
    ),
)


@dataclass
class Card:
    key: str
    spec: Spec
    box: Bbox
    texts: list = field(default_factory=list)


@dataclass
class Link:
    source: str
    target: str
    points: list[tuple[float, float]]
    kind: str


class Scene:
    def __init__(self, name):
        self.name = name
        # Handwritten-font hinting differs substantially at small raster sizes.
        # Measure at preview resolution, then convert pixel extents back to points.
        self.fig = plt.figure(figsize=(WIDTH / 72, 8), dpi=200)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(0, 576))
        self.ax.set_axis_off()
        self.fig.canvas.draw()
        self.cards: dict[str, Card] = {}
        self.links: list[Link] = []
        self.labels: list = []

    def measure(self, text, size, weight="normal"):
        artist = self.ax.text(0, 0, text, fontsize=size, fontweight=weight,
                              linespacing=1.25, ha="left", va="top")
        bounds = artist.get_window_extent(self.fig.canvas.get_renderer())
        artist.remove()
        return bounds.width * 72 / self.fig.dpi + 0.8, bounds.height * 72 / self.fig.dpi + 1.0

    def wrap(self, text, width, size, weight="normal"):
        result = []
        for explicit_line in text.split("\n"):
            line = ""
            for word in explicit_line.split():
                candidate = f"{line} {word}".strip()
                if self.measure(candidate, size, weight)[0] > width:
                    if not line:
                        raise ValueError(f"Unbreakable word does not fit: {word!r}")
                    result.append(line)
                    line = word
                    if self.measure(line, size, weight)[0] > width:
                        raise ValueError(f"Unbreakable word does not fit: {word!r}")
                else:
                    line = candidate
            result.append(line)
        return "\n".join(result)

    def prepare(self, s, width):
        inner = width - 2 * PAD
        title = self.wrap(s.title, inner, TITLE_SIZE, "bold")
        body = self.wrap("\n".join(s.lines), inner, BODY_SIZE)
        foot = self.wrap(s.foot, inner, FOOT_SIZE)
        heights = (self.measure(title, TITLE_SIZE, "bold")[1],
                   self.measure(body, BODY_SIZE)[1], self.measure(foot, FOOT_SIZE)[1])
        # Dedicated header, body and footer zones with an actual measured gap.
        required_height = 2 * PAD + sum(heights) + 2 * LAYOUT_GAP
        return (title, body, foot), heights, required_height

    def draw_card(self, key, s, x, top, width, height, prepared):
        strings, heights, _ = prepared
        box = Bbox.from_bounds(x, top - height, width, height)
        patch = FancyBboxPatch((box.x0, box.y0), width, height,
                               boxstyle="round,pad=0,rounding_size=3",
                               facecolor=FILLS[s.grade], edgecolor=EDGES[s.grade],
                               linewidth=1.15, linestyle=(0, (4, 3)) if s.grade in ("prior", "U") else "-",
                               zorder=2)
        self.ax.add_patch(sketch(patch))
        title_y = top - PAD
        foot_y = box.y0 + PAD + heights[2]
        body_top, body_bottom = title_y - heights[0] - LAYOUT_GAP, foot_y + LAYOUT_GAP
        body_y = (body_top + body_bottom + heights[1]) / 2
        texts = []
        for text, y, size, weight, color in (
                (strings[0], title_y, TITLE_SIZE, "bold", INK),
                (strings[1], body_y, BODY_SIZE, "normal", SOFT),
                (strings[2], foot_y, FOOT_SIZE, "normal", EDGES[s.grade])):
            texts.append(self.ax.text(x + width / 2, y, text, fontsize=size,
                                      fontweight=weight, color=color, ha="center", va="top",
                                      linespacing=1.25, zorder=3))
        self.cards[key] = Card(key, s, box, texts)

    def associate(self, source, target, port_fraction=0.5):
        src, dst = self.cards[source], self.cards[target]
        left = src.box.x0 < dst.box.x0
        sx = src.box.x1 if left else src.box.x0
        tx = dst.box.x0 if left else dst.box.x1
        sy, ty = (src.box.y0 + src.box.height / 2,
                  dst.box.y0 + dst.box.height * port_fraction)
        # The elbow's vertical segment is in the empty gutter, NEVER on/inside a card.
        gx = (sx + tx) / 2
        points = [(sx, sy), (gx, sy), (gx, ty), (tx, ty)]
        color = EDGES[src.spec.grade]
        line, = self.ax.plot(*zip(*points), color=color, linewidth=1.15,
                             linestyle=(0, (4, 3)) if src.spec.grade in ("prior", "U") else "-",
                             solid_capstyle="butt", dash_capstyle="butt", zorder=1)
        sketch(line)
        # A dash pattern can end in a gap at a port. Short solid terminal caps make
        # even prior-only links visibly touch both card edges, without a stray dot.
        direction = 1 if left else -1
        for x, y, inward in ((sx, sy, direction), (tx, ty, -direction)):
            self.ax.plot([x, x + inward * 2], [y, y], color=color, linewidth=1.15,
                         solid_capstyle="butt", zorder=1)
        self.links.append(Link(source, target, points, "association"))

    def arrow(self, source, target):
        src, dst = self.cards[source].box, self.cards[target].box
        points = [(src.x0 + src.width / 2, src.y0),
                  (dst.x0 + dst.width / 2, dst.y1)]
        self.ax.add_patch(sketch(FancyArrowPatch(*points, arrowstyle="-|>", mutation_scale=10,
                                                shrinkA=0, shrinkB=0, color=SOFT, linewidth=1.2,
                                                zorder=1)))
        self.links.append(Link(source, target, points, "functional order"))

    def text_box(self, artist):
        return artist.get_window_extent(self.fig.canvas.get_renderer()).transformed(
            self.ax.transData.inverted())

    @staticmethod
    def segment_enters_box(a, b, box, tolerance=1e-5):
        """Strict-interior intersection: endpoints touching the intended edge are OK."""
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        if abs(a[0] - b[0]) < tolerance:
            return (box.x0 + tolerance < a[0] < box.x1 - tolerance
                    and min(y1, box.y1) - max(y0, box.y0) > tolerance)
        if abs(a[1] - b[1]) < tolerance:
            return (box.y0 + tolerance < a[1] < box.y1 - tolerance
                    and min(x1, box.x1) - max(x0, box.x0) > tolerance)
        raise AssertionError("Only orthogonal routes are allowed")

    @staticmethod
    def on_edge(point, box, tolerance=1e-4):
        x, y = point
        return ((abs(x - box.x0) < tolerance or abs(x - box.x1) < tolerance)
                and box.y0 - tolerance <= y <= box.y1 + tolerance
                or (abs(y - box.y0) < tolerance or abs(y - box.y1) < tolerance)
                and box.x0 - tolerance <= x <= box.x1 + tolerance)

    def validate(self):
        self.fig.canvas.draw()
        gaps, padding = [], []
        for card in self.cards.values():
            bounds = [self.text_box(t) for t in card.texts]
            for box in bounds:
                pad = min(box.x0 - card.box.x0, card.box.x1 - box.x1,
                          box.y0 - card.box.y0, card.box.y1 - box.y1)
                padding.append(pad)
                assert pad >= PAD - 0.1, f"{self.name}/{card.key}: text outside padded interior"
            for upper, lower in zip(bounds, bounds[1:]):
                gap = upper.y0 - lower.y1
                gaps.append(gap)
                assert gap >= TEXT_GAP - 0.1, f"{self.name}/{card.key}: text blocks collide"
        cards = list(self.cards.values())
        for i, card in enumerate(cards):
            for other in cards[i + 1:]:
                assert not card.box.overlaps(other.box), "Card rectangles overlap"
        for link in self.links:
            assert self.on_edge(link.points[0], self.cards[link.source].box), "Floating source"
            assert self.on_edge(link.points[-1], self.cards[link.target].box), "Floating target"
            for a, b in zip(link.points, link.points[1:]):
                for card in cards:
                    assert not self.segment_enters_box(a, b, card.box), (
                        f"{self.name}: {link.source}->{link.target} crosses {card.key}")
        canvas = Bbox.from_bounds(0, 0, WIDTH, self.ax.get_ylim()[1])
        label_boxes = [self.text_box(t) for t in self.labels]
        for i, box in enumerate(label_boxes):
            assert canvas.contains(box.x0, box.y0) and canvas.contains(box.x1, box.y1)
            for other in label_boxes[i + 1:]:
                assert not box.overlaps(other), "Figure labels overlap"
            for card in cards:
                assert not box.overlaps(card.box), "Figure label overlaps a card"
        return {"cards": len(cards), "links": len(self.links),
                "min_text_gap_pt": round(min(gaps), 3),
                "min_inner_padding_pt": round(min(padding), 3),
                "body_font_pt_at_full_text_width": BODY_SIZE,
                "foot_font_pt_at_full_text_width": FOOT_SIZE,
                "connector_card_crossings": 0, "floating_endpoints": 0,
                "style": "v2 Excalidraw-inspired", "font": HAND_FONT,
                "figure_width_pt": round(WIDTH, 3),
                "figure_height_pt": round(canvas.height, 3)}

    def close(self):
        plt.close(self.fig)


def create_diagram(d: Diagram):
    plt.rcParams.update({"font.family": [HAND_FONT, "DejaVu Sans"],
                         "figure.facecolor": "white", "text.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    scene = Scene(d.name)
    center = [scene.prepare(s, CENTER_WIDTH) for s in d.center]
    sides = [[scene.prepare(s, SIDE_WIDTH) for s in column[:2]]
             for column in (d.left, d.right)]
    aux_width = (WIDTH - 2 * MARGIN - GUTTER) / 2
    aux = [scene.prepare(column[2], aux_width) for column in (d.left, d.right)]
    boundary = scene.prepare(d.boundary, WIDTH - 2 * MARGIN)
    # Staggered side boxes and a wider spine reproduce v2, not a dashboard grid.
    side_heights = [max(sides[0][i][2], sides[1][i][2]) for i in range(2)]
    central_height = sum(p[2] for p in center) + 2 * ROW_GAP
    graph_height = max(central_height, 18 + sum(side_heights) + ROW_GAP)
    aux_height = max(p[2] for p in aux)
    header_height = 74.0
    height = header_height + graph_height + 35 + aux_height + 20 + boundary[2] + MARGIN
    scene.fig.set_size_inches(WIDTH / 72, height / 72)
    scene.ax.set_ylim(0, height)

    scene.labels.append(scene.ax.text(WIDTH / 2, height - 8, d.title,
                                      fontsize=11, fontweight="bold", ha="center", va="top"))
    for x, label, grade in ((16, "Observed", "measured"), (127, "Candidate", "mechanism"),
                             (243, "Prior-only / U", "prior")):
        y = height - 35
        scene.ax.add_patch(sketch(FancyBboxPatch((x, y - 9), 8, 8,
                              boxstyle="round,pad=0,rounding_size=1",
                              facecolor=FILLS[grade], edgecolor=EDGES[grade], linewidth=1)))
        scene.labels.append(scene.ax.text(x + 13, y, label, fontsize=7.8,
                                          ha="left", va="top", color=SOFT))
    scene.labels.append(scene.ax.text(WIDTH / 2, height - 54,
                         "Conceptual paths and evidence links, not recovered physical wiring.",
                         fontsize=7.2, color=SOFT, ha="center", va="top"))
    graph_top = height - header_height
    center_x = MARGIN + SIDE_WIDTH + GUTTER
    top = graph_top
    for i, p in enumerate(center):
        scene.draw_card(f"center{i}", d.center[i], center_x, top, CENTER_WIDTH, p[2], p)
        top -= p[2] + ROW_GAP
    for col, (prefix, specs) in enumerate((("left", d.left), ("right", d.right))):
        x = MARGIN if col == 0 else center_x + CENTER_WIDTH + GUTTER
        top = graph_top - 18
        for i, h in enumerate(side_heights):
            scene.draw_card(f"{prefix}{i}", specs[i], x, top, SIDE_WIDTH, h, sides[col][i])
            top -= h + ROW_GAP
        # Port order and gutter turns prevent the dangling/border-following v2 links.
        for i, target in enumerate((d.left_targets if col == 0 else d.right_targets)[:2]):
            scene.associate(f"{prefix}{i}", f"center{target}", port_fraction=0.78 if i == 0 else 0.25)
    scene.arrow("center0", "center1")
    scene.arrow("center1", "center2")

    aux_top = graph_top - graph_height - 35
    scene.labels.append(scene.ax.text(WIDTH / 2, aux_top + 22,
                                      "Additional evidence / limits (not execution stages)",
                                      fontsize=7.5, color=SOFT, ha="center", va="top"))
    for i, column in enumerate((d.left, d.right)):
        s = column[2]
        # V2's blue auxiliary cards for measured interface/serving facts.
        if d.name != "jev":
            s = Spec(s.title, s.lines, "aux", s.foot)
        scene.draw_card(f"{'left' if i == 0 else 'right'}2", s,
                        MARGIN + i * (aux_width + GUTTER), aux_top,
                        aux_width, aux_height, aux[i])
    boundary_top = aux_top - aux_height - 20
    scene.draw_card("boundary", d.boundary, MARGIN, boundary_top,
                    WIDTH - 2 * MARGIN, boundary[2], boundary)
    return scene


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    qa = {}
    for d in DIAGRAMS:
        scene = create_diagram(d)
        qa[d.name] = scene.validate()  # Fail before exporting if any geometry is unsafe.
        # No tight bounding box: the canvas width is the paper's actual text width.
        scene.fig.savefig(FIG / f"fig_v4_recon_{d.name}.pdf")
        scene.fig.savefig(FIG / f"fig_v4_recon_{d.name}.png", dpi=200)
        print(f"wrote {d.name}: {qa[d.name]}")
        scene.close()
    QA.parent.mkdir(parents=True, exist_ok=True)
    QA.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
