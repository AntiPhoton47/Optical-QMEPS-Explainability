from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "input": "#E7EEF8",
    "semantic": "#EAF4EA",
    "monitor": "#F7E8C7",
    "output": "#F0EAF8",
    "edge": "#5A6673",
    "text": "#20242A",
    "blue": "#2F6DB3",
    "red": "#B84A4A",
}


def add_panel(ax, label, title):
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.45, 3.25)
    ax.axis("off")
    ax.text(
        0.05,
        3.08,
        label,
        fontsize=12,
        fontweight="bold",
        color=COLORS["text"],
        va="top",
    )
    ax.text(
        0.74,
        3.08,
        title,
        fontsize=12,
        fontweight="bold",
        color=COLORS["text"],
        va="top",
    )


def node(ax, xy, text, fill, radius=0.26, fontsize=8.5, edge="#8A95A1"):
    circ = Circle(xy, radius, facecolor=fill, edgecolor=edge, lw=1.0, zorder=3)
    ax.add_patch(circ)
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        zorder=4,
    )


def arrow(ax, start, end, lw=1.0, alpha=0.72, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        lw=lw,
        color=COLORS["edge"],
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arr)


def layer_label(ax, x, title, subtitle, y=0.18):
    ax.text(x, y, title, ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLORS["text"])
    ax.text(x, y - 0.27, subtitle, ha="center", va="bottom", fontsize=7.5, color="#5B6470")


def monitor_box(ax, x, y0=0.48, y1=2.42, label="monitor", label_offset=0.08):
    box = FancyBboxPatch(
        (x - 0.55, y0),
        1.1,
        y1 - y0,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        facecolor=COLORS["monitor"],
        edgecolor="#C59C45",
        lw=0.9,
        alpha=0.45,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(x, y1 + label_offset, label, ha="center", va="bottom", fontsize=8.2, color="#6F5513")


def monitor_tag(ax, x, y, text, width=1.35):
    tag = FancyBboxPatch(
        (x - width / 2, y - 0.14),
        width,
        0.28,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=COLORS["monitor"],
        edgecolor="#C59C45",
        lw=0.85,
        alpha=0.65,
        zorder=6,
    )
    ax.add_patch(tag)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.7, color="#6F5513", zorder=7)


def rounded_box(ax, xy, width, height, fill, edge="#8A95A1", lw=1.0, zorder=3):
    rect = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.07",
        facecolor=fill,
        edgecolor=edge,
        lw=lw,
        zorder=zorder,
    )
    ax.add_patch(rect)
    return rect


def mini_node(ax, xy, text, fill, edge="#8A95A1", radius=0.18, fontsize=7.5):
    circ = Circle(xy, radius, facecolor=fill, edgecolor=edge, lw=0.9, zorder=5)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=COLORS["text"], zorder=6)


def elbow_arrow(ax, points, lw=0.95, alpha=0.58):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=COLORS["edge"], lw=lw, alpha=alpha, zorder=2)
    arrow(ax, points[-2], points[-1], lw=lw, alpha=alpha)


def draw_topology(ax):
    add_panel(ax, "(a)", "Topology Calibration")
    xs = [1.4, 5.2, 9.0]
    inputs = [(xs[0], 1.69), (xs[0], 0.89)]
    middle = [(xs[1], 1.96), (xs[1], 1.26), (xs[1], 0.56)]
    outputs = [(xs[2], 1.69), (xs[2], 0.89)]
    monitor_box(ax, xs[1], 0.25, 2.25, "ancilla on L2", label_offset=0.15)
    for i, p in enumerate(inputs, start=1):
        node(ax, p, f"in {i}", COLORS["input"])
    for j, p in enumerate(middle, start=1):
        node(ax, p, f"m{j}", COLORS["semantic"])
    for k, p in enumerate(outputs, start=1):
        node(ax, p, f"a{k}", COLORS["output"])
    for s in inputs:
        for t in middle:
            arrow(ax, (s[0] + 0.26, s[1]), (t[0] - 0.26, t[1]), lw=0.85, alpha=0.45)
    for s in middle:
        for t in outputs:
            arrow(ax, (s[0] + 0.26, s[1]), (t[0] - 0.26, t[1]), lw=0.85, alpha=0.45)
    ax.text(5.2, 2.75, "connected support", ha="center", fontsize=8.4, color="#4C5A66")
    layer_label(ax, xs[0], "Input", "2 photons", y=-0.1)
    layer_label(ax, xs[1], "Middle", "3 modes, active support", y=-0.1)
    layer_label(ax, xs[2], "Action", "2 modes", y=-0.1)


def draw_blue_square(ax):
    add_panel(ax, "(b)", "Blue-Square Feature Classifier")
    xs = [1.35, 4.7, 8.2, 10.7]
    percepts = [(xs[0], 2.24), (xs[0], 1.68), (xs[0], 1.12), (xs[0], 0.56)]
    features = [(xs[1], 2.24), (xs[1], 1.68), (xs[1], 1.12), (xs[1], 0.56)]
    outputs = [(xs[2], 1.84), (xs[2], 1.04)]
    monitor_box(ax, xs[1], 0.28, 2.48, "feature monitor", label_offset=0.10)
    for lab, p in zip(["RS", "RC", "BS", "BC"], percepts):
        node(ax, p, lab, COLORS["input"], radius=0.23)
    for lab, p, fill in zip(["R", "B", "S", "C"], features, [COLORS["red"], COLORS["blue"], COLORS["semantic"], COLORS["semantic"]]):
        node(ax, p, lab, fill, radius=0.23)
    for lab, p in zip(["no", "yes"], outputs):
        node(ax, p, lab, COLORS["output"], radius=0.28)
    for s, targets in zip(percepts, [(features[0], features[2]), (features[0], features[3]), (features[1], features[2]), (features[1], features[3])]):
        for t in targets:
            arrow(ax, (s[0] + 0.24, s[1]), (t[0] - 0.24, t[1]), lw=0.9, alpha=0.55)
    for s in features:
        arrow(ax, (s[0] + 0.24, s[1]), (outputs[0][0] - 0.28, outputs[0][1]), lw=0.75, alpha=0.35)
        arrow(ax, (s[0] + 0.24, s[1]), (outputs[1][0] - 0.28, outputs[1][1]), lw=0.95, alpha=0.45)
    rule = FancyBboxPatch(
        (9.35, 0.56),
        2.35,
        1.52,
        boxstyle="round,pad=0.12,rounding_size=0.08",
        facecolor="#F7F8FA",
        edgecolor="#B8C0C8",
        lw=0.9,
    )
    ax.add_patch(rule)
    ax.text(10.525, 1.77, "semantic rule", ha="center", fontsize=8.5, fontweight="bold", color=COLORS["text"])
    ax.text(10.525, 1.42, "yes iff B and S", ha="center", fontsize=8.4, color=COLORS["text"])
    ax.text(10.525, 1.12, "1-photon test", ha="center", fontsize=7.6, color="#5B6470")
    ax.text(10.525, 0.84, "2-photon follow-up", ha="center", fontsize=7.6, color="#5B6470")
    layer_label(ax, xs[0], "Percept", "4 objects", y=-0.05)
    layer_label(ax, xs[1], "Features", "R, B, S, C", y=-0.05)
    layer_label(ax, xs[2], "Readout", "yes/no", y=-0.05)


def draw_cm(ax):
    add_panel(ax, "(c)", "Modified Computer Maintenance Environment")
    ax.text(10.05, 2.64, r"$N=5,\ M=31$", ha="center", fontsize=8.2, color="#4C5A66")

    symptom_x = 1.05
    symptom_y = [1.88, 1.51, 1.14, 0.77, 0.40]
    rounded_box(ax, (0.35, 0.24), 1.4, 2.26, COLORS["input"], zorder=1)
    ax.text(symptom_x, 2.25, "symptoms", ha="center", va="center", fontsize=8.6, fontweight="bold", color=COLORS["text"])
    for lab, y in zip([r"$S_1$", r"$S_2$", r"$S_3$", r"$\cdots$", r"$S_{10}$"], symptom_y):
        mini_node(ax, (symptom_x, y), lab, "#CFE6F7", radius=0.15, fontsize=7.1)
    ax.text(symptom_x, -0.12, "symptom pair", ha="center", va="bottom", fontsize=7.4, color="#5B6470")

    diag_x = 3.55
    rounded_box(ax, (2.48, 0.30), 2.14, 2.04, COLORS["semantic"], zorder=1)
    monitor_tag(ax, diag_x, 2.55, "diagnosis record", width=1.55)
    ax.text(diag_x, 2.13, "component-cause", ha="center", fontsize=8.4, fontweight="bold", color=COLORS["text"])
    ax.text(diag_x, 1.87, "semantic layer", ha="center", fontsize=7.3, color="#5B6470")
    for lab, y in zip([r"$A_1$", r"$A_2$", r"$A_5$"], [1.55, 1.13, 0.71]):
        mini_node(ax, (3.05, y), lab, "#DDE8F4", radius=0.17)
    for lab, y in zip([r"$C_1$", r"$C_3$", r"$C_5$"], [1.55, 1.13, 0.71]):
        mini_node(ax, (4.03, y), lab, "#F7D3A8", radius=0.17)
    ax.text(3.05, 0.36, "components", ha="center", fontsize=6.9, color="#5B6470")
    ax.text(4.03, 0.36, "causes", ha="center", fontsize=6.9, color="#5B6470")

    repair_x = 6.35
    rounded_box(ax, (5.22, 0.30), 2.26, 2.04, COLORS["semantic"], zorder=1)
    monitor_tag(ax, repair_x, 2.55, "repair record", width=1.25)
    ax.text(repair_x, 2.13, "component-fix", ha="center", fontsize=8.4, fontweight="bold", color=COLORS["text"])
    ax.text(repair_x, 1.87, "candidate layer", ha="center", fontsize=7.3, color="#5B6470")
    for lab, y in zip([r"$A_1$", r"$A_2$", r"$A_5$"], [1.55, 1.13, 0.71]):
        mini_node(ax, (5.82, y), lab, "#DDE8F4", radius=0.17)
    for lab, y in zip([r"$F_1$", r"$F_2$", r"$F_4$"], [1.55, 1.13, 0.71]):
        mini_node(ax, (6.88, y), lab, "#F4A7B8", radius=0.17)
    ax.text(5.82, 0.36, "components", ha="center", fontsize=6.9, color="#5B6470")
    ax.text(6.88, 0.36, "fixes", ha="center", fontsize=6.9, color="#5B6470")

    readout_x = 9.05
    rounded_box(ax, (8.36, 0.82), 1.38, 1.36, COLORS["output"], zorder=3)
    ax.text(readout_x, 1.64, "yes/no", ha="center", fontsize=8.9, fontweight="bold", color=COLORS["text"])
    ax.text(readout_x, 1.22, "repair fixes\nscenario?", ha="center", fontsize=7.2, color="#5B6470")

    candidate_x = 10.95
    rounded_box(ax, (10.05, 0.78), 1.8, 1.44, "#F7F8FA", edge="#B8C0C8", lw=0.9, zorder=2)
    ax.text(candidate_x, 1.7, "repair", ha="center", fontsize=8.6, fontweight="bold", color=COLORS["text"])
    ax.text(candidate_x, 1.42, "candidate", ha="center", fontsize=8.6, fontweight="bold", color=COLORS["text"])
    ax.text(candidate_x, 1.02, "44 scenarios", ha="center", fontsize=7.4, color="#5B6470")

    arrow(ax, (1.78, 1.42), (2.44, 1.42), lw=1.2)
    arrow(ax, (4.66, 1.42), (5.18, 1.42), lw=1.2)
    arrow(ax, (7.52, 1.42), (8.32, 1.42), lw=1.2)
    elbow_arrow(ax, [(10.05, 0.74), (9.95, 0.08), (7.76, 0.08), (7.52, 0.62)], lw=0.9, alpha=0.55)
    ax.text(2.1, 1.64, r"$U_1$", ha="center", fontsize=7.5, color="#5B6470")
    ax.text(4.92, 1.64, r"$U_2^{(a)}$", ha="center", fontsize=7.5, color="#5B6470")
    ax.text(7.92, 1.64, r"$U_3^{(s)}$", ha="center", fontsize=7.5, color="#5B6470")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.25), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.045, hspace=0.24)
    draw_topology(axes[0])
    draw_blue_square(axes[1])
    draw_cm(axes[2])
    fig.savefig(OUT_DIR / "fig45_environment_schematic.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig45_environment_schematic.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
