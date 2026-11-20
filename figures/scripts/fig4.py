"""Render Fig. 4 from the frozen operating-point and source-paired MRC tables."""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd

from .resources import DATA, FONT_FAMILY, save_svg

NAMES = ("Random", "Replace", "Add", "Delete", "Content", "Pose", "Color", "Material", "Background", "Style")
PAIRS = ("SU", "SF", "UF")
SCALES = (.25, .5, .75, 1., 1.25)


WIDTH = 7.16
HEIGHT = WIDTH * 169 / 510
COLORS = {"Spatial": "#1F5A8C", "Uniform": "#C85A32",
          "Scalar": "#6E7781", "FlowEdit": "#2A8C76"}
ARMS = {"Spatial": ("o", "-"), "Uniform": ("s", "--"), "FlowEdit": ("^", "-.")}
CONTROLS = {"two_constant": ("Scalar", "D", 3.6),
            "uniform_at_local": ("Uniform", "s", 6.2),
            "spatial_at_uniform": ("Spatial", "o", 6.8)}
INK = "#23282C"
MUTED = "#4B555C"
EXTENSION_STYLE = (0, (1.0, 1.65))


def panel(fig, x, y, w, h):
    """Layout in typographic points at the final print size."""
    return fig.add_axes([x / (72 * WIDTH), y / (72 * HEIGHT),
                         w / (72 * WIDTH), h / (72 * HEIGHT)])


def text(fig, x, y, s, **kw):
    return fig.text(x / (72 * WIDTH), y / (72 * HEIGHT), s, va="center", **kw)


def intervals(ax, rows, x, y, color):
    """Each segment is a marginal source-bootstrap interval for one observed setting."""
    ax.hlines(rows[y], rows[x + "_ci_low"], rows[x + "_ci_high"],
              color=color, alpha=.49, lw=.66, zorder=1)
    ax.vlines(rows[x], rows[y + "_ci_low"], rows[y + "_ci_high"],
              color=color, alpha=.49, lw=.66, zorder=1)


def control(ax, x, y, method):
    color, marker, size = CONTROLS[method]
    ax.plot(x, y, marker=marker, ms=size, mec=COLORS[color],
            mfc=COLORS[color] if color == "Scalar" else "none",
            mew=.85, linestyle="none", zorder=5)


def operating_plane(ax, points, x, y, offsets):
    for arm, (marker, style) in ARMS.items():
        rows = points[points.arm == arm].sort_values("setting")
        knobs = (12, 16, 20, 24, 28) if arm == "FlowEdit" else SCALES
        assert tuple(rows.setting) == tuple(knobs) and rows.sources.eq(42).all()
        color = COLORS[arm]
        intervals(ax, rows, x, y, color)
        native_range = rows if arm == "FlowEdit" else rows[rows.setting <= 1]
        ax.plot(native_range[x], native_range[y], ls=style, lw=1.0, color=color, marker=marker,
                mfc="white", mec=color, ms=3.45, mew=.8, zorder=3)
        if arm in ("Spatial", "Uniform"):
            extension = rows[rows.setting >= 1]
            ax.plot(extension[x], extension[y], ls=EXTENSION_STYLE, lw=1.25, color=color,
                    marker=marker, markevery=[-1], ms=4.6, mfc="white", mec=color,
                    mew=1.0, zorder=3, gid=f"extension-{arm.lower()}-{y}")
        default = rows[rows.setting == (24 if arm == "FlowEdit" else 1)].iloc[0]
        ax.plot(default[x], default[y], marker=marker, ms=3.8, color=color, zorder=4)
        end = rows.iloc[-1]
        dx, dy, align = offsets[arm]
        label = arm
        if y == "selectivity" and arm in ("Spatial", "Uniform"):
            label = f"{arm} {end[y]:.2f}× at {end.setting:g}"
        ax.annotate(label, (end[x], end[y]), xytext=(dx, dy), textcoords="offset points",
                    ha=align, va="center", fontsize=7.4, color=color, zorder=7)
    for method, (color, _, _) in CONTROLS.items():
        row = points[points.method == method]
        assert len(row) == 1 and row.sources.iloc[0] == 42
        intervals(ax, row, x, y, COLORS[color])
        control(ax, row[x].iloc[0], row[y].iloc[0], method)


def log_ticks(ax, ticks, labels):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())


def estimate(ax, row, y):
    if row.supported >= 2:
        ax.plot([row.ci_low, row.ci_high], [y, y], color=INK, lw=1.0,
                marker="|", ms=2.5, mew=.6, zorder=4)
    ax.plot(row["mean"], y, "D", ms=2.75, mfc=INK, mec=INK, mew=.4, zorder=5)


def category_panel(ax, sources, stats, bounded, bounded_values):
    ax.axvline(0, color="#8A9399", lw=.64, ls=(0, (2, 2)), zorder=0)
    y_ticks, labels = [], []
    for i, name in enumerate(NAMES):
        y = len(NAMES) - i + 4
        values = sources.loc[(sources.pair == "SU") & (sources.category_name == name), "spatial"]
        row = stats.loc[("SU", name)]
        assert len(values) == row.supported
        # The separate lanes preserve x exactly, even when a source equals its category mean.
        # Stagger neighbors within the source lane only when marker footprints overlap.
        previous_x, neighbor = -np.inf, 0
        for value in values.sort_values():
            neighbor = neighbor + 1 if value - previous_x < .022 else 0
            previous_x = value
            offset = (0, -.15, .13)[neighbor % 3]
            negative = value < 0
            ax.scatter(value, y-.20+offset, marker="v" if negative else "o",
                       s=11 if negative else 8, linewidths=.4, alpha=.65,
                       color=COLORS["Uniform" if negative else "Spatial"], zorder=2)
        estimate(ax, row, y+.22)
        y_ticks.append(y)
        labels.append(f"{name} ({len(values)})")
    ax.axhline(4.2, color="#B3B9BD", lw=.5)
    overall = [("4-setting U − S", bounded, bounded_values)]
    for pair, label in zip(PAIRS, ["5-setting U − S", "F − S", "F − U"]):
        vals = sources.loc[sources.pair == pair, "spatial"]
        overall.append((label, stats.loc[(pair, "All")], vals))
    for (label, row, vals), y in zip(overall, [3.5, 2.43, 1.36, .29]):
        assert len(vals) == row.supported and np.isclose(vals.mean(), row["mean"])
        ax.plot(vals, np.full(len(vals), y-.2), "|", ms=2.7, mew=.6,
                color=COLORS["Scalar"], alpha=.46, zorder=2)
        estimate(ax, row, y+.19)
        y_ticks.append(y)
        labels.append(f"{label} ({len(vals)})")
    ax.set(xlim=(-.455, .51), ylim=(-.28, 14.67), yticks=y_ticks, yticklabels=labels,
           xticks=[-.4,-.2,0,.2,.4], xticklabels=["−0.4","−0.2","0","0.2","0.4"],
           xlabel="Paired difference in background LPIPS")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2.6)


def figure():
    mpl.rcParams.update({
        "font.family": FONT_FAMILY, "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.6,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#677178",
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .55,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.3, "ytick.major.size": 2.3, "axes.labelpad": 2,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "v19-selectivity",
    })
    points = pd.read_csv(DATA / "operating_points.csv")
    sources = pd.read_csv(DATA / "five_setting_sources.csv", dtype={"case": str})
    stats = pd.read_csv(DATA / "five_setting_summary.csv").set_index(["pair", "category"])
    bounded = pd.read_csv(DATA / "four_setting_summary.csv").rename(columns={
        "n_valid": "supported", "matched": "mean", "matched_lo": "ci_low", "matched_hi": "ci_high"
    }).iloc[0]
    bounded_values = pd.read_csv(DATA / "four_setting_sources.csv")["matched"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")

    for x, title, sub in [
        (1.5, "(a)  Observed operating points", "Directional CLIP   ·   42 sources"),
        (170, "(b)  Change and selectivity", r"Edit / background MAE  $\uparrow$  more concentrated"),
        (331, "(c)  Matched background LPIPS", "5-setting Uniform − Spatial · category mean [95% CI]"),
    ]:
        text(fig, x, 164, title, size=8.5, weight="bold")
        text(fig, x, 151, sub, size=7.25, color=MUTED)

    a = panel(fig, 18, 42, 143, 97)
    log_ticks(a, [.01,.03,.1,.3], ["0.01","0.03","0.1","0.3"])
    a.set(xlim=(.0091,.77), ylim=(-.014,.204), xlabel="Background LPIPS (log scale)",
          yticks=[0,.05,.10,.15,.20], yticklabels=["0","0.05","0.1","0.15","0.2"])
    operating_plane(a, points, "lpips_spatial", "directional_clip",
                    {"Spatial": (-4,5,"right"), "Uniform": (3,-10,"left"),
                     "FlowEdit": (-3,7,"right")})

    b = panel(fig, 185, 42, 137, 97)
    log_ticks(b, [.015,.03,.06,.12,.24], ["0.015","0.03","0.06","0.12","0.24"])
    b.set(xlim=(.010,.255), ylim=(.9,6.65), yticks=[1,2,3,4,5,6],
          yticklabels=[f"{n}×" for n in range(1, 7)],
          xlabel=r"Whole-image MAE (log)  ·  $\leftarrow$ lighter")
    operating_plane(b, points, "total_change", "selectivity",
                    {"Spatial": (4,5,"left"), "Uniform": (1,-9,"right"),
                     "FlowEdit": (-2,11,"right")})

    c = panel(fig, 384, 31.5, 127.5, 111)
    category_panel(c, sources, stats, bounded, bounded_values)
    text(fig, 331, 4.2, f"Four-setting U − S: {bounded['mean']:.4f} "
         f"[{bounded.ci_low:.4f}, {bounded.ci_high:.4f}]", size=7.05)

    text(fig, 1.5, 14.5, "SAM scale  0.25, 0.5, 0.75, 1", size=7.05, color=MUTED)
    fig.add_artist(plt.Line2D(np.array([98, 112]) / (72 * WIDTH), [14.5 / (72 * HEIGHT)] * 2,
                              transform=fig.transFigure, color=MUTED, lw=1.25, ls=EXTENSION_STYLE))
    text(fig, 116, 14.5, "1.25 extension", size=7.05, color=MUTED)
    text(fig, 1.5, 5.7, "FlowEdit steps  12, 16, 20, 24, 28", size=7.05, color=MUTED)
    key = panel(fig, 165, 1.2, 164, 18)
    key.set(xlim=(0,164), ylim=(0,18))
    key.axis("off")
    for x, method, label in [(4,"two_constant","Scalar"), (45,"uniform_at_local","Uniform @ S"),
                             (109,"spatial_at_uniform","Spatial @ U")]:
        control(key, x, 13.3, method)
        key.text(x+5.8,13.3,label,va="center",fontsize=7.05)
    key.text(0,4.5,"Filled = default    ·    Bars = 95% source CI",va="center",fontsize=7.05,color=MUTED)

    save_svg(fig, "selectivity", dpi=400)
    plt.close(fig)
