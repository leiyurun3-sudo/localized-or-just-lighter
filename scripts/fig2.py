"""Rebuild only Fig. 2 from measured traces, the archived correction field and scores.

Repository entry point: python figures/redraw.py 2
The two offset sheets show the measured source (with its native P=0.5 contour)
and its 64 x 64 correction multiplier. The oblique projection is a display
transform shared by both sheets. The four-cell replay diagram is schematic.
The correction archive logs all 28 steps. Both departing profiles log only
active steps 2..27. Open symbols show a subset of separately measured replays.
"""
from .resources import DATA, FONT_FAMILY, flower_source, read_data, save_svg

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib import patheffects
import numpy as np

SAM_ARMS = {"SS": "sam_native", "UU": "uniform", "SU": "spatial_at_uniform", "US": "uniform_at_local"}
SPATIAL, UNIFORM, SCALAR, FLOWEDIT = "#1F5A8C", "#C85A32", "#6E7781", "#2A8C76"
COL = {"S": SPATIAL, "U": UNIFORM}
MARK = {"S": "o", "U": "s"}
INK, SUB, HAIR = "#20262B", "#4B545B", "#BCC4CA"
W, H = 7.16 * 72, 168
FONT = {"font.family": FONT_FAMILY, "font.size": 7.5, "mathtext.fontset": "stix",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": SUB, "ytick.color": SUB,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": SCALAR, "axes.linewidth": .55,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "v20-fig2-final"}
FIELD = LinearSegmentedColormap.from_list("protection_weight", ["#404040", "#FFFFFF"])


def measurements():
    photo = np.asarray(flower_source())
    weight = np.load(DATA / "flower_gate.npy", allow_pickle=False)
    data = read_data("profiles.json")
    branches = {
        field: {key: [step[field] for step in data["departing"][arm]] for key, arm in SAM_ARMS.items()}
        for field in ("model_rms", "visible_rms")
    }
    sweeps = {key: np.array(points) for key, points in data["curves"].items()}
    return (photo, weight, data["restoring"], branches, sweeps, data["two_constants"],
            data["pieces"], data["difference"])


def axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def text(fig, x, y, value, **style):
    fig.text(x / W, y / H, value, va="center", **style)


def line(fig, x0, y0, x1, y1, **style):
    fig.add_artist(Line2D([x0 / W, x1 / W], [y0 / H, y1 / H], transform=fig.transFigure, **style))


def images(fig, photo, weight):
    ax = axes(fig, 1, 112, 83, 37.8)
    source = Affine2D.from_values(40.5, 1.9, -4.2, -30.6, 5.1, 35.0)
    field = source + Affine2D().translate(31.9, -4.0)
    source_view, field_view = source + ax.transData, field + ax.transData
    ax.imshow(photo, origin="upper", extent=(0, 1, 1, 0), transform=source_view,
              interpolation="none", zorder=1)
    # Register the contour and photo before shifting the front sheet.
    xy = (np.arange(weight.shape[0]) + .5) / weight.shape[0]
    contour = ax.contour(xy, xy, weight, levels=[.5], transform=source_view,
                         colors="white", linewidths=.32, zorder=2)
    contour.set_path_effects([patheffects.Stroke(linewidth=.60,foreground="#27343d"),
                              patheffects.Normal()])
    ax.imshow(weight, origin="upper", extent=(0, 1, 1, 0), transform=field_view,
              cmap=FIELD, vmin=.25, vmax=1, interpolation="none", zorder=4)
    for view, order in ((source_view, 3), (field_view, 5)):
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=view, fill=False,
                               edgecolor="#6f787e", linewidth=.42, zorder=order))
    ax.set(xlim=(0,83), ylim=(0,37.8), aspect="auto")
    ax.set_axis_off()
    text(fig, 19.5, 152.6, "Source", size=7.2, ha="center")
    text(fig, 61.5, 152.6, "Weight P", size=7.2, ha="center")
    text(fig, 1, 108.3, "P = 0.5 line", size=7, color=SUB)
    bar = axes(fig, 61, 107.3, 13, 2.5)
    bar.imshow(np.linspace(.25,1,100)[None,:], cmap=FIELD, vmin=.25,vmax=1, aspect="auto")
    bar.set_axis_off()
    text(fig, 59, 108.3, "0.25", ha="right", size=7, color=SUB)
    text(fig, 76, 108.3, "1", size=7, color=SUB)


def schematic(fig):
    text(fig, 90, 149.5, "Own-state replay", size=8.1, weight="bold")
    text(fig, 90, 137.3, r"Record native profiles $b^S$ and $b^U$.", size=7.55)
    text(fig, 90, 126.2, r"Evaluate $d_i(z_i)$ on its own state.", size=7.55)
    text(fig, 90, 112.7, r"$\hat d_i=b_i^p\,d_i(z_i)/\mathrm{RMS}(d_i(z_i))$", size=8.35)
    text(fig, 266, 150, "Replay grid (schematic)", size=7.15, ha="center", color=SUB)
    x0, y0, cw, rh = 245, 109.2, 28, 12
    for j, ref in enumerate(("S","U")):
        text(fig, x0+(j+.5)*cw, y0+2*rh+5.4, f"at {ref}", size=7.15, color=COL[ref], ha="center")
    for i, alloc in enumerate(("U","S")):
        y = y0+i*rh
        text(fig, x0-3.3, y+rh/2, "Spatial" if alloc=="S" else "Uniform", ha="right", size=7.1,
             color=COL[alloc])
        for j, ref in enumerate(("S","U")):
            x = x0+j*cw
            fig.add_artist(Rectangle((x/W,y/H),cw/W,rh/H,transform=fig.transFigure,
                                     facecolor="#F1F3F4" if alloc==ref else "white",
                                     edgecolor=HAIR,linewidth=.52))
            text(fig, x+cw/2, y+rh/2, "native" if alloc==ref else "replay",
                 ha="center", size=7.15, weight="normal" if alloc==ref else "bold")


def trace(fig, x, width, title, steps, paths, ticks, gap_color="#E8EDF0", title_offset=0):
    text(fig, x+width/2+title_offset, 95.4, title, ha="center", weight="bold", size=7.7)
    ax = axes(fig, x, 31, width, 56.5)
    ax.set(xlim=(-.5,27.5), ylim=(0,ticks[-1]), xticks=[steps[0],14,27],
           yticks=ticks, yticklabels=[f"{t:g}" for t in ticks])
    ax.tick_params(length=2, pad=1.6, width=.55)
    ax.grid(axis="y",color="#DBE0E3",lw=.42)
    ax.set_axisbelow(True)
    ax.fill_between(steps, paths["SS"], paths["UU"], facecolor=gap_color, lw=0, zorder=1)
    for alloc in ("S","U"):
        ax.plot(steps, paths[alloc*2], c=COL[alloc], ls="-" if alloc=="S" else (0,(3,1.35)),
                lw=1.15, zorder=2.5)
    indices = np.rint(np.linspace(0, len(steps)-1, 6)).astype(int)
    for key in ("US","SU"):
        ax.plot(steps[indices], np.array(paths[key])[indices], marker=MARK[key[0]],
                ls="none", color=COL[key[0]], mfc="white", mew=.83, ms=3.35, zorder=4)
    return ax


def legend(fig):
    for x, alloc, name in [(3,"S","Spatial"),(53,"U","Uniform")]:
        line(fig,x,8.1,x+10.5,8.1,color=COL[alloc],lw=1.2,ls="-" if alloc=="S" else (0,(3,1.35)))
        text(fig,x+13.5,8.1,name,size=7.35)
    for x, alloc, name in [(113,"S","Spatial at U"),(181,"U","Uniform at S")]:
        line(fig,x,8.1,x,8.1,marker=MARK[alloc],mfc="white",mec=COL[alloc],ms=3.65,mew=.83)
        text(fig,x+5,8.1,name,size=7.35)
    text(fig, 302, 8.1, "x = solver step", size=7.05, color=SUB, ha="right")


def mrc(fig, sweeps, scalar, pieces, gap):
    text(fig,317,148.8,"Add flowers to dog’s mouth · 0.25, 0.50, 0.75, 1.00, 1.25",size=7.4)
    ax = axes(fig,345,36,164,101.5)
    lo, hi = pieces[0]["lo"], pieces[-1]["hi"]
    ax.axvspan(lo,hi,color="#EFF3F5",lw=0)
    for piece in pieces:
        ax.fill_between([piece["lo"],piece["hi"]],[piece["S_lo"],piece["S_hi"]],
                        [piece["U_lo"],piece["U_hi"]],color="#CFDCE3",lw=0)
    for x in (lo,hi):
        ax.axvline(x,color="#9EAAB1",lw=.56,ls=(0,(2.3,1.6)),zorder=1)
    for alloc, pts in sweeps.items():
        ax.plot(*pts.T,c=COL[alloc],lw=1.15,marker=MARK[alloc],ms=3.25,mew=.7,mfc="white",
                ls="-" if alloc=="S" else (0,(3,1.35)),zorder=3)
        ax.plot(*pts[3],MARK[alloc],color=COL[alloc],ms=4,zorder=4)
    ax.plot(*scalar,"D",color=SCALAR,mfc=SCALAR,mec="white",mew=.35,ms=4.7,zorder=5)
    ax.annotate("Two constants",xy=scalar,xytext=(.014,.108),size=7.15,color=SUB,
                arrowprops=dict(arrowstyle="-",color=SCALAR,lw=.55,shrinkB=2))
    for key, x, y in [("Uniform",.113,.157),("Spatial",.132,.025)]:
        ax.text(x,y,key,size=7.7,color=COL[key[0]])
    ax.text(.004,.223,"Shared response interval",color=SUB,size=7.4)
    ax.set(xlim=(-.062,.236), ylim=(0,.245), xticks=[-.05,0,.10,.20],yticks=[0,.1,.2],
           xticklabels=["−0.05","0","0.10","0.20"],yticklabels=["0","0.1","0.2"],
           xlabel="Directional CLIP",ylabel="Background LPIPS")
    ax.tick_params(length=2, pad=2, width=.55)
    ax.xaxis.labelpad=1.2
    ax.yaxis.labelpad=1.8
    text(fig,317,7.9,f"Mean U − S over shared range = {gap:+.4f} LPIPS",size=7.6)


def main():
    photo, weight, restoring, branches, sweeps, scalar, pieces, gap = measurements()
    with mpl.rc_context(FONT):
        fig = plt.figure(figsize=(W/72,H/72),facecolor="white")
        text(fig,1,161.9,"(a)  Crossed profile replay",weight="bold",size=8.5)
        text(fig,317,161.9,"(b)  Matched-response comparison",weight="bold",size=8.5)
        line(fig,308.7,3,308.7,165,color=HAIR,lw=.57)
        images(fig,photo,weight)
        schematic(fig)
        line(fig,1,102,302,102,color="#D7DDE0",lw=.48)
        left = trace(fig,27,70,"Restoring · correction",np.arange(28),restoring,[0,.4,.8],
                     gap_color="#EAEAEA")
        left.set_ylabel("Measured RMS",size=7.3,labelpad=2.3)
        trace(fig,119,78,"SAM model (next step)",np.arange(2,28),branches["model_rms"],[0,.15,.30])
        trace(fig,222,78,"SAM visible (final decode)",np.arange(2,28),branches["visible_rms"],[0,.15,.30],
              title_offset=-5)
        legend(fig)
        mrc(fig,sweeps,scalar,pieces,gap)
        save_svg(fig, "protocols", dpi=400)
        plt.close(fig)
