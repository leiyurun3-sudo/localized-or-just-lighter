"""Teaser. The two arrows in each sketch indicate the gated term's direction.
Only the four endpoints in each right-hand chart have a quantitative ordinate.
"""
from .resources import DATA, FONT_FAMILY, case_frame, flower_source, read_data, save_svg

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as DrawPath
from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

WIDTH, HEIGHT = 251, 185
S, U = "#1F5A8C", "#C85A32"
INK, SUB, RULE = "#24292F", "#4C565D", "#D4D9DC"

def prepare_images():
    weights = np.load(DATA / "flower_gate.npy", allow_pickle=False)
    gate = Image.fromarray(np.rint(weights * 255).astype("uint8"))
    case = "211000000000"
    return flower_source(), gate, case_frame(case, "source"), case_frame(case, "sam_native")

def label(ax, x, y, s, *, size=7.3, color=INK, ha="left", bold=False):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, fontweight="bold" if bold else "normal")

def tile(ax, image, x, y, w, h, depth, z, gate=False):
    raster = np.asarray(image.convert("RGB"))
    # x grows right, y grows down. A slight parallel projection exposes the two slices.
    transform = Affine2D.from_values(w, -depth, 0, h, x, y) + ax.transData
    artist = ax.imshow(raster, origin="upper", extent=(0,1,1,0), transform=transform, zorder=z,
              aspect="auto", interpolation="nearest" if gate else "antialiased")
    outline = [(x,y),(x+w,y-depth),(x+w,y+h-depth),(x,y+h)]
    artist.set_clip_path(Polygon(outline, transform=ax.transData))
    ax.add_patch(Polygon(outline, closed=True,
                         facecolor="none", edgecolor="#737D81", lw=.45, zorder=z+.1))

def vector(ax, points, color, width, dashed=False):
    p=DrawPath(points,[DrawPath.MOVETO,DrawPath.CURVE4,DrawPath.CURVE4,DrawPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=p,arrowstyle="->",color=color,linewidth=width,
                                 mutation_scale=6.5, linestyle=(0,(2.8,1.5)) if dashed else "-",
                                 joinstyle="round",capstyle="round",zorder=3))

def segment(ax, a, b, color, width, style="-"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=5.4,linewidth=width,
                                 color=color,linestyle=style,shrinkA=2.7,shrinkB=3,
                                 capstyle="round",zorder=2))

def signed(x):
    return f"{x:+.4f}".replace("-", "−")

def quantitative(ax, c, y, diagnostic):
    top, bottom = y+3, y+35
    low, high = min(c.values()), max(c.values())
    def pos(key):
        return (165 if key[0]=="U" else 213, bottom-(bottom-top)*(c[key]-low)/(high-low))
    for first,last in (("UU","US"),("SU","SS")):
        segment(ax,pos(first),pos(last),"#A3ABAF",.55)
    for first,last,dash in (("US","SS","-"),("UU","SU",(0,(1.6,1.4)))):
        segment(ax,pos(first),pos(last),SUB,.7,dash)
    default=c["SS"]-c["UU"]
    segment(ax,pos("UU"),pos("SS"),INK,1.05+2.4*abs(default))
    for key,value in c.items():
        x,cy=pos(key)
        spatial=key[0]=="S"
        col=S if spatial else U
        native=key[0]==key[1]
        ax.plot(x,cy,marker="o" if spatial else "s",ls="none",ms=4.4,mew=.9,
                mfc=col if native else "white",mec=col,zorder=5)
        label(ax,x+(4.3 if spatial else -4.3),cy,f"{value:.4f}",
              size=7.1,ha="left" if spatial else "right",bold=native)
    for x, title, contrast in ((138,"Default",default), (183,"At S",c["SS"]-c["US"]),
                               (228,"At U†" if diagnostic else "At U",c["SU"]-c["UU"])):
        label(ax,x,y+44,title,size=7.3,color=SUB,ha="center")
        label(ax,x,y+53,signed(contrast),size=7.7,bold=True,ha="center")

def sketch(ax, y, restoring):
    anchor=(64,y+32)
    if restoring:
        start=(104,y+9)
        vector(ax,[start,(93,y+8),(83,y+15),(68,y+28)],U,1.15,True)
        vector(ax,[start,(101,y+17),(94,y+21),(86,y+23)],S,1.35)
        ax.plot(*start,".",color=INK,ms=4.1)
        label(ax,82,y+10,"U",color=U,bold=True,size=7.1)
        label(ax,97,y+25,"S",color=S,bold=True,size=7.1)
    else:
        vector(ax,[anchor,(72,y+19),(88,y+10),(105,y+9)],U,1.15,True)
        vector(ax,[anchor,(77,y+27),(80,y+25),(84,y+20)],S,1.35)
        label(ax,93,y+5,"U",color=U,bold=True,size=7.1)
        label(ax,87,y+25,"S",color=S,bold=True,size=7.1)
    ax.plot(*anchor,marker="o",ms=3.7,mec=INK,mfc="white",mew=.9,zorder=6)
    label(ax,61,y+42,"Reference" if restoring else "Anchor",size=7.1,color=SUB)

def render():
    imgs=prepare_images()
    cells=read_data("cells.json")
    with mpl.rc_context({"font.family":FONT_FAMILY,"font.size":7.3,"pdf.fonttype":42,"svg.fonttype":"none",
                         "svg.hashsalt":"two-gated-terms","mathtext.fontset":"stix"}):
        fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),facecolor="white")
        ax=fig.add_axes([0,0,1,1],xlim=(0,WIDTH),ylim=(HEIGHT,0))
        ax.set_axis_off()
        label(ax,1,5,"Gated term",bold=True)
        label(ax,119,5,"Background LPIPS ↓  ·  Δ = S − U",bold=True)
        label(ax,165,16,"Uniform",ha="center",color=U)
        label(ax,213,16,"Spatial",ha="center",color=S)
        ax.plot([112,112],[0,159],color=RULE,lw=.45)
        for j,(title,desc) in enumerate([("Restoring correction","Source / gate P"),("Departing projection","Source / edited")]):
            y=24+j*72
            label(ax,1,y-4,("(a)  " if not j else "(b)  ")+title,bold=True,size=8.15)
            # Match the first image and the front of the second without hiding the salient content.
            tile(ax,imgs[2*j],2,y+11,33,29,4,1)
            tile(ax,imgs[2*j+1],22,y+20,32,27,4,2,gate=(j==0))
            label(ax,1,y+53,desc,size=7.1,color=SUB)
            sketch(ax,y+3,j==0)
            quantitative(ax,cells[j],y+3,diagnostic=(j==1))
        ax.plot([1,247],[87,87],color=RULE,lw=.55)
        ax.plot([1,247],[158,158],color=RULE,lw=.5)
        segment(ax,(1,166),(14,166),INK,1.1)
        label(ax,17,166,"Native pair",size=7.15)
        ax.plot([61,72],[166,166],color=SUB,lw=.78)
        label(ax,75,166,"At S profile",size=7.15)
        ax.plot([120,131],[166,166],color=SUB,lw=.78,ls=(0,(1.6,1.4)))
        label(ax,134,166,"At U profile",size=7.15)
        ax.plot(182,166,marker="o",ms=3.5,mfc="white",mec=SUB,ls="none",mew=.8)
        label(ax,188,166,"Replayed",size=7.15)
        label(ax,1,178.5,"†  Spatial at U exceeds the native [0, 1] gate range in (b).",size=7.1)
        save_svg(fig, "cross", dpi=450)
        plt.close(fig)
