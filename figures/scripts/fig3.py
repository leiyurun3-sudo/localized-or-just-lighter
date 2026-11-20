"""Recompose Fig. 3 from archival frames, masks, crop coordinates and recorded scores."""
import base64
from functools import lru_cache
import io
from xml.sax.saxutils import escape

import contourpy
from fontTools.ttLib import TTFont
import matplotlib
from matplotlib import font_manager
import numpy as np
from PIL import Image

from .resources import FONT_FAMILY, OUTPUT, case_frame, raster, read_data

ARMS = [
    ("source", "Source", "#3C4147"),
    ("uniform", "Uniform", "#C85A32"),
    ("two_constant", "Two constants", "#6E7781"),
    ("sam_native", "Spatial", "#1F5A8C"),
    ("flowedit_24", "FlowEdit", "#2A8C76"),
]
WIDTH, HEIGHT = 515.52, 400
NATIVE_PX, CROP_PX = 512, 96
FRAME, MAP, ZOOM_FACTOR, CLIP = 64.0, 43.2, 2.5, 0.25
DETAIL = FRAME * CROP_PX / NATIVE_PX * ZOOM_FACTOR
CELL = FRAME + 2 + DETAIL
PITCH = (WIDTH - 4 - CELL) / (len(ARMS) - 1)
ROI_COLORS = ["#F2C55F", "#65C8E5"]
PALETTE = np.rint(matplotlib.colormaps["inferno"](np.arange(256)/255)[:, :3] * 255).astype("uint8")


@lru_cache(maxsize=2)
def advances(bold):
    prop = font_manager.FontProperties(family=FONT_FAMILY, weight="bold" if bold else "normal")
    with TTFont(font_manager.findfont(prop)) as face:
        em = face["head"].unitsPerEm
        metrics = face["hmtx"].metrics
        return {code: metrics[glyph][0] / em for code, glyph in face.getBestCmap().items()}


def rect(x, y, w, h, style):
    return f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" {style}/>'


class Plate:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.items = [rect(0, 0, width, height, 'fill="#FFFFFF"')]

    def label(self, x, y, s, size=7.3, bold=False, align="left", gray=False, color=None):
        width = size * sum(advances(bold)[ord(char)] for char in s)
        x -= width * {"left": 0, "center": .5, "right": 1}[align]
        col = color or ("#444444" if gray else "#171717")
        self.items.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="{FONT_FAMILY}" font-size="{size}" '
                          f'font-weight="{700 if bold else 400}" fill="{col}">{escape(s)}</text>')

    def image(self, name, im, x, y, w, h=None):
        b = io.BytesIO()
        im.save(b, "PNG", optimize=True)
        raw = base64.b64encode(b.getvalue()).decode()
        self.items.append(f'<image id="{name}" x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{(h or w):.3f}" '
                          f'preserveAspectRatio="none" xlink:href="data:image/png;base64,{raw}"/>')

    def write(self):
        start = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                 f'width="{self.width}pt" height="{self.height}pt" viewBox="0 0 {self.width} {self.height}">')
        (OUTPUT / "v19_changes.svg").write_text("\n".join([start, *self.items, "</svg>"]), encoding="utf-8")


def colored(q):
    im = Image.frombytes("P", (q.shape[1], q.shape[0]), q.astype("uint8").tobytes())
    im.putpalette(PALETTE.ravel().tolist())
    return im


def outline(mask):
    padded = np.pad(mask.astype(float), 1)
    coords = np.arange(padded.shape[0]) - .5
    lines = contourpy.contour_generator(x=coords, y=coords, z=padded).lines(.5)
    result = []
    for line in lines:
        p = line[:-1] if np.array_equal(line[0], line[-1]) else line
        keep = np.any((p-np.roll(p,1,axis=0)) != (np.roll(p,-1,axis=0)-p), axis=1)
        points = p[keep]
        if len(points) < 3:
            continue
        # Bake in the gaps so SVG importers and the PDF renderer preserve the dashes.
        points = np.vstack((points, points[0]))
        distances = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
        unit = NATIVE_PX / MAP
        for start in np.arange(0, distances[-1], 2.63 * unit):
            end = min(start + 1.55 * unit, distances[-1])
            i, j = np.searchsorted(distances, [start,end], side="right")
            xs = np.interp([start,end], distances, points[:,0])
            ys = np.interp([start,end], distances, points[:,1])
            run = np.vstack(([xs[0],ys[0]], points[i:j], [xs[1],ys[1]]))
            result.append("M"+"L".join(f"{x:.2f},{y:.2f}" for x,y in run))
    return "".join(result)


def header(p):
    # The three SAM-Flow columns form one comparison; FlowEdit is an external reference.
    p.label(2 + CELL/2, 6.9, "PIE mask below", 7.0, align="center", gray=True)
    center = 2 + 2 * PITCH + CELL / 2
    p.label(center, 6.9, "SAM-Flow", 7.0, align="center", gray=True)
    for start, end in [(2+PITCH, center-21), (center+21, 2+3*PITCH+CELL)]:
        p.items.append(rect(start, 4.4, end-start, .45, 'fill="#B5BBC0"'))
    p.label(2 + 4*PITCH + CELL/2, 6.9, "Reference editor", 7.0, align="center", gray=True)
    for j, (_, label, color) in enumerate(ARMS):
        x = 2 + j*PITCH
        p.label(x+CELL/2, 16.5, label, 8, True, "center")
        p.items.append(rect(x, 19, CELL, 1.25, f'fill="{color}"'))


def location_box(p, x, y, size, color):
    for border, weight in [("#25282B", 1.32), (color, .80)]:
        p.items.append(rect(x, y, size, size, f'fill="none" stroke="{border}" stroke-width="{weight}"'))


def heatmap(p, tag, delta, contour, x, y):
    levels = np.rint(255 * np.clip(delta / CLIP, 0, 1))
    p.image(tag+"-change", colored(levels), x, y, MAP)
    scale = MAP / NATIVE_PX
    # The dark hairline keeps the white dashes visible on bright inferred changes.
    for color, width in [("#1C1C1C", .86), ("#FFFFFF", .52)]:
        p.items.append(f'<path d="{contour}" transform="translate({x} {y}) scale({scale})" '
                       f'fill="none" stroke="{color}" stroke-width="{width/scale:.4f}"/>')


def footer(p):
    y = 396
    p.label(2, y, "Abs. RGB change", 7.2, True)
    p.label(74, y, "0", 7.1, align="right")
    ramp = colored(np.tile(np.arange(256), (12, 1)))
    p.image("inferno-scale", ramp, 78, y-5.4, 48, 5.0)
    p.label(130, y, "0.25+", 7.1)
    p.items.append(rect(169, y-6, 16, 6.5, 'fill="#382F40"'))
    p.items.append(f'<path d="M170 {y-2.8}h2m1.2 0h2m1.2 0h2m1.2 0h2" '
                   'fill="none" stroke="#FFFFFF" stroke-width=".58"/>')
    p.label(189, y, "PIE edit-mask boundary", 7.1)
    p.label(284, y, "96 × 96 px crops at 2.5×", 7.1)
    p.label(WIDTH-2, y, "Per-image scores  ·  seed 0", 7.1, align="right")


def render():
    data = read_data("cases.json")
    scores = {(row["case"], row["arm"]): row for row in data["scores"]}
    p = Plate(WIDTH, HEIGHT)
    header(p)
    for i, sample in enumerate(data["cases"]):
        case = sample["case"]
        top = 32 + i * 122.2
        src = case_frame(case, "source")
        ref = np.asarray(src, dtype=np.int16)
        mask_values = np.array(raster("changes", f"case-{case}-source-PIE-mask").convert("L"))
        mask = mask_values == 255
        boundary = outline(mask)
        regions = sample["regions"]
        boxes = [region["xyxy"] for region in regions]
        instruction = sample["instruction"]
        p.items.append(f'<g id="case-{case}">')
        p.label(2, top-3.1, f"({chr(97+i)})  {instruction}", 7.9, True)
        if case == "513000000001":
            default_text = f"{data['jar_comparison']['default_difference']:+.4f}".replace("-", "−")
            matched_text = f"{data['jar_comparison']['matched_difference']:+.4f}".replace("-", "−")
            caption_gap = f"BG LPIPS (U−S): default {default_text} / five-setting MRC {matched_text}"
            p.label(211, top-3.1, caption_gap, 7.1)
        # Identify the two landmarks once per row. Their frames use the same order and color in all columns.
        for k, (region, color) in enumerate(zip(regions, ROI_COLORS)):
            name = region["label"]
            lx = WIDTH-105+57*k
            location_box(p, lx, top-8.1, 4.9, color)
            p.label(lx+8.2, top-3.2, name, 7.1)
        for j, (arm, _, _) in enumerate(ARMS):
            x = 2 + j*PITCH
            tag = f"case-{case}-{arm}"
            p.items.append(f'<g id="{tag}">')
            im = src if arm == "source" else case_frame(case, arm)
            p.image(tag+"-frame", im, x, top, FRAME)
            for k, (box, color) in enumerate(zip(boxes, ROI_COLORS)):
                if j == 0:
                    scale = FRAME/NATIVE_PX
                    location_box(p, x+box[0]*scale, top+box[1]*scale, CROP_PX*scale, color)
                crop_x, crop_y = x+FRAME+2, top+k*(FRAME-DETAIL)
                p.image(tag+f"-crop-{k+1}", im.crop(box), crop_x, crop_y, DETAIL)
                location_box(p, crop_x, crop_y, DETAIL, color)
            my = top+FRAME+1.6
            if arm == "source":
                # A binary annotation, in the same position as the change maps in the four result columns.
                p.image(tag+"-PIE-mask", Image.fromarray(mask_values), x, my, MAP)
                p.items.append(rect(x, my, MAP, MAP, 'fill="none" stroke="#72777C" stroke-width=".34"'))
                entries = ("Bg. LPIPS", "Dir. CLIP", "Edit/bg MAE")
            else:
                delta = np.abs(np.asarray(im, dtype=np.int16)-ref).mean(axis=2)/255
                heatmap(p, tag, delta, boundary, x, my)
                inside, outside = delta[mask].mean(), delta[~mask].mean()
                score = scores[case, arm]
                if not np.allclose([inside, outside], [score["mae_edit"], score["mae_background"]],
                                   rtol=0, atol=1e-7):
                    raise ValueError(f"Archived pixels differ from measured scores for {case}/{arm}")
                lpips, clip = score["bg_lpips"], score["directional_clip"]
                entries = (f"{lpips:.3f}", f"{clip:.3f}".replace("-", "−"), f"{inside/outside:.2f}×")
            for k, value in enumerate(entries):
                p.label(x+70, my+11+12.0*k, value, 7.5 if j else 7.1, bool(j), "center")
            p.items.append("</g>")
        p.items.append("</g>")
    footer(p)
    p.write()
