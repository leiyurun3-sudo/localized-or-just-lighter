"""Shared local inputs, SVG export and portable font selection."""
import base64
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from xml.etree import ElementTree

from matplotlib import font_manager
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figures"
ARCHIVE = ROOT / "figures" / "archive"
OUTPUT = ROOT / "figures" / "generated"


def select_font():
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    raise RuntimeError("Install a sans-serif font or restore Matplotlib's bundled DejaVu Sans.")


FONT_FAMILY = select_font()


def read_data(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def image_layers(stem):
    svg = ElementTree.parse(ARCHIVE / f"v19_{stem}.svg")
    return {node.get("id"): node for node in svg.findall(".//{*}image")}


def raster(stem, layer):
    node = image_layers(stem)[layer]
    uri = node.attrib["{http://www.w3.org/1999/xlink}href"]
    header, encoded = uri.split(",", 1)
    if header != "data:image/png;base64":
        raise ValueError(f"Expected an embedded PNG in {stem}, layer {layer}")
    with Image.open(BytesIO(base64.b64decode(encoded))) as im:
        return im.copy()


def flower_source():
    return raster("protocols", "imagea32c6dfa4b").convert("RGB")


def case_frame(case, arm):
    return raster("changes", f"case-{case}-{arm}-frame").convert("RGB")


def save_svg(fig, name, dpi):
    fig.savefig(OUTPUT / f"v19_{name}.svg", dpi=dpi, facecolor="white", metadata={"Date": None})
