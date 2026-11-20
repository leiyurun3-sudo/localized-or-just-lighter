"""Regenerate the selected main figures as editable SVGs using only bundled plot inputs."""
import argparse

from scripts import fig1, fig2, fig3, fig4
from scripts.resources import OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", metavar="N", nargs="*", type=int, help="Figure numbers 1–4 (default: all).")
    args = parser.parse_args()
    renderers = {1: fig1.render, 2: fig2.main, 3: fig3.render, 4: fig4.figure}
    selected = args.figures or list(renderers)
    if any(n not in renderers for n in selected):
        parser.error("choose figure numbers from 1 to 4")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for n in dict.fromkeys(selected):
        renderers[n]()
        print(f"Fig. {n} ready in {OUTPUT}")


if __name__ == "__main__":
    main()
