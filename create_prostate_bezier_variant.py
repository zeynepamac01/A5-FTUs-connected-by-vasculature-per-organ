"""Create an organic Bézier-link copy of the manually edited prostate SVG.

The source SVG is never modified. The heart connections become two independent
curves, and the FTU transition becomes separate inflow/outflow curves connected
to the single capillary node by two clean capillary links. Typography, nodes, labels, FTU styling,
colors, and canvas metadata remain byte-for-byte identical elsewhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = {
    # Two independent, asymmetric chamber-to-heart curves. The compound path
    # deliberately leaves a gap behind the heart symbol instead of implying a
    # direct chamber-to-chamber vessel.
    (
        'd="M82.45,152.08h-37.92c-.93,0-1.68.75-1.68,1.68-.'
        '1,22.37-.12,38.78,0,39.92.07.77.49,1.19.49,1.19s.42.42,'
        '1.19.49c1.15.11,16.7.1,37.92,0"'
    ): (
        'd="M82.45,152.16 C66.20,152.16 53.74,160.12 42.86,168.24 '
        'M42.86,185.18 C53.74,192.18 66.20,195.36 82.44,195.36"'
    ),
    # Direct arterial approach to the FTU transition.
    'd="M431.84,152.16h95.17c4.33,0,7.83,3.51,7.83,7.83v1.77"': (
        'd="M431.84,152.16 C450.20,152.16 469.10,153.80 485.00,157.50"'
    ),
    # Direct venous departure from the FTU transition.
    'd="M534.84,183.36v4.17c0,4.33-3.51,7.83-7.83,7.83h-106.37"': (
        'd="M485.00,185.00 C468.00,189.10 446.00,195.36 420.64,195.36"'
    ),
    # Two capillary links meet at the existing node: inflow and outflow. There
    # is intentionally no third, decorative branch through the middle.
    'd="M534.84,161.76v9.6"': (
        'd="M485.00,157.50 C503.00,161.80 520.00,168.00 534.84,171.36"'
    ),
    'd="M534.84,171.36v12"': (
        'd="M534.84,171.36 C520.00,175.00 502.00,181.50 485.00,185.00"'
    ),
}


def create_variant(source: Path, destination: Path) -> None:
    svg = source.read_text()
    for original, replacement in REPLACEMENTS.items():
        count = svg.count(original)
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one source path, found {count}: {original}"
            )
        svg = svg.replace(original, replacement)

    generator_comment = (
        "<!-- Generator: Adobe Illustrator 30.8.1, SVG Export Plug-In . "
        "SVG Version: 2.1.4 Build 1)  -->"
    )
    variant_comment = (
        "<!-- Prostate organic Bezier-link variant; source artwork preserved; "
        "heart and FTU vascular transitions changed only. -->"
    )
    svg = svg.replace(
        generator_comment,
        f"{generator_comment}\n  {variant_comment}",
        1,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    create_variant(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
