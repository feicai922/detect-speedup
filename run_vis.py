#!/usr/bin/env python
"""Standalone visualization for inference results (instance segmentation by default).

Examples:
    python run_vis.py ./output/rtmdet/0622_1653/json
    python run_vis.py ./output/yolo/0622_1631/json --geometry bbox --no-fill
    python run_vis.py ./output/yolo/0622_1631/json --image-dir /path/to/source_images
    python run_vis.py ./output/yolo/0622_1631/json --show-legend --min-area 0
"""

import argparse
import os

try:
    from detect_speedup.visualize_predictions import run_visualization
except ImportError as e:
    raise SystemExit(f"Visualization requires PIL/Pillow: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize inference JSON results (instance segmentation by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_vis.py ./output/yolo/0622_1631/json
  python run_vis.py ./output/yolo/0622_1631/json --geometry bbox --no-fill
  python run_vis.py ./output/yolo/0622_1631/json --show-legend
        """,
    )
    parser.add_argument("input_dir", help="Directory containing prediction JSON files")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <input_dir>/../vis)")
    parser.add_argument("--image-dir", default=None,
                        help="Source image directory (default: auto-detect)")
    parser.add_argument("--geometry", choices=("mask", "bbox", "obb"), default="mask",
                        help="Geometry: mask (default), bbox, or obb")
    parser.add_argument("--no-fill", action="store_true",
                        help="Outline only, no semi-transparent fill")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.4],
                        help="Confidence thresholds")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--jpg-quality", type=int, default=92)
    parser.add_argument("--min-area", type=int, default=25,
                        help="Minimum mask area in pixels (default: 25, set 0 to disable)")
    parser.add_argument("--show-legend", action="store_true",
                        help="Show confidence color legend on the image")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(input_dir.rstrip("/")), "vis")

    if args.image_dir is None:
        args.image_dir = os.path.dirname(os.path.dirname(input_dir.rstrip("/")))

    config = {
        "input_dir": input_dir,
        "output_dir": args.output,
        "image_dir": args.image_dir,
        "geometry": args.geometry,
        "thresholds": args.thresholds,
        "line_width": args.line_width,
        "jpg_quality": args.jpg_quality,
        "recursive": args.recursive,
        "mask_fallback_bbox": True,
        "fill_alpha": 0 if args.no_fill else 80,
        "min_mask_area": args.min_area,
        "show_legend": args.show_legend,
    }

    print(
        f"[vis] geometry={config['geometry']}, fill_alpha={config['fill_alpha']}, "
        f"min_area={config['min_mask_area']}, legend={config['show_legend']}, "
        f"input={config['input_dir']}, output={config['output_dir']}",
        flush=True,
    )
    run_visualization(config)


if __name__ == "__main__":
    main()
