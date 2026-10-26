#!/usr/bin/env python3
"""Plot available trigger patterns with a configurable scalar intensity."""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TRIGGER_VARIANTS = [
    {"position": "bottom_right", "size": 1, "intensity": 0.5},
    {"position": "bottom_right", "size": 3, "intensity": 0.5},
    {"position": "bottom_right", "size": 5, "intensity": 0.5},
    {"position": "center", "size": 3, "intensity": 0.5},
    {"position": "top_right", "size": 3, "intensity": 0.5},
    {"position": "center_cross", "size": 5, "intensity": 0.5},
    {"position": "top_center_left", "size": 1, "intensity": 0.5},
    {"position": "twin_lines", "size": 3, "intensity": 0.5},
]

TRIGGER_POSITION_ALIASES = {
    "topright": "top_right",
    "top_right": "top_right",
    "bottomright": "bottom_right",
    "bottom_right": "bottom_right",
    "center": "center",
    "both": "both",
    "topcenterleft": "top_center_left",
    "top_center_left": "top_center_left",
    "centercross": "center_cross",
    "center_cross": "center_cross",
    "twinlines": "twin_lines",
    "twin_lines": "twin_lines",
    "twin_stacked_lines": "twin_lines",
    "twin_top_stacked_lines": "twin_lines",
}


def normalize_trigger_position(trigger_position):
    normalized = str(trigger_position).strip().lower().replace("-", "_").replace(" ", "_")
    compact = normalized.replace("_", "")
    return TRIGGER_POSITION_ALIASES.get(normalized, TRIGGER_POSITION_ALIASES.get(compact, normalized))


def clip_bounds(height, width, h_start, h_end, w_start, w_end):
    h_start = max(0, h_start)
    w_start = max(0, w_start)
    h_end = min(height, h_end)
    w_end = min(width, w_end)
    if h_start >= h_end or w_start >= w_end:
        return None
    return (h_start, h_end, w_start, w_end)


def get_trigger_bounds(height, width, trigger_size, trigger_position):
    trigger_size = max(1, int(trigger_size))
    trigger_position = normalize_trigger_position(trigger_position)
    bounds = []

    if trigger_position == "center":
        h_start = (height - trigger_size) // 2
        w_start = (width - trigger_size) // 2
        bounds.append((h_start, h_start + trigger_size, w_start, w_start + trigger_size))
    elif trigger_position == "both":
        bounds.extend(
            [
                (height - trigger_size, height, width - trigger_size, width),
                (0, trigger_size, 0, trigger_size),
            ]
        )
    elif trigger_position == "top_right":
        bounds.append((0, trigger_size, width - trigger_size, width))
    elif trigger_position == "top_center_left":
        h_start = 0
        w_start = max(0, (width // 2) - trigger_size - 1)
        bounds.append((h_start, h_start + trigger_size, w_start, w_start + trigger_size))
    elif trigger_position == "center_cross":
        center_h = height // 2
        center_w = width // 2
        arm = max(1, trigger_size - 1)
        for d in range(-arm, arm + 1):
            # Rotated 45 degrees: thin diagonal X with wider span than the square trigger.
            bounds.append((center_h + d, center_h + d + 1, center_w + d, center_w + d + 1))
            bounds.append((center_h + d, center_h + d + 1, center_w - d, center_w - d + 1))
    elif trigger_position == "twin_lines":
        gap = 2
        line_len = trigger_size
        total_width = 2 * line_len + gap
        left = max(0, (width - total_width) // 2)
        second_left = left + line_len + gap
        bounds.extend(
            [
                (0, 1, left, left + line_len),
                (2, 3, left, left + line_len),
                (0, 1, second_left, second_left + line_len),
                (2, 3, second_left, second_left + line_len),
            ]
        )
    else:  # bottom_right
        bounds.append((height - trigger_size, height, width - trigger_size, width))

    clipped_bounds = []
    for h_start, h_end, w_start, w_end in bounds:
        clipped = clip_bounds(height, width, h_start, h_end, w_start, w_end)
        if clipped is not None:
            clipped_bounds.append(clipped)
    return clipped_bounds


def parse_intensity(trigger_value):
    value = float(trigger_value)
    return np.array([value, value, value], dtype=np.float32)


def pretty_position_name(position):
    return normalize_trigger_position(position).replace("_", " ")


def parse_trigger_spec(spec):
    """Parse a trigger spec string in the form: position:size:intensity."""
    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --trigger_spec '{spec}'. Expected format: position:size:intensity"
        )
    position, size_text, intensity_text = parts
    size = max(1, int(size_text))
    intensity = float(intensity_text)
    return {
        "position": normalize_trigger_position(position),
        "size": size,
        "intensity": intensity,
    }


def render_trigger_canvas(image_size, trigger_size, trigger_position, intensity_rgb):
    canvas = np.ones((image_size, image_size, 3), dtype=np.float32) * 0.08
    for h_start, h_end, w_start, w_end in get_trigger_bounds(
        image_size, image_size, trigger_size, trigger_position
    ):
        canvas[h_start:h_end, w_start:w_end, :] = intensity_rgb
    return np.clip(canvas, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Plot all trigger pattern types")
    parser.add_argument("--image_size", type=int, default=32, help="Canvas side length in pixels")
    parser.add_argument("--trigger_size", type=int, default=3, help="Trigger size parameter")
    parser.add_argument(
        "--trigger_value",
        type=float,
        default=0.5,
        help="Trigger scalar intensity in [0, 1]",
    )
    parser.add_argument(
        "--trigger_spec",
        action="append",
        default=[],
        help=(
            "Override trigger variants with entries formatted as "
            "position:size:intensity (example: bottom_right:7:0.5). "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--title_fontsize",
        type=float,
        default=14,
        help="Font size for each subplot title",
    )
    parser.add_argument("--output", type=str, default="trigger_types.png", help="Output image path")

    args = parser.parse_args()
    if args.trigger_spec:
        trigger_variants = [parse_trigger_spec(spec) for spec in args.trigger_spec]
    else:
        trigger_variants = []
        for variant in DEFAULT_TRIGGER_VARIANTS:
            trigger_variants.append(
                {
                    "position": normalize_trigger_position(variant["position"]),
                    "size": max(1, int(variant["size"])),
                    "intensity": float(variant.get("intensity", args.trigger_value)),
                }
            )

    cols = 8
    rows = math.ceil(len(trigger_variants) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for idx, variant in enumerate(trigger_variants):
        position = variant["position"]
        size = variant["size"]
        intensity = variant["intensity"]
        ax = axes[idx]
        canvas = render_trigger_canvas(
            args.image_size,
            size,
            position,
            parse_intensity(intensity),
        )
        ax.imshow(canvas)
        ax.set_title(
            f"{pretty_position_name(position)} | {size}x{size}",
            fontsize=args.title_fontsize,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(trigger_variants), len(axes)):
        axes[idx].axis("off")

    fig.savefig(args.output, dpi=180)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
