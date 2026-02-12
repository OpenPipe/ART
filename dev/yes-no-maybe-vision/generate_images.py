from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - clear import guidance
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc
# (resampling constants removed; we no longer scale bitmap masks)


# Minimal scalable font loader (raises if not found; use --font-path to provide one)
def _load_font(font_path: str | Path | None, preferred_size: int) -> Any:
    candidates: list[str] = []
    if font_path is not None:
        candidates.append(str(font_path))
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, preferred_size)
        except Exception:
            continue
    # Fallback to PIL's default bitmap font so the script always runs
    return ImageFont.load_default()


# (scalable font check removed; scalable fonts are required by _load_font)


def generate_yes_no_maybe_prompts() -> list[str]:
    from itertools import permutations

    prompts = []
    for prefix in ("respond", "just respond"):
        for n in (3, 2):
            for words in permutations(("yes", "no", "maybe"), n):
                prompts.append(
                    f"{prefix} with {', '.join(words)}"
                    if n == 3
                    else f"{prefix} with {words[0]} or {words[1]}"
                )
    return prompts


# _load_font implemented above


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int
) -> list[str]:
    """Greedy word-wrapping that ensures each line fits within `max_width`."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = (" ".join(current + [word])).strip()
        left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
        if right - left <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


# (fit helper removed; we use a single binary search to maximize font size)


def _max_fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    image_width: int,
    image_height: int,
    margin_px: int,
    min_size: int = 12,
    max_size: int | None = None,
    font_path: str | Path | None = None,
) -> Any:
    """Binary search the largest font that fits the canvas with wrapping.

    This aggressively grows the font size to fill the available space
    (subject to margins), then returns the largest working size.
    """
    if max_size is None:
        max_size = min(image_width, image_height)

    low = min_size
    high = max_size
    best_font: Any = _load_font(font_path, min_size)

    while low <= high:
        mid = (low + high) // 2
        font = _load_font(font_path, mid)
        lines = _wrap_text_to_width(draw, text, font, image_width - 2 * margin_px)
        # measure
        max_line_width = 0
        line_heights: list[int] = []
        for line in lines:
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            max_line_width = max(max_line_width, int(right - left))
            line_heights.append(int(bottom - top))
        avg_line_height = (
            int(sum(line_heights) / len(line_heights)) if line_heights else mid
        )
        line_spacing = max(1, avg_line_height // 4)
        total_height = sum(line_heights) + (len(lines) - 1) * line_spacing

        fits = (
            max_line_width <= image_width - 2 * margin_px
            and total_height <= image_height - 2 * margin_px
        )
        if fits:
            best_font = font
            low = mid + 2
        else:
            high = mid - 2

    return best_font


def save_prompt_images(
    prompts: Sequence[str] | Iterable[str],
    output_dir: str | Path,
    image_size: tuple[int, int] = (512, 512),
    margin_px: int = 16,
    font_path: str | Path | None = None,
    font_size: int | None = None,
    text_color: tuple[int, int, int] = (0, 0, 0),
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> list[Path]:
    """
    Render each prompt as centered text on a white background and save as PNG.

    Args:
        prompts: Sequence of prompt strings.
        output_dir: Directory to write images into (created if missing).
        image_size: (width, height) for output images.
        margin_px: Padding inside the canvas for text layout.
        font_path: Optional path to a .ttf/.otf font.
        font_size: Optional explicit font size; if None, will auto-fit.
        text_color: RGB color for text.
        background_color: RGB background color.

    Returns:
        List of file Paths written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    used_names: set[str] = set()
    width, height = image_size

    for idx, raw_prompt in enumerate(prompts):
        prompt = str(raw_prompt).strip()
        if not prompt:
            continue

        image = Image.new("RGB", (width, height), color=background_color)
        draw = ImageDraw.Draw(image)

        if font_size is not None:
            font = _load_font(font_path, font_size)
        else:
            # Aggressively maximize font size within margins.
            max_size = min(width, height)
            font = _max_fit_font_size(
                draw,
                prompt,
                width,
                height,
                margin_px,
                min_size=12,
                max_size=max_size,
                font_path=font_path,
            )

        lines = _wrap_text_to_width(draw, prompt, font, width - 2 * margin_px)

        # Single drawing path: measure, center, render
        line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_heights = [int(b[3] - b[1]) for b in line_bboxes]
        max_line_width = max((b[2] - b[0]) for b in line_bboxes) if line_bboxes else 0
        avg_line_height = (
            int(sum(line_heights) / len(line_heights)) if line_heights else 0
        )
        line_spacing = max(1, avg_line_height // 4) if avg_line_height else 8
        total_text_height = sum(line_heights) + (len(lines) - 1) * line_spacing

        y_start = (height - total_text_height) // 2
        cursor_y = y_start
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = int(bbox[2] - bbox[0])
            x = (width - line_width) // 2
            draw.text((x, cursor_y), line, font=font, fill=text_color)
            cursor_y += line_heights[i]
            if i < len(lines) - 1:
                cursor_y += line_spacing

        # Build a deterministic, safe filename
        slug = _slugify(prompt)
        base_name = slug if slug else f"prompt_{idx:04d}"
        name = base_name
        suffix_counter = 1
        while name in used_names:
            name = f"{base_name}_{suffix_counter}"
            suffix_counter += 1
        used_names.add(name)

        out_path = output_path / f"{name}.png"
        image.save(out_path, format="PNG")
        written_paths.append(out_path)

    return written_paths


def _slugify(text: str, max_length: int = 80) -> str:
    """Create a filesystem-friendly slug from text."""
    import re

    # Lowercase, replace whitespace with underscores, strip quotes, keep alnum and _-
    cleaned = text.lower().strip()
    cleaned = cleaned.replace("'", "").replace('"', "")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_\-]", "", cleaned)
    return cleaned[:max_length].strip("_-")


if __name__ == "__main__":
    # Generate prompts and render images
    import argparse

    parser = argparse.ArgumentParser(description="Render prompts as text images")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="dev/yes-no-maybe-vision/images",
        help="Directory to write images (default: dev/yes-no-maybe-images)",
    )
    parser.add_argument(
        "--size", type=int, default=None, help="Square px (overrides width/height)"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Width px (default 256)"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Height px (default 256)"
    )
    parser.add_argument("--margin", type=int, default=16, help="Margin px (default 16)")
    parser.add_argument(
        "--font-path", type=str, default=None, help="Path to .ttf/.otf font"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.size is not None:
        width = height = int(args.size)
    else:
        width = int(args.width) if args.width is not None else 256
        height = int(args.height) if args.height is not None else 256
    margin = int(args.margin)

    saved = save_prompt_images(
        generate_yes_no_maybe_prompts(),
        output_dir,
        image_size=(width, height),
        margin_px=margin,
        font_path=args.font_path,
    )
    print(
        f"Wrote {len(saved)} images to {output_dir.resolve()} at size {width}x{height}"
    )
