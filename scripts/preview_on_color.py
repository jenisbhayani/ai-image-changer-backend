import argparse
import os
import sys
from PIL import Image, ImageDraw


def preview_on(rgba: Image.Image, color=(255, 0, 255)) -> Image.Image:
    """Composites an RGBA image over a solid background color."""
    bg = Image.new("RGBA", rgba.size, color + (255,))
    return Image.alpha_composite(bg, rgba.convert("RGBA"))


def make_checkerboard(size, square_size=20) -> Image.Image:
    """Generates a gray/white checkerboard pattern."""
    w, h = size
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    gray = (200, 200, 200, 255)
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 1:
                draw.rectangle([x, y, x + square_size - 1, y + square_size - 1], fill=gray)
    return img


def preview_on_checkerboard(rgba: Image.Image, square_size=20) -> Image.Image:
    """Composites an RGBA image over a checkerboard background."""
    bg = make_checkerboard(rgba.size, square_size=square_size)
    return Image.alpha_composite(bg, rgba.convert("RGBA"))


def create_contact_sheet(rgba_path: str, output_path: str = None) -> str:
    """
    Renders 4 preview panels into a 2x2 contact sheet:
    - Top-left: Magenta (255, 0, 255)
    - Top-right: Green (0, 160, 0)
    - Bottom-left: White (255, 255, 255)
    - Bottom-right: Checkerboard
    """
    rgba = Image.open(rgba_path).convert("RGBA")
    w, h = rgba.size

    # Resize panels if overly huge for the contact sheet (> 1500px)
    max_dim = 1200
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        panel_w = int(w * scale)
        panel_h = int(h * scale)
        panel_rgba = rgba.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    else:
        panel_w, panel_h = w, h
        panel_rgba = rgba

    p_magenta = preview_on(panel_rgba, (255, 0, 255))
    p_green = preview_on(panel_rgba, (0, 160, 0))
    p_white = preview_on(panel_rgba, (255, 255, 255))
    p_check = preview_on_checkerboard(panel_rgba, square_size=max(10, panel_w // 40))

    sheet = Image.new("RGB", (panel_w * 2, panel_h * 2), (0, 0, 0))
    sheet.paste(p_magenta.convert("RGB"), (0, 0))
    sheet.paste(p_green.convert("RGB"), (panel_w, 0))
    sheet.paste(p_white.convert("RGB"), (0, panel_h))
    sheet.paste(p_check.convert("RGB"), (panel_w, panel_h))

    if output_path is None:
        base, _ = os.path.splitext(rgba_path)
        output_path = f"{base}_contact_sheet.png"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=6)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate color preview contact sheet for an RGBA image.")
    parser.add_argument("input", help="Path to input RGBA image")
    parser.add_argument("--output", "-o", help="Path to output contact sheet PNG", default=None)
    args = parser.parse_args()

    out = create_contact_sheet(args.input, args.output)
    print(f"Contact sheet saved to: {out}")
