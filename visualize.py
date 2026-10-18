"""Draws the Visual Audit overlay on the shelf image."""
from __future__ import annotations

from PIL import Image, ImageDraw

COLORS = {"OK": (46, 160, 67), "EMPTY": (220, 38, 38), "MISPLACED": (245, 158, 11),
          "UNKNOWN": (168, 85, 247), "LOW": (234, 179, 8)}


def annotate(image: Image.Image, evaluation: dict, price_issues: list[dict], rows: int, cols: int) -> Image.Image:
    img = image.convert("RGB").copy()
    w, h = img.size
    cw, ch = w / cols, h / rows
    d = ImageDraw.Draw(img, "RGBA")
    flagged = {p["loc"] for p in price_issues}
    for s in evaluation["slots"]:
        x0, y0 = (s["slot"] - 1) * cw, (s["row"] - 1) * ch
        x1, y1 = x0 + cw, y0 + ch
        key = "LOW" if (s["status"] == "OK" and s["stock"] == "low") else s["status"]
        col = COLORS[key]
        d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=col + (255,), width=4, fill=col + (28 if key != "OK" else 0,))
        label = s["loc"] if key == "OK" else f"{s['loc']} {key}"
        d.rectangle([x0 + 4, y0 + 4, x0 + 12 + 7 * len(label), y0 + 20], fill=col + (230,))
        d.text((x0 + 8, y0 + 6), label, fill=(255, 255, 255))
        if s["loc"] in flagged:
            d.rectangle([x0 + 4, y1 - 12, x1 - 4, y1 - 5], fill=(217, 70, 239, 230))
    return img
