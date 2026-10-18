"""Shelf-label price extraction and comparison against the product catalogue."""
from __future__ import annotations

import re

import pandas as pd

from . import config

_PRICE_RE = re.compile(r"(\d{1,5}(?:[.,]\d{1,2})?)")
_reader = None


def parse_price(text) -> float | None:
    """Pull a price out of OCR / VLM text such as 'Rs 45', '₹45.00', 'MRP 40/-'."""
    if text is None:
        return None
    t = str(text).replace("O", "0") if re.search(r"\d", str(text)) else str(text)
    m = _PRICE_RE.search(t)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def read_tag_with_easyocr(image, row: int, slot: int, rows: int, cols: int) -> str:
    """Optional OCR fallback: crop the price-tag strip of one grid cell and read it with EasyOCR."""
    global _reader
    try:
        import easyocr  # type: ignore
        import numpy as np
    except ImportError:
        return ""
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    w, h = image.size
    cw, ch = w / cols, h / rows
    x0, y0 = int((slot - 1) * cw), int((row - 1) * ch + ch * 0.78)
    crop = image.crop((x0, y0, int(x0 + cw), int(row * ch)))
    return " ".join(_reader.readtext(np.array(crop), detail=0))


def check_prices(evaluation: dict, catalog: pd.DataFrame) -> list[dict]:
    """Flag price mismatches, unreadable / missing labels and label-product mismatches.

    The price on a label is compared with the catalogue price of the product the label
    names (falls back to the product physically on the shelf).
    """
    cat = catalog.set_index("SKU_ID")
    issues = []
    from .planogram import match_sku

    for s in evaluation["slots"]:
        if s["status"] == "EMPTY":
            continue  # nothing to price-check; availability handles it
        tag = str(s.get("price_tag") or "").strip()
        label_sku = None
        if s.get("label_product"):
            label_sku, _ = match_sku(s["label_product"], catalog)
        ref_sku = label_sku or s["found_sku"]

        base = {"loc": s["loc"], "row": s["row"], "slot": s["slot"], "sku": ref_sku}

        if s["status"] == "OK" and label_sku and label_sku != s["found_sku"]:
            issues.append({**base, "type": "LABEL_PRODUCT_MISMATCH", "tag_price": parse_price(tag),
                           "catalog_price": None,
                           "detail": f"Label names '{cat.loc[label_sku,'Name']}' but shelf holds '{s['found_name']}'"})
            continue
        if ref_sku is None:
            continue
        cat_price = float(cat.loc[ref_sku, "Expected_Price"])
        if not tag:
            issues.append({**base, "type": "MISSING_LABEL", "tag_price": None, "catalog_price": cat_price,
                           "detail": "No price label found for a stocked product"})
            continue
        price = parse_price(tag)
        if price is None:
            issues.append({**base, "type": "PRICE_UNREADABLE", "tag_price": None, "catalog_price": cat_price,
                           "detail": f"Label text '{tag}' could not be read - verify manually"})
        elif abs(price - cat_price) > config.PRICE_TOLERANCE:
            kind = "OVERPRICED" if price > cat_price else "UNDERPRICED"
            issues.append({**base, "type": f"PRICE_MISMATCH_{kind}", "tag_price": price, "catalog_price": cat_price,
                           "detail": f"Label shows {price:g}, catalogue says {cat_price:g}"})
    return issues
