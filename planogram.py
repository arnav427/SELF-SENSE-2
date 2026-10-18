"""Planogram compliance: match detections to SKUs, find missing / misplaced items, score."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import pandas as pd


def load_planogram(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_catalog(path) -> pd.DataFrame:
    return pd.read_csv(path)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


def match_sku(name: str, catalog: pd.DataFrame, threshold: float = 0.6):
    """Fuzzy-match a free-text product name to a catalogue SKU. Returns (sku_id | None, score)."""
    n = _norm(name)
    if not n:
        return None, 0.0
    best, best_score = None, 0.0
    for _, row in catalog.iterrows():
        c = _norm(row["Name"])
        score = SequenceMatcher(None, n, c).ratio()
        if n in c or c in n:  # containment boost ("Pepsi" -> "Pepsi 500ml")
            score = max(score, 0.85)
        if score > best_score:
            best, best_score = row["SKU_ID"], score
    return (best, best_score) if best_score >= threshold else (None, best_score)


def evaluate(slots: list[dict], planogram: dict, catalog: pd.DataFrame) -> dict:
    """Compare detected slots with the expected planogram.

    Status per slot: OK | EMPTY | MISPLACED | UNKNOWN.
    Compliance score = row-weighted share of slots holding the right SKU
    (eye-level / golden-zone row carries a higher weight).
    """
    names = dict(zip(catalog["SKU_ID"], catalog["Name"]))
    weights = {int(k): float(v) for k, v in planogram["row_weights"].items()}
    detected = {(int(s["row"]), int(s["slot"])): s for s in slots}

    results = []
    for ps in planogram["slots"]:
        r, c, exp = ps["row"], ps["slot"], ps["sku"]
        d = detected.get((r, c), {})
        product = str(d.get("product") or "").strip()
        stock = str(d.get("stock") or "full").lower()
        if stock not in ("full", "low", "empty"):
            stock = "full"

        found_sku, conf = None, 0.0
        if product.upper() in ("", "EMPTY", "NONE", "NULL") or stock == "empty":
            status, stock = "EMPTY", "empty"
        else:
            found_sku, conf = match_sku(product, catalog)
            if found_sku is None:
                status = "UNKNOWN"
            elif found_sku == exp:
                status = "OK"
            else:
                status = "MISPLACED"

        results.append({
            "row": r, "slot": c, "loc": f"R{r}S{c}", "weight": weights.get(r, 1.0),
            "expected_sku": exp, "expected_name": names[exp],
            "found_sku": found_sku, "found_name": names.get(found_sku, product or None),
            "match_conf": round(conf, 2), "status": status, "stock": stock,
            "price_tag": d.get("price_tag", ""), "label_product": d.get("label_product", ""),
        })

    total_w = sum(x["weight"] for x in results) or 1.0
    ok_w = sum(x["weight"] for x in results if x["status"] == "OK")
    avail_w = sum(x["weight"] * {"full": 1.0, "low": 0.5, "empty": 0.0}[x["stock"]] for x in results)

    row_scores = {}
    for r in sorted({x["row"] for x in results}):
        rr = [x for x in results if x["row"] == r]
        row_scores[r] = round(100 * sum(x["status"] == "OK" for x in rr) / len(rr), 1)

    counts = {k: sum(x["status"] == k for x in results) for k in ("OK", "EMPTY", "MISPLACED", "UNKNOWN")}
    counts["LOW"] = sum(x["stock"] == "low" and x["status"] != "EMPTY" for x in results)

    return {
        "slots": results,
        "score": round(100 * ok_w / total_w, 1),
        "availability_score": round(100 * avail_w / total_w, 1),
        "row_scores": row_scores,
        "counts": counts,
    }
