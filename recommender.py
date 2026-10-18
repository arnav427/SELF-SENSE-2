"""Turns violations into store-associate fix instructions ranked by revenue at risk."""
from __future__ import annotations

import pandas as pd

from . import config


def _severity(v: float) -> str:
    return "Critical" if v >= config.CRITICAL_AT else "High" if v >= config.HIGH_AT else "Medium"


def build_actions(evaluation: dict, price_issues: list[dict], catalog: pd.DataFrame,
                  planogram: dict, store_hours: float | None = None) -> list[dict]:
    hours = store_hours or planogram["store_hours_per_day"]
    cat = catalog.set_index("SKU_ID")
    weights = {int(k): float(v) for k, v in planogram["row_weights"].items()}
    slots = {(s["row"], s["slot"]): s for s in evaluation["slots"]}
    home = {p["sku"]: (p["row"], p["slot"]) for p in planogram["slots"]}

    def profit_per_day(sku, row, factor):
        r = cat.loc[sku]
        return float(r["Units_Per_Hour"] * hours * r["Expected_Price"] * r["Margin"] * weights.get(row, 1.0) * factor)

    actions, handled = [], set()

    def add(kind, loc, instruction, risk, detail=""):
        actions.append({"type": kind, "location": loc, "instruction": instruction,
                        "revenue_at_risk": round(risk, 0), "severity": _severity(risk), "detail": detail})

    for s in evaluation["slots"]:
        key, exp = (s["row"], s["slot"]), s["expected_sku"]
        backroom = int(cat.loc[exp, "Stock_Backroom"])
        eye = " [Eye-level 1.5x]" if s["row"] == planogram.get("golden_row") else ""

        if s["status"] == "EMPTY":
            risk = profit_per_day(exp, s["row"], config.LOSS_FACTORS["EMPTY"])
            msg = (f"Refill {s['expected_name']} at {s['loc']} from backroom ({backroom} units available)"
                   if backroom > 0 else
                   f"URGENT: {s['expected_name']} at {s['loc']} is out of stock with 0 backroom units - raise a reorder")
            add("STOCK_OUT", s["loc"], msg + eye, risk, "Empty shelf space")

        elif s["status"] in ("MISPLACED", "UNKNOWN") and key not in handled:
            handled.add(key)
            found = s["found_sku"]
            if found is None:
                risk = profit_per_day(exp, s["row"], config.LOSS_FACTORS["UNKNOWN"])
                add("UNKNOWN_PRODUCT", s["loc"],
                    f"Remove unrecognised product '{s['found_name']}' from {s['loc']} and stock {s['expected_name']}" + eye,
                    risk, "Product not in catalogue")
                continue
            hr, hc = home[found]
            other = slots[(hr, hc)]
            if other["found_sku"] == exp:  # clean swap
                handled.add((hr, hc))
                risk = (profit_per_day(exp, s["row"], config.LOSS_FACTORS["MISPLACED"])
                        + profit_per_day(found, hr, config.LOSS_FACTORS["MISPLACED"]))
                add("SWAP", f"{s['loc']} <-> {other['loc']}",
                    f"Swap {s['found_name']} (at {s['loc']}) with {other['found_name']} (at {other['loc']})" + eye,
                    risk, "Two products are in each other's slots")
            else:
                risk = profit_per_day(exp, s["row"], config.LOSS_FACTORS["MISPLACED"])
                extra = (f"; then stock {s['expected_name']} here from backroom ({backroom} units)"
                         if backroom > 0 else f"; {s['expected_name']} has no backroom stock - reorder")
                add("MISPLACED", s["loc"],
                    f"Move {s['found_name']} from {s['loc']} to {other['loc']} (its planogram slot){extra}" + eye,
                    risk, "Wrong product in slot")

        if s["status"] == "OK" and s["stock"] == "low":
            risk = profit_per_day(exp, s["row"], config.LOSS_FACTORS["LOW"])
            msg = (f"Replenish {s['expected_name']} at {s['loc']} - low stock ({backroom} units in backroom)"
                   if backroom > 0 else f"{s['expected_name']} at {s['loc']} is running low and backroom is empty - reorder")
            add("LOW_STOCK", s["loc"], msg + eye, risk, "Low facings")

    for p in price_issues:
        sku = p["sku"]
        if sku is None:
            continue
        r = cat.loc[sku]
        vel_day = float(r["Units_Per_Hour"] * hours * weights.get(p["row"], 1.0))
        t = p["type"]
        if t == "PRICE_MISMATCH_OVERPRICED":
            risk = (p["tag_price"] - p["catalog_price"]) * vel_day
            msg = f"Correct label at {p['loc']}: shows {p['tag_price']:g}, should be {p['catalog_price']:g} (overcharging risk)"
        elif t == "PRICE_MISMATCH_UNDERPRICED":
            risk = (p["catalog_price"] - p["tag_price"]) * vel_day
            msg = f"Correct label at {p['loc']}: shows {p['tag_price']:g}, should be {p['catalog_price']:g} (margin leak)"
        elif t == "LABEL_PRODUCT_MISMATCH":
            risk = profit_per_day(sku, p["row"], config.LOSS_FACTORS["LABEL"]) * 2
            msg = f"Replace label at {p['loc']}: {p['detail']}"
        else:  # missing / unreadable
            risk = profit_per_day(sku, p["row"], config.LOSS_FACTORS["LABEL"])
            msg = f"Print or re-attach a legible label at {p['loc']} ({r['Name']}, {p['catalog_price']:g})"
        add(t, p["loc"], msg, risk, p["detail"])

    actions.sort(key=lambda a: -a["revenue_at_risk"])
    for i, a in enumerate(actions, 1):
        a["priority"] = i
    return actions


def markdown_report(result: dict) -> str:
    ev, acts = result["evaluation"], result["actions"]
    lines = [
        "# ShelfSense AI - Shelf Audit Report", "",
        f"- **Image:** {result['image_name']}",
        f"- **Detector:** {result['detection']['mode']} ({result['detection'].get('model') or 'n/a'})",
        f"- **Planogram compliance score:** {ev['score']}%",
        f"- **Availability score:** {ev['availability_score']}%",
        f"- **Total revenue at risk:** Rs {result['total_risk']:,.0f}/day", "",
        "## Prioritised corrective actions", "",
        "| # | Severity | Location | Action | Rs/day at risk |", "|---|---|---|---|---|",
    ]
    for a in acts:
        lines.append(f"| {a['priority']} | {a['severity']} | {a['location']} | {a['instruction']} | {a['revenue_at_risk']:,.0f} |")
    return "\n".join(lines) + "\n"
