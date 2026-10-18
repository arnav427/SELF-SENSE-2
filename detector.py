"""Shelf detector: Vision SLM (Qwen2.5-VL via Ollama) with a Demo / Simulated fallback."""
from __future__ import annotations

import base64
import io
import json
import random
import re
from dataclasses import dataclass, field

import pandas as pd
import requests
from PIL import Image

from . import config
from .edge import EdgeMonitor, ollama_model_info


@dataclass
class DetectorSettings:
    mode: str = "demo"                      # "demo" | "live"
    model: str = config.DEFAULT_MODEL
    host: str = config.OLLAMA_HOST
    rows: int = 4
    cols: int = 5
    edge: bool = False                      # edge-simulation mode
    max_side: int = 1024                    # image down-scaling (px)
    num_thread: int | None = None           # CPU thread cap for edge simulation
    num_ctx: int = 4096
    timeout_s: int = 300


@dataclass
class Detection:
    slots: list[dict]
    mode: str
    model: str | None = None
    note: str = ""
    raw: str = ""
    metrics: dict = field(default_factory=dict)


def _prompt(rows: int, cols: int, names: list[str]) -> str:
    return (
        "You are a retail shelf auditor. The image shows one shelf bay with "
        f"{rows} rows (row 1 = top, row {rows} = bottom) and {cols} slots per row (slot 1 = leftmost).\n"
        "For EVERY slot report:\n"
        '- "product": exact name from the known list, "UNKNOWN" if the product is not in the list, or "EMPTY" if no product\n'
        '- "stock": "full" (3+ facings), "low" (1-2 facings) or "empty"\n'
        '- "price_tag": the price text printed on the shelf label for that slot, exactly as printed ("" if none or unreadable)\n'
        '- "label_product": the product name printed on that label ("" if none)\n'
        "Known products: " + "; ".join(names) + "\n"
        'Respond with ONLY valid JSON: {"slots":[{"row":1,"slot":1,"product":"","stock":"","price_tag":"","label_product":""}, ...]}'
    )


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(text[start:end + 1])


def _normalise(slots: list[dict], rows: int, cols: int) -> list[dict]:
    by_key = {}
    for s in slots:
        try:
            by_key[(int(s["row"]), int(s["slot"]))] = s
        except (KeyError, ValueError, TypeError):
            continue
    out = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            s = by_key.get((r, c), {"product": "EMPTY", "stock": "empty", "price_tag": "", "label_product": ""})
            out.append({"row": r, "slot": c, "product": s.get("product", "EMPTY"),
                        "stock": s.get("stock", "full"), "price_tag": str(s.get("price_tag", "") or ""),
                        "label_product": str(s.get("label_product", "") or "")})
    return out


def _to_b64(image: Image.Image, max_side: int) -> str:
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _detect_live(image: Image.Image, cfg: DetectorSettings, names: list[str]) -> tuple[list[dict], str]:
    options = {"temperature": 0, "num_ctx": cfg.num_ctx}
    if cfg.num_thread:
        options["num_thread"] = cfg.num_thread
    payload = {
        "model": cfg.model, "stream": False, "format": "json", "options": options,
        "messages": [{"role": "user", "content": _prompt(cfg.rows, cfg.cols, names),
                      "images": [_to_b64(image, cfg.max_side)]}],
    }
    resp = requests.post(f"{cfg.host}/api/chat", json=payload, timeout=cfg.timeout_s)
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    return _normalise(_parse_json(raw).get("slots", []), cfg.rows, cfg.cols), raw


def simulate(planogram: dict, catalog: pd.DataFrame, seed: int = 7) -> list[dict]:
    """Deterministic simulated shelf with a few injected errors (used when no sidecar exists)."""
    rng = random.Random(seed)
    names = dict(zip(catalog["SKU_ID"], catalog["Name"]))
    price = dict(zip(catalog["SKU_ID"], catalog["Expected_Price"]))
    slots = [{"row": p["row"], "slot": p["slot"], "product": names[p["sku"]], "stock": "full",
              "price_tag": f"Rs {price[p['sku']]:g}", "label_product": names[p["sku"]]} for p in planogram["slots"]]
    idx = list(range(len(slots)))
    rng.shuffle(idx)
    a, b, c, d, e = idx[:5]
    slots[a].update(product="EMPTY", stock="empty")
    slots[b].update(stock="low")
    slots[c]["price_tag"] = f"Rs {price[planogram['slots'][c]['sku']] + 5:g}"
    slots[d]["product"], slots[e]["product"] = slots[e]["product"], slots[d]["product"]
    return slots


def detect(image: Image.Image, cfg: DetectorSettings, catalog: pd.DataFrame, planogram: dict,
           sidecar_json: str | None = None) -> Detection:
    """Run detection. `sidecar_json` is a path to demo ground-truth for bundled sample images."""
    names = catalog["Name"].tolist()
    note = ""
    with EdgeMonitor() as mon:
        if cfg.mode == "live":
            try:
                slots, raw = _detect_live(image, cfg, names)
                det = Detection(slots, "live", cfg.model, raw=raw)
            except Exception as exc:  # fall back so the demo never dies
                note = f"Live model unavailable ({type(exc).__name__}: {exc}). Fell back to demo mode."
                slots = None
        else:
            slots = None
        if slots is None or cfg.mode != "live":
            if sidecar_json:
                with open(sidecar_json, encoding="utf-8") as f:
                    slots = _normalise(json.load(f)["slots"], cfg.rows, cfg.cols)
                kind = "demo (recorded sample)"
            else:
                slots = simulate(planogram, catalog)
                kind = "demo (simulated errors)"
            det = Detection(slots, kind, None, note=note)
    det.metrics = mon.metrics()
    det.metrics["model_params_b"] = config.MODEL_PARAMS_B.get(cfg.model)
    if det.mode == "live":
        det.metrics.update(ollama_model_info(cfg.host, cfg.model))
    return det
