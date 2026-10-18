"""End-to-end audit: image -> detection -> planogram score -> price check -> prioritised actions."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from . import config, ocr_pricing, planogram as pg, recommender
from .detector import DetectorSettings, detect
from .visualize import annotate


def load_media(data: bytes, filename: str) -> Image.Image:
    """Open an image, or grab the middle frame of a video (needs opencv-python-headless)."""
    if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        import tempfile
        import cv2  # type: ignore
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(data)
        cap = cv2.VideoCapture(tmp.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2, 0))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Could not read a frame from the video")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return Image.open(io.BytesIO(data)).convert("RGB")


def run_audit(image: Image.Image, image_name: str, cfg: DetectorSettings, catalog, planogram,
              store_hours: float | None = None) -> dict:
    sidecar = config.SAMPLES_DIR / f"{Path(image_name).stem}.json"
    det = detect(image, cfg, catalog, planogram, str(sidecar) if sidecar.exists() else None)

    # OCR fallback: if the VLM returned no tag text for a stocked slot, try EasyOCR (if installed).
    if det.mode == "live":
        for s in det.slots:
            if not s["price_tag"] and s["product"].upper() not in ("", "EMPTY"):
                s["price_tag"] = ocr_pricing.read_tag_with_easyocr(image, s["row"], s["slot"], cfg.rows, cfg.cols)

    evaluation = pg.evaluate(det.slots, planogram, catalog)
    price_issues = ocr_pricing.check_prices(evaluation, catalog)
    actions = recommender.build_actions(evaluation, price_issues, catalog, planogram, store_hours)
    return {
        "image_name": image_name, "image": image, "detection": det.__dict__,
        "evaluation": evaluation, "price_issues": price_issues, "actions": actions,
        "total_risk": sum(a["revenue_at_risk"] for a in actions),
        "annotated": annotate(image, evaluation, price_issues, cfg.rows, cfg.cols),
    }
