"""ShelfSense AI - Autonomous Retail Compliance Agent (Streamlit dashboard).

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from shelfsense import config
from shelfsense.detector import DetectorSettings
from shelfsense.pipeline import load_media, run_audit
from shelfsense.planogram import load_catalog, load_planogram
from shelfsense.recommender import markdown_report

st.set_page_config(page_title="ShelfSense AI", page_icon="🛒", layout="wide")

catalog = load_catalog(config.CATALOG_PATH)
planogram = load_planogram(config.PLANOGRAM_PATH)
rows, cols = planogram["grid"]["rows"], planogram["grid"]["cols"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛒 ShelfSense AI")
    st.caption("Autonomous Retail Compliance Agent")

    mode = st.radio("Detector", ["Demo / Simulated", "Live (Ollama VLM)"],
                    help="Demo mode needs no GPU or model; Live calls a local Qwen2.5-VL through Ollama.")
    known = list(config.MODEL_PARAMS_B)
    choice = st.selectbox("Vision SLM", known + ["custom..."], index=known.index(config.DEFAULT_MODEL))
    model = st.text_input("Custom model tag", "qwen2.5vl:7b") if choice == "custom..." else choice
    host = st.text_input("Ollama host", config.OLLAMA_HOST)

    params = config.MODEL_PARAMS_B.get(model)
    if params is not None:
        ok = params < config.SLM_LIMIT_B
        st.success(f"SLM: {model} ({params}B params < {config.SLM_LIMIT_B}B) ✅") if ok else st.warning(f"{model} exceeds the 20B SLM bonus limit")

    st.divider()
    edge = st.toggle("⚡ Edge Simulation Mode", help="Down-scales the image, caps CPU threads and context, and shows edge metrics.")
    max_side, threads = 1024, None
    if edge:
        max_side = st.select_slider("Max image side (px)", [320, 480, 640, 800, 1024], value=640)
        threads = st.slider("CPU threads (Raspberry Pi 5 ~ 4, Jetson ~ 6)", 1, 8, 4)
        budget = st.number_input("Latency budget (ms)", 500, 120000, 30000, step=500)

    st.divider()
    hours = st.number_input("Store hours per day", 4, 24, int(planogram["store_hours_per_day"]))
    st.caption(f"Golden zone: Row {planogram['golden_row']} (eye level) carries a "
               f"{planogram['row_weights'][str(planogram['golden_row'])]}x weight.")

st.title("ShelfSense AI - Shelf Compliance Audit")
tab_upload, tab_visual, tab_report, tab_assume = st.tabs(["1 · Upload", "2 · Visual Audit", "3 · Action Report", "Assumptions"])

# ------------------------------------------------------------------ 1. upload
with tab_upload:
    left, right = st.columns([1, 1])
    with left:
        samples = sorted(p.name for p in config.SAMPLES_DIR.glob("*.png"))
        sample = st.selectbox("Bundled demo shelf (intentional errors)", ["-"] + samples)
        up = st.file_uploader("...or upload a shelf photo / video", type=["png", "jpg", "jpeg", "mp4", "mov"])
        st.caption("Images are mapped to the planogram grid (4 rows x 5 slots). For videos the middle frame is audited.")
    image, name = None, None
    if up is not None:
        image, name = load_media(up.getvalue(), up.name), up.name
    elif sample != "-":
        from PIL import Image
        image, name = Image.open(config.SAMPLES_DIR / sample).convert("RGB"), sample
    with right:
        if image is not None:
            st.image(image, caption=name, use_container_width=True)

    if st.button("🔍 Run shelf audit", type="primary", disabled=image is None):
        cfg = DetectorSettings(mode="live" if mode.startswith("Live") else "demo", model=model, host=host,
                               rows=rows, cols=cols, edge=edge, max_side=max_side, num_thread=threads)
        with st.spinner("Auditing shelf..."):
            st.session_state["result"] = run_audit(image, name, cfg, catalog, planogram, hours)
            st.session_state["edge"] = {"on": edge, "budget": budget if edge else None}
        st.success("Audit complete - open the Visual Audit and Action Report tabs.")

res = st.session_state.get("result")

# ------------------------------------------------------------------ 2. visual audit
with tab_visual:
    if not res:
        st.info("Run an audit first.")
    else:
        det = res["detection"]
        if det.get("note"):
            st.warning(det["note"])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(res["annotated"], use_container_width=True)
            st.caption("🟩 OK  🟥 empty  🟧 misplaced  🟪 unknown product  🟨 low stock  🟣 bar = price/label issue")
        with c2:
            ev = res["evaluation"]
            st.metric("Planogram compliance", f"{ev['score']}%")
            st.progress(min(ev["score"] / 100, 1.0))
            st.metric("Availability score", f"{ev['availability_score']}%")
            st.write("**Compliance by row**")
            st.bar_chart(pd.Series({f"Row {r}": v for r, v in ev["row_scores"].items()}))
        st.subheader("Slot-by-slot detail")
        df = pd.DataFrame(ev["slots"])[["loc", "expected_name", "found_name", "status", "stock", "price_tag", "weight"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ 3. action report
with tab_report:
    if not res:
        st.info("Run an audit first.")
    else:
        ev, acts = res["evaluation"], res["actions"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Compliance score", f"{ev['score']}%")
        k2.metric("Violations", len(acts))
        k3.metric("Critical", sum(a["severity"] == "Critical" for a in acts))
        k4.metric("Revenue at risk", f"₹{res['total_risk']:,.0f}/day")

        if acts:
            top = acts[0]
            st.info(f"**Fix first:** {top['location']} - saves about ₹{top['revenue_at_risk']:,.0f}/day. {top['instruction']}")

        st.subheader("Prioritised corrective actions")
        adf = pd.DataFrame(acts)
        if not adf.empty:
            st.dataframe(adf[["priority", "severity", "location", "instruction", "revenue_at_risk"]],
                         use_container_width=True, hide_index=True,
                         column_config={"revenue_at_risk": st.column_config.NumberColumn("₹/day at risk", format="₹%d")})
            st.bar_chart(adf.set_index("location")["revenue_at_risk"])

        st.subheader("Label & price verification")
        if res["price_issues"]:
            st.dataframe(pd.DataFrame(res["price_issues"])[["loc", "type", "tag_price", "catalog_price", "detail"]],
                         use_container_width=True, hide_index=True)
        else:
            st.success("All shelf labels match the catalogue.")

        st.subheader("Shelf availability")
        a = ev["counts"]
        st.write(f"Empty slots: **{a['EMPTY']}** · Low stock: **{a['LOW']}** · Misplaced: **{a['MISPLACED'] + a['UNKNOWN']}**")

        m = res["detection"]["metrics"]
        if st.session_state.get("edge", {}).get("on"):
            st.subheader("⚡ Edge metrics")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Inference latency", f"{m['latency_ms']:.0f} ms")
            e2.metric("Throughput", f"{m['fps']:.3f} FPS")
            e3.metric("App memory", f"{m['app_memory_mb']:.0f} MB")
            size = m.get("model_disk_mb")
            e4.metric("Quantised model size", f"{size:.0f} MB" if size else "n/a (demo)")
            budget = st.session_state["edge"]["budget"]
            (st.success if m["latency_ms"] <= budget else st.error)(
                f"Latency {'within' if m['latency_ms'] <= budget else 'over'} the {budget} ms edge budget")
            if m.get("model_loaded_mb"):
                st.caption(f"Model resident in memory: {m['model_loaded_mb']:.0f} MB · quantisation: {m.get('quantization')}")

        st.subheader("Export")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Report (Markdown)", markdown_report(res), "shelf_report.md")
        d2.download_button("⬇ Actions (CSV)", adf.to_csv(index=False) if not adf.empty else "", "actions.csv")
        d3.download_button("⬇ Full audit (JSON)", json.dumps(
            {"score": ev["score"], "slots": ev["slots"], "price_issues": res["price_issues"], "actions": acts},
            indent=2, default=str), "audit.json")

# ------------------------------------------------------------------ assumptions
with tab_assume:
    st.markdown(f"""
### Stated assumptions
1. **Grid mapping** - one camera frame covers one shelf bay of {rows} rows x {cols} slots; each slot holds exactly one SKU (one planogram facing group).
2. **Golden zone** - Row {planogram['golden_row']} (eye level) violations carry a **1.5x** weight; bottom row 0.8x. Weights live in `expected_planogram.json`.
3. **Compliance score** - row-weighted share of slots holding the planogram SKU. Empty, misplaced and unknown products all count as non-compliant.
4. **Availability** - full = 1, low (1-2 facings) = 0.5, empty = 0.
5. **Revenue at risk (₹/day)** = units/hour x store hours x price x margin x row weight x loss factor
   (empty 1.0 · misplaced 0.5 · low stock 0.35 · label issues 0.1). Overpriced/underpriced labels use the price gap x daily units instead.
6. **Labels** are compared with the catalogue price of the product the label names; the price tolerance is ₹{config.PRICE_TOLERANCE}.
7. **Backroom stock** comes from the catalogue and decides whether the fix is "refill" or "reorder".
8. **Demo mode** replays recorded detections for the bundled images; unknown images get deterministic simulated errors.
9. **Uncertainty** - unreadable labels are flagged for manual verification, never guessed.
""")
