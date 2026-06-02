"""
app.py — Streamlit UI for Agricultural Multimodal RAG Pipeline
"""

import streamlit as st
import sys
import os
import io
import time

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgriLM — Crop Disease Diagnostics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0f5132 0%, #198754 50%, #20c997 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(15, 81, 50, 0.25);
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border-left: 4px solid #198754;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f5132; }
.metric-card .label { font-size: 0.78rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }

.result-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.result-card h3 { margin-top: 0; color: #0f5132; font-size: 1.1rem; }

.severity-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.severity-critical { background: #f8d7da; color: #842029; }
.severity-high { background: #fff3cd; color: #664d03; }
.severity-medium { background: #cff4fc; color: #055160; }
.severity-low { background: #d1e7dd; color: #0f5132; }

/* ── Latency table ── */
.latency-row {
    display: flex; justify-content: space-between;
    padding: 0.35rem 0; border-bottom: 1px solid #f0f0f0;
    font-size: 0.85rem;
}
.latency-row .layer { color: #495057; }
.latency-row .time { font-weight: 600; color: #0f5132; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa, #e9ecef);
}

/* ── Architecture diagram ── */
.arch-step {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0.8rem; margin: 0.3rem 0;
    background: white; border-radius: 8px;
    border-left: 3px solid #198754;
    font-size: 0.82rem;
}
.arch-step .num {
    background: #198754; color: white; width: 22px; height: 22px;
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 0.7rem; font-weight: 700;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Pipeline (cached singleton) ─────────────────────────────
@st.cache_resource(show_spinner="Loading models... (first time takes ~30s)")
def load_pipeline():
    from config import FAISS_INDEX_PATH
    meta_path = os.path.join(FAISS_INDEX_PATH, "metadata.pkl")
    if not os.path.exists(meta_path):
        st.error("FAISS index not found. Run `python build_index.py` first.")
        st.stop()
    from models.pipeline import AgriMultimodalPipeline
    return AgriMultimodalPipeline()


# ─── Terminal Output ──────────────────────────────────────────
def _print_terminal(result, elapsed):
    """Print diagnosis results and metrics to the terminal (cmd)."""
    print()
    print("=" * 70)
    print("  DIAGNOSIS RESULT  (inference: {:.2f}s)".format(elapsed))
    print("=" * 70)
    print("  Confidence : {:.1f}%".format(result.confidence * 100))
    print("  Severity   : {}".format(result.severity.upper()))
    print("-" * 70)

    print()
    print("[DIAGNOSIS]")
    print(result.diagnosis)

    print()
    print("[EXPLANATION]")
    print(result.explanation)

    print()
    print("[RECOMMENDATION]")
    print(result.recommendation)

    if result.evidence_sources:
        print()
        print("[EVIDENCE SOURCES]")
        print("  {:<35} {:<15} {}".format("Disease", "Crop", "Score"))
        print("  {} {} {}".format("-" * 35, "-" * 15, "-" * 8))
        for src in result.evidence_sources:
            score = "{:.4f}".format(src["score"])
            print("  {:<35} {:<15} {}".format(src["disease"], src["crop"], score))

    m = result.metrics
    if m:
        print()
        print("=" * 70)
        print("  EVALUATION METRICS")
        print("=" * 70)

        print()
        print("  Modalities Used   : {}".format(", ".join(m.get("modalities_used", []))))
        print("  Top-K Retrieved   : {}".format(m.get("top_k", 0)))

        print()
        print("  --- Retrieval Scores ---")
        print("  Top-1 Score       : {:.4f}".format(m.get("top1_score", 0)))
        print("  Mean Score        : {:.4f}".format(m.get("mean_score", 0)))
        print("  Min Score         : {:.4f}".format(m.get("min_score", 0)))
        print("  Max Score         : {:.4f}".format(m.get("max_score", 0)))
        print("  Score Spread      : {:.4f}".format(m.get("score_spread", 0)))
        print("  Score Margin(1-2) : {:.4f}".format(m.get("score_margin", 0)))

        print()
        print("  --- Retrieval Quality ---")
        print("  Precision@K       : {:.4f}  ({:.1f}%)".format(
            m.get("precision_at_k", 0), m.get("precision_at_k", 0) * 100))
        print("  Recall@K          : {:.4f}  ({:.1f}%)".format(
            m.get("recall_at_k", 0), m.get("recall_at_k", 0) * 100))
        print("  F1@K              : {:.4f}  ({:.1f}%)".format(
            m.get("f1_at_k", 0), m.get("f1_at_k", 0) * 100))
        print("  MRR               : {:.4f}".format(m.get("mrr", 0)))
        print("  NDCG@K            : {:.4f}".format(m.get("ndcg_at_k", 0)))
        print("  Fusion Score      : {:.4f}".format(m.get("fusion_score", 0)))

        timings = m.get("timings", {})
        if timings:
            print()
            print("  --- Latency Breakdown ---")
            for layer in ["embedding_text", "embedding_image", "embedding_doc",
                          "fusion", "cross_modal", "multi_vector",
                          "retrieval", "rag_generation"]:
                if layer in timings:
                    label = layer.replace("_", " ").title()
                    print("  {:<20}: {:>8.2f} ms".format(label, timings[layer] * 1000))
            print("  {:<20}: {:>8.2f} ms".format("Total", timings.get("total", 0) * 1000))

    print("=" * 70)
    print()
    sys.stdout.flush()


# ─── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌾 AgriLM — Crop Disease Diagnostics</h1>
    <p>Multimodal RAG Pipeline • CLIP + FAISS + Cross-Modal Transformer</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Pipeline Architecture")
    layers = [
        ("1", "📥 Input", "Image + Text + Document"),
        ("2", "🔢 Embedding", "CLIP ViT-B/32 + MiniLM"),
        ("3", "🔗 Fusion", "αE_T + βE_I + γE_D"),
        ("4", "🤖 Cross-Modal", "2-layer Transformer"),
        ("5", "⭐ Multi-Vector", "k=8 ColBERT-style"),
        ("6", "🔍 FAISS Retrieval", "Top-5 evidence search"),
        ("7", "📝 RAG Generation", "Evidence-conditioned"),
        ("8", "📊 Output", "Diagnosis + Metrics"),
    ]
    for num, name, desc in layers:
        st.markdown(f"""
        <div class="arch-step">
            <div class="num">{num}</div>
            <div><strong>{name}</strong><br/><span style="color:#6c757d;font-size:0.75rem">{desc}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Evidence docs (Top-K)", 3, 10, 5)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#6c757d;font-size:0.75rem'>"
        "Built with CLIP • FAISS • PyTorch<br/>No API keys • Runs locally"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Input Section ────────────────────────────────────────────
col_input, col_result = st.columns([1, 1.4], gap="large")

with col_input:
    st.markdown("### 📋 Describe Symptoms")

    text_query = st.text_area(
        "Symptom description",
        placeholder="e.g. My rice leaves have diamond-shaped gray lesions with brown borders...",
        height=120,
        label_visibility="collapsed",
    )

    uploaded_image = st.file_uploader(
        "Upload crop/leaf image (optional)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    doc_text = st.text_area(
        "Additional document / lab report (optional)",
        placeholder="Paste any additional context, lab reports, or soil data...",
        height=80,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        diagnose_btn = st.button("🩺 Diagnose", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # Example queries
    st.markdown("---")
    st.markdown("##### 💡 Try an example")
    examples = [
        "My rice leaves have diamond-shaped gray lesions with brown borders",
        "Tomato plant leaves show large dark water-soaked spots with white fuzz",
        "My wheat crop has orange-brown pustules scattered on leaf surfaces",
        "Coffee plant leaves show yellow-orange powdery spots on the underside",
        "Banana plant has yellowing lower leaves and brown vascular discoloration",
        "Corn leaves show large cigar-shaped gray-green lesions",
    ]
    for ex in examples:
        if st.button(f"→ {ex[:55]}...", key=ex, use_container_width=True):
            st.session_state["example_query"] = ex
            st.rerun()


# ─── Use example if clicked ──────────────────────────────────
if "example_query" in st.session_state:
    text_query = st.session_state.pop("example_query")


# ─── Run Diagnosis ────────────────────────────────────────────
with col_result:
    if diagnose_btn or text_query:
        if not text_query and not uploaded_image and not doc_text:
            st.warning("Please provide at least a symptom description or an image.")
        else:
            pipeline = load_pipeline()

            # Process image
            image = None
            if uploaded_image:
                from PIL import Image
                image = Image.open(uploaded_image).convert("RGB")

            # Run pipeline
            with st.spinner("Running 8-layer diagnosis pipeline..."):
                start = time.time()
                result = pipeline.predict(
                    image=image,
                    text=text_query,
                    document=doc_text,
                    top_k=top_k,
                )
                elapsed = time.time() - start

            # ── Print to Terminal ──
            _print_terminal(result, elapsed)

            # ── Top Metrics Row ──
            severity_map = {
                "critical": ("🔴", "severity-critical"),
                "high": ("🟠", "severity-high"),
                "medium": ("🟡", "severity-medium"),
                "low": ("🟢", "severity-low"),
            }
            sev_emoji, sev_class = severity_map.get(result.severity, ("⚪", "severity-low"))

            m = result.metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{result.confidence*100:.1f}%</div>
                    <div class="label">Confidence</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{sev_emoji} {result.severity.upper()}</div>
                    <div class="label">Severity</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{elapsed:.2f}s</div>
                    <div class="label">Inference Time</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{m.get('top1_score',0):.3f}</div>
                    <div class="label">Top-1 Similarity</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # ── Diagnosis ──
            st.markdown(f"""<div class="result-card">
                <h3>🩺 Diagnosis</h3>
                <span class="severity-badge {sev_class}">{sev_emoji} {result.severity.upper()}</span>
                <div style="margin-top:0.8rem">{result.diagnosis}</div>
            </div>""", unsafe_allow_html=True)

            # ── Explanation ──
            st.markdown(f"""<div class="result-card">
                <h3>📋 Explanation</h3>
                <div>{result.explanation}</div>
            </div>""", unsafe_allow_html=True)

            # ── Recommendation ──
            st.markdown(f"""<div class="result-card">
                <h3>💊 Recommendation</h3>
                <div>{result.recommendation}</div>
            </div>""", unsafe_allow_html=True)

            # ── Evidence Sources ──
            if result.evidence_sources:
                st.markdown(f"""<div class="result-card">
                    <h3>📚 Evidence Sources</h3>
                </div>""", unsafe_allow_html=True)

                import pandas as pd
                df = pd.DataFrame(result.evidence_sources)
                df.columns = [c.title() for c in df.columns]
                if "Score" in df.columns:
                    df["Score"] = df["Score"].apply(lambda x: f"{x:.4f}")
                st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Evaluation Metrics ──
            if m:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("### 📊 Evaluation Metrics")

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                metrics_display = [
                    (mc1, "Precision@K", m.get("precision_at_k", 0)),
                    (mc2, "Recall@K", m.get("recall_at_k", 0)),
                    (mc3, "F1@K", m.get("f1_at_k", 0)),
                    (mc4, "NDCG@K", m.get("ndcg_at_k", 0)),
                    (mc5, "MRR", m.get("mrr", 0)),
                ]
                for col, label, val in metrics_display:
                    with col:
                        st.metric(label, f"{val*100:.1f}%")

                # Score stats + Latency in two columns
                stat_col, lat_col = st.columns(2)

                with stat_col:
                    st.markdown("##### 📈 Retrieval Scores")
                    stats = {
                        "Top-1 Score": f"{m.get('top1_score',0):.4f}",
                        "Mean Score": f"{m.get('mean_score',0):.4f}",
                        "Min Score": f"{m.get('min_score',0):.4f}",
                        "Max Score": f"{m.get('max_score',0):.4f}",
                        "Score Spread": f"{m.get('score_spread',0):.4f}",
                        "Score Margin (1→2)": f"{m.get('score_margin',0):.4f}",
                        "Fusion Score": f"{m.get('fusion_score',0):.4f}",
                        "Modalities": ", ".join(m.get("modalities_used", [])),
                    }
                    for k, v in stats.items():
                        st.markdown(f"""<div class="latency-row">
                            <span class="layer">{k}</span>
                            <span class="time">{v}</span>
                        </div>""", unsafe_allow_html=True)

                with lat_col:
                    st.markdown("##### ⏱️ Latency Breakdown")
                    timings = m.get("timings", {})
                    layer_names = {
                        "embedding_text": "Text Embedding",
                        "embedding_image": "Image Embedding",
                        "embedding_doc": "Doc Embedding",
                        "fusion": "Fusion",
                        "cross_modal": "Cross-Modal",
                        "multi_vector": "Multi-Vector",
                        "retrieval": "FAISS Retrieval",
                        "rag_generation": "RAG Generation",
                        "total": "Total",
                    }
                    for key, label in layer_names.items():
                        if key in timings:
                            ms = timings[key] * 1000
                            style = "font-weight:700;" if key == "total" else ""
                            st.markdown(f"""<div class="latency-row">
                                <span class="layer" style="{style}">{label}</span>
                                <span class="time" style="{style}">{ms:.2f} ms</span>
                            </div>""", unsafe_allow_html=True)

            # Show uploaded image if present
            if uploaded_image:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", width=300)

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#6c757d;">
            <div style="font-size:4rem; margin-bottom:1rem;">🌿</div>
            <h3 style="color:#0f5132;">Enter symptoms to get a diagnosis</h3>
            <p>Describe what you see on your crops, upload a leaf image,<br/>
            or try one of the example queries on the left.</p>
            <div style="margin-top:2rem; font-size:0.85rem;">
                <strong>Supported inputs:</strong> Text descriptions • Leaf/crop images • Lab reports
            </div>
        </div>
        """, unsafe_allow_html=True)
