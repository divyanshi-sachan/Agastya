"""Streamlit demo app for the Phase 3 hybrid pipeline.

Renders the Smart Report (risk + clause explanation + conflict reasoning +
top-3 risk factors) and surfaces calibration warnings (distribution shift)
in red.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO

import pandas as pd
import pdfplumber
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase3.bayesian.bootstrap import ensure_seed_model
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline
from src.phase3.ocr.extractor import extract_text
from src.phase3.smart_report import format_text_report

st.set_page_config(page_title="Agastya", page_icon="⚖", layout="wide")

st.markdown(
    """
    <style>
    .stButton>button { width: 100%; text-align: left; border-radius: 5px; margin-bottom: 5px; }
    .active-clause { border: 2px solid #007bff !important; background-color: #e7f1ff !important; }
    .risk-indicator { float: right; font-size: 10px; padding: 2px 6px; border-radius: 10px; color: white; }
    .risk-high { background-color: #dc3545; }
    .risk-low { background-color: #28a745; }
    .calibration-warning { background-color: #ffe4e6; color: #b00020; padding: 12px; border-radius: 6px; border: 1px solid #b00020; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agastya - Contract Risk Analyzer")
st.caption("Hybrid AI: LoRA Legal-BERT + Bayesian Network with bidirectional feedback")


@st.cache_resource(show_spinner="Loading hybrid pipeline...")
def load_pipeline() -> AgastyaHybridPipeline:
    model_path = ensure_seed_model("results/phase3/bayesian_network.pkl")
    return AgastyaHybridPipeline(
        bn_model_path=model_path,
        bert_checkpoint_path="results/phase2/models/legal_bert_phase2.pt",
        label_map_path="results/phase2/label2id.json",
        adapter_path="results/phase2/models/legal_bert_lora_adapter",
    )


@st.cache_data(show_spinner="Analyzing contract...")
def run_prediction(text: str) -> dict:
    pipeline = load_pipeline()
    return pipeline.predict(text)


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "selected_index" not in st.session_state:
    st.session_state.selected_index = 0
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

uploaded = st.file_uploader("Upload contract", type=["pdf"])

if uploaded is not None:
    if st.session_state.pdf_bytes != uploaded.getvalue():
        st.session_state.pdf_bytes = uploaded.getvalue()
        text = extract_text(uploaded)
        st.session_state.analysis_result = run_prediction(text)
        st.session_state.selected_index = 0


def _render_calibration_warning(warning: str | None) -> None:
    if not warning:
        return
    st.markdown(
        f"<div class='calibration-warning'>⚠ {warning}</div>",
        unsafe_allow_html=True,
    )


def _render_smart_report(report: dict) -> None:
    rs = report.get("risk_score", {})
    st.subheader("📊 Risk Score")
    cols = st.columns(4)
    cols[0].metric("Level", rs.get("level", "?"))
    cols[1].metric("Confidence", f"{rs.get('confidence', 0.0):.1%}")
    cols[2].metric("Uncertainty (entropy)", f"{rs.get('uncertainty', 0.0):.3f}")
    high_p = rs.get("probabilities", {}).get("High", 0.0)
    cols[3].metric("High-risk prob", f"{high_p:.1%}")

    st.subheader("⚠ Top Risk Factors")
    for i, factor in enumerate(report.get("top_risk_factors", []), start=1):
        with st.container(border=True):
            st.markdown(
                f"**{i}. {factor.get('label')}** "
                f"— importance `{factor.get('importance', 0.0):.3f}` "
                f"(posterior `{factor.get('posterior_risky', 0.0):.2f}`, "
                f"prior `{factor.get('prior_risky', 0.0):.2f}`)"
            )
            st.caption(factor.get("reason", ""))

    st.subheader("🧩 Clause Explanation")
    ce = report.get("clause_explanation", {})
    agg = ce.get("aggregated_scores", {})
    counts = ce.get("clause_count", {})
    thresholds = ce.get("thresholds", {})
    absence = ce.get("absence_penalty", {})
    rows = []
    for node, score in agg.items():
        rows.append(
            {
                "Node": node,
                "Aggregated Score": round(score, 3),
                "Threshold": round(float(thresholds.get(node, 0.0)), 3),
                "Hard Evidence": ce.get("hard_evidence", {}).get(node, "?"),
                "Clause Count": counts.get(node, 0),
                "Absence Penalty": round(float(absence.get(node, 0.0)), 3),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("🔁 Conflict Reasoning")
    cr = report.get("conflict_reasoning", {})
    cols = st.columns(3)
    cols[0].metric("Raw conflict", f"{cr.get('raw_conflict_signal', 0.0):.3f}")
    calibrated = cr.get("calibrated_conflict_signal")
    cols[1].metric("Calibrated (ECDF)", f"{calibrated:.3f}" if calibrated is not None else "—")
    cols[2].metric("Iterations", len(cr.get("iteration_trace", [])))
    if cr.get("iteration_trace"):
        with st.expander("Feedback iteration trace"):
            st.json(cr["iteration_trace"])


if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    smart_report = result.get("smart_report") or {}

    _render_calibration_warning(result.get("calibration_warning"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Level", result["risk_level"])
    m2.metric("High Risk Prob", f'{result["risk_probabilities"].get("High", 0.0):.1%}')
    m3.metric("Conflict Signal", f'{result["conflict_signal"]:.2f}')
    st.divider()

    col_nav, col_viewer = st.columns([1, 2])

    with col_nav:
        st.subheader("📑 Identified Clauses")
        named_clauses = [(i, d) for i, d in enumerate(result["bert_details"]) if d.get("clause_type") != "Other"]
        other_clauses = [(i, d) for i, d in enumerate(result["bert_details"]) if d.get("clause_type") == "Other"]
        tab_named, tab_other = st.tabs([f"Named ({len(named_clauses)})", f"Other ({len(other_clauses)})"])
        with tab_named:
            with st.container(height=600):
                for i, detail in named_clauses:
                    ctype = detail.get("clause_type", "Unknown")
                    is_risky = ctype in ["Termination", "Liability"] and detail.get("confidence", 0) > 0.3
                    risk_label = "!" if is_risky else "✓"
                    btn_label = f"{risk_label} #{i+1}: {ctype}"
                    if st.button(btn_label, key=f"btn_named_{i}"):
                        st.session_state.selected_index = i
        with tab_other:
            with st.container(height=600):
                for i, detail in other_clauses:
                    btn_label = f"Clause #{i+1}: Segment"
                    if st.button(btn_label, key=f"btn_other_{i}"):
                        st.session_state.selected_index = i

    with col_viewer:
        idx = st.session_state.selected_index
        idx = max(0, min(idx, len(result["bert_details"]) - 1))
        active_detail = result["bert_details"][idx]
        clause_text = active_detail.get("clause_text", "")
        st.subheader(f"🔍 Viewing Clause #{idx+1}")
        with st.container(height=600):
            found_highlight = False
            if st.session_state.pdf_bytes:
                with pdfplumber.open(BytesIO(st.session_state.pdf_bytes)) as pdf:
                    search_query = clause_text[:100].strip()
                    for page_num, page in enumerate(pdf.pages):
                        matches = page.search(search_query)
                        if matches:
                            im = page.to_image(resolution=150)
                            for match in matches:
                                im.draw_rect(match, stroke="#ff0000", stroke_width=3, fill="#ff000033")
                            st.image(im.annotated, caption=f"Page {page_num + 1}", use_column_width=True)
                            found_highlight = True
                            break
            if not found_highlight:
                st.warning("Could not locate the exact text in the PDF viewer. Displaying raw text instead.")
                st.info(clause_text)

    st.divider()
    _render_smart_report(smart_report)

    with st.expander("Raw Smart Report (JSON)"):
        st.json(smart_report)
    with st.expander("Plaintext Smart Report"):
        st.code(format_text_report(smart_report))
