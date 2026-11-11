"""
Model Performance & Employability ML Page
=========================================
Benchmark matrix of the 6 trained employability classification models,
evaluation metrics, and live interactive soft-skill employability predictor.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components.ui import page_header, section_header, status_pill
from frontend.services.api_client import client


def render_model_performance(nav_to_fn):
    page_header(
        kicker="Machine Learning Analytics",
        title="Model Performance & Employability AI",
        description="Review rigorous evaluation metrics for all 6 trained supervised models and test real-time candidate employability classification.",
    )

    t_bench, t_pred = st.tabs(["📊 Model Evaluation Matrix", "🎯 Interactive Employability Predictor"])

    # ── TAB 1: MODEL BENCHMARK MATRIX ────────────────────────────────────────
    with t_bench:
        section_header("Supervised Classification Benchmarks", "Evaluated on held-out test split of 597 candidates (Train-only 5-fold CV)")

        meta = client.get_model_meta()
        models_data = meta.get("models", {})

        benchmark_rows = [
            {
                "Model": "Gradient Boosting (Production)",
                "Val Accuracy": "76.38%",
                "Val Macro F1": "0.7470",
                "Val ROC-AUC": "0.8347",
                "Test Accuracy": "76.88%",
                "Test Precision": "74.19%",
                "Test Recall": "92.20%",
                "Test Binary F1": "0.8222",
                "Test Macro F1": "0.7460",
                "Test ROC-AUC": "0.8248",
            },
            {
                "Model": "Support Vector Machine (RBF)",
                "Val Accuracy": "73.03%",
                "Val Macro F1": "0.7215",
                "Val ROC-AUC": "0.7700",
                "Test Accuracy": "75.88%",
                "Test Precision": "74.40%",
                "Test Recall": "89.02%",
                "Test Binary F1": "0.8105",
                "Test Macro F1": "0.7394",
                "Test ROC-AUC": "0.7926",
            },
            {
                "Model": "Random Forest",
                "Val Accuracy": "70.52%",
                "Val Macro F1": "0.7034",
                "Val ROC-AUC": "0.8075",
                "Test Accuracy": "74.20%",
                "Test Precision": "76.97%",
                "Test Recall": "79.19%",
                "Test Binary F1": "0.7806",
                "Test Macro F1": "0.7338",
                "Test ROC-AUC": "0.8132",
            },
            {
                "Model": "K-Nearest Neighbors (KNN)",
                "Val Accuracy": "69.18%",
                "Val Macro F1": "0.6841",
                "Val ROC-AUC": "0.7898",
                "Test Accuracy": "71.19%",
                "Test Precision": "75.74%",
                "Test Recall": "73.99%",
                "Test Binary F1": "0.7485",
                "Test Macro F1": "0.7056",
                "Test ROC-AUC": "0.7849",
            },
            {
                "Model": "Decision Tree",
                "Val Accuracy": "64.32%",
                "Val Macro F1": "0.6431",
                "Val ROC-AUC": "0.7069",
                "Test Accuracy": "69.85%",
                "Test Precision": "71.39%",
                "Test Recall": "80.06%",
                "Test Binary F1": "0.7548",
                "Test Macro F1": "0.6817",
                "Test ROC-AUC": "0.7766",
            },
            {
                "Model": "Logistic Regression",
                "Val Accuracy": "57.96%",
                "Val Macro F1": "0.5764",
                "Val ROC-AUC": "0.6001",
                "Test Accuracy": "59.30%",
                "Test Precision": "66.04%",
                "Test Recall": "61.27%",
                "Test Binary F1": "0.6357",
                "Test Macro F1": "0.5873",
                "Test ROC-AUC": "0.6316",
            },
        ]

        df_bench = pd.DataFrame(benchmark_rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.write("")
        c1, c2 = st.columns(2, gap="large")
        with c1, st.container(border=True):
            section_header("Test Accuracy Comparison (%)")
            acc_data = {"Gradient Boosting": 76.88, "SVM": 75.88, "Random Forest": 74.20, "KNN": 71.19, "Decision Tree": 69.85, "Logistic Regression": 59.30}
            st.bar_chart(pd.DataFrame(list(acc_data.items()), columns=["Model", "Accuracy (%)"]).set_index("Model"), height=240)
        with c2, st.container(border=True):
            section_header("Test ROC-AUC Comparison")
            auc_data = {"Gradient Boosting": 0.8248, "SVM": 0.7926, "Random Forest": 0.8132, "KNN": 0.7849, "Decision Tree": 0.7766, "Logistic Regression": 0.6316}
            st.bar_chart(pd.DataFrame(list(auc_data.items()), columns=["Model", "ROC-AUC"]).set_index("Model"), height=240)

    # ── TAB 2: INTERACTIVE EMPLOYABILITY PREDICTOR ────────────────────────────
    with t_pred:
        section_header("Candidate Soft-Skill Employability Inference", "Live inference using the trained Gradient Boosting classification pipeline")
        
        st.markdown(
            """
            Input the candidate's evaluated soft skill ratings (scale: 1 to 5).
            The pipeline automatically engineers interaction terms `soft_skill_mean` and `soft_skill_min`
            before feeding features to `gradient_boosting_employability.joblib`.
            """
        )

        with st.form("form_employability_predictor"):
            c1, c2, c3 = st.columns(3)
            with c1:
                manner = st.slider("Manner of Speaking (1–5)", 1, 5, 4, help="Confidence, tone, and professional composure")
                mental = st.slider("Mental Alertness (1–5)", 1, 5, 4, help="Speed and responsiveness under questioning")
            with c2:
                conf = st.slider("Self-Confidence (1–5)", 1, 5, 4, help="Poise and clarity during answer delivery")
                ideas = st.slider("Ability to Present Ideas (1–5)", 1, 5, 4, help="Logical coherence and technical storytelling")
            with c3:
                comm = st.slider("Communication Skills (1–5)", 1, 5, 4, help="Grammar, vocabulary, and articulated phrasing")

            predict_submitted = st.form_submit_button("Run Employability Inference", type="primary", use_container_width=True)

        if predict_submitted:
            scores_dict = {
                "manner_of_speaking": manner,
                "mental_alertness": mental,
                "self_confidence": conf,
                "ability_to_present_ideas": ideas,
                "communication_skills": comm,
            }
            with st.spinner("Running inference pipeline…"):
                ok, pred_res = client.predict_employability(scores_dict)
                if ok:
                    pred_class = pred_res.get("prediction", "Unknown")
                    prob_emp = pred_res.get("probability_employable", 0.5)

                    st.write("")
                    with st.container(border=True):
                        p_col1, p_col2 = st.columns([1.2, 1.8], gap="large")
                        with p_col1:
                            st.markdown('<div class="page-eyebrow">PREDICTION OUTCOME</div>', unsafe_allow_html=True)
                            if pred_class == "Employable":
                                st.success(f"### Result: **{pred_class}**")
                            else:
                                st.warning(f"### Result: **{pred_class}**")
                            st.caption(f"Estimated Probability: **{prob_emp:.1%}**")
                            st.progress(prob_emp)
                        with p_col2:
                            st.markdown('<div class="page-eyebrow">ENGINEERED FEATURE SUMMARY</div>', unsafe_allow_html=True)
                            raw_vals = [manner, mental, conf, ideas, comm]
                            m_val = sum(raw_vals) / len(raw_vals)
                            min_val = min(raw_vals)
                            st.markdown(f"- **Soft Skill Mean:** `{m_val:.2f} / 5.0`")
                            st.markdown(f"- **Soft Skill Minimum:** `{min_val} / 5`")
                            st.markdown(f"- **Primary Driving Feature:** `Ability to Present Ideas` (Highest feature importance in model)")
                            st.caption("Disclaimer: This prediction is generated by a supervised ML model trained on academic datasets for interview practice. It should not be used as an automated hiring decision.")
                else:
                    st.error(f"Inference failed: {pred_res}")

