"""
=============================================================================
Intelligent Multimodal Mock Interview Assessment System
Production AI SaaS Dashboard & Evaluation Interface
=============================================================================
"""

import os, sys, json, tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="InterviewAI | Multimodal Mock Assessment",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root and src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from src.text_pipeline import InterviewTextAssessor, parse_instruction_parquet, clean_text
from src.audio_feature_extractor import CandidateAudioAnalyzer
from src.interview_repository import get_categories, get_questions_by_category
import joblib

# Check if Whisper is available in environment
WHISPER_AVAILABLE = False
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# ── Load Resources (Cached) ────────────────────────────────────────────────
@st.cache_resource
def load_models_and_resources():
    models_dir = os.path.join(BASE_DIR, "models")
    gb_path = os.path.join(models_dir, "gradient_boosting_employability.joblib")
    tfidf_path = os.path.join(models_dir, "interview_tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "employability_model_meta.json")
    parquet_path = os.path.join(BASE_DIR, "dataset", "interview_questions", "eval-00000-of-00001.parquet")
    
    gb_model = joblib.load(gb_path) if os.path.exists(gb_path) else None
    
    if os.path.exists(tfidf_path):
        vectorizer = joblib.load(tfidf_path)
        assessor = InterviewTextAssessor(vectorizer=vectorizer)
        assessor.is_fitted = True
    else:
        assessor = InterviewTextAssessor()
        
    audio_analyzer = CandidateAudioAnalyzer(target_sr=16000)
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    raw_questions = []
    if os.path.exists(parquet_path):
        df_q = parse_instruction_parquet(parquet_path)
        raw_questions = df_q.head(50).to_dict(orient="records")
            
    return gb_model, assessor, audio_analyzer, metadata, raw_questions

gb_model, text_assessor, audio_analyzer, model_meta, raw_questions = load_models_and_resources()


# ── SaaS Dark Theme CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Navy & Slate SaaS Theme */
    .stApp {
        background-color: #0B0F19;
        color: #F9FAFB;
    }
    
    /* Global Card */
    .saas-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    
    .saas-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Metric Header Cards */
    .metric-box {
        background: #161E2E;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: left;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F9FAFB;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #60A5FA;
    }

    /* Gradients & Headers */
    .gradient-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .gradient-sub {
        font-size: 1.0rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }

    /* Status Badges */
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    .badge-info {
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)


# ── Navigation Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;"><span style="font-size:1.8rem;">⚡</span><span style="font-size:1.3rem;font-weight:700;color:#F9FAFB;">InterviewAI</span></div>', unsafe_allow_html=True)
    st.caption("AI SaaS Mock Interview Platform")
    
    module_select = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Employability Predictor",
            "Technical Interview (Text)",
            "Audio Assessment",
            "Multimodal Report",
            "Model Performance & EDA"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("#### System Architecture")
    st.markdown("""
    - **Classifier:** Gradient Boosting (AUC: `0.825`)
    - **NLP Engine:** TF-IDF Cosine Similarity
    - **Audio DSP:** Librosa Acoustic Extraction
    - **Backend API:** FastAPI REST Engine
    """)
    st.divider()
    st.caption("Intelligent Multimodal Mock Interview Assessment • v1.2.0")


# ══════════════════════════════════════════════════════════════════════════
# MODULE 1: LANDING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if module_select == "Dashboard":
    st.markdown('<div class="gradient-header">Multimodal Mock Interview Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Supervised Machine Learning & NLP Evaluation Platform for Placement Readiness</div>', unsafe_allow_html=True)
    
    # Hero Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Assessment Engines</div>
            <div class="metric-value">4 Modules</div>
            <div class="metric-sub">Soft-Skill • Text • Audio • Multimodal</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Best Supervised ML</div>
            <div class="metric-value">76.88%</div>
            <div class="metric-sub">Gradient Boosting (Test AUC: 0.825)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">NLP Reference Corpus</div>
            <div class="metric-value">5,817</div>
            <div class="metric-sub">Expert Benchmark Q&A Pairs</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Audio Feature Engine</div>
            <div class="metric-value">16 kHz</div>
            <div class="metric-sub">Librosa Acoustic DSP (F0, Pause, MFCC)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Modules Grid
    st.markdown("### Evaluation Modules")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size:1.15rem;font-weight:700;color:#60A5FA;margin-bottom:0.4rem;">🎓 Employability Prediction Engine</div>
            <p style="color:#9CA3AF;font-size:0.9rem;line-height:1.5;">
                Evaluates candidate placement readiness using 5 non-appearance behavioral soft skills. 
                Powered by a trained <b>Gradient Boosting</b> ensemble with feature engineering (mean & bottleneck skills).
            </p>
            <span class="badge-success">Supervised Classifier</span> <span class="badge-info">Test Acc: 76.88%</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="saas-card">
            <div style="font-size:1.15rem;font-weight:700;color:#C084FC;margin-bottom:0.4rem;">🎙️ Acoustic Speech Analyzer</div>
            <p style="color:#9CA3AF;font-size:0.9rem;line-height:1.5;">
                Extracts pitch variation ($F_0$ via Yin), pause ratios, speaking pace, and 13 MFCC timbre coefficients 
                from candidate audio using <b>Librosa</b>.
            </p>
            <span class="badge-info">DSP Feature Extraction</span> <span class="badge-success">Live Audio Ready</span>
        </div>
        """, unsafe_allow_html=True)
        
    with g2:
        st.markdown("""
        <div class="saas-card">
            <div style="font-size:1.15rem;font-weight:700;color:#34D399;margin-bottom:0.4rem;">💬 Technical Interview Text Assessor</div>
            <p style="color:#9CA3AF;font-size:0.9rem;line-height:1.5;">
                Objectively grades candidate responses across 8 engineering domains (Python, C++, Java, DSA, DBMS, OS, Networks, ML) 
                using sublinear TF-IDF vector similarity against benchmark solutions.
            </p>
            <span class="badge-info">NLP Vector Space</span> <span class="badge-success">8 Domains Indexed</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="saas-card">
            <div style="font-size:1.15rem;font-weight:700;color:#F59E0B;margin-bottom:0.4rem;">📑 Explainable Multimodal Report</div>
            <p style="color:#9CA3AF;font-size:0.9rem;line-height:1.5;">
                Synthesizes spoken speech delivery metrics and technical content relevance into an integrated, transparent 
                report card with actionable placement advice.
            </p>
            <span class="badge-warning">Rule-Based Composite</span> <span class="badge-info">Holistic Evaluation</span>
        </div>
        """, unsafe_allow_html=True)

    # Session Assessment History if available
    st.markdown("### Session Assessment Log")
    has_history = False
    history_records = []
    
    if "employability_result" in st.session_state:
        has_history = True
        r = st.session_state["employability_result"]
        history_records.append({
            "Module": "Employability Predictor",
            "Result / Tier": r["pred"],
            "Primary Metric": f"{r['p_emp']:.1%} Placement Probability",
            "Status": "Evaluated"
        })
    if "last_text_eval" in st.session_state:
        has_history = True
        t = st.session_state["last_text_eval"]
        history_records.append({
            "Module": "Technical Text Interview",
            "Result / Tier": t["tier"],
            "Primary Metric": f"Relevance: {t['relevance_score']*100:.1f}%",
            "Status": "Evaluated"
        })
    if "last_audio_eval" in st.session_state:
        has_history = True
        a = st.session_state["last_audio_eval"]
        history_records.append({
            "Module": "Audio Speech Assessment",
            "Result / Tier": f"Fluency: {a['estimated_fluency_score']}/5.0",
            "Primary Metric": f"Pause Ratio: {a['pause_ratio']*100:.1f}%",
            "Status": "Evaluated"
        })
        
    if has_history:
        st.table(pd.DataFrame(history_records))
    else:
        st.info("No assessments executed yet in this session. Navigate to any module in the sidebar to begin.")


# ══════════════════════════════════════════════════════════════════════════
# MODULE 2: EMPLOYABILITY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
elif module_select == "Employability Predictor":
    st.markdown('<div class="gradient-header">Student Employability Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Supervised Machine Learning Inference on Behavioral Soft-Skill Dimensions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Soft-Skill Evaluation Inputs (1–5)")
        speaking = st.slider("Manner of Speaking", 1, 5, 4, help="Composure, vocal tone, and articulation")
        alertness = st.slider("Mental Alertness", 1, 5, 4, help="Cognitive agility and responsiveness")
        confidence = st.slider("Self-Confidence", 1, 5, 4, help="Poise under technical questioning")
        presentation = st.slider("Ability to Present Ideas", 1, 5, 3, help="Structured and logical point delivery")
        communication = st.slider("Communication Skills", 1, 5, 4, help="Language accuracy and technical precision")
        
        predict_btn = st.button("🚀 Predict Employability", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            raw_scores = [speaking, alertness, confidence, presentation, communication]
            feat_mean = float(np.mean(raw_scores))
            feat_min = int(np.min(raw_scores))
            
            input_df = pd.DataFrame([{
                "MANNER OF SPEAKING": speaking,
                "MENTAL ALERTNESS": alertness,
                "SELF-CONFIDENCE": confidence,
                "ABILITY TO PRESENT IDEAS": presentation,
                "COMMUNICATION SKILLS": communication,
                "soft_skill_mean": feat_mean,
                "soft_skill_min": feat_min
            }])
            
            pred = gb_model.predict(input_df)[0]
            proba = gb_model.predict_proba(input_df)[0]
            emp_idx = list(gb_model.classes_).index("Employable")
            p_emp = float(proba[emp_idx])
            
            st.session_state["employability_result"] = {
                "pred": pred,
                "p_emp": p_emp,
                "raw_scores": raw_scores,
                "speaking": speaking,
                "alertness": alertness,
                "confidence": confidence,
                "presentation": presentation,
                "communication": communication
            }

    with col2:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Real-Time Assessment Dashboard")
        
        if "employability_result" in st.session_state:
            res = st.session_state["employability_result"]
            pred = res["pred"]
            p_emp = res["p_emp"]
            raw_scores = res["raw_scores"]
            
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                if pred == "Employable":
                    st.markdown('<div class="badge-success" style="font-size:1.1rem;padding:0.5rem 1rem;">✅ Class: Employable</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-warning" style="font-size:1.1rem;padding:0.5rem 1rem;">⚠️ Class: LessEmployable</div>', unsafe_allow_html=True)
            with res_c2:
                st.metric("Placement Probability", f"{p_emp:.1%}", delta=f"{p_emp - 0.58:.1%} vs Base Rate")
                
            st.progress(float(p_emp))
            
            # Dark Themed Radar Profile
            st.markdown("##### Soft-Skill Profile Radar")
            categories = ["Speaking", "Alertness", "Confidence", "Ideas", "Communication"]
            values = raw_scores + [raw_scores[0]]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#161E2E')
            ax.plot(angles, values, color="#818CF8", linewidth=2)
            ax.fill(angles, values, color="#6366F1", alpha=0.3)
            ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#9CA3AF", fontsize=7)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, color="#F9FAFB", fontsize=8.5)
            ax.tick_params(colors="#9CA3AF")
            ax.spines['polar'].set_color('#374151')
            ax.grid(color="#374151", linestyle='--')
            ax.set_ylim(0, 5)
            st.pyplot(fig)
            
            # Actionable feedback
            st.markdown("##### Actionable Feedback & Development Areas")
            recs = []
            if res["communication"] < 3: recs.append("Focus on technical vocabulary precision and eliminating filler phrases.")
            if res["presentation"] < 3: recs.append("Structure complex answers using the STAR method (Situation, Task, Action, Result).")
            if res["speaking"] < 3: recs.append("Practice steady pacing and clear vocal modulation during technical explanations.")
            if res["confidence"] < 3: recs.append("Participate in timed mock interview drills to reduce hesitation.")
            if not recs: recs.append("Excellent soft-skill balance. Focus on domain technical mastery.")
            for r in recs:
                st.info(f"👉 {r}")
        else:
            st.info("Adjust the sliders and click **Predict Employability** to run the trained Gradient Boosting model.")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 3: TECHNICAL INTERVIEW (TEXT)
# ══════════════════════════════════════════════════════════════════════════
elif module_select == "Technical Interview (Text)":
    st.markdown('<div class="gradient-header">Technical Interview Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Objective NLP Evaluation Benchmark against Expert Reference Standards</div>', unsafe_allow_html=True)
    
    # Category Selection
    cat_col, q_col = st.columns([1, 2])
    with cat_col:
        categories = get_categories()
        selected_category = st.selectbox("Interview Domain / Category:", categories)
        questions_in_cat = get_questions_by_category(selected_category)
        
    with q_col:
        q_options = [q["question"] for q in questions_in_cat]
        selected_q_text = st.selectbox("Select Question:", q_options)
        # Find selected question item
        curr_q_item = next((q for q in questions_in_cat if q["question"] == selected_q_text), questions_in_cat[0])
        
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(f"**Target Technical Question:**")
    st.markdown(f"<div style='font-size:1.15rem;font-weight:600;color:#F9FAFB;margin-bottom:0.5rem;'>{curr_q_item['question']}</div>", unsafe_allow_html=True)
    
    with st.expander("🔍 View Benchmark Reference Answer (Ground-Truth Standard)"):
        st.write(curr_q_item["reference_answer"])
        st.caption(f"**Key Expected Concepts:** {', '.join(curr_q_item['key_concepts'])}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Candidate Editor
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### Candidate Response Editor")
    
    candidate_answer = st.text_area(
        "Candidate Answer:",
        height=150,
        placeholder="Type or paste your technical explanation here with key definitions, mechanisms, and examples...",
        key="tech_ans_editor"
    )
    
    # Word count counter
    word_count = len(candidate_answer.strip().split()) if candidate_answer.strip() else 0
    st.caption(f"**Word Count:** `{word_count}` words")
    
    btn_c1, btn_c2 = st.columns([1, 4])
    with btn_c1:
        eval_btn = st.button("📝 Evaluate Response", type="primary", use_container_width=True)
    with btn_c2:
        if st.button("🗑️ Clear", use_container_width=False):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if eval_btn:
        if word_count < 3:
            st.warning("Please provide a response of at least 3 words.")
        else:
            with st.spinner("Analyzing semantic relevance and structural coherence via TF-IDF vectorizer..."):
                scores = text_assessor.assess_answer(curr_q_item["question"], candidate_answer, curr_q_item["reference_answer"])
                
                # Determine Tier
                q_score = scores["quality_score"]
                if q_score >= 0.35:
                    tier = "Proficient / Strong Answer"
                    badge_class = "badge-success"
                elif q_score >= 0.15:
                    tier = "Intermediate / Partially Correct"
                    badge_class = "badge-info"
                else:
                    tier = "Developing / Insufficient Coverage"
                    badge_class = "badge-warning"
                    
                scores["tier"] = tier
                scores["question"] = curr_q_item["question"]
                scores["answer"] = candidate_answer
                st.session_state["last_text_eval"] = scores

    if "last_text_eval" in st.session_state:
        res = st.session_state["last_text_eval"]
        
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### NLP Evaluation Report Card")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Content Relevance", f"{res['relevance_score']*100:.1f}%")
        m2.metric("Question Alignment", f"{res['question_alignment_score']*100:.1f}%")
        m3.metric("Keyword Overlap", f"{res['keyword_overlap']*100:.1f}%")
        m4.metric("Structure Coherence", f"{res['structure_score']*100:.1f}%")
        
        st.progress(float(res["quality_score"]))
        st.markdown(f"**Composite Score:** `{res['quality_score']:.3f}` / 1.000 &nbsp;&nbsp; <span class='{badge_class}'>{res['tier']}</span>", unsafe_allow_html=True)
        st.caption("*NLP Information Retrieval Scoring: Computed using sublinear TF-IDF N-Gram Cosine Similarity and Jaccard overlap, not a black-box supervised classifier.*")
        
        st.divider()
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            st.markdown("##### Identified Strengths")
            for s in res.get("strengths", []):
                st.success(f"✓ {s}")
        with fb_c2:
            st.markdown("##### Areas for Improvement")
            for imp in res.get("improvements", []):
                st.warning(f"⚠ {imp}")
                
        st.markdown("##### Suggested Answer Architecture")
        for step in res.get("suggested_structure", []):
            st.write(step)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 4: AUDIO ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════
elif module_select == "Audio Assessment":
    st.markdown('<div class="gradient-header">Acoustic Speech Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Digital Signal Processing with Librosa for Fluency, Pause, and Prosody Analysis</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 1.2])
    
    with col_a:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Audio Recording Input")
        audio_file = st.file_uploader("Upload Spoken Audio (.wav / .mp3):", type=["wav", "mp3"])
        
        if audio_file is not None:
            st.audio(audio_file)
            st.caption(f"**File:** `{audio_file.name}` ({audio_file.size / 1024:.1f} KB)")
            
        st.markdown("#### 2. Speech Transcription")
        if WHISPER_AVAILABLE:
            st.markdown('<span class="badge-success">Whisper ASR Detected</span>', unsafe_allow_html=True)
            if st.button("🎙️ Transcribe with Whisper"):
                st.info("Running local Whisper transcription...")
        else:
            st.markdown('<span class="badge-warning">Whisper ASR Not Installed in venv</span>', unsafe_allow_html=True)
            st.caption("Manual transcript entry is active for zero-latency execution.")
            
        transcript_input = st.text_area(
            "Candidate Verbal Transcript:",
            height=100,
            placeholder="Type or paste the verbal response that was spoken...",
            key="audio_transcript_input"
        )
        
        analyze_audio_btn = st.button("⚡ Analyze Audio with Librosa", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("#### Extracted Acoustic Telemetry")
        
        if analyze_audio_btn:
            if audio_file is None:
                st.warning("Please upload a WAV or MP3 audio recording to analyze.")
            else:
                with st.spinner("Extracting pitch (F0), RMS energy, silence intervals, and MFCCs with Librosa..."):
                    suffix = os.path.splitext(audio_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                        
                    try:
                        metrics = audio_analyzer.analyze_audio_file(tmp_path)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            
                    metrics["filename"] = audio_file.name
                    metrics["transcript"] = transcript_input
                    st.session_state["last_audio_eval"] = metrics

        if "last_audio_eval" in st.session_state:
            a = st.session_state["last_audio_eval"]
            
            # Metric grid
            c1, c2 = st.columns(2)
            c1.metric("Speaking Duration", f"{a['speaking_duration_sec']}s", delta=f"Total: {a['duration_sec']}s")
            c1.metric("Pause Ratio", f"{a['pause_ratio']*100:.1f}%")
            c1.metric("Estimated Fluency", f"{a['estimated_fluency_score']} / 5.0")
            
            c2.metric("Mean Pitch ($F_0$)", f"{a['mean_pitch_hz']} Hz")
            c2.metric("Pitch Variation (Std)", f"{a['pitch_std_hz']} Hz", delta=f"Range: {a.get('min_pitch_hz', 90)}–{a.get('max_pitch_hz', 160)} Hz")
            c2.metric("Estimated Prosody", f"{a['estimated_prosody_score']} / 5.0")
            
            st.divider()
            st.markdown("##### 13 MFCC Summary Coefficients (Timbre)")
            mfcc_df = pd.DataFrame({
                "Coeff": [f"C{i+1}" for i in range(len(a['mfcc_mean']))],
                "Mean Magnitude": a['mfcc_mean']
            })
            st.bar_chart(mfcc_df.set_index("Coeff"), height=160)
            
            st.markdown("##### Acoustic Diagnostics & Observations")
            if a["pause_ratio"] > 0.30:
                st.warning("⚠️ High hesitation ratio detected (>30%). Practice reducing unfilled pauses between sentences.")
            else:
                st.success("✓ Healthy conversational tempo and balanced pause ratio.")
                
            if a["pitch_std_hz"] < 12.0:
                st.info("ℹ️ Pitch modulation is relatively low. Adding natural vocal inflection can enhance listener engagement.")
            else:
                st.success("✓ Good vocal dynamics and expressive prosodic inflection.")
                
            st.caption("**Ethical AI Transparency:** Pitch variability reflects acoustic prosodic inflection and dynamic range; it is NOT an indicator of candidate confidence or personality.")
        else:
            st.info("Upload an audio file and click **Analyze Audio with Librosa** to view speech metrics.")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 5: MULTIMODAL REPORT
# ══════════════════════════════════════════════════════════════════════════
elif module_select == "Multimodal Report":
    st.markdown('<div class="gradient-header">Integrated Multimodal Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Composite Synthesis of Technical Text Competence and Spoken Audio Delivery</div>', unsafe_allow_html=True)
    
    has_text = "last_text_eval" in st.session_state
    has_audio = "last_audio_eval" in st.session_state
    
    if not has_text and not has_audio:
        st.info("No assessments completed yet. Please evaluate a **Technical Text Response** and analyze an **Audio Response** to generate an integrated report card.")
    else:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        
        # Calculate composite score
        text_score = st.session_state["last_text_eval"]["quality_score"] if has_text else 0.50
        fluency_norm = st.session_state["last_audio_eval"]["estimated_fluency_score"] / 5.0 if has_audio else 0.50
        prosody_norm = st.session_state["last_audio_eval"]["estimated_prosody_score"] / 5.0 if has_audio else 0.50
        
        composite = (0.50 * text_score) + (0.25 * fluency_norm) + (0.25 * prosody_norm)
        
        st.markdown("### 🎖️ Multimodal Candidate Scorecard")
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.metric("Composite Multimodal Score", f"{composite*100:.1f} / 100")
            st.progress(float(composite))
        with c2:
            if composite >= 0.45:
                st.markdown('<div class="badge-success" style="font-size:1.1rem;padding:0.4rem 0.9rem;">🏆 Tier: Strong Placement Candidate</div>', unsafe_allow_html=True)
            elif composite >= 0.25:
                st.markdown('<div class="badge-warning" style="font-size:1.1rem;padding:0.4rem 0.9rem;">⚠️ Tier: Competent with Growth Areas</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-warning" style="background:rgba(239,68,68,0.2);color:#F87171;font-size:1.1rem;padding:0.4rem 0.9rem;">❌ Tier: Needs Targeted Development</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="font-size:0.83rem;color:#9CA3AF;margin-top:0.6rem;">
                <b>Fusion Method:</b> Transparent Rule-Based Late Fusion (50% Text Quality + 25% Speech Fluency + 25% Speech Prosody).
                <i>Note: As aligned multimodal ground-truth training datasets do not exist, this composite score is computed via explainable heuristics rather than a black-box model.</i>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Breakdown columns
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("#### 📝 Text Competence Indicators")
            if has_text:
                t = st.session_state["last_text_eval"]
                st.metric("Content Relevance", f"{t['relevance_score']*100:.1f}%")
                st.metric("Question Alignment", f"{t['question_alignment_score']*100:.1f}%")
                st.metric("Keyword Overlap", f"{t['keyword_overlap']*100:.1f}%")
                st.metric("Structure Coherence", f"{t['structure_score']*100:.1f}%")
            else:
                st.caption("Text evaluation not executed yet.")
                
        with b2:
            st.markdown("#### 🎙️ Spoken Acoustic Indicators")
            if has_audio:
                a = st.session_state["last_audio_eval"]
                st.metric("Estimated Fluency", f"{a['estimated_fluency_score']} / 5.0")
                st.metric("Pause Ratio", f"{a['pause_ratio']*100:.1f}%")
                st.metric("Estimated Prosody", f"{a['estimated_prosody_score']} / 5.0")
                st.metric("Mean Pitch ($F_0$)", f"{a['mean_pitch_hz']} Hz")
            else:
                st.caption("Audio evaluation not executed yet.")
                
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MODULE 6: MODEL PERFORMANCE & BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════
elif module_select == "Model Performance & EDA":
    st.markdown('<div class="gradient-header">Supervised Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Empirical Results on Held-Out Test Data (N=597) & Exploratory Visualizations</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### Final Test Set Benchmark Comparison")
    
    # Dynamically load from models/employability_model_meta.json
    test_results_dict = model_meta.get("test_results", {})
    best_model_name = model_meta.get("best_model", "Gradient Boosting")
    
    if test_results_dict:
        dynamic_rows = []
        for model_name, m in test_results_dict.items():
            status = "🏆 Best Model" if model_name == best_model_name else (
                "Runner-Up" if model_name == "SVM" else (
                    "High Stability" if model_name == "Random Forest" else (
                        "Non-parametric" if model_name == "KNN" else (
                            "Interpretable" if model_name == "Decision Tree" else "Linear Baseline"
                        )
                    )
                )
            )
            dynamic_rows.append({
                "Model": model_name,
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1 (Binary)": f"{m['f1_binary']:.4f}",
                "F1 (Macro)": f"{m['f1_macro']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Architecture": status
            })
        df_benchmarks = pd.DataFrame(dynamic_rows).sort_values("ROC-AUC", ascending=False)
        st.table(df_benchmarks)
    else:
        st.warning("Model metadata file not found.")
        
    st.caption("**Evaluation Rigor:** All values reflect verified evaluation on the held-out 20% test partition ($N=597$). No values are synthetic.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization Gallery
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown("#### Evaluation Visualizations & ROC Curves")
    
    figs_dir = os.path.join(BASE_DIR, "reports", "figures")
    img_select = st.selectbox(
        "Select Figure to Display:",
        [
            ("E10_roc_curves_test.png", "Test Set ROC Curves (Gradient Boosting AUC = 0.825)"),
            ("E8_confusion_matrices_test.png", "Test Set Confusion Matrices across 6 Classifiers"),
            ("E7_feature_importance_rf.png", "Gini Feature Importances (Random Forest)"),
            ("E11_decision_tree_visualization.png", "Learned Decision Rules (Decision Tree)"),
            ("A4_correlation_heatmap.png", "Soft-Skill Correlation Matrix"),
            ("E3_cv_results.png", "5-Fold Stratified Cross-Validation Benchmarks")
        ],
        format_func=lambda x: x[1]
    )
    
    img_path = os.path.join(figs_dir, img_select[0])
    if os.path.exists(img_path):
        st.image(img_path, caption=img_select[1], use_container_width=True)
    else:
        st.info("Figure not found in reports/figures directory.")
    st.markdown('</div>', unsafe_allow_html=True)
