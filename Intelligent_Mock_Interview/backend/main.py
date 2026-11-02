"""
=============================================================================
FastAPI Backend Server
Intelligent Multimodal Mock Interview Assessment System
=============================================================================
"""

import os, sys, tempfile, shutil
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Add project root and src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from src.text_pipeline import InterviewTextAssessor, parse_instruction_parquet, clean_text
from src.audio_feature_extractor import CandidateAudioAnalyzer

# Initialize FastAPI application
app = FastAPI(
    title="Intelligent Multimodal Mock Interview Assessment API",
    description="Backend ML & NLP Inference Service for Employability Prediction, Text Assessment, and Audio Analysis.",
    version="1.0.0"
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Pre-Trained Models on Startup ─────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
GB_MODEL_PATH = os.path.join(MODELS_DIR, "gradient_boosting_employability.joblib")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "interview_tfidf_vectorizer.joblib")
QUESTIONS_PARQUET = os.path.join(BASE_DIR, "dataset", "interview_questions", "eval-00000-of-00001.parquet")

employability_model = None
text_assessor = None
audio_analyzer = None
cached_questions = []

try:
    if os.path.exists(GB_MODEL_PATH):
        employability_model = joblib.load(GB_MODEL_PATH)
        print(f"[OK] Loaded Gradient Boosting Model from {GB_MODEL_PATH}")
    else:
        print(f"[WARNING] Model not found at {GB_MODEL_PATH}")

    if os.path.exists(TFIDF_MODEL_PATH):
        vectorizer = joblib.load(TFIDF_MODEL_PATH)
        text_assessor = InterviewTextAssessor(vectorizer=vectorizer)
        text_assessor.is_fitted = True
        print(f"[OK] Loaded TF-IDF Vectorizer from {TFIDF_MODEL_PATH}")
    else:
        text_assessor = InterviewTextAssessor()
        print(f"[WARNING] TF-IDF Vectorizer not found, initialized empty")

    audio_analyzer = CandidateAudioAnalyzer(target_sr=16000)
    print("[OK] Initialized CandidateAudioAnalyzer with Librosa")

    if os.path.exists(QUESTIONS_PARQUET):
        df_q = parse_instruction_parquet(QUESTIONS_PARQUET)
        # Select top 50 diverse questions for mock interviews
        cached_questions = df_q.head(50).to_dict(orient="records")
        print(f"[OK] Cached {len(cached_questions)} reference interview questions")
except Exception as e:
    print(f"[ERROR] Startup loading error: {e}")


# ── Pydantic Request & Response Schemas ───────────────────────────────────
class EmployabilityRequest(BaseModel):
    manner_of_speaking: int = Field(..., ge=1, le=5, description="Speaking manner & clarity (1-5)")
    mental_alertness: int = Field(..., ge=1, le=5, description="Comprehension & mental agility (1-5)")
    self_confidence: int = Field(..., ge=1, le=5, description="Poise and confidence (1-5)")
    ability_to_present_ideas: int = Field(..., ge=1, le=5, description="Structure & presentation (1-5)")
    communication_skills: int = Field(..., ge=1, le=5, description="Overall communication (1-5)")

class EmployabilityResponse(BaseModel):
    prediction: str
    probability_employable: float
    probability_less_employable: float
    confidence_score: float
    soft_skill_mean: float
    soft_skill_min: int
    recommendations: List[str]

class TextAssessmentRequest(BaseModel):
    question: str
    candidate_answer: str
    reference_answer: Optional[str] = ""

class TextAssessmentResponse(BaseModel):
    relevance_score: float
    question_alignment_score: float
    keyword_overlap: float
    structure_score: float
    quality_score: float
    performance_tier: str
    feedback: List[str]


# ── API Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Intelligent Multimodal Mock Interview Assessment System",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/questions",
            "/api/v1/predict/employability",
            "/api/v1/assess/text",
            "/api/v1/assess/audio",
            "/api/v1/assess/multimodal"
        ]
    }

@app.get("/api/v1/questions")
def get_interview_questions(limit: int = 15):
    """Returns curated mock interview questions with expert reference answers."""
    if not cached_questions:
        return {"questions": []}
    return {"questions": cached_questions[:limit]}


@app.post("/api/v1/predict/employability", response_model=EmployabilityResponse)
def predict_employability(req: EmployabilityRequest):
    """
    Predicts student employability based on 5 non-appearance soft-skill features.
    Computes engineered soft_skill_mean and soft_skill_min before inference.
    """
    if employability_model is None:
        raise HTTPException(status_code=503, detail="Employability model pipeline is not loaded.")
    
    # Feature engineering matching the trained model pipeline
    features_dict = {
        "MANNER OF SPEAKING": req.manner_of_speaking,
        "MENTAL ALERTNESS": req.mental_alertness,
        "SELF-CONFIDENCE": req.self_confidence,
        "ABILITY TO PRESENT IDEAS": req.ability_to_present_ideas,
        "COMMUNICATION SKILLS": req.communication_skills,
    }
    raw_vals = list(features_dict.values())
    features_dict["soft_skill_mean"] = float(np.mean(raw_vals))
    features_dict["soft_skill_min"] = int(np.min(raw_vals))
    
    input_df = pd.DataFrame([features_dict])
    
    # Predict
    pred = employability_model.predict(input_df)[0]
    proba = employability_model.predict_proba(input_df)[0]
    
    emp_idx = list(employability_model.classes_).index("Employable")
    less_idx = 1 - emp_idx
    
    p_emp = float(proba[emp_idx])
    p_less = float(proba[less_idx])
    
    # Generate tailored actionable feedback
    feedback = []
    if req.communication_skills < 3:
        feedback.append("Focus on verbal conciseness and vocabulary precision in technical descriptions.")
    if req.manner_of_speaking < 3:
        feedback.append("Practice steady pacing and clear vocal articulation to enhance delivery.")
    if req.self_confidence < 3:
        feedback.append("Engage in timed mock interviews to reduce hesitation under pressure.")
    if req.ability_to_present_ideas < 3:
        feedback.append("Utilize the STAR method (Situation, Task, Action, Result) to organize answers.")
    if req.mental_alertness < 3:
        feedback.append("Take 3 seconds to structure your thoughts before answering complex questions.")
    if not feedback:
        feedback.append("Excellent soft-skill profile! Continue reinforcing technical domain knowledge.")

    return EmployabilityResponse(
        prediction=pred,
        probability_employable=round(p_emp, 4),
        probability_less_employable=round(p_less, 4),
        confidence_score=round(max(p_emp, p_less) * 100, 1),
        soft_skill_mean=round(features_dict["soft_skill_mean"], 2),
        soft_skill_min=features_dict["soft_skill_min"],
        recommendations=feedback
    )


@app.post("/api/v1/assess/text", response_model=TextAssessmentResponse)
def assess_interview_text(req: TextAssessmentRequest):
    """
    Evaluates candidate typed or transcribed answer against question and reference answer.
    """
    if text_assessor is None:
        raise HTTPException(status_code=503, detail="Text assessment engine not loaded.")
    
    q = req.question.strip()
    ans = req.candidate_answer.strip()
    ref = req.reference_answer.strip() if req.reference_answer else q
    
    if len(ans.split()) < 3:
        raise HTTPException(status_code=400, detail="Answer is too short to evaluate (minimum 3 words).")
        
    scores = text_assessor.assess_answer(q, ans, ref)
    q_score = scores["quality_score"]
    
    if q_score >= 0.35:
        tier = "Proficient / Strong Answer"
    elif q_score >= 0.15:
        tier = "Intermediate / Partially Correct"
    else:
        tier = "Developing / Insufficient Coverage"
        
    feedback = []
    if scores["relevance_score"] < 0.20:
        feedback.append("Content relevance is low compared to the reference standard; ensure you address specific technical keywords.")
    else:
        feedback.append("Good semantic alignment with the expected technical concepts.")
        
    if scores["structure_score"] < 0.40:
        feedback.append("Response could be improved by using logical transitions (e.g. 'for example', 'specifically', 'therefore').")
    else:
        feedback.append("Well-structured sentence organization and adequate response length.")
        
    return TextAssessmentResponse(
        relevance_score=scores["relevance_score"],
        question_alignment_score=scores["question_alignment_score"],
        keyword_overlap=scores["keyword_overlap"],
        structure_score=scores["structure_score"],
        quality_score=q_score,
        performance_tier=tier,
        feedback=feedback
    )


@app.post("/api/v1/assess/audio")
async def assess_interview_audio(file: UploadFile = File(...)):
    """
    Accepts uploaded audio clip (WAV/MP3), extracts pitch, pauses, energy, and estimates fluency/prosody.
    """
    if audio_analyzer is None:
        raise HTTPException(status_code=503, detail="Audio analyzer not loaded.")
        
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".wav", ".mp3", ".ogg", ".flac"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Please upload WAV or MP3.")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        metrics = audio_analyzer.analyze_audio_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    # Generate speech feedback
    audio_feedback = []
    if metrics["pause_ratio"] > 0.35:
        audio_feedback.append("High silence ratio detected (>35%); try to reduce long hesitations between sentences.")
    elif metrics["pause_ratio"] < 0.08:
        audio_feedback.append("Very low pause ratio (<8%); remember to breathe and pause between major thoughts.")
    else:
        audio_feedback.append("Healthy speech rhythm and natural pacing.")
        
    if metrics["pitch_std_hz"] < 10.0:
        audio_feedback.append("Vocal pitch shows low modulation; adding natural inflection can increase engagement.")
    else:
        audio_feedback.append("Expressive prosody and good vocal variation.")
        
    return {
        "audio_metrics": metrics,
        "speech_feedback": audio_feedback
    }


@app.post("/api/v1/assess/multimodal")
async def assess_multimodal(
    question: str = Form(...),
    reference_answer: str = Form(""),
    candidate_answer: str = Form(...),
    manner_of_speaking: int = Form(4),
    mental_alertness: int = Form(4),
    self_confidence: int = Form(4),
    ability_to_present_ideas: int = Form(4),
    communication_skills: int = Form(4),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Comprehensive Multimodal Assessment: Text + Soft Skills + Optional Audio.
    """
    # 1. Text assessment
    text_res = text_assessor.assess_answer(question, candidate_answer, reference_answer)
    
    # 2. Soft skills employability prediction
    features_dict = {
        "MANNER OF SPEAKING": manner_of_speaking,
        "MENTAL ALERTNESS": mental_alertness,
        "SELF-CONFIDENCE": self_confidence,
        "ABILITY TO PRESENT IDEAS": ability_to_present_ideas,
        "COMMUNICATION SKILLS": communication_skills,
    }
    raw_vals = list(features_dict.values())
    features_dict["soft_skill_mean"] = float(np.mean(raw_vals))
    features_dict["soft_skill_min"] = int(np.min(raw_vals))
    
    emp_pred = employability_model.predict(pd.DataFrame([features_dict]))[0]
    proba = employability_model.predict_proba(pd.DataFrame([features_dict]))[0]
    emp_idx = list(employability_model.classes_).index("Employable")
    p_emp = float(proba[emp_idx])
    
    # 3. Audio assessment if file provided
    audio_metrics = None
    if audio_file is not None:
        suffix = os.path.splitext(audio_file.filename)[1].lower()
        if suffix in [".wav", ".mp3", ".ogg", ".flac"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(audio_file.file, tmp)
                tmp_path = tmp.name
            try:
                audio_metrics = audio_analyzer.analyze_audio_file(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
    return {
        "overall_status": "Evaluation Complete",
        "employability_result": {
            "prediction": emp_pred,
            "employability_probability": round(p_emp, 4)
        },
        "text_evaluation": text_res,
        "speech_acoustic_evaluation": audio_metrics
    }

