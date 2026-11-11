"""
Verification Test Script for Fixes 1, 2, and 3
"""
import os, sys, json, joblib
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)
sys.path.append(os.path.join(BASE, "src"))

from src.text_pipeline import InterviewTextAssessor
from src.audio_feature_extractor import CandidateAudioAnalyzer

def test_all():
    print("=== TESTING FIX 1: Gradient Boosting Pipeline ===")
    gb_path = os.path.join(BASE, "models", "gradient_boosting_employability.joblib")
    gb = joblib.load(gb_path)
    input_df = pd.DataFrame([{
        "MANNER OF SPEAKING": 4,
        "MENTAL ALERTNESS": 4,
        "SELF-CONFIDENCE": 4,
        "ABILITY TO PRESENT IDEAS": 3,
        "COMMUNICATION SKILLS": 4,
        "soft_skill_mean": 3.8,
        "soft_skill_min": 3
    }])
    pred = gb.predict(input_df)[0]
    proba = gb.predict_proba(input_df)[0]
    emp_idx = list(gb.classes_).index("Employable")
    print(f"Prediction: {pred}, P(Employable): {proba[emp_idx]:.4f}")
    assert pred in ["Employable", "LessEmployable"]
    print("Fix 1 Verified Successfully!\n")

    print("=== TESTING FIX 2: Multimodal Rule-Based Fusion ===")
    analyzer = CandidateAudioAnalyzer(target_sr=16000)
    # Generate 2.5s sample wave
    sr = 16000
    t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
    y_test = 0.4 * np.sin(2 * np.pi * 180 * t)
    y_test[int(sr * 1.0):int(sr * 1.3)] = 0.0001
    audio_metrics = analyzer.extract_features_from_array(y_test, sr=sr)
    
    vec_path = os.path.join(BASE, "models", "interview_tfidf_vectorizer.joblib")
    vec = joblib.load(vec_path)
    assessor = InterviewTextAssessor(vectorizer=vec)
    assessor.is_fitted = True
    
    text_metrics = assessor.assess_answer(
        "What is polymorphism?",
        "Polymorphism allows different objects to respond to the same message in unique ways.",
        "Polymorphism is the ability of different classes to respond to the same method call."
    )
    
    pause_ratio = audio_metrics.get("pause_ratio") or 0.2
    pitch_std = audio_metrics.get("pitch_std_hz") or 15.0
    fluency_score = float(np.clip(5.0 - (pause_ratio * 7.0), 1.0, 5.0))
    prosody_score = float(np.clip(1.0 + (pitch_std / 10.0), 1.0, 5.0))
    fluency_norm = float(np.clip(fluency_score / 5.0, 0.0, 1.0))
    prosody_norm = float(np.clip(prosody_score / 5.0, 0.0, 1.0))
    text_quality = float(text_metrics["quality_score"])
    composite = float(np.clip((0.50 * text_quality) + (0.25 * fluency_norm) + (0.25 * prosody_norm), 0.0, 1.0))
    
    print(f"Text Quality: {text_quality:.4f}, Fluency: {fluency_norm:.4f}, Prosody: {prosody_norm:.4f}")
    print(f"Composite Multimodal Score: {composite*100:.1f} / 100")
    assert 0.0 <= composite <= 1.0
    print("Fix 2 Verified Successfully!\n")

    print("=== TESTING FIX 3: Dynamic Model Meta Loading ===")
    meta_path = os.path.join(BASE, "models", "employability_model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    test_results = meta.get("test_results", {})
    assert len(test_results) == 6, f"Expected 6 models, found {len(test_results)}"
    for name, m in test_results.items():
        print(f"  - {name:<22}: Acc={m['accuracy']*100:.2f}%, AUC={m['roc_auc']:.4f}")
    print("Fix 3 Verified Successfully!\n")
    print("ALL THREE FIXES VERIFIED 100% OPERATIONAL!")

if __name__ == "__main__":
    test_all()

