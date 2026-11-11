"""
Live HTTP Integration Tests against Running FastAPI Server (http://127.0.0.1:8000)
Migrated to Google Gemini API
"""
import os
import sys
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000"
SAMPLE_WAV = os.path.join(str(BASE_DIR), "reports", "test_candidate_sample.wav")

def test_live_server():
    print("=" * 60)
    print(f"TESTING LIVE FASTAPI SERVER AT: {API_URL}")
    print("=" * 60)

    # 1. Root
    r = requests.get(f"{API_URL}/")
    print(f"1. GET /: Status {r.status_code}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    print(f"   Response: status={data['status']}, provider={data.get('transcription_provider')}, configured={data['transcription_configured']}")

    # 2. Questions
    r = requests.get(f"{API_URL}/api/v1/questions?limit=3")
    print(f"2. GET /api/v1/questions: Status {r.status_code}")
    assert r.status_code == 200
    qs = r.json()["questions"]
    print(f"   Returned {len(qs)} questions. First: {qs[0]['question'][:50]}...")

    # 3. Employability Predictor
    r = requests.post(f"{API_URL}/api/v1/predict/employability", json={
        "manner_of_speaking": 4,
        "mental_alertness": 4,
        "self_confidence": 4,
        "ability_to_present_ideas": 3,
        "communication_skills": 4
    })
    print(f"3. POST /api/v1/predict/employability: Status {r.status_code}")
    assert r.status_code == 200
    emp = r.json()
    print(f"   Prediction: {emp['prediction']} (P(Employable)={emp['probability_employable']:.2%})")

    # 4. Text Assessment
    r = requests.post(f"{API_URL}/api/v1/assess/text", json={
        "question": "What is overfitting in machine learning?",
        "candidate_answer": "Overfitting happens when a model learns the training noise instead of the general distribution, resulting in poor generalization.",
        "reference_answer": "Overfitting occurs when a statistical model memorizes training noise rather than capturing general underlying relationships."
    })
    print(f"4. POST /api/v1/assess/text: Status {r.status_code}")
    assert r.status_code == 200
    txt = r.json()
    print(f"   Quality Score: {txt['quality_score']:.4f}, Tier: {txt['performance_tier']}")

    # 5. Audio Assessment
    with open(SAMPLE_WAV, "rb") as f:
        r = requests.post(f"{API_URL}/api/v1/assess/audio", files={"file": ("candidate.wav", f, "audio/wav")})
    print(f"5. POST /api/v1/assess/audio: Status {r.status_code}")
    assert r.status_code == 200
    aud = r.json()
    print(f"   Duration: {aud['audio_metrics']['duration_sec']}s, Pause: {aud['audio_metrics']['pause_ratio']:.1%}, RMS: {aud['audio_metrics']['mean_energy_rms']}")

    # 6. Multimodal Assessment
    with open(SAMPLE_WAV, "rb") as f:
        r = requests.post(
            f"{API_URL}/api/v1/assess/multimodal",
            data={
                "question": "What is overfitting?",
                "candidate_answer": "Overfitting happens when a model fits training data too closely.",
                "manner_of_speaking": 4,
                "mental_alertness": 4,
                "self_confidence": 4,
                "ability_to_present_ideas": 3,
                "communication_skills": 4
            },
            files={"audio_file": ("candidate.wav", f, "audio/wav")}
        )
    print(f"6. POST /api/v1/assess/multimodal: Status {r.status_code}")
    assert r.status_code == 200
    mul = r.json()
    print(f"   Status: {mul['overall_status']}, Employability: {mul['employability_result']['prediction']}")

    # 7. Multimodal Report
    with open(SAMPLE_WAV, "rb") as f:
        r = requests.post(
            f"{API_URL}/api/interview/multimodal-report",
            data={
                "question": "What is overfitting in machine learning?",
                "reference_answer": "Overfitting occurs when a model learns training noise.",
                "corrected_transcript": "Um overfitting is when a model memorizes noise like you know.",
            },
            files={"file": ("candidate.wav", f, "audio/wav")}
        )
    print(f"7. POST /api/interview/multimodal-report: Status {r.status_code}")
    assert r.status_code == 200
    rep = r.json()
    print(f"   Report: Fillers={rep['voice_assessment']['filler_word_count']}, WPM={rep['voice_assessment']['speaking_rate_wpm']}")

    # 8. Google Gemini Audio Transcription
    with open(SAMPLE_WAV, "rb") as f:
        r = requests.post(
            f"{API_URL}/api/audio/transcribe",
            files={"file": ("candidate.wav", f, "audio/wav")}
        )
    print(f"8. POST /api/audio/transcribe: Status {r.status_code}")
    if r.status_code == 200:
        print(f"   Gemini Transcription SUCCESS: '{r.json().get('transcript')}' (Model: {r.json().get('model')})")
    elif r.status_code == 503:
        print(f"   Gemini Not Configured (Expected if GEMINI_API_KEY is not yet in .env): {r.json().get('detail')}")
    elif r.status_code in (401, 429):
        print(f"   Gemini API Response ({r.status_code}): {r.json().get('detail')}")
    else:
        print(f"   Gemini Transcription response ({r.status_code}): {r.text[:200]}")

    print("\nALL LIVE INTEGRATION CHECKS COMPLETED!")

if __name__ == "__main__":
    test_live_server()
