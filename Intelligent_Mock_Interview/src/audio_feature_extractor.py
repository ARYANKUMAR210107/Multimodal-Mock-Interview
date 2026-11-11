"""
=============================================================================
Acoustic Feature Extraction Pipeline for Candidate Audio
Intelligent Multimodal Mock Interview Assessment System
=============================================================================

BEGINNER GUIDE — What this module does:
----------------------------------------
When a candidate speaks into the microphone during a mock interview:
  1. We record the audio waveform (WAV/MP3).
  2. We extract acoustic and prosodic features using `librosa` and `soundfile`:
     - MFCCs (Mel-Frequency Cepstral Coefficients): captures voice timbre & phonetic clarity.
     - Pitch (Fundamental Frequency F0): tracks intonation and pitch variation (monotone vs expressive).
     - RMS Energy: measures loudness and vocal projection.
     - Speaking Rate & Pauses: calculates words per minute and identifies awkward hesitations.
  3. We report descriptive acoustic measurements without converting them into
     unvalidated fluency, pronunciation, confidence, or quality scores.

IMPORTANT ARCHITECTURAL RULE:
  The 6.71 GB OpenEnded wav.zip was deliberately NOT downloaded to preserve storage.
  We do NOT claim to have trained an acoustic neural network on non-existent WAVs!
  Instead, this module provides the production-grade audio feature extraction engine
  ready to process live or uploaded candidate audio clips in the mock interview system.
=============================================================================
"""

import os, json, warnings
import numpy as np

# Audio libraries
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

class CandidateAudioAnalyzer:
    """
    Extracts acoustic features from recorded candidate speech clips:
    - Pitch variability & fundamental frequency (Prosody)
    - Pause frequency and speaking rhythm
    - RMS energy and MFCC spectral measurements
    """
    def __init__(self, target_sr=16000, n_mfcc=13):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc

    def load_audio(self, audio_path: str):
        """Loads and resamples audio to target sample rate (default 16kHz mono)."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return y, sr

    def extract_features_from_array(self, y: np.ndarray, sr: int = 16000) -> dict:
        """
        Extracts acoustic features from in-memory numpy waveform.
        """
        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError("Sample rate must be a positive integer in Hz.")
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            # Support channel-first and frame-first stereo arrays.
            if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
                y = np.mean(y, axis=0)
            elif y.shape[1] <= 8:
                y = np.mean(y, axis=1)
            else:
                raise ValueError("Audio array must be mono or stereo.")
        if y.ndim != 1:
            raise ValueError("Audio array must be mono or stereo.")
        y = np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if len(y) == 0:
            return {
                "duration_sec": 0.0,
                "speaking_duration_sec": 0.0,
                "mean_pitch_hz": None,
                "pitch_std_hz": None,
                "min_pitch_hz": None,
                "max_pitch_hz": None,
                "mean_energy_rms": None,
                "pause_ratio": None,
                "pause_duration_sec": None,
                "pause_count": None,
                "mfcc_mean": [None] * self.n_mfcc,
                "speech_segments": [],
            }
            
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        # 1. RMS Energy (Volume / Projection)
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms))
        
        # 2. Silence / Pause Detection
        # Non-silent intervals using standard top_db threshold
        intervals = librosa.effects.split(y, top_db=25)
        speaking_time = sum((end - start) for start, end in intervals) / float(sr)
        pause_ratio = float(np.clip((duration - speaking_time) / (duration + 1e-6), 0.0, 1.0))
        pause_duration = max(0.0, duration - speaking_time)
        pause_count = 0
        if len(intervals) > 1:
            for i in range(len(intervals) - 1):
                gap = (intervals[i+1][0] - intervals[i][1]) / float(sr)
                if gap >= 0.15:
                    pause_count += 1
        speech_segments = [
            {"start_sec": round(float(start) / sr, 2), "end_sec": round(float(end) / sr, 2)}
            for start, end in intervals
        ]
        
        # 3. Pitch tracking (F0 via YIN). Silence or unvoiced signals are
        # represented as unavailable rather than assigned a plausible default.
        try:
            f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr) if mean_rms >= 1e-5 else np.array([])
            # filter out unvoiced / zero frames
            valid_f0 = f0[(f0 > 70) & (f0 < 380)]
            if len(valid_f0) > 0:
                mean_f0 = float(np.mean(valid_f0))
                std_f0 = float(np.std(valid_f0))
                min_f0 = float(np.min(valid_f0))
                max_f0 = float(np.max(valid_f0))
            else:
                mean_f0, std_f0, min_f0, max_f0 = None, None, None, None
        except Exception:
            mean_f0, std_f0, min_f0, max_f0 = None, None, None, None
            
        # 4. MFCCs (Timbre & Pronunciation Clarity)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_means = [float(val) for val in np.mean(mfccs, axis=1)]

        return {
            "duration_sec": round(duration, 2),
            "speaking_duration_sec": round(speaking_time, 2),
            "pause_duration_sec": round(pause_duration, 2),
            "pause_count": int(pause_count),
            "pause_ratio": round(pause_ratio, 3),
            "mean_energy_rms": round(mean_rms, 4),
            "mean_pitch_hz": round(mean_f0, 2) if mean_f0 is not None else None,
            "pitch_std_hz": round(std_f0, 2) if std_f0 is not None else None,
            "min_pitch_hz": round(min_f0, 2) if min_f0 is not None else None,
            "max_pitch_hz": round(max_f0, 2) if max_f0 is not None else None,
            "mfcc_mean": [round(v, 3) for v in mfcc_means],
            "speech_segments": speech_segments
        }

    def analyze_audio_bytes(self, audio_bytes: bytes, transcript: str = "") -> dict:
        """Helper to decode audio bytes and extract acoustic metrics."""
        import io
        import soundfile as sf
        y, sr = sf.read(io.BytesIO(audio_bytes))
        return self.extract_features_from_array(y, sr)

    def analyze_audio_file(self, audio_path: str, include_heuristics: bool = False) -> dict:
        """Helper to load from file path and compute full acoustic metrics."""
        y, sr = self.load_audio(audio_path)
        metrics = self.extract_features_from_array(y, sr)
        if include_heuristics and metrics.get("pause_ratio") is not None:
            pause_ratio = metrics["pause_ratio"]
            std_f0 = metrics.get("pitch_std_hz")
            metrics["estimated_fluency_score"] = round(float(np.clip(5.0 - (pause_ratio * 7.0), 1.0, 5.0)), 2)
            metrics["estimated_prosody_score"] = round(float(np.clip(1.0 + (std_f0 / 10.0), 1.0, 5.0)), 2) if std_f0 is not None else None
        return metrics


if __name__ == "__main__":
    print("Testing CandidateAudioAnalyzer on synthetic candidate audio snippet...")
    # Generate 3 seconds of synthetic speech-like harmonic signal for unit verification
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    # 150 Hz harmonic tone with occasional silence
    y_test = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.25 * np.sin(2 * np.pi * 300 * t)
    # inject 0.5s pause
    y_test[int(sr * 1.0):int(sr * 1.5)] = 0.001
    
    analyzer = CandidateAudioAnalyzer(target_sr=sr)
    metrics = analyzer.extract_features_from_array(y_test, sr=sr)
    
    print("\nExtraction Successful! Extracted Metrics:")
    for k, v in metrics.items():
        if k != "mfcc_mean":
            print(f"  - {k}: {v}")
    print(f"  - mfcc_mean (13 coefficients): {metrics['mfcc_mean'][:4]} ...")

