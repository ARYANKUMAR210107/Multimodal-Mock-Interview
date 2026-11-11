# Multimodal report score

The report exposes `text_assessment_score` as a normalized value from 0 to 1 and displays it as a percentage. It reuses the text assessor's existing weighted heuristic:

```text
Text Assessment Score =
    0.50 × TF-IDF cosine similarity to the reference answer
  + 0.20 × TF-IDF cosine similarity to the question
  + 0.15 × Jaccard keyword overlap with the reference answer
  + 0.15 × answer structure heuristic
```

All four components must be finite numeric values within [0, 1]. The backend recalculates the weighted score from those components; a missing or invalid component yields a `null` score rather than a fabricated zero. This ensures the returned report score follows the same formula as `InterviewTextAssessor`.

Pitch, RMS energy, MFCCs, duration, pause counts and ratios are returned as separate voice measurements. They are not normalized or included in the text score because this project has no validated audio-quality scale. `overall_multimodal_assessment.status` is `Available` only when a valid text score and at least one substantive acoustic measurement are present. That status indicates both modalities have data; it is not a combined numeric quality rating.

The score is a reference-based practice heuristic. The interview corpus contains reference question/answer pairs, not human-rated candidate submissions, so the score is not calibrated against expert grading and must not be interpreted as a hiring recommendation.
