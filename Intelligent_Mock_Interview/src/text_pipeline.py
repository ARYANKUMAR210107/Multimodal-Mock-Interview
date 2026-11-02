"""
=============================================================================
Interview Text Assessment Pipeline & Feature Extraction Engine
Intelligent Multimodal Mock Interview Assessment System
=============================================================================

BEGINNER GUIDE — What this module does:
----------------------------------------
In an interview, a candidate is asked a question (e.g. "What is an abstract class?").
We have a reference/expert answer from our dataset. We need an objective,
mathematically grounded NLP system to evaluate:
  1. Content Relevance: Did the candidate answer the question asked?
  2. Answer Completeness: How well does it match key technical concepts in the reference answer?
  3. Structural Coherence: Is the response well-structured and grammatically coherent?
  4. Vocabulary Richness: Does the candidate exhibit professional technical fluency?

IMPORTANT ARCHITECTURAL RULE:
  The K-areem/AI-Interview-Questions dataset does NOT contain human quality ratings.
  It contains question-answer pairs intended as instruction data.
  Treating them as graded submissions would be an ML fallacy!
  Instead, we use them as an expert reference knowledge base against which
  candidate answers are scored.
=============================================================================
"""

import os, re, json, string
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Configuration & Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "interview_questions")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CUSTOM_DIR = os.path.join(BASE_DIR, "dataset", "custom")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CUSTOM_DIR, exist_ok=True)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves"
}


# ── Text Cleaning & Normalization ──────────────────────────────────────────
def clean_text(text: str) -> str:
    """Standardizes input string: lowercases, strips tags, removes punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # remove residual LLM prompt formatting if present
    text = re.sub(r'<s>|</s>|\[inst\]|\[/inst\]', '', text, flags=re.IGNORECASE)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_content_words(text: str) -> list:
    """Returns list of non-stopword alphanumeric tokens."""
    cleaned = clean_text(text)
    words = cleaned.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def parse_instruction_parquet(file_path: str) -> pd.DataFrame:
    """
    Parses parquet file containing `<s>[INST] question [/INST] answer </s>`.
    Returns structured DataFrame with separated question and answer columns.
    """
    df_raw = pd.read_parquet(file_path)
    
    questions = []
    answers = []
    for text in df_raw["text"]:
        q_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
        a_match = re.search(r'\[/INST\](.*?)(?:</s>|$)', text, re.DOTALL)
        
        q = q_match.group(1).strip() if q_match else ""
        a = a_match.group(1).strip() if a_match else ""
        
        # strip leftover markers
        q = re.sub(r'^<s>', '', q).strip()
        a = re.sub(r'</s>$', '', a).strip()
        
        questions.append(q)
        answers.append(a)
        
    df_clean = pd.DataFrame({
        "question": questions,
        "reference_answer": answers,
        "clean_question": [clean_text(q) for q in questions],
        "clean_reference_answer": [clean_text(a) for a in answers]
    })
    return df_clean


# ── Interview Text Scoring Engine ─────────────────────────────────────────
class InterviewTextAssessor:
    """
    Multimodal Text Feature Extractor & Answer Assessor.
    Combines TF-IDF Semantic Relevance, Lexical Overlap, and Structural Quality.
    """
    def __init__(self, vectorizer=None):
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=10000,
                stop_words="english"
            )
        self.is_fitted = False

    def fit_corpus(self, texts: list):
        """Fits TF-IDF on corpus of interview questions and reference answers."""
        cleaned_corpus = [clean_text(t) for t in texts if t and isinstance(t, str)]
        self.vectorizer.fit(cleaned_corpus)
        self.is_fitted = True
        return self

    def compute_tfidf_similarity(self, candidate_answer: str, reference_answer: str) -> float:
        """Computes Cosine Similarity between TF-IDF representations."""
        if not self.is_fitted:
            raise ValueError("TF-IDF Vectorizer has not been fitted on any reference corpus.")
        c_clean = clean_text(candidate_answer)
        r_clean = clean_text(reference_answer)
        if not c_clean or not r_clean:
            return 0.0
        
        vecs = self.vectorizer.transform([c_clean, r_clean])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def compute_jaccard_overlap(self, text_a: str, text_b: str) -> float:
        """Computes Jaccard Similarity on content keywords."""
        set_a = set(tokenize_content_words(text_a))
        set_b = set(tokenize_content_words(text_b))
        if not set_a or not set_b:
            return 0.0
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return float(len(intersection) / len(union))

    def compute_structural_score(self, text: str) -> float:
        """
        Heuristic structural assessment:
        - Evaluates sentence count and punctuation coherence.
        - Evaluates presence of logical transitional keywords (e.g. 'because', 'therefore', 'firstly').
        """
        if not text or not isinstance(text, str):
            return 0.0
        words = text.split()
        if len(words) < 5:
            return 0.2  # severely incomplete answer
        
        # Transition markers
        connectors = {"because", "therefore", "firstly", "secondly", "additionally", 
                      "for example", "specifically", "however", "consequently", "furthermore"}
        text_lower = text.lower()
        connector_count = sum(1 for c in connectors if c in text_lower)
        connector_score = min(connector_count * 0.25, 0.5)
        
        # Length adequacy (around 30-100 words is a typical concise answer)
        word_count = len(words)
        if 25 <= word_count <= 150:
            length_score = 0.5
        elif word_count > 150:
            length_score = 0.4
        else:
            length_score = (word_count / 25.0) * 0.5
            
        return float(np.clip(connector_score + length_score, 0.1, 1.0))

    def assess_answer(self, question: str, candidate_answer: str, reference_answer: str) -> dict:
        """
        Produces complete multidimensional assessment scores:
        - relevance_score: Content match with reference answer (0.0 to 1.0)
        - question_alignment_score: Topic match directly against the question (0.0 to 1.0)
        - keyword_overlap: Jaccard overlap of key technical concepts
        - structure_score: Structural and grammatical completeness
        - quality_score: Composite weighted assessment
        - strengths: Specific positive technical attributes
        - improvements: Targeted growth areas
        - suggested_structure: Recommended answer architecture
        """
        relevance = self.compute_tfidf_similarity(candidate_answer, reference_answer)
        q_align = self.compute_tfidf_similarity(candidate_answer, question)
        jaccard = self.compute_jaccard_overlap(candidate_answer, reference_answer)
        structure = self.compute_structural_score(candidate_answer)
        
        # Composite score calculation (explainable weighting)
        composite = (0.50 * relevance) + (0.20 * q_align) + (0.15 * jaccard) + (0.15 * structure)
        q_score = float(np.clip(composite, 0.0, 1.0))
        
        # Qualitative feedback generation
        strengths = []
        improvements = []
        
        if relevance >= 0.35:
            strengths.append("Strong conceptual alignment with the technical benchmark answer.")
        elif relevance >= 0.20:
            strengths.append("Basic understanding of the core concept demonstrated.")
        else:
            improvements.append("Response lacks critical technical terminology present in the reference answer.")
            
        if jaccard >= 0.20:
            strengths.append("High overlap of technical domain keywords.")
        else:
            improvements.append("Incorporate more exact technical terms and key definitions.")
            
        if structure >= 0.60:
            strengths.append("Effective use of transitional markers and logical sequencing.")
        elif len(candidate_answer.split()) < 20:
            improvements.append("Answer is too brief. Expand with practical details and mechanisms.")
        else:
            improvements.append("Use structuring connectors (e.g. 'Firstly', 'Specifically', 'In contrast') to organize points.")
            
        if not strengths:
            strengths.append("Attempted to address the question prompt directly.")
            
        suggested_structure = [
            "1. Concise Definition: Begin with a direct 1-sentence technical definition.",
            "2. Mechanism / How it Works: Detail internal operations, algorithms, or protocols.",
            "3. Trade-offs & Complexity: Highlight time/space complexities or architectural tradeoffs.",
            "4. Practical Example: Conclude with a real-world scenario or industry use-case."
        ]
        
        return {
            "relevance_score": round(relevance, 4),
            "question_alignment_score": round(q_align, 4),
            "keyword_overlap": round(jaccard, 4),
            "structure_score": round(structure, 4),
            "quality_score": round(q_score, 4),
            "strengths": strengths,
            "improvements": improvements,
            "suggested_structure": suggested_structure
        }


# ── Execution Pipeline ────────────────────────────────────────────────────
def run_text_pipeline():
    print("=" * 70)
    print("  STEP 1: INSPECTION OF AI INTERVIEW QUESTIONS DATASET")
    print("=" * 70)
    
    train_parquet = os.path.join(DATASET_DIR, "train-00000-of-00001.parquet")
    eval_parquet = os.path.join(DATASET_DIR, "eval-00000-of-00001.parquet")
    
    print(f"Reading train parquet: {train_parquet}")
    df_train = parse_instruction_parquet(train_parquet)
    print(f"Train records parsed: {len(df_train)}")
    
    print(f"Reading eval parquet: {eval_parquet}")
    df_eval = parse_instruction_parquet(eval_parquet)
    print(f"Eval records parsed: {len(df_eval)}")
    
    print("\nSample Parsed Q&A:")
    print("Question:", df_train.iloc[0]["question"])
    print("Reference Answer:", df_train.iloc[0]["reference_answer"][:160] + "...")
    
    print("\n" + "=" * 70)
    print("  STEP 2: FITTING TF-IDF REFERENCE REPOSITORY VECTORIZER")
    print("=" * 70)
    
    corpus = list(df_train["question"]) + list(df_train["reference_answer"]) + \
             list(df_eval["question"]) + list(df_eval["reference_answer"])
    print(f"Total reference text segments for TF-IDF dictionary: {len(corpus)}")
    
    assessor = InterviewTextAssessor()
    assessor.fit_corpus(corpus)
    print(f"TF-IDF Dictionary Size: {len(assessor.vectorizer.vocabulary_)} n-grams")
    
    # Save vectorizer and assessor pipeline
    vectorizer_path = os.path.join(MODELS_DIR, "interview_tfidf_vectorizer.joblib")
    joblib.dump(assessor.vectorizer, vectorizer_path)
    print(f"Saved fitted vectorizer to: {vectorizer_path}")
    
    print("\n" + "=" * 70)
    print("  STEP 3: BENCHMARK DEMONSTRATION ACROSS ANSWER QUALITY TIERS")
    print("=" * 70)
    
    sample_q = df_train.iloc[0]["question"]
    sample_ref = df_train.iloc[0]["reference_answer"]
    
    # 3 distinct test responses
    candidate_responses = {
        "High Quality (Thorough & Accurate)": (
            "UML facilitates specification in the analysis, design, and implementation "
            "of software-intensive systems by offering comprehensive visual diagrams "
            "and notations. It covers both structural and behavioral aspects from concept to execution."
        ),
        "Moderate Quality (Partial understanding)": (
            "UML is used by software engineers to draw diagrams like class diagrams "
            "and use case diagrams when designing an application."
        ),
        "Poor Quality (Irrelevant / Vague)": (
            "I think software is very important for modern computers and people write code in it."
        )
    }
    
    demo_results = []
    print(f"Target Question: {sample_q}\n")
    for tier, answer in candidate_responses.items():
        scores = assessor.assess_answer(sample_q, answer, sample_ref)
        scores["tier"] = tier
        scores["answer_snippet"] = answer[:80] + "..."
        demo_results.append(scores)
        print(f"[{tier}]")
        print(f"  Relevance: {scores['relevance_score']} | Alignment: {scores['question_alignment_score']} | Structure: {scores['structure_score']} | Overall: {scores['quality_score']}\n")
        
    print("=" * 70)
    print("  STEP 4: VERIFY CUSTOM DATASET TEMPLATE FOR HUMAN ANNOTATION")
    print("=" * 70)
    
    template_path = os.path.join(CUSTOM_DIR, "interview_answer_assessment_template.csv")
    print(f"Template path: {template_path}")
    
    # Check if template exists
    if os.path.exists(template_path):
        tmpl_df = pd.read_csv(template_path)
        print(f"Template columns: {list(tmpl_df.columns)}")
        print(f"Records currently in template: {len(tmpl_df)}")
        print("Confirmed: Training labels remain UNTOUCHED until authentic human evaluation is collected.")
        
    report_path = os.path.join(REPORTS_DIR, "interview_text_pipeline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("INTERVIEW TEXT NLP PIPELINE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"1. Dataset: K-areem/AI-Interview-Questions\n")
        f.write(f"   - Train records: {len(df_train)}\n")
        f.write(f"   - Eval records: {len(df_eval)}\n")
        f.write(f"   - Genuine Human Quality Labels: NONE (Architectural note: These are reference Q&A pairs, not graded candidate submissions)\n\n")
        f.write(f"2. TF-IDF Vocabulary Size: {len(assessor.vectorizer.vocabulary_)} features\n")
        f.write(f"3. Scoring Dimensions: TF-IDF Cosine Relevance, Question Alignment, Jaccard Keyword Overlap, Structural Coherence\n")
        f.write(f"4. Demonstration Benchmark:\n")
        for res in demo_results:
            f.write(f"   - Tier: {res['tier']}\n")
            f.write(f"     Quality Score: {res['quality_score']} (Relevance: {res['relevance_score']}, Structure: {res['structure_score']})\n")
    print(f"\nSaved report to: {report_path}")

if __name__ == "__main__":
    run_text_pipeline()

