import re
import numpy as np
import sacrebleu
from sentence_transformers import SentenceTransformer
from vendi_score import vendi

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def split_ideas(text):
    """
    Extract idea-like items from model output.
    Works with numbered lists, bullet lists, or paragraphs.
    """
    lines = text.splitlines()
    ideas = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # match bullets or numbered items
        if re.match(r"^(\d+[\).\s]|[-*•])\s*", line):
            cleaned = re.sub(r"^(\d+[\).\s]|[-*•])\s*", "", line).strip()
            if len(cleaned.split()) >= 5:
                ideas.append(cleaned)
                
    # fallback: split by paragraphs if no list items found
    if len(ideas) < 2:
        paragraphs = [p.strip()]
        ideas = paragraphs

    return ideas

def compute_distinct_n(ideas, n=2):
    ngrams = []
    total = 0

    for idea in ideas:
        tokens = idea.lower().split()
        if len(tokens) < n: continue

        current = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        ngrams.extend(current)
        total += len(current)

    if total == 0: return 0
    return len(set(ngrams)) / total

def compute_self_bleu(ideas):
    if len(ideas) < 2: return 0
    scores = []

    for i, hypothesis in enumerate(ideas):
        ref = [idea for j, idea in enumerate(ideas) if j != i]
        if not ref: continue

        score = sacrebleu.sentence_bleu(hypothesis, ref).score
        scores.append(score)
    
    if not scores: return 0
    return float(np.mean(scores))

def compute_embedding_diversity_metrics(ideas):
    """
    Computes reference-free, non-LLM diversity metrics for a set of ideas.
    """
    if len(ideas) < 2:
        return {
            "idea_count": len(ideas),
            "vendi_score": 0,
            "mean_pairwise_distance": 0,
            "distinct_1": 0,
            "distinct_2": 0,
            "self_bleu": 0,
        }
    
    model = get_embedding_model()
    X = model.encode(ideas)
    vendi_score = float(vendi.score_X(X))
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    similarity_matrix = Xn @ Xn.T

    upper_triangle = np.triu_indices(len(ideas), 1)
    mean_pairwise_similarity = similarity_matrix[upper_triangle].mean()
    mean_pairwise_distance = float(1 - mean_pairwise_similarity)

    distinct_1 = compute_distinct_n(ideas, n=1)
    distinct_2 = compute_distinct_n(ideas, n=2)
    self_bleu = compute_self_bleu(ideas)

    return {
        "idea_count": len(ideas),
        "vendi_score": round(vendi_score, 3),
        "mean_pairwise_distance": round(mean_pairwise_distance, 3),
        "distinct_1": round(distinct_1, 3),
        "distinct_2": round(distinct_2, 3),
        "self_bleu": round(self_bleu, 3),
    }

def evaluate_idea_set_diversity(output_text):
    ideas = split_ideas(output_text)
    metrics = compute_embedding_diversity_metrics(ideas)
    return {
        "ideas": ideas,
        "metrics": metrics,
    }