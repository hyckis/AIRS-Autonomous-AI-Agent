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
    text = text.strip()
    text = re.sub(r"(?<!\n)(\d+\.\s*Title:)", r"\n\1", text)
    text = re.sub(r"^Okay,[\s\S]*?(?=\n\s*1\.\s*Title:|\n\s*Title:|\n\s*\d+[\).\s]+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?(I believe these directions|Do you want me|Would you like me)[\s\S]*$", "", text, flags=re.IGNORECASE)

    ideas = []

    # case 1: lens agent format "1. Title: ..."
    pattern = r"(?:^|\n)\s*\d+\.\s*Title:\s*([\s\S]*?)(?=\n\s*\d+\.\s*Title:|\Z)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    if matches:
        for match in matches:
            idea = "Title: "  + match.strip()
            idea = re.sub(r"http\S+", "", idea)
            idea = re.sub(r"\s+", " ", idea).strip()
            if len(idea.split()) >= 8: ideas.append(idea)
        return ideas
    
    # case 2: strong prompt format: repeated "Title: ..."
    pattern = r"(?:^|\n)\s*Title:\s*([\s\S]*?)(?=\n\s*Title:|\Z)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    if matches:
        for match in matches:
            idea = "Title: " + match.strip()
            idea = re.sub(r"http\S+", "", idea)
            idea = re.sub(r"\s+", " ", idea).strip()
            if len(idea.split()) >= 8: ideas.append(idea)
        return ideas
    
    # case 3: naive format: numbered/bullet list
    lines = text.splitlines()
    current_idea = []

    metadata_prefixes = (
        "note on",
        "potential homogenization",
        "do you want me",
        "would you like me",
    )

    # metadata_prefixes = (
    #     "differentiation:",
    #     "challenges dominant assumption:",
    #     "cognitive diversity preservation",
    #     "supporting paper:",
    #     "speculative extension:",
    #     "evidence:",
    #     "rationale:",
    #     "method:",
    #     "evaluation:",
    # )

    def flush():
        nonlocal current_idea
        if current_idea:
            idea = " ".join(current_idea).strip()
            idea = re.sub(r"\s+", " ", idea)
            if len(idea.split()) >= 8: ideas.append(idea)
            current_idea = []

    for line in lines:
        # line = line.strip()
        # if not line: continue

        lower = line.lower()
        if lower.startswith(metadata_prefixes): 
            flush()
            continue

        # start of a numbered idea
        if re.match(r"^\d+[\).\s]+", line):
            flush()
            current_idea.append(re.sub(r"^\d+[\).\s]+", "", line).strip())
            continue
        # # start of a title line
        # if line.lower().startswith("title"):
        #     flush()
        #     current_idea.append(line)
        #     continue
        # # add only high level description, not metadata fields
        # if line.lower().startswith("description"):
        #     current_idea.append(line)
        #     continue
        # if line.lower().startswith((metadata_prefixes)): continue
        # if current_idea: current_idea.append(line)
        elif current_idea: current_idea.append(line)

    flush()

    # fallback: bullet/numbered list extraction
    # if len(ideas) < 2:
    #     ideas = []
    #     for line in lines:
    #         line = line.strip()
    #         m = re.match(r"^(\d+[\).\s]|[-*•])\s*", line)
    #         if m:
    #             cleaned = re.sub(r"^(\d+[\).\s]|[-*•])\s*", "", line).strip()
    #             if len(cleaned.split()) >= 5: ideas.append(cleaned)
        # chunks = re.split(r"\n\s*\n", text)
        # ideas = [
        #     chunk.strip()
        #     for chunk in chunks
        #     if len(chunk.strip().split()) >= 8 
        # ]
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
    if len(ideas) == 0:
        return {
            "idea_count": 0,
            "vendi_score": 0,
            "mean_pairwise_distance": 0,
            "distinct_1": 0,
            "distinct_2": 0,
            "self_bleu": 0,
        }
    
    if len(ideas) == 1:
        return {
            "idea_count": 1,
            "vendi_score": 1,
            "mean_pairwise_distance": 0,
            "distinct_1": round(compute_distinct_n(ideas, n=1), 3),
            "distinct_2": round(compute_distinct_n(ideas, n=2), 3),
            "self_bleu": 0,
        }
    # if len(ideas) < 2:
    #     return {
    #         "idea_count": len(ideas),
    #         "vendi_score": 0,
    #         "mean_pairwise_distance": 0,
    #         "distinct_1": 0,
    #         "distinct_2": 0,
    #         "self_bleu": 0,
    #     }
    
    model = get_embedding_model()
    X = model.encode(ideas)
    vendi_score = float(vendi.score_X(X))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1
    Xn = X / norms
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