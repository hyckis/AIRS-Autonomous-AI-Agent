import numpy as np
from diversity_metrics import get_embedding_model


def paper_to_text(paper):
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    return f"{title}. {abstract}".strip()


def normalize_embeddings(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def compute_literature_grounded_metrics(
    ideas,
    papers,
    support_threshold=0.55,
    top_n_sources=3,
):
    """
    Computes per-idea literature-grounded metrics.

    Novelty:
        1 - max cosine similarity between idea and retrieved papers.

    Evidence-support proxy:
        Number/proportion of retrieved papers whose cosine similarity
        to the idea exceeds support_threshold.

    Important:
        This is a retrieved-literature proxy, not claim-level evidence validation.
    """
    if not ideas:
        return {
            "per_idea": [],
            "average_novelty": 0,
            "average_evidence_count": 0,
            "average_evidence_ratio": 0,
        }

    if not papers:
        return {
            "per_idea": [
                {
                    "idea": idea,
                    "novelty": 0,
                    "max_similarity": 0,
                    "evidence_count": 0,
                    "evidence_ratio": 0,
                    "closest_paper": None,
                    "supporting_sources": [],
                }
                for idea in ideas
            ],
            "average_novelty": 0,
            "average_evidence_count": 0,
            "average_evidence_ratio": 0,
        }

    model = get_embedding_model()

    idea_embeddings = model.encode(ideas)
    paper_texts = [paper_to_text(paper) for paper in papers]
    paper_embeddings = model.encode(paper_texts)

    idea_embeddings = normalize_embeddings(idea_embeddings)
    paper_embeddings = normalize_embeddings(paper_embeddings)

    similarity_matrix = idea_embeddings @ paper_embeddings.T

    per_idea = []

    for i, idea in enumerate(ideas):
        sims = similarity_matrix[i]

        max_idx = int(np.argmax(sims))
        max_similarity = float(sims[max_idx])
        novelty = float(1 - max_similarity)

        support_indices = np.where(sims >= support_threshold)[0]
        evidence_count = int(len(support_indices))
        evidence_ratio = float(evidence_count / len(papers))

        top_indices = np.argsort(sims)[::-1][:top_n_sources]
        supporting_sources = []

        for idx in top_indices:
            supporting_sources.append({
                "title": papers[idx].get("title", ""),
                "url": papers[idx].get("url", ""),
                "similarity": round(float(sims[idx]), 3),
                "above_threshold": bool(sims[idx] >= support_threshold),
            })

        per_idea.append({
            "idea": idea,
            "novelty": round(novelty, 3),
            "max_similarity": round(max_similarity, 3),
            "evidence_count": evidence_count,
            "evidence_ratio": round(evidence_ratio, 3),
            "closest_paper": {
                "title": papers[max_idx].get("title", ""),
                "url": papers[max_idx].get("url", ""),
                "similarity": round(max_similarity, 3),
            },
            "supporting_sources": supporting_sources,
        })

    return {
        "per_idea": per_idea,
        "average_novelty": round(float(np.mean([x["novelty"] for x in per_idea])), 3),
        "average_evidence_count": round(float(np.mean([x["evidence_count"] for x in per_idea])), 3),
        "average_evidence_ratio": round(float(np.mean([x["evidence_ratio"] for x in per_idea])), 3),
    }
