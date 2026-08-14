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
    # "research direction #"
    text = re.sub(r"(?<!\n)(Research Direction\s*\d+\s*:)", r"\n\1", text, flags=re.IGNORECASE,)
    text = re.sub(r"(?<!\n)(\d+\.\s*Title:)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(\d+\.\s*Title\s*:)", r"\n\1", text, flags=re.IGNORECASE,)
    text = re.sub(r"^Okay,[\s\S]*?(?=\n\s*1\.\s*Title:|\n\s*Title:|\n\s*Research Direction\s*\d+\s*:|\n\s*\d+[\).\s]+)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?(I believe these directions|Do you want me|Would you like me)[\s\S]*$", "", text, flags=re.IGNORECASE)

    ideas = []

    # case 0: arm c nested format
    rd_pattern = (
        r"(?:^|\n)\s*Research Direction\s+\d+\s*:?\s*"
        r"([\s\S]*?)(?=\n\s*Research Direction\s*\d+\s*:|\Z)"
    )
    rd_matches = re.findall(rd_pattern, text, flags=re.IGNORECASE)
    if rd_matches:
        for match in rd_matches:
            idea = match.strip()
            # remove internal numbering from fields
            idea = re.sub(
                r"(?:^|\n)\s*\d+[\).\s]+(?=(Title|Description|Challenges|Challenge|Preserves|Supporting|Support|Speculative|Evidence|Rationale|Method|Evaluation)\b)", 
                r"\n", idea, flags=re.IGNORECASE,
            )
            idea = re.sub(r"http\S+", "", idea)
            idea = re.sub(r"\s+", " ", idea).strip()
            if len(idea.split()) >= 8: ideas.append(idea)

        return ideas

    # only normalize numbered title format after research direction parsing
    text = re.sub(r"(?:^|\n)(\d+\.\s*Title:)", r"\n\1", text, flags=re.IGNORECASE,)

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
        line = line.strip()
        if not line: continue

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

def extract_core_concept_text(idea):
    """
    Extract the core research concept from an idea block.
    Removes template/meta fields such as supporting paper, assumption challenge,
    cognitive diversity preservation, and implementation notes.

    Goal:
    full idea text -> core concept text for cleaner embedding-based diversity.
    keep only the idea itself: title + description / research direction

    Remove
    - differentiation
    - challenged assumption
    - cognitive diversity metadata
    - supporting papers
    - potential outcomes
    - persona
    - other evaluation metadata
    """
    if not idea: return ""

    text = re.sub(r"http\S+", "", idea)
    # Remove markdown bold markers: **Description** => Description
    text.replace("**", "")
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove parenthetical implementation notes that can distort embeddings
    text = re.sub(
        r"\([^)]*(LLM strength|Potential Limitation|Potential limitation)[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    stop_pattern = re.compile(
        r"\b(?:"
        r"Differentiation|"
        r"Challenges?\s+(?:Dominant\s+)?Assumption|"
        r"Preserves?\s+Cognitive\s+Diversity|"
        r"Cognitive\s+Diversity\s+Preservation|"
        r"Supporting\s+Paper|"
        r"Speculative\s+Extension|"
        r"Potential\s+Outcome|"
        r"Persona|"
        r"Evidence|"
        r"Rationale|"
        r"Method|"
        r"Evaluation|"
        r")\s*:",
        flags=re.IGNORECASE,
    )

    stop_fields = (
        "Differentiation",
        "Challenges Dominant Assumption",
        "Challenges Assumption",
        "Challenge Dominant Assumption",
        "Challenge Assumption",
        "Preserves Cognitive Diversity",
        "Cognitive Diversity Preservation",
        "Supporting Paper",
        "Speculative Extension",
        "Potential Outcome",
        "Persona",
        "Evidence",
        "Rationale",
        "Method",
        "Evaluation",
    )    

    #stop_pattern = "|".join(re.escape(field) for field in stop_fields)

    # Case: Arm B/C
    description_marker = re.search(
        r"\bDescription\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )
    if description_marker: 
        title = text[:description_marker.start()].strip()
        title = re.sub(
            r"^\s*Title\s*:\s*",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
        remainder = text[description_marker.end():].strip()
        stop_match = stop_pattern.search(remainder)
        if stop_match: description = remainder[:stop_match.start()].strip()
        else: description = remainder
        if title and description: return re.sub(r"\s+", " ", f"{title}: {description}").strip()
        if description: return description
        return title

    # Case: Arm A
    if ":" in text:
        heading, body = text.split(":", 1)
        heading = heading.strip()
        body = body.strip()

        body = re.sub(
            r"\s*\([^)]*"
            r"(potential homogenization|potential assumption|"
            r"high-risk|risk:|likely focuses)"
            r"[^)]*\)\s*$",
            "",
            body,
            flags=re.IGNORECASE,
        ).strip()
        # body = re.split(
        #     rf"\s+(?:{stop_pattern})\s*:",
        #     body,
        #     maxsplit=1,
        #     flags=re.IGNORECASE,
        # )[0]

        if 2<= len(heading.split()) <= 18: return f"{heading}: {body}"

        # if 2<= len(heading.split()) <= 16:
        #     # Keep heading plus first 1-2 sentences of body
        #     sentences = re.split(r"(?<=[.!?])\s+", body)
        #     body_summary = " ".join(sentences[:2]).strip()
        #     core = f"{heading}: {body_summary}"
        #     return re.sub(r"\s+", " ", core).strip()

    # fallback
    stop_match = stop_pattern.search(text)
    if stop_match: text = text[:stop_match.start()].strip()
    return re.sub(r"\s+", " ", text).strip()

    # # Case 1: structured format with Title and Description
    # title_match = re.search(
    #     #rf"\bTitle:\s*(.*?)(?=\s+Description:|\s+(?:{stop_pattern})\s*:|$)",
    #     r"\bTitle:\s*(.*?)(?=\s+Description:|$)",
    #     text,
    #     flags=re.IGNORECASE,
    # )

    # desc_match = re.search(
    #     rf"\bDescription:\s*(.*?)(?=\s+(?:{stop_pattern})\s*:|$)",
    #     text,
    #     flags=re.IGNORECASE,
    # )

    # if desc_match: 
    #     description = desc_match.group(1).strip()
    #     if title_match:
    #         title = title_match.group(1).strip()
    #         core = f"{title}: {description}"
    #     else: core = description
    #     return re.sub(r"\s+", " ", core).strip()

    # # parts = []

    # # if title_match: parts.append(title_match.group(1).strip())
    # # if desc_match: parts.append(desc_match.group(1).strip())
    # # if parts:
    # #     core = " ".join(parts)
    # #     return re.sub(r"\s+", " ", core).strip()
    
    # # Case 2: naive A format "Idea title: research direction: ..."
    # research_match = re.search(
    #     rf"\bResearch Direction:\s*(.*?)(?=\s+(?:{stop_pattern})\s*:|$)",
    #     text,
    #     flags=re.IGNORECASE,
    # )
    # if research_match:
    #     description = research_match.group(1).strip()
    #     heading_match = re.match(
    #         r"^(.*?):\s*Research Direction:",
    #         text,
    #         flags=re.IGNORECASE,
    #     )
    #     if heading_match:
    #         title = heading_match.group(1).strip()
    #         core = f"{title}: {description}"
    #     else: core = description
    #     return re.sub(r"\s+", " ", core).strip()

    # # Case 3: generic "Heading: body"
    # if ":" in text:
    #     heading, body = text.split(":", 1)
    #     heading = heading.strip()
    #     body = body.strip()

    #     body = re.split(
    #         rf"\s+(?:{stop_pattern})\s*:",
    #         body,
    #         maxsplit=1,
    #         flags=re.IGNORECASE,
    #     )[0]

    #     if 2<= len(heading.split()) <= 16:
    #         # Keep heading plus first 1-2 sentences of body
    #         sentences = re.split(r"(?<=[.!?])\s+", body)
    #         body_summary = " ".join(sentences[:2]).strip()
    #         core = f"{heading}: {body_summary}"
    #         return re.sub(r"\s+", " ", core).strip()
    
    # # Case 4: fallback to first 2 sentences
    # text = re.split(
    #     rf"\s+(?:{stop_pattern}\s*:)",
    #     text,
    #     maxsplit=1,
    #     flags=re.IGNORECASE,
    # )[0]
    # sentences = re.split(r"(?<=[.!?])\s+", text)
    # core = " ".join(sentences[:2]).strip()

    # return re.sub(r"\s+", " ", core).strip()

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

def compute_embedding_spread_metrics(texts):
    """
    Compute embedding-based diversity metrics for a list of text units.
    Used by both full ideas and core concepts.
    """
    if len(texts) == 0:
        return {
            "vendi_score": 0,
            "mean_pairwise_distance": 0,
        }
    
    if len(texts) == 1:
        return {
            "vendi_score": 1,
            "mean_pairwise_distance": 0,
        }
    
    model = get_embedding_model()
    X = model.encode(texts)

    vendi_score = float(vendi.score_X(X))

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Xn = X / norms

    similarity_matrix = Xn @ Xn.T
    upper_triangle = np.triu_indices(len(texts), 1)

    mean_pairwise_similarity = similarity_matrix[upper_triangle].mean()
    mean_pairwise_distance = float(1 - mean_pairwise_similarity)

    return {
        "vendi_score": round(vendi_score, 3),
        "mean_pairwise_distance": round(mean_pairwise_distance, 3),
    }


def compute_embedding_diversity_metrics(ideas):
    """
    Computes reference-free, non-LLM diversity metrics for a set of ideas.
    Includes:
    - full-text embedding diversity
    - core-concept embedding diversity
    - lexical diversity diagnostics
    """
    if len(ideas) == 0:
        return {
            "idea_count": 0,
            "vendi_score": 0,
            "mean_pairwise_distance": 0,
            "core_concept_vendi_score": 0,
            "core_concept_mean_pairwise_distance": 0,
            "distinct_1": 0,
            "distinct_2": 0,
            "self_bleu": 0,
        }
    
    core_concepts = [
        extract_core_concept_text(idea) for idea in ideas
    ]
    # remove accidental empty core concepts by falling back to original idea
    core_concepts = [
        core if core else idea
        for core, idea in zip(core_concepts, ideas)
    ]

    full_embedding_metrics = compute_embedding_spread_metrics(ideas)
    core_embedding_metrics = compute_embedding_spread_metrics(core_concepts)

    distinct_1 = compute_distinct_n(ideas, n=1)
    distinct_2 = compute_distinct_n(ideas, n=2)
    self_bleu = compute_self_bleu(ideas)

    return {
        "idea_count": len(ideas),
        # full idea text embedding metrics
        "vendi_score": full_embedding_metrics["vendi_score"],
        "mean_pairwise_distance": full_embedding_metrics["mean_pairwise_distance"],
        # core concept embedding metrics
        "core_concept_vendi_score": core_embedding_metrics["vendi_score"],
        "core_concept_mean_pairwise_distance": core_embedding_metrics["mean_pairwise_distance"],
        # lexical diagnostics
        "distinct_1": round(distinct_1, 3),
        "distinct_2": round(distinct_2, 3),
        "self_bleu": round(self_bleu, 3),
    }

def evaluate_idea_set_diversity(output_text):
    ideas = split_ideas(output_text)
    metrics = compute_embedding_diversity_metrics(ideas)
    core_concepts = [
        extract_core_concept_text(idea)
        for idea in ideas
    ]
    core_concepts = [
        core if core else idea
        for core, idea in zip(core_concepts, ideas)
    ]
    return {
        "ideas": ideas,
        "metrics": metrics,
        "core_concepts": core_concepts,
    }