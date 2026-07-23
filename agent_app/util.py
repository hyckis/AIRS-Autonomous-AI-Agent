import json
import re

# sometimes json output contains “”; ‘’ rather than "" that makes parsing failed
def normalize_quotes(text):
    return (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

def extract_json(text):
    text = text.strip()
    text = normalize_quotes(text)

    # case 1: ```json
    text = re.sub(r"^```json\s", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # case 2: direct parsing
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # case 3: extract first json obj inside the txt
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    
    raise ValueError("No json obj found")


def extract_idea_title(idea_text, max_words=14):
    """
    Extract a short idea title from an idea block.
    Works with:
    - Title: ...
    - 1. Title: ...
    - Heading: description
    - fallback first N words
    """
    if not idea_text: return ""

    text = re.sub(r"\s+", " ", idea_text).strip()

    # case 1: title:... description:
    m = re.search(
        r"Title:\s*(.*?)(?:\s+Description:|\s+Description:|\s+Challenges|\s+Cognitive Diversity|\s+Supporting Paper:|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m: return m.group(1).strip()

    # case 2: heading: description
    if ":" in text: 
        prefix = text.split(":", 1)[0].strip()
        if 2 <= len(prefix.split()) <= max_words: return prefix
    
    # case 3: fallback
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def normalize_text_for_match(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_assumption_id(value, valid_assumption_ids):
    """
    Normalize assumption id from llm output
    accepts: 1, "1", "Assumption 1", null/None
    """
    if value is None: return None

    if isinstance(value, int): assumption_id = value
    else:
        match = re.search(r"\d+", str(value))
        if not match: return None
        assumption_id = int(match.group(0))
    if assumption_id in valid_assumption_ids: return assumption_id

    return None

def infer_assumption_id_from_text(challenged_assumption, assumption_by_id):
    """
    Infer assumption id from the challenged assumption text when the llm forgets to return assumption id
    """
    if not challenged_assumption: 
        print("DEBUG: not challenged assumption")
        return None

    raw = str(challenged_assumption).strip()
    challenged = normalize_text_for_match(raw)
    print("DEBUG: challenged_assumption: ", challenged)
    if not challenged: 
        print("DEBUG: no challenged")
        return None

    if "no specific assumption" in challenged: 
        print("DEBUG: no specific assumption")
        return None

    # Case 1: "Assumption 3" / "ID 3" / "3" 
    id_match = re.search(r"\b(?:assumption|id)?\s*(\d+)\b", raw, flags=re.IGNORECASE) 
    if id_match: 
        candidate_id = int(id_match.group(1)) 
        if candidate_id in assumption_by_id: return candidate_id 

    # Case 2: exact / containment match 
    for assumption_id, item in assumption_by_id.items(): 
        assumption_text = normalize_text_for_match(item.get("assumption", "")) 
        if not assumption_text: continue 
        if challenged == assumption_text: return assumption_id 
        if assumption_text in challenged or challenged in assumption_text: return assumption_id 

    # Case 3: token overlap fallback 
    challenged_tokens = set(challenged.split()) 
    best_id = None 
    best_overlap = 0 

    for assumption_id, item in assumption_by_id.items(): 
        assumption_text = normalize_text_for_match(item.get("assumption", "")) 

        assumption_tokens = set(assumption_text.split()) 
        if not assumption_tokens: continue 

        overlap = len(challenged_tokens & assumption_tokens) / len(assumption_tokens) 
        if overlap > best_overlap: 
            best_overlap = overlap 
            best_id = assumption_id 

    # threshold can be adjusted 
    if best_overlap >= 0.6: return best_id 

    return None


def extract_challenged_assumption_from_idea(idea_text):
    """
    Extract challenges dominant assumption field from a structured idea block
    """
    if not idea_text: return None
    text = re.sub(r"\s+", " ", str(idea_text)).strip()
    match = re.search(
        r"Challenges?\s+(?:Dominant\s+)?Assumption\s*:\s*"
        r"(.*?)(?=\s+(?:Preserves|Cognitive Diversity|Suppporting|Support|Speculative|Evidence|Rationale|Method|Evaluation|Title|Description)\s*:|$)",
        text, flags=re.IGNORECASE,
    )
    if match: return match.group(1).strip()

    return None

def normalize_idea_scores(llm_scores, ideas, assumption_bank):
    """
    Add idea_title and normalize assumption_id / challenged_assumption.
    """

    if ideas is None: ideas = []
    if assumption_bank is None: assumption_bank = []

    raw_idea_scores = llm_scores.get("idea_scores", [])

    # print("DEBUG assumption_bank: ", assumption_bank)
    print("DEBUG assumption_bank type: ", type(assumption_bank))

    assumption_by_id = {
        int(item.get("id", i)): item
        for i, item in enumerate(assumption_bank, start=1) if isinstance(item, dict)
    }

    valid_assumption_ids = set(assumption_by_id.keys())
    normalized_scores = []

    print("DEBUG assumption by id: ", assumption_by_id)
    print("DEBUG valid assumption ids: ", valid_assumption_ids)
    print("DEBUG raw first idea score: ", raw_idea_scores[0] if raw_idea_scores else None)

    for i, idea in enumerate(ideas, start=1):
        # try to align by position
        item = raw_idea_scores[i - 1] if i - 1 < len(raw_idea_scores) else {}
        if not isinstance(item, dict): item = {}
        idea_title = item.get("idea_title") or extract_idea_title(idea)

        assumption_id = normalize_assumption_id(
            item.get("assumption_id"), valid_assumption_ids,
        )
        # fallback 1: infer from evaluator's challenged assumption
        if assumption_id is None:
            assumption_id = infer_assumption_id_from_text(
                item.get("challenged_assumption"), assumption_by_id,
            )

        # fallback 2: infer from the original idea block's challenges dominant assumption field
        idea_declared_assumption = extract_challenged_assumption_from_idea(idea)

        if assumption_id is None: 
            assumption_id = infer_assumption_id_from_text(idea_declared_assumption, assumption_by_id)
            challenged_assumption = "No specific assumption directly addressed"
        else:
            challenged_assumption = assumption_by_id[assumption_id].get(
                "assumption", item.get("challenged_assumption", "") or idea_declared_assumption or "",
            )

        normalized_scores.append({
            "idea_index": item.get("idea_index", i),
            "idea_title": idea_title,
            "assumption_id": assumption_id,
            "challenged_assumption": challenged_assumption,
            "rationale": item.get("rationale", ""),
            "novelty": item.get("novelty", 0),
            "diversity": item.get("diversity", 0),
            "usefulness": item.get("usefulness", 0),
            "assumption_challenge": item.get("assumption_challenge", 0),
        })

        print("DEBUG: idea declared assumption: ", idea_declared_assumption)
        print("DEBUG: evaluator challenged assumption: ", item.get("challenged_assumption"))
        print("DEBUG: inferred assumption id: ", assumption_id)

    llm_scores["idea_scores"] = normalized_scores
    llm_scores["raw_idea_score_count"] = len(raw_idea_scores)
    llm_scores["normalized_idea_score_count"] = len(normalized_scores)

    return llm_scores