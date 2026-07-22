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
        except json.JSONDecodeError: "No json obj found"
    
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

def normalize_idea_scores(llm_scores, ideas, assumption_bank):
    """
    Add idea_title and normalize assumption_id / challenged_assumption.
    """
    if ideas is None: ideas = []
    if assumption_bank is None: assumption_bank = []

    raw_idea_scores = llm_scores.get("idea_scores", [])
    normalized_scores = []
    valid_assumption_ids = set(range(1, len(assumption_bank) + 1))

    for i, idea in enumerate(ideas, start=1):
        # try to align by position
        item = raw_idea_scores[i - 1] if i - 1 < len(raw_idea_scores) else {}
        if not isinstance(item, dict): item = {}
        idea_title = item.get("idea_title") or extract_idea_title(idea)

        try: assumption_id = int(assumption_id) if assumption_id is not None else None
        except Exception: assumption_id = None

        if assumption_id not in valid_assumption_ids: assumption_id = None
        if assumption_id is None: 
            challenged_assumption = item.get(
                "challenged_assumption",
                "No specific assumption directly addressed",
            )
            if not challenged_assumption:
                challenged_assumption = "No specific assumption directly addressed"
        else:
            challenged_assumption = item.get(
                "challenged_assumption",
                "No specific assumption directly addressed",
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

        llm_scores["idea_scores"] = normalized_scores
        return llm_scores