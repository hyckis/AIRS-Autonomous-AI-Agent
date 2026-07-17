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