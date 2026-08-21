# used as a parser for json and output text
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

FIELD_NAMES = [
    "Description", "Research Direction", "Focus", "Rationale",
    "Metrics", "Potential Homogenization Risk", "Differentiation",
    "Challenges", "Challenge", "Challenged Assumption", "Challenges Assumption", "Challenges Dominant Assumption",
    "Preserves Cognitive Diversity", "Cognitive Diversity Preservation", 
    "Supporting Paper", "Supporting Papers", "Speculative Extension",
    "Key Metrics", "Closet Paper", "Novelty", 
    "Evidence Count", "Evidence Ratio", "Pairwise Win Rate",
]

FIELD_RE = "|".join(re.escape(x) for x in FIELD_NAMES)

def normalize_text(text):
    if text is None: return ""
    text = str(text)

    # normalize common markdown/unicode artifacts
    text = text.replace("\u00a0", " ") 
    text = text.replace("\\*", "*") 
    text = text.replace("–", "–") 
    text = text.replace("—", "—")

    # remove markdown links but keep link text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_markdown(text, max_len=None):
    text = normalize_text(text)

    # remove markdown wrappers
    text = re.sub(r"^\s*[-*]\s+", "", text) 
    text = text.replace("**", "") 
    text = text.replace("__", "") 
    text = text.replace("*", "") 
    text = text.replace("`", "") 
    # Remove dangling field labels 
    text = re.sub(rf"\b(?:{FIELD_RE})\s*:\s*$", "", text, flags=re.IGNORECASE) 
    # Normalize spaces 
    text = re.sub(r"\s+", " ", text).strip(" -:;|\n\t")

    if max_len is not None and len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text

def remove_outro(text):
    """
    Remove common llm outro text that sometimes gets attached to the last idea
    """
    text = normalize_text(text)
    outro_patterns = [
        r"\n\s*---\s*\n\s*How does this.*$", 
        r"\n\s*How does this.*$", 
        r"\n\s*Do you want me.*$", 
        r"\n\s*Would you like me.*$", 
        r"\n\s*Let me know.*$",
    ]

    for pattern in outro_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()

def split_ideas(response_text):
    """
    Split model output into idea blocks
    """
    text = remove_outro(response_text)
    if not text: return []

    # Normalize section headings so they become split points 
    text = re.sub( 
        r"(?m)^\s*\*\*\s*Research Direction\s+(\d+)\s*:\s*\*\*\s*$", 
        r"\n@@IDEA_SPLIT@@\1. ", 
        text, 
        flags=re.IGNORECASE, 
    ) 
    text = re.sub( 
        r"(?m)^\s*Research Direction\s+(\d+)\s*:\s*$", 
        r"\n@@IDEA_SPLIT@@\1. ", 
        text, 
        flags=re.IGNORECASE, 
    ) 
    # Normalize numbered list starts 
    text = re.sub( 
        r"(?m)^\s*(?:\*\*)?\s*(\d+)\s*[\.\)]\s+(?=(?:\*\*)?\s*(?:Title\s*:|[A-Z\"“]))", 
        r"\n@@IDEA_SPLIT@@\1. ", 
        text, 
    )
    # If model put next idea on same line, insert split before "2. Title:" 
    text = re.sub( 
        r"(?<!\n)(\s+)(\d+\.\s*(?:\*\*)?\s*Title\s*:)", 
        r"\n@@IDEA_SPLIT@@\2", 
        text, 
        flags=re.IGNORECASE, 
    ) 

    chunks = [c.strip() for c in text.split("@@IDEA_SPLIT@@")]
    ideas = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk: continue
        if not re.match(r"^\d+\s*[\.\)]", chunk): continue
        ideas.append(chunk)

    return ideas

def extract_field(idea_text, field_names):
    """
    Extract content after a metadata field until the next metadata field
    """
    text = normalize_text(idea_text)
    if isinstance(field_names, str): field_names = [field_names]
    field_alt = "|".join(re.escape(x) for x in field_names)
    match = re.search(
        rf"(?:^|\n|\s|-)\s*(?:\*\*)?\s*(?:{field_alt})\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*" 
        rf"(.+?)" 
        rf"(?=\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:|\n\s*\n|$)", 
        text, 
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match: return ""
    return clean_markdown(match.group(1))

def extract_idea_title(idea_text, max_len=160):
    """
    Extract clean title only
    """
    text = normalize_text(idea_text)
    # remove leading number
    text = re.sub(r"^\s*\d+\s*[\.\)]\s", "", text).strip()

    # case 1: explicit title field
    title = extract_field(text, "Title")
    if title: return clean_markdown(title, max_len=max_len)

    # case 2: numbered markdown title
    match = re.search(
        rf"^\s*(?:\*\*)?\s*(.+?)(?:\*\*)?\s*:\s*(?=(?:\*\*)?\s*(?:Focus|Research Direction|Description|Rationale)\s*:|\*)", 
        text, 
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match: return clean_markdown(match.group(1), max_len=max_len)

    # case 3: first line fallback, cut before known fields
    first_line = text.split("\n")[0]
    first_line = re.split(
        rf"\s+(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:",
        first_line,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    return clean_markdown(first_line, max_len=max_len)


def extract_idea_description(idea_text, max_len=700):
    """
    Extract description-like body
    """
    text = normalize_text(idea_text)

    for field in ["Description", "Focus", "Research Direction"]:
        value = extract_field(text, field)
        if value: return clean_markdown(value, max_len=max_len)

    title = extract_idea_title(text, max_len=None)

    # remove leading number and title from fallback
    body = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text).strip() 
    if title: body = body.replace(title, "", 1) 

    # Cut before metadata fields that should not be part of description 
    body = re.split( 
        rf"\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:Challenges?|Challenges Dominant Assumption|Cognitive Diversity|Cognitive Diversity Preservation|Preserves Cognitive Diversity|Supporting Papers?|Differentiation|Metrics|Potential Homogenization Risk)\s*(?:\*\*)?\s*:", 
        body, 
        maxsplit=1, 
        flags=re.IGNORECASE, 
    )[0]

    return clean_markdown(body, max_len=max_len)

def extract_core_concept(idea_text, max_len=900):
    """
    Core concept = clean title + description-like content
    this should be used for core concept vendi
    """
    title = extract_idea_title(idea_text, max_len=180)
    description = extract_idea_description(idea_text, max_len=max_len)

    if title and description:
        if description.lower().startswith(title.lower()): return clean_markdown(description, max_len=max_len)
        return clean_markdown(f"{title}: {description}", max_len=max_len)

    if title: return clean_markdown(title, max_len=180)

    return clean_markdown(idea_text, max_len=max_len)

def short_title(title, max_len=90): return clean_markdown