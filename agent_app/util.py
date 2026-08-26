# used as a parser for json and output text
import json
import re
import html

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

# FIELD_NAMES = [
#     "Description", "Research Direction", "Focus", "Rationale",
#     "Metrics", "Potential Homogenization Risk", "Differentiation",
#     "Challenges", "Challenge", "Challenged Assumption", "Challenges Assumption", "Challenges Dominant Assumption",
#     "Preserves Cognitive Diversity", "Cognitive Diversity Preservation", 
#     "Supporting Paper", "Supporting Papers", "Speculative Extension",
#     "Key Metrics", "Closet Paper", "Novelty", 
#     "Evidence Count", "Evidence Ratio", "Pairwise Win Rate",
# ]

# FIELD_RE = "|".join(re.escape(x) for x in FIELD_NAMES)

TITLE_FIELDS = [ "Title", ] 
DESCRIPTION_FIELDS = [ "Description", "Research Direction", "Focus", ] 
CHALLENGE_FIELDS = [ "Challenges Dominant Assumption", "Challenges Assumption", "Challenged Assumption", "Challenges", "Challenge", ] 
DIVERSITY_FIELDS = [ "Preserves Cognitive Diversity", "Cognitive Diversity Preservation", "Cognitive Diversity", ] 
SUPPORT_FIELDS = [ "Supporting Papers", "Supporting Paper", "Closest Paper", "Closet Paper",  ] # keep typo for backward compatibility 
SPECULATIVE_FIELDS = [ "Speculative Extension", ] 
META_FIELDS = [ "Differentiation", "Metrics", "Key Metrics", "Key Focus", "Potential Homogenization Concern", "Potential Homogenization Risk", "Evidence Count", "Evidence Ratio", "Pairwise Win Rate", "Novelty", ] 
ALL_FIELDS = ( TITLE_FIELDS + DESCRIPTION_FIELDS + CHALLENGE_FIELDS + DIVERSITY_FIELDS + SUPPORT_FIELDS + SPECULATIVE_FIELDS + META_FIELDS ) 
# Important: longer fields first. 
# # Example: "Supporting Papers" should be matched before "Supporting Paper". 
ALL_FIELDS = sorted(set(ALL_FIELDS), key=len, reverse=True) 
FIELD_RE = "|".join(re.escape(x) for x in ALL_FIELDS)


def normalize_text(text):
    if text is None: return ""
    text = str(text)
    text = html.unescape(text)

    # normalize common markdown/unicode artifacts
    text = text.replace("\u00a0", " ") 
    text = text.replace("\\*", "*") 
    text = text.replace("–", "–") 
    text = text.replace("—", "—")

    # remove markdown links but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_core_concept_text(text, max_len=None):
    """ 
    Final cleaner for titles and core concepts. 
    This should be the final gate before: - displaying title - displaying core concept - computing Core Concept Vendi 
    """ 
    if text is None: return "" 
    text = normalize_text(text) 
    text = text.replace("\\*", "*") 
    # Remove markdown wrappers. 
    text = text.replace("**", "") 
    text = text.replace("__", "") 
    text = text.replace("*", "") 
    text = text.replace("`", "") 
    # Remove leading "1. Title:" / "2. Description:" / etc. 
    text = re.sub( r"^\s*\d+\s*[\.\)]\s*(?:Title|Description|Focus|Research Direction)\s*:\s*", "", text, flags=re.IGNORECASE, ) 
    # Remove plain leading numbering.
    text = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text) 
    # Remove leading bullet. 
    text = re.sub(r"^\s*[-•]\s*", "", text) 
    # Remove leading field label. 
    text = re.sub( r"^\s*(?:Title|Description|Focus|Research Direction)\s*:\s*", "", text, flags=re.IGNORECASE, ) 
    # Remove quote wrappers. 
    text = text.strip("“”\"' ") 
    # Remove metadata that should not enter embedding. 
    text = re.sub( r"\s*Key\s+(?:Focus|Metrics)\s*:\s*.*$", "", text, flags=re.IGNORECASE | re.DOTALL, ) 
    text = re.sub( r"\s*\(Focus\s*:\s*.*?\)\s*$", "", text, flags=re.IGNORECASE | re.DOTALL, ) 
    text = re.sub( r"\s*\((?:Technical Researcher Focus|Policy/Ethics focus)\)\s*$", "", text, flags=re.IGNORECASE | re.DOTALL, ) 
    # Remove dangling field labels. 
    text = re.sub( rf"\b(?:{FIELD_RE})\s*:\s*$", "", text, flags=re.IGNORECASE, ) 
    text = re.sub(r"\s+", " ", text).strip(" -:;|\n\t") 

    if max_len is not None and len(text) > max_len: text = text[:max_len].rstrip() + "..." 
    return text

def clean_markdown(text, max_len=None):    
    return clean_core_concept_text(text, max_len=max_len)

def remove_meta_notes(text): 
    text = normalize_text(text) 
    meta_patterns = [ 
        r"\s*Potential Homogenization Concern\s*:\s*.*$", 
        r"\s*Potential Homogenization Risk\s*:\s*.*$", 
        r"\s*Key\s+(?:Focus|Metrics)\s*:\s*.*$", 
        r"\s*\(Likely LLM Focus\s*:\s*.*?\)", 
        r"\s*\(Potential homogenization\s*:\s*.*?\)", 
        r"\s*\(Potential assumption\s*:\s*.*?\)", 
        r"\s*\(Focus\s*:\s*.*?\)",
    ] 
    for pattern in meta_patterns: text = re.sub( pattern, "", text, flags=re.IGNORECASE | re.DOTALL, ) 
    return clean_markdown(text)

def remove_outro(text):
    """
    Remove common llm outro text that sometimes gets attached to the last idea
    """
    text = normalize_text(text)
    outro_patterns = [
        r"\n\s*---\s*\n\s*\*\*Important Note:\*\*.*$", 
        r"\n\s*\*\*Important Note:\*\*.*$", 
        r"\n\s*Important Note:.*$", 
        # Common LLM follow-up prompts 
        r"\n\s*To help me refine.*$", 
        r"\n\s*To provide a more targeted critique.*$", 
        r"\n\s*How do these alternative directions.*$", 
        r"\n\s*Does this revised set of directions.*$", 
        r"\n\s*Do you want me to.*$", 
        r"\n\s*Do you wish me to.*$",
        r"\n\s*Would you like me to.*$", 
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
        chunk = remove_outro(chunk)
        ideas.append(chunk)

    return ideas

def normalize_field_boundaries(text): 
    """ 
    Arm C often returns fields inline: 
    **Title:** ... **Description:** ... **Challenges Assumption:** ... 
    This function inserts newlines before known fields so extract_field() can parse both inline and multiline formats. 
    """ 
    text = normalize_text(text) 
    text = text.replace("\\*", "*") 
    # Handle weird escaped/broken title markers. 
    text = re.sub(r"Title:\*\*", "Title:", text, flags=re.IGNORECASE) 
    text = re.sub(r"Title\s*:\s*\*\*", "Title:", text, flags=re.IGNORECASE) 
    # Insert newline before known fields when they appear inline. 
    text = re.sub( rf"\s+(?=(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:)", "\n", text, flags=re.IGNORECASE, ) 
    return text.strip()

def extract_field(idea_text, field_names):
    """
    Extract content after a metadata field until the next metadata field
    """
    text = normalize_field_boundaries(idea_text)
    if isinstance(field_names, str): field_names = [field_names]
    field_names = sorted(field_names, key=len, reverse=True)
    field_alt = "|".join(re.escape(x) for x in field_names)
    pattern = ( 
        rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*" 
        rf"(?:{field_alt})" 
        rf"\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*" 
        rf"(.+?)" 
        rf"(?=\n\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:|\Z)" 
    ) 
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match: return ""
    return clean_core_concept_text(match.group(1))

def parse_idea_block(idea_text): 
    """
    Parse one idea block into a standardized dict. 
    This is the single source of truth for: - title - description - challenge - diversity rationale - supporting paper - speculative extension - raw cleaned idea block 
    """ 
    raw = remove_outro(idea_text) 
    text = normalize_field_boundaries(raw) 
    title = extract_field(text, TITLE_FIELDS) 
    description = extract_field(text, DESCRIPTION_FIELDS) 
    # Fallback for A-arm style: 
    # # 1. **Title:** body... 
    # # 1. **Title:** "body..." (Focus: ...) 
    if not title: 
        match = re.match( 
            r"^\s*\d+\s*[\.\)]\s*(?:\*\*)?\s*(.+?)(?:\*\*)?\s*:\s*(.+)$", 
            text, flags=re.IGNORECASE | re.DOTALL, 
        ) 
        if match: 
            title = clean_core_concept_text(match.group(1)) 
            body = match.group(2) 

            # Cut body before any metadata field. 
            body = re.split( 
                rf"\n?\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:", 
                body, maxsplit=1, flags=re.IGNORECASE, 
            )[0] 
            description = clean_core_concept_text(body) 

    # Fallback: explicit title exists but description not found. 
    # # This can happen if the model emits: 
    # # 1. **Title:** Some Title **Description:** ... 
    # # but the boundary normalization missed something. 
    if title and not description: 
        body = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text).strip()
        # Remove explicit Title field if present. 
        body = re.sub( 
            r"^(?:\*\*)?\s*Title\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*", "", 
            body, flags=re.IGNORECASE, 
        ) 
        # Remove title string once. 
        if title: body = body.replace(title, "", 1) 
        # Cut before any metadata field. 
        body = re.split( 
            rf"\n?\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:", 
            body, maxsplit=1, flags=re.IGNORECASE, 
        )[0] 
        description = clean_core_concept_text(body) 

    challenge = extract_field(text, CHALLENGE_FIELDS) 
    diversity = extract_field(text, DIVERSITY_FIELDS) 
    support = extract_field(text, SUPPORT_FIELDS) 
    speculative = extract_field(text, SPECULATIVE_FIELDS) 

    return { 
        "title": clean_core_concept_text(title), 
        "description": clean_core_concept_text(description), 
        "challenge": clean_core_concept_text(challenge), 
        "diversity": clean_core_concept_text(diversity), 
        "support": clean_core_concept_text(support), 
        "speculative": clean_core_concept_text(speculative), 
        "raw": raw, 
    } 


def extract_idea_title(idea_text, max_len=160):
    """
    Extract clean title only
    """
    parsed = parse_idea_block(idea_text)
    title = parsed.get("title", "")
    if title: return clean_core_concept_text(title, max_len=max_len)

    text = normalize_text(idea_text)
    first_line = text.split("\n")[0]
    # remove leading number
    first_line = re.sub(r"^\s*\d+\s*[\.\)]\s", "", first_line)

    first_line = re.split(
        rf"\s+(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:",
        first_line, maxsplit=1, flags=re.IGNORECASE,
    )[0]

    return clean_core_concept_text(first_line, max_len=max_len)

    # # case 1: explicit title field
    # title = extract_field(text, "Title")
    # if title: return clean_markdown(title, max_len=max_len)

    # # case 2: numbered markdown title
    # match = re.search(
    #     rf"^\s*(?:\*\*)?\s*(.+?)(?:\*\*)?\s*:\s*(?=(?:\*\*)?\s*(?:Focus|Research Direction|Description|Rationale)\s*:|\*)", 
    #     text, 
    #     flags=re.IGNORECASE | re.DOTALL,
    # )
    # if match: return clean_markdown(match.group(1), max_len=max_len)

    # # case 3: first line fallback, cut before known fields
    # first_line = text.split("\n")[0]
    # first_line = re.split(
    #     rf"\s+(?:\*\*)?\s*(?:{FIELD_RE})\s*(?:\*\*)?\s*:",
    #     first_line,
    #     maxsplit=1,
    #     flags=re.IGNORECASE,
    # )[0]

    # return clean_markdown(first_line, max_len=max_len)


def extract_idea_description(idea_text, max_len=700):
    """
    Extract description-like body
    """
    parsed = parse_idea_block(idea_text)
    description = parsed.get("description", "")
    description = remove_meta_notes(description)
    return clean_core_concept_text(description, max_len=max_len)
    # text = normalize_text(idea_text)

    # for field in ["Description", "Focus", "Research Direction"]:
    #     value = extract_field(text, field)
    #     if value: return clean_markdown(value, max_len=max_len)

    # title = extract_idea_title(text, max_len=None)

    # # remove leading number and title from fallback
    # body = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text).strip() 
    # if title: body = body.replace(title, "", 1) 

    # # Cut before metadata fields that should not be part of description 
    # body = re.split( 
    #     rf"\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:Challenges?|Challenges Dominant Assumption|Cognitive Diversity|Cognitive Diversity Preservation|Preserves Cognitive Diversity|Supporting Papers?|Differentiation|Metrics|Potential Homogenization Risk)\s*(?:\*\*)?\s*:", 
    #     body, 
    #     maxsplit=1, 
    #     flags=re.IGNORECASE, 
    # )[0]

    # return clean_markdown(body, max_len=max_len)

def extract_core_concept(idea_text, max_len=900):
    """
    Core concept = clean title + description-like content
    this should be used for core concept vendi
    """
    parsed = parse_idea_block(idea_text)
    title = parsed.get("title", "")
    description = parsed.get("description", "")
    # title = extract_idea_title(idea_text, max_len=180)
    # description = extract_idea_description(idea_text, max_len=max_len)

    if title and description:
        if description.lower().startswith(title.lower()): core = description
        else: core = f"{title}: {description}"
        # core = remove_meta_notes(core)
        # return clean_markdown(core, max_len=max_len)
    elif title: core = title
    else: core = parsed.get("raw", "")
    
    core = remove_meta_notes(core)
    return clean_core_concept_text(core, max_len=max_len)


def short_title(title, max_len=90): return clean_markdown