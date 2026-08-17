"""
run_builder.py — the adapter.

Shapes the existing pipeline's outputs (the objects app.py already puts in
st.session_state) into a single `run` dict matching run.schema.json, which
render_airs.render() turns into the Divergence Studio page.

It computes nothing new: arms come from the *_diversity["ideas"] the pipeline
already parsed, Vendi/crosschecks from *_diversity["metrics"], assumptions from
the shared assumption_bank, evidence from the literature-grounded metrics. The
only work here is *shaping* — pulling title/oneline/detail/lens/breaks/evidence
out of the free-text ideas the agents produced, with safe fallbacks so a messy
LLM response degrades gracefully instead of raising.
"""
import os
import re
import sys
from datetime import datetime, timezone

# Make the agent_app package dir importable whether this runs under Streamlit
# (cwd = agent_app) or standalone (cwd = agent_app/ui).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assumption_bank import (  # noqa: E402
    extract_challenged_assumption_from_idea,
)

# The eight lenses the diversity_expander_prompt asks for. Order matters: the
# first keyword group that matches the idea text wins, so put the more specific
# phrases first.
LENS_KEYWORDS = [
    ("Anti-efficiency", ["anti-efficiency", "anti efficiency", "efficiency", "scarcity", "surplus"]),
    ("Marginalized users", ["marginalized", "overlooked", "underrepresented", "underserved", "food-insecure", "excluded"]),
    ("Historical analogy", ["historical analogy", "historical", "precedent", "analogy", "history"]),
    ("Failure-mode", ["failure-mode", "failure mode", "failure", "unintended", "misuse", "backfire", "shaming"]),
    ("Cultural difference", ["cultural difference", "cross-cultural", "cultural", "culture"]),
    ("Counter-mainstream", ["counter-mainstream", "counter mainstream", "contrarian", "against the grain"]),
    ("Collective intelligence", ["collective intelligence", "commons", "self-govern", "crowd", "grassroots", "guild"]),
    ("Long-term risk", ["long-term", "long term", "speculative risk", "future risk", "structural"]),
]

STOP_FIELDS = (
    "Description", "Short description", "Differentiation",
    "Challenges Dominant Assumption", "Challenges Assumption", "Challenges",
    "Preserves Cognitive Diversity", "Cognitive Diversity Preservation", "Preserves",
    "Supporting Paper", "Supporting Papers", "Supporting", "Speculative Extension",
    "Evidence", "Rationale", "Method", "Evaluation", "Why", "Title",
)
_STOP_RE = "|".join(re.escape(f) for f in STOP_FIELDS)


def _clean(text):
    text = re.sub(r"\*+", "", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _field(idea, *labels):
    """Pull the value of a labelled field like 'Description: ...' up to the next field."""
    for label in labels:
        m = re.search(
            rf"\b{re.escape(label)}\s*:\s*(.*?)(?=\s+(?:{_STOP_RE})\s*:|$)",
            idea, flags=re.IGNORECASE,
        )
        if m and m.group(1).strip():
            return _clean(m.group(1))
    return ""


def _first_sentence(text, max_chars=150):
    text = _clean(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    first = parts[0].strip()
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0] + "…"
    return first


def _sentence_split(text):
    """Split prose into (first_sentence, rest) so a card's one-liner and its
    expand body don't repeat the same opening sentence."""
    text = _clean(text)
    if not text:
        return "", ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    first = parts[0].strip()
    rest = " ".join(p.strip() for p in parts[1:]).strip()
    return first, rest


def _split_question(text):
    """Split a human-in-the-loop prompt into (question, rationale). Keeps the
    question itself terse; everything after it becomes optional rationale."""
    text = _clean(text)
    if not text:
        return "", ""
    idx = text.find("?")
    if idx == -1:
        return _sentence_split(text)  # no question mark: first sentence is the ask
    question = text[:idx + 1].strip()
    rationale = text[idx + 1:].strip()
    # Drop a short leading preamble like "A strategic question: <the ask>?"
    if ":" in question:
        pre, after = question.split(":", 1)
        if "?" in after and len(pre.split()) <= 8:
            question = after.strip()
    return question, rationale


# Splits an idea at the first structured field label, so the text before it can
# be recovered as a title. gemma3 often runs the Title / Description / Challenges
# / Preserves sections together in one block with no explicit "Title:" prefix.
_FIELD_SPLIT_RE = re.compile(
    r"\s+(?:Short\s+description|Description|Challeng\w*|Preserv\w*|"
    r"Cognitive\s+Diversity|Supporting|Speculativ\w*|Differentiation|"
    r"Evidence|Rationale|Method|Evaluation)\b\s*:?",
    re.IGNORECASE,
)


def _short_title(idea, max_words=12):
    """A short, punchy card title.

    Cut the idea at the first field label, then: if the remaining head reads
    "Catchy Name: explanatory subtitle", keep just the catchy name; otherwise
    cap the length. This stops the whole description from bleeding into the
    title when the model doesn't use a clean "Title:" prefix.
    """
    head = _FIELD_SPLIT_RE.split(_clean(idea), maxsplit=1)[0]
    head = re.sub(r"^\s*\d+[\).\s]+", "", head)          # leading "1." / "2)"
    head = re.sub(r"^\s*Title\s*[:\-–—]\s*", "", head, flags=re.IGNORECASE)  # leading "Title:"
    head = head.strip(" -–—:*\"'")
    if not head:
        return _first_sentence(idea, max_chars=80)
    if ":" in head:
        name = head.split(":", 1)[0].strip(" \"'")
        if 2 <= len(name.split()) <= 9:
            return name
    words = head.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(",;:-") + "…"
    return head


def _detect_lens(idea):
    low = idea.lower()
    for lens, keywords in LENS_KEYWORDS:
        if any(k in low for k in keywords):
            return lens
    return ""


def _detect_evidence(idea, lit_item):
    """speculative vs paper. URLs are stripped from idea text, so paper details
    come from the literature-grounded metrics computed on the same idea."""
    if "speculative" in idea.lower():
        return {"type": "speculative"}
    if lit_item:
        closest = lit_item.get("closest_paper") or {}
        if closest.get("title") and lit_item.get("evidence_count", 0) > 0:
            return {
                "type": "paper",
                "citation": closest.get("title", ""),
                "url": closest.get("url", ""),
            }
    return {"type": "speculative"}


def _baseline_ideas(diversity):
    """Arm A / Arm B: flat {title, detail}."""
    ideas = (diversity or {}).get("ideas", []) or []
    out = []
    for idea in ideas:
        clean = _clean(idea)
        title = _short_title(idea, max_words=9)
        detail = _field(idea, "Description", "Short description")
        if not detail:
            parts = _FIELD_SPLIT_RE.split(clean, maxsplit=1)
            if len(parts) > 1:
                detail = _clean(parts[1])
            elif ":" in clean:
                detail = clean.split(":", 1)[1].strip()
            elif clean != title:
                detail = clean
        out.append({"title": title, "detail": detail or ""})
    return out


def _c_directions(diversity, literature_metrics):
    """Arm C: structured {title, lens, oneline, detail, breaks, evidence}."""
    ideas = (diversity or {}).get("ideas", []) or []
    per_idea = (literature_metrics or {}).get("per_idea", []) or []
    out = []
    for i, idea in enumerate(ideas):
        lit_item = per_idea[i] if i < len(per_idea) else None
        clean = _clean(idea)
        parts = _FIELD_SPLIT_RE.split(clean, maxsplit=1)
        remainder = _clean(parts[1]) if len(parts) > 1 else ""
        description = _field(idea, "Description", "Short description") or remainder or clean
        # one-liner = first sentence (always shown); detail = the rest (on expand),
        # so the expand never repeats the sentence already visible above it.
        oneline, detail = _sentence_split(description)
        breaks = _field(
            idea,
            "Challenges Dominant Assumption", "Challenges Assumption",
            "Challenged Assumption", "Challenges",
        ) or _clean(extract_challenged_assumption_from_idea(idea) or "")
        if "no specific assumption" in breaks.lower():
            breaks = ""
        out.append({
            "title": _short_title(idea),
            "lens": _detect_lens(idea),
            "oneline": oneline,
            "detail": detail,
            "breaks": breaks,
            "evidence": _detect_evidence(idea, lit_item),
        })
    return out


def _fmt(value, is_bleu=False):
    """Format a metric for the crosscheck row. Self-BLEU comes back on a 0-100
    scale from sacrebleu, so normalise it to 0-1 for display consistency."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if is_bleu:
        v = v / 100.0
    return f"{v:.2f}"


def _metric(diversity, key, default=0):
    return (diversity or {}).get("metrics", {}).get(key, default)


def build_run(session_state, topic):
    """Build the run dict from st.session_state-like mapping and the topic string.

    `session_state` only needs to support .get(); a plain dict works too, which
    keeps this testable without Streamlit.
    """
    get = session_state.get

    baseline_div = get("baseline_diversity") or {}
    strong_div = get("strong_diversity") or {}
    expanded_div = get("expanded_diversity") or {}
    assumption_bank = get("assumption_bank") or []
    expanded_lit = get("expanded_literature_metrics") or {}

    discussion_question, discussion_rationale = _split_question(get("human_question") or "")

    run = {
        "topic": topic,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discussion_prompt": discussion_question,
        "discussion_rationale": discussion_rationale,
        "arms": {
            "A": {
                "label": "Naive baseline",
                "sublabel": "Arm A",
                "ideas": _baseline_ideas(baseline_div),
            },
            "B": {
                "label": "Strong ChatGPT baseline",
                "sublabel": "Arm B",
                "ideas": _baseline_ideas(strong_div),
            },
            "C": {
                "label": "Divergent directions",
                "sublabel": "Assumption-critique + 8 cognitive lenses - Arm C",
                "directions": _c_directions(expanded_div, expanded_lit),
            },
        },
        "assumptions": [
            {
                "assumed": _clean(item.get("assumption", "")),
                "reframe": _clean(item.get("challenge_criteria", "")),
                "lens": "",
            }
            for item in assumption_bank
            if isinstance(item, dict) and item.get("assumption")
        ][:3],
        "metrics": {
            "vendi": {
                "A": _metric(baseline_div, "vendi_score"),
                "B": _metric(strong_div, "vendi_score"),
                "C": _metric(expanded_div, "vendi_score"),
            },
            "crosschecks": [
                {
                    "name": "Mean pairwise dist",
                    "c": _fmt(_metric(expanded_div, "mean_pairwise_distance")),
                    "b": _fmt(_metric(strong_div, "mean_pairwise_distance")),
                    "better": "higher",
                },
                {
                    "name": "Distinct-2",
                    "c": _fmt(_metric(expanded_div, "distinct_2")),
                    "b": _fmt(_metric(strong_div, "distinct_2")),
                    "better": "higher",
                },
                {
                    "name": "Self-BLEU",
                    "c": _fmt(_metric(expanded_div, "self_bleu"), is_bleu=True),
                    "b": _fmt(_metric(strong_div, "self_bleu"), is_bleu=True),
                    "better": "lower",
                },
            ],
        },
        "human_eval": {
            "ratings": {
                "A": get("human_baseline_diversity") or 0,
                "B": get("human_strong_diversity") or 0,
                "C": get("human_agent_diversity") or 0,
            },
            "note": "Class ratings are captured with the sliders in the detailed "
                    "view. Where humans and Vendi disagree is the interesting "
                    "discussion.",
        },
    }
    return run
