"""
run_builder.py — the adapter.

Shapes the existing pipeline's outputs (the objects app.py already puts in
st.session_state) into a single `run` dict matching run.schema.json, which
render_airs.render() turns into the Divergence Studio page.

It computes nothing new: arms come from the *_diversity["ideas"] the pipeline
already parsed, Vendi/crosschecks from *_diversity["metrics"], assumptions from
the shared assumption_bank, evidence from the literature-grounded metrics, and
the evaluation summary / per-idea judge, debate and pairwise details straight
out of the *_eval objects. The only work here is *shaping* — pulling fields out
of the free-text ideas and eval dicts, with safe fallbacks so a messy or partial
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
    # Drop a leading preamble that ends in a colon and reads like a lead-in, e.g.
    # "Given the identified homogenization..., I'd like to ask: <the ask>?"
    if ":" in question:
        before, _, after = question.rpartition(":")
        after = after.strip()
        cues = ("ask", "question", "following", "pose", "wonder", "consider",
                "propose", "raise", "like to", "want to", "here is", "here's",
                "i'd", "i would", "let me", "given", "my ")
        if (after.endswith("?") and len(after.split()) >= 3
                and any(c in before.lower() for c in cues)):
            question = after
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


# ---------------------------------------------------------------- eval shaping

def _num_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _idea_scores_by_index(eval_obj):
    """{idea_index: idea_score dict} from an *_eval object's llm_scores."""
    ls = (eval_obj or {}).get("llm_scores", {}) or {}
    out = {}
    for it in ls.get("idea_scores", []) or []:
        if isinstance(it, dict) and it.get("idea_index") is not None:
            out[it.get("idea_index")] = it
    return out


def _judge_summary_for(eval_obj):
    """Arm-level averaged judge scores (1-5) already computed in llm_scores."""
    ls = (eval_obj or {}).get("llm_scores", {}) or {}
    return {m: _num_or_none(ls.get(m)) for m in
            ("novelty", "diversity", "usefulness", "assumption_challenge")}


def _lit_summary_for(lit_obj):
    """Arm-level averaged literature-grounding numbers."""
    lit = lit_obj or {}
    return {
        "avg_novelty": _num_or_none(lit.get("average_novelty")),
        "avg_evidence_count": _num_or_none(lit.get("average_evidence_count")),
        "avg_evidence_ratio": _num_or_none(lit.get("average_evidence_ratio")),
    }


def _direction_eval(score, lit_item):
    """Per-idea eval block attached to one Arm C direction."""
    if not score and not lit_item:
        return None
    ev = {}
    if score:
        ev["judge"] = {
            "novelty": _num_or_none(score.get("novelty")),
            "diversity": _num_or_none(score.get("diversity")),
            "usefulness": _num_or_none(score.get("usefulness")),
            # debate-adjusted score when the idea was debated, else the raw judge score
            "assumption_challenge": _num_or_none(
                score.get("assumption_challenge_final",
                          score.get("assumption_challenge"))),
            "rationale": _clean(score.get("rationale", "")),
        }
        ev["debated"] = bool(score.get("debate_selected"))
        rank = score.get("usefulness_rank")
        ev["usefulness_rank"] = rank if isinstance(rank, int) else None
    if lit_item:
        ev["literature"] = {
            "novelty": _num_or_none(lit_item.get("novelty")),
            "evidence_count": lit_item.get("evidence_count", 0),
            "evidence_ratio": _num_or_none(lit_item.get("evidence_ratio")),
            "closest_paper": lit_item.get("closest_paper"),
            "supporting_sources": lit_item.get("supporting_sources", []) or [],
        }
    return ev


def _debate_top(eval_obj):
    """The debated (top-k) ideas -> transcript rows, ranked by debate outcome."""
    ls = (eval_obj or {}).get("llm_scores", {}) or {}
    debated = [it for it in ls.get("idea_scores", []) or []
               if isinstance(it, dict) and it.get("debate_selected")]
    debated.sort(key=lambda x: x.get("assumption_challenge_rank") or 999)
    out = []
    for it in debated:
        out.append({
            "idea_title": it.get("idea_title") or _short_title(it.get("idea_text", "")),
            "decision": _clean(it.get("debate_decision", "")),
            "llm_score": _num_or_none(it.get("assumption_challenge_llm")),
            "final_score": _num_or_none(it.get("assumption_challenge_final")),
            "confidence": _num_or_none(it.get("confidence")),
            "advocate_argument": _clean(it.get("advocate_argument", "")),
            "skeptic_argument": _clean(it.get("skeptic_argument", "")),
        })
    return out


def _pairwise(eval_obj, max_comparisons=6):
    """Pairwise usefulness tournament -> top-3 ranking + sample comparisons."""
    ls = (eval_obj or {}).get("llm_scores", {}) or {}
    top = []
    for it in ls.get("usefulness_pairwise_top3", []) or []:
        top.append({
            "idea_title": it.get("idea_title") or _short_title(it.get("idea_text", "")),
            "rank": it.get("usefulness_rank"),
            "score": _num_or_none(it.get("usefulness_pairwise")),
            "win_rate": _num_or_none(it.get("usefulness_win_rate")),
            "wins": it.get("pairwise_wins", 0),
            "losses": it.get("pairwise_losses", 0),
            "ties": it.get("pairwise_ties", 0),
        })

    title_by_index = {it.get("idea_index"): (it.get("idea_title") or "")
                      for it in ls.get("idea_scores", []) or []}
    comps = []
    for c in (ls.get("usefulness_pairwise_comparisons", []) or []):
        if len(comps) >= max_comparisons:
            break
        winner = c.get("winner")
        at = title_by_index.get(c.get("idea_a_index"), "")
        bt = title_by_index.get(c.get("idea_b_index"), "")
        if winner == "A":
            a, b = at, bt
        elif winner == "B":
            a, b = bt, at
        else:
            continue  # skip ties in the sample list
        if not a or not b:
            continue
        comps.append({"a": a, "b": b, "reason": _clean(c.get("reason", ""))})
    return {"top": top, "comparisons": comps}


def _generation_context(get):
    """Run-level metadata for the bottom panel, from run_params + literature +
    the parsed idea sets. Everything degrades to a safe default if absent."""
    literature = get("literature") or {}
    ideas_by_arm = get("ideas_by_arm") or {}
    params = get("run_params") or {}
    papers = literature.get("papers", []) or []
    return {
        "backend": params.get("backend", "local_ollama"),
        "model": params.get("model"),
        "params": {
            "paper_limit": params.get("paper_limit"),
            "use_llm_queries": params.get("use_llm_queries"),
            "run_debate": params.get("run_debate"),
        },
        "arms": [
            {"key": "A", "label": "Naive LLM", "temperature": 0.8},
            {"key": "B", "label": "Strong Prompt", "temperature": 0.8},
            {"key": "C", "label": "Lens Agent (assumption-critique + 8 lenses)", "temperature": 0.8},
        ],
        "literature": {
            "query_source": literature.get("query_source", ""),
            "paper_count": len(papers),
            "queries": literature.get("queries", []) or [],
        },
        "idea_counts": {
            "A": len(ideas_by_arm.get("A: Naive LLM", []) or []),
            "B": len(ideas_by_arm.get("B: Strong Prompt", []) or []),
            "C": len(ideas_by_arm.get("C: Lens Agent", []) or []),
        },
    }


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


def _c_directions(diversity, literature_metrics, eval_obj=None):
    """Arm C: structured {title, lens, oneline, detail, breaks, evidence, eval}."""
    ideas = (diversity or {}).get("ideas", []) or []
    per_idea = (literature_metrics or {}).get("per_idea", []) or []
    scores_by_idx = _idea_scores_by_index(eval_obj)
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
        # idea_index in the eval output is 1-based and aligned to this same list.
        score = scores_by_idx.get(i + 1)
        out.append({
            "title": _short_title(idea),
            "lens": _detect_lens(idea),
            "oneline": oneline,
            "detail": detail,
            "breaks": breaks,
            "evidence": _detect_evidence(idea, lit_item),
            "eval": _direction_eval(score, lit_item),
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
    baseline_lit = get("baseline_literature_metrics") or {}
    strong_lit = get("strong_literature_metrics") or {}
    expanded_lit = get("expanded_literature_metrics") or {}
    baseline_eval = get("baseline_eval") or {}
    strong_eval = get("strong_eval") or {}
    expanded_eval = get("expanded_eval") or {}

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
                "directions": _c_directions(expanded_div, expanded_lit, expanded_eval),
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
        "evaluation": {
            "judge_summary": {
                "A": _judge_summary_for(baseline_eval),
                "B": _judge_summary_for(strong_eval),
                "C": _judge_summary_for(expanded_eval),
            },
            "literature_summary": {
                "A": _lit_summary_for(baseline_lit),
                "B": _lit_summary_for(strong_lit),
                "C": _lit_summary_for(expanded_lit),
            },
            "debate": {"top": _debate_top(expanded_eval)},
            "pairwise": _pairwise(expanded_eval),
        },
        "generation_context": _generation_context(get),
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
