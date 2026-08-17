import re
import json
import numpy as np
from util import extract_json
from llm_backend import call_llm
from prompts_evaluator import (
    evaluate_with_llm_prompt,
    evaluate_single_idea_prompt,
    evaluate_cognitive_diversity_prompt,
)
from assumption_bank import (
    format_assumption_bank_for_prompt,
    extract_idea_title,
    normalize_idea_scores,
)
from pairwise_tournament import run_usefulness_pairwise
from debate_evaluator import apply_assumption_debate_to_scores

SCORE_METRICS = [
    "novelty",
    "diversity",
    "usefulness",
    "assumption_challenge"
]

def validate_score_dict(score):
    """
    All score keys must exist and be numeric(1-5)
    """
    validated = {}

    for key in SCORE_METRICS:
        value = score.get(key, 0)
        try: value = float(value)
        except(TypeError, ValueError): value = 0
        if value > 0: 
            value = max(1, min(5, value))
        validated[key] = value
    
    validated["summary"] = score.get("summary", "")
    return validated

def evaluate_with_llm(topic, response_text, ideas=None, assumption_bank=None, backend="local_ollama", model=None):
    """
    G-Eval inspired per-idea LLM rubric evaluation.
    Scores each idea separately, then averages scores in Python.
    """
    if ideas is None: ideas = [response_text] if response_text else []
    bank_text = format_assumption_bank_for_prompt(assumption_bank)
    idea_payloads = []
    for i, idea in enumerate(ideas, start=1):
        idea_payloads.append({
            "idea_index": i,
            "idea_title": extract_idea_title(idea),
            "idea_text": idea,
        })
    prompt = evaluate_with_llm_prompt(topic, bank_text, idea_payloads)
    
    raw_result = call_llm(
        prompt,
        backend = backend,
        model = model,
        temperature = 0.1
    )

    try:
        parsed = extract_json(raw_result)
        idea_scores = parsed.get("idea_scores", [])
        cleaned_idea_scores = []
        
        for item in idea_scores:
            cleaned = {
                "idea_index": item.get("idea_index", len(cleaned_idea_scores) + 1),
                "idea_title": item.get("idea_title", ""),
                "assumption_id": item.get("assumption_id", None),
                #"default_assumption": item.get("default_assumption", ""),
                "challenged_assumption": item.get("challenged_assumption", ""),
                "rationale": item.get("rationale", ""),
            }
            for metric in SCORE_METRICS:
                value = item.get(metric, 0)
                try: value = float(value)
                except (TypeError, ValueError): value = 0
                if value > 0: value = max(1, min(5, value))
                cleaned[metric] = value

            cleaned_idea_scores.append(cleaned)

        if not cleaned_idea_scores: raise ValueError("No valid idea_scores returned")

        averaged = {
            metric: round(float(np.mean([item[metric] for item in cleaned_idea_scores])), 3)
            for metric in SCORE_METRICS
        }
        # averaged = {}
        # for metric in SCORE_METRICS:
        #     averaged[metric] = round(
        #         float(np.mean([item[metric] for item in cleaned_idea_scores])),
        #         3
        #     )
        averaged["summary"] = parsed.get("summary", "")
        averaged["idea_scores"] = cleaned_idea_scores
        averaged["parse_failed"] = False

        return averaged
    
    except Exception as e:
        return {
            "novelty": None,
            "diversity": None,
            "usefulness": None,
            "assumption_challenge": None,
            "summary": f"Evaluation parsing failed: {e}. Raw output: {raw_result[:500]}",
            "idea_scores": [],
            "parse_failed": True,
            "raw_output": raw_result[:3000] if raw_result else "",
        }    

def evaluate_with_llm_idea_batch(topic, response_text, ideas=None, assumption_bank=None, backend="local_ollama", model=None):
    """
    First try batch evaluation.
    If batch parsing fails, fall back to evaluating one idea at a time
    using evaluate_with_llm 
    """
    # 1. try original batch judge
    batch_result = evaluate_with_llm(
        topic=topic,
        response_text=response_text,
        ideas=ideas,
        assumption_bank=assumption_bank,
        backend=backend,
        model=model,
    )
    if not batch_result.get("parse_failed"):
        batch_result["evaluation_mode"] = ""
        return batch_result
    print("Batch LLM judge failed, falling back to per-idea judge")

    # 2. fallback: call evaluate_with_llm once per idea
    per_idea_scores = []
    for i, idea in enumerate(ideas, start=1):
        single_result = evaluate_with_llm(
            topic=topic,
            response_text=response_text,
            ideas=[idea],
            assumption_bank=assumption_bank,
            backend=backend,
            model=model,
        )
        if single_result.get("parse_failed"):
            per_idea_scores.append({
                "idea_index": i,
                "idea_title": extract_idea_title(idea),
                "assumption_id": None,
                "challenged_assumption": "Evaluation parsing failed",
                "rationale": single_result.get("summary", ""),
                "novelty": None,
                "diversity": None,
                "usefulness": None,
                "assumption_challenge": None,
                "parse_failed": True,
            })
            continue

        idea_scores = single_result.get("idea_scores", [])
        if idea_scores:
            item = idea_scores[0]
            item["idea_index"] = i
            item["idea_title"] = item.get("idea_title") or extract_idea_title(idea)
            item["parse_failed"] = False
            per_idea_scores.append(item)
        else:
            per_idea_scores.append({
                "idea_index": i,
                "idea_title": extract_idea_title(idea),
                "assumption_id": None,
                "challenged_assumption": "No idea score returned",
                "rationale": single_result.get("summary", ""),
                "novelty": None,
                "diversity": None,
                "usefulness": None,
                "assumption_challenge": None,
                "parse_failed": True,
            })
    
    # 3. average per-idea scores
    averaged = {}
    for metric in SCORE_METRICS:
        values = []
        for item in per_idea_scores:
            value = item.get(metric, None)
            if value is None: continue
            try: values.append(float(value))
            except Exception: pass
        averaged[metric] = round(sum(values) / len(values), 3) if values else None
    
    parse_failed_count = sum(
        1 for item in per_idea_scores
        if item.get("parse_failed")
    )

    averaged["summary"] = (
        f"Batch judge failed; used per-idea fallback. "
        f"Parse failures: {parse_failed_count}/{len(per_idea_scores)}."
    )
    averaged["idea_scores"] = per_idea_scores
    averaged["parse_failed"] = parse_failed_count == len(per_idea_scores)
    averaged["parse_failed_count"] = parse_failed_count
    averaged["evaluation_mode"] = "per_idea_fallback"

    return averaged


def compute_simple_metrics(response_text):
    words = response_text.split()
    word_cnt = len(words)

    diversity_lens_keywords = {
        "contrarian": ["contrarian", "opposite", "challenge", "counter", "alternative"],
        "historical": ["historical", "history", "past", "analogy", "precedent"],
        "cross_disciplinary": ["cross-disciplinary", "interdisciplinary", "biology", "sociology", "psychology", "economics"],
        "failure_mode": ["failure", "risk", "unintended", "breakdown", "misuse"],
        "cultural": ["cultural", "culture", "linguistic", "local", "context"],
        "stakeholder": ["stakeholder", "underserved", "marginalized", "student", "teacher", "community"],
        "long_term": ["long-term", "future", "scaling", "institutional", "societal"],
    }

    lower = response_text.lower()
    lens_coverage = {}
    for lens, keywords in diversity_lens_keywords.items():
        lens_coverage[lens] = any(keyword in lower for keyword in keywords)
    lens_coverage_cnt = sum(lens_coverage.values())

    return {
        "word_count": word_cnt,
        "lens_coverage_count": lens_coverage_cnt,
        "lens_coverage": lens_coverage,
    }

def evaluate_output(
        topic, 
        response_text, 
        ideas=None, 
        assumption_bank=None, 
        backend="local_ollama", 
        model=None,
        run_debate=False
    ):
    """
    Combined evaluation
    1: LLM as a judge
    2. Pairwise usefulness tournament
    3: Simple non LLM diagnostic metrics
    """
    if ideas is None: ideas = []
    if assumption_bank is None: assumption_bank = []

    # LLM as a judge evaluation
    #llm_scores = evaluate_with_llm(
    llm_scores = evaluate_with_llm_idea_batch(
        topic=topic,
        response_text=response_text,
        ideas=ideas,
        assumption_bank=assumption_bank,
        backend=backend,
        model=model,
    )
    print("DEBUG after evaluate with llm: ")
    print(llm_scores)

    # Add idea_title and normalize assumption_id/challenged_assumption
    llm_scores = normalize_idea_scores(
        llm_scores=llm_scores,
        ideas=ideas,
        assumption_bank=assumption_bank,
    )

    # multi agent debate for assumption challenge
    if run_debate: 
        debate_result = apply_assumption_debate_to_scores(
            topic=topic,
            idea_scores=llm_scores.get("idea_scores", []),
            ideas=ideas,
            assumption_bank=assumption_bank,
            backend=backend,
            model=model,
            top_k=3,
        )
        llm_scores["idea_scores"] = debate_result.get(
            "idea_scores",
            llm_scores.get("idea_scores", []),
        )
        llm_scores["unmatched_assumptions"] = debate_result.get(
            "unmatched_assumptions", [],
        )
    else:
        llm_scores["assumption_debate_top3"] = []
        llm_scores["unmatched_assumptions"] = []

    # Pairwise tournament - usefulness
    pairwise_result = run_usefulness_pairwise(
        topic=topic,
        ideas=ideas,
        backend=backend,
        model=model,
    )
    pairwise_scores = pairwise_result.get(
        "idea_scores", [],
    )
    ranked_pairwise = pairwise_result.get(
        "ranked_idea_scores", None,
    )
    if ranked_pairwise is None:
        ranked_pairwise = sorted(
            pairwise_scores,
            key=lambda x: (
                x.get("usefulness_win_rate", 0),
                x.get("pairwise_wins", 0),
                -x.get("pairwise_losses", 0),
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked_pairwise, start=1): item["usefulness_rank"] = rank

    title_by_index = {
        item.get("idea_index"): item.get("idea_title", "")
        for item in llm_scores.get("idea_scores", [])
    }
    for item in ranked_pairwise:
        idx = item.get("idea_index")
        item["idea_title"] = title_by_index.get(idx, "")
        if (isinstance(idx, int) and 1 <= idx <= len(ideas)): item["idea_text"] = ideas[idx-1]
        else: item["idea_text"] = ""

    pairwise_by_index = {
        item["idea_index"]: item
        for item in ranked_pairwise #pairwise_scores["idea_scores"]
    }

    for item in llm_scores.get("idea_scores", []):
        idx = item.get("idea_index")
        pairwise_item = pairwise_by_index.get(idx)
        if not pairwise_item: continue

        item["usefulness_pairwise"] = pairwise_item.get("usefulness_pairwise", None)
        item["usefulness_win_rate"] = pairwise_item.get("usefulness_win_rate", None)
        item["usefulness_rank"] = pairwise_item.get("usefulness_rank", None)
        item["pairwise_wins"] = pairwise_item.get("pairwise_wins", None)
        item["pairwise_losses"] = pairwise_item.get("pairwise_losses", None)
        item["pairwise_ties"] = pairwise_item.get("pairwise_ties", None)

        # if pairwise_item:
        #     # keep original llm absolute usefulness score
        #     item["usefulness_llm"] = item.get("usefulness", 0)
        #     # add pairwise fields
        #     item["usefulness_pairwise"] = pairwise_item["usefulness_pairwise"]
        #     item["usefulness_win_rate"] = pairwise_item["usefulness_win_rate"]
        #     # replace original usefulness score with pairwise score
        #     item["usefulness"] = pairwise_item["usefulness_pairwise"]

    llm_scores["usefulness_pairwise_top3"] = ranked_pairwise[:3]
    llm_scores["usefulness_pairwise_ranked"] = ranked_pairwise
    llm_scores["usefulness_pairwise_comparisons"] = pairwise_result.get(
        "comparisons", [],
    )

    # Recompute average scores from normalized per-idea scores
    idea_scores = llm_scores.get("idea_scores", [])

    for metric in SCORE_METRICS: 
        values = []
        for item in idea_scores:
            value = item.get(metric, None)
            if value is None: continue
            try: values.append(float(value))
            #try: values.append(float(item.get(metric, 0)))
            except Exception: pass
        llm_scores[metric] = round(sum(values) / len(values), 3) if values else None

    # simple non llm diagnostics
    simple_metrics = compute_simple_metrics(response_text)

    # print("DEBUG: ideas len = ", len(ideas))
    # print("DEBUG: raw idea scores len = ", len(llm_scores.get("idea_scores", [])))
    # print("DEBUG: raw idea scores = ", llm_scores.get("idea_scores", []))
    # print("DEBUG normalized idea scores len = ", len(llm_scores.get("idea_scores", [])))

    return {
        "llm_scores": llm_scores,
        "simple_metrics": simple_metrics,
    }


# used as backward wrapper
def evaluate_cognitive_diversity(topic, response_text):
    prompt = evaluate_cognitive_diversity_prompt(topic, response_text)
    result = call_llm(prompt, "local_ollama", temperature=0.2)

    try: return extract_json(result)
        #return json.loads(result)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "novelty": 0,
            "diversity": 0,
            "usefulness": 0,
            "assumption_challenge": 0,
            "summary": f"Evaluation parsing failed: {e}. Raw output: {result[:500]}"
        }

def audit_evaluation_results(eval_df, assumption_bank, expected_idea_count=None):
    """
    Basic sanity checks for per-idea LLM judge outputs.
    Returns a list of warnings.
    """
    warnings = []

    if eval_df is None or eval_df.empty: return ["Evaluation result is empty"]

    # 1. idea count check
    idea_count = len(eval_df)
    if expected_idea_count is not None and idea_count != expected_idea_count: 
        warnings.append(
            f"Idea count mismatch: expected {expected_idea_count}, got {idea_count}"
        )

    # 2. assumption bank usage check
    bank_assumptions = [
        item.get("assumption", "").strip().lower()
        for item in assumption_bank
        if isinstance(item, dict) and item.get("assumption")
    ]
    for _, row in eval_df.iterrows():
        idx = row.get("idea_index", "unknown")
        challenged = str(row.get("challenged_assumption", "")).strip()
        challenged_lower = challenged.lower()
        score = row.get("assumption_challenge", None)
        rationale = str(row.get("rationale", "")).lower()

        try: score = float(row.get("assumption_challenge", 0))
        except Exception: score = 0

        # 3. no assumption but score too high
        if "no specific assumption" in challenged_lower and score >= 3:
            warnings.append(
                f"Idea {idx}: says no specific assumption is addressed,"
                f"but assumption_challenge score is {score}."
            )

        # 4. weak rationale but high score
        weak_words = ["implicitly", "partially", "doesn't directly", 
                      "does not directly", "indirectly", "not directly",]

        if score >= 4 and any(word in rationale for word in weak_words):
            warnings.append(
                f"Idea {idx}: rationale sounds weak/indirect, "
                f"but assumption_challenge score is high({score})."
            )

        # 5. check if challenged assumption seems connected to bank
        if (challenged and "no specific assumption" not in challenged_lower and bank_assumptions):
            matched = any(
                bank_assumption in challenged_lower
                or challenged_lower in bank_assumption
                for bank_assumption in bank_assumptions
            )
            # allow labels like "Assumption 1"
            has_assumption_number = "assumption " in challenged_lower

            if not matched and not has_assumption_number:
                warnings.append(
                    f"Idea {idx}: challenged assumption may not come from the assumption bank: "
                    f"{challenged}"
                )
    
    return warnings


def extract_json_old(text):
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )
    if match:
        return json.loads(
            match.group()
        )
    raise ValueError("No JSON file found")