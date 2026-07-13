import re
import json
import numpy as np
from util import extract_json
from llm_backend import call_llm
from prompts_evaluator import (
    evaluate_with_llm_prompt,
    evaluate_cognitive_diversity_prompt,
)
from assumption_bank import format_assumption_bank_for_prompt

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
    prompt = evaluate_with_llm_prompt(topic, bank_text, ideas, response_text)
    
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

        return averaged
    
    except Exception as e:
        return {
            "novelty": 0,
            "diversity": 0,
            "usefulness": 0,
            "assumption_challenge": 0,
            "summary": f"Evaluation parsing failed: {e}. Raw output: {raw_result[:500]}",
        }    

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

def evaluate_output(topic, response_text, ideas=None, assumption_bank=None, backend="local_ollama", model=None):
    """
    Combined evaluation
    1: LLM as a judge
    2: Simple non LLM diagnostic metrics
    """
    llm_scores = evaluate_with_llm(
        topic=topic,
        response_text=response_text,
        ideas=ideas,
        assumption_bank=assumption_bank,
        backend=backend,
        model=model
    )

    simple_metrics = compute_simple_metrics(response_text)

    return {
        "llm_scores": llm_scores,
        "simple_metrics": simple_metrics
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