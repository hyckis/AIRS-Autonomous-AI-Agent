from llm_backend import call_llm
from util import extract_json
from prompts_evaluator import (
    advocate_prompt,
    skeptic_prompt,
    judge_prompt,
)
from assumption_bank import normalize_assumption_id

def clamp_score(value, low=1, high=5):
    try: value = float(value)
    except Exception: return 0
    if value <= 0: return 0
    return max(low, min(high, value))

def get_assumption_by_id(assumption_bank):
    if assumption_bank is None: assumption_bank = []
    return {
        int(item.get("id", i)): item
        for i, item in enumerate(assumption_bank, start=1)
        if isinstance(item, dict)
    }

def assumption_challenge_debate(
    topic,
    idea_text,
    assumption_item,
    backend="local_ollama",
    model=None,
):
    """
    Multi-agent debate for assumption_challenge only.

    Returns a dict with:
    - assumption_challenge_debate
    - advocate_argument
    - skeptic_argument
    - debate_decision
    - confidence
    """
    if not assumption_item: 
        return {
            "assumption_challenge_debate": 1,
            "debate_decision": "No matched assumption was available for debate.",
            "advocate_argument": "",
            "skeptic_argument": "",
            "confidence": 0,
            "debate_parse_failed": False,
        }
    
    try: 
        advocate_raw = call_llm(
            advocate_prompt(topic, idea_text, assumption_item),
            backend=backend,
            model=model,
            temperature=0.2,
        )
        advocate_result = extract_json(advocate_raw)

        skeptic_raw = call_llm(
            skeptic_prompt(
                topic=topic, 
                idea_text=idea_text,
                assumption_item=assumption_item,
                advocate_argument=advocate_result.get("advocate_argument", ""),
            ),
            backend=backend,
            model=model,
            temperature=0.2,
        )
        skeptic_result = extract_json(skeptic_raw)

        judge_raw = call_llm(
            judge_prompt(
                topic=topic,
                idea_text=idea_text,
                assumption_item=assumption_item,
                advocate_result=advocate_result,
                skeptic_result=skeptic_result,
            ),
            backend=backend,
            model=model,
            temperature=0.1,
        )
        judge_result = extract_json(judge_raw)

        final_score = clamp_score(
            judge_result.get("assumption_challenge", 0)
        )

        return {
            "assumption_challenge_debate": final_score,
            "debate_decision": judge_result.get("final_decision", ""),
            "advocate_argument": advocate_result.get("advocate_argument", ""),
            "skeptic_argument": skeptic_result.get("skeptic_argument", ""),
            "advocate_suggested_score": clamp_score(
                advocate_result.get("suggested_score", 0)
            ),
            "skeptic_suggested_score": clamp_score(
                skeptic_result.get("suggested_score", 0)
            ),
            "confidence": judge_result.get("confidence", 0),
            "debate_parse_failed": False,
        }
    
    except Exception as e:
        return {
            "assumption_challenge_debate": 0,
            "debate_decision": f"Debate parsing failed: {e}",
            "advocate_argument": "",
            "skeptic_argument": "",
            "confidence": 0,
            "debate_parse_failed": True,
        } 

def apply_assumption_debate_to_scores(
    topic,
    idea_scores,
    ideas,
    assumption_bank,
    backend="local_ollama",
    model=None,
):
    """
    Apply multi-agent debate to each idea's assumption_challenge score.

    This preserves the original LLM judge score as assumption_challenge_llm,
    then replaces assumption_challenge with the debate score when available.
    """
    assumption_by_id = get_assumption_by_id(assumption_bank)
    updated_scores = []

    for i, item in enumerate(idea_scores, start=1):
        if not isinstance(item, dict): item = {}
        idea_index = item.get("idea_index", i)

        try: idea_index_int = int(idea_index)
        except Exception: idea_index_int = i

        idea_text = ideas[idea_index_int - 1] if 0 <= idea_index_int - 1 < len(ideas) else ""

        assumption_id = normalize_assumption_id(
            item.get("assumption_id"),
            set(assumption_by_id.keys())
        )
        assumption_item = assumption_by_id.get(assumption_id)

        item["assumption_challenge_llm"] = item.get("assumption_challenge", 0)

        if assumption_id is None or assumption_item is None:
            item["assumption_challenge_debate"] = 1
            item["assumption_challenge"] = min(
                float(item.get("assumption_challenge_llm", 0) or 0),
                2,
            )
            item["debate_decision"] = "No matched assumption_id; debate skipped."
            item["debate_parse_failed"] = False
            updated_scores.append(item)
            continue
        
        debate_result = assumption_challenge_debate(
            topic=topic,
            idea_text=idea_text,
            assumption_item=assumption_item,
            backend=backend,
            model=model,
        )

        if not debate_result.get("debate_parse_failed"):
            item["assumption_challenge"] = debate_result.get(
                "assumption_challenge_debate",
                item.get("assumption_challenge_llm", 0),
            )

        updated_scores.append(item)
        item.update(debate_result)
    
    return updated_scores