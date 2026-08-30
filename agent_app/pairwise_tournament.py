import itertools
import json
import re
from llm_backend import call_llm
from util import extract_json
from prompts_evaluator import pairwise_usefulness_prompt

def run_usefulness_pairwise(
        topic, ideas, backend="local_ollama", model=None, temperature=0.1
):
    """
    Pairwise tournament for usefulness.
    Returns per-idea win rates and 1-5 usefulness scores.
    """
    n = len(ideas)
    if n == 0: return []
    if n == 1:
        return [{
            "idea_index": 1,
            "pairwise_wins": 0,
            "pairwise_losses": 0,
            "pairwise_ties": 0,
            "pairwise_matches": 0,
            "usefulness_win_rate": 1.0,
            "usefulness_pairwise": 5.0,
        }]
    
    records = {
        i: {
            "idea_index": i + 1,
            "pairwise_wins": 0,
            "pairwise_losses": 0,
            "pairwise_ties": 0,
            "pairwise_matches": 0,
        } for i in range(n)
    }

    comparisons = []

    for i, j in itertools.combinations(range(n), 2):
        prompt = pairwise_usefulness_prompt(
            topic=topic,
            idea_a=ideas[i],
            idea_b=ideas[j],
        )
        raw = call_llm(prompt, backend=backend, model=model, temperature=temperature)
        try:
            parsed = extract_json(raw)
            winner = str(parsed.get("winner", "Tie")).strip()
            reason = parsed.get("reason", "")
        except Exception:
            winner = "Tie"
            reason = "Parsing failed; counted as tie"    

        records[i]["pairwise_matches"] += 1
        records[j]["pairwise_matches"] += 1

        if winner == "A":              
            records[i]["pairwise_wins"] += 1
            records[j]["pairwise_losses"] += 1
        elif winner == "B":
            records[j]["pairwise_wins"] += 1
            records[i]["pairwise_losses"] += 1
        else:
            records[i]["pairwise_ties"] += 1
            records[j]["pairwise_ties"] += 1

        comparisons.append({
            "idea_a_index": i + 1,
            "idea_b_index": j + 1,
            "winner": winner,
            "reason": reason,
        })

    results = []

    for i in range(n):
        rec = records[i]
        matches = rec["pairwise_matches"]

        # tie counts as half win
        win_rate = (
            rec["pairwise_wins"] + 0.5 * rec["pairwise_ties"]
        ) / matches if matches else 0
        usefulness_pairwise = 1 + 4 * win_rate

        rec["usefulness_win_rate"] = round(win_rate, 3)
        rec["usefulness_pairwise"] = round(usefulness_pairwise, 3)

        results.append(rec)

    ranked_results = sorted(
        results,
        key=lambda x: (
            x["usefulness_win_rate"],
            x["pairwise_wins"],
            -x["pairwise_losses"],
        ),
        reverse=True,
    )
    for rank, rec in enumerate(ranked_results, start=1):
        rec["usefulness_rank"] = rank
        idea_idx = rec["idea_index"]
        rec["idea_text"] = ideas[idea_idx - 1]

    top_3 = ranked_results[:3]

    return {
        "idea_scores": results,
        "ranked_idea_scores": ranked_results,
        "top_3": top_3,
        "comparisons": comparisons,
    }