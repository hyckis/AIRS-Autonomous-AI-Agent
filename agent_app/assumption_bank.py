import json
from llm_backend import call_llm
from util import extract_json
from prompts_evaluator import assumption_bank_prompt

def generate_assumption_bank(topic, critique, backend="local_ollama", model=None):
    """ 
    Topic-level assumption bank, generated ONCE per topic and shared across 
    all arms (baseline / strong_prompt / lens_agent) so that 
    assumption_challenge scoring uses a consistent yardstick. 
    """
    prompt =  assumption_bank_prompt(topic, critique)
    raw = call_llm(prompt, backend=backend, model=model, temperature=0.3)

    try:
        parsed = extract_json(raw)
        bank = parsed.get("assumptions", [])
        bank = [
            item for item in bank
            if isinstance(item, dict) and item.get("assumption") and item.get("challenge_criteria")
        ]
        return bank
    except Exception: return []

def format_assumption_bank_for_prompt(bank):
    if not bank: return "No pre-identified assumptions available; identify the assumption yourself."
    lines = []
    for i, item in enumerate(bank, start=1):
        id = item.get("id", 1)
        assumption = item.get("assumption", "")
        challenge_criteria = item.get(
            "challenge_criteria",
            "Evaluate whether the idea explicitly challenges this assumption."
        )
        lines.append(
            f"Assumption ID: {id}\n" 
            f"Assumption: {assumption}\n "
            f"Challenge criteria: {challenge_criteria}"
        )
    return "\n".join(lines)