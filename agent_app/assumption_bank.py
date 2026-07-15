import json
from llm_backend import call_llm
from evaluator import extract_json
from prompts_evaluator import assumption_bank_prompt

def generate_assumption_bank(topic, critique, backend="local_ollama", model=None):
    """ 
    Topic-level assumption bank, generated ONCE per topic and shared across 
    all arms (baseline / strong_prompt / lens_agent) so that 
    assumption_challenge scoring uses a consistent yardstick. 
    """
    prompt =  assumption_bank_prompt(topic, critique)
    raw = call_llm(prompt, backend=backend, model=model, temperature=0.3)
    print(f"RAW OUTPUT: {raw} \n =============")

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
        lines.append(f"{i}. Assumption: {item['assumption']}\n Challenge criteria: {item['challenge_criteria']}")
    return "\n".join(lines)