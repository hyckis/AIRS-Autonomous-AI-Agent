import json
from pathlib import Path
from finance_agents import (
    finance_naive_agent,
    finance_strong_agent,
    finance_lens_agent,
    finance_detect_homogeneity,
)
#from evaluator import evaluate_output
from scenario_context import build_fixed_context
from finance_prompts import (
    prompt_arm_a,
    prompt_arm_b,
    prompt_arm_c,
    build_retrieval,
)
from finance_corpus import (
    load_bank_profile,
    retrieve_finance,
    format_retrieved_context,
)

# paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# helper
def save_result(result, bank_id, run_id=1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"bank_{bank_id.lower()}_run_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path

def parse_json_response(text):
    if text is None: raise ValueError("LLM returned None instead of JSON.")
    cleaned = text.strip()
    if not cleaned: raise ValueError("LLM returned None instead of JSON.")
    if cleaned.startswith("```json"): cleaned = cleaned[len("```json"):].strip()
    if cleaned.startswith("```"): cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"): cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No valid JSON object found in LLM response.\n"
            f"Raw response:\n{text}"
        )

    json_text = cleaned[start:end + 1]
    try: return json.loads(json_text)
    except json.JSONDecodeError as e:
        print("\n========== JSON PARSE FAILED ==========")
        print(json_text)
        print("=======================================\n")
        raise ValueError(
            f"LLM returned malformed JSON: {e}"
        ) from e


def run_finance_case(bank_id="A", run_id=1, top_k=5, backend="local_ollama", model=None):
    bank_label = f"Bank {bank_id}"

    # Fixed context
    fixed_context = build_fixed_context(bank_id)
    bank_profile = load_bank_profile(bank_id)

    # -----------------------------
    # Arm A
    # -----------------------------
    baseline_output = finance_naive_agent(
        bank_label=bank_label,
        fixed_context=fixed_context,
        backend=backend,
        model=model,
    )

    # -----------------------------
    # Shared retrieval for B + C
    # -----------------------------
    retrieval_query = build_retrieval(
        bank_profile
    )

    retrieved_chunks = retrieve_finance(
        retrieval_query,
        top_k=top_k,
    )
    print("\n========== RETRIEVED CHUNKS ==========") 
    for chunk in retrieved_chunks: 
        print( f"\n{chunk['chunk_id']} \nsource={chunk['source']} \nsection={chunk['section']} \nscore={chunk['score']:.4f}" ) 

    evidence = format_retrieved_context(
        retrieved_chunks
    )

    # -----------------------------
    # Arm B
    # -----------------------------
    strong_output = finance_strong_agent(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=evidence,
        backend=backend,
        model=model,
    )

    # -----------------------------
    # Convergence detection
    # -----------------------------
    
    critique_raw = finance_detect_homogeneity(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=evidence,
        baseline_response=baseline_output,
        strong_response=strong_output,
        backend=backend,
        model=model,
    )
    critique = parse_json_response(critique_raw)
    supported_directions = critique.get("supported_directions", [])
    supported_directions_text = json.dumps(supported_directions, indent=2)
    # -----------------------------
    # Arm C
    # -----------------------------

    lens_output = finance_lens_agent(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=evidence,
        baseline_response=baseline_output,
        strong_response=strong_output,
        supported_directions=supported_directions_text,
        backend=backend,
        model=model,
    )

    return {
        "bank_id": bank_id,
        "run_id": run_id,
        "retrieval": {
            "query": retrieval_query,
            "chunk_ids": [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
            ],
        },
        "baseline": baseline_output,
        "strong": strong_output,
        "critique": critique,
        "lens": lens_output,
    }

if __name__ == "__main__":
    results = run_finance_case(bank_id="A", run_id=1, top_k=5)
    print(
        "\n"
        "========================================\n"
        "FINANCE CASE COMPLETE\n"
        "========================================"
    )

    print("\nArm A:")
    print(results["baseline"])

    print("\nArm B:")
    print(results["strong"])

    print("\nCritique:")
    print(results["critique"])

    print("\nArm C:")
    print(results["lens"])