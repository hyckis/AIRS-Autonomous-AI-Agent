import json
from pathlib import Path
from agents import detect_homogeneity
from agent_app.finance_case.finance_agents import (
    finance_naive_agent,
    finance_strong_agent,
    finance_lens_agent,
)
from evaluator import evaluate_output
from agent_app.finance_case.scenario_context import build_fixed_context
from agent_app.finance_case.finance_prompts import (
    prompt_arm_a,
    prompt_arm_b,
    prompt_arm_c,
    build_retrieval,
)
from agent_app.finance_case.finance_corpus import (
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

def run_finance_case(bank_id="A", run_id=1, top_k=5, save=True):
    bank_id = bank_id.upper()
    if bank_id not in {"A", "B"}: raise ValueError("bank id must be either A or B")

    bank_label = f"Bank {bank_id}"
    print(f"\nRunning finance case: {bank_label}")

    # fixed context
    fixed_context = build_fixed_context(bank_id)
    bank_profile = load_bank_profile(bank_id)

    # arm a
    print("\n[1/5] Running Arm A...")
    a_prompt = prompt_arm_a(bank_label=bank_label, fixed_context=fixed_context)
    baseline_output = finance_naive_agent(a_prompt)

    # retrieval: retrieved once and passed to B/C
    print("\n[2/5] Retrieving evidence...")
    retrieval_query = build_retrieval(bank_profile)
    retrieved_chunks = retrieve_finance(retrieval_query, top_k=top_k)
    evidence = format_retrieved_context(retrieved_chunks)

    # arm b
    print("\n[3/5] Running Arm B...")
    b_prompt = prompt_arm_b(bank_label=bank_label, fixed_context=fixed_context, retrieved_context=evidence)
    strong_output = finance_strong_agent(b_prompt)

    # critique
    print("\n[4/5] Detecting convergence...")
    combined_initial_outputs = f"""
ARM A OUTPUT
============
{baseline_output}


ARM B OUTPUT
============
{strong_output}
""".strip()
    
    critique = detect_homogeneity(combined_initial_outputs)
    
    # arm c
    print("\n[5/5] Running Arm C...")
    c_prompt = prompt_arm_c(bank_label=bank_label, fixed_context=fixed_context, retrieved_context=evidence, convergence_analysis=critique)
    lens_output = finance_lens_agent(c_prompt)

    # result
    result = {
        "bank_id": bank_id,
        "bank_label": bank_label,
        "run_id": run_id,
        "retrieval": {
            "query": retrieval_query,
            "top_k": top_k,
            "chunk_ids": [item["chunk_id"] for item in retrieved_chunks],
            "chunks": retrieved_chunks,
        },
        "outputs": {
            "baseline": baseline_output,
            "strong": strong_output,
            "lens": lens_output,
        },
        "convergence_analysis": critique,
    }

    if save:
        result_path = save_result(result, bank_id=bank_id, run_id=run_id)
        print(f"\nResult saved to: {result_path}")

    return result

if __name__ == "__main__":
    results = run_finance_case(bank_id="A", run_id=1, top_k=5)
    print(
        "\n"
        "========================================\n"
        "FINANCE CASE COMPLETE\n"
        "========================================"
    )

    print("\nArm A:")
    print(results["outputs"]["baseline"])

    print("\nArm B:")
    print(results["outputs"]["strong"])

    print("\nArm C:")
    print(results["outputs"]["lens"])