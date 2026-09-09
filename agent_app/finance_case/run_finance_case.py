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


def validate_supported_directions(
    supported_directions,
    retrieved_chunks,
    allowed_exposure_keys,
):
    """
    Validate the structure and retrieved-evidence references of
    homogeneity-agent supported directions.

    This does NOT determine whether the reasoning is semantically correct.
    It only prevents malformed entries or fabricated CHUNK_IDs from
    being passed to the Lens Agent.
    """

    allowed_chunk_ids = {
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    }

    validated = []

    for item in supported_directions:
        if not isinstance(item, dict): continue

        direction = item.get("direction", "").strip()
        exposure_key = item.get(
            "required_exposure_keys",
            ["unsupported"],
        )
        # if exposure_key not in allowed_exposure_keys:
        #     print(
        #         f"[FILTERED] unsupported target exposure "
        #         f"'{exposure_key}': {direction}",
        #         flush=True,
        #     )
        #     continue

        invalid_exposure_keys = [
            key for key in exposure_key if key not in allowed_exposure_keys]

        if invalid_exposure_keys:
            print(
            f"[FILTERED] unsupported target exposure(s) "
            f"{invalid_exposure_keys}: {direction}",
            flush=True,
            )
            continue


        # Keep your existing semantic flags
        if item.get("exposure_explicitly_supported") is not True:
            print(
                f"[FILTERED] exposure not explicitly supported: "
                f"{direction}",
                flush=True,
            )
            continue

        unsupported = item.get(
            "unsupported_assumptions_required",
            [],
        )

        if unsupported:
            print(
                f"[FILTERED] unsupported assumptions "
                f"{unsupported}: {direction}",
                flush=True,
            )
            continue

        # -------------------------
        # CHUNK_ID validation
        # -------------------------
        external_chunk_ids = item.get(
            "external_chunk_ids",
            [],
        )

        invalid_ids = [
            cid
            for cid in external_chunk_ids
            if cid not in allowed_chunk_ids
        ]

        if invalid_ids:
            print(
                f"[FILTERED] invalid CHUNK_ID(s) "
                f"{invalid_ids}: {direction}",
                flush=True,
            )
            continue

        validated.append(item)

    print(
        f"[SUPPORTED DIRECTIONS] "
        f"{len(validated)}/{len(supported_directions)} "
        f"passed validation",
        flush=True,
    )

    return validated



SUPPORTED_EXPOSURES_BY_BANK = {
    "A": {
        "total_loans",
        "uninsured_deposits",
        "afs_securities",
        "htm_securities",
        "securities_unrealized_losses",
        "customer_concentration",
    },

    "B": {
        "total_loans",
        "cre_loans",
        "uninsured_deposits",
        "afs_securities",
        "htm_securities",
        "securities_unrealized_losses",
    },
}


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
        allowed_exposure_keys=SUPPORTED_EXPOSURES_BY_BANK[bank_id],
        backend=backend,
        model=model,
    )
    critique = parse_json_response(critique_raw)
    validated_supported = validate_supported_directions(
        critique.get("supported_directions", []), 
        retrieved_chunks,
        SUPPORTED_EXPOSURES_BY_BANK[bank_id]
    )
    print("VALIDATED: ", validated_supported, flush=True)
    for i, item in enumerate(validated_supported, start=1): item["direction_id"] = f"D{i}"
    lens_supported_directions = [{
        "direction_id": item["direction_id"],
        "direction": item["direction"],
        "required_exposure_keys": item["required_exposure_keys"],
        "target_bank_support": item["target_bank_support"],
        "external_chunk_ids": item["external_chunk_ids"],
    } for item in validated_supported]

    for item in validated_supported:
        exposure_supported = item.get("exposure_explicitly_supported", False)
        unsupported = item.get("unsupported_assumptions_required", []) 
        if unsupported: 
            print( f"[FILTERED] unsupported assumptions " f"{unsupported}", flush=True, ) 
            continue 


    dominant_framing = critique.get("dominant_framing", [])
    overlapping_mechanisms = critique.get("overlapping_mechanisms", [])
    #supported_directions = critique.get("supported_directions", [])
    #supported_directions_text = json.dumps(supported_directions, indent=2)
    

    # -----------------------------
    # Arm C
    # -----------------------------

    if not validated_supported: 
        print("[ARM C ABSTAINED] No validated supported directions", flush=True)
        lens_output = ("No evidence-supported alternative direction is available from the supplied context.")
        lens_status = "not generated"

    else:
        lens_status = "generated"
        lens_output = finance_lens_agent(
            bank_label=bank_label,
            fixed_context=fixed_context,
            retrieved_context=evidence,
            #baseline_response=baseline_output,
            #strong_response=strong_output,
            supported_directions=lens_supported_directions,
            dominant_framing=dominant_framing,
            overlapping_mechanisms=overlapping_mechanisms,
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
        "lens_status": lens_status
    }

if __name__ == "__main__":
    results = run_finance_case(bank_id="B", run_id=5, top_k=5, model="gemma3:12b")
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