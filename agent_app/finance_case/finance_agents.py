from finance_llm_backend import call_llm

from finance_prompts import (
    prompt_arm_a,
    prompt_arm_b,
    prompt_arm_c,
    prompt_homogeneity,
)


# =========================================================
# Arm A — Naive
# =========================================================

def finance_naive_agent(
    bank_label,
    fixed_context,
    backend="local_ollama",
    model=None,
):
    prompt = prompt_arm_a(
        bank_label=bank_label,
        fixed_context=fixed_context,
    )

    return call_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=0.8,
    )


# =========================================================
# Arm B — Strong Prompt
# =========================================================

def finance_strong_agent(
    bank_label,
    fixed_context,
    retrieved_context,
    backend="local_ollama",
    model=None,
):
    prompt = prompt_arm_b(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=retrieved_context,
    )

    return call_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=0.8,
    )


# =========================================================
# Homogeneity / Convergence Detection
# =========================================================

def finance_detect_homogeneity(
    bank_label,
    fixed_context,
    retrieved_context,
    baseline_response,
    strong_response,
    allowed_exposure_keys,
    backend="local_ollama",
    model=None,
):
    prompt = prompt_homogeneity(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=retrieved_context,
        baseline_response=baseline_response,
        strong_response=strong_response,
        allowed_exposure_keys=allowed_exposure_keys,
    )

    return call_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=0.5,
    )


# =========================================================
# Arm C — Lens Agent
# =========================================================

def finance_lens_agent(
    bank_label,
    fixed_context,
    retrieved_context,
    supported_directions,
    dominant_framing,
    overlapping_mechanisms,
    backend="local_ollama",
    model=None,
):
    prompt = prompt_arm_c(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=retrieved_context,
        supported_directions=supported_directions,
        dominant_framing=dominant_framing,
        overlapping_mechanisms=overlapping_mechanisms
    )

    return call_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=0.8,
    )