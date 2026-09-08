from llm_backend import call_llm

from .finance_prompts import (
    build_naive_prompt,
    build_strong_prompt,
    build_homogeneity_prompt,
    build_lens_prompt,
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
    prompt = build_naive_prompt(
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
    prompt = build_strong_prompt(
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
    baseline_response,
    strong_response,
    backend="local_ollama",
    model=None,
):
    prompt = build_homogeneity_prompt(
        bank_label=bank_label,
        baseline_response=baseline_response,
        strong_response=strong_response,
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
    baseline_response,
    strong_response,
    critique,
    backend="local_ollama",
    model=None,
):
    prompt = build_lens_prompt(
        bank_label=bank_label,
        fixed_context=fixed_context,
        retrieved_context=retrieved_context,
        baseline_response=baseline_response,
        strong_response=strong_response,
        convergence_analysis=critique,
    )

    return call_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=0.8,
    )