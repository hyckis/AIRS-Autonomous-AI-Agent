from llm_backend_openai import call_openai
from llm_backend_ollama import call_ollama
from llm_backend_jarvis import call_jarvis

def call_llm(prompt, backend="openai", model=None, temperature=0.8):
    if backend == "openai":
        return call_openai(prompt, model or "gpt-4o", temperature)
    elif backend == "local_ollama":
        return call_ollama(prompt, model or "gemma3", temperature)
    elif backend == "jarvis":
        return call_jarvis(prompt, model, temperature)
    elif backend == "dummy":
        return call_dummy_agent(prompt, model, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def call_dummy_agent(prompt, model=None, temperature=0.8):
    return "fake local model response for testing backend"
