import os
import streamlit as st
import requests
from openai import OpenAI

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

# openai gpt4o
def get_openai_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")
    
def get_openai_client():
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found.")
    return OpenAI(api_key = api_key)

def call_openai(prompt, model="gpt-4o-mini", temperature=0.8):
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous AI research assistant. "
                    "You help identify idea homogenization, dominant assumptions, "
                    "and cognitively diverse research directions."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# local llm ollama
def call_ollama(prompt, model="gemma3", temperature=0.8):
    url = ""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["response"]

# local llm jarvis
def call_jarvis(prompt, model, temperature=0.8):
    return True