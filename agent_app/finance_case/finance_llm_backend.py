import os
import streamlit as st
import requests
from openai import OpenAI

def get_ollama_client():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

def call_ollama(prompt, model=None, temperature=0.8):
    provider = os.getenv("LLM_PROVIDER", "ollama")

    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma3")

    client = get_ollama_client()

    response = client.chat.completions.create(
        model = model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous AI research assistant."
                    "You help identify idea homogenization, dominant assumptions, "
                    "and cognitively diverse research directions."
                )
            },
            {
                "role": "user",
                "content": prompt
            }],
            temperature=temperature
    )

    print(f"Provider={provider} Model={model}")
    return response.choices[0].message.content

def call_llm(prompt, backend="local_ollama", model=None, temperature=0.8):
    if backend == "local_ollama":
        return call_ollama(prompt, model or "gemma3", temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")