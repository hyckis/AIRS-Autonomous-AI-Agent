import os
import streamlit as st
from openai import OpenAI

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

def call_llm_old(prompt, model="gpt-4o-mini", temperature=0.8):
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

#LOCAL LLM PROVIDER SUPPORT

def get_client():
    provider = os.getenv("LLM_PROVIDER", "ollama")

    if provider == "openai":
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )


def call_llm(prompt,
             model=None,
             temperature=0.8):

    provider = os.getenv("LLM_PROVIDER", "ollama")

    if model is None:
        if provider == "openai":
            model = os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            )
        else:
            model = os.getenv(
                "OLLAMA_MODEL",
                "gemma3"
            )

    client = get_client()

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
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content