import os
import requests
import streamlit as st

def get_secret(name, default=None):
    try: return st.secrets[name]
    except Exception: return os.getenv(name, default)

# local llm jarvis
def call_jarvis(prompt, model=None, temperature=0.8):
    base_url = get_secret("JARVIS_BASE_URL")
    api_key = get_secret("JARVIS_API_KEY")
    default_model = get_secret("JARVIS_MODEL", "default")

    if not base_url: raise ValueError("JARVIS_BASE_URL is not set")

    model = model or default_model
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    playload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "temperature": temperature
    }

    response = requests.post(
        url,
        headers=headers,
        json=playload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]