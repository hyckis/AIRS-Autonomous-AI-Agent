import json
from llm_backend import call_llm
import re


def extract_json(text):
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:
        return json.loads(
            match.group()
        )

    raise ValueError(
        "No JSON found"
    )

def evaluate_cognitive_diversity(topic, response_text):
    prompt = f"""
    Evaluate the following research-agent output for the topic below.

    Topic:
    {topic}

    Output:
    {response_text}

    Score from 1 to 5:
    - novelty: originality of the research directions
    - diversity: conceptual variety across directions
    - usefulness: practical or scholarly value
    - assumption_challenge: how well it challenges mainstream assumptions

    Return only valid JSON:
    {{
      "novelty": 0,
      "diversity": 0,
      "usefulness": 0,
      "assumption_challenge": 0,
      "summary": ""
    }}
    """
    result = call_llm(prompt, temperature=0.2)

    try:
        return extract_json(result)
    except json.JSONDecodeError:
        return {
            "novelty": 0,
            "diversity": 0,
            "usefulness": 0,
            "assumption_challenge": 0,
            "summary": "Evaluation parsing failed."
        }