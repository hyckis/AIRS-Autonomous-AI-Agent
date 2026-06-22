import json
import re
from llm_backend import call_llm

def extract_json(text):
    text = text.strip()

    # case 1: ```json
    text = re.sub(r"^```json\s", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # case 2: direct parsing
    try: return json.loads(text)
    except json.JSONDecodeError: pass

    # case 3: extract first json obj inside the txt
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError: "No json obj found"

def extract_json_old(text):
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )
    if match:
        return json.loads(
            match.group()
        )
    raise ValueError("No JSON file found")

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

    Evaluation Rubric:
    Evaluate each individual research direction separately for all four metrics,
    then calculate the final metric scores as the average across all directions.
    - Novelty measures how original and unexpected a research direction is relative to mainstream research trends.
        - A score of 1 indicates a highly conventional or commonly suggested direction
        - 2 indicates a minor variation on established ideas
        - 3 indicates a moderately original extension of existing work
        - 4 indicates a clearly unconventional or underexplored direction
        - 5 indicates a highly original direction that introduces uncommon perspectives or challenges established thinking.
    - Diversity measures how conceptually distinct each direction is from the other directions in the set.
        - A score of 1 indicates substantial overlap with the other directions
        - 2 indicates minor conceptual variation
        - 3 indicates moderate conceptual differentiation
        - 4 indicates substantial conceptual differences
        - 5 indicates a direction that explores a significantly different perspective, stakeholder, discipline, methodology, or problem framing. 
    - Usefulness measures the practical and scholarly value of the direction. 
        - A score of 1 indicates a vague, unrealistic, or unproductive direction
        - 2 indicates limited research value
        - 3 indicates moderate research potential
        - 4 indicates a clear and actionable research direction
        - 5 indicates a highly impactful, feasible, and researchable direction likely to generate meaningful insights. 
    - Assumption Challenge measures how strongly the direction questions, reverses, or undermines dominant assumptions present in the topic area. 
        - A score of 1 indicates no challenge and fully accepts dominant assumptions
        - 2 indicates a minor variation that changes implementation while accepting the assumptions
        - 3 indicates a partial challenge that questions some aspects of the assumptions
        - 4 indicates a strong challenge that directly questions major assumptions
        - 5 indicates a fundamental challenge that reverses, contradicts, or undermines dominant assumptions. After evaluating all directions individually, compute the average score for each metric and return only the final averaged scores.

    Return only valid JSON:
    {{
      "novelty": 0,
      "diversity": 0,
      "usefulness": 0,
      "assumption_challenge": 0,
      "summary": ""
    }}
    """
    result = call_llm(prompt, "local_ollama", temperature=0.2)

    try:
        return extract_json(result)
        #return json.loads(result)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "novelty": 0,
            "diversity": 0,
            "usefulness": 0,
            "assumption_challenge": 0,
            "summary": f"Evaluation parsing failed: {e}. Raw output: {result[:500]}"
        }