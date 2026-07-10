import re
import json
import numpy as np
from llm_backend import call_llm

SCORE_METRICS = [
    "novelty",
    "diversity",
    "usefulness",
    "assumption_challenge"
]

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
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: "No json obj found"
    
    raise ValueError("No json obj found")

def validate_score_dict(score):
    """
    All score keys must exist and be numeric(1-5)
    """
    validated = {}

    for key in SCORE_METRICS:
        value = score.get(key, 0)
        try: value = float(value)
        except(TypeError, ValueError): value = 0
        if value > 0: 
            value = max(1, min(5, value))
        validated[key] = value
    
    validated["summary"] = score.get("summary", "")
    return validated

def evaluate_with_llm(topic, response_text, ideas=None, backend="local_ollama", model=None):
    """
    G-Eval inspired per-idea LLM rubric evaluation.
    Scores each idea separately, then averages scores in Python.
    """
    if ideas is None: ideas = [response_text] if response_text else []

    prompt = f"""
        You are a strict JSON-only evaluation engine.
        Evaluate the research-agent output below.

        Return ONLY one valid JSON object.
        Do not include markdown.
        Do not wrap the JSON in ```json.
        Do not include explanations outside the JSON.

        Topic:
        {topic}

        Ideas:
        {json.dumps(ideas, ensure_ascii=False, indent=2)}

        Use a 1-5 scale for each metric.

        Metric definitions:
        - novelty: Are the research directions original, non-obvious, or unexpected?
        - diversity: Does the output explore multiple distinct perspectives, stakeholders, methods, or assumptions?
        - usefulness: Are the ideas relevant and potentially valuable for research planning?
        - assumption_challenge: Does the output challenge dominant assumptions instead of simply extending mainstream ideas?

        Topic:
        {topic}

        Research-agent output:
        {response_text}

        Use a 1-5 scale for each metric.

Metric definitions:

1. novelty:
How original, non-obvious, or unexpected is this idea relative to mainstream research directions?
1 = very conventional
2 = minor variation on common ideas
3 = moderately original extension
4 = clearly underexplored or unconventional
5 = highly original and surprising

2. diversity:
How conceptually distinct is this idea from the other ideas in the same set?
1 = highly overlapping with other ideas
2 = minor variation
3 = moderately distinct
4 = substantially distinct
5 = explores a very different perspective, stakeholder, method, or framing

3. usefulness:
How useful, researchable, and valuable is this idea for research planning?
1 = vague or unrealistic
2 = limited research value
3 = moderately useful
4 = clear and actionable
5 = highly valuable, feasible, and researchable

4. assumption_challenge:
Does this idea challenge a dominant assumption rather than simply extending mainstream ideas?
1 = conformist; accepts dominant assumptions
2 = surface variation; changes implementation but keeps assumptions intact
3 = partial challenge; questions a secondary assumption
4 = strong challenge; questions an important assumption
5 = core premise reversal; reverses or undermines a central premise

Important rule for assumption_challenge:
Do not give a score of 4 or 5 unless you can explicitly name the assumption being challenged.

Return exactly this JSON structure:
{{
"idea_scores": [
{{
"idea_index": 1,
"novelty": 1,
"diversity": 1,
"usefulness": 1,
"assumption_challenge": 1,
"default_assumption": "...",
"challenged_assumption": "...",
"rationale": "One short explanation."
}}
],
"summary": "Brief explanation of the overall evaluation."
}}

        """
    
    raw_result = call_llm(
        prompt,
        backend = backend,
        model = model,
        temperature = 0.1
    )

    try:
        parsed = extract_json(raw_result)
        idea_scores = parsed.get("idea_scores", [])
        cleaned_idea_scores = []
        for item in idea_scores:
            cleaned = {
                "idea_index": item.get("idea_index", len(cleaned_idea_scores) + 1),
                "default_assumption": item.get("default_assumption", ""),
                "challenged_assumption": item.get("challenged_assumption", ""),
                "rationale": item.get("rationale", ""),
            }
            for metric in SCORE_METRICS:
                value = item.get(metric, 0)
                try: value = float(value)
                except (TypeError, ValueError): value = 0
                if value > 0: value = max(1, min(5, value))
                cleaned[metric] = value

            cleaned_idea_scores.append(cleaned)

        if not cleaned_idea_scores: raise ValueError("No valid idea_scores returned")

        averaged = {}
        for metric in SCORE_METRICS:
            averaged[metric] = round(
                float(np.mean([item[metric] for item in cleaned_idea_scores])),
                3
            )
        averaged["summary"] = parsed.get("summary", "")
        averaged["idea_scores"] = cleaned_idea_scores

        return averaged
    
    except Exception as e:
        return {
            "novelty": 0,
            "diversity": 0,
            "usefulness": 0,
            "assumption_challenge": 0,
            "summary": f"Evaluation parsing failed: {e}. Raw output: {raw_result[:500]}",
        }    

def compute_simple_metrics(response_text):
    words = response_text.split()
    word_cnt = len(words)

    diversity_lens_keywords = {
        "contrarian": ["contrarian", "opposite", "challenge", "counter", "alternative"],
        "historical": ["historical", "history", "past", "analogy", "precedent"],
        "cross_disciplinary": ["cross-disciplinary", "interdisciplinary", "biology", "sociology", "psychology", "economics"],
        "failure_mode": ["failure", "risk", "unintended", "breakdown", "misuse"],
        "cultural": ["cultural", "culture", "linguistic", "local", "context"],
        "stakeholder": ["stakeholder", "underserved", "marginalized", "student", "teacher", "community"],
        "long_term": ["long-term", "future", "scaling", "institutional", "societal"],
    }

    lower = response_text.lower()
    lens_coverage = {}
    for lens, keywords in diversity_lens_keywords.items():
        lens_coverage[lens] = any(keyword in lower for keyword in keywords)
    lens_coverage_cnt = sum(lens_coverage.values())

    return {
        "word_count": word_cnt,
        "lens_coverage_count": lens_coverage_cnt,
        "lens_coverage": lens_coverage,
    }

def evaluate_output(topic, response_text, ideas=None, backend="local_ollama", model=None):
    """
    Combined evaluation
    1: LLM as a judge
    2: Simple non LLM diagnostic metrics
    """
    llm_scores = evaluate_with_llm(
        topic=topic,
        response_text=response_text,
        backend=backend,
        model=model
    )

    simple_metrics = compute_simple_metrics(response_text)

    return {
        "llm_scores": llm_scores,
        "simple_metrics": simple_metrics
    }


# used as backward wrapper
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