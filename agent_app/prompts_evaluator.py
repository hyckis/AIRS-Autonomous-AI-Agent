import json

def assumption_bank_prompt(topic, critique):
    return f""" 
You are identifying the dominant assumptions in a research field, based on a homogeneity critique of standard LLM responses. 

Topic: 
{topic} 

Homogeneity critique: 
{critique} 

Identify 3–5 dominant assumptions that researchers in this field typically accept without questioning.

Definition:
An assumption is a belief that much of the literature implicitly treats as true.
It is not:
- a design choice,
- an implementation detail,
- a benchmark,
- or a recommendation.

For each assumption, describe what would constitute genuinely challenging it.

A genuine challenge:
- questions whether the assumption itself should hold,
- proposes an alternative worldview,
- or demonstrates that the assumption is fundamentally flawed.

A genuine challenge is NOT:
- improving the assumption,
- making it more efficient,
- changing the implementation,
- or applying it in another domain.

Requirements:
- Focus on assumptions of the research field, not a single paper.
- Make each assumption distinct and non-overlapping.
- Assumptions should be concise (under 20 words).
- Challenge criteria should be under 80 words.

Output must be exactly one JSON.
Return exactly this JSON structure: 
{{ 
    "assumptions": [ 
        {{ 
            "assumption": "A short, named statement of the dominant assumption.", 
            "challenge_criteria": "What a research direction would need to do to genuinely challenge this assumption, as opposed to just varying its implementation." 
        }} 
    ] 
}} 

Do not include markdown.
Do not wrap the JSON in ```json.
Do not include explanations outside the JSON.
""" 

def evaluate_with_llm_prompt(topic, bank_text, ideas, response_text):
    return f"""
You are a strict JSON-only evaluation engine.
Evaluate the research-agent output below.

Return ONLY one valid JSON object.
Do not include markdown.
Do not wrap the JSON in ```json.
Do not include explanations outside the JSON.

Topic:
{topic}

Pre-identified dominant assumptions in this field 
(use these as your primary reference; only identify a new assumption if none of these apply):
{bank_text}

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
First, decide which assumption from the list above (if any) this idea relates to. 
Then judge how strongly it challenges that assumption according to its stated challenge_criteria. 

1 = conformist; accepts the assumption 
2 = surface variation; changes implementation but the assumption from the bank still holds 
3 = partial challenge; meets the challenge_criteria for a secondary aspect of the assumption 
4 = strong challenge; meets the challenge_criteria for a core aspect 
5 = core premise reversal; the challenge_criteria is fully met and the assumption no longer holds if this idea is adopted 

Do not give a score of 4 or 5 unless the idea's challenged_assumption field names 
one of the assumptions listed above (or explicitly explains why none apply), 
AND meets that assumption's stated challenge_criteria. 

Return exactly this JSON structure: 
{{ 
    "idea_scores": [ 
    {{ 
        "idea_index": 1, 
        "novelty": 1, 
        "diversity": 1, 
        "usefulness": 1, 
        "assumption_challenge": 1, 
        "challenged_assumption": "...", 
        "rationale": "One short explanation referencing the challenge_criteria." 
    }} 
    ], 
    "summary": "..." 
}}

        """


def evaluate_cognitive_diversity_prompt(topic, response_text):
    return f"""
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
Evaluate each individual research direction separately for all four metrics, then calculate the final metric scores as the average across all directions.
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