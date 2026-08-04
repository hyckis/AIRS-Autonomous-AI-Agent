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
      "id": 1,
      "assumption": "A short statement of the dominant assumption.",
      "challenge_criteria": "What a research direction would need to do to genuinely challenge this assumption."
    }},
    {{
      "id": 2,
      "assumption": "A short statement of another dominant assumption.",
      "challenge_criteria": "What a research direction would need to do to genuinely challenge this assumption."
    }}
  ]
}}

Do not include markdown.
Do not wrap the JSON in ```json.
Do not include explanations outside the JSON.
""" 

def evaluate_with_llm_prompt(topic, bank_text, idea_payloads):
    return f"""
You are a strict JSON-only evaluation engine.

Evaluate each research idea below using the provided topic-level assumption bank.

Return ONLY one valid JSON object.
Do not include markdown.
Do not wrap the JSON in ```json.
Do not include explanations outside the JSON.

Topic:
{topic}

Pre-identified dominant assumptions:
Use these assumptions as the primary reference for assumption_challenge scoring.
Only use assumption_id = null if none of the listed assumptions clearly apply.

{bank_text}

Ideas to evaluate:
{json.dumps(idea_payloads, ensure_ascii=False, indent=2)}

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
First, identify which assumption from the assumption bank the idea challenges.
Then judge how strongly the idea challenges that assumption according to the assumption's challenge_criteria.

1 = conformist; accepts the dominant assumption
2 = surface variation; changes implementation but the dominant assumption still holds
3 = partial challenge; challenges a secondary aspect of the assumption
4 = strong challenge; challenges a core aspect of the assumption
5 = core premise reversal; the assumption would no longer hold if this idea were adopted

Rules for assumption_id:
- assumption_id must be one of the IDs from the assumption bank.
- If no listed assumption is clearly challenged, set assumption_id to null.
- If assumption_id is null, challenged_assumption must be "No specific assumption directly addressed".
- If assumption_id is null, assumption_challenge must be 1 or 2.
- Do not assign assumption_challenge >= 3 unless the idea clearly challenges one listed assumption.
- Do not assign assumption_challenge >= 4 unless the idea directly reverses, undermines, or reframes a core part of one listed assumption.
- idea_title must copy the provided idea_title exactly.

Return exactly this JSON structure:

{{
  "idea_scores": [
    {{
      "idea_index": 1,
      "idea_title": "copy the provided idea_title exactly",
      "assumption_id": 1,
      "challenged_assumption": "copy the matched assumption text, or 'No specific assumption directly addressed'",
      "novelty": 1,
      "diversity": 1,
      "usefulness": 1,
      "assumption_challenge": 1,
      "rationale": "One short explanation referencing the matched assumption and challenge_criteria."
    }}
  ],
  "summary": "One short summary of the evaluation."
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

def pairwise_usefulness_prompt(topic, idea_a, idea_b):
    return f"""
You are evaluating research idea usefulness.

Topic:
{topic}

Compare the two research ideas below.

Usefulness means:
- relevant to the research topic
- feasible to investigate
- clear enough to guide research planning
- likely to produce meaningful findings
- not merely interesting but difficult to operationalize

Idea A:
{idea_a}

Idea B:
{idea_b}

Choose which idea is more useful for research planning.

Return ONLY valid JSON.
Do not include markdown.
Do not wrap the JSON in ```json.

Return exactly this structure:
{{
  "winner": "A",
  "reason": "One short explanation."
}}

Rules:
- winner must be "A", "B", or "Tie".
- Choose "Tie" only if both ideas are similarly useful.
"""

# for multi-agent debate
def advocate_prompt(topic, idea_text, assumption_item):
    return f"""
You are the Advocate in a research evaluation debate.

Your role:
Argue that the research idea genuinely challenges the given dominant assumption.

Topic:
{topic}

Research idea:
{idea_text}

Dominant assumption:
{assumption_item.get("assumption", "")}

Challenge criteria:
{assumption_item.get("challenge_criteria", "")}

Task:
Explain the strongest case that this idea satisfies the challenge criteria.

Return ONLY valid JSON.
Do not include markdown.

Return exactly:
{{
  "advocate_argument": "A concise argument for why the idea challenges the assumption.",
  "criteria_met": "Which part of the challenge criteria is met.",
  "suggested_score": 1
}}

Score guide:
1 = does not challenge the assumption
2 = surface variation
3 = partial challenge
4 = strong challenge
5 = core premise reversal
"""


def skeptic_prompt(topic, idea_text, assumption_item, advocate_argument):
    return f"""
You are the Skeptic in a research evaluation debate.

Your role:
Critically examine whether the research idea actually challenges the dominant assumption,
or whether it only changes implementation details while preserving the assumption.

Topic:
{topic}

Research idea:
{idea_text}

Dominant assumption:
{assumption_item.get("assumption", "")}

Challenge criteria:
{assumption_item.get("challenge_criteria", "")}

Advocate argument:
{advocate_argument}

Task:
Identify the strongest reason this idea may NOT genuinely satisfy the challenge criteria.

Return ONLY valid JSON.
Do not include markdown.

Return exactly:
{{
  "skeptic_argument": "A concise critique of the advocate's claim.",
  "criteria_gap": "Which part of the challenge criteria is missing or weak.",
  "suggested_score": 1
}}

Score guide:
1 = does not challenge the assumption
2 = surface variation
3 = partial challenge
4 = strong challenge
5 = core premise reversal
"""


def judge_prompt(topic, idea_text, assumption_item, advocate_result, skeptic_result):
    return f"""
You are the Judge in a research evaluation debate.

Your role:
Decide the final assumption_challenge score based on:
- the research idea,
- the dominant assumption,
- the challenge criteria,
- the Advocate's argument,
- the Skeptic's critique.

Topic:
{topic}

Research idea:
{idea_text}

Dominant assumption:
{assumption_item.get("assumption", "")}

Challenge criteria:
{assumption_item.get("challenge_criteria", "")}

Advocate result:
{advocate_result}

Skeptic result:
{skeptic_result}

Scoring rubric:
1 = conformist; accepts the dominant assumption
2 = surface variation; changes implementation but the dominant assumption still holds
3 = partial challenge; challenges a secondary aspect of the assumption
4 = strong challenge; challenges a core aspect of the assumption
5 = core premise reversal; the assumption would no longer hold if this idea were adopted

Important rules:
- Do not give 4 or 5 unless the idea clearly satisfies the challenge criteria.
- Do not give 5 unless the idea reverses or invalidates the core premise of the assumption.
- If the idea is interesting but does not challenge the assumption itself, give 1 or 2.
- If the idea only improves, optimizes, or implements the assumption differently, give 2.

Return ONLY valid JSON.
Do not include markdown.

Return exactly:
{{
  "assumption_challenge": 1,
  "final_decision": "A concise explanation of the final score.",
  "advocate_validity": "How convincing the advocate was.",
  "skeptic_validity": "How convincing the skeptic was.",
  "confidence": 0.0
}}
"""