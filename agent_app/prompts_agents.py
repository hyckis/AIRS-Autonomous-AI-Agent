from datetime import date

# agents.py
def baseline_research_prompt(topic):
    return f"""
A researcher asks for promising research directions on this topic:

Topic:
{topic}

Generate 8 plausible research directions that a standard LLM research assistant would likely suggest.

Keep them useful and clear.
Return as a numbered list.
    """

def diversity_expander_prompt(topic, baseline_response, critique, literature_context):
    return f"""
You are a cognitive-diversity-preserving research agent.

Topic:
{topic}

Standard LLM response:
{baseline_response}

Homogeneity critique:
{critique}

Retrieved literature from arXiv:
{literature_context}

Your goal is to generate research directions that reduce idea homogenization while staying grounded in the retrieved literature where possible.

Use these lenses:
    - counter-mainstream hypothesis
    - marginalized or overlooked user group
    - historical analogy
    - failure-mode analysis
    - cultural difference
    - anti-efficiency perspective
    - collective intelligence perspective
    - long-term speculative risk

Generate 8 alternative research directions.

For each direction, include:
    1. Title
    2. Short description
    3. Which dominant assumption it challenges
    4. Why it preserves cognitive diversity
    5. Supporting papers (cite title and URL from the retrieved literature above),
       or label as "speculative extension" if not directly supported by a retrieved paper

Only cite papers from the retrieved literature list. Do not invent citations.
    """


def strong_prompt_baseline_prompt(topic):
    return f"""
You are an expert research ideation assistant.

Your goal is to generate a diverse set of research directions for the topic below.

Use the following strategy:
1. Think from multiple expert personas:
   - technical researcher
   - social scientist
   - educator or practitioner
   - policy or ethics expert
   - skeptical critic
2. Make the ideas intentionally different from each other.
3. Avoid repeating the same assumption across ideas.
4. Include both mainstream and non-obvious directions.
5. Generate 8 concise but researchable ideas.

Topic:
{topic}

Return the output as a numbered list of 8 distinct research directions.
For each direction, include:
- title
- one-sentence description
- why it is different from the others
"""

def detect_homogeneity_prompt(topic, baseline_response):
    return f"""
You are analyzing whether a standard LLM response suffers from idea homogenization.

Topic:
{topic}

Standard LLM research directions:
{baseline_response}

Analyze:
    1. What dominant assumptions appear repeatedly?
    2. Which ideas are semantically different but conceptually similar?
    3. What mainstream trends are being amplified?
    4. What perspectives are missing?

Return a concise critique with bullet points.
    """

def human_question_prompt(topic, baseline_response, critique):
    return f"""
You are a semi-autonomous research agent that keeps the human researcher in the loop.

Topic:
{topic}

Standard LLM response:
{baseline_response}

Homogeneity critique:
{critique}

Ask ONE high-value question that would help the researcher choose what kind of originality they want to pursue.

The question should not be generic. It should reveal a meaningful strategic choice.
    """

def arxiv_prompt(topic, min_year):
    return f"""
You generate arXiv API search queries for a research topic.

Topic:
{topic}

Rules:
    - Return ONLY a JSON array of 2-3 query strings.
    - Each query must use arXiv boolean syntax with ti: and/or abs: field prefixes.
    - Derive all search terms from the topic itself. Do not assume a fixed domain.
    - Each query should combine multiple topic-specific terms with AND so generic
      single-word matches are avoided.
    - Prefer quoted phrases for multi-word concepts from the topic.
    - Include this recency filter in every query (use today's date, not *):
      submittedDate:[{min_year}0101 TO {date.today().strftime('%Y%m%d')}]
    - Keep each query reasonably short so arXiv does not reject it.
    - Use at most 2-3 AND clauses per query.

Example for topic "protein folding with deep learning":
    [
      "(abs:\\"protein folding\\" OR ti:\\"protein folding\\") AND (abs:learning) AND submittedDate:[20200101 TO 20260101]",
      "(abs:protein AND abs:folding AND abs:learning) AND submittedDate:[20200101 TO 20260101]"
    ]
    """

def assumption_bank_prompt(topic, literature_context):
    return f"""
Topic: 
{topic} 

Retrieved literature: 
{literature_context} 

List 3-5 dominant assumptions that mainstream research in this area typically accepts without question. 
For each, give a short name and one-sentence description. 
Return as JSON array of 
{{"name": "...", "description": "..."}}.
"""