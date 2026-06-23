from llm_backend import call_llm
from retrieval import (
    DEFAULT_MIN_YEAR,
    build_fallback_queries,
    normalize_queries,
    parse_query_list,
    retrieve_papers,
)

def baseline_research_agent(topic):
    prompt = f"""
    A researcher asks for promising research directions on this topic:

    Topic:
    {topic}

    Generate 8 plausible research directions that a standard LLM research assistant
    would likely suggest.

    Keep them useful and clear.
    Return as a numbered list.
    """
    return call_llm(prompt, temperature=0.7)


def detect_homogeneity(topic, baseline_response):
    prompt = f"""
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
    return call_llm(prompt, temperature=0.5)


def diversity_expander_agent(topic, baseline_response, critique):
    prompt = f"""
    You are a cognitive-diversity-preserving research agent.

    Topic:
    {topic}

    Standard LLM response:
    {baseline_response}

    Homogeneity critique:
    {critique}

    Your goal is to generate research directions that reduce idea homogenization.

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
    """
    return call_llm(prompt, temperature=0.95)


def generate_arxiv_queries(topic, min_year=DEFAULT_MIN_YEAR):
    prompt = f"""
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
    - Include this recency filter in every query:
      submittedDate:[{min_year}0101 TO *]
    - Keep each query reasonably short so arXiv does not reject it.

    Example for topic "protein folding with deep learning":
    [
      "(abs:\\"protein folding\\" OR ti:\\"protein folding\\") AND (abs:\\"deep learning\\" OR abs:learning) AND submittedDate:[20200101 TO *]",
      "(abs:protein AND abs:folding AND abs:learning) AND submittedDate:[20200101 TO *]"
    ]
    """
    try:
        result = call_llm(prompt, temperature=0.2)
        queries = parse_query_list(result)
        if queries:
            return normalize_queries(queries, min_year=min_year), "llm"
    except Exception:
        pass

    return build_fallback_queries(topic, min_year=min_year), "fallback"


def retrieve_literature(topic, limit=5, min_year=DEFAULT_MIN_YEAR, use_llm_queries=True):
    if use_llm_queries:
        queries, query_source = generate_arxiv_queries(topic, min_year=min_year)
    else:
        queries = build_fallback_queries(topic, min_year=min_year)
        query_source = "fallback"

    papers = retrieve_papers(queries, limit=limit)
    return {
        "queries": queries,
        "papers": papers,
        "query_source": query_source,
    }


def generate_human_question(topic, baseline_response, critique):
    prompt = f"""
    You are a semi-autonomous research agent that keeps the human researcher in the loop.

    Topic:
    {topic}

    Standard LLM response:
    {baseline_response}

    Homogeneity critique:
    {critique}

    Ask ONE high-value question that would help the researcher choose what kind of originality
    they want to pursue.

    The question should not be generic. It should reveal a meaningful strategic choice.
    """
    return call_llm(prompt, temperature=0.7)


