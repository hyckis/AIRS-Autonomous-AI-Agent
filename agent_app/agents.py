from llm_backend import call_llm
from datetime import date
from retrieval import (
    DEFAULT_MIN_YEAR,
    build_fallback_queries,
    format_papers_for_prompt,
    normalize_queries,
    parse_query_list,
    retrieve_papers,
)
from prompts_agents import (
    baseline_research_prompt,
    diversity_expander_prompt,
    strong_prompt_baseline_prompt,
    detect_homogeneity_prompt,
    human_question_prompt,
    arxiv_prompt,
)

# baseline agent
def baseline_research_agent(topic):
    prompt = baseline_research_prompt(topic)
    return call_llm(prompt, temperature=0.8)
    #return call_llm(prompt, temperature=0.7)

# homogeneity detect: baseline agent / naive llm
def detect_homogeneity(topic, baseline_response):
    prompt = detect_homogeneity_prompt(topic, baseline_response)
    return call_llm(prompt, "local_ollama", temperature=0.5)
    #return call_llm(prompt, temperature=0.5)

# diversity agent
def diversity_expander_agent(topic, baseline_response, critique, papers=None):
    literature_context = format_papers_for_prompt(papers or [])
    prompt = diversity_expander_prompt(topic, baseline_response, critique, literature_context)
    return call_llm(prompt, temperature=0.8)
    #return call_llm(prompt, temperature=0.95)

# strong agent
def strong_prompt_baseline_agent(topic, backend="local_ollama", model=None):
    prompt = strong_prompt_baseline_prompt(topic)
    return call_llm(prompt, backend=backend, model=model, temperature=0.8,)

# arxiv searching existing papers
def generate_arxiv_queries(topic, min_year=DEFAULT_MIN_YEAR):
    prompt = arxiv_prompt(topic, min_year)
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

    papers, failed_queries = retrieve_papers(queries, limit=limit)

    if not papers and query_source == "llm":
        fallback_queries = build_fallback_queries(topic, min_year=min_year)
        papers, fallback_failed = retrieve_papers(fallback_queries, limit=limit)
        queries = queries + fallback_queries
        failed_queries = failed_queries + fallback_failed
        if papers:
            query_source = "llm+fallback"

    return {
        "queries": queries,
        "papers": papers,
        "query_source": query_source,
        "failed_queries": failed_queries,
    }

# human question
def generate_human_question(topic, baseline_response, critique):
    prompt = human_question_prompt(topic, baseline_response, critique)
    return call_llm(prompt, temperature=0.7)

# for assumption_challenge evaluation
def build_assumption_bank(topic, papers, backend="local_ollama", model=None):
    literature_context = format_pepers_for_prompt(papers or [])
    prompt = assumption_bank_prompt(topic, literature_context)