import json
import re
from datetime import date

import arxiv

DEFAULT_MIN_YEAR = 2020
DEFAULT_PAPERS_PER_QUERY = 5

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "but",
    "with", "from", "by", "as", "into", "about", "that", "this", "using", "via",
    "through", "between", "among", "their", "its", "are", "was", "were", "be",
    "been", "being", "how", "what", "when", "where", "which", "who", "whom",
}

SHORT_KEEP = {"ai", "ml", "llm", "nlp", "cv", "rl", "iot", "xr"}


def _recency_clause(min_year=DEFAULT_MIN_YEAR):
    return f"submittedDate:[{min_year}0101 TO {date.today().strftime('%Y%m%d')}]"


def _term_clause(term):
    if " " in term:
        return f'(abs:"{term}" OR ti:"{term}")'
    return f"(abs:{term} OR ti:{term})"


def extract_topic_terms(topic):
    """
    Pull searchable phrases and terms from a free-form research topic.
    """
    topic = topic.strip()
    phrases = [match.strip() for match in re.findall(r'"([^"]+)"', topic) if match.strip()]

    words = re.findall(r"[a-zA-Z0-9]+", re.sub(r'"[^"]+"', " ", topic))
    terms = []
    for word in words:
        lower = word.lower()
        if lower in STOP_WORDS:
            continue
        if len(word) <= 2 and lower not in SHORT_KEEP:
            continue
        terms.append(word)

    # Preserve likely multi-word phrases from the original topic.
    raw_parts = re.split(r"\s+", re.sub(r"[^\w\s]", " ", topic.lower()))
    chunk = []
    for part in raw_parts:
        if not part or part in STOP_WORDS:
            if len(chunk) >= 2:
                phrases.append(" ".join(chunk))
            chunk = []
            continue
        if len(part) <= 2 and part not in SHORT_KEEP:
            if len(chunk) >= 2:
                phrases.append(" ".join(chunk))
            chunk = []
            continue
        chunk.append(part)
    if len(chunk) >= 2:
        phrases.append(" ".join(chunk))

    deduped_phrases = []
    seen = set()
    for phrase in phrases:
        normalized = phrase.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped_phrases.append(phrase)

    deduped_terms = []
    seen_terms = set()
    for term in terms:
        normalized = term.lower()
        if normalized not in seen_terms:
            seen_terms.add(normalized)
            deduped_terms.append(term)

    return deduped_phrases, deduped_terms


def build_fallback_queries(topic, min_year=DEFAULT_MIN_YEAR):
    """
    Build arXiv boolean queries without an LLM.
    Derives phrases and terms from the topic itself instead of hardcoding
    any domain such as education or AI agents.
    """
    recency = _recency_clause(min_year)
    phrases, terms = extract_topic_terms(topic)
    queries = []

    quoted_topic = topic.strip().replace('"', "")
    if quoted_topic:
        queries.append(f'all:"{quoted_topic}" AND {recency}')

    if phrases:
        phrase_clause = " OR ".join(_term_clause(phrase) for phrase in phrases[:3])
        queries.append(f"({phrase_clause}) AND {recency}")

    if len(terms) >= 2:
        and_clause = " AND ".join(_term_clause(term) for term in terms[:4])
        queries.append(f"({and_clause}) AND {recency}")

    if len(terms) >= 4:
        midpoint = len(terms) // 2
        first_group = " OR ".join(_term_clause(term) for term in terms[:midpoint])
        second_group = " OR ".join(_term_clause(term) for term in terms[midpoint:4])
        queries.append(f"(({first_group}) AND ({second_group})) AND {recency}")

    if len(terms) == 1:
        queries.append(f"{_term_clause(terms[0])} AND {recency}")

    # Deduplicate while preserving order.
    unique_queries = []
    seen = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            unique_queries.append(query)

    return unique_queries or [f'all:"{quoted_topic}" AND {recency}']


def parse_query_list(raw_text):
    """Extract a JSON array of query strings from an LLM response."""
    raw_text = raw_text.strip()

    try:
        queries = json.loads(raw_text)
        if isinstance(queries, list):
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", raw_text)
    if match:
        try:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        except json.JSONDecodeError:
            pass

    return []


def normalize_queries(queries, min_year=DEFAULT_MIN_YEAR):
    """Ensure each query includes a valid arXiv recency filter."""
    recency = _recency_clause(min_year)
    end_date = date.today().strftime("%Y%m%d")
    normalized = []

    for query in queries:
        query = query.strip()
        if not query:
            continue

        # arXiv rejects open-ended ranges like "TO *" — use a concrete end date.
        query = re.sub(r"TO\s+\*", f"TO {end_date}", query, flags=re.IGNORECASE)
        # Normalize dashed dates: 2020-01-01 -> 20200101
        query = re.sub(
            r"(\d{4})-(\d{2})-(\d{2})",
            lambda match: f"{match.group(1)}{match.group(2)}{match.group(3)}",
            query,
        )

        if "submittedDate:" not in query:
            query = f"({query}) AND {recency}"

        if len(query) > 1000:
            continue

        normalized.append(query)

    return normalized


def _paper_id(url):
    path = url.rstrip("/").split("/abs/")[-1]
    return path.split("v", 1)[0]


def dedupe_papers(papers):
    seen = set()
    unique = []

    for paper in papers:
        key = _paper_id(paper["url"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)

    return unique


def _paper_from_result(result):
    return {
        "title": result.title,
        "abstract": result.summary,
        "authors": [author.name for author in result.authors],
        "published": str(result.published.date()),
        "url": result.entry_id,
        "query": None,
    }


def retrieve_with_query(query, limit=DEFAULT_PAPERS_PER_QUERY):
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()

    papers = []
    try:
        for result in client.results(search):
            paper = _paper_from_result(result)
            paper["query"] = query
            papers.append(paper)
    except arxiv.HTTPError:
        pass

    return papers


def retrieve_papers(queries, limit=5, papers_per_query=DEFAULT_PAPERS_PER_QUERY):
    """
    Retrieve papers for one or more arXiv query strings.
    Results are deduplicated and truncated to `limit`.
    Also returns which queries failed (returned no papers).
    """
    if isinstance(queries, str):
        queries = [queries]

    all_papers = []
    failed_queries = []
    for query in queries:
        batch = retrieve_with_query(query, limit=papers_per_query)
        if batch:
            all_papers.extend(batch)
        else:
            failed_queries.append(query)

    return dedupe_papers(all_papers)[:limit], failed_queries


def format_papers_for_prompt(papers, max_abstract_chars=400):
    """Format retrieved papers as context for LLM prompts."""
    if not papers:
        return "No retrieved papers available."

    blocks = []
    for index, paper in enumerate(papers, start=1):
        abstract = paper["abstract"]
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars] + "..."

        authors = ", ".join(paper["authors"][:3])
        blocks.append(
            f"Paper {index}:\n"
            f"Title: {paper['title']}\n"
            f"Authors: {authors}\n"
            f"Published: {paper['published']}\n"
            f"URL: {paper['url']}\n"
            f"Abstract: {abstract}"
        )

    return "\n\n".join(blocks)
