from agents import generate_arxiv_queries, retrieve_literature
from retrieval import build_fallback_queries, extract_topic_terms, retrieve_papers

TOPICS = [
    "AI agents in education",
    "protein folding with deep learning",
]


def print_papers(papers):
    for paper in papers:
        print(f"  - {paper['published']} | {paper['title']}")


def test_fallback_queries(topic):
    print("\n" + "=" * 80)
    print(f"FALLBACK QUERIES | TOPIC: {topic}")

    phrases, terms = extract_topic_terms(topic)
    print(f"Phrases: {phrases}")
    print(f"Terms: {terms}")

    queries = build_fallback_queries(topic)
    print(f"\nQueries ({len(queries)}):")
    for index, query in enumerate(queries, start=1):
        print(f"  {index}. {query}")

    papers = retrieve_papers(queries, limit=5)
    print(f"\nRetrieved {len(papers)} papers:")
    print_papers(papers)


def test_retrieve_literature(topic, use_llm_queries):
    label = "LLM" if use_llm_queries else "FALLBACK"
    print("\n" + "=" * 80)
    print(f"retrieve_literature ({label}) | TOPIC: {topic}")

    result = retrieve_literature(topic, limit=5, use_llm_queries=use_llm_queries)
    print(f"Query source: {result['query_source']}")
    print(f"Queries ({len(result['queries'])}):")
    for index, query in enumerate(result["queries"], start=1):
        print(f"  {index}. {query}")

    print(f"\nRetrieved {len(result['papers'])} papers:")
    print_papers(result["papers"])


def test_llm_query_generation(topic):
    print("\n" + "=" * 80)
    print(f"generate_arxiv_queries | TOPIC: {topic}")

    queries, query_source = generate_arxiv_queries(topic)
    print(f"Query source: {query_source}")
    for index, query in enumerate(queries, start=1):
        print(f"  {index}. {query}")


if __name__ == "__main__":
    for topic in TOPICS:
        test_fallback_queries(topic)
        test_retrieve_literature(topic, use_llm_queries=False)

    try:
        test_llm_query_generation(TOPICS[0])
        test_retrieve_literature(TOPICS[0], use_llm_queries=True)
    except Exception as error:
        print(f"\nSkipped LLM tests: {error}")
