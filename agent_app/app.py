import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from agents import (
    baseline_research_agent,
    detect_homogeneity,
    diversity_expander_agent,
    generate_human_question,
    retrieve_literature,
)
from evaluator import evaluate_cognitive_diversity

st.title("Cognitive Diversity Research Agent")

st.write(
    "This prototype compares a traditional LLM research assistant with a "
    "cognitive-diversity-preserving research agent."
)

topic = st.text_area(
    "Enter a research topic:",
    "AI agents in education"
)

use_llm_queries = st.checkbox(
    "Use LLM-generated arXiv queries",
    value=True,
    help=(
        "When enabled, Ollama/OpenAI generates search queries. "
        "If that fails, the app falls back to topic-based queries automatically."
    ),
)

paper_limit = st.slider("Number of papers to retrieve", min_value=3, max_value=10, value=5)

if st.button("Run Agent Comparison"):
    with st.spinner("Generating traditional LLM response..."):
        baseline = baseline_research_agent(topic)

    with st.spinner("Detecting idea homogenization..."):
        critique = detect_homogeneity(topic, baseline)

    with st.spinner("Retrieving literature from arXiv..."):
        literature = retrieve_literature(
            topic,
            limit=paper_limit,
            use_llm_queries=use_llm_queries,
        )

    with st.spinner("Generating cognitively diverse, literature-informed research directions..."):
        expanded = diversity_expander_agent(
            topic,
            baseline,
            critique,
            papers=literature["papers"],
        )

    with st.spinner("Generating researcher-in-the-loop question..."):
        human_question = generate_human_question(topic, baseline, critique)

    st.subheader("1. Traditional LLM Research Directions")
    st.write(baseline)

    st.subheader("2. Homogeneity Critique")
    st.write(critique)

    st.subheader("3. Retrieved Literature")
    source_label = {
        "llm": "LLM-generated",
        "fallback": "topic-based fallback",
        "llm+fallback": "LLM-generated with fallback retrieval",
    }.get(literature["query_source"], literature["query_source"])
    st.caption(f"Query source: {source_label}")

    if literature.get("failed_queries"):
        st.warning(
            f"{len(literature['failed_queries'])} arXiv query(s) failed; "
            "results may come from other queries or fallback."
        )

    with st.expander("arXiv search queries used", expanded=False):
        for index, query in enumerate(literature["queries"], start=1):
            st.code(query, language=None)

    if literature["papers"]:
        for index, paper in enumerate(literature["papers"], start=1):
            with st.expander(f"{index}. {paper['title']} ({paper['published']})"):
                st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}")
                st.markdown(f"**URL:** {paper['url']}")
                st.write(paper["abstract"][:600] + ("..." if len(paper["abstract"]) > 600 else ""))
    else:
        st.warning("No papers retrieved. Try a broader topic or disable LLM-generated queries.")

    st.subheader("4. Cognitive-Diversity-Preserving Agent")
    st.caption("Directions below are generated using the retrieved arXiv papers above.")
    st.write(expanded)

    st.subheader("5. Human-in-the-Loop Question")
    st.info(human_question)

    with st.spinner("Evaluating cognitive diversity..."):
        baseline_score = evaluate_cognitive_diversity(topic, baseline)
        expanded_score = evaluate_cognitive_diversity(topic, expanded)

    df = pd.DataFrame([
        {"Agent": "Traditional LLM", **baseline_score},
        {"Agent": "Diversity-Preserving Agent", **expanded_score}
    ])

    st.subheader("Evaluation Scores")
    st.dataframe(df)

    score_df = df.set_index("Agent")[
        ["novelty", "diversity", "usefulness", "assumption_challenge"]
    ]

    fig, ax = plt.subplots()
    score_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Score")
    ax.set_title("Cognitive Diversity Evaluation")
    st.pyplot(fig)

    st.subheader("Evaluation Summary")
    for _, row in df.iterrows():
        st.markdown(f"**{row['Agent']}**: {row['summary']}")
