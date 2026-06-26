import os
from datetime import datetime
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
from evaluator import (
    evaluate_cognitive_diversity,
    evaluate_output,
    SCORE_METRICS
)

st.title("Cognitive Diversity Research Agent")

st.write(
    "This prototype compares a traditional LLM research assistant with a "
    "cognitive-diversity-preserving research agent."
)

topic = st.text_area(
    "Enter a research topic:",
    "AI agents in education"
)

# arXiv usage 
use_llm_queries = st.checkbox(
    "Use LLM-generated arXiv queries",
    value=True,
    help=(
        "When enabled, Ollama/OpenAI generates search queries. "
        "If that fails, the app falls back to topic-based queries automatically."
    ),
)

paper_limit = st.slider("Number of papers to retrieve", min_value=3, max_value=10, value=5)

# DF for Comparing traditional LLM and the agent
def make_score_df(baseline_eval, expanded_eval):
    return pd.DataFrame({
        "Metric": SCORE_METRICS,
        "Traditional LLM": [
            baseline_eval["llm_scores"][metric] for metric in SCORE_METRICS
        ],
        "Diversity-Preserving Agent": [
            expanded_eval["llm_scores"][metric] for metric in SCORE_METRICS
        ],
    })

# Human-in-the-loop scoring
def human_evaluation_widget(title, key_prefix):
    st.markdown(f"### {title}")
    human_scores = {}
    cols = st.columns(len(SCORE_METRICS))
    for i, metric in enumerate(SCORE_METRICS):
        with cols[i]:
            human_scores[metric] = st.slider(
                label=metric.replace("_", " ").title(),
                min_value=1,
                max_value=5,
                value=3,
                key=f"{key_prefix}_{metric}",
            )
    avg_score = sum(human_scores.values()) / len(human_scores)

    st.metric(
        label = "Average Human Score",
        value = round(avg_score, 2)
    )

    human_df = pd.DataFrame({
        "Metric": SCORE_METRICS,
        "Human Score": [
            human_scores[metric] for metric in SCORE_METRICS
        ],
    })

    st.dataframe(human_df, use_container_width=True)
    st.bar_chart(human_df.set_index("Metric"))

    return human_scores

# save human evaluation to csv file
def save_human_evaluation(topic, output_type, human_scores, comment):
    os.makedirs("results", exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "output_type": output_type,
        **human_scores,
        "comment": comment,
    }

    file_path = "results/human_evaluations.csv"

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        new_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([row])

    new_df.to_csv(file_path, index=False)

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

    with st.spinner("Evaluating outputs..."):
        baseline_eval = evaluate_output(
            topic=topic,
            response_text=baseline,
            backend="local_ollama",
            model=None,
        )
        expanded_eval = evaluate_output(
            topic=topic,
            response_text=expanded,
            backend="local_ollama",
            model=None,
        )
    
    st.session_state["baseline"] = baseline
    st.session_state["critique"] = critique
    st.session_state["expanded"] = expanded
    st.session_state["human_question"] = human_question
    st.session_state["baseline_eval"] = baseline_eval
    st.session_state["expanded_eval"] = expanded_eval

if "baseline" in st.session_state:
    baseline = st.session_state["baseline"]
    critique = st.session_state["critique"]
    expanded = st.session_state["expanded"]
    human_question = st.session_state["human_question"]
    baseline_eval = st.session_state["baseline_eval"]
    expanded_eval = st.session_state["expanded_eval"]

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

    st.subheader("5. LLM-as-a-Judge Evaluation")
    score_df = make_score_df(baseline_eval, expanded_eval)
    st.dataframe(score_df, use_container_width=True)
    st.bar_chart(score_df.set_index("Metric"))
    with st.expander("Traditional LLM Evaluation Summary"):
        st.write(baseline_eval["llm_scores"].get("summary", ""))
    with st.expander("Diversity-Preserving Agent Evaluation Summary"):
        st.write(expanded_eval["llm_scores"].get("summary", ""))

    st.subheader("6. Diagnostic Metrics")
    diagnostic_df = pd.DataFrame({
        "Metric": ["Word Count", "Diversity Lens Coverage"],
        "Traditional LLM": [
            baseline_eval["simple_metrics"]["word_count"],
            baseline_eval["simple_metrics"]["lens_coverage_count"],
        ],
        "Diversity-Preserving Agent": [
            expanded_eval["simple_metrics"]["word_count"],
            expanded_eval["simple_metrics"]["lens_coverage_count"],
        ],
    })

    st.dataframe(diagnostic_df, use_container_width=True)

    with st.expander("Detected Diversity Lenses"):
        st.write("Traditional LLM")
        st.json(baseline_eval["simple_metrics"]["lens_coverage"])
        st.write("Diversity-Preserving Agent")
        st.json(expanded_eval["simple_metrics"]["lens_coverage"])
        
    st.subheader("7. Human Evaluation")
    human_expanded_scores = human_evaluation_widget(
        "Human Evaluation: Diversity-Preserving Agent",
        key_prefix="human_expanded",
    )

    st.subheader("8. Human vs LLM Evaluation")
    human_vs_llm_df = pd.DataFrame({
        "Metric": SCORE_METRICS,
        "LLM Judge": [
            expanded_eval["llm_scores"][metric] for metric in SCORE_METRICS
        ],
        "Human Evaluation": [
            human_expanded_scores[metric] for metric in SCORE_METRICS
        ],
    })

    st.markdown("### Diversity-Preserving Agent: Human vs LLM")
    st.dataframe(human_vs_llm_df, use_container_width=True)
    st.bar_chart(human_vs_llm_df.set_index("Metric"))

    st.subheader("9. Save Human Evaluation")
    human_comment = st.text_area("Optional comment", key="human_comment")

    if st.button("Save Human Evaluation"):
        save_human_evaluation(
            topic=topic,
            output_type="traditional_llm",
            human_scores=human_expanded_scores,
            comment=human_comment,
        )
        save_human_evaluation(
            topic=topic,
            output_type="diversity_preserving_agent",
            human_scores=human_expanded_scores,
            comment=human_comment,
        )
        st.success("Human evaluation saved.")


# the old evaluator
    # with st.spinner("Evaluating cognitive diversity..."):
    #     baseline_score = evaluate_cognitive_diversity(topic, baseline)
    #     expanded_score = evaluate_cognitive_diversity(topic, expanded)

    # df = pd.DataFrame([
    #     {"Agent": "Traditional LLM", **baseline_score},
    #     {"Agent": "Diversity-Preserving Agent", **expanded_score}
    # ])

    # st.subheader("Evaluation Scores")
    # st.dataframe(df)

    # score_df = df.set_index("Agent")[
    #     ["novelty", "diversity", "usefulness", "assumption_challenge"]
    # ]

    # fig, ax = plt.subplots()
    # score_df.plot(kind="bar", ax=ax)
    # ax.set_ylim(0, 5)
    # ax.set_ylabel("Score")
    # ax.set_title("Cognitive Diversity Evaluation")
    # st.pyplot(fig)

    # st.subheader("Evaluation Summary")
    # for _, row in df.iterrows():
    #     st.markdown(f"**{row['Agent']}**: {row['summary']}")


    # st.subheader("Evaluation")
    # with st.spinner("Evaluating outputs..."):
    #     baseline_eval = evaluate_output(
    #         topic=topic,
    #         response_text=baseline,
    #         backend="local_ollama",
    #         model=None,
    #     )

