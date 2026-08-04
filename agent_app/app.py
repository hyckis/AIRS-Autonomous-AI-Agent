import os
from datetime import datetime
import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
from agents import (
    baseline_research_agent,
    detect_homogeneity,
    diversity_expander_agent,
    generate_human_question,
    retrieve_literature,
    strong_prompt_baseline_agent,
)
from evaluator import (
    # evaluate_cognitive_diversity,
    evaluate_output,
    audit_evaluation_results,
    SCORE_METRICS
)
from diversity_metrics import evaluate_idea_set_diversity
from literature_metrics import compute_literature_grounded_metrics
from assumption_bank import generate_assumption_bank

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

# multi agent debate for challenge assumption
run_debate = st.checkbox(
    "Run multi-agent debate for assumption challenge",
    value=False,
)

paper_limit = st.slider("Number of papers to retrieve", min_value=3, max_value=10, value=5)

# DF for Comparing traditional LLM and the agent
def make_score_df(baseline_eval, strong_eval, expanded_eval):
    return pd.DataFrame({
        "Metric": SCORE_METRICS,
        "A: Naive LLM": [
            baseline_eval["llm_scores"][metric] for metric in SCORE_METRICS
        ],
        "B: Strong Prompt": [
            strong_eval["llm_scores"][metric] for metric in SCORE_METRICS
        ],
        "C: Lens Agent": [
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
    with st.spinner("Generating naive LLM response..."):
        baseline = baseline_research_agent(topic)
    
    with st.spinner("Generating strong prompt baseline response..."):
        strong_baseline = strong_prompt_baseline_agent(topic)

    with st.spinner("Detecting idea homogenization..."):
        critique = detect_homogeneity(topic, baseline)
    
    with st.spinner("Building assumption bank..."):
        assumption_bank = generate_assumption_bank(topic, critique)

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

    with st.spinner("Computing non-LLM diversity metrics..."):
        baseline_diversity = evaluate_idea_set_diversity(baseline)
        strong_diversity = evaluate_idea_set_diversity(strong_baseline)
        expanded_diversity = evaluate_idea_set_diversity(expanded)

    with st.spinner("Computing literature-grounded novelty and evidence-support metrics..."):
        baseline_literature_metrics = compute_literature_grounded_metrics(
            baseline_diversity["ideas"],
            literature["papers"],
        )
        strong_literature_metrics = compute_literature_grounded_metrics(
            strong_diversity["ideas"],
            literature["papers"],
        )
        expanded_literature_metrics = compute_literature_grounded_metrics(
            expanded_diversity["ideas"],
            literature["papers"],
        )

    with st.spinner("Evaluating outputs..."):
        baseline_eval = evaluate_output(
            topic=topic,
            response_text=baseline,
            ideas=baseline_diversity["ideas"],
            assumption_bank=assumption_bank,
            backend="local_ollama",
            model=None,
            run_debate=False,
        )
        strong_eval = evaluate_output(
            topic=topic,
            response_text=strong_baseline,
            ideas=strong_diversity["ideas"],
            assumption_bank=assumption_bank,
            backend="local_ollama",
            model=None,
            run_debate=False,
        )
        expanded_eval = evaluate_output(
            topic=topic,
            response_text=expanded,
            ideas=expanded_diversity["ideas"],
            assumption_bank=assumption_bank,
            backend="local_ollama",
            model=None,
            run_debate=run_debate,
        )
    
    st.session_state["baseline"] = baseline
    st.session_state["critique"] = critique
    st.session_state["expanded"] = expanded
    st.session_state["literature"] = literature
    st.session_state["strong_baseline"] = strong_baseline
    st.session_state["human_question"] = human_question
    st.session_state["baseline_literature_metrics"] = baseline_literature_metrics
    st.session_state["strong_literature_metrics"] = strong_literature_metrics
    st.session_state["expanded_literature_metrics"] = expanded_literature_metrics
    st.session_state["assumption_bank"] = assumption_bank
    st.session_state["baseline_eval"] = baseline_eval
    st.session_state["strong_eval"] = strong_eval
    st.session_state["expanded_eval"] = expanded_eval
    st.session_state["baseline_diversity"] = baseline_diversity
    st.session_state["strong_diversity"] = strong_diversity
    st.session_state["expanded_diversity"] = expanded_diversity

if "baseline" in st.session_state:
    baseline = st.session_state["baseline"]
    critique = st.session_state["critique"]
    expanded = st.session_state["expanded"]
    literature = st.session_state["literature"]
    strong_baseline = st.session_state["strong_baseline"]
    human_question = st.session_state["human_question"]
    baseline_literature_metrics = st.session_state["baseline_literature_metrics"]
    strong_literature_metrics = st.session_state["strong_literature_metrics"]
    expanded_literature_metrics = st.session_state["expanded_literature_metrics"]
    assumption_bank = st.session_state["assumption_bank"]
    baseline_eval = st.session_state["baseline_eval"]
    strong_eval = st.session_state["strong_eval"]
    expanded_eval = st.session_state["expanded_eval"]
    baseline_diversity = st.session_state["baseline_diversity"]
    strong_diversity = st.session_state["strong_diversity"]
    expanded_diversity = st.session_state["expanded_diversity"]

    st.subheader("1. Three-Arm Idea Generation Comparison")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### Arm A: Naive LLM")
        st.write(baseline)
    with col_b:
        st.markdown("### Arm B: Strong Prompt")
        st.write(strong_baseline)
    with col_c:
        st.markdown("### Arm C: Lens Agent")
        st.write(expanded)

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

    with st.expander("Assumption Bank (shared across all arms)"):
        print(f"assumption_bank {assumption_bank[0]["id"]}:", assumption_bank)
        if not assumption_bank: st.markdown("No assumption bank generated or parsing failed.")
        for item in assumption_bank:
            st.markdown(f"**{item['assumption']}**")
            st.markdown(item['challenge_criteria'])

    st.subheader("Non-LLM Diversity Metrics")
    diversity_df = pd.DataFrame([
        {
            "Arm": "A: Naive LLM",
            **baseline_diversity["metrics"],
        },
        {
            "Arm": "B: Strong Prompt",
            **strong_diversity["metrics"],
        }, 
        {
            "Arm": "C: Lens Agent",
            **expanded_diversity["metrics"],
        },
    ])
    st.dataframe(diversity_df, use_container_width=True)
    diversity_chart_df = diversity_df.set_index("Arm")[[
        "vendi_score", 
        "core_concept_vendi_score",
        "mean_pairwise_distance", 
        "distinct_2"
    ]]
    st.bar_chart(diversity_chart_df)
    st.caption(
        "Vendi Score estimates the effective number of distinct ideas. "
        "Core Concept Vendi Score recomputes diversity after extracting each idea's title/description-level core concept, "
        "Mean pairwise distance measures semantic spread in embedding space. "
        "Distinct-2 measures lexical diversity. Higher values generally indicate greater diversity. "
        "Self-BLEU is shown in the table; lower Self-BLEU indicates less repetition."
    )

    with st.expander("Extracted Core Concepts"):
        selected_core_arm = st.selectbox(
        "Select arm for core concepts",
        ["A: Naive LLM", "B: Strong Prompt", "C: Lens Agent"],
        key="core_concept_arm",
    )

        if selected_core_arm == "A: Naive LLM":
            core_concepts = baseline_diversity.get("core_concepts", [])
        elif selected_core_arm == "B: Strong Prompt":
            core_concepts = strong_diversity.get("core_concepts", [])
        else:
            core_concepts = expanded_diversity.get("core_concepts", [])

        for i, concept in enumerate(core_concepts, start=1):
            st.markdown(f"**Core concept {i}**")
            st.write(concept)


    st.subheader("Literature-Grounded Metrics")
    literature_metric_df = pd.DataFrame([
        {
            "Arm": "A: Naive LLM",
            "Avg Novelty": baseline_literature_metrics["average_novelty"],
            "Avg Evidence Count": baseline_literature_metrics["average_evidence_count"],
            "Avg Evidence Ratio": baseline_literature_metrics["average_evidence_ratio"],
        },
        {
            "Arm": "B: Strong Prompt",
            "Avg Novelty": strong_literature_metrics["average_novelty"],
            "Avg Evidence Count": strong_literature_metrics["average_evidence_count"],
            "Avg Evidence Ratio": strong_literature_metrics["average_evidence_ratio"],
        },
        {
            "Arm": "C: Lens Agent",
            "Avg Novelty": expanded_literature_metrics["average_novelty"],
            "Avg Evidence Count": expanded_literature_metrics["average_evidence_count"],
            "Avg Evidence Ratio": expanded_literature_metrics["average_evidence_ratio"],
        },
    ])

    st.dataframe(literature_metric_df, use_container_width=True)

    st.bar_chart(
        literature_metric_df.set_index("Arm")[
            ["Avg Novelty", "Avg Evidence Ratio"]
        ]
    )

    st.caption(
        "Novelty is computed as 1 minus the maximum cosine similarity between an idea and the retrieved papers. "
        "Evidence Ratio is the proportion of retrieved papers whose similarity to the idea exceeds the support threshold. "
        "These are retrieved-literature proxies, not claim-level proof."
    )

    with st.expander("Per-Idea Literature-Grounded Details"):
        selected_arm = st.selectbox(
            "Select arm",
            ["A: Naive LLM", "B: Strong Prompt", "C: Lens Agent"],
            key="literature_detail_arm",
        )

        if selected_arm == "A: Naive LLM":
            details = baseline_literature_metrics["per_idea"]
        elif selected_arm == "B: Strong Prompt":
            details = strong_literature_metrics["per_idea"]
        else:
            details = expanded_literature_metrics["per_idea"]

        for idx, item in enumerate(details, start=1):
            st.markdown(f"### Idea {idx}")
            st.write(item["idea"])
            st.markdown(f"**Novelty:** {item['novelty']}")
            st.markdown(f"**Evidence Count:** {item['evidence_count']}")
            st.markdown(f"**Evidence Ratio:** {item['evidence_ratio']}")

            if item["closest_paper"]:
                st.markdown("**Closest Paper:**")
                st.write(item["closest_paper"]["title"])
                st.write(item["closest_paper"]["url"])
                st.write(f"Similarity: {item['closest_paper']['similarity']}")

    st.subheader("LLM-as-a-Judge Evaluation")
    score_df = make_score_df(baseline_eval, strong_eval, expanded_eval)
    st.dataframe(score_df, use_container_width=True)
    st.bar_chart(score_df.set_index("Metric"))
    with st.expander("Traditional LLM Evaluation Summary"):
        st.write(baseline_eval["llm_scores"].get("summary", ""))
    with st.expander("Diversity-Preserving Agent Evaluation Summary"):
        st.write(expanded_eval["llm_scores"].get("summary", ""))

    with st.expander("Per-Idea LLM Judge Details"):
        selected_arm = st.selectbox(
            "Select arm for LLM judge details",
            ["A: Naive LLM", "B: Strong Prompt", "C: Lens Agent"],
            key="llm_detail_arm",
        )

        if selected_arm == "A: Naive LLM":
            idea_scores = baseline_eval["llm_scores"].get("idea_scores", [])
        elif selected_arm == "B: Strong Prompt":
            idea_scores = strong_eval["llm_scores"].get("idea_scores", [])
        else:
            idea_scores = expanded_eval["llm_scores"].get("idea_scores", [])

        if idea_scores:
            st.dataframe(pd.DataFrame(idea_scores), use_container_width=True)
        else:
            st.warning("No per-idea LLM judge details available.")

    st.subheader("Evaluator Sanity Check")
    eval_dfs = {
        "A: Naive LLM": pd.DataFrame(baseline_eval["llm_scores"].get("idea_scores", [])),
        "B: Strong Prompt": pd.DataFrame(strong_eval["llm_scores"].get("idea_scores", [])),
        "C: Lens Agent": pd.DataFrame(expanded_eval["llm_scores"].get("idea_scores", [])),
    }
    expected_counts = {
        "A: Naive LLM": len(baseline_diversity.get("ideas", [])),
        "B: Strong Prompt": len(strong_diversity.get("ideas", [])),
        "C: Lens Agent": len(expanded_diversity.get("ideas", [])),
    }
    for arm_name, eval_df in eval_dfs.items():
        warnings = audit_evaluation_results(
            eval_df=eval_df,
            assumption_bank=assumption_bank,
            expected_idea_count=expected_counts[arm_name]
        )
        with st.expander(f"{arm_name} sanity check"):
            st.write(f"Expected ideas: {expected_counts[arm_name]}")
            st.write(f"Evaluated ideas: {len(eval_df)}")

            if warnings:
                st.warning(f"{len(warnings)} potential evaluator issue(s) found.")
                for warning in warnings: st.write("Warning: ", warning)
            else: st.success("No obvious evaluator issues found.")

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
        
    st.subheader("7. Human Evaluation - Perceived Diversity")
    human_baseline_diversity = st.slider(
        "Human score for Arm A - Naive LLM Diversity",
        1,
        5,
        3,
        key="human_baseline_diversity",
    )
    human_strong_diversity = st.slider(
        "Human score for Arm B - Strong Prompt Diversity",
        1,
        5,
        3,
        key="human_strong_diversity",
    )
    human_agent_diversity = st.slider(
        "Human score for Arm C - Lens Agent",
        1,
        5,
        3,
        key="human_agent_diversity",
    )
    human_diversity_df = pd.DataFrame([
        {
            "Arm": "A - Naive LLM",
            "Human Diversity Score": human_baseline_diversity,
            "Vendi Score": baseline_diversity["metrics"]["vendi_score"],
        }, 
        {
            "Arm": "B - Strong Prompt",
            "Human Diversity Score": human_strong_diversity,
            "Vendi Score": strong_diversity["metrics"]["vendi_score"],
        }, 
        {
            "Arm": "C - Lens Agent",
            "Human Diversity Score": human_agent_diversity,
            "Vendi Score": expanded_diversity["metrics"]["vendi_score"],
        },
    ])
    st.dataframe(human_diversity_df, use_container_width=True)
    
    human_chart_df = human_diversity_df.set_index("Arm")[[
        "Human Diversity Score",
        "Vendi Score",
    ]]
    st.bar_chart(human_chart_df)
    
    # human_expanded_scores = human_evaluation_widget(
    #     "Human Evaluation: Diversity-Preserving Agent",
    #     key_prefix="human_expanded",
    # )

    # st.subheader("8. Human vs Non-LLM Diversity Evaluation")
    # human_vs_llm_df = pd.DataFrame({
    #     "Metric": SCORE_METRICS,
    #     "LLM Judge": [
    #         expanded_eval["llm_scores"][metric] for metric in SCORE_METRICS
    #     ],
    #     "Human Evaluation": [
    #         human_expanded_scores[metric] for metric in SCORE_METRICS
    #     ],
    # })

    # st.markdown("### Diversity-Preserving Agent: Human vs LLM")
    # st.dataframe(human_vs_llm_df, use_container_width=True)
    # st.bar_chart(human_vs_llm_df.set_index("Metric"))

    # st.subheader("9. Save Human Evaluation")
    # human_comment = st.text_area("Optional comment", key="human_comment")

    # if st.button("Save Human Evaluation"):
    #     save_human_evaluation(
    #         topic=topic,
    #         output_type="traditional_llm",
    #         human_scores=human_expanded_scores,
    #         comment=human_comment,
    #     )
    #     save_human_evaluation(
    #         topic=topic,
    #         output_type="diversity_preserving_agent",
    #         human_scores=human_expanded_scores,
    #         comment=human_comment,
    #     )
    #     st.success("Human evaluation saved.")


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

