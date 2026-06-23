import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from agents import (
    baseline_research_agent,
    detect_homogeneity,
    diversity_expander_agent,
    generate_human_question,
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

if st.button("Run Agent Comparison"):
    with st.spinner("Generating traditional LLM response..."):
        baseline = baseline_research_agent(topic)

    with st.spinner("Detecting idea homogenization..."):
        critique = detect_homogeneity(topic, baseline)

    with st.spinner("Generating cognitively diverse research directions..."):
        expanded = diversity_expander_agent(topic, baseline, critique)

    with st.spinner("Generating researcher-in-the-loop question..."):
        human_question = generate_human_question(topic, baseline, critique)

    st.subheader("1. Traditional LLM Research Directions")
    st.write(baseline)

    st.subheader("2. Homogeneity Critique")
    st.write(critique)

    st.subheader("3. Cognitive-Diversity-Preserving Agent")
    st.write(expanded)

    st.subheader("4. Human-in-the-Loop Question")
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


    st.subheader("Evaluation")
    with st.spinner("Evaluating outputs..."):
        baseline_eval = evaluate_output(
            topic=topic,
            response_text=baseline_response,
            backend="local_ollama",
            model=None,
        )

    st.subheader("Human Evaluation")

    with st.form("human_evaluation_form"):
        st.write("Rate the diversity-preserving agent output: ")

        human_novelty = st.slider("Novelty", 1, 5, 3)
        human_diversity = st.slider("Diversity", 1, 5, 3)
        human_usefulness = st.slider("Usefulness", 1, 5, 3)
        human_assumption_challenge = st.slider("Assumption Challenge", 1, 5, 3)
        human_comment = st.text_area("Optional Comment")
        submitted = st.form_submit_button("Save Human Evaluation")

        if submitted: 
            os.makedirs("results", exist = True)

            row = {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "novelty": human_novelty,
                "diversity": human_diversity,
                "usefulness": human_usefulness,
                "assumption_challenge": human_assumption_challenge,
                "comment": human_comment,
            }

            file_path = "results/human_evaluations.csv"
