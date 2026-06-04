from agents import (
    baseline_research_agent,
    detect_homogeneity,
    diversity_expander_agent,
    generate_human_question,
)
from evaluator import evaluate_cognitive_diversity

def main():
    topic = "AI agents in education"

    print("Testing baseline agent...")
    baseline = baseline_research_agent(topic)
    print(baseline[:500])

    print("\nTesting homogeneity detector...")
    critique = detect_homogeneity(topic, baseline)
    print(critique[:500])

    print("\nTesting diversity expander...")
    expanded = diversity_expander_agent(topic, baseline, critique)
    print(expanded[:500])

    print("\nTesting human question generator...")
    question = generate_human_question(topic, baseline, critique)
    print(question)

    print("\nTesting evaluator...")
    score = evaluate_cognitive_diversity(topic, expanded)
    print(score)


if __name__ == "__main__":
    main()