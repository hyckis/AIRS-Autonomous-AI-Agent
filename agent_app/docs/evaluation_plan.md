## Current Evaluation Metrics
The current prototype evaluates outputs using four LLM-based metrics:
1. Novelty
    Compute each generated idea's embedding-based distance from the closet retrieved prior work
    - Distance from existing literature
        - Max cosine between the idea
        - Top-k retrieved papers
            - Scientific embedding: SPECTER, SPECTER2
    - LLM rubric score, panel-only
    - Novelty = 1 - max_p∈P_k(cos(e_idea, e_p))
    - Facet-level novelty
2. Diversity
    - Vendi Score: https://arxiv.org/abs/2210.02410
        - Effective number of distinct ideas, 
        - choose similarity function
3. Usefulness
    - G-Eval: https://arxiv.org/abs/2303.16634
    - jury/poll LLM-judge
4. Assumption challenge
    - Assumption bank (form-filling paradigm from G-eval)
    - Prompt: idea's assumption; classify challenge depth; rationale; score
    - Multi-agent debate
    - Reward hacking prevention
    - Human evaluation correction
    - Few-shot exemplar
5. Evidence-support
    - Count retrieved sources whose cosine to the idea exceeds a threshold
    - Evidence-support(i)=∣{p∈Pk​:cos(ei​,ep​)>τ}∣
    - SupportCount(i)= ∣{p∈Pk​:cos(ei​,ep​)>τ}∣
    - SupportRatio(i)= ∣{p∈Pk​:cos(ei​,ep​)>τ}∣ / k​
    - RAGAS
    - ALCE
    - panel-only rigorous check

## Future Evaluation Metrics
Layer 1: LLM-as-a-judge
- novelty
- diversity
- usefulness
- assumption_challenge

Layer 2: Simple non-LLM metrics
- output length
- number of distinct ideas
- keyword / lens coverage

Layer 3: Future evidence-aware metrics
- citation grounding
- source diversity
- evidence quality
