#### see /update_diary for daily progress record

## Autonomous AI Research agent Summary
The problem: Monotony in ideas, trend slop (not good business advice)  
Agent vision: Not fully autonomous, interacts with researcher (more like a conversation)

## GOAL
reduce homogenization of ideas, maintain overall diversity of ideas, avoid loss of cognitive diversity, safeguard human jobs

## System Architecture
User Input Research Topic
→ Traditional LLM Generator
→ Homogeneity Critique Agent
→ Diversity-Preserving Agent
→ Human-in-the-Loop Question
→ Evaluation Module
→ Dashboard + Result Log 

Backend implementations of User data storage
Memory Loop ← High-scoring creative directions → Reusable creativity tactics 


## Prototype flow
1: User Research Topic
User enters research topic

2: Traditional LLM
LLM generates ideas
Agent clusters ideas into themes

3: Homogeneity Critic
Agent detects homogeneity
System analyzes baseline output 
 - repeated themes
 - dominant assumptions
 - mainstream research directions
 - missing perspectives
 - possible blind spots

4: Cognitive Diversity Agent
expands the idea space by applying alternative divergence lenses
 - contrarian perspectives
 - historical analogy
 - cross-disciplinary thinking
 - failure modes
 - cultural and social context
 - underrepresented stakeholders
 - long-term consequences

Output: 
 - Human-in-the-Loop Question
 - Evaluation in Traditional LLM & Diversity Agent
 Metrics:
  - novelty
  - diversity
  - usefulness
  - assumption challenge
 - Score table & Bar chart & Summary

## Long-term system goal
1: Research Planning Agent
2: Multi-Source Retrieval Layer
3: Evidence Synthesis Agent
4: Consensus and Assumption Analysis
5: Cognitive Diversity Engine
6: Human Steering Layer
7: Autonomous Exploration Agent
8: Map Generator
9: Evaluation and Diversity Metrics

## TO RUN ON LAB COMPUTER
git pull  
cd /agent_app  
git config --system core.longpaths true
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  

## TO RUN ON POWERSHELL
git pull  
cd /agent_app  
python3 -m venv venv  
.\venv\Scripts\Activate.ps1  
pip install -r requirements.txt