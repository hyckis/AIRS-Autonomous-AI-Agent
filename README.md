#### /update_diary

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


## Research flow
1: Traditional LLM
User enters research topic
LLM generates ideas
Agent clusters ideas into themes

2: Homogeneity Critic
Agent detects homogeneity
Cognitive-Diversity-Preserving Agent output
Human-in-the-Loop Question
Evaluation in Traditional LLM & Diversity Agent
Score table & Bar chart & Summary

## TO RUN ON LAB COMPUTER
git pull
cd /agent_app
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  