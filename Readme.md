# Personal Research Agent

> An AI agent that thinks, plans, and uses tools to answer questions autonomously.

---

## How It Works

Uses the **ReAct framework** (Reasoning + Acting) to chain multiple steps:
```
User Question
     ↓
Agent thinks: "What tool do I need?"
     ↓
Calls tool → Gets result
     ↓
Thinks again → Calls another tool if needed
     ↓
Final Answer
```

## Tools Available

| Tool | Purpose |
|------|---------|
| web_search | Finds current information from the web |
| calculator | Handles mathematical expressions |
| get_current_date | Returns today's date |

## Tech Stack

- **LLM:** Groq LLaMA 3.3 70B (temperature=0)
- **Framework:** LangChain ReAct Agent
- **Search:** DuckDuckGo
- **UI:** Coming Day 14

## Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/personal-research-agent
cd personal-research-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add to `.env`:
```
GROQ_API_KEY=your_key_here
```
```bash
python agent.py
```

## 👨Author
Built as part of a 30-day AI Engineer bootcamp.  
Portfolio: [https://github.com/Salman0452/ai-engineer-portfolio.git]