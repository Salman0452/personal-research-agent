## Live Demo
[Try it here](https://personal-research-agent-cwsq2mku9sjxqinhxtquew.streamlit.app)

# Personal Research Agent

> An AI agent that thinks, plans, and uses tools autonomously — 
> powered by Groq LLaMA 3.3 70B and the ReAct framework.

---

## How It Works

Uses the **ReAct framework** (Reasoning + Acting):
```
User Question
     ↓
Agent thinks: "Which tool do I need?"
     ↓
Calls tool → Gets result
     ↓
Thinks again → Chains another tool if needed
     ↓
Final Answer
```

---

## Tools

| Tool | Purpose |
|------|---------|
| web_search | Current information from the internet |
| calculator | Mathematical expressions |
| get_current_date | Today's date |
| company_document_search | Internal HR policy documents via RAG |

---

## ✨ Features

- **Autonomous tool selection** — decides which tool to use based on your question
- **Tool chaining** — combines multiple tools in one response
- **RAG integration** — searches company documents semantically
- **Hallucination prevention** — grounds answers in real sources
- **Clean Streamlit UI** — chat interface with tool sidebar

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq LLaMA 3.3 70B (temperature=0) |
| Agent Framework | LangChain ReAct Agent |
| Web Search | DuckDuckGo |
| RAG | ChromaDB + Cohere Embeddings |
| UI | Streamlit |

---

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
COHERE_API_KEY=your_key_here
```
```bash
python agent.py        # terminal mode with verbose output
streamlit run app.py   # UI mode
```

---

## Example Queries

- *"What is the employee relocation policy?"* → uses company docs
- *"What is 1250 multiplied by 24?"* → uses calculator
- *"Latest AI trends in Europe?"* → uses web search
- *"Today's date and days until Eid?"* → chains date + calculator

---

## Author
Built as part of a 30-day AI Engineer bootcamp.  
Portfolio: [https://github.com/Salman0452/ai-engineer-portfolio.git]