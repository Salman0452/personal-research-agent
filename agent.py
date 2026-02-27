import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.prompts import PromptTemplate
from langchain_classic import hub

load_dotenv()

# ── 1. LLM ─────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,       # 0 for agents — you want pure logic, zero creativity
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ── 2. DEFINE TOOLS ────────────────────────────────────────────────────────────
# Tool 1: Web Search
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web-search",
    func=search.run,
    description="""Useful for searching current information from the web.
    Use this when you need up to date information about any topic.
    Input should be a search query string."""
)

# Tool 2: Calculator
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely"""
    try:
        # eval is safe here because we control the input scope
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description="""Useful for mathematical calculations.
    Input should be a valid mathematical expression like '2 + 2' or '15 * 8'.
    Always use this for any math instead of calculating yourself."""
)

# Tool 3: Current Date
from datetime import datetime
def get_current_date(_: str) -> str:
    return datetime.now().strftime("%A, %B %d, %Y")

date_tool = Tool(
    name="get_current_date",
    func=get_current_date,
    description="Returns today's date. Use this when you need to know the current date."
)

tools = [search_tool, calculator_tool, date_tool]

# ── 3. AGENT PROMPT ────────────────────────────────────────────────────────────
# We pull the standard ReAct prompt from LangChain hub
# This is the prompt that teaches the LLM to think in Thought/Action/Observation
prompt = hub.pull("hwchase17/react")

# ── 4. CREATE AGENT ────────────────────────────────────────────────────────────
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# AgentExecutor is the runtime that actually runs the ReAct loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # shows Thought/Action/Observation in terminal
    max_iterations=5,       # prevents infinite loops
    handle_parsing_errors=True
)

# ── 5. TEST IT ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1 — forces web search tool
    result = agent_executor.invoke({
        "input": "What is the latest version of Python and when was it released?"
    })
    print("\n🤖 Answer:", result["output"])
    print("\n" + "="*50 + "\n")

    # Test 2 — forces calculator tool
    result = agent_executor.invoke({
        "input": "What is today's date?"
    })
    print("\n🤖 Answer:", result["output"])

