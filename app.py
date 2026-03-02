import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic import hub
from datetime import datetime
from rag_tool import load_rag_tool

load_dotenv()

st.set_page_config(page_title="AI Research Agent", page_icon="🤖")
st.title("Personal Research Agent")
st.caption("Searches web, does math, queries company documents — and remembers context.")

# ── LOAD AGENT ─────────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="web_search",
        func=search.run,
        description="Use for current or general information from internet. Input: search query."
    )

    def calculator(expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error: {str(e)}"

    calculator_tool = Tool(
        name="calculator",
        func=calculator,
        description="Use for math calculations. Input: math expression like '15 * 200'."
    )

    def get_current_date(_: str) -> str:
        return datetime.now().strftime("%A, %B %d, %Y")

    date_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="Returns today's date. No input needed."
    )

    rag_search = load_rag_tool()
    rag_tool = Tool(
        name="company_document_search",
        func=rag_search,
        description="""Use for company HR policies, employee rules, relocation,
        travel expenses, disciplinary action. Input: policy question."""
    )

    tools = [search_tool, calculator_tool, date_tool, rag_tool]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True
    )

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# This stores last 3 turns as plain text — injected into every prompt
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── DISPLAY CHAT HISTORY ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── HANDLE INPUT ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            agent_executor = build_agent()

            # Build context string from last 3 turns
            history_text = ""
            if st.session_state.chat_history:
                history_text = "\n\nPrevious conversation:\n"
                # Only keep last 3 turns
                recent = st.session_state.chat_history[-3:]
                for turn in recent:
                    history_text += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n"

            # Inject history into the current question
            full_input = f"{prompt}{history_text}"

            result = agent_executor.invoke({"input": full_input})
            answer = result["output"]

        st.markdown(answer)

    # Save this turn to history
    st.session_state.chat_history.append({
        "human": prompt,
        "assistant": answer
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.title("Available Tools")
st.sidebar.markdown("""
- **Web Search** — current information
- **Calculator** — math expressions
- **Date** — today's date
- **Company Docs** — HR policies
""")
st.sidebar.markdown("---")

# Clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Groq LLaMA 3.3 70B")