import streamlit as st
import pandas as pd
import io
import os
import sqlite3
import re
import matplotlib.pyplot as plt
from functools import lru_cache

# LangChain & OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

# -------------------------------------------
# 💰  Pricing helpers
# -------------------------------------------
# ↓  UPDATE these when you switch / add models!
MODEL_PRICES = {
    # dollars per 1K tokens
    "gpt-4.1-mini-2025-04-14": {"prompt": 0.0008, "completion": 0.0032},
    # "gpt-4o":               {"prompt": 0.0025, "completion": 0.0100},
}

def dollars_for_call(model_name: str, usage: dict) -> float:
    """
    Return USD cost for a single OpenAI chat completion call.

    Parameters
    ----------
    model_name : str
        e.g. "gpt-4.1-mini-2025-04-14"
    usage : dict
        {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        (exact structure OpenAI returns via LangChain)
    """
    price = MODEL_PRICES.get(model_name)
    if not price or not usage:
        return 0.0
    in_cost  = usage.get("prompt_tokens", 0)      * price["prompt"]     / 1000
    out_cost = usage.get("completion_tokens", 0)  * price["completion"] / 1000
    return in_cost + out_cost

# ------------------------------------------------------
# 🔑 Credentials helper
# ------------------------------------------------------

def get_credentials():
    """Loads environment variables and sets OPENAI_API_KEY into the current
    process. Nothing is returned – the key is picked‑up automatically by
    LangChain’s ChatOpenAI class once it lives in the env."""
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        st.error("OPENAI_API_KEY not found in environment. Set it in a .env file or in the OS env vars.")
        st.stop()

# ------------------------------------------------------
#  🔍 Database schema introspection helpers
# ------------------------------------------------------

@lru_cache(maxsize=8)
def get_db_schema(db_path: str) -> dict[str, list[str]]:
    """Return {table_name: [columns]} mapping for *SQLite* database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        schema: dict[str, list[str]] = {}
        for t in tables:
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({t})")]
            schema[t] = cols
        return schema


def schema_to_str(schema: dict[str, list[str]]) -> str:
    """Format schema as `table(col, col, …)` lines suitable for prompt."""
    lines = [f"{tbl}({', '.join(cols)})" for tbl, cols in schema.items()]
    return "\n".join(lines)

# ------------------------------------------------------
#  💬 Generic LLM caller
# ------------------------------------------------------

DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

def llm_complete(system_prompt: str,
                 human_prompt: str,
                 *,
                 model_name: str = DEFAULT_MODEL,
                 temperature: float = 0.0,
                 max_tokens: int = 300) -> str:

    get_credentials()
    llm = ChatOpenAI(model_name=model_name,
                     temperature=temperature,
                     max_tokens=max_tokens)
    response = llm.invoke([
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=human_prompt.strip())
    ])

    # ----- cost tracking -----
    usage = response.response_metadata.get("token_usage", {})
    call_cost = dollars_for_call(model_name, usage)
    st.session_state.session_cost += call_cost
    st.session_state.last_call_cost = call_cost
    # -------------------------

    return response.content.strip()


# ------------------------------------------------------
#  🧮 Pandas code generation & execution
# ------------------------------------------------------

PYTHON_PROMPT_TEMPLATE = (
    "You are an expert in converting English questions to Python (pandas) code for talking to a table!\n\n"
    "The DataFrame is already available in the variable `df`. Columns are:\n{columns}\n"
    "Return **ONLY** executable Python code (no ``` fences, no explanations)."
)


def generate_pandas_query(question: str, df: pd.DataFrame) -> str:
    prompt = PYTHON_PROMPT_TEMPLATE.format(columns=", ".join(map(str, df.columns)))
    code = llm_complete(prompt, question)
    # keep only the first non‑empty line – defensive if LLM adds blanks
    lines = [l for l in code.splitlines() if l.strip()]
    return lines[0] if lines else ""


def execute_pandas_query(code: str, df: pd.DataFrame):
    local_vars = {"df": df, "pd": pd}
    try:
        result = eval(code, {}, local_vars)
        return result
    except Exception as e:
        st.error(f"🚨 Error executing generated code:\n{e}")
        return None

# ------------------------------------------------------
#  🖼️ Plot generation (Matplotlib)
# ------------------------------------------------------

PLOT_PROMPT_TEMPLATE = (
    "You are an expert in converting natural language questions into Python code for creating graphs using Matplotlib and Pandas.\n"
    "The DataFrame is available as `df` with columns {columns}.\n"
    "Return ONLY the plotting Python code (no fences / comments / imports)."
)

def generate_plot_code(question: str, df: pd.DataFrame) -> str:
    prompt = PLOT_PROMPT_TEMPLATE.format(columns=", ".join(map(str, df.columns)))
    return llm_complete(prompt, question)


def execute_plot_code(code: str, df: pd.DataFrame):
    local_vars = {"df": df, "plt": plt, "pd": pd}
    try:
        exec(code, {}, local_vars)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"🚨 Plot code execution error:\n{e}")

# ------------------------------------------------------
#  🗄️ SQL generation, validation & execution (multi‑table)
# ------------------------------------------------------

SQL_PROMPT_TEMPLATE = (
    "You are an expert SQL assistant. The database schema is:\n{schema}\n"
    "Use SQLite dialect and JOIN tables as needed to answer the user's request.\n"
    "Return a syntactically correct SQL query only – no code fences, no commentary."
)

REFINE_PROMPT_TEMPLATE = (
    "The previous SQL query produced an error: {error}. Please return a fixed query only (no explanation).\n\n"
    "Original question: {question}\nPrevious query: {query}"
)


def generate_sql_query(question: str, schema_str: str) -> str:
    prompt = SQL_PROMPT_TEMPLATE.format(schema=schema_str)
    return llm_complete(prompt, question)


def execute_sql_query(sql_query: str, db_path: str):
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(sql_query, conn)
            return True, df
        except Exception as e:
            return False, str(e)


def generate_and_validate_sql_query(question: str, db_path: str, max_attempts: int = 3):
    schema_dict = get_db_schema(db_path)
    schema_str = schema_to_str(schema_dict)

    attempt = 0
    query = generate_sql_query(question, schema_str)

    while attempt < max_attempts:
        ok, result = execute_sql_query(query, db_path)
        if ok:
            return query, result
        # refine
        refine_prompt = REFINE_PROMPT_TEMPLATE.format(error=result, question=question, query=query)
        prompt = SQL_PROMPT_TEMPLATE.format(schema=schema_str)  # same schema context
        query = llm_complete(prompt, refine_prompt)
        attempt += 1
        st.info(f"🔄 Attempt {attempt + 1} – refining query…")
    return None, result  # final error message

# ------------------------------------------------------
#  🎛️ Streamlit UI
# ------------------------------------------------------

st.set_page_config(page_title="Talk2Table", page_icon="📊")

# Replace with your own logo / hero image
st.image("D:\\WORK\\PANDASAI\\ChatGPT Image Apr 29, 2025, 01_12_56 PM.png", use_column_width="auto")

# tiny CSS tweak
st.markdown(
    """
    <style>
        .big-font {font-size:20px !important;}
        .stfile_uploader, .sttext_input {width: 300px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Select Your Datasource")

if "mode" not in st.session_state:
    st.session_state.mode = None
    
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "last_call_cost" not in st.session_state:
    st.session_state.last_call_cost = 0.0

with st.sidebar:
    st.metric("💸 Cost of last call", f"${st.session_state.last_call_cost:,.4f}")
    st.metric("📈 Session total",     f"${st.session_state.session_cost:,.4f}")


col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("SQL Database"):
        st.session_state.mode = "sql"
with col3:
    st.markdown('<p class="big-font">OR</p>', unsafe_allow_html=True)
with col4:
    if st.button("Excel / CSV File"):
        st.session_state.mode = "excel"

# ------------- SQL mode (multi‑table) -------------
if st.session_state.mode == "sql":
    db_path = st.text_input("Enter the SQLite database file path:")

    if db_path:
        try:
            schema_dict = get_db_schema(db_path)
        except Exception as e:
            st.error(str(e))
            st.stop()

        with st.expander("📜 Detected schema"):
            for tbl, cols in schema_dict.items():
                st.write(f"**{tbl}**: {', '.join(cols)}")

        question = st.text_input("Ask a question (multi‑table joins supported):")

        if question:
            sql_query, result_or_error = generate_and_validate_sql_query(question, db_path)
            if sql_query:
                st.subheader("Query Results")
                st.dataframe(result_or_error)
                with st.expander("Generated SQL query"):
                    st.code(sql_query, language="sql")
            else:
                st.error(result_or_error)

# ------------- Excel / CSV mode -------------
if st.session_state.mode == "excel":
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

        with st.expander("🔎 Dataframe preview"):
            st.write(df.head())

        q = st.text_area("🗣️ Chat with dataframe (type 'plot …' for visualisations)")
        if q and st.button("Run"):
            if "plot" in q.lower():
                code = generate_plot_code(q, df)
                with st.expander("Generated plot code"):
                    st.code(code, language="python")
                execute_plot_code(code, df)
            else:
                code = generate_pandas_query(q, df)
                with st.expander("Generated pandas code"):
                    st.code(code, language="python")
                result = execute_pandas_query(code, df)
                if isinstance(result, pd.DataFrame):
                    st.subheader("Query results")
                    st.dataframe(result)
                else:
                    st.write(result)
