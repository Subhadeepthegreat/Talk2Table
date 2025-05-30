from __future__ import annotations


import contextlib, io, os, signal, sqlite3, tempfile, textwrap, traceback, re, ast, sys, multiprocessing, queue as std_queue
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
# import streamlit as st # Will be conditionally imported or mocked
from matplotlib.figure import Figure 
import uuid  
import matplotlib.pyplot as plt
# import google_adk # Assuming placeholder

try:
    from libsql_client import Client, Transaction, ResultSet, LibsqlError
except ImportError:
    Client, Transaction, ResultSet, LibsqlError = None, None, None, None # type: ignore
    print("WARNING: libsql_client not installed. Please install it: pip install libsql-client")

# Turso configuration - loaded from .env via os.getenv()
TURSO_DB_URL = os.getenv("TURSO_DATABASE_URL") # Default to local file if not in .env
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

try:
    import openai
except ImportError:
    openai = None

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Constants & Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME           = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
SQL_ROW_LIMIT        = 5_000
EXECUTION_TIMEOUT_S  = 300
MAX_RETURN_ROWS_CHAT = 200
MAX_ADK_RETRIES      = 2 
os.environ["OPENAI_TRACING"] = "1"

SQL_TABLE_EXTRACTION_RE = re.compile(r"\bFROM\s+([a-zA-Z0-9_]+)\b|\bJOIN\s+([a-zA-Z0-9_]+)\b", re.IGNORECASE)
SQL_COLUMN_EXTRACTION_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b(?=\s*(?:AS\s+\w+|FROM|,|\s+WHERE|\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|=|<|>|<=|>=|<>|LIKE|IN|BETWEEN|IS\s+NULL|IS\s+NOT\s+NULL|DESC|ASC|$))")
SQL_SELECT_FROM_RE = re.compile(r"\bSELECT\b.*?\bFROM\b", re.IGNORECASE | re.DOTALL)
SQL_HARMFUL_PATTERNS = {
    "DROP": re.compile(r"\bDROP\s+(TABLE|DATABASE|INDEX|VIEW)\b", re.IGNORECASE),
    "DELETE_NO_WHERE": re.compile(r"\bDELETE\s+FROM\s+\w+\s*;?\s*$", re.IGNORECASE),
    "TRUNCATE": re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    "UPDATE_NO_WHERE": re.compile(r"\bUPDATE\s+\w+\s+SET\s+.+(?!\s+WHERE\s+)", re.IGNORECASE)
}
SQL_DELETE_WITH_WHERE_RE = re.compile(r"\bDELETE\s+FROM\s+\w+\s+WHERE\s+", re.IGNORECASE)
SQL_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
SQL_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
SQL_GENERAL_WHERE_RE = re.compile(r"\bWHERE\s+1\s*=\s*1\b", re.IGNORECASE)
USER_QUERY_LIMIT_RE = re.compile(r"\b(top|limit|show\s*(?:me\s+)?only|first|last)\s+(\d+)\b", re.IGNORECASE)
PANDAS_HEAD_TAIL_RE = re.compile(r"\.(head|tail)\s*\(\s*(\d+)\s*\)")
PANDAS_FILTER_RE = re.compile(r"\[df\[.*\]\]|\.query\(|\.loc\[|\.iloc\[")



DB_PATH = "conversations_2.db" 
CREATE_SESSIONS_SQL = "CREATE TABLE IF NOT EXISTS sessions (session_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT DEFAULT 'Untitled', data_type TEXT DEFAULT 'Mixed', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
# MODIFIED: Added agent_type column
CREATE_MESSAGES_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    session_id INTEGER NOT NULL, 
    idx INTEGER NOT NULL, 
    role TEXT NOT NULL CHECK(role IN ('user','assistant')), 
    content TEXT, 
    code TEXT, 
    agent_type TEXT, 
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

# Global 'st' and 'ss' for mocking when not in Streamlit context
st = None
ss = {}

if "streamlit" in sys.modules:
    import streamlit as real_st 
    st = real_st
    ss = st.session_state
else:
    class MockStreamlitModule:
        def __init__(self):
            self.session_state = {} 
            self.warning_messages = []
            self.error_messages = []
        def warning(self, msg): self.warning_messages.append(msg); print(f"ST.WARNING_MOCK: {msg}", flush=True)
        def error(self, msg): self.error_messages.append(msg); print(f"ST.ERROR_MOCK: {msg}", flush=True)
        def info(self, msg): print(f"ST.INFO_MOCK: {msg}", flush=True)
        def success(self, msg): print(f"ST.SUCCESS_MOCK: {msg}", flush=True)
        def empty(self): 
            class MockEmpty:
                def chat_message(self, *args, **kwargs): return self
                def __enter__(self): return self
                def __exit__(self, *args, **kwargs): pass
                def markdown(self, txt): print(f"ST.EMPTY.MARKDOWN_MOCK: {txt}", flush=True)
                def empty(self): pass 
            return MockEmpty()
        def __getattr__(self, name): 
            print(f"ST.{name}_MOCK called (no-op)", flush=True)
            return lambda *args, **kwargs: None 
    st = MockStreamlitModule() # type: ignore
    ss = st.session_state # type: ignore

def get_turso_client() -> Client | None:
    """Creates and returns a Turso client instance using globally defined URL and Token."""
    if not Client:
        # Use st.error if st is available, otherwise print
        err_msg = "libsql_client is not installed. Cannot connect to Turso."
        if st and hasattr(st, 'error'): st.error(err_msg)
        else: print(f"ERROR: {err_msg}", flush=True)
        return None
        
    if not TURSO_DB_URL or TURSO_DB_URL == "file:local.db" and not TURSO_DB_URL.startswith("libsql:"):
        # If TURSO_DB_URL is still the default "file:local.db" and not a proper libsql URL,
        # it means it wasn't set in .env for a remote DB.
        # We allow "file:local.db" for local dev without a token.
        if TURSO_DB_URL == "file:local.db":
             print("INFO: Using local SQLite file 'local.db' via Turso client.", flush=True)
        else:
            err_msg = "Turso Database URL (TURSO_DATABASE_URL) is not configured correctly in your .env file."
            if st and hasattr(st, 'error'): st.error(err_msg)
            else: print(f"ERROR: {err_msg}", flush=True)
            return None

    try:
        # Auth token is optional for local file URLs like "file:local.db"
        # but typically required for remote Turso URLs (libsql://...).
        # The Client will handle if token is needed based on URL.
        client = Client(url=TURSO_DB_URL, auth_token=TURSO_AUTH_TOKEN if TURSO_DB_URL.startswith("libsql:") else None)
        return client
    except Exception as e:
        err_msg = f"Failed to create Turso client: {e}"
        if st and hasattr(st, 'error'): st.error(err_msg)
        else: print(f"ERROR: {err_msg}", flush=True)
        return None



def init_db() -> None:
    """Initializes the database schema in Turso."""
    client = get_turso_client()
    if not client:
        return
    try:
        with client.transaction() as tx: # type: ignore
            tx.execute(CREATE_SESSIONS_SQL)
            tx.execute(CREATE_MESSAGES_SQL)
        print("Database schema initialized/verified in Turso.", flush=True)
    except LibsqlError as e: # type: ignore
        err_msg = f"Turso DB Error during schema initialization: {e}"
        if st and hasattr(st, 'error'): st.error(err_msg)
        else: print(f"ERROR: {err_msg}", flush=True)
    except Exception as e:
        err_msg = f"An unexpected error occurred during schema initialization: {e}"
        if st and hasattr(st, 'error'): st.error(err_msg)
        else: print(f"ERROR: {err_msg}", flush=True)

def create_session(t: str = "Untitled", dt: str = "Mixed") -> int | None:
    """Creates a new session in the Turso database."""
    client = get_turso_client()
    if not client:
        return None
    try:
        rs: ResultSet = client.execute("INSERT INTO sessions(title, data_type) VALUES (?, ?)", [t, dt]) # type: ignore
        return rs.last_insert_rowid
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error creating session: {e}")
        else: print(f"ERROR: Turso DB Error creating session: {e}", flush=True)
        return None
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred creating session: {e}")
        else: print(f"ERROR: An unexpected error occurred creating session: {e}", flush=True)
        return None

def update_session_title(session_id: int, title: str) -> None:
    """Updates the title of a session in the Turso database."""
    client = get_turso_client()
    if not client:
        return
    try:
        client.execute("UPDATE sessions SET title = ? WHERE session_id = ?", [title, session_id]) # type: ignore
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error updating session title: {e}")
        else: print(f"ERROR: Turso DB Error updating session title: {e}", flush=True)
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred updating session title: {e}")
        else: print(f"ERROR: An unexpected error occurred updating session title: {e}", flush=True)


# MODIFIED: Added agent_type parameter
def save_message(session_id: int, idx: int, role: str, content: str, code: str | None = None, agent_type: str | None = None) -> None:
    """Saves a chat message to the Turso database."""
    client = get_turso_client()
    if not client:
        return
    try:
        client.execute( # type: ignore
            "INSERT INTO messages(session_id, idx, role, content, code, agent_type) VALUES(?,?,?,?,?,?)",
            (session_id, idx, role, content, code, agent_type),
        )
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error saving message: {e}")
        else: print(f"ERROR: Turso DB Error saving message: {e}", flush=True)
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred saving message: {e}")
        else: print(f"ERROR: An unexpected error occurred saving message: {e}", flush=True)

def delete_session(session_id: int) -> None:
    """Deletes a session and all its associated messages from the Turso database."""
    client = get_turso_client()
    if not client:
        return
    try:
        with client.transaction() as tx: # type: ignore
            tx.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            tx.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        print(f"Session {session_id} and its messages deleted successfully from Turso.", flush=True)
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB error while deleting session {session_id}: {e}")
        else: print(f"ERROR: Turso DB error while deleting session {session_id}: {e}", flush=True)
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred deleting session {session_id}: {e}")
        else: print(f"ERROR: An unexpected error occurred deleting session {session_id}: {e}", flush=True)

# MODIFIED: Fetches code and agent_type, prepares a more complete message dictionary
def load_messages(session_id: int) -> list[dict[str, Any]]:
    """Loads messages for a session from the Turso database."""
    client = get_turso_client()
    if not client:
        return []
    history = []
    try:
        rs: ResultSet = client.execute( # type: ignore
            "SELECT role, content, code, agent_type FROM messages WHERE session_id = ? ORDER BY idx", (session_id,)
        )
        for r_idx, row in enumerate(rs.rows):
            msg = {
                "role": row[0],
                "content": row[1],
                "code": row[2],
                "agent": row[3],
                "dataframe": None,
                "figure": None,
                "verification_notes": [],
                "executed": False,
                "session_id": str(session_id),
            }
            history.append(msg)
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error loading messages: {e}")
        else: print(f"ERROR: Turso DB Error loading messages: {e}", flush=True)
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred loading messages: {e}")
        else: print(f"ERROR: An unexpected error occurred loading messages: {e}", flush=True)
    return history

def next_msg_index(session_id: int) -> int:
    """Gets the next message index for a session from Turso."""
    client = get_turso_client()
    if not client:
        return 0
    try:
        rs: ResultSet = client.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM messages WHERE session_id = ?", (session_id,)) # type: ignore
        if rs.rows and rs.rows[0] and rs.rows[0][0] is not None:
            return int(rs.rows[0][0])
        return 0
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error getting next message index: {e}")
        else: print(f"ERROR: Turso DB Error getting next message index: {e}", flush=True)
        return 0
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred getting next message index: {e}")
        else: print(f"ERROR: An unexpected error occurred getting next message index: {e}", flush=True)
        return 0


def user_msg_count(session_id: int) -> int:
    """Counts user messages in a session from Turso."""
    client = get_turso_client()
    if not client:
        return 0
    try:
        rs: ResultSet = client.execute("SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'", (session_id,)) # type: ignore
        if rs.rows and rs.rows[0] and rs.rows[0][0] is not None:
            return int(rs.rows[0][0])
        return 0
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error counting user messages: {e}")
        else: print(f"ERROR: Turso DB Error counting user messages: {e}", flush=True)
        return 0
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred counting user messages: {e}")
        else: print(f"ERROR: An unexpected error occurred counting user messages: {e}", flush=True)
        return 0
def recent_sessions(limit: int = 20) -> list[tuple]:
    """Fetches recent sessions from Turso."""
    client = get_turso_client()
    if not client:
        return []
    try:
        query = """
        SELECT s.session_id, s.title, 
               strftime('%d %b %H:%M', COALESCE((SELECT ts FROM messages m WHERE m.session_id = s.session_id ORDER BY ts DESC LIMIT 1), s.created_at)) AS last_activity_ts 
        FROM sessions s 
        ORDER BY last_activity_ts DESC 
        LIMIT ?
        """
        rs: ResultSet = client.execute(query, (limit,)) # type: ignore
        return rs.rows
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error fetching recent sessions: {e}")
        else: print(f"ERROR: Turso DB Error fetching recent sessions: {e}", flush=True)
        return []
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred fetching recent sessions: {e}")
        else: print(f"ERROR: An unexpected error occurred fetching recent sessions: {e}", flush=True)
        return []

# Initialize DB first
if "db_initialized" not in ss: # type: ignore
    init_db()
    ss["db_initialized"] = True # type: ignore

# Initialize session state variables
for k, v in {"chat_history": [], "costs": [], "total_cost": 0.0, "schemas": {}, "tables": {}, "db_paths": {}, "session_id": None, "current_session_id": None}.items():
    ss.setdefault(k, v) # type: ignore

if ss.get("session_id") is None: # type: ignore
    new_db_session_id = create_session()
    if new_db_session_id is not None:
        ss["session_id"] = new_db_session_id # type: ignore
        ss["current_session_id"] = str(new_db_session_id) # type: ignore
        print(f"Initial DB session created in Turso: ID {new_db_session_id}", flush=True)
    else:
        if st and hasattr(st, 'error'): st.error("Failed to create initial session in Turso.")
        else: print("ERROR: Failed to create initial session in Turso.", flush=True)


if not ss.get("chat_history") and ss.get("session_id") is not None: # type: ignore
    ss["chat_history"] = load_messages(ss["session_id"]) # type: ignore


def infer_db_schema_turso(client: Client) -> dict:
    """Infers schema from a Turso database."""
    if not client:
        return {}
    
    schema_info = {}
    try:
        tables_rs: ResultSet = client.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_litestream_%' AND name NOT LIKE 'libsql_%';") # type: ignore
        table_names = [row[0] for row in tables_rs.rows if row[0]]

        for table_name in table_names:
            cols_rs: ResultSet = client.execute(f"PRAGMA table_info({table_name});") # type: ignore
            columns = [row[1] for row in cols_rs.rows] 
            pks = [row[1] for row in cols_rs.rows if row[5]] 
            unique_cols = []
            try:
                index_list_rs: ResultSet = client.execute(f"PRAGMA index_list({table_name});") # type: ignore
                for index_row in index_list_rs.rows:
                    if index_row[2] == 1: 
                        index_name = index_row[1]
                        index_info_rs: ResultSet = client.execute(f"PRAGMA index_info({index_name});") # type: ignore
                        for index_info_row in index_info_rs.rows:
                            unique_cols.append(index_info_row[2]) 
            except Exception as e_idx:
                print(f"Could not reliably determine unique columns for {table_name} via PRAGMA index_list/info: {e_idx}", flush=True)

            count_rs: ResultSet = client.execute(f"SELECT COUNT(*) FROM {table_name};") # type: ignore
            row_count = count_rs.rows[0][0] if count_rs.rows and count_rs.rows[0] else 0
            
            candidate_uniques = [] # Skipping for performance by default

            schema_info[table_name] = {
                "cols": columns, "pk": pks, "uniques": list(set(unique_cols)),
                "candidates": candidate_uniques, "row_count": row_count
            }
    except LibsqlError as e: # type: ignore
        if st and hasattr(st, 'error'): st.error(f"Turso DB Error inferring schema: {e}")
        else: print(f"ERROR: Turso DB Error inferring schema: {e}", flush=True)
        return {}
    except Exception as e:
        if st and hasattr(st, 'error'): st.error(f"An unexpected error occurred inferring schema: {e}")
        else: print(f"ERROR: An unexpected error occurred inferring schema: {e}", flush=True)
        return {}
    return schema_info

def infer_db_schema(db_identifier: Any = None): 
    if Client and isinstance(db_identifier, Client): # type: ignore
        return infer_db_schema_turso(db_identifier)
    else: 
        client = get_turso_client()
        if client:
            return infer_db_schema_turso(client)
    return {}

# ─────────────────────────────────────────────────────────────────────────────
#  Multiprocessing Execution Target
# ─────────────────────────────────────────────────────────────────────────────
def _mp_exec_target(code_to_exec_str: str, df_obj: pd.DataFrame | None, env_keys: list[str], result_q: multiprocessing.Queue): # type: ignore
    """
    Target function for multiprocessing to execute code in a sandboxed environment.
    Imports necessary modules and sets up a global environment for exec.
    """
    try:
        from matplotlib.figure import Figure # Import Figure here
        
        globals_for_exec = {}
        if 'pd' in env_keys:
            import pandas as pd_module
            globals_for_exec['pd'] = pd_module
        if 'plt' in env_keys:
            import matplotlib.pyplot as plt_module
            globals_for_exec['plt'] = plt_module
        
        if df_obj is not None and 'df' in env_keys:
            globals_for_exec['df'] = df_obj

        local_capture_dict = {}
        exec(code_to_exec_str, globals_for_exec, local_capture_dict)
        final_result = local_capture_dict.get("_ret")
        
        if isinstance(final_result, Figure):
            try:
                # Ensure the figure is fully rendered before pickling
                final_result.canvas.draw()
                print("[_mp_exec_target] Called canvas.draw() on Figure object.", flush=True)
            except AttributeError:
                 # This can happen if the figure is using a non-drawing backend or has no canvas
                 print("[_mp_exec_target] Figure has no canvas or draw method, skipping canvas.draw().", flush=True)
            except Exception as e:
                print(f"[_mp_exec_target] Error calling fig.canvas.draw(): {e}", flush=True)
                
        result_q.put(final_result)
    except Exception as e:
        # Ensure any exception during exec or setup is sent back
        result_q.put(e)

def execute_plot_code(code: str, df: pd.DataFrame, plt_module: Any, st_module: Any) -> Any:
    """
    Executes the provided Python code string designed to generate a Matplotlib plot.

    Args:
        code: The Python code string to execute.
        df: The Pandas DataFrame to be used as 'df' in the executed code.
        plt_module: The imported matplotlib.pyplot module.
        st_module: The imported streamlit module (used for error reporting).

    Returns:
        The Matplotlib figure object if successful, None otherwise.
    """
    local_vars = {"df": df, "plt": plt_module, "pd": pd} # Assuming pd is available globally or passed
    fig = None
    try:
        # Ensure the code string is suitable for exec.
        # If it's an expression, it might need to be assigned to a variable.
        # For now, assume the code handles figure creation and plt.gcf() or similar
        # is implicitly or explicitly part of the 'code'.
        exec(code, {}, local_vars)
        fig = plt_module.gcf()  # Get the current figure
        return fig
    except Exception as e:
        st_module.error(f"🚨 Plot code execution error:\n{e}")
        # Optionally, print traceback for server-side logging
        # import traceback
        # print(f"Plot code execution error: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class InvalidPandasOutputError(ValueError):
    """Custom exception for when Pandas code does not return a DataFrame or Series."""
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class InvalidPandasOutputError(ValueError):
    """Custom exception for when Pandas code does not return a DataFrame or Series."""
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  ADK Agent Base Class & Subclasses
# ─────────────────────────────────────────────────────────────────────────────
def infer_excel_schema(b,f):
    dfs,s={}, {}; # type: ignore
    xls=pd.ExcelFile(io.BytesIO(b)); # type: ignore
    for sh in xls.sheet_names:df=xls.parse(sh); # type: ignore
    dfs[f"{f}:{sh}"]=df; # type: ignore
    s[sh]={"cols":df.columns.tolist(),"pk":[],"uniques":[],"candidates":[c for c in df.columns if df[c].is_unique]}; # type: ignore
    return s,dfs # type: ignore
def infer_csv_schema(b,f):
    df=pd.read_csv(io.BytesIO(b)); # type: ignore
    return{"File":{"cols":df.columns.tolist(),"pk":[],"uniques":[],"candidates":[c for c in df.columns if df[c].is_unique]}},df # type: ignore

class ADKAgent:
    def __init__(self,mn=MODEL_NAME,ok=OPENAI_API_KEY):self.model_name=mn;self.openai_api_key=ok
    def call_openai_llm(self,up,sp,mt=1024): # type: ignore
        if not self.openai_api_key or not openai: print("Warning: OPENAI_API_KEY not set or OpenAI module not available. Echoing prompt.", flush=True); return up,0.0
        try:r=openai.chat.completions.create(model=self.model_name,messages=[{"role":"system","content":sp},{"role":"user","content":up}],max_tokens=mt);return r.choices[0].message.content.strip(),(r.usage.total_tokens if r.usage else 0)*0.00001 # type: ignore
        except Exception as e:print(f"OpenAI API Error: {e}", flush=True);return f"Error: OpenAI API call failed. Details: {e}",0.0 # type: ignore

    def _sandbox_exec(self, code_str_in: str, env: dict[str, Any]): # type: ignore
        """
        Executes a string of Python code in a sandboxed environment using multiprocessing
        for timeout control.
        """
        # Prepare the code string: split lines, dedent, and append _ret for capturing the last expression.
        lines = code_str_in.strip().split("\n")
        if len(lines) > 1:
            body, last_line = lines[:-1], lines[-1]
            code_to_exec_str = textwrap.dedent("\n".join(body) + f"\n_ret={last_line}")
        else:
            code_to_exec_str = textwrap.dedent(f"_ret={lines[0]}")

        # Extract necessary components from the environment for the subprocess
        df_obj = env.get('df')  # Assuming 'df' is the key for pandas DataFrame
        # Pass all original env keys so the target can try to reconstruct them
        env_keys = list(env.keys())

        result_q = multiprocessing.Queue()
        
        # Ensure the context is fork for macOS/Windows compatibility if not already default
        # context = multiprocessing.get_context("fork") # Use 'spawn' or 'forkserver' if 'fork' is problematic
        # p = context.Process(target=_mp_exec_target, args=(code_to_exec_str, df_obj, env_keys, result_q))
        p = multiprocessing.Process(target=_mp_exec_target, args=(code_to_exec_str, df_obj, env_keys, result_q))


        p.start()
        p.join(timeout=EXECUTION_TIMEOUT_S)

        if p.is_alive():
            p.terminate() # Terminate the process if it's still alive (timed out)
            p.join(timeout=1) # Wait a bit for termination
            if p.is_alive(): # Still alive after terminate?
                 p.kill() # Force kill
                 p.join() # Wait for kill
            raise TimeoutError("Code execution timed out.")
        else:
            try:
                # Non-blocking get from queue; process finished, item should be there.
                result = result_q.get_nowait() 
                if isinstance(result, Exception):
                    # If the subprocess put an exception in the queue, re-raise it.
                    raise result
                return result
            except std_queue.Empty:
                # This case means the process finished (or crashed) but didn't put anything in the queue.
                # It might happen if the process is killed abruptly or crashes before result_q.put().
                # Check exit code if available and meaningful.
                exitcode = p.exitcode
                if exitcode is not None and exitcode != 0:
                    raise RuntimeError(f"Execution process ended with non-zero exit code {exitcode} and no result. Potential crash in subprocess.")
                raise RuntimeError("Execution process ended unexpectedly without a result.")
            except Exception as e: # Catch any other errors during result retrieval
                raise RuntimeError(f"Failed to retrieve result from execution process: {e}")

    # def _alternative_sandbox_exec_stub(self, code_str_in: str, env: dict[str, Any]):
    #     # This is a placeholder for an alternative sandboxing mechanism.
    #     # It would execute code_str_in in a different sandboxed environment.
    #     #
    #     # Args:
    #     #     code_str_in: The string of Python code to execute.
    #     #     env: A dictionary representing the global environment for the code.
    #     #
    #     # Returns:
    #     #     The result of the execution.
    #     #
    #     # Raises:
    #     #     TimeoutError: If execution exceeds a time limit.
    #     #     Exception: For other execution errors.
    #
    #     # --- Begin custom sandboxing logic (example placeholder) ---
    #     # if "some_condition_for_this_sandbox":
    #     #     try:
    #     #         # ... (preparation of code_to_exec_str as in _sandbox_exec) ...
    #     #         # ... (actual sandboxed execution logic) ...
    #     #         # result = ...
    #     #         # return result
    #     #         pass # Replace with actual logic
    #     #     except Exception as e:
    #     #         # ... (error handling) ...
    #     #         raise e
    #     # --- End custom sandboxing logic ---
    #
    #     # Fallback or error if not implemented:
    #     print("WARNING: _alternative_sandbox_exec_stub is a stub and not implemented.", flush=True)
    #     raise NotImplementedError("This sandboxing method is a stub.")

    def run(self,uq,**kw):raise NotImplementedError

class SQLAgent(ADKAgent): 
    system_prompt_template=(
        "You are an expert SQL assistant. The database schema is:\n{schema}\n"
        "Use SQLite dialect and JOIN tables as needed to answer the user's request.\n"
        "Return a syntactically correct SQL query only – no code fences, no commentary."
    )
    _v_sql_syntax=lambda s,c:None if SQL_SELECT_FROM_RE.search(c)else"SQL Syntax Issue...";_v_schema_conf=lambda s,c,sc:next((f"Schema Error: Table '{t}' not in DB."for t in{tbl for grp in SQL_TABLE_EXTRACTION_RE.findall(c)for tbl in grp if tbl}if t not in sc)or next((f"Schema Error: Col '{cl}' missing."for t in{tbl for grp in SQL_TABLE_EXTRACTION_RE.findall(c)for tbl in grp if tbl}if t in sc for cl in{col[0]if isinstance(col,tuple)else col for col in SQL_COLUMN_EXTRACTION_RE.findall(c)}if not(cl.upper()in{'SELECT','FROM','WHERE'}or cl.isdigit()or cl in sc[t].get("cols",[]))and any(f"{t}.{cl}"in c or f" {cl} "in c)and not any(qt in sc and cl in sc[qt].get("cols",[])for qt in{tbl for grp in SQL_TABLE_EXTRACTION_RE.findall(c)for tbl in grp if tbl})) ,None),None);_v_harmful_sql=lambda s,c:next((m for p,m in SQL_HARMFUL_PATTERNS.items()if re.search(p,c,re.I)and not(p==SQL_HARMFUL_PATTERNS["DELETE_NO_WHERE"]and SQL_DELETE_WITH_WHERE_RE.search(c))),None);_v_df_r=lambda s,d,u,c:f"Note: Expected {er} rows, got {len(d)}."if(er:=(int(ls.group(1))if(ls:=SQL_LIMIT_RE.search(c))else int(lr.group(2))if(lr:=USER_QUERY_LIMIT_RE.search(u))else None))and len(d)<er else None;_v_df_e=lambda s,d,c:"Note: Query returned no data."if d.empty and(not SQL_WHERE_RE.search(c)or SQL_GENERAL_WHERE_RE.search(c))else None
    def run(self, user_query, schema_info, db_path, db_name):# type: ignore
        sp=self.system_prompt_template.format(schema=textwrap.indent(str(schema_info),"    "));c,cost=self.call_openai_llm(user_query,sp);rp={"agent":"sql","code":c,"dataframe":None,"figure":None,"cost":cost,"verification_notes":[]}
        if "Error:"in c:rp["content"]=f"LLM Error:{c}";return rp

        # Clean SQL code from LLM
        # Handles optional "sql" and varying newlines/spaces around fences.
        match = re.search(r"^(?:```(?:sql)?\s*\n)?(.*?)(?:\s*\n```)?$", c, re.DOTALL | re.IGNORECASE)
        if match:
            c = match.group(1).strip()
        else:
            c = c.strip() # Fallback to simple strip if regex doesn't match (e.g. no fences)
        rp["code"] = c # Update code in response payload as well

        for vf,va in[(self._v_sql_syntax,c),(self._v_harmful_sql,c),(self._v_schema_conf,(c,schema_info))]:
            # The validation function vf is called with arguments va.
            # It returns an error message string if validation fails, or None if it passes.
            validation_error_message = vf(*va) if isinstance(va, tuple) else vf(va)
            if validation_error_message:
                # If there's a validation error, set the content and return the response payload.
                rp["content"] = f"Validation Error:{validation_error_message}"
                return rp
        # If the loop completes without returning, it means all validations passed.
        # rp["content"] should not be set with a validation error at this point.
        # The original rp["content"] (likely None or an LLM error message if that check failed earlier)
        # will be overwritten by the success message or execution error message later in the try/except block.
        try:df=self._execute_sql(db_path,c);vn=[];rp.update({"dataframe":df,"content":f"SQL on **{db_name}**:\n```sql\n{c}\n```\nReturned {len(df)} rows."});[vn.append(n)for n in[self._v_df_r(df,user_query,c),self._v_df_e(df,c)]if n];rp["verification_notes"]=vn
        except Exception as e:rp["content"]=f"❗ SQL Error:{e}\n{traceback.format_exc()}"
        return rp
    def _execute_sql(self,p,q):cq=q.strip("` ")[:-1]if q.strip("` ").endswith(";")else q.strip("` ");conn=sqlite3.connect(p);return pd.read_sql(f"SELECT * FROM({cq})LIMIT {SQL_ROW_LIMIT}",conn)
class PandasAgent(ADKAgent): 
    system_prompt_template = (
        "You are an expert in converting English questions to Python (pandas) code for talking to a table!\n\n"
        "The DataFrame is already available in the variable `df`. Columns are:\n{columns}\n"
        "Return **ONLY** executable Python code (no ``` fences, no explanations, no import statements)."
    )
    _v_py_syntax=lambda s,c:None if not(e:=_exec_py_syntax_check(c))else f"Python Syntax Error:{e}";
    _v_col_exists=lambda s,c,cs:next((f"Column Error:'{cn}' not in DataFrame."for n in ast.walk(ast.parse(c))if isinstance(n,ast.Subscript)and isinstance(n.value,ast.Name)and n.value.id=='df'and isinstance(n.slice,ast.Constant)and isinstance(n.slice.value,str)and(cn:=n.slice.value)not in cs),None)if not _exec_py_syntax_check(c)else None;
    _v_df_r=lambda s,d,u,c:f"Note:Code might limit to {er} rows,result has {len(d)}."if(er:=(int(next(filter(None,lc.groups()),"0")[1])if(lc:=PANDAS_HEAD_TAIL_RE.search(c))else int(lr.group(2))if(lr:=USER_QUERY_LIMIT_RE.search(u))else None))and len(d)<er else None;
    _v_df_e=lambda s,d,c:"Note:Operation returned empty."if d.empty and not PANDAS_FILTER_RE.search(c)else None
    def run(self,u,df_n,df_i): # type: ignore
        sp=self.system_prompt_template.format(columns=list(df_i.columns));c,cost=self.call_openai_llm(u,sp);rp={"agent":"pandas","code":c,"dataframe":None,"figure":None,"cost":cost,"verification_notes":[]}
        
        if "Error:"in c: # Check for LLM error first
            rp["content"]=f"LLM Error:{c}";return rp

        # +++ START OF NEW CODE CLEANING +++
        # 1. Remove markdown fences
        #    Handles optional "python" and varying newlines/spaces around fences.
        match = re.search(r"^(?:```(?:python)?\s*\n)?(.*?)(?:\s*\n```)?$", c, re.DOTALL | re.IGNORECASE)
        if match:
            c = match.group(1).strip()
        else:
            c = c.strip() # Fallback to simple strip if regex doesn't match (e.g. no fences)

        # 2. Remove pandas import statements
        lines = c.split('\n')
        lines = [line for line in lines if not re.match(r"^\s*import\s+pandas(?:\s+as\s+pd)?\s*$", line)]
        c = '\n'.join(lines).strip()
        # +++ END OF NEW CODE CLEANING +++

        # Update the code in the response payload as well, because validation might happen on 'c'
        rp["code"] = c 

        for vf,va in[(self._v_py_syntax,c),(self._v_col_exists,(c,list(df_i.columns)))]:
            e=vf(*va)if isinstance(va,tuple)else vf(va)
            if e: # If a validation error occurs
                rp["content"]=f"Validation Error:{e}"
                return rp # Return the response payload with the error
            # If 'e' is None (validation passed), the loop continues to the next check
            # or execution proceeds if all checks pass.
        print(f"[PandasAgent] Attempting to execute Pandas code:\n{c}", flush=True)
        try:
            print(f"[PandasAgent] Processing DataFrame '{df_n}' with shape: {df_i.shape}", flush=True)
            rdf=self._execute_pandas(df_i,c);vn=[];rp.update({"dataframe":rdf,"content":f"Pandas on **{df_n}**:"});[vn.append(n)for n in[self._v_df_r(rdf,u,c),self._v_df_e(rdf,c)]if n];rp["verification_notes"]=vn
        except InvalidPandasOutputError as e: # Catch specific error
            rp["content"]=f"❗ Pandas Execution Error: {e}\nCode:\n```python\n{c}\n```" # More specific error message
        except Exception as e:rp["content"]=f"❗ Pandas Error:{e}\n{traceback.format_exc()}"
        return rp
    
    def _execute_pandas(self, df: pd.DataFrame, code_str: str): # type: ignore
        # --- Experimental Sandboxing Hooks (Commented Out) ---
        # # Set a flag to choose execution mode, e.g., by class attribute or config
        # execution_mode = "direct"  # Options: "direct", "sandbox_original", "sandbox_alternative"
        #
        # if execution_mode == "sandbox_original":
        #     # Option 1: Use the original multiprocessing sandbox
        #     # This was found to be slow for large DataFrames.
        #     print("[PandasAgent] EXPERIMENTAL: Using original _sandbox_exec", flush=True)
        #     return self._sandbox_exec(code_str, {"pd": pd, "df": df.copy()}) # Pass original df and pd
        # elif execution_mode == "sandbox_alternative":
        #     # Option 2: Use the alternative sandbox stub
        #     # This requires implementing _alternative_sandbox_exec_stub
        #     print("[PandasAgent] EXPERIMENTAL: Using _alternative_sandbox_exec_stub", flush=True)
        #     return self._alternative_sandbox_exec_stub(code_str, {"pd": pd, "df": df.copy()})
        #
        # # If execution_mode is "direct" or no specific mode matched,
        # # it will fall through to the direct execution code below.
        # # Ensure the direct execution code is still present and active by default.
        # --- End Experimental Sandboxing Hooks ---
        env = {"pd": pd, "df": df.copy()}
        
        # Prepare code to capture the result of the last expression
        lines = code_str.strip().split('\n')
        if len(lines) > 1:
            body, last_line = lines[:-1], lines[-1]
            # Ensure last_line is not empty or just whitespace
            if last_line.strip():
                code_to_exec_str = textwrap.dedent("\n".join(body) + f"\n_ret={last_line}")
            else: # Only whitespace or empty last line
                code_to_exec_str = textwrap.dedent("\n".join(body)) # Execute body only
                # Potentially set _ret to None or handle cases where no expression is returned
                env['_ret'] = None # Initialize _ret in case the code doesn't assign to it
        elif lines and lines[0].strip(): # Single line of code
            code_to_exec_str = textwrap.dedent(f"_ret={lines[0]}")
        else: # Empty or whitespace-only code_str
            raise InvalidPandasOutputError("Pandas code was empty or contained only whitespace.")

        local_vars = {}
        try:
            exec(code_to_exec_str, env, local_vars)
            result = local_vars.get("_ret")
        except Exception as e:
            # Re-raise execution errors to be caught by the calling `run` method
            raise e

        if isinstance(result, pd.Series):
            return result.to_frame()
        if isinstance(result, pd.DataFrame):
            return result
        
        # If not a Series or DataFrame, raise the custom error
        raise InvalidPandasOutputError(f"Pandas code did not return a DataFrame or Series. Got type: {type(result)}. Executed code:\n{code_str}")

def _exec_py_syntax_check(c):
    try:
        ast.parse(c)
        return None
    except SyntaxError as e:
        return e.msg # Changed from e.message to e.msg
class PlotAgent(ADKAgent): 
    system_prompt_template=(
        "You are an expert in converting natural language questions into Python code for creating graphs using Matplotlib and Pandas.\n"
        "The DataFrame is available as `df` with columns {columns}.\n"
        "Return ONLY the plotting Python code (no fences / comments / imports).\n"
        "IMPORTANT: Ensure the Matplotlib figure object is the VERY LAST line of your code block (e.g., assign it to a variable `fig` and make `fig` the last line, or end with `plt.gcf()`).\n"
        "Do NOT include `plt.show()` in your code."
    )
    _v_plot_sanity=lambda s,f:([n for n in["Critical Plot Error:Not a Figure."if not isinstance(f,Figure)else"Plotting Note:Plot empty(no axes)."if not f.get_axes()else None]if n])
    def run(self, user_query, df_ref_name, df_initial): # type: ignore
        sp=self.system_prompt_template.format(columns=list(df_initial.columns));c,cost=self.call_openai_llm(user_query,sp);rp={"agent":"plot","code":c,"dataframe":None,"figure":None,"cost":cost,"verification_notes":[]}
        if "Error:"in c:rp["content"]=f"LLM Error:{c}";return rp

        # Clean Python code from LLM
        # Handles optional "python" and varying newlines/spaces around fences.
        match = re.search(r"^(?:```(?:python)?\s*\n)?(.*?)(?:\s*\n```)?$", c, re.DOTALL | re.IGNORECASE)
        if match:
            c = match.group(1).strip()
        else:
            c = c.strip() # Fallback to simple strip if regex doesn't match (e.g. no fences)
        rp["code"] = c # Update code in response payload as well

        pa=PandasAgent()
        for vf,va in[(pa._v_py_syntax,c),(pa._v_col_exists,(c,list(df_initial.columns)))]:
            e=vf(*va)if isinstance(va,tuple)else vf(va)
            if e: # If a validation error occurs
                rp["content"]=f"Validation Error (Plot Code):{e}"
                return rp # Return the response payload with the error
            # If 'e' is None (validation passed), the loop continues to the next check
            # or execution proceeds if all checks pass.
        try:fig=self._execute_plot(df_initial,c);vn=self._v_plot_sanity(fig);rp.update({"figure":fig,"verification_notes":vn});rp["content"]="\n".join(vn)if any("Critical"in n for n in vn)else f"Plot from **{df_ref_name}**."
        except Exception as e:rp["content"]=f"❗ Plot Error:{e}\n{traceback.format_exc()}";rp["verification_notes"].extend(self._v_plot_sanity(locals().get('fig')))
        return rp
    def _execute_plot(self, df: pd.DataFrame, code_str: str):
        """
        Executes the plot code using the new execute_plot_code function.
        Ensures that plt and st are passed correctly.
        """
        # Ensure 'plt' and 'st' are available.
        # 'plt' is 'matplotlib.pyplot' and 'st' is 'streamlit'.
        # These are typically imported at the top of talk2table_ui.py.
        import matplotlib.pyplot as plt_module # Ensure it's the module
        import streamlit as st_module # Ensure it's the module
        
        # Make sure pandas is available as pd for execute_plot_code's local_vars
        import pandas as pd_module

        # Call the new function
        # The execute_plot_code function now handles st.pyplot() directly.
        # It also returns the figure object.
        fig = execute_plot_code(code_str, df.copy(), plt_module, st_module)

        if fig is not None and isinstance(fig, Figure):
            return fig
        elif fig is None:
            # Error already handled by st.error in execute_plot_code
            # We might want to raise an exception to signal failure to the caller
            # so that it doesn't try to use a None figure.
            raise ValueError("Plot generation failed. See error message above.")
        else:
            # This case handles if execute_plot_code returns something unexpected.
            raise ValueError(f"Plot code did not return a Figure object. Got type: {type(fig)}")


def chat_renderer(msg: dict[str, Any], idx: int) -> None:
    """Renders a single chat message, including verification notes and re-run buttons."""
    # Ensure st is the real Streamlit or a working mock
    global st 
    if "streamlit" in sys.modules and st.__class__.__name__ == "MockStreamlitModule": # Re-assign if real st became available
        import streamlit as real_st_render
        st = real_st_render

    with st.chat_message(msg["role"]): # type: ignore
        # Determine if this is a pandas scenario where msg["content"] is identical to msg["code"]
        # and msg["content"] is not an error message itself.
        is_pandas_pure_code_content = \
            msg.get("agent") == "pandas" and \
            msg.get("code") is not None and \
            msg.get("content") == msg.get("code") and \
            not any(err_keyword in msg.get("content", "").lower() for err_keyword in ["error:", "failed:", "exception:"])

        if not is_pandas_pure_code_content:
            # For non-pandas agents, or pandas error messages (where content is the error),
            # or pandas cases where content might legitimately differ from code, render content using markdown.
            st.markdown(msg["content"])
        # If it IS a pandas pure code content scenario, we skip the initial st.markdown(msg["content"]).
        # The code will be displayed by a dedicated st.code() call later.
        
        if msg.get("role") == "assistant" and (verification_notes := msg.get("verification_notes", [])):
            for note in verification_notes:
                if "Critical" in note: st.error(f"⚠️ {note}") # type: ignore
                elif "Warning" in note or "Issue" in note: st.warning(f"🔍 {note}") # type: ignore
                else: st.info(f"ℹ️ {note}") # type: ignore

        if msg.get("role") == "assistant" and (code_from_msg := msg.get("code")): # Use a distinct variable name
            is_critical_error_present = any("Critical" in n for n in msg.get("verification_notes", []))
            
            # 'executed' implies a dataframe or figure was successfully generated by the workflow's execution of the code.
            should_show_df_or_figure = msg.get("executed", False) and not is_critical_error_present

            if should_show_df_or_figure:
                if (dataframe := msg.get("dataframe")) is not None:
                    st.dataframe(dataframe.head(MAX_RETURN_ROWS_CHAT), use_container_width=True) # type: ignore
                if (figure := msg.get("figure")) and isinstance(figure, Figure):
                    st.pyplot(figure, use_container_width=True) # type: ignore
                
                # If a df/figure is shown AND it was a pandas pure code scenario (initial markdown skipped),
                # we must explicitly show the code here as well.
                if is_pandas_pure_code_content:
                    st.code(code_from_msg, language="python")
            
            elif code_from_msg: # No df/figure to show, but there is code in code_from_msg
                                # This path will be taken by pandas pure code if no df/figure was generated.
                st.code(code_from_msg, language="sql" if msg.get("agent") == "sql" else "python") # type: ignore
            
            # "Run again" button - applies if code was shown by either of the st.code() calls above.
            # This means it should be outside the direct elif code_from_msg.
            # It should appear if code_from_msg is present and no critical error.
            if code_from_msg and not is_critical_error_present:
                if st.button("▶ Run again", key=f"run_{idx}_{msg.get('session_id')}_{msg.get('code_hash', hash(code_from_msg))}"): # type: ignore
                    try: 
                        agent_type = msg.get("agent")
                        current_code = msg.get("code", "")
                        # Ensure global 'ss' is accessible for db_paths and tables
                        global_ss = st.session_state if "streamlit" in sys.modules else ss

                        if agent_type == "sql" and global_ss.get("db_paths"): # type: ignore
                            db_path_rerun = next(iter(global_ss["db_paths"].values())) # type: ignore
                            new_df = SQLAgent()._execute_sql(db_path_rerun, current_code)
                            msg["dataframe"] = new_df; msg["figure"] = None
                        elif agent_type == "pandas" and global_ss.get("tables"): # type: ignore
                            df_initial_rerun = next(iter(global_ss["tables"].values())) # type: ignore
                            new_df = PandasAgent()._execute_pandas(df_initial_rerun, current_code)
                            msg["dataframe"] = new_df; msg["figure"] = None
                        elif agent_type == "plot" and global_ss.get("tables"): # type: ignore
                            df_initial_rerun = next(iter(global_ss["tables"].values())) # type: ignore
                            new_fig = PlotAgent()._execute_plot(df_initial_rerun, current_code)
                            msg["figure"] = new_fig; msg["dataframe"] = None
                        else:
                            st.error(f"Cannot re-run {agent_type}: Missing required context (data or schema).") # type: ignore
                            msg["executed"] = False; st.rerun(); return # type: ignore
                        
                        msg["executed"] = True
                        msg["content"] = f"Re-executed {agent_type} code." # Update content to reflect re-run
                        msg["verification_notes"] = [] # Clear old notes on re-run
                        st.rerun() # type: ignore
                    except Exception as e: 
                        st.error(f"Re-run failed: {e}") # type: ignore


def run_adk_workflow(user_query: str, schemas: dict, tables: dict, db_paths: dict, original_user_query: str, ui_feedback_placeholder: Any) -> dict[str, Any]:
    # ... (run_adk_workflow implementation remains unchanged from previous correct version with verbose logging) ...
    print(f"\n[ADK Workflow START] Original Query: '{original_user_query}'", flush=True)
    response_payload: dict[str, Any] = {} 
    current_query = user_query           
    accumulated_status_messages = ""    

    def update_ui_status(message: str, is_major_step: bool = True) -> None:
        nonlocal accumulated_status_messages
        if is_major_step: accumulated_status_messages = message 
        else: accumulated_status_messages += f"\n{message}"   
        
        print(f"[UI STATUS MOCK IN WORKFLOW] {message}", flush=True) 
        if ui_feedback_placeholder and hasattr(ui_feedback_placeholder, 'chat_message') and callable(getattr(ui_feedback_placeholder, 'chat_message')):
            try: 
                with ui_feedback_placeholder.chat_message("assistant"): 
                    st.markdown(accumulated_status_messages) # type: ignore
            except Exception as e_ui:
                print(f"[UI MOCK ERROR] Failed to update mock UI: {e_ui}", flush=True)


    for attempt in range(MAX_ADK_RETRIES + 1):
        print(f"[ADK Workflow Attempt {attempt + 1}] Current Query Snippet: '{current_query[:100]}...'", flush=True)
        update_ui_status(f"⚙️ Processing request (Attempt {attempt + 1} of {MAX_ADK_RETRIES + 1})...")

        wants_plot = any(k in current_query.lower() for k in ["plot","graph","chart","histogram","scatter"])
        selected_agent_type: str|None = None; agent_instance: ADKAgent|None = None; run_kwargs: dict[str,Any] = {}

        if wants_plot: 
            selected_agent_type="plot"; agent_instance=PlotAgent()
            if not tables: print("[ADK Workflow] No tables for plot.", flush=True); return {"agent":"plot","content":"Plotting Error: No data tables found.","cost":0.0,"code":None,"verification_notes":["Plotting requires data."]}
            ref,df_initial=next(iter(tables.items())); run_kwargs={"user_query":current_query,"df_ref_name":ref,"df_initial":df_initial}
        elif db_paths:
            selected_agent_type="sql"; agent_instance=SQLAgent()
            if not schemas: print("[ADK Workflow] No schema for SQL.", flush=True); return {"agent":"sql","content":"Database Error: No schema found.","cost":0.0,"code":None,"verification_notes":["SQL requires schema."]}
            db_n,db_p=next(iter(db_paths.items())); schema_info=schemas.get(db_n,{}); run_kwargs={"user_query":current_query,"schema_info":schema_info,"db_path":db_p,"db_name":db_n}
        elif tables:
            selected_agent_type="pandas"; agent_instance=PandasAgent()
            ref,df_initial=next(iter(tables.items())); run_kwargs={"u":current_query,"df_n":ref,"df_i":df_initial}
        else: print("[ADK Workflow] No data source.", flush=True); return {"agent":"none","content":"Data Error: No data source found.","cost":0.0,"code":None,"verification_notes":["No data source."]}

        if not agent_instance: print("[ADK Workflow] Agent init failed.", flush=True); return {"agent":"none","content":"System Error: Agent init failed.","cost":0.0,"code":None,"verification_notes":["Agent init failed."]}
        
        update_ui_status(f"⏳ Generating code with {selected_agent_type.capitalize()} agent...", is_major_step=False)
        print(f"[ADK Workflow] Calling agent.run() for {selected_agent_type}", flush=True)
        response_payload = agent_instance.run(**run_kwargs)
        print(f"[ADK Workflow] Agent {selected_agent_type} returned. Content snippet: {str(response_payload.get('content'))[:100]}", flush=True)
        response_payload.setdefault("verification_notes", [])

        content = response_payload.get("content", ""); code = response_payload.get("code")
        is_validation_error = any(p in content for p in ["Validation Error:","Syntax Error:","Harmful Query:","Schema Error:","Column Error:"]) and not any(e in content for e in ["OpenAI API Error:","Runtime Error"])
        
        if is_validation_error:
            update_ui_status(f"⚠️ Validation failed: {content.split(': ',1)[-1]}", is_major_step=False)
            if attempt < MAX_ADK_RETRIES:
                corrective_prompt = f"Original request: '{original_user_query}'. Previous code:\n```\n{code or 'N/A'}\n```\nError: '{content}'. Fix the code."
                current_query = corrective_prompt 
                update_ui_status(f"🔄 Retrying (Attempt {attempt + 2}/{MAX_ADK_RETRIES + 1})...", is_major_step=True)
                response_payload["verification_notes"].append(f"Self-Correction (Attempt {attempt+1}): Pre-execution validation failed. Error: {content}")
                print(f"[ADK Workflow] Pre-execution validation failed. Retrying. New query: {current_query[:100]}...", flush=True)
                continue
            else:
                # MODIFIED: Handle Pandas agent error content
                error_message_detail = content.split(': ',1)[-1]
                if selected_agent_type == "pandas":
                    response_payload["content"] = code or error_message_detail # Prefer code, fallback to error detail
                else:
                    response_payload["content"] = f"🚫 Correction Failed (Validation): {error_message_detail}"
                response_payload["verification_notes"].append("Self-correction failed: Persistent pre-execution validation errors.")
                print("[ADK Workflow] Max retries reached for pre-execution validation.", flush=True)
                return response_payload

        actionable_notes = [n for n in response_payload.get("verification_notes", []) if "Critical" in n or (selected_agent_type == "plot" and "Plot empty" in n)]
        if actionable_notes:
            notes_summary = "; ".join(actionable_notes)
            update_ui_status(f"🔍 Verification issues: {notes_summary}", is_major_step=False)
            if attempt < MAX_ADK_RETRIES:
                corrective_prompt = f"Original request: '{original_user_query}'. Previous code:\n```\n{code or 'N/A'}\n```\nVerification issues: '{notes_summary}'. Generate improved code."
                current_query = corrective_prompt
                update_ui_status(f"🔄 Retrying (Attempt {attempt + 2}/{MAX_ADK_RETRIES + 1})...", is_major_step=True)
                response_payload["verification_notes"].append(f"Self-Correction (Attempt {attempt+1}): Post-execution verification issues. Details: {notes_summary}")
                print(f"[ADK Workflow] Post-execution verification failed. Retrying. New query: {current_query[:100]}...", flush=True)
                continue
            else:
                # MODIFIED: Handle Pandas agent error content
                if selected_agent_type == "pandas":
                    response_payload["content"] = code or notes_summary # Prefer code, fallback to notes_summary
                else:
                    response_payload["content"] = f"🚫 Correction Failed (Verification): {notes_summary}"
                response_payload["verification_notes"].append("Self-correction failed: Persistent post-execution verification issues.")
                print("[ADK Workflow] Max retries reached for post-execution verification.", flush=True)
                return response_payload
        
        # MODIFIED: Handle Pandas agent success content
        # Check if the agent's run was successful (no critical errors, content from agent isn't an error message itself)
        agent_content = response_payload.get("content", "")
        agent_code = response_payload.get("code")
        is_agent_run_successful = not any(err_msg in agent_content for err_msg in ["Error:", "Validation Error:", "Syntax Error:", "Harmful Query:", "Schema Error:", "Column Error:", "❗"])

        if selected_agent_type == "pandas" and agent_code and is_agent_run_successful:
            response_payload["content"] = agent_code # Set content to only the code
            if attempt > 0:
                # Add correction note to verification_notes, not to content
                response_payload["verification_notes"].append(f"Self-Correction: Successful on attempt {attempt + 1}.")
        else:
            # Original logic for non-pandas agents or pandas agents with errors from their run method
            final_message = agent_content if agent_content else "Processing complete."
            if attempt > 0:
                # Prepend success message for non-pandas or pandas initial error cases
                # For pandas, if it was an error from agent.run, it's already in agent_content
                if selected_agent_type != "pandas" or not is_agent_run_successful :
                     final_message = f"✅ Correction successful after {attempt} attempt(s)!\n\n{final_message}"
                response_payload["verification_notes"].append(f"Self-Correction: Successful on attempt {attempt + 1}.")
            response_payload["content"] = final_message
        
        update_ui_status("✨ Processing complete!", is_major_step=True)
        print("[ADK Workflow] Success.", flush=True)
        return response_payload

    # MODIFIED: Handle Pandas agent fallback content
    if selected_agent_type == "pandas" and response_payload.get("code"):
        response_payload.setdefault("content", response_payload["code"]) # Fallback to last code for pandas
    else:
        response_payload.setdefault("content", "❗ Self-correction attempts exhausted.")
        
    if not any("Correction Failed" in (response_payload.get("content") or "") for _ in range(1)) and selected_agent_type != "pandas": # Avoid double notes for pandas if content is code
         response_payload.setdefault("verification_notes", []).append("Self-correction attempts exhausted without clear success.")
    elif not any("Correction Failed" in (note for note in response_payload.get("verification_notes", []))): # Check notes for pandas
         response_payload.setdefault("verification_notes", []).append("Self-correction attempts exhausted without clear success.")

    print("[ADK Workflow] Exhausted retries without explicit success.", flush=True)
    return response_payload


# This block should be the main Streamlit app execution
if "streamlit" in sys.modules and hasattr(st, 'sidebar'): # Check if it's the real st
    # Initialize DB if not already done (e.g. if script was imported before)
    # init_db() # init_db is now called right after ss/st setup

    # Initialize session state if it wasn't fully done (e.g. if imported then run)
    # for k,v in {"chat_history":[],"costs":[],"total_cost":0.0,"schemas":{},"tables":{},"db_paths":{},"session_id":None,"current_session_id":None}.items(): ss.setdefault(k,v)
    # if ss.get("session_id") is None: 
    #     new_db_session_id = create_session()
    #     ss["session_id"] = new_db_session_id
    #     ss["current_session_id"] = str(new_db_session_id) 
    # if not ss.get("chat_history"): ss["chat_history"] = load_messages(ss["session_id"])

    st.sidebar.image("ChatGPT Image Apr 29, 2025, 01_12_56 PM.png", use_column_width=True)
    st.sidebar.header("📂 Data Sources"); files=st.sidebar.file_uploader("Upload CSV/Excel/SQLite",type=["csv","xlsx","db","sqlite"],accept_multiple_files=True)
    if files:
        for up in files:
            if up.name in ss["schemas"]: continue # type: ignore
            ext=Path(up.name).suffix.lower()
            if ext in {".db",".sqlite"}: tmp=Path(tempfile.gettempdir())/f"{datetime.now().timestamp()}_{up.name}"; tmp.write_bytes(up.getbuffer()); ss["db_paths"][up.name]=str(tmp); ss["schemas"][up.name]=infer_db_schema(str(tmp)) # type: ignore
            elif ext==".xlsx": sch,dfs=infer_excel_schema(up.getbuffer(),up.name); ss["schemas"][up.name]=sch; ss["tables"].update(dfs) # type: ignore
            else: sch,df=infer_csv_schema(up.getbuffer(),up.name); ss["schemas"][up.name]=sch; ss["tables"][up.name]=df # type: ignore
    print("[UI Info] Note: Processing very large datasets might take longer and could lead to timeouts if complexity is high.", flush=True)
    
    # MODIFIED "New Chat" button logic
    if st.sidebar.button("➕ New Chat"): 
        new_db_session_id = create_session(t="Untitled (New)") # Create new session in DB
        ss["session_id"] = new_db_session_id                        # Update main DB session ID
        ss["current_session_id"] = str(new_db_session_id)          # Align UI session ID with new DB session
        ss["chat_history"] = []                                      # Clear UI history
        ss["costs"] = []                                             # Clear costs
        ss["total_cost"] = 0.0
        # Optionally, clear schemas, tables, db_paths if new chat should start fresh from data sources
        # ss["schemas"] = {}; ss["tables"] = {}; ss["db_paths"] = {}
        st.rerun() # type: ignore

    st.sidebar.divider(); st.sidebar.subheader("💰 Cost Meter (USD)"); [st.sidebar.write(f"• {i+1}: ${c:,.4f}") for i,c in enumerate(ss["costs"])]; st.sidebar.metric("Total",f"${ss['total_cost']:.4f}"); st.sidebar.divider(); st.sidebar.subheader("🕑 Recent Chats") # type: ignore

    def clear_other_action_states(current_action_key_prefix: str | None = None, current_sess_id: int | None = None):
        """Clears session state flags for delete confirmations or title edits, optionally preserving the current one."""
        keys_to_delete = []
        for k_ss in ss.keys():
            is_delete_confirm = k_ss.startswith("confirm_delete_")
            is_edit_title = k_ss.startswith("edit_title_")
            
            if is_delete_confirm or is_edit_title:
                if current_action_key_prefix and current_sess_id is not None:
                    # Check if the key belongs to the currently activated action for the current session
                    if k_ss.startswith(current_action_key_prefix) and k_ss.endswith(str(current_sess_id)):
                        continue # Don't delete if it's for the current action being initiated
                    if k_ss.startswith("current_edit_title_") and current_action_key_prefix == "edit_title_" and k_ss.endswith(str(current_sess_id)): # preserve current_edit_title_ for rename
                        continue
                keys_to_delete.append(k_ss)
        for k_del in keys_to_delete:
            del ss[k_del]

    # MODIFIED Recent Chats loading with Delete and Rename Functionality
    for sess_id_db, title, dt_str in recent_sessions():
        confirm_delete_key = f"confirm_delete_{sess_id_db}"
        edit_title_key = f"edit_title_{sess_id_db}"
        current_edit_title_key = f"current_edit_title_{sess_id_db}"
        title_input_key = f"title_input_{sess_id_db}"

        if ss.get(edit_title_key): # Edit mode for this session
            st.sidebar.text_input(
                "New title:", 
                value=ss.get(current_edit_title_key, title), 
                key=title_input_key,
                help="Enter new title and press Save."
            )
            col_save, col_cancel_edit = st.sidebar.columns(2)
            if col_save.button("💾 Save", key=f"save_title_{sess_id_db}"):
                new_title = ss.get(title_input_key, title).strip()
                if new_title: # Ensure title is not empty
                    update_session_title(sess_id_db, new_title)
                del ss[edit_title_key]
                if current_edit_title_key in ss: del ss[current_edit_title_key]
                st.rerun()
            if col_cancel_edit.button("❌ Cancel", key=f"cancel_rename_{sess_id_db}"):
                del ss[edit_title_key]
                if current_edit_title_key in ss: del ss[current_edit_title_key]
                st.rerun()

        elif ss.get(confirm_delete_key): # Delete confirmation mode
            st.sidebar.warning(f"Delete '{title}'?")
            col_confirm, col_cancel_delete = st.sidebar.columns(2)
            if col_confirm.button("🗑️ Yes, Delete", key=f"confirm_yes_delete_{sess_id_db}", type="primary"):
                delete_session(sess_id_db)
                del ss[confirm_delete_key]
                if str(sess_id_db) == ss.get("current_session_id") or sess_id_db == ss.get("session_id"):
                    ss["current_session_id"] = None; ss["chat_history"] = []; ss["costs"] = []; ss["total_cost"] = 0.0; ss["session_id"] = None
                st.rerun()
            if col_cancel_delete.button("❌ Cancel", key=f"confirm_cancel_delete_{sess_id_db}"):
                del ss[confirm_delete_key]
                st.rerun()
        
        else: # Normal display mode
            col_open, col_rename_btn, col_delete_btn = st.sidebar.columns([5, 1, 1]) # Adjusted ratio
            
            with col_open:
                if st.button(f"{title} — {dt_str}", key=f"open_{sess_id_db}"):
                    clear_other_action_states()
                    ss["session_id"] = sess_id_db
                    ss["current_session_id"] = str(sess_id_db)
                    ss["chat_history"] = load_messages(sess_id_db)
                    ss["costs"] = []; ss["total_cost"] = 0.0
                    st.rerun()
            
            with col_rename_btn:
                if st.button("✏️", key=f"rename_btn_{sess_id_db}", help=f"Rename chat: {title}"):
                    clear_other_action_states("edit_title_", sess_id_db)
                    ss[edit_title_key] = True
                    ss[current_edit_title_key] = title 
                    st.rerun()

            with col_delete_btn:
                if st.button("🗑️", key=f"delete_btn_{sess_id_db}", help=f"Delete chat: {title}"):
                    clear_other_action_states("confirm_delete_", sess_id_db)
                    ss[confirm_delete_key] = True
                    st.rerun()

    # st.title("Talk‑2‑Table :speech_balloon:"); st.caption(", ".join(ss["schemas"].keys()) or "Upload one or more data files to begin.") # type: ignore
    st.warning("⚠️ **Performance Mode:** Pandas operations are currently running directly for faster results. This means standard timeout/sandboxing is bypassed for these operations. Please be mindful of the queries.")
    for fname,schema_val in ss["schemas"].items(): st.expander(f"📑 Schema: {fname}").json(schema_val,expanded=False) # type: ignore
    
    # Render chat history
    current_chat_history = ss.get("chat_history", [])
    for i,m_val in enumerate(current_chat_history): chat_renderer(m_val,i) # type: ignore

    if user_msg_count(ss["session_id"]) < 1000: # Increased limit for easier testing # type: ignore
        user_input = st.chat_input("Ask about your data …")
        if user_input:
            with st.chat_message("user"): st.markdown(user_input) # type: ignore
            
            # Use current ss.session_id (which is the active DB session ID)
            active_db_session_id = ss["session_id"] # type: ignore
            current_turn_idx = next_msg_index(active_db_session_id)
            save_message(active_db_session_id, current_turn_idx, "user", user_input)
            
            # Append to in-memory chat history for the current UI session
            # Ensure 'session_id' in msg is the UI session_id for chat_renderer logic
            ss["chat_history"].append({"role":"user","content":user_input,"session_id":ss["current_session_id"]}) # type: ignore
            
            thinking_msg_placeholder = st.empty() # type: ignore
            with thinking_msg_placeholder.chat_message("assistant"): st.markdown("⚙️ Processing your request...") # type: ignore

            assistant_response = run_adk_workflow(user_input,ss["schemas"],ss["tables"],ss["db_paths"], original_user_query=user_input, ui_feedback_placeholder=thinking_msg_placeholder) # type: ignore
            
            thinking_msg_placeholder.empty() 

            cost=assistant_response.pop("cost",0.0); ss["costs"].append(cost); ss["total_cost"]+=cost # type: ignore
            
            history_payload={
                "role":"assistant",
                "session_id":ss["current_session_id"], # Tag with UI session ID
                **assistant_response 
            }
            history_payload["executed"] = history_payload.get("dataframe") is not None or history_payload.get("figure") is not None
            
            # Render final assistant response (will also be part of chat_history for next loop)
            # chat_renderer(history_payload, len(ss["chat_history"])) # Render immediately before appending
            
            ss["chat_history"].append(history_payload) # type: ignore
            # MODIFIED: Pass agent_type to save_message
            save_message(
                active_db_session_id, 
                next_msg_index(active_db_session_id), 
                "assistant", 
                history_payload["content"], 
                history_payload.get("code"),
                history_payload.get("agent") # Pass agent type
            )
            
            if current_turn_idx == 0 and user_input: # Update title for new chats
                update_session_title(active_db_session_id, user_input.split("?",1)[0][:60].strip() or "Untitled")
            
            if hasattr(st,"rerun"): st.rerun() # type: ignore
    else: 
        st.info("⚠️ This chat reached its 1000‑message limit. Click *New Chat* to start another.") # type: ignore

    st.caption("_Prototype – The model can make mistakes. The team is working to make your experience better everyday._") # type: ignore

# Ensure google_adk.init() is called, but suppress errors if it's a placeholder
# Commented out for now to ensure no side effects during testing phase
# with contextlib.suppress(Exception): 
#   import google_adk # Assuming it's a placeholder or properly installed
#   google_adk.init() # type: ignore

# [end of talk2table_ui.py]
