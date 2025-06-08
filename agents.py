from __future__ import annotations

import re, ast, textwrap, traceback, multiprocessing, queue as std_queue, sys
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

try:
    import openai
except ImportError:  # pragma: no cover - openai might not be installed
    openai = None

from sherlock_protocol import (
    _canon,
    fuzzy_filter,
    best_match,
    list_distinct,
    numeric_range,
    safe_agg,
    standardise_df,
    standardise_column,
    # Sherlock row-filter helpers
    equal_filter,
    prefix_filter,
    substring_filter,
    regex_filter,
    numeric_filter,
    null_filter,
    duplicate_filter,
    row_any,
)


from config import MODEL_NAME, OPENAI_API_KEY, SQL_ROW_LIMIT, EXECUTION_TIMEOUT_S

SQL_TABLE_EXTRACTION_RE = re.compile(r"\bFROM\s+([a-zA-Z0-9_]+)\b|\bJOIN\s+([a-zA-Z0-9_]+)\b", re.IGNORECASE)
SQL_COLUMN_EXTRACTION_RE = re.compile(r"\b([a-zA-Z0-9_]+)\b(?=\s*(?:AS\s+\w+|FROM|,|\s+WHERE|\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|=|<|>|<=|>=|<>|LIKE|IN|BETWEEN|IS\s+NULL|IS\s+NOT\s+NULL|DESC|ASC|$))")
SQL_SELECT_FROM_RE = re.compile(r"\bSELECT\b.*?\bFROM\b", re.IGNORECASE | re.DOTALL)
SQL_HARMFUL_PATTERNS = {
    "DROP": re.compile(r"\bDROP\s+(TABLE|DATABASE|INDEX|VIEW)\b", re.IGNORECASE),
    "DELETE_NO_WHERE": re.compile(r"\bDELETE\s+FROM\s+\w+\s*;?\s*$", re.IGNORECASE),
    "TRUNCATE": re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    "UPDATE_NO_WHERE": re.compile(r"\bUPDATE\s+\w+\s+SET\s+.+(?!\s+WHERE\s+)", re.IGNORECASE),
}
SQL_DELETE_WITH_WHERE_RE = re.compile(r"\bDELETE\s+FROM\s+\w+\s+WHERE\s+", re.IGNORECASE)
SQL_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
SQL_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
SQL_GENERAL_WHERE_RE = re.compile(r"\bWHERE\s+1\s*=\s*1\b", re.IGNORECASE)
USER_QUERY_LIMIT_RE = re.compile(r"\b(top|limit|show\s*(?:me\s+)?only|first|last)\s+(\d+)\b", re.IGNORECASE)
PANDAS_HEAD_TAIL_RE = re.compile(r"\.(head|tail)\s*\(\s*(\d+)\s*\)")
PANDAS_FILTER_RE = re.compile(r"\[df\[.*\]\]|\.query\(|\.loc\[|\.iloc\[")


def _to_dataframe_if_needed(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name or "value")
    if isinstance(obj, (list, tuple, set)):
        return pd.DataFrame({"value": list(obj)})
    return pd.DataFrame({"value": [obj]})


def _mp_exec_target(code_to_exec_str: str, df_obj: pd.DataFrame | None, env_keys: list[str], result_q: multiprocessing.Queue) -> None:
    try:
        globals_for_exec: dict[str, Any] = {}
        if 'pd' in env_keys:
            globals_for_exec['pd'] = pd
        if 'plt' in env_keys:
            import matplotlib.pyplot as plt_module
            globals_for_exec['plt'] = plt_module
        if df_obj is not None and 'df' in env_keys:
            globals_for_exec['df'] = df_obj
        local_capture_dict: dict[str, Any] = {}
        exec(code_to_exec_str, globals_for_exec, local_capture_dict)
        final_result = local_capture_dict.get("_ret")
        if isinstance(final_result, Figure):
            try:
                final_result.canvas.draw()
            except Exception:
                pass
        result_q.put(final_result)
    except Exception as e:  # pragma: no cover - execution errors
        result_q.put(e)


def execute_plot_code(code: str, df: pd.DataFrame, plt_module: Any, st_module: Any) -> Any:
    local_vars = {"df": df, "plt": plt_module, "pd": pd}
    try:
        exec(code, {}, local_vars)
        return plt_module.gcf()
    except Exception as e:
        st_module.error(f"🚨 Plot code execution error:\n{e}")
        return None


class InvalidPandasOutputError(ValueError):
    pass


class ADKAgent:
    def __init__(self, mn: str = MODEL_NAME, ok: str = OPENAI_API_KEY):
        self.model_name = mn
        self.openai_api_key = ok

    def call_openai_llm(self, up: str, sys_prompts, mt: int = 1024):
        if isinstance(sys_prompts, str):
            sys_prompts = [sys_prompts]
        if not self.openai_api_key or not openai:
            print("Warning: OPENAI_API_KEY not set or OpenAI unavailable.", flush=True)
            return up, 0.0
        messages = ([{"role": "system", "content": p} for p in sys_prompts] + [{"role": "user", "content": up}])
        try:
            r = openai.chat.completions.create(model=self.model_name, messages=messages, max_completion_tokens=mt)
            cost = (r.usage.total_tokens if r.usage else 0) * 0.00001
            return r.choices[0].message.content.strip(), cost
        except Exception as e:
            print(f"OpenAI API Error: {e}", flush=True)
            return f"Error: OpenAI API call failed. Details: {e}", 0.0

    def _sandbox_exec(self, code_str_in: str, env: dict[str, Any]):
        lines = code_str_in.strip().split("\n")
        if len(lines) > 1:
            body, last_line = lines[:-1], lines[-1]
            code_to_exec_str = textwrap.dedent("\n".join(body) + f"\n_ret={last_line}")
        else:
            code_to_exec_str = textwrap.dedent(f"_ret={lines[0]}")
        df_obj = env.get('df')
        env_keys = list(env.keys())
        result_q: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_mp_exec_target, args=(code_to_exec_str, df_obj, env_keys, result_q))
        p.start()
        p.join(timeout=EXECUTION_TIMEOUT_S)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            if p.is_alive():
                p.kill(); p.join()
            raise TimeoutError("Code execution timed out.")
        try:
            result = result_q.get_nowait()
            if isinstance(result, Exception):
                raise result
            return result
        except std_queue.Empty:
            raise RuntimeError("Execution process ended unexpectedly without a result.")


class SQLAgent(ADKAgent):
    system_prompt_template = (
        "You are an expert SQL assistant. The database schema is:\n{schema}\n"
        "Use SQLite dialect and JOIN tables as needed to answer the user's request.\n"
        "Return a syntactically correct SQL query only – no code fences, no commentary."
    )
    _v_sql_syntax = lambda s, c: None if SQL_SELECT_FROM_RE.search(c) else "SQL Syntax Issue..."
    _v_schema_conf = lambda s, c, sc: next((f"Schema Error: Table '{t}' not in DB." for t in {tbl for grp in SQL_TABLE_EXTRACTION_RE.findall(c) for tbl in grp if tbl} if t not in sc) or next((f"Schema Error: Col '{cl}' missing." for t in {tbl for grp in SQL_TABLE_EXTRACTION_RE.findall(c) for tbl in grp if tbl} if t in sc for cl in {col[0] if isinstance(col, tuple) else col for col in SQL_COLUMN_EXTRACTION_RE.findall(c)} if not (cl.upper() in {'SELECT','FROM','WHERE'} or cl.isdigit() or cl in sc[t].get('cols',[])) and any(f"{t}.{cl}" in c or f" {cl} " in c) and not any(qt in sc and cl in sc[qt].get('cols',[]) for qt in {tbl for grp in SQL_TABLE_EXTRACTION_RE.findall(c) for tbl in grp if tbl})), None), None)
    _v_harmful_sql = lambda s, c: next((m for p, m in SQL_HARMFUL_PATTERNS.items() if re.search(p, c, re.I) and not (p == SQL_HARMFUL_PATTERNS["DELETE_NO_WHERE"] and SQL_DELETE_WITH_WHERE_RE.search(c))), None)
    _v_df_r = lambda s, d, u, c: f"Note: Expected {er} rows, got {len(d)}." if (er := (int(ls.group(1)) if (ls := SQL_LIMIT_RE.search(c)) else int(lr.group(2)) if (lr := USER_QUERY_LIMIT_RE.search(u)) else None)) and len(d) < er else None
    _v_df_e = lambda s, d, c: "Note: Query returned no data." if d.empty and (not SQL_WHERE_RE.search(c) or SQL_GENERAL_WHERE_RE.search(c)) else None

    def run(self, user_query, schema_info, db_path, db_name):
        sp = self.system_prompt_template.format(schema=textwrap.indent(str(schema_info), "    "))
        c, cost = self.call_openai_llm(user_query, sp)
        rp = {"agent": "sql", "code": c, "dataframe": None, "figure": None, "cost": cost, "verification_notes": []}
        if "Error:" in c:
            rp["content"] = f"LLM Error:{c}"; return rp
        match = re.search(r"^(?:```(?:sql)?\s*\n)?(.*?)(?:\s*\n```)?$", c, re.DOTALL | re.IGNORECASE)
        c = match.group(1).strip() if match else c.strip()
        rp["code"] = c
        for vf, va in [(self._v_sql_syntax, c), (self._v_harmful_sql, c), (self._v_schema_conf, (c, schema_info))]:
            validation_error_message = vf(*va) if isinstance(va, tuple) else vf(va)
            if validation_error_message:
                rp["content"] = f"Validation Error:{validation_error_message}"; return rp
        try:
            df = self._execute_sql(db_path, c)
            vn: list[str] = []
            rp.update({"dataframe": df, "content": f"SQL on **{db_name}**:\n```sql\n{c}\n```\nReturned {len(df)} rows."})
            for n in [self._v_df_r(df, user_query, c), self._v_df_e(df, c)]:
                if n: vn.append(n)
            rp["verification_notes"] = vn
        except Exception as e:
            rp["content"] = f"❗ SQL Error:{e}\n{traceback.format_exc()}"
        return rp

    def _execute_sql(self, p, q):
        cq = q.strip("` ")[:-1] if q.strip("` ").endswith(";") else q.strip("` ")
        conn = sqlite3.connect(p)
        return pd.read_sql(f"SELECT * FROM({cq})LIMIT {SQL_ROW_LIMIT}", conn)


class PandasAgent(ADKAgent):
    _v_py_syntax = lambda s, c: None if not (e := _exec_py_syntax_check(c)) else f"Python Syntax Error:{e}"
    _v_col_exists = lambda s, c, cs: next((f"Column Error:'{cn}' not in DataFrame." for n in ast.walk(ast.parse(c)) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'df' and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and (cn := n.slice.value) not in cs), None) if not _exec_py_syntax_check(c) else None
    _v_df_r = lambda s, d, u, c: f"Note:Code might limit to {er} rows,result has {len(d)}." if (er := (int(next(filter(None, lc.groups()), "0")[1]) if (lc := PANDAS_HEAD_TAIL_RE.search(c)) else int(lr.group(2)) if (lr := USER_QUERY_LIMIT_RE.search(u)) else None)) and len(d) < er else None
    _v_df_e = lambda s, d, c: "Note:Operation returned empty." if d.empty and not PANDAS_FILTER_RE.search(c) else None

    SYSTEM_TOOLS = """
You have the following SAFE helpers. They never leak whole tables—use them
instead of raw pandas when you feel there might be a need for approximating the answers.

Core cleaning & inspection
--------------------------
• standardise_df(
    df: pd.DataFrame,
    *,
    strings: bool = True,
    numerics: bool = False,
    datetimes: bool = False
  ) -> pd.DataFrame
    – return a cleaned copy (lower-case strings; “42%”→0.42; parse dates).

• standardise_column(
    df: pd.DataFrame,
    column: str,
    *,
    kind: str = "string"   # or "numeric" / "datetime"
  ) -> pd.Series
    – clean a single column; does NOT mutate df.

Fuzzy look-ups
--------------
• best_match(
    df: pd.DataFrame,
    column: str,
    query: str,
    *,
    start: float = 0.9,
    floor: float = 0.5,
    step: float = 0.1,
    top: int = 3
  ) -> tuple[pd.DataFrame, float]
    – progressive fuzzy matcher; returns (subset, used_cutoff).

• fuzzy_filter(
    df: pd.DataFrame,
    column: str,
    query: str,
    *,
    cutoff: float = 0.8,
    top: int = 5
  ) -> pd.DataFrame
    – one-shot fuzzy subset.

• list_distinct(
    df: pd.DataFrame,
    column: str,
    *,
    limit: int = 50
  ) -> list[str]
    – sample of distinct values (helpful for suggesting choices).

Numeric helpers
---------------
• numeric_range(
    df: pd.DataFrame,
    column: str
  ) -> tuple[min, max]
    – fast min/max after cleaning symbols and %.

• safe_agg(
    df: pd.DataFrame,
    group_by: list[str] | None,
    metrics: dict[str, list[str]]
  ) -> pd.DataFrame
    – secure wrapper around .groupby(...).agg(...).

Row-subset primitives (Sherlock filters)
---------------------------------------
All return **copies** of the matching rows:

• equal_filter(df, column, value)
    – exact match (strings canonicalised).

• prefix_filter(df, column, prefix, *, case: bool = False)
    – rows whose string starts with *prefix*.

• substring_filter(df, column, text, *, case: bool = False)
    – rows containing *text*.

• regex_filter(df, column, pattern, *, flags=re.I)
    – regex search.

• numeric_filter(df, column, op, value)
    – numeric/date compare: op ∈ {'==','!=','>','>=','<','<=','between'/'range'}.

• null_filter(df, column, *, is_null: bool = True)
    – rows where column IS (or IS NOT) null.

• duplicate_filter(df, column, *, canonise: bool = True)
    – rows that have duplicate values in *column*.

• row_any(df, *dfs)
    – union of row subsets (logical OR).

------------------------------------------------------------
"""

    SYSTEM_PLANNER = """
        As *Planner*, think step-by-step which helper(s) to call.
        Reply with two sections exactly:

        PLAN:
        # free-text, bullet reasoning

        CODE:
        ```python
        # executable Python that calls only the helpers above
        """

    def ask_llm(self, user_query: str, df_signature: str, max_tokens: int = 1024) -> dict[str, str]:
        sys_prompts = [self.SYSTEM_TOOLS, self.SYSTEM_PLANNER, f"Table schema:\n{df_signature}"]
        result = self.call_openai_llm(user_query, sys_prompts, mt=max_tokens)
        if isinstance(result, tuple):
            assistant_msg, cost_val = result
        else:
            assistant_msg, cost_val = result, 0.0
        try:
            plan_part, code_part = assistant_msg.split("CODE:", 1)
            code_clean = re.search(r"```python(.*?)```", code_part, re.S).group(1)
        except Exception:
            plan_part, code_clean = "(no PLAN section)", assistant_msg
        return {"plan": plan_part.strip(), "code": code_clean.strip(), "cost": cost_val}

    def run(self, u, df_n, df_i):
        sig = ", ".join(f"{col}:{dtype}" for col, dtype in zip(df_i.columns, df_i.dtypes))
        plan_code = self.ask_llm(u, sig)
        c, plan, cost = plan_code["code"], plan_code["plan"], plan_code["cost"]
        rp = {"agent": "pandas", "code": c, "plan": plan, "dataframe": None, "figure": None, "cost": cost, "verification_notes": []}
        if "Error:" in c:
            rp["content"] = f"LLM Error:{c}"; return rp
        match = re.search(r"^(?:```(?:python)?\s*\n)?(.*?)(?:\s*\n```)?$", c, re.DOTALL | re.IGNORECASE)
        c = match.group(1).strip() if match else c.strip()
        lines = c.split('\n')
        lines = [line for line in lines if not re.match(r"^\s*import\s+pandas(?:\s+as\s+pd)?\s*$", line)]
        c = '\n'.join(lines).strip()
        rp["code"] = c
        for vf, va in [(self._v_py_syntax, c), (self._v_col_exists, (c, list(df_i.columns)))]:
            e = vf(*va) if isinstance(va, tuple) else vf(va)
            if e:
                rp["content"] = f"Validation Error:{e}"; return rp
        try:
            rdf = self._execute_pandas(df_i, c)
            if isinstance(rdf, pd.Series):
                rdf = rdf.to_frame()
            rp.update({"dataframe": rdf, "content": f"Pandas code executed."})
            vn = []
            for n in [self._v_df_r(rdf, u, c), self._v_df_e(rdf, c)]:
                if n: vn.append(n)
            rp["verification_notes"] = vn
        except Exception as e:
            rp["content"] = f"❗ Pandas Error:{e}\n{traceback.format_exc()}"
        return rp

    def _execute_pandas(self, df_initial: pd.DataFrame, code_str: str):
        env: dict[str, Any] = {
    # core modules / data ----------------------------------------------
    "pd": pd,
    "df": df_initial.copy(),            # always pass a *copy* to protect the original

    # universal cleaning helpers ---------------------------------------
    "standardise_df": standardise_df,
    "standardise_column": standardise_column,

    # fuzzy / text helpers ---------------------------------------------
    "best_match": best_match,
    "fuzzy_filter": fuzzy_filter,
    "list_distinct": list_distinct,

    # numeric & aggregation helpers ------------------------------------
    "numeric_range": numeric_range,
    "safe_agg": safe_agg,

    # Sherlock row-subset primitives -----------------------------------
    "equal_filter": equal_filter,
    "prefix_filter": prefix_filter,
    "substring_filter": substring_filter,
    "regex_filter": regex_filter,
    "numeric_filter": numeric_filter,
    "null_filter": null_filter,
    "duplicate_filter": duplicate_filter,
    "row_any": row_any,

    # (optional) low-level canoniser – handy for ad-hoc tests ----------
    "_canon": _canon,
}

        lines = code_str.strip().split("\n")
        if len(lines) > 1:
            body, last_line = lines[:-1], lines[-1]
            code_to_exec_str = textwrap.dedent("\n".join(body + [f"_ret={last_line}"]))
        else:
            if lines and lines[0].strip():
                code_to_exec_str = textwrap.dedent(f"_ret={lines[0]}")
            else:
                raise InvalidPandasOutputError("Pandas code was empty or contained only whitespace.")
        local_vars: dict[str, Any] = {}
        exec(code_to_exec_str, env, local_vars)
        result = local_vars.get("_ret")
        result = _to_dataframe_if_needed(result)
        if isinstance(result, pd.Series):
            return result.to_frame()
        if isinstance(result, pd.DataFrame):
            return result
        raise InvalidPandasOutputError(
            f"Pandas code did not return a DataFrame or Series. Got type: {type(result)}. Executed code:\n{code_str}"
        )


def _exec_py_syntax_check(c):
    try:
        ast.parse(c)
        return None
    except SyntaxError as e:
        return e.msg


class PlotAgent(ADKAgent):
    system_prompt_template = (
        "You are an expert in converting natural language questions into Python code for creating graphs using Matplotlib and Pandas.\n"
        "The DataFrame is available as `df` with columns {columns}.\n"
        "Return ONLY the plotting Python code (no fences / comments / imports).\n"
        "IMPORTANT: Ensure the Matplotlib figure object is the VERY LAST line of your code block (e.g., assign it to a variable `fig` and make `fig` the last line, or end with `plt.gcf()`).\n"
        "Do NOT include `plt.show()` in your code."
    )
    _v_plot_sanity = lambda s, f: [n for n in ["Critical Plot Error:Not a Figure." if not isinstance(f, Figure) else "Plotting Note:Plot empty(no axes)." if not f.get_axes() else None] if n]

    def run(self, user_query, df_ref_name, df_initial):
        sp = self.system_prompt_template.format(columns=list(df_initial.columns))
        c, cost = self.call_openai_llm(user_query, sp)
        rp = {"agent": "plot", "code": c, "dataframe": None, "figure": None, "cost": cost, "verification_notes": []}
        if "Error:" in c:
            rp["content"] = f"LLM Error:{c}"; return rp
        match = re.search(r"^(?:```(?:python)?\s*\n)?(.*?)(?:\s*\n```)?$", c, re.DOTALL | re.IGNORECASE)
        c = match.group(1).strip() if match else c.strip()
        rp["code"] = c
        pa = PandasAgent()
        for vf, va in [(pa._v_py_syntax, c), (pa._v_col_exists, (c, list(df_initial.columns)))]:
            e = vf(*va) if isinstance(va, tuple) else vf(va)
            if e:
                rp["content"] = f"Validation Error (Plot Code):{e}"; return rp
        try:
            fig = self._execute_plot(df_initial, c)
            vn = self._v_plot_sanity(fig)
            rp.update({"figure": fig, "verification_notes": vn})
            rp["content"] = "\n".join(vn) if any("Critical" in n for n in vn) else f"Plot from **{df_ref_name}**."
        except Exception as e:
            rp["content"] = f"❗ Plot Error:{e}\n{traceback.format_exc()}"
            rp["verification_notes"].extend(self._v_plot_sanity(locals().get('fig')))
        return rp

    def _execute_plot(self, df: pd.DataFrame, code_str: str):
        import matplotlib.pyplot as plt_module
        import streamlit as st_module
        import pandas as pd_module
        fig = execute_plot_code(code_str, df.copy(), plt_module, st_module)
        if fig is not None and isinstance(fig, Figure):
            return fig
        if fig is None:
            raise ValueError("Plot generation failed. See error message above.")
        raise ValueError(f"Plot code did not return a Figure object. Got type: {type(fig)}")
