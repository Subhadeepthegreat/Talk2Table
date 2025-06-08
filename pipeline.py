from __future__ import annotations

import sys
from typing import Any

from config import MAX_RETURN_ROWS_CHAT, MAX_ADK_RETRIES
from agents import SQLAgent, PandasAgent, PlotAgent, ADKAgent

# Streamlit setup or mock -----------------------------------------------------
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
        def warning(self, msg):
            self.warning_messages.append(msg)
            print(f"ST.WARNING_MOCK: {msg}", flush=True)
        def error(self, msg):
            self.error_messages.append(msg)
            print(f"ST.ERROR_MOCK: {msg}", flush=True)
        def info(self, msg):
            print(f"ST.INFO_MOCK: {msg}", flush=True)
        def success(self, msg):
            print(f"ST.SUCCESS_MOCK: {msg}", flush=True)
        def empty(self):
            class MockEmpty:
                def chat_message(self, *a, **k):
                    return self
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass
                def markdown(self, txt):
                    print(f"ST.EMPTY.MARKDOWN_MOCK: {txt}", flush=True)
                def empty(self):
                    pass
            return MockEmpty()
        def __getattr__(self, name):
            print(f"ST.{name}_MOCK called (no-op)", flush=True)
            return lambda *a, **k: None
    st = MockStreamlitModule()  # type: ignore
    ss = st.session_state  # type: ignore


def chat_renderer(msg: dict[str, Any], idx: int) -> None:
    global st
    if "streamlit" in sys.modules and st.__class__.__name__ == "MockStreamlitModule":
        import streamlit as real_st_render
        st = real_st_render

    with st.chat_message(msg["role"]):  # type: ignore
        is_pandas_pure_code_content = (
            msg.get("agent") == "pandas"
            and msg.get("code") is not None
            and msg.get("content") == msg.get("code")
            and not any(err_keyword in msg.get("content", "").lower() for err_keyword in ["error:", "failed:", "exception:"])
        )
        if not is_pandas_pure_code_content:
            st.markdown(msg["content"])

        if msg.get("role") == "assistant" and (verification_notes := msg.get("verification_notes", [])):
            for note in verification_notes:
                if "Critical" in note:
                    st.error(f"⚠️ {note}")  # type: ignore
                elif "Warning" in note or "Issue" in note:
                    st.warning(f"🔍 {note}")  # type: ignore
                else:
                    st.info(f"ℹ️ {note}")  # type: ignore

        if msg.get("role") == "assistant" and (code_from_msg := msg.get("code")):
            is_critical_error_present = any("Critical" in n for n in msg.get("verification_notes", []))
            should_show_df_or_figure = (
                (msg.get("dataframe") is not None or msg.get("figure") is not None)
                and not is_critical_error_present
            )
            if should_show_df_or_figure:
                if (dataframe := msg.get("dataframe")) is not None:
                    st.dataframe(dataframe.head(MAX_RETURN_ROWS_CHAT), use_container_width=True)
                if (figure := msg.get("figure")) and hasattr(figure, "__class__"):
                    st.pyplot(figure, use_container_width=True)
                if code_from_msg:
                    st.code(code_from_msg, language="sql" if msg.get("agent") == "sql" else "python")
            elif code_from_msg:
                st.code(code_from_msg, language="sql" if msg.get("agent") == "sql" else "python")
            if code_from_msg and not is_critical_error_present:
                if st.button("▶ Run again", key=f"run_{idx}_{msg.get('session_id')}_{msg.get('code_hash', hash(code_from_msg))}"):
                    try:
                        agent_type = msg.get("agent")
                        current_code = msg.get("code", "")
                        global_ss = st.session_state if "streamlit" in sys.modules else ss
                        if agent_type == "sql" and global_ss.get("db_paths"):
                            db_path_rerun = next(iter(global_ss["db_paths"].values()))
                            new_df = SQLAgent()._execute_sql(db_path_rerun, current_code)
                            msg["dataframe"] = new_df; msg["figure"] = None
                        elif agent_type == "pandas" and global_ss.get("tables"):
                            df_initial_rerun = next(iter(global_ss["tables"].values()))
                            new_df = PandasAgent()._execute_pandas(df_initial_rerun, current_code)
                            msg["dataframe"] = new_df; msg["figure"] = None
                        elif agent_type == "plot" and global_ss.get("tables"):
                            df_initial_rerun = next(iter(global_ss["tables"].values()))
                            new_fig = PlotAgent()._execute_plot(df_initial_rerun, current_code)
                            msg["figure"] = new_fig; msg["dataframe"] = None
                        else:
                            st.error(f"Cannot re-run {agent_type}: Missing required context (data or schema).")
                            msg["executed"] = False; st.rerun(); return
                        msg["executed"] = True
                        msg["content"] = f"Re-executed {agent_type} code."
                        msg["verification_notes"] = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Re-run failed: {e}")
        plan_txt = msg.get("plan")
        if plan_txt:
            with st.expander("🤖 Model’s thinking", expanded=False):
                st.markdown(f"```text\n{plan_txt}\n```")


def run_adk_workflow(user_query: str, schemas: dict, tables: dict, db_paths: dict, original_user_query: str, ui_feedback_placeholder: Any) -> dict[str, Any]:
    print(f"\n[ADK Workflow START] Original Query: '{original_user_query}'", flush=True)
    response_payload: dict[str, Any] = {}
    current_query = user_query
    accumulated_status_messages = ""

    def update_ui_status(message: str, is_major_step: bool = True) -> None:
        nonlocal accumulated_status_messages
        if is_major_step:
            accumulated_status_messages = message
        else:
            accumulated_status_messages += f"\n{message}"
        print(f"[UI STATUS MOCK IN WORKFLOW] {message}", flush=True)
        if ui_feedback_placeholder and hasattr(ui_feedback_placeholder, 'chat_message') and callable(getattr(ui_feedback_placeholder, 'chat_message')):
            try:
                with ui_feedback_placeholder.chat_message("assistant"):
                    st.markdown(accumulated_status_messages)  # type: ignore
            except Exception as e_ui:
                print(f"[UI MOCK ERROR] Failed to update mock UI: {e_ui}", flush=True)

    for attempt in range(MAX_ADK_RETRIES + 1):
        print(f"[ADK Workflow Attempt {attempt + 1}] Current Query Snippet: '{current_query[:100]}...'", flush=True)
        update_ui_status(f"⚙️ Processing request (Attempt {attempt + 1} of {MAX_ADK_RETRIES + 1})...")
        wants_plot = any(k in current_query.lower() for k in ["plot","graph","chart","histogram","scatter"])
        selected_agent_type: str | None = None
        agent_instance: ADKAgent | None = None
        run_kwargs: dict[str, Any] = {}
        if wants_plot:
            selected_agent_type = "plot"; agent_instance = PlotAgent()
            if not tables:
                print("[ADK Workflow] No tables for plot.", flush=True)
                return {"agent": "plot", "content": "Plotting Error: No data tables found.", "cost": 0.0, "code": None, "verification_notes": ["Plotting requires data."]}
            ref, df_initial = next(iter(tables.items()))
            run_kwargs = {"user_query": current_query, "df_ref_name": ref, "df_initial": df_initial}
        elif db_paths:
            selected_agent_type = "sql"; agent_instance = SQLAgent()
            if not schemas:
                print("[ADK Workflow] No schema for SQL.", flush=True)
                return {"agent": "sql", "content": "Database Error: No schema found.", "cost": 0.0, "code": None, "verification_notes": ["SQL requires schema."]}
            db_n, db_p = next(iter(db_paths.items()))
            schema_info = schemas.get(db_n, {})
            run_kwargs = {"user_query": current_query, "schema_info": schema_info, "db_path": db_p, "db_name": db_n}
        elif tables:
            selected_agent_type = "pandas"; agent_instance = PandasAgent()
            ref, df_initial = next(iter(tables.items()))
            run_kwargs = {"u": current_query, "df_n": ref, "df_i": df_initial}
        else:
            print("[ADK Workflow] No data source.", flush=True)
            return {"agent": "none", "content": "Data Error: No data source found.", "cost": 0.0, "code": None, "verification_notes": ["No data source."]}
        if not agent_instance:
            print("[ADK Workflow] Agent init failed.", flush=True)
            return {"agent": "none", "content": "System Error: Agent init failed.", "cost": 0.0, "code": None, "verification_notes": ["Agent init failed."]}
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
                response_payload["verification_notes"].append(f"Self-Correction(Attempt {attempt+1}): Pre-execution validation failed. Error: {content}")
                print(f"[ADK Workflow] Pre-execution validation failed. Retrying. New query: {current_query[:100]}...", flush=True)
                continue
            else:
                error_message_detail = content.split(': ',1)[-1]
                if selected_agent_type == "pandas":
                    response_payload["content"] = code or error_message_detail
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
                response_payload["verification_notes"].append(f"Self-Correction(Attempt {attempt+1}): Post-execution verification issues. Details: {notes_summary}")
                print(f"[ADK Workflow] Post-execution verification failed. Retrying. New query: {current_query[:100]}...", flush=True)
                continue
            else:
                if selected_agent_type == "pandas":
                    response_payload["content"] = code or notes_summary
                else:
                    response_payload["content"] = f"🚫 Correction Failed (Verification): {notes_summary}"
                response_payload["verification_notes"].append("Self-correction failed: Persistent post-execution verification issues.")
                print("[ADK Workflow] Max retries reached for post-execution verification.", flush=True)
                return response_payload
        agent_content = response_payload.get("content", "")
        agent_code = response_payload.get("code")
        is_agent_run_successful = not any(err_msg in agent_content for err_msg in ["Error:", "Validation Error:", "Syntax Error:", "Harmful Query:", "Schema Error:", "Column Error:", "❗"])
        if selected_agent_type == "pandas" and agent_code and is_agent_run_successful:
            response_payload["content"] = agent_code
            if attempt > 0:
                response_payload["verification_notes"].append(f"Self-Correction: Successful on attempt {attempt + 1}.")
        else:
            final_message = agent_content if agent_content else "Processing complete."
            if attempt > 0:
                if selected_agent_type != "pandas" or not is_agent_run_successful:
                    final_message = f"✅ Correction successful after {attempt} attempt(s)!\n\n{final_message}"
                response_payload["verification_notes"].append(f"Self-Correction: Successful on attempt {attempt + 1}.")
            response_payload["content"] = final_message
        update_ui_status("✨ Processing complete!", is_major_step=True)
        print("[ADK Workflow] Success.", flush=True)
        return response_payload
    if selected_agent_type == "pandas" and response_payload.get("code"):
        response_payload.setdefault("content", response_payload["code"])
    else:
        response_payload.setdefault("content", "❗ Self-correction attempts exhausted.")
    if not any("Correction Failed" in (response_payload.get("content") or "") for _ in range(1)) and selected_agent_type != "pandas":
        response_payload.setdefault("verification_notes", []).append("Self-correction attempts exhausted without clear success.")
    elif not any("Correction Failed" in (note for note in response_payload.get("verification_notes", []))):
        response_payload.setdefault("verification_notes", []).append("Self-correction attempts exhausted without clear success.")
    print("[ADK Workflow] Exhausted retries without explicit success.", flush=True)
    return response_payload
