import sys
from pathlib import Path
from datetime import datetime
import tempfile

from pipeline import st, ss, chat_renderer, run_adk_workflow
from utility import (
    init_db,
    create_session,
    update_session_title,
    save_message,
    delete_session,
    load_messages,
    next_msg_index,
    user_msg_count,
    recent_sessions,
    infer_db_schema,
    infer_excel_schema,
    infer_csv_schema,
)

# initialise database and session state --------------------------------------
init_db()
for k, v in {
    "chat_history": [],
    "costs": [],
    "total_cost": 0.0,
    "schemas": {},
    "tables": {},
    "db_paths": {},
    "session_id": None,
    "current_session_id": None,
}.items():
    ss.setdefault(k, v)

if ss.get("session_id") is None:
    new_db_session_id = create_session()
    ss["session_id"] = new_db_session_id
    ss["current_session_id"] = str(new_db_session_id)
    print(f"Initial DB session created: ID {new_db_session_id}", flush=True)

if not ss.get("chat_history") and ss.get("session_id") is not None:
    ss["chat_history"] = load_messages(ss["session_id"])


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
if "streamlit" in sys.modules and hasattr(st, "sidebar"):
    st.sidebar.image("ChatGPT Image Apr 29, 2025, 01_12_56 PM.png", use_column_width=True)
    st.sidebar.header("📂 Data Sources")
    files = st.sidebar.file_uploader(
        "Upload CSV/Excel/SQLite", type=["csv", "xlsx", "db", "sqlite"], accept_multiple_files=True
    )
    if files:
        for up in files:
            if up.name in ss["schemas"]:
                continue
            ext = Path(up.name).suffix.lower()
            if ext in {".db", ".sqlite"}:
                tmp = Path(tempfile.gettempdir()) / f"{datetime.now().timestamp()}_{up.name}"
                tmp.write_bytes(up.getbuffer())
                ss["db_paths"][up.name] = str(tmp)
                ss["schemas"][up.name] = infer_db_schema(str(tmp))
            elif ext == ".xlsx":
                sch, dfs = infer_excel_schema(up.getbuffer(), up.name)
                ss["schemas"][up.name] = sch
                ss["tables"].update(dfs)
            else:
                sch, df = infer_csv_schema(up.getbuffer(), up.name)
                ss["schemas"][up.name] = sch
                ss["tables"][up.name] = df
    print(
        "[UI Info] Note: Processing very large datasets might take longer and could lead to timeouts if complexity is high.",
        flush=True,
    )

    if st.sidebar.button("➕ New Chat"):
        new_db_session_id = create_session(t="Untitled (New)")
        ss["session_id"] = new_db_session_id
        ss["current_session_id"] = str(new_db_session_id)
        ss["chat_history"] = []
        ss["costs"] = []
        ss["total_cost"] = 0.0
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("💰 Cost Meter (USD)")
    [st.sidebar.write(f"• {i+1}: ${c:,.4f}") for i, c in enumerate(ss["costs"])]
    st.sidebar.metric("Total", f"${ss['total_cost']:.4f}")
    st.sidebar.divider()
    st.sidebar.subheader("🕑 Recent Chats")

    def clear_other_action_states(current_action_key_prefix: str | None = None, current_sess_id: int | None = None):
        keys_to_delete = []
        for k_ss in ss.keys():
            is_delete_confirm = k_ss.startswith("confirm_delete_")
            is_edit_title = k_ss.startswith("edit_title_")
            if is_delete_confirm or is_edit_title:
                if current_action_key_prefix and current_sess_id is not None:
                    if k_ss.startswith(current_action_key_prefix) and k_ss.endswith(str(current_sess_id)):
                        continue
                    if k_ss.startswith("current_edit_title_") and current_action_key_prefix == "edit_title_" and k_ss.endswith(str(current_sess_id)):
                        continue
                keys_to_delete.append(k_ss)
        for k_del in keys_to_delete:
            del ss[k_del]

    for sess_id_db, title, dt_str in recent_sessions():
        confirm_delete_key = f"confirm_delete_{sess_id_db}"
        edit_title_key = f"edit_title_{sess_id_db}"
        current_edit_title_key = f"current_edit_title_{sess_id_db}"
        title_input_key = f"title_input_{sess_id_db}"
        if ss.get(edit_title_key):
            st.sidebar.text_input(
                "New title:",
                value=ss.get(current_edit_title_key, title),
                key=title_input_key,
                help="Enter new title and press Save.",
            )
            col_save, col_cancel_edit = st.sidebar.columns(2)
            if col_save.button("💾 Save", key=f"save_title_{sess_id_db}"):
                new_title = ss.get(title_input_key, title).strip()
                if new_title:
                    update_session_title(sess_id_db, new_title)
                del ss[edit_title_key]
                if current_edit_title_key in ss:
                    del ss[current_edit_title_key]
                st.rerun()
            if col_cancel_edit.button("❌ Cancel", key=f"cancel_rename_{sess_id_db}"):
                del ss[edit_title_key]
                if current_edit_title_key in ss:
                    del ss[current_edit_title_key]
                st.rerun()
        elif ss.get(confirm_delete_key):
            st.sidebar.warning(f"Delete '{title}'?")
            col_confirm, col_cancel_delete = st.sidebar.columns(2)
            if col_confirm.button("🗑️ Yes, Delete", key=f"confirm_yes_delete_{sess_id_db}", type="primary"):
                delete_session(sess_id_db)
                del ss[confirm_delete_key]
                if str(sess_id_db) == ss.get("current_session_id") or sess_id_db == ss.get("session_id"):
                    ss["current_session_id"] = None
                    ss["chat_history"] = []
                    ss["costs"] = []
                    ss["total_cost"] = 0.0
                    ss["session_id"] = None
                st.rerun()
            if col_cancel_delete.button("❌ Cancel", key=f"confirm_cancel_delete_{sess_id_db}"):
                del ss[confirm_delete_key]
                st.rerun()
        else:
            col_open, col_rename_btn, col_delete_btn = st.sidebar.columns([5, 1, 1])
            with col_open:
                if st.button(f"{title} — {dt_str}", key=f"open_{sess_id_db}"):
                    clear_other_action_states()
                    ss["session_id"] = sess_id_db
                    ss["current_session_id"] = str(sess_id_db)
                    ss["chat_history"] = load_messages(sess_id_db)
                    ss["costs"] = []
                    ss["total_cost"] = 0.0
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

    # st.warning("⚠️ **Performance Mode:** Pandas operations are currently running directly for faster results. This means standard timeout/sandboxing is bypassed for these operations. Please be mindful of the queries.")
    st.info(
    "🔍 **Powered by the Sherlock Protocol** — our privacy‑first, blind‑query "
    "engine lets the AI analyse your data without ever seeing a single row. "
    "[Learn&nbsp;more](https://github.com/Subhadeepthegreat/Talk2Table/blob/main/README.md)"
)
    for fname, schema_val in ss["schemas"].items():
        st.expander(f"📑 Schema: {fname}").json(schema_val, expanded=False)

    current_chat_history = ss.get("chat_history", [])
    for i, m_val in enumerate(current_chat_history):
        chat_renderer(m_val, i)

    if user_msg_count(ss["session_id"]) < 1000:
        user_input = st.chat_input("Ask about your data …")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            active_db_session_id = ss["session_id"]
            current_turn_idx = next_msg_index(active_db_session_id)
            save_message(active_db_session_id, current_turn_idx, "user", user_input)
            ss["chat_history"].append({"role": "user", "content": user_input, "session_id": ss["current_session_id"]})
            thinking_msg_placeholder = st.empty()
            with thinking_msg_placeholder.chat_message("assistant"):
                st.markdown("⚙️ Processing your request...")
            assistant_response = run_adk_workflow(user_input, ss["schemas"], ss["tables"], ss["db_paths"], original_user_query=user_input, ui_feedback_placeholder=thinking_msg_placeholder)
            thinking_msg_placeholder.empty()
            cost = assistant_response.pop("cost", 0.0)
            ss["costs"].append(cost)
            ss["total_cost"] += cost
            history_payload = {"role": "assistant", "session_id": ss["current_session_id"], **assistant_response}
            history_payload["executed"] = history_payload.get("dataframe") is not None or history_payload.get("figure") is not None
            ss["chat_history"].append(history_payload)
            chat_renderer(history_payload, len(ss["chat_history"]) - 1)
            save_message(active_db_session_id, next_msg_index(active_db_session_id), "assistant", history_payload["content"], history_payload.get("code"), history_payload.get("agent"))
            if current_turn_idx == 0 and user_input:
                update_session_title(active_db_session_id, user_input.split("?", 1)[0][:60].strip() or "Untitled")
            if hasattr(st, "rerun"):
                st.rerun()
    else:
        st.info("⚠️ This chat reached its 1000‑message limit. Click *New Chat* to start another.")

    st.caption("_Prototype – The model can make mistakes. The team is working to make your experience better everyday._")

