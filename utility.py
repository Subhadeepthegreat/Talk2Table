from __future__ import annotations

import sqlite3
import pandas as pd
import io
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Any

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
engine = create_engine(DB_URL, pool_pre_ping=True)

CREATE_SESSIONS_SQL = """
create table if not exists sessions (
    session_id  serial primary key,
    title       text    default 'Untitled',
    data_type   text    default 'Mixed',
    created_at  timestamp default current_timestamp
);
"""

CREATE_MESSAGES_SQL = """
create table if not exists messages (
    id          serial primary key,
    session_id  int     references sessions(session_id) on delete cascade,
    idx         int     not null,
    role        text    not null check (role in ('user','assistant')),
    content     text,
    code        text,
    agent_type  text,
    ts          timestamp default current_timestamp
);
"""

def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_SESSIONS_SQL))
        conn.execute(text(CREATE_MESSAGES_SQL))

def create_session(t: str = "Untitled", dt: str = "Mixed") -> int:
    stmt = text("insert into sessions(title,data_type) values (:t,:dt) returning session_id")
    with engine.begin() as conn:
        return conn.execute(stmt, {"t": t, "dt": dt}).scalar_one()

def update_session_title(s: int, t: str) -> None:
    with engine.begin() as conn:
        conn.execute(text("update sessions set title = :t where session_id = :s"), {"t": t, "s": s})

def save_message(session_id: int, idx: int, role: str, content: str, code: str | None = None, agent_type: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            """
            insert into messages
            (session_id, idx, role, content, code, agent_type)
            values (:sid, :idx, :role, :content, :code, :agent)
            """),
            {"sid": session_id, "idx": idx, "role": role, "content": content, "code": code, "agent": agent_type},
        )

def delete_session(session_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(text("delete from sessions where session_id = :sid"), {"sid": session_id})

def load_messages(session_id: int) -> list[dict]:
    q = text(
        """select role, content, code, agent_type from messages where session_id = :sid order by idx"""
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"sid": session_id}).fetchall()

    history: list[dict] = []
    for role, content, code, agent in rows:
        history.append({
            "role": role,
            "content": content,
            "code": code,
            "agent": agent,
            "dataframe": None,
            "figure": None,
            "verification_notes": [],
            "executed": False,
            "session_id": str(session_id),
        })
    return history

def next_msg_index(s: int) -> int:
    q = text("select coalesce(max(idx), -1) + 1 from messages where session_id = :sid")
    with engine.connect() as conn:
        return conn.execute(q, {"sid": s}).scalar_one()

def user_msg_count(s: int) -> int:
    q = text("select count(*) from messages where session_id = :sid and role = 'user'")
    with engine.connect() as conn:
        return conn.execute(q, {"sid": s}).scalar_one()

def recent_sessions(l: int = 20):
    q = text(
        """
        select s.session_id,
               s.title,
               to_char(coalesce((select max(ts) from messages m where m.session_id = s.session_id), s.created_at), 'DD Mon HH24:MI') as last_activity_ts
        from sessions s
        order by last_activity_ts desc
        limit :lim
    """
    )
    with engine.connect() as conn:
        return conn.execute(q, {"lim": l}).fetchall()

def infer_db_schema(p: str) -> dict[str, Any]:
    s = {}
    conn = sqlite3.connect(p)
    cur = conn.cursor()
    ts = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in ts:
        i = cur.execute(f"PRAGMA table_info({t})").fetchall()
        cs = [r[1] for r in i]
        pk = [r[1] for r in i if r[5]]
        uq = []
        for n, _, iu, *_ in cur.execute(f"PRAGMA index_list({t})"):
            if iu:
                uq += [r[2] for r in cur.execute(f"PRAGMA index_info({n})").fetchall()]
        rc = cur.execute(f"SELECT COUNT(*)FROM {t}").fetchone()[0]
        ca = [c for c in cs if cur.execute(f"SELECT COUNT(DISTINCT {c})FROM {t}").fetchone()[0] == rc]
        s[t] = {"cols": cs, "pk": pk, "uniques": list(set(uq)), "candidates": ca}
    return s

def infer_excel_schema(b: bytes, f: str):
    dfs, s = {}, {}
    xls = pd.ExcelFile(io.BytesIO(b))
    for sh in xls.sheet_names:
        df = xls.parse(sh)
        dfs[f"{f}:{sh}"] = df
        s[sh] = {
            "cols": df.columns.tolist(),
            "pk": [],
            "uniques": [],
            "candidates": [c for c in df.columns if df[c].is_unique],
        }
    return s, dfs

def infer_csv_schema(b: bytes, f: str):
    df = pd.read_csv(io.BytesIO(b))
    return {"File": {"cols": df.columns.tolist(), "pk": [], "uniques": [], "candidates": [c for c in df.columns if df[c].is_unique]}}, df
