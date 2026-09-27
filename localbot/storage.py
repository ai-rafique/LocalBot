"""Local persistence: one SQLite file for conversations, logged answers,
question sets and experiment runs. Everything stays in DATA_DIR."""
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, title TEXT, created TEXT, updated TEXT
);
-- One row per answer: the question, the exact messages the model saw,
-- every retrieved passage, settings, metrics, checks, and the user's grade.
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY, ts TEXT, origin TEXT,
    conversation_id TEXT, run_id TEXT, qid TEXT,
    question TEXT, answer TEXT,
    messages TEXT, hits TEXT, memory TEXT, settings TEXT, metrics TEXT,
    weak_tokens TEXT, checks TEXT,
    grade TEXT, tags TEXT, reason TEXT, correction TEXT, graded_at TEXT,
    -- NULL: graded by the user. "carried:<id>": copied from that identical,
    -- user-graded answer to the same question.
    grade_source TEXT
);
CREATE INDEX IF NOT EXISTS ix_interactions_conv ON interactions(conversation_id, ts);
CREATE INDEX IF NOT EXISTS ix_interactions_run ON interactions(run_id);
CREATE TABLE IF NOT EXISTS question_sets (
    id TEXT PRIMARY KEY, name TEXT, created TEXT, items TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, name TEXT, question_set_id TEXT, config TEXT,
    status TEXT, created TEXT, started TEXT, finished TEXT,
    done INTEGER DEFAULT 0, total INTEGER DEFAULT 0, error TEXT
);
CREATE TABLE IF NOT EXISTS run_results (
    run_id TEXT, qid TEXT, interaction_id TEXT, result TEXT,
    PRIMARY KEY (run_id, qid)
);
"""

JSON_COLUMNS = ("messages", "hits", "memory", "settings", "metrics", "weak_tokens", "checks", "tags",
                "items", "config", "result")
LIST_COLUMNS = ("hits", "memory", "weak_tokens", "tags", "items", "messages")


def now():
    return datetime.now().isoformat(timespec="seconds")


def new_id():
    return uuid.uuid4().hex[:12]


def query(sql, params=()):
    # A connection per call: the server handles requests on worker threads.
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn, conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def decode(row):
    """Parse the JSON columns of a row in place."""
    if row is None:
        return None
    for col in JSON_COLUMNS:
        if col in row:
            row[col] = json.loads(row[col]) if row[col] else ([] if col in LIST_COLUMNS else {})
    return row


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.executescript(SCHEMA)
        # Databases from earlier versions lack newer columns; add them in place.
        have = {r[1] for r in conn.execute("PRAGMA table_info(interactions)")}
        if "grade_source" not in have:
            conn.execute("ALTER TABLE interactions ADD COLUMN grade_source TEXT")


# --- Conversations -----------------------------------------------------------
def create_conversation(title="New chat"):
    cid = new_id()
    query("INSERT INTO conversations VALUES (?, ?, ?, ?)", (cid, title, now(), now()))
    return get_conversation(cid)


def get_conversation(cid):
    rows = query("SELECT * FROM conversations WHERE id = ?", (cid,))
    return rows[0] if rows else None


def list_conversations():
    return query(
        "SELECT c.*, COUNT(i.id) AS turns FROM conversations c "
        "LEFT JOIN interactions i ON i.conversation_id = c.id "
        "GROUP BY c.id ORDER BY c.updated DESC"
    )


def rename_conversation(cid, title):
    query("UPDATE conversations SET title = ? WHERE id = ?", (title.strip()[:120] or "Untitled", cid))


def touch_conversation(cid):
    query("UPDATE conversations SET updated = ? WHERE id = ?", (now(), cid))


def delete_conversation(cid):
    """Deletes the conversation; its logged answers stay for evaluation."""
    query("UPDATE interactions SET conversation_id = NULL WHERE conversation_id = ?", (cid,))
    query("DELETE FROM conversations WHERE id = ?", (cid,))


def conversation_turns(cid):
    return [decode(r) for r in query(
        "SELECT * FROM interactions WHERE conversation_id = ? ORDER BY ts, rowid", (cid,))]


# --- Interactions ------------------------------------------------------------
def log_interaction(**fields):
    iid = new_id()
    row = {"id": iid, "ts": now(), **fields}
    for col in JSON_COLUMNS:
        if col in row and not isinstance(row[col], str):
            row[col] = json.dumps(row[col], ensure_ascii=False)
    cols = ", ".join(row)
    query(f"INSERT INTO interactions ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
    return iid


def get_interaction(iid):
    rows = query("SELECT * FROM interactions WHERE id = ?", (iid,))
    return decode(rows[0]) if rows else None
