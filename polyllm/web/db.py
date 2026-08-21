"""Conversation storage.

A single SQLite file, written to with plain ``sqlite3`` — the schema is two tables
and the app is a single-user localhost UI, so an ORM would cost more than it saves.

Only the transcript is stored. The sidebar settings and the Raw tab's provider JSON
stay per-session, deliberately: settings are how you want to answer the *next* turn,
not a property of the chat, and the raw log is a debugging view of this process.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TITLE = "New chat"

#: Longest title derived from a first message before it gets an ellipsis.
_TITLE_CHARS = 60


def db_path() -> Path:
    """Where the conversations live. Override with ``POLYLLM_DB``."""
    override = os.getenv("POLYLLM_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".polyllm" / "conversations.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """A fresh connection for one operation.

    FastAPI runs the sync endpoints in a threadpool, and a connection can't be
    shared across threads, so each call opens its own rather than reusing one.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        # WAL keeps a read (loading a conversation) from blocking the write at the
        # end of a stream. It's a property of the file, so setting it once sticks.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              title      TEXT NOT NULL DEFAULT 'New chat',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
              id              INTEGER PRIMARY KEY AUTOINCREMENT,
              conversation_id INTEGER NOT NULL
                              REFERENCES conversations(id) ON DELETE CASCADE,
              role            TEXT NOT NULL,
              content         TEXT NOT NULL,
              created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS messages_by_conversation
              ON messages(conversation_id, id);
            """
        )


def create_conversation() -> dict:
    now = _now()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (DEFAULT_TITLE, now, now),
        )
    return {"id": cursor.lastrowid, "title": DEFAULT_TITLE, "updated_at": now}


def list_conversations() -> list[dict]:
    """Every conversation, most recently used first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations"
            " ORDER BY updated_at DESC, id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: int) -> dict | None:
    """One conversation with its messages in order, or ``None`` if it's gone."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return {**dict(row), "messages": [dict(message) for message in messages]}


def rename_conversation(conversation_id: int, title: str) -> bool:
    """Set the title by hand. False if there's no such conversation."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title.strip() or DEFAULT_TITLE, conversation_id),
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id: int) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
    return cursor.rowcount > 0


def _derive_title(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    if not first_line:
        return DEFAULT_TITLE
    if len(first_line) <= _TITLE_CHARS:
        return first_line
    return first_line[:_TITLE_CHARS].rstrip() + "…"


def add_message(conversation_id: int, role: str, content: str) -> bool:
    """Append a turn, bumping the conversation's place in the list.

    The first user turn also names the conversation, unless it's been renamed.
    Returns False if the conversation no longer exists — a stale id from a tab
    left open across a delete shouldn't be an error, just a message that isn't
    stored.
    """
    now = _now()
    with _connect() as conn:
        exists = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if exists is None:
            return False

        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at)"
            " VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        if role == "user" and exists["title"] == DEFAULT_TITLE:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (_derive_title(content), now, conversation_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
    return True
