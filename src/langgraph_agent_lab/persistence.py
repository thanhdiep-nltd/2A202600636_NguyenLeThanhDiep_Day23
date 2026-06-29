"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> Any | None:  # noqa: ANN401
    """Return a LangGraph checkpointer.

    TODO(student): implement SQLite support for the persistence extension track.
    The starter provides MemorySaver only — SQLite/Postgres are extension tasks.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3
        from pathlib import Path

        from langgraph.checkpoint.sqlite import SqliteSaver
        
        db_path = database_url or "outputs/checkpoints.sqlite"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn)
    if kind == "postgres":
        # Fallback to sqlite or raise if not configured
        raise NotImplementedError(
            "Postgres checkpointer is not configured. Please use sqlite or memory checkpointer."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
