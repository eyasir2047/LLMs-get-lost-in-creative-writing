"""
SQLite helper for the annotation app.
Stores all ratings with full metadata (hidden from annotators in the UI).
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "ratings.db"

_CREATE = """
CREATE TABLE IF NOT EXISTS ratings (
    username    TEXT    NOT NULL,
    story_uid   TEXT    NOT NULL,
    domain      TEXT,
    item_id     TEXT,
    condition   TEXT,
    model       TEXT,
    relevance   INTEGER,
    coherence   INTEGER,
    empathy     INTEGER,
    surprise    INTEGER,
    engagement  INTEGER,
    complexity  INTEGER,
    rated_at    TEXT,
    PRIMARY KEY (username, story_uid)
);
"""


def _conn():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def init_db():
    with _conn() as c:
        c.execute(_CREATE)
        c.commit()


def save_rating(username: str, story: dict, scores: dict):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            """
            INSERT OR REPLACE INTO ratings
              (username, story_uid, domain, item_id, condition, model,
               relevance, coherence, empathy, surprise, engagement, complexity,
               rated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                username,
                story["story_uid"],
                story["domain"],
                story["item_id"],
                story["condition"],
                story["model"],
                scores["relevance"],
                scores["coherence"],
                scores["empathy"],
                scores["surprise"],
                scores["engagement"],
                scores["complexity"],
                now,
            ),
        )
        c.commit()


def get_user_ratings(username: str) -> dict:
    """Returns {story_uid: {criterion: score}} for all rated stories by user."""
    with _conn() as c:
        cur = c.execute(
            """SELECT story_uid, relevance, coherence, empathy,
                      surprise, engagement, complexity
               FROM ratings WHERE username = ?""",
            (username,),
        )
        rows = cur.fetchall()
    result = {}
    for row in rows:
        uid, rel, coh, emp, sur, eng, com = row
        result[uid] = dict(
            relevance=rel, coherence=coh, empathy=emp,
            surprise=sur, engagement=eng, complexity=com,
        )
    return result


def get_all_ratings() -> list[dict]:
    """Returns all rows as a list of dicts (for CSV export)."""
    with _conn() as c:
        cur = c.execute("SELECT * FROM ratings ORDER BY username, rated_at")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def get_progress() -> dict:
    """Returns {username: n_rated} summary."""
    with _conn() as c:
        cur = c.execute(
            "SELECT username, COUNT(*) FROM ratings GROUP BY username"
        )
        return dict(cur.fetchall())
