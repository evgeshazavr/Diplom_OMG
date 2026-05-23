import sqlite3
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edupath.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER NOT NULL,
                applicant_type      TEXT,
                interests           TEXT,
                work_format         TEXT,
                it_level            TEXT,
                ege_scores          TEXT,
                recommendation_text TEXT,
                top_directions      TEXT,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)


# ── Пользователи ───────────────────────────────────────────────────

def create_user(name: str, email: str, password: str):
    """Создаёт пользователя. Возвращает его id или None если email занят."""
    try:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_by_email(email: str):
    """Возвращает строку пользователя или None."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None


def verify_user(email: str, password: str):
    """Проверяет пароль. Возвращает dict пользователя или None."""
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# ── Рекомендации ───────────────────────────────────────────────────

def save_recommendation(user_id: int, applicant: dict, rec_text: str, top_dirs: list):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO recommendations
               (user_id, applicant_type, interests, work_format, it_level,
                ege_scores, recommendation_text, top_directions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                applicant.get("applicant_type", ""),
                applicant.get("interests", ""),
                applicant.get("work_format", ""),
                applicant.get("it_level", ""),
                json.dumps(applicant.get("ege_scores", {}), ensure_ascii=False),
                rec_text,
                json.dumps(top_dirs, ensure_ascii=False),
            ),
        )


def get_user_recommendations(user_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM recommendations WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        result = []
        for row in rows:
            r = dict(row)
            r["ege_scores"]     = json.loads(r["ege_scores"] or "{}")
            r["top_directions"] = json.loads(r["top_directions"] or "[]")
            result.append(r)
        return result
