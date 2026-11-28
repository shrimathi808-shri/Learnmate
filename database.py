"""
database.py - LearnMate SQLite Database & Progress Tracking Module
------------------------------------------------------------------
Author: Deepshika (Backend & Database Management)
Project: LearnMate - AI-Powered Personal Tutor (HOMINGO HACK DAYS)

This module manages all persistent data for LearnMate using Python's built-in
`sqlite3` library and standard-library cryptography (`hashlib`, `secrets`, `hmac`).

Tables managed:
1. students         - Registered student accounts with salted PBKDF2 password hashes
2. learning_history - Saved AI tutor questions, responses, topics, and levels
3. quiz_scores      - Saved quiz scores linked to each student
4. completed_topics - Topics marked as completed by each student
"""

from contextlib import contextmanager
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from typing import Any, Dict, Iterator, List, Optional


# Store the SQLite database file next to this module by default so it works
# reliably regardless of which folder the app is launched from.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH: str = os.path.join(BASE_DIR, "learnmate.db")

# Password hashing settings (PBKDF2-HMAC-SHA256)
HASH_ALGORITHM: str = "pbkdf2_sha256"
HASH_ITERATIONS: int = 200_000
SALT_BYTES: int = 16
MIN_PASSWORD_LENGTH: int = 6

# Allowed learning levels matching tutor.py
VALID_LEARNING_LEVELS: Dict[str, str] = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
}

# Basic email format pattern (e.g., student@example.com)
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Database Connection & Initialization Helpers
# ---------------------------------------------------------------------------

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open a SQLite connection with foreign-key enforcement enabled.

    Args:
        db_path: Path to the SQLite database file (or ':memory:').

    Returns:
        sqlite3.Connection: Configured SQLite connection returning dict-like rows.
    """
    if db_path != ":memory:":
        parent_dir = os.path.dirname(os.path.abspath(db_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # SQLite disables foreign-key checks by default, so enable them per connection.
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def _managed_connection(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection and guarantee it is closed when the block exits."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _apply_safe_migrations(conn: sqlite3.Connection) -> None:
    """
    Safely add optional metadata columns to existing tables if an older database
    file does not have them yet. Existing data is never deleted or overwritten.
    """
    existing_quiz_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(quiz_scores);").fetchall()
    }
    optional_quiz_cols = {
        "max_score": "REAL",
        "percentage": "REAL",
        "difficulty": "TEXT",
    }
    for col_name, col_type in optional_quiz_cols.items():
        if col_name not in existing_quiz_cols:
            conn.execute(f"ALTER TABLE quiz_scores ADD COLUMN {col_name} {col_type};")


def init_db(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Initialize the LearnMate SQLite database and create tables if they do not exist.
    Safe to call multiple times: existing tables and student records are preserved.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        bool: True if initialization succeeded, False on database error.
    """
    try:
        with _managed_connection(db_path) as conn:
            # 1. Students table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 2. Learning History table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    learning_level TEXT NOT NULL CHECK (
                        learning_level IN ('Beginner', 'Intermediate', 'Advanced')
                    ),
                    student_question TEXT NOT NULL,
                    tutor_response TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id)
                        REFERENCES students (student_id)
                        ON DELETE CASCADE
                );
                """
            )

            # 3. Quiz Scores table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quiz_scores (
                    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    quiz_topic TEXT NOT NULL,
                    score REAL NOT NULL,
                    total_questions INTEGER NOT NULL,
                    max_score REAL,
                    percentage REAL,
                    difficulty TEXT,
                    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id)
                        REFERENCES students (student_id)
                        ON DELETE CASCADE
                );
                """
            )

            # 4. Completed Topics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_topics (
                    completion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (student_id, topic),
                    FOREIGN KEY (student_id)
                        REFERENCES students (student_id)
                        ON DELETE CASCADE
                );
                """
            )

            # Create indexes for faster per-student lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learning_history_student "
                "ON learning_history (student_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quiz_scores_student "
                "ON quiz_scores (student_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_completed_topics_student "
                "ON completed_topics (student_id);"
            )

            _apply_safe_migrations(conn)
            conn.commit()
        return True
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Internal Validation & Security Helpers
# ---------------------------------------------------------------------------

def _normalize_email(email: Optional[str]) -> str:
    """Trim whitespace and convert email to lowercase for consistent matching."""
    return (email or "").strip().lower()


def _normalize_learning_level(level: Optional[str]) -> Optional[str]:
    """Normalize learning level to 'Beginner', 'Intermediate', or 'Advanced'."""
    cleaned = (level or "").strip().lower()
    return VALID_LEARNING_LEVELS.get(cleaned)


def _hash_password(password: str) -> str:
    """
    Hash a plain-text password using PBKDF2-HMAC-SHA256 with a random salt.

    Stored format:
        pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    salt = secrets.token_bytes(SALT_BYTES)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS,
    )
    return f"{HASH_ALGORITHM}${HASH_ITERATIONS}${salt.hex()}${derived_key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plain-text password against a stored PBKDF2-HMAC-SHA256 hash."""
    if not password or not stored_hash:
        return False

    try:
        algorithm, iterations_str, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != HASH_ALGORITHM:
            return False

        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(derived_key, expected_key)


def _student_exists(conn: sqlite3.Connection, student_id: Any) -> bool:
    """Return True if a valid student record exists for the given student_id."""
    if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
        return False
    row = conn.execute(
        "SELECT 1 FROM students WHERE student_id = ?;",
        (student_id,),
    ).fetchone()
    return row is not None


def _get_performance_level(percentage: float) -> str:
    """
    Map a percentage score to a student-friendly performance label.
    Matches the performance scale used in quiz.py.
    """
    if percentage >= 90.0:
        return "Excellent"
    if percentage >= 75.0:
        return "Very Good"
    if percentage >= 50.0:
        return "Good"
    return "Needs Improvement"


# ---------------------------------------------------------------------------
# Student Registration & Login
# ---------------------------------------------------------------------------

def register_student(
    name: str,
    email: str,
    password: str,
    confirm_password: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Register a new student account with a securely hashed password.

    Args:
        name: Student's full name.
        email: Unique email address for login.
        password: Account password (minimum 6 characters).
        confirm_password: Optional confirmation password. If provided, must match `password`.
        db_path: Path to the SQLite database file.

    Returns:
        Dict with keys: 'success' (bool), 'student_id' (int or None), 'message' (str).
    """
    clean_name = (name or "").strip()
    clean_email = _normalize_email(email)

    if not clean_name:
        return {"success": False, "student_id": None, "message": "Name is required."}

    if not clean_email or not EMAIL_REGEX.match(clean_email):
        return {
            "success": False,
            "student_id": None,
            "message": "Please enter a valid email address.",
        }

    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return {
            "success": False,
            "student_id": None,
            "message": f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.",
        }

    if confirm_password is not None and password != confirm_password:
        return {
            "success": False,
            "student_id": None,
            "message": "Passwords do not match. Please confirm your password.",
        }

    init_db(db_path)
    password_hash = _hash_password(password)

    try:
        with _managed_connection(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO students (name, email, password_hash)
                VALUES (?, ?, ?);
                """,
                (clean_name, clean_email, password_hash),
            )
            conn.commit()
            return {
                "success": True,
                "student_id": cursor.lastrowid,
                "name": clean_name,
                "email": clean_email,
                "message": "Registration successful.",
            }
    except sqlite3.IntegrityError:
        return {
            "success": False,
            "student_id": None,
            "message": "An account with this email address already exists.",
        }
    except sqlite3.Error as exc:
        return {
            "success": False,
            "student_id": None,
            "message": f"Database error during registration: {str(exc)}",
        }


def login_student(
    email: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Authenticate a student using email and password.

    Args:
        email: Registered email address.
        password: Password entered by the student.
        db_path: Path to the SQLite database file.

    Returns:
        Dict with keys: 'success' (bool), 'student_id' (int or None),
        'name' (str or None), 'email' (str or None), 'message' (str).
    """
    clean_email = _normalize_email(email)

    if not clean_email or not password:
        return {
            "success": False,
            "student_id": None,
            "name": None,
            "email": None,
            "message": "Email and password are required.",
        }

    init_db(db_path)

    try:
        with _managed_connection(db_path) as conn:
            row = conn.execute(
                """
                SELECT student_id, name, email, password_hash, created_at
                FROM students
                WHERE email = ?;
                """,
                (clean_email,),
            ).fetchone()

            if row is None or not _verify_password(password, row["password_hash"]):
                return {
                    "success": False,
                    "student_id": None,
                    "name": None,
                    "email": None,
                    "message": "Invalid email or password.",
                }

            return {
                "success": True,
                "student_id": row["student_id"],
                "name": row["name"],
                "email": row["email"],
                "created_at": row["created_at"],
                "message": "Login successful.",
            }
    except sqlite3.Error as exc:
        return {
            "success": False,
            "student_id": None,
            "name": None,
            "email": None,
            "message": f"Database error during login: {str(exc)}",
        }


def get_student_by_id(
    student_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Fetch non-sensitive profile details for an authenticated student.
    Never returns the password hash.
    """
    if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
        return None

    init_db(db_path)

    try:
        with _managed_connection(db_path) as conn:
            row = conn.execute(
                """
                SELECT student_id, name, email, created_at
                FROM students
                WHERE student_id = ?;
                """,
                (student_id,),
            ).fetchone()
            return dict(row) if row else None
    except sqlite3.Error:
        return None


# ---------------------------------------------------------------------------
# Learning History Functions (Tutor Integration)
# ---------------------------------------------------------------------------

def save_learning_history(
    student_id: int,
    topic: str,
    learning_level: str,
    student_question: str,
    tutor_response: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Save a student's tutoring interaction (topic, level, question, and AI response).

    Args:
        student_id: Authenticated student's ID.
        topic: Topic studied (e.g., 'Photosynthesis').
        learning_level: 'Beginner', 'Intermediate', or 'Advanced'.
        student_question: Question asked by the student.
        tutor_response: Response returned by tutor.ask_tutor() or explain_differently().
        db_path: Path to the SQLite database file.

    Returns:
        Dict with 'success' (bool), 'history_id' (int or None), and 'message' (str).
    """
    clean_topic = (topic or "").strip()
    clean_level = _normalize_learning_level(learning_level)
    clean_question = (student_question or "").strip()
    clean_response = (tutor_response or "").strip()

    if not clean_topic:
        return {"success": False, "history_id": None, "message": "Topic is required."}
    if not clean_level:
        return {
            "success": False,
            "history_id": None,
            "message": "Learning level must be Beginner, Intermediate, or Advanced.",
        }
    if not clean_question:
        return {
            "success": False,
            "history_id": None,
            "message": "Student question is required.",
        }
    if not clean_response:
        return {
            "success": False,
            "history_id": None,
            "message": "Tutor response is required.",
        }

    init_db(db_path)

    try:
        with _managed_connection(db_path) as conn:
            if not _student_exists(conn, student_id):
                return {
                    "success": False,
                    "history_id": None,
                    "message": "Student account not found.",
                }

            cursor = conn.execute(
                """
                INSERT INTO learning_history (
                    student_id, topic, learning_level, student_question, tutor_response
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                (student_id, clean_topic, clean_level, clean_question, clean_response),
            )
            conn.commit()
            return {
                "success": True,
                "history_id": cursor.lastrowid,
                "message": "Learning history saved.",
            }
    except sqlite3.Error as exc:
        return {
            "success": False,
            "history_id": None,
            "message": f"Database error while saving learning history: {str(exc)}",
        }


def get_learning_history(
    student_id: int,
    topic: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieve learning history records belonging strictly to the given student.

    Args:
        student_id: Authenticated student's ID.
        topic: Optional topic filter (case-insensitive).
        limit: Optional maximum number of recent records to return.
        db_path: Path to the SQLite database file.

    Returns:
        List of learning history dictionaries ordered from newest to oldest.
    """
    if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
        return []

    init_db(db_path)

    query = """
        SELECT history_id, student_id, topic, learning_level,
               student_question, tutor_response, created_at
        FROM learning_history
        WHERE student_id = ?
    """
    params: List[Any] = [student_id]

    clean_topic = (topic or "").strip()
    if clean_topic:
        query += " AND LOWER(topic) = LOWER(?)"
        params.append(clean_topic)

    query += " ORDER BY created_at DESC, history_id DESC"

    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    query += ";"

    try:
        with _managed_connection(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


# ---------------------------------------------------------------------------
# Quiz Score Functions (Quiz Integration)
# ---------------------------------------------------------------------------

def save_quiz_score(
    student_id: int,
    quiz_topic: str,
    score: float,
    total_questions: int,
    max_score: Optional[float] = None,
    difficulty: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Save a completed quiz score for a specific student.

    Args:
        student_id: Authenticated student's ID.
        quiz_topic: Topic of the quiz.
        score: Score obtained (number of correct answers or total marks obtained).
        total_questions: Total number of questions in the quiz.
        max_score: Optional maximum possible marks (defaults to total_questions).
        difficulty: Optional quiz difficulty ('Beginner', 'Intermediate', 'Advanced').
        db_path: Path to the SQLite database file.

    Returns:
        Dict with 'success', 'score_id', 'percentage', 'performance_level', and 'message'.
    """
    clean_topic = (quiz_topic or "").strip()
    if not clean_topic:
        return {"success": False, "score_id": None, "message": "Quiz topic is required."}

    if isinstance(total_questions, bool) or not isinstance(total_questions, int) or total_questions <= 0:
        return {
            "success": False,
            "score_id": None,
            "message": "Total questions must be a positive whole number.",
        }

    if isinstance(score, bool) or not isinstance(score, (int, float)) or score < 0:
        return {
            "success": False,
            "score_id": None,
            "message": "Score must be a non-negative number.",
        }

    effective_max = (
        float(max_score)
        if isinstance(max_score, (int, float)) and not isinstance(max_score, bool) and max_score > 0
        else float(total_questions)
    )
    numeric_score = float(score)
    if numeric_score > effective_max:
        return {
            "success": False,
            "score_id": None,
            "message": "Score cannot exceed the maximum possible score.",
        }

    percentage = round((numeric_score / effective_max) * 100.0, 2)
    clean_difficulty = _normalize_learning_level(difficulty) if difficulty else None

    init_db(db_path)

    try:
        with _managed_connection(db_path) as conn:
            if not _student_exists(conn, student_id):
                return {
                    "success": False,
                    "score_id": None,
                    "message": "Student account not found.",
                }

            cursor = conn.execute(
                """
                INSERT INTO quiz_scores (
                    student_id, quiz_topic, score, total_questions,
                    max_score, percentage, difficulty
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    student_id,
                    clean_topic,
                    round(numeric_score, 2),
                    total_questions,
                    round(effective_max, 2),
                    percentage,
                    clean_difficulty,
                ),
            )
            conn.commit()
            return {
                "success": True,
                "score_id": cursor.lastrowid,
                "percentage": percentage,
                "performance_level": _get_performance_level(percentage),
                "message": "Quiz score saved.",
            }
    except sqlite3.Error as exc:
        return {
            "success": False,
            "score_id": None,
            "message": f"Database error while saving quiz score: {str(exc)}",
        }


def save_quiz_result(
    student_id: int,
    quiz_result: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Save a quiz result dictionary directly from Selvarani's `quiz.evaluate_quiz()`
    or `quiz_ui.render_quiz_ui()` without duplicating evaluation logic.

    Args:
        student_id: Authenticated student's ID.
        quiz_result: Dictionary returned by `evaluate_quiz()`.
        db_path: Path to the SQLite database file.

    Returns:
        Dict returned by `save_quiz_score()`.
    """
    if not isinstance(quiz_result, dict) or not isinstance(quiz_result.get("summary"), dict):
        return {
            "success": False,
            "score_id": None,
            "message": "Invalid quiz result object.",
        }

    summary = quiz_result["summary"]
    topic = quiz_result.get("topic", "")
    difficulty = quiz_result.get("difficulty")
    total_questions = summary.get("total_questions", 0)
    score = summary.get("total_marks_obtained", summary.get("correct_answers", 0))
    max_score = summary.get("max_possible_marks", total_questions)

    return save_quiz_score(
        student_id=student_id,
        quiz_topic=topic,
        score=score,
        total_questions=total_questions,
        max_score=max_score,
        difficulty=difficulty,
        db_path=db_path,
    )


def get_quiz_scores(
    student_id: int,
    quiz_topic: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieve quiz score history belonging strictly to the given student.

    Args:
        student_id: Authenticated student's ID.
        quiz_topic: Optional topic filter (case-insensitive).
        db_path: Path to the SQLite database file.

    Returns:
        List of quiz score dicts including calculated `percentage` and `performance_level`.
    """
    if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
        return []

    init_db(db_path)

    query = """
        SELECT score_id, student_id, quiz_topic, score, total_questions,
               max_score, percentage, difficulty, completed_at
        FROM quiz_scores
        WHERE student_id = ?
    """
    params: List[Any] = [student_id]

    clean_topic = (quiz_topic or "").strip()
    if clean_topic:
        query += " AND LOWER(quiz_topic) = LOWER(?)"
        params.append(clean_topic)

    query += " ORDER BY completed_at DESC, score_id DESC;"

    try:
        with _managed_connection(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                # Fallback percentage calculation if an older DB lacks stored percentage
                if item.get("percentage") is None:
                    denom = item.get("max_score") or item["total_questions"] or 1
                    item["percentage"] = round((float(item["score"]) / float(denom)) * 100.0, 2)
                item["performance_level"] = _get_performance_level(float(item["percentage"]))
                results.append(item)
            return results
    except sqlite3.Error:
        return []


# ---------------------------------------------------------------------------
# Completed Topics & Learning Progress Functions
# ---------------------------------------------------------------------------

def mark_topic_completed(
    student_id: int,
    topic: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Mark a topic as completed for a student without creating duplicate records.

    Args:
        student_id: Authenticated student's ID.
        topic: Name of the completed topic.
        db_path: Path to the SQLite database file.

    Returns:
        Dict with 'success', 'completion_id', 'already_completed', and 'message'.
    """
    clean_topic = (topic or "").strip()
    if not clean_topic:
        return {
            "success": False,
            "completion_id": None,
            "already_completed": False,
            "message": "Topic name is required.",
        }

    init_db(db_path)

    try:
        with _managed_connection(db_path) as conn:
            if not _student_exists(conn, student_id):
                return {
                    "success": False,
                    "completion_id": None,
                    "already_completed": False,
                    "message": "Student account not found.",
                }

            # Case-insensitive check prevents duplicates like 'Python' and 'python'
            existing = conn.execute(
                """
                SELECT completion_id
                FROM completed_topics
                WHERE student_id = ? AND LOWER(topic) = LOWER(?);
                """,
                (student_id, clean_topic),
            ).fetchone()

            if existing:
                return {
                    "success": True,
                    "completion_id": existing["completion_id"],
                    "already_completed": True,
                    "message": "Topic is already marked as completed.",
                }

            cursor = conn.execute(
                """
                INSERT INTO completed_topics (student_id, topic)
                VALUES (?, ?);
                """,
                (student_id, clean_topic),
            )
            conn.commit()
            return {
                "success": True,
                "completion_id": cursor.lastrowid,
                "already_completed": False,
                "message": "Topic marked as completed.",
            }
    except sqlite3.Error as exc:
        return {
            "success": False,
            "completion_id": None,
            "already_completed": False,
            "message": f"Database error while marking topic completed: {str(exc)}",
        }


def get_completed_topics(
    student_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieve all topics completed by the specified student.

    Args:
        student_id: Authenticated student's ID.
        db_path: Path to the SQLite database file.

    Returns:
        List of completed topic dicts ordered from newest to oldest.
    """
    if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
        return []

    init_db(db_path)

    try:
        with _managed_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT completion_id, student_id, topic, completed_at
                FROM completed_topics
                WHERE student_id = ?
                ORDER BY completed_at DESC, completion_id DESC;
                """,
                (student_id,),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def get_student_progress(
    student_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """
    Build a personal learning and quiz performance summary for one student.
    Only data belonging to `student_id` is included.

    Args:
        student_id: Authenticated student's ID.
        db_path: Path to the SQLite database file.

    Returns:
        Dict summarizing learning activity, completed topics, quiz performance,
        and recent history for the student's dashboard.
    """
    student = get_student_by_id(student_id, db_path=db_path)
    if not student:
        return {
            "success": False,
            "message": "Student account not found.",
        }

    history = get_learning_history(student_id, db_path=db_path)
    quizzes = get_quiz_scores(student_id, db_path=db_path)
    completed = get_completed_topics(student_id, db_path=db_path)

    # Preserve display names while deduplicating topics case-insensitively
    studied_topics_map: Dict[str, str] = {}
    for entry in history:
        key = entry["topic"].strip().lower()
        if key not in studied_topics_map:
            studied_topics_map[key] = entry["topic"].strip()

    completed_topics_map: Dict[str, str] = {}
    for entry in completed:
        key = entry["topic"].strip().lower()
        if key not in completed_topics_map:
            completed_topics_map[key] = entry["topic"].strip()

    quiz_topics_map: Dict[str, str] = {}
    for entry in quizzes:
        key = entry["quiz_topic"].strip().lower()
        if key not in quiz_topics_map:
            quiz_topics_map[key] = entry["quiz_topic"].strip()

    all_engaged_keys = set(studied_topics_map) | set(completed_topics_map) | set(quiz_topics_map)
    completion_rate = (
        round((len(completed_topics_map) / len(all_engaged_keys)) * 100.0, 2)
        if all_engaged_keys
        else 0.0
    )

    quiz_percentages = [float(q["percentage"]) for q in quizzes]
    average_quiz_score = (
        round(sum(quiz_percentages) / len(quiz_percentages), 2)
        if quiz_percentages
        else 0.0
    )
    best_quiz_score = round(max(quiz_percentages), 2) if quiz_percentages else 0.0
    overall_performance = (
        _get_performance_level(average_quiz_score)
        if quizzes
        else "Not Assessed Yet"
    )

    return {
        "success": True,
        "student": student,
        "total_questions_asked": len(history),
        "topics_studied": list(studied_topics_map.values()),
        "topics_studied_count": len(studied_topics_map),
        "completed_topics": list(completed_topics_map.values()),
        "completed_topics_details": completed,
        "completed_topics_count": len(completed_topics_map),
        "completion_rate": completion_rate,
        "total_quizzes_taken": len(quizzes),
        "average_quiz_score": average_quiz_score,
        "best_quiz_score": best_quiz_score,
        "overall_performance_level": overall_performance,
        "latest_quiz": quizzes[0] if quizzes else None,
        "recent_learning_activity": history[:5],
        "recent_quiz_activity": quizzes[:5],
    }
