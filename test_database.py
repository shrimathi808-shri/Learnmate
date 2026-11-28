"""
test_database.py - Offline Unit Tests for LearnMate Database Module
-------------------------------------------------------------------
Author: Deepshika (Backend & Database Management)
Project: LearnMate - AI-Powered Personal Tutor

Tests SQLite connectivity, foreign-key enforcement, student registration and login,
salted PBKDF2 password hashing, learning history, quiz scores, completed topics,
progress calculation, and non-destructive schema preservation.
"""

from contextlib import closing
import os
import sqlite3
import tempfile
import unittest

import database


class TestDatabaseModule(unittest.TestCase):
    """Comprehensive unit tests for database.py using isolated temp SQLite files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_learnmate.db")
        self.assertTrue(database.init_db(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_creates_tables_and_preserves_existing_data(self):
        """Verify tables exist, foreign keys are active, and re-running init_db keeps data."""
        reg = database.register_student(
            "Deepshika", "deepshika@example.com", "secret123", "secret123", db_path=self.db_path
        )
        self.assertTrue(reg["success"])

        # Re-running init_db must not wipe existing students
        self.assertTrue(database.init_db(self.db_path))
        student = database.get_student_by_id(reg["student_id"], db_path=self.db_path)
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], "Deepshika")

        with closing(database.get_connection(self.db_path)) as conn:
            fk_status = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            self.assertEqual(fk_status, 1)

            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall()
            }
            self.assertTrue(
                {"students", "learning_history", "quiz_scores", "completed_topics"}.issubset(tables)
            )

    def test_register_student_and_password_hashing(self):
        """Verify registration creates unique IDs and hashes passwords with random salts."""
        res1 = database.register_student(
            "Alice", "Alice@Example.com", "password123", "password123", db_path=self.db_path
        )
        res2 = database.register_student(
            "Bob", "bob@example.com", "password123", "password123", db_path=self.db_path
        )

        self.assertTrue(res1["success"])
        self.assertTrue(res2["success"])
        self.assertNotEqual(res1["student_id"], res2["student_id"])

        with closing(database.get_connection(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT email, password_hash FROM students ORDER BY student_id;"
            ).fetchall()
            self.assertEqual(rows[0]["email"], "alice@example.com")
            self.assertNotEqual(rows[0]["password_hash"], "password123")
            self.assertTrue(rows[0]["password_hash"].startswith("pbkdf2_sha256$"))
            # Same password must produce different hashes due to unique random salts
            self.assertNotEqual(rows[0]["password_hash"], rows[1]["password_hash"])

    def test_register_rejects_duplicate_email_and_password_mismatch(self):
        """Verify duplicate emails (case-insensitive) and mismatched passwords are rejected."""
        mismatch = database.register_student(
            "Alice", "alice@example.com", "password123", "different123", db_path=self.db_path
        )
        self.assertFalse(mismatch["success"])
        self.assertIn("do not match", mismatch["message"])

        first = database.register_student(
            "Alice", "alice@example.com", "password123", "password123", db_path=self.db_path
        )
        self.assertTrue(first["success"])

        duplicate = database.register_student(
            "Alice Clone", "ALICE@example.com", "anotherpass", "anotherpass", db_path=self.db_path
        )
        self.assertFalse(duplicate["success"])
        self.assertIsNone(duplicate["student_id"])
        self.assertIn("already exists", duplicate["message"])

    def test_login_student_success_and_rejection(self):
        """Verify login returns student_id on valid credentials and rejects wrong ones."""
        reg = database.register_student(
            "Zeba", "zeba@example.com", "learnmate99", "learnmate99", db_path=self.db_path
        )
        self.assertTrue(reg["success"])

        ok_login = database.login_student("ZEBA@example.com", "learnmate99", db_path=self.db_path)
        self.assertTrue(ok_login["success"])
        self.assertEqual(ok_login["student_id"], reg["student_id"])
        self.assertEqual(ok_login["name"], "Zeba")

        wrong_pw = database.login_student("zeba@example.com", "wrongpass", db_path=self.db_path)
        self.assertFalse(wrong_pw["success"])
        self.assertIsNone(wrong_pw["student_id"])

        unknown_user = database.login_student("nobody@example.com", "learnmate99", db_path=self.db_path)
        self.assertFalse(unknown_user["success"])
        self.assertIsNone(unknown_user["student_id"])

    def test_learning_history_save_retrieve_and_isolation(self):
        """Verify learning history saves per student and cannot leak across students."""
        s1 = database.register_student("Student One", "s1@example.com", "pass1234", db_path=self.db_path)
        s2 = database.register_student("Student Two", "s2@example.com", "pass1234", db_path=self.db_path)

        saved1 = database.save_learning_history(
            student_id=s1["student_id"],
            topic="Photosynthesis",
            learning_level="beginner",
            student_question="How do plants make food?",
            tutor_response="Plants use sunlight, water, and CO2 to make sugar.",
            db_path=self.db_path,
        )
        self.assertTrue(saved1["success"])

        saved2 = database.save_learning_history(
            student_id=s2["student_id"],
            topic="Recursion",
            learning_level="Intermediate",
            student_question="What is a base case?",
            tutor_response="The stopping condition of a recursive function.",
            db_path=self.db_path,
        )
        self.assertTrue(saved2["success"])

        s1_history = database.get_learning_history(s1["student_id"], db_path=self.db_path)
        self.assertEqual(len(s1_history), 1)
        self.assertEqual(s1_history[0]["topic"], "Photosynthesis")
        self.assertEqual(s1_history[0]["learning_level"], "Beginner")

        s2_history = database.get_learning_history(s2["student_id"], db_path=self.db_path)
        self.assertEqual(len(s2_history), 1)
        self.assertEqual(s2_history[0]["topic"], "Recursion")

        # Invalid student ID or invalid learning level must fail cleanly
        bad_student = database.save_learning_history(
            9999, "Math", "Beginner", "What is 2+2?", "4", db_path=self.db_path
        )
        self.assertFalse(bad_student["success"])

        bad_level = database.save_learning_history(
            s1["student_id"], "Math", "Expert", "What is 2+2?", "4", db_path=self.db_path
        )
        self.assertFalse(bad_level["success"])

    def test_quiz_scores_and_quiz_result_integration(self):
        """Verify direct quiz score saving and Selvarani's quiz_result adapter."""
        s1 = database.register_student("Selva Test", "selva@example.com", "pass1234", db_path=self.db_path)
        sid = s1["student_id"]

        res_direct = database.save_quiz_score(
            student_id=sid,
            quiz_topic="Binary Search",
            score=4,
            total_questions=5,
            db_path=self.db_path,
        )
        self.assertTrue(res_direct["success"])
        self.assertEqual(res_direct["percentage"], 80.0)
        self.assertEqual(res_direct["performance_level"], "Very Good")

        # Simulated output matching quiz.evaluate_quiz()
        mock_quiz_result = {
            "topic": "Photosynthesis",
            "difficulty": "Beginner",
            "question_type": "Mixed",
            "evaluations": [],
            "summary": {
                "total_questions": 4,
                "correct_answers": 3,
                "partially_correct_answers": 1,
                "incorrect_answers": 0,
                "total_marks_obtained": 3.5,
                "max_possible_marks": 4.0,
                "percentage": 87.5,
                "performance_level": "Very Good",
            },
        }
        res_adapter = database.save_quiz_result(sid, mock_quiz_result, db_path=self.db_path)
        self.assertTrue(res_adapter["success"])
        self.assertEqual(res_adapter["percentage"], 87.5)

        scores = database.get_quiz_scores(sid, db_path=self.db_path)
        self.assertEqual(len(scores), 2)
        topics = {item["quiz_topic"] for item in scores}
        self.assertEqual(topics, {"Binary Search", "Photosynthesis"})

    def test_completed_topics_and_student_progress(self):
        """Verify completed topic deduplication and student progress summary calculations."""
        s1 = database.register_student("Shrimathi", "shri@example.com", "pass1234", db_path=self.db_path)
        s2 = database.register_student("Other", "other@example.com", "pass1234", db_path=self.db_path)
        sid = s1["student_id"]

        database.save_learning_history(
            sid, "Photosynthesis", "Beginner", "What is chlorophyll?", "A green pigment.", db_path=self.db_path
        )
        database.save_learning_history(
            sid, "Python Lists", "Intermediate", "How to slice a list?", "Use list[start:end].", db_path=self.db_path
        )
        database.save_quiz_score(sid, "Photosynthesis", 5, 5, db_path=self.db_path)
        database.save_quiz_score(sid, "Python Lists", 3, 5, db_path=self.db_path)

        # Save data for another student to verify isolation
        database.save_quiz_score(s2["student_id"], "Thermodynamics", 1, 10, db_path=self.db_path)

        comp1 = database.mark_topic_completed(sid, "Photosynthesis", db_path=self.db_path)
        self.assertTrue(comp1["success"])
        self.assertFalse(comp1["already_completed"])

        # Marking same topic again (even with different casing) should not duplicate
        comp_dup = database.mark_topic_completed(sid, "photosynthesis", db_path=self.db_path)
        self.assertTrue(comp_dup["success"])
        self.assertTrue(comp_dup["already_completed"])

        completed_list = database.get_completed_topics(sid, db_path=self.db_path)
        self.assertEqual(len(completed_list), 1)

        progress = database.get_student_progress(sid, db_path=self.db_path)
        self.assertTrue(progress["success"])
        self.assertEqual(progress["student"]["name"], "Shrimathi")
        self.assertEqual(progress["total_questions_asked"], 2)
        self.assertEqual(progress["topics_studied_count"], 2)
        self.assertEqual(progress["completed_topics_count"], 1)
        self.assertEqual(progress["completion_rate"], 50.0)
        self.assertEqual(progress["total_quizzes_taken"], 2)
        self.assertEqual(progress["average_quiz_score"], 80.0)
        self.assertEqual(progress["best_quiz_score"], 100.0)
        self.assertEqual(progress["overall_performance_level"], "Very Good")

    def test_safe_migration_preserves_older_quiz_table(self):
        """Verify init_db safely upgrades an older 6-column quiz_scores table without losing rows."""
        legacy_db = os.path.join(self.temp_dir.name, "legacy.db")
        with closing(sqlite3.connect(legacy_db)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE students (
                    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE quiz_scores (
                    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    quiz_topic TEXT NOT NULL,
                    score REAL NOT NULL,
                    total_questions INTEGER NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                );
                """
            )
            conn.execute(
                "INSERT INTO students (name, email, password_hash) VALUES ('Legacy', 'legacy@example.com', 'hash');"
            )
            conn.execute(
                "INSERT INTO quiz_scores (student_id, quiz_topic, score, total_questions) VALUES (1, 'Algebra', 4, 5);"
            )
            conn.commit()

        self.assertTrue(database.init_db(legacy_db))
        scores = database.get_quiz_scores(1, db_path=legacy_db)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["quiz_topic"], "Algebra")
        self.assertEqual(scores[0]["percentage"], 80.0)
        self.assertEqual(scores[0]["performance_level"], "Very Good")


if __name__ == "__main__":
    unittest.main()
