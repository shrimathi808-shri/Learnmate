"""
test_quiz.py - Comprehensive Offline Unit Tests for LearnMate Quiz Module
-------------------------------------------------------------------------
Author: Selvarani (CSE Student – Quiz Generation & Answer Evaluation)
Project: LearnMate – AI-Powered Personal Tutor

All tests run completely offline using mocked Gemini API responses.
Compatible with both `pytest` and `unittest`.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from google.genai.errors import APIError

from quiz import (
    DEFAULT_MCQ_MARKS,
    DEFAULT_SHORT_ANSWER_MARKS,
    QuizAPIError,
    QuizConfigurationError,
    QuizInputError,
    QuizResponseError,
    calculate_score,
    evaluate_mcq,
    evaluate_quiz,
    evaluate_short_answer,
    generate_feedback,
    generate_quiz,
    get_performance_level,
    get_sample_quiz,
)


def _mock_gemini_json_response(payload) -> MagicMock:
    """Helper to wrap a Python dict/list or string in a mock Gemini response object."""
    mock_resp = MagicMock()
    mock_resp.text = payload if isinstance(payload, str) else json.dumps(payload)
    return mock_resp


class TestQuizGeneration(unittest.TestCase):
    """Tests for AI-based quiz generation (MCQ, Short Answer, Mixed, and Difficulty)."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_generate_mcq_quiz_success(self):
        """Verify MCQ generation validates 4 options (A-D), correct answer, and explanation."""
        payload = {
            "questions": [
                {
                    "type": "mcq",
                    "question": "Which organelle performs photosynthesis in plant cells?",
                    "options": {
                        "A": "Mitochondria",
                        "B": "Chloroplast",
                        "C": "Ribosome",
                        "D": "Nucleus",
                    },
                    "correct_answer": "B",
                    "explanation": "Chloroplasts contain chlorophyll, which captures sunlight for photosynthesis.",
                },
                {
                    "type": "mcq",
                    "question": "What gas do plants primarily absorb for photosynthesis?",
                    "options": {
                        "A": "Nitrogen",
                        "B": "Oxygen",
                        "C": "Carbon dioxide",
                        "D": "Hydrogen",
                    },
                    "correct_answer": "C",
                    "explanation": "Plants take in carbon dioxide through stomata to make glucose.",
                },
            ]
        }
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(payload)

        quiz = generate_quiz(
            topic="Photosynthesis",
            difficulty="Beginner",
            num_questions=2,
            question_type="MCQ",
            client=self.mock_client,
        )

        self.assertEqual(quiz["topic"], "Photosynthesis")
        self.assertEqual(quiz["difficulty"], "Beginner")
        self.assertEqual(quiz["question_type"], "MCQ")
        self.assertEqual(quiz["num_questions"], 2)
        self.assertEqual(quiz["total_max_marks"], 2.0)

        first = quiz["questions"][0]
        self.assertEqual(first["id"], 1)
        self.assertEqual(first["type"], "mcq")
        self.assertEqual(set(first["options"].keys()), {"A", "B", "C", "D"})
        self.assertEqual(first["correct_answer"], "B")
        self.assertIn("chlorophyll", first["explanation"].lower())
        self.assertEqual(first["max_marks"], DEFAULT_MCQ_MARKS)

    def test_generate_short_answer_quiz_success(self):
        """Verify Short Answer generation includes expected_answer, keywords, and rubric."""
        payload = {
            "questions": [
                {
                    "type": "short_answer",
                    "question": "Explain why binary search requires a sorted array.",
                    "expected_answer": (
                        "Binary search compares the middle element with the target and discards half "
                        "of the array, which only works when elements are in sorted order."
                    ),
                    "keywords": ["sorted", "middle element", "discard half", "O(log n)"],
                    "rubric": "5 marks for explaining middle comparison and half elimination; partial for partial answer.",
                }
            ]
        }
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(payload)

        quiz = generate_quiz(
            topic="Binary Search",
            difficulty="Intermediate",
            num_questions=1,
            question_type="Short Answer",
            client=self.mock_client,
        )

        self.assertEqual(quiz["difficulty"], "Intermediate")
        self.assertEqual(quiz["question_type"], "Short Answer")
        self.assertEqual(quiz["total_max_marks"], DEFAULT_SHORT_ANSWER_MARKS)

        q = quiz["questions"][0]
        self.assertEqual(q["type"], "short_answer")
        self.assertTrue(q["expected_answer"])
        self.assertEqual(len(q["keywords"]), 4)
        self.assertIn("5 marks", q["rubric"])

    def test_generate_mixed_quiz_with_markdown_fenced_json(self):
        """Verify mixed quiz generation and stripping of markdown code fences around JSON."""
        fenced_json = """```json
        {
          "questions": [
            {
              "type": "mcq",
              "question": "What does LIFO stand for?",
              "options": ["A) Last In First Out", "B) Last In Final Out", "C) Linear In First Out", "D) List In File Out"],
              "correct_answer": "A",
              "explanation": "A stack removes the most recently added item first."
            },
            {
              "type": "short_answer",
              "question": "Name two real-world uses of a stack.",
              "expected_answer": "Undo/redo in text editors and function call stacks in programming.",
              "keywords": ["undo", "call stack", "browser history"],
              "rubric": "5 marks for two clear uses; 2.5 for one use."
            }
          ]
        }
        ```"""
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(fenced_json)

        quiz = generate_quiz(
            topic="Stacks",
            difficulty="advanced",
            num_questions=2,
            question_type="Mixed",
            client=self.mock_client,
        )

        self.assertEqual(quiz["difficulty"], "Advanced")
        self.assertEqual(quiz["questions"][0]["type"], "mcq")
        self.assertEqual(quiz["questions"][0]["options"]["A"], "Last In First Out")
        self.assertEqual(quiz["questions"][1]["type"], "short_answer")
        self.assertEqual(quiz["total_max_marks"], 6.0)

    def test_generate_quiz_reuses_tutor_get_gemini_client(self):
        """Verify generate_quiz calls tutor.get_gemini_client when no explicit client is provided."""
        payload = {
            "questions": [
                {
                    "type": "mcq",
                    "question": "2 + 2 = ?",
                    "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
                    "correct_answer": "B",
                    "explanation": "Two plus two equals four.",
                }
            ]
        }
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(payload)

        with patch("quiz.get_gemini_client", return_value=self.mock_client) as mock_get_client:
            quiz = generate_quiz("Arithmetic", "Beginner", 1, "MCQ")
            mock_get_client.assert_called_once()
            self.assertEqual(quiz["num_questions"], 1)


class TestMCQEvaluation(unittest.TestCase):
    """Tests for deterministic Multiple Choice Question evaluation."""

    def setUp(self):
        self.mcq = {
            "id": 1,
            "type": "mcq",
            "question": "Which keyword defines a function in Python?",
            "options": {
                "A": "func",
                "B": "def",
                "C": "function",
                "D": "define",
            },
            "correct_answer": "B",
            "explanation": "Python uses the 'def' keyword to define functions.",
            "max_marks": 1.0,
        }

    def test_correct_mcq_evaluation(self):
        """Verify exact and lowercase/formatted correct selections receive full marks."""
        for ans in ("B", "b", "B)", "def"):
            res = evaluate_mcq(self.mcq, ans)
            self.assertEqual(res["status"], "correct")
            self.assertTrue(res["is_correct"])
            self.assertEqual(res["marks_obtained"], 1.0)
            self.assertEqual(res["max_marks"], 1.0)
            self.assertEqual(res["missed"], [])
            self.assertIn("Result: Correct", res["feedback_text"])

    def test_incorrect_mcq_evaluation(self):
        """Verify wrong option receives 0 marks and explains what was missed."""
        res = evaluate_mcq(self.mcq, "A")
        self.assertEqual(res["status"], "incorrect")
        self.assertFalse(res["is_correct"])
        self.assertEqual(res["marks_obtained"], 0.0)
        self.assertIn("B) def", res["correct_answer"])
        self.assertTrue(any("A) func" in item for item in res["missed"]))
        self.assertIn("Result: Incorrect", res["feedback_text"])

    def test_empty_mcq_answer(self):
        """Verify empty/None MCQ selection is marked incorrect with helpful guidance."""
        res = evaluate_mcq(self.mcq, "")
        self.assertEqual(res["status"], "incorrect")
        self.assertEqual(res["marks_obtained"], 0.0)
        self.assertIn("did not select", res["missed"][0].lower())


class TestShortAnswerEvaluation(unittest.TestCase):
    """Tests for AI-based Short Answer conceptual evaluation and partial marking."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.short_q = {
            "id": 2,
            "type": "short_answer",
            "question": "What is the difference between a list and a tuple in Python?",
            "expected_answer": "Lists are mutable (can be modified), whereas tuples are immutable (cannot be modified).",
            "keywords": ["mutable", "immutable", "modify"],
            "rubric": "5 marks for both; 2.5-3 for one; 0 for wrong.",
            "max_marks": 5.0,
        }

    def test_short_answer_full_marks_conceptual_match(self):
        """Verify paraphrased conceptually accurate answer gets full marks."""
        ai_resp = {
            "status": "correct",
            "marks_obtained": 5,
            "explanation": "You clearly explained that lists can be changed while tuples stay fixed.",
            "missed": [],
            "incorrect_points": [],
            "suggestions": "Awesome work!",
        }
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(ai_resp)

        res = evaluate_short_answer(
            self.short_q,
            "You can edit items inside a list after making it, but a tuple stays locked once created.",
            client=self.mock_client,
        )

        self.assertEqual(res["status"], "correct")
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["marks_obtained"], 5.0)
        self.assertEqual(res["max_marks"], 5.0)
        self.assertEqual(res["missed"], [])

    def test_short_answer_partial_marking(self):
        """Verify partially correct short answer awards partial marks and lists missing ideas."""
        ai_resp = {
            "status": "partially_correct",
            "marks_obtained": 2.5,
            "explanation": "You correctly noted that lists can change, which is half of the comparison.",
            "missed": ["Did not mention that tuples are immutable (cannot be changed)."],
            "incorrect_points": [],
            "suggestions": "Remember to compare both sides—mention tuples are immutable too!",
        }
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(ai_resp)

        res = evaluate_short_answer(
            self.short_q,
            "A list is mutable so we can append items to it.",
            client=self.mock_client,
        )

        self.assertEqual(res["status"], "partially_correct")
        self.assertFalse(res["is_correct"])
        self.assertEqual(res["marks_obtained"], 2.5)
        self.assertEqual(len(res["missed"]), 1)
        self.assertIn("immutable", res["missed"][0].lower())
        self.assertIn("Result: Partially Correct", res["feedback_text"])

    def test_short_answer_incorrect_not_blindly_accepted(self):
        """Verify factually incorrect answer receives 0 marks and highlights the error."""
        ai_resp = {
            "status": "incorrect",
            "marks_obtained": 0,
            "explanation": "Actually, lists are mutable and tuples are immutable—not the other way around.",
            "missed": ["Lists are mutable", "Tuples are immutable"],
            "incorrect_points": ["Claimed lists cannot be changed."],
            "suggestions": "Try creating a small list and tuple in Python to see which one allows .append().",
        }
        self.mock_client.models.generate_content.return_value = _mock_gemini_json_response(ai_resp)

        res = evaluate_short_answer(
            self.short_q,
            "Lists cannot be modified at all, and tuples are used for changing data.",
            client=self.mock_client,
        )

        self.assertEqual(res["status"], "incorrect")
        self.assertEqual(res["marks_obtained"], 0.0)
        self.assertTrue(any("Incorrect point:" in m for m in res["missed"]))

    def test_short_answer_empty_skips_api_call(self):
        """Verify empty/whitespace short answer gets 0 marks without invoking Gemini API."""
        res = evaluate_short_answer(self.short_q, "   ", client=self.mock_client)

        self.mock_client.models.generate_content.assert_not_called()
        self.assertEqual(res["status"], "incorrect")
        self.assertEqual(res["marks_obtained"], 0.0)
        self.assertEqual(res["missed"], self.short_q["keywords"])


class TestScoringAndFeedback(unittest.TestCase):
    """Tests for score calculation, percentage accuracy, performance levels, and feedback."""

    def test_calculate_score_mixed_evaluations(self):
        """Verify counts, total marks, percentage, and performance category for mixed results."""
        evaluations = [
            {"status": "correct", "marks_obtained": 1.0, "max_marks": 1.0},
            {"status": "incorrect", "marks_obtained": 0.0, "max_marks": 1.0},
            {"status": "partially_correct", "marks_obtained": 3.0, "max_marks": 5.0},
            {"status": "correct", "marks_obtained": 5.0, "max_marks": 5.0},
        ]

        summary = calculate_score(evaluations)

        self.assertEqual(summary["total_questions"], 4)
        self.assertEqual(summary["correct_answers"], 2)
        self.assertEqual(summary["partially_correct_answers"], 1)
        self.assertEqual(summary["incorrect_answers"], 1)
        self.assertEqual(summary["total_marks_obtained"], 9.0)
        self.assertEqual(summary["max_possible_marks"], 12.0)
        self.assertEqual(summary["percentage"], 75.0)
        self.assertEqual(summary["performance_level"], "Very Good")

    def test_performance_level_thresholds(self):
        """Verify exact performance boundaries: 90+, 75-89, 50-74, <50."""
        self.assertEqual(get_performance_level(100.0), "Excellent")
        self.assertEqual(get_performance_level(90.0), "Excellent")
        self.assertEqual(get_performance_level(89.99), "Very Good")
        self.assertEqual(get_performance_level(75.0), "Very Good")
        self.assertEqual(get_performance_level(74.99), "Good")
        self.assertEqual(get_performance_level(50.0), "Good")
        self.assertEqual(get_performance_level(49.99), "Needs Improvement")
        self.assertEqual(get_performance_level(0.0), "Needs Improvement")

    def test_calculate_score_empty_list(self):
        """Verify empty evaluations list returns zeroed summary safely."""
        summary = calculate_score([])
        self.assertEqual(summary["total_questions"], 0)
        self.assertEqual(summary["percentage"], 0.0)
        self.assertEqual(summary["performance_level"], "Needs Improvement")

    def test_generate_feedback_contains_all_required_sections(self):
        """Verify generate_feedback formats all 6 student-friendly feedback items."""
        fb = generate_feedback(
            {
                "status": "partially_correct",
                "marks_obtained": 3.0,
                "max_marks": 5.0,
                "correct_answer": "Plants convert light energy into chemical energy.",
                "explanation": "Chlorophyll absorbs light to produce glucose and oxygen.",
                "missed": ["Mentioned glucose, missed oxygen release."],
                "suggestions": "Include both products of photosynthesis next time.",
            }
        )

        text = fb["feedback_text"]
        self.assertIn("Result: Partially Correct", text)
        self.assertIn("Marks Obtained: 3 / 5", text)
        self.assertIn("Correct Answer: Plants convert light energy", text)
        self.assertIn("Explanation: Chlorophyll absorbs light", text)
        self.assertIn("What You Missed: Mentioned glucose", text)
        self.assertIn("Suggestions for Improvement: Include both products", text)

    def test_evaluate_quiz_end_to_end_with_sample_quiz(self):
        """Verify full evaluate_quiz pipeline across MCQ and Short Answer questions."""
        sample_quiz = get_sample_quiz()
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_gemini_json_response(
            {
                "status": "partially_correct",
                "marks_obtained": 4.0,
                "explanation": "Good explanation of mutability.",
                "missed": ["Could briefly mention syntax difference ([] vs ())."],
                "incorrect_points": [],
                "suggestions": "Great work—adding a quick example makes it even stronger.",
            }
        )

        answers = {
            1: "C",  # Correct (1/1)
            2: "A",  # Incorrect (0/1)
            3: "Lists are mutable and can change, while tuples are immutable.",  # 4/5
        }

        result = evaluate_quiz(sample_quiz, answers, client=mock_client)
        summary = result["summary"]

        self.assertEqual(summary["total_questions"], 3)
        self.assertEqual(summary["correct_answers"], 1)
        self.assertEqual(summary["partially_correct_answers"], 1)
        self.assertEqual(summary["incorrect_answers"], 1)
        self.assertEqual(summary["total_marks_obtained"], 5.0)
        self.assertEqual(summary["max_possible_marks"], 7.0)
        self.assertEqual(summary["percentage"], 71.43)
        self.assertEqual(summary["performance_level"], "Good")


class TestErrorHandlingAndValidation(unittest.TestCase):
    """Tests for invalid inputs, missing API keys, API failures, and bad AI responses."""

    def test_invalid_inputs_generate_quiz(self):
        """Verify QuizInputError is raised for empty topic, invalid difficulty/type, or bad count."""
        with self.assertRaises(QuizInputError):
            generate_quiz("", "Beginner", 3, "MCQ")
        with self.assertRaises(QuizInputError):
            generate_quiz("Python", "Expert", 3, "MCQ")
        with self.assertRaises(QuizInputError):
            generate_quiz("Python", "Beginner", 3, "Essay")
        with self.assertRaises(QuizInputError):
            generate_quiz("Python", "Beginner", 0, "MCQ")
        with self.assertRaises(QuizInputError):
            generate_quiz("Python", "Beginner", 25, "MCQ")
        with self.assertRaises(QuizInputError):
            generate_quiz("Python", "Beginner", True, "MCQ")

    def test_invalid_question_dicts_evaluation(self):
        """Verify malformed question dicts raise QuizInputError cleanly."""
        with self.assertRaises(QuizInputError):
            evaluate_mcq({"question": "Missing options"}, "A")
        with self.assertRaises(QuizInputError):
            evaluate_short_answer({"question": "Missing expected answer"}, "some answer")

    def test_missing_api_key_handling(self):
        """Verify missing GEMINI_API_KEY raises QuizConfigurationError without crashing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with patch("quiz.load_dotenv", create=True), patch("tutor.load_dotenv"):
                with self.assertRaises(QuizConfigurationError) as ctx:
                    generate_quiz("Python", "Beginner", 2, "MCQ")
                self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_gemini_api_rate_limit_and_network_failure(self):
        """Verify Gemini APIError (429) and network exceptions raise QuizAPIError cleanly."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = APIError(
            code=429, response_json={"error": {"message": "Quota exceeded"}}
        )
        with self.assertRaises(QuizAPIError) as ctx:
            generate_quiz("Biology", "Beginner", 2, "MCQ", client=mock_client)
        self.assertIn("rate limit", str(ctx.exception).lower())

        mock_client.models.generate_content.side_effect = ConnectionError("Network unreachable")
        with self.assertRaises(QuizAPIError) as ctx2:
            evaluate_short_answer(
                {"id": 1, "question": "What is DNA?", "expected_answer": "Genetic material."},
                "DNA stores genetic information.",
                client=mock_client,
            )
        self.assertIn("network error", str(ctx2.exception).lower())

    def test_invalid_ai_json_and_empty_response_handling(self):
        """Verify unparseable JSON and empty responses raise QuizResponseError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_gemini_json_response("Not JSON at all!")
        with self.assertRaises(QuizResponseError):
            generate_quiz("Physics", "Beginner", 1, "MCQ", client=mock_client)

        mock_client.models.generate_content.return_value = _mock_gemini_json_response("   ")
        with self.assertRaises(QuizResponseError):
            generate_quiz("Physics", "Beginner", 1, "MCQ", client=mock_client)

        # Missing required options in MCQ
        bad_mcq = {"questions": [{"type": "mcq", "question": "Q1?", "options": {"A": "1"}, "correct_answer": "A"}]}
        mock_client.models.generate_content.return_value = _mock_gemini_json_response(bad_mcq)
        with self.assertRaises(QuizResponseError):
            generate_quiz("Physics", "Beginner", 1, "MCQ", client=mock_client)


if __name__ == "__main__":
    unittest.main()
