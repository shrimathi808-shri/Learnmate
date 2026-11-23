"""
test_tutor.py - Comprehensive Unit Tests for LearnMate AI Tutor Module
---------------------------------------------------------------------
Author: Shrimathi (Team Lead + LLM/API Integration)
Project: LearnMate – AI-Powered Personal Tutor

All tests run completely offline using mocked Gemini API responses.
No real API calls or active API keys are required.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from google.genai.errors import APIError

from tutor import (
    ask_tutor,
    explain_differently,
    get_gemini_client,
    SYSTEM_PROMPTS,
    EXPLAIN_DIFFERENTLY_SYSTEM_PROMPT,
)


class TestTutorLearningModes(unittest.TestCase):
    """Tests for the three personalized learning modes (Beginner, Intermediate, Advanced)."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_beginner_mode(self):
        """Verify that Beginner mode uses beginner-appropriate system instructions and returns response."""
        mock_response = MagicMock()
        mock_response.text = (
            "Think of photosynthesis like a plant's kitchen! "
            "Plants use sunlight like a stove, water from their roots, and air to cook their own food (sugar)."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        topic = "Photosynthesis"
        question = "How do plants make their food?"
        level = "Beginner"

        result = ask_tutor(topic=topic, level=level, question=question, client=self.mock_client)

        self.assertEqual(result, mock_response.text)
        self.mock_client.models.generate_content.assert_called_once()
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.system_instruction, SYSTEM_PROMPTS["beginner"])
        self.assertIn("simple, clear English", config.system_instruction)
        self.assertIn("everyday analogies", config.system_instruction)

    def test_intermediate_mode(self):
        """Verify that Intermediate mode uses structured technical explanations with practical examples."""
        mock_response = MagicMock()
        mock_response.text = (
            "Photosynthesis is a biochemical process in chloroplasts where chlorophyll captures solar energy "
            "to convert 6CO2 and 6H2O into glucose (C6H12O6) and oxygen (6O2)."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        topic = "Photosynthesis"
        question = "Explain the biochemical equation and stages."
        level = "Intermediate"

        result = ask_tutor(topic=topic, level=level, question=question, client=self.mock_client)

        self.assertEqual(result, mock_response.text)
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.system_instruction, SYSTEM_PROMPTS["intermediate"])
        self.assertIn("Technical terminology", config.system_instruction)

    def test_advanced_mode(self):
        """Verify that Advanced mode includes rigorous academic details and underlying mechanisms."""
        mock_response = MagicMock()
        mock_response.text = (
            "Photosynthesis encompasses two phases: light-dependent reactions within the thylakoid membrane "
            "involving Photosystems II and I (Z-scheme electron transport chain and ATP synthase chemiosmosis), "
            "and the Calvin-Benson cycle in the stroma mediated by RuBisCO."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        topic = "Photosynthesis"
        question = "Detail the Z-scheme electron transport chain and photophosphorylation."
        level = "Advanced"

        result = ask_tutor(topic=topic, level=level, question=question, client=self.mock_client)

        self.assertEqual(result, mock_response.text)
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.system_instruction, SYSTEM_PROMPTS["advanced"])
        self.assertIn("deep, rigorous conceptual and theoretical analysis", config.system_instruction)

    def test_case_insensitivity_and_default_level(self):
        """Verify that learning level input is case-insensitive and defaults gracefully to beginner."""
        mock_response = MagicMock()
        mock_response.text = "Newton's laws explained."
        self.mock_client.models.generate_content.return_value = mock_response

        # Test uppercase
        ask_tutor("Physics", "INTERMEDIATE", "Explain inertia", client=self.mock_client)
        config1 = self.mock_client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config1.system_instruction, SYSTEM_PROMPTS["intermediate"])

        # Test unrecognized level defaults to beginner
        ask_tutor("Physics", "unknown_level", "Explain inertia", client=self.mock_client)
        config2 = self.mock_client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config2.system_instruction, SYSTEM_PROMPTS["beginner"])


class TestConversationHistory(unittest.TestCase):
    """Tests for conversation history and follow-up questions support."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_ask_tutor_with_history_dicts(self):
        """Verify that previous turns are included in contents passed to Gemini."""
        mock_response = MagicMock()
        mock_response.text = "In Python, a list is mutable while a tuple is immutable."
        self.mock_client.models.generate_content.return_value = mock_response

        history = [
            {"role": "user", "content": "What is a data structure in Python?"},
            {"role": "assistant", "content": "A data structure is a way of organizing and storing data."},
        ]

        result = ask_tutor(
            topic="Python",
            level="Beginner",
            question="What is the difference between a list and a tuple?",
            history=history,
            client=self.mock_client,
        )

        self.assertEqual(result, mock_response.text)
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]

        # History should have 2 previous turns + 1 current turn = 3 total turns
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "What is a data structure in Python?")
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(contents[1].parts[0].text, "A data structure is a way of organizing and storing data.")
        self.assertEqual(contents[2].role, "user")
        self.assertIn("What is the difference between a list and a tuple?", contents[2].parts[0].text)

    def test_ask_tutor_with_history_tuples(self):
        """Verify that history passed as tuples is also handled properly."""
        mock_response = MagicMock()
        mock_response.text = "A stack follows LIFO order."
        self.mock_client.models.generate_content.return_value = mock_response

        history = [
            ("user", "What is a stack?"),
            ("model", "A stack is a linear data structure."),
        ]

        result = ask_tutor(
            topic="Data Structures",
            level="Intermediate",
            question="What order does it follow?",
            history=history,
            client=self.mock_client,
        )

        self.assertEqual(result, mock_response.text)
        contents = self.mock_client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[1].role, "model")


class TestExplainDifferently(unittest.TestCase):
    """Tests for the explain_differently function."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_explain_differently_success(self):
        """Verify alternative explanation uses fresh analogies and avoids repeating the previous explanation."""
        mock_response = MagicMock()
        mock_response.text = (
            "Imagine Russian Matryoshka nesting dolls. "
            "To reach the tiny prize inside, you open a doll, find a smaller doll, and repeat until the smallest doll (base case)."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        topic = "Recursion"
        question = "How does a recursive function terminate?"
        previous = "A recursive function calls itself on smaller inputs until meeting a base case condition."

        result = explain_differently(
            topic=topic,
            question=question,
            previous_explanation=previous,
            client=self.mock_client,
        )

        self.assertEqual(result, mock_response.text)
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        self.assertEqual(config.system_instruction, EXPLAIN_DIFFERENTLY_SYSTEM_PROMPT)

        contents = call_kwargs["contents"]
        prompt_text = contents[0].parts[0].text
        self.assertNotIn("Russian Matryoshka", prompt_text)
        self.assertIn(topic, prompt_text)
        self.assertIn(question, prompt_text)
        self.assertIn(previous, prompt_text)
        self.assertIn("DIFFERENT mental model", prompt_text)

    def test_explain_differently_empty_inputs(self):
        """Verify friendly prompt when no topic or question is supplied."""
        result = explain_differently(topic="", question="", previous_explanation="")
        self.assertIn("Please specify the topic or question", result)


class TestErrorHandlingAndAPIKey(unittest.TestCase):
    """Tests for missing API keys, API errors, empty responses, and edge cases."""

    def test_missing_api_key_ask_tutor(self):
        """Verify ask_tutor returns configuration error when GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GEMINI_API_KEY is not in env
            os.environ.pop("GEMINI_API_KEY", None)
            result = ask_tutor(topic="Math", level="Beginner", question="What is 2+2?")
            self.assertIn("Configuration Error", result)
            self.assertIn("GEMINI_API_KEY", result)

    def test_missing_api_key_explain_differently(self):
        """Verify explain_differently returns configuration error when GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            result = explain_differently(
                topic="Math", question="What is 2+2?", previous_explanation="It equals 4."
            )
            self.assertIn("Configuration Error", result)
            self.assertIn("GEMINI_API_KEY", result)

    def test_get_gemini_client_raises_value_error(self):
        """Verify get_gemini_client raises ValueError when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with self.assertRaises(ValueError) as context:
                get_gemini_client()
            self.assertIn("GEMINI_API_KEY", str(context.exception))

    def test_api_error_handling(self):
        """Verify that Gemini APIError is caught and a clean message is returned without crashing."""
        mock_client = MagicMock()
        mock_error = APIError(code=429, response_json={"error": {"message": "Resource exhausted: Quota exceeded"}})
        mock_client.models.generate_content.side_effect = mock_error

        result = ask_tutor("Chemistry", "Beginner", "What is an atom?", client=mock_client)

        self.assertIn("Gemini API Error", result)
        self.assertIn("Quota exceeded", result)

    def test_generic_exception_handling(self):
        """Verify unexpected network or runtime exceptions are handled gracefully."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ConnectionResetError("Connection reset by peer")

        result = ask_tutor("Chemistry", "Beginner", "What is an atom?", client=mock_client)

        self.assertIn("Error", result)
        self.assertIn("Connection reset by peer", result)

    def test_empty_response_handling(self):
        """Verify that empty response text returns an informative fallback message."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "   "  # whitespace only
        mock_client.models.generate_content.return_value = mock_response

        result = ask_tutor("Biology", "Intermediate", "Explain mitosis", client=mock_client)

        self.assertIn("could not generate an explanation", result)

    def test_empty_question_handling(self):
        """Verify that asking with empty question text returns a polite prompt."""
        result = ask_tutor(topic="Biology", level="Beginner", question="")
        self.assertIn("Please ask a question", result)


if __name__ == "__main__":
    unittest.main()
