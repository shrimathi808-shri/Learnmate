"""
quiz.py - LearnMate AI Quiz Generation & Answer Evaluation Module
-----------------------------------------------------------------
Author: Selvarani (CSE Student – Quiz Generation & Answer Evaluation)
Project: LearnMate – AI-Powered Personal Tutor

This module provides UI-independent AI quiz functionality for LearnMate:
- Reuses `get_gemini_client()` from `tutor.py` (never hardcodes API keys).
- Generates Beginner, Intermediate, and Advanced quizzes (MCQ, Short Answer, Mixed).
- Evaluates MCQs deterministically and Short Answer responses conceptually via Gemini.
- Awards partial marks for partially correct short answers while rejecting wrong ones.
- Calculates accurate scores, percentages, and performance categories.
- Produces clear, student-friendly feedback and explanations.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from google import genai
from google.genai import types
from google.genai.errors import APIError

from tutor import DEFAULT_MODEL, get_gemini_client

VALID_DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")
VALID_QUESTION_TYPES = ("MCQ", "Short Answer", "Mixed")

DEFAULT_MCQ_MARKS: float = 1.0
DEFAULT_SHORT_ANSWER_MARKS: float = 5.0
MAX_QUESTIONS_PER_QUIZ: int = 20

STATUS_LABELS: Dict[str, str] = {
    "correct": "Correct",
    "partially_correct": "Partially Correct",
    "incorrect": "Incorrect",
}


class QuizError(Exception):
    """Base exception for LearnMate quiz module errors."""


class QuizInputError(QuizError, ValueError):
    """Raised when user or caller inputs to the quiz module are invalid."""


class QuizConfigurationError(QuizError, ValueError):
    """Raised when the Gemini API key or configuration is missing or invalid."""


class QuizAPIError(QuizError):
    """Raised when a Gemini API call fails due to network, quota, or server errors."""


class QuizResponseError(QuizError):
    """Raised when the Gemini API returns empty, unparseable, or invalid JSON data."""


QUIZ_GENERATION_SYSTEM_PROMPT: str = (
    "You are LearnMate's expert educational quiz designer.\n"
    "Your job is to create clear, accurate, fair, and pedagogically sound quiz questions "
    "tailored to the student's topic and difficulty level.\n"
    "Rules:\n"
    "1. Always return valid JSON matching the exact requested schema.\n"
    "2. Never include markdown fences, comments, or extra prose outside the JSON.\n"
    "3. Difficulty calibration:\n"
    "   - Beginner: Basic concepts, clear simple English, foundational understanding.\n"
    "   - Intermediate: Applied understanding, standard technical terms, cause-and-effect.\n"
    "   - Advanced: Deep analytical reasoning, trade-offs, edge cases, and mechanisms.\n"
    "4. For MCQs: Provide exactly 4 distinct options labeled A, B, C, and D, with only ONE "
    "unambiguously correct answer and a simple explanation of why it is correct.\n"
    "5. For Short Answer questions: Provide a clear question, a concise model expected_answer, "
    "3 to 6 important keywords/concepts expected in a good answer, and a brief evaluation rubric."
)

SHORT_ANSWER_EVALUATION_SYSTEM_PROMPT: str = (
    "You are LearnMate's fair, encouraging, and accurate grading assistant.\n"
    "Your job is to evaluate a student's short-answer response against the expected answer and rubric.\n"
    "Rules:\n"
    "1. Evaluate conceptual understanding rather than requiring exact word-for-word matching.\n"
    "2. Do NOT blindly accept incorrect, off-topic, nonsensical, or vague placeholder answers.\n"
    "3. Award full marks when the student covers all key concepts accurately.\n"
    "4. Award partial marks fairly when the student demonstrates partial understanding.\n"
    "5. Award 0 marks when the answer is factually wrong, irrelevant, or shows no understanding.\n"
    "6. Use simple, warm, student-friendly English in explanations and suggestions.\n"
    "7. Always return valid JSON matching the requested schema with no extra text."
)


def _normalize_difficulty(difficulty: Optional[str]) -> str:
    """Validate and normalize difficulty level to Beginner, Intermediate, or Advanced."""
    cleaned = (difficulty or "").strip().lower()
    mapping = {
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
    }
    if cleaned not in mapping:
        raise QuizInputError(
            f"Invalid difficulty '{difficulty}'. Choose from: Beginner, Intermediate, or Advanced."
        )
    return mapping[cleaned]


def _normalize_question_type(question_type: Optional[str]) -> str:
    """Validate and normalize question type to MCQ, Short Answer, or Mixed."""
    cleaned = (question_type or "").strip().lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    mapping = {
        "mcq": "MCQ",
        "mcqs": "MCQ",
        "multiple choice": "MCQ",
        "short answer": "Short Answer",
        "short": "Short Answer",
        "mixed": "Mixed",
    }
    if cleaned not in mapping:
        raise QuizInputError(
            f"Invalid question type '{question_type}'. Choose from: MCQ, Short Answer, or Mixed."
        )
    return mapping[cleaned]


def _validate_num_questions(num_questions: Any) -> int:
    """Validate that num_questions is an integer within [1, MAX_QUESTIONS_PER_QUIZ]."""
    if isinstance(num_questions, bool) or not isinstance(num_questions, int):
        raise QuizInputError("Number of questions must be a whole number (integer).")
    if num_questions < 1 or num_questions > MAX_QUESTIONS_PER_QUIZ:
        raise QuizInputError(
            f"Number of questions must be between 1 and {MAX_QUESTIONS_PER_QUIZ}."
        )
    return num_questions


def _extract_json_payload(raw_text: str) -> Any:
    """Parse JSON from Gemini output, stripping optional markdown code fences."""
    if not raw_text or not raw_text.strip():
        raise QuizResponseError("Gemini returned an empty response instead of valid JSON.")

    text = raw_text.strip()

    # Strip markdown code fences if present (e.g. ```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to locate the outermost JSON object or array
        start_obj = text.find("{")
        start_arr = text.find("[")
        candidates = [idx for idx in (start_obj, start_arr) if idx != -1]
        if candidates:
            start = min(candidates)
            end_char = "}" if text[start] == "{" else "]"
            end = text.rfind(end_char)
            if end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass

        raise QuizResponseError(
            "Unable to parse AI response as valid JSON. Please try generating again."
        )


def _call_gemini_json(
    prompt: str,
    system_instruction: str,
    client: Optional[genai.Client] = None,
    temperature: float = 0.4,
) -> Any:
    """
    Execute a Gemini API call using tutor.get_gemini_client() and parse its JSON response.
    Translates configuration, rate-limit, network, and parsing failures into QuizError subclasses.
    """
    try:
        gemini_client = client or get_gemini_client()
    except ValueError as exc:
        raise QuizConfigurationError(str(exc)) from exc
    except Exception as exc:
        raise QuizConfigurationError(
            f"Unable to initialize Gemini client: {str(exc)}"
        ) from exc

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
    except APIError as exc:
        error_msg = getattr(exc, "message", str(exc)) or str(exc)
        status_code = getattr(exc, "code", None)
        lowered = error_msg.lower()
        if status_code == 429 or "quota" in lowered or "rate limit" in lowered or "resource exhausted" in lowered:
            raise QuizAPIError(
                f"Gemini API rate limit or quota exceeded: {error_msg}. Please wait a moment and try again."
            ) from exc
        if status_code in (401, 403) or "api key" in lowered or "unauthorized" in lowered:
            raise QuizConfigurationError(
                f"Gemini API authentication failed: {error_msg}. Please check your GEMINI_API_KEY."
            ) from exc
        raise QuizAPIError(
            f"Gemini API Error: {error_msg}. Please check your connection and try again."
        ) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise QuizAPIError(
            f"Network error while contacting Gemini API: {str(exc)}. Please check your internet connection."
        ) from exc
    except Exception as exc:
        raise QuizAPIError(
            f"Unexpected error while calling Gemini API: {str(exc)}"
        ) from exc

    raw_text = getattr(response, "text", None) if response else None
    if not raw_text or not str(raw_text).strip():
        raise QuizResponseError(
            "Gemini returned an empty response. Please try again."
        )

    return _extract_json_payload(str(raw_text))


def _normalize_mcq_options(raw_options: Any) -> Dict[str, str]:
    """Normalize MCQ options from a dict or 4-item list into {'A': ..., 'B': ..., 'C': ..., 'D': ...}."""
    labels = ("A", "B", "C", "D")
    normalized: Dict[str, str] = {}

    if isinstance(raw_options, dict):
        for key, value in raw_options.items():
            clean_key = str(key).strip().upper().rstrip(").:")
            if clean_key in labels and str(value).strip():
                normalized[clean_key] = str(value).strip()
    elif isinstance(raw_options, list) and len(raw_options) == 4:
        for label, item in zip(labels, raw_options):
            text = re.sub(r"^[A-Da-d][\)\.\:\-\s]+", "", str(item).strip()).strip()
            if text:
                normalized[label] = text

    if set(normalized.keys()) != set(labels):
        raise QuizResponseError(
            "Generated MCQ does not contain valid options A, B, C, and D."
        )
    return {label: normalized[label] for label in labels}


def _normalize_correct_option(raw_answer: Any, options: Dict[str, str]) -> str:
    """Normalize an MCQ correct answer into one of 'A', 'B', 'C', or 'D'."""
    answer_str = str(raw_answer or "").strip()
    if not answer_str:
        raise QuizResponseError("Generated MCQ is missing a correct answer.")

    # Direct letter or letter with punctuation (e.g., "A", "b)", "C.")
    letter_match = re.match(r"^([A-Da-d])(?:[\)\.\:\s]|$)", answer_str)
    if letter_match:
        return letter_match.group(1).upper()

    # Fallback: match against option text
    lowered = answer_str.lower()
    for label, text in options.items():
        if text.strip().lower() == lowered:
            return label

    raise QuizResponseError(
        f"Generated MCQ has an invalid correct_answer '{answer_str}'."
    )


def _validate_and_normalize_questions(
    raw_data: Any,
    expected_type: str,
    num_questions: int,
) -> List[Dict[str, Any]]:
    """Validate AI-generated quiz JSON and normalize each question structure."""
    if isinstance(raw_data, dict):
        raw_questions = raw_data.get("questions")
    elif isinstance(raw_data, list):
        raw_questions = raw_data
    else:
        raise QuizResponseError("AI response must be a JSON object containing a 'questions' list.")

    if not isinstance(raw_questions, list) or not raw_questions:
        raise QuizResponseError("AI did not generate any quiz questions.")

    normalized_questions: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_questions[:num_questions], start=1):
        if not isinstance(item, dict):
            raise QuizResponseError(f"Question {idx} is malformed.")

        q_text = str(item.get("question", "")).strip()
        if not q_text:
            raise QuizResponseError(f"Question {idx} is missing question text.")

        raw_q_type = str(item.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if not raw_q_type:
            if "options" in item:
                raw_q_type = "mcq"
            elif expected_type == "Short Answer":
                raw_q_type = "short_answer"
            elif expected_type == "MCQ":
                raw_q_type = "mcq"

        if raw_q_type in ("mcq", "multiple_choice"):
            if expected_type == "Short Answer":
                raise QuizResponseError("AI generated an MCQ when Short Answer was requested.")
            options = _normalize_mcq_options(item.get("options"))
            correct_answer = _normalize_correct_option(item.get("correct_answer"), options)
            explanation = str(item.get("explanation", "")).strip()
            if not explanation:
                explanation = (
                    f"Option {correct_answer} ({options[correct_answer]}) is the correct answer."
                )

            normalized_questions.append(
                {
                    "id": idx,
                    "type": "mcq",
                    "question": q_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "explanation": explanation,
                    "max_marks": DEFAULT_MCQ_MARKS,
                }
            )
        elif raw_q_type in ("short_answer", "short"):
            if expected_type == "MCQ":
                raise QuizResponseError("AI generated a Short Answer question when MCQ was requested.")
            expected_answer = str(item.get("expected_answer", "")).strip()
            if not expected_answer:
                raise QuizResponseError(f"Short answer question {idx} is missing expected_answer.")

            raw_keywords = item.get("keywords", [])
            if isinstance(raw_keywords, str):
                keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
            elif isinstance(raw_keywords, list):
                keywords = [str(kw).strip() for kw in raw_keywords if str(kw).strip()]
            else:
                keywords = []
            if not keywords:
                raise QuizResponseError(f"Short answer question {idx} is missing keywords.")

            rubric = str(item.get("rubric", "")).strip()
            if not rubric:
                rubric = (
                    f"Award full marks ({int(DEFAULT_SHORT_ANSWER_MARKS)}) if the student accurately "
                    f"explains the core concept covering: {', '.join(keywords)}. "
                    "Award partial marks for incomplete but accurate points."
                )

            normalized_questions.append(
                {
                    "id": idx,
                    "type": "short_answer",
                    "question": q_text,
                    "expected_answer": expected_answer,
                    "keywords": keywords,
                    "rubric": rubric,
                    "max_marks": DEFAULT_SHORT_ANSWER_MARKS,
                }
            )
        else:
            raise QuizResponseError(f"Question {idx} has unsupported type '{raw_q_type}'.")

    if len(normalized_questions) < num_questions:
        raise QuizResponseError(
            f"Expected {num_questions} questions, but AI only returned {len(normalized_questions)}."
        )

    return normalized_questions


def generate_quiz(
    topic: str,
    difficulty: str = "Beginner",
    num_questions: int = 5,
    question_type: str = "Mixed",
    client: Optional[genai.Client] = None,
) -> Dict[str, Any]:
    """
    Generate an AI-powered quiz for a given topic, difficulty, count, and question type.

    Args:
        topic: Subject or topic for the quiz (e.g., 'Photosynthesis', 'Binary Search').
        difficulty: 'Beginner', 'Intermediate', or 'Advanced'.
        num_questions: Number of questions to generate (1 to 20).
        question_type: 'MCQ', 'Short Answer', or 'Mixed'.
        client: Optional pre-configured genai.Client (defaults to tutor.get_gemini_client()).

    Returns:
        Dict containing quiz metadata and a list of normalized question dicts.

    Raises:
        QuizInputError: If any input parameter is empty or invalid.
        QuizConfigurationError: If the Gemini API key is missing or invalid.
        QuizAPIError: If the Gemini API request fails.
        QuizResponseError: If the AI response cannot be parsed or validated.
    """
    clean_topic = (topic or "").strip()
    if not clean_topic:
        raise QuizInputError("Topic cannot be empty. Please enter a subject or topic.")

    norm_difficulty = _normalize_difficulty(difficulty)
    norm_q_type = _normalize_question_type(question_type)
    valid_count = _validate_num_questions(num_questions)

    if norm_q_type == "MCQ":
        type_instruction = f"Generate exactly {valid_count} Multiple Choice Questions (type: 'mcq')."
    elif norm_q_type == "Short Answer":
        type_instruction = f"Generate exactly {valid_count} Short Answer Questions (type: 'short_answer')."
    else:
        if valid_count == 1:
            type_instruction = "Generate 1 question (either 'mcq' or 'short_answer')."
        else:
            mcq_count = (valid_count + 1) // 2
            short_count = valid_count - mcq_count
            type_instruction = (
                f"Generate a MIXED quiz with exactly {mcq_count} MCQ question(s) (type: 'mcq') "
                f"and {short_count} Short Answer question(s) (type: 'short_answer'), totaling {valid_count} questions."
            )

    prompt = (
        f"Create a quiz on the topic: \"{clean_topic}\"\n"
        f"Difficulty Level: {norm_difficulty}\n"
        f"Total Questions: {valid_count}\n"
        f"Question Mix: {type_instruction}\n\n"
        "Return a JSON object with this exact structure:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "type": "mcq",\n'
        '      "question": "Clear question text?",\n'
        '      "options": {"A": "Option 1", "B": "Option 2", "C": "Option 3", "D": "Option 4"},\n'
        '      "correct_answer": "A",\n'
        '      "explanation": "Simple, student-friendly explanation of why option A is right."\n'
        "    },\n"
        "    {\n"
        '      "type": "short_answer",\n'
        '      "question": "Clear open-ended question?",\n'
        '      "expected_answer": "Concise, accurate model answer.",\n'
        '      "keywords": ["concept 1", "concept 2", "concept 3"],\n'
        '      "rubric": "Full marks (5) for covering all concepts; partial marks (2-4) for partial explanation; 0 for incorrect."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Include only the requested question type(s) and ensure every field is populated accurately."
    )

    raw_data = _call_gemini_json(
        prompt=prompt,
        system_instruction=QUIZ_GENERATION_SYSTEM_PROMPT,
        client=client,
        temperature=0.5,
    )
    questions = _validate_and_normalize_questions(
        raw_data=raw_data,
        expected_type=norm_q_type,
        num_questions=valid_count,
    )
    total_max_marks = round(sum(float(q["max_marks"]) for q in questions), 2)

    return {
        "topic": clean_topic,
        "difficulty": norm_difficulty,
        "question_type": norm_q_type,
        "num_questions": len(questions),
        "total_max_marks": total_max_marks,
        "questions": questions,
    }


def _normalize_student_mcq_choice(
    student_answer: Optional[str],
    options: Dict[str, str],
) -> str:
    """Normalize a student's MCQ selection ('A', 'a)', option text, etc.) or return '' if empty."""
    if student_answer is None:
        return ""
    raw = str(student_answer).strip()
    if not raw:
        return ""

    letter_match = re.match(r"^([A-Da-d])(?:[\)\.\:\s]|$)", raw)
    if letter_match:
        return letter_match.group(1).upper()

    lowered = raw.lower()
    for label, text in options.items():
        if text.strip().lower() == lowered:
            return label

    return raw


def generate_feedback(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build standardized, student-friendly feedback from an evaluation dictionary.

    Ensures all six required feedback components are present:
    - Answer status (Correct / Partially Correct / Incorrect)
    - Marks obtained (with maximum marks)
    - Correct answer
    - Simple explanation
    - What the student missed
    - Suggestions for improvement

    Args:
        evaluation: Dict containing evaluation fields.

    Returns:
        Dict containing the enriched evaluation fields plus `status_label` and `feedback_text`.
    """
    status = str(evaluation.get("status", "incorrect")).strip().lower()
    if status not in STATUS_LABELS:
        status = "incorrect"

    status_label = STATUS_LABELS[status]
    marks_obtained = round(float(evaluation.get("marks_obtained", 0.0)), 2)
    max_marks = round(float(evaluation.get("max_marks", 1.0)), 2)
    correct_answer = str(evaluation.get("correct_answer", "Not provided")).strip()
    explanation = str(
        evaluation.get("explanation", "Review the core concept behind this question.")
    ).strip()

    raw_missed = evaluation.get("missed", [])
    if isinstance(raw_missed, str):
        missed = [raw_missed.strip()] if raw_missed.strip() else []
    elif isinstance(raw_missed, list):
        missed = [str(item).strip() for item in raw_missed if str(item).strip()]
    else:
        missed = []

    suggestions = str(evaluation.get("suggestions", "")).strip()
    if not suggestions:
        if status == "correct":
            suggestions = "Great job! Keep practicing slightly harder questions to reinforce this concept."
        elif status == "partially_correct":
            suggestions = "You are on the right track. Review the missing concepts above to make your answer complete."
        else:
            suggestions = "Take a moment to review the explanation and try explaining the main idea in your own words."

    missed_display = ", ".join(missed) if missed else "Nothing — you covered all key points!"

    feedback_text = (
        f"Result: {status_label}\n"
        f"Marks Obtained: {marks_obtained:g} / {max_marks:g}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Explanation: {explanation}\n"
        f"What You Missed: {missed_display}\n"
        f"Suggestions for Improvement: {suggestions}"
    )

    result = dict(evaluation)
    result.update(
        {
            "status": status,
            "status_label": status_label,
            "marks_obtained": marks_obtained,
            "max_marks": max_marks,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "missed": missed,
            "suggestions": suggestions,
            "feedback_text": feedback_text,
        }
    )
    return result


def evaluate_mcq(
    question: Dict[str, Any],
    student_answer: Optional[str],
) -> Dict[str, Any]:
    """
    Evaluate a student's Multiple Choice Question answer against the correct option.

    Args:
        question: Normalized MCQ dictionary containing 'options', 'correct_answer', and 'explanation'.
        student_answer: Student's selected option (e.g., 'A', 'b', or full option text).

    Returns:
        Dict with evaluation results, marks, and student-friendly feedback.

    Raises:
        QuizInputError: If the question dictionary is missing required MCQ fields.
    """
    if not isinstance(question, dict) or "options" not in question or "correct_answer" not in question:
        raise QuizInputError("Invalid MCQ question data: missing 'options' or 'correct_answer'.")

    options = _normalize_mcq_options(question["options"])
    correct_key = _normalize_correct_option(question["correct_answer"], options)
    max_marks = float(question.get("max_marks", DEFAULT_MCQ_MARKS))
    if max_marks <= 0:
        max_marks = DEFAULT_MCQ_MARKS

    correct_text = options[correct_key]
    formatted_correct = f"{correct_key}) {correct_text}"
    explanation = str(
        question.get("explanation")
        or f"Option {correct_key} ({correct_text}) is the right choice for this question."
    ).strip()

    choice = _normalize_student_mcq_choice(student_answer, options)

    if not choice:
        evaluation = {
            "question_id": question.get("id"),
            "question_type": "mcq",
            "question": str(question.get("question", "")).strip(),
            "student_answer": "",
            "correct_answer": formatted_correct,
            "correct_option": correct_key,
            "status": "incorrect",
            "is_correct": False,
            "marks_obtained": 0.0,
            "max_marks": max_marks,
            "explanation": explanation,
            "missed": ["You did not select any option."],
            "suggestions": "Make sure to choose one of the options (A, B, C, or D) before submitting.",
        }
        return generate_feedback(evaluation)

    if choice == correct_key:
        evaluation = {
            "question_id": question.get("id"),
            "question_type": "mcq",
            "question": str(question.get("question", "")).strip(),
            "student_answer": choice,
            "correct_answer": formatted_correct,
            "correct_option": correct_key,
            "status": "correct",
            "is_correct": True,
            "marks_obtained": max_marks,
            "max_marks": max_marks,
            "explanation": explanation,
            "missed": [],
            "suggestions": "Well done! Your understanding of this concept is solid.",
        }
        return generate_feedback(evaluation)

    if choice in options:
        selected_desc = f"{choice}) {options[choice]}"
        missed_msg = f"You chose {selected_desc}, instead of {formatted_correct}."
    else:
        missed_msg = f"'{choice}' is not a valid option (choose A, B, C, or D)."

    evaluation = {
        "question_id": question.get("id"),
        "question_type": "mcq",
        "question": str(question.get("question", "")).strip(),
        "student_answer": choice,
        "correct_answer": formatted_correct,
        "correct_option": correct_key,
        "status": "incorrect",
        "is_correct": False,
        "marks_obtained": 0.0,
        "max_marks": max_marks,
        "explanation": explanation,
        "missed": [missed_msg],
        "suggestions": (
            f"Re-read the question carefully and focus on why {formatted_correct} fits best."
        ),
    }
    return generate_feedback(evaluation)


def evaluate_short_answer(
    question: Dict[str, Any],
    student_answer: Optional[str],
    client: Optional[genai.Client] = None,
) -> Dict[str, Any]:
    """
    Evaluate a student's short-answer response conceptually using the Gemini API.

    Supports partial marking, identifies missing concepts and factual errors, and
    rejects empty or incorrect answers fairly.

    Args:
        question: Normalized short-answer dict with 'question', 'expected_answer', 'keywords', 'rubric'.
        student_answer: Student's written answer string.
        client: Optional pre-configured genai.Client.

    Returns:
        Dict with evaluation results, marks, and student-friendly feedback.

    Raises:
        QuizInputError: If the question dictionary is missing required fields.
        QuizConfigurationError: If Gemini API key is missing.
        QuizAPIError: If the Gemini API call fails.
        QuizResponseError: If the AI evaluation response is invalid JSON.
    """
    if not isinstance(question, dict):
        raise QuizInputError("Question data must be a dictionary.")

    q_text = str(question.get("question", "")).strip()
    expected_answer = str(question.get("expected_answer", "")).strip()
    if not q_text or not expected_answer:
        raise QuizInputError(
            "Invalid short-answer question data: missing 'question' or 'expected_answer'."
        )

    raw_keywords = question.get("keywords", [])
    if isinstance(raw_keywords, str):
        keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
    elif isinstance(raw_keywords, list):
        keywords = [str(kw).strip() for kw in raw_keywords if str(kw).strip()]
    else:
        keywords = []

    rubric = str(question.get("rubric", "")).strip() or (
        "Award full marks for accurate conceptual coverage, partial marks for partial understanding, "
        "and 0 for incorrect or irrelevant answers."
    )
    max_marks = float(question.get("max_marks", DEFAULT_SHORT_ANSWER_MARKS))
    if max_marks <= 0:
        max_marks = DEFAULT_SHORT_ANSWER_MARKS

    cleaned_answer = str(student_answer or "").strip()

    # Empty or non-alphanumeric answers receive 0 marks without calling the API
    if not cleaned_answer or not re.search(r"[A-Za-z0-9]", cleaned_answer):
        missed_items = keywords if keywords else ["Key concepts from the expected answer."]
        evaluation = {
            "question_id": question.get("id"),
            "question_type": "short_answer",
            "question": q_text,
            "student_answer": cleaned_answer,
            "correct_answer": expected_answer,
            "status": "incorrect",
            "is_correct": False,
            "marks_obtained": 0.0,
            "max_marks": max_marks,
            "explanation": f"No valid answer was submitted. Model answer: {expected_answer}",
            "missed": missed_items,
            "incorrect_points": [],
            "suggestions": (
                "Write a clear answer in your own words covering the main ideas and key terms."
            ),
        }
        return generate_feedback(evaluation)

    prompt = (
        f"Question: {q_text}\n"
        f"Expected Answer: {expected_answer}\n"
        f"Expected Keywords/Concepts: {', '.join(keywords) if keywords else 'N/A'}\n"
        f"Evaluation Rubric: {rubric}\n"
        f"Maximum Marks: {max_marks:g}\n\n"
        f"Student's Answer: \"\"\"\n{cleaned_answer}\n\"\"\"\n\n"
        "Evaluate the student's conceptual understanding (do not require exact wording).\n"
        "IMPORTANT: If the student's answer is wrong, irrelevant, gibberish, or says things like 'I don't know', "
        "award 0 marks and mark status as 'incorrect'.\n"
        "Return a JSON object with this exact schema:\n"
        "{\n"
        '  "status": "correct" | "partially_correct" | "incorrect",\n'
        f'  "marks_obtained": <number from 0 to {max_marks:g}>,\n'
        '  "explanation": "Simple, student-friendly explanation of the concept and evaluation.",\n'
        '  "missed": ["List of key concepts or ideas the student missed (empty list if none)"],\n'
        '  "incorrect_points": ["List of factually incorrect claims made by the student (empty list if none)"],\n'
        '  "suggestions": "Specific, encouraging advice on how the student can improve."\n'
        "}"
    )

    ai_eval = _call_gemini_json(
        prompt=prompt,
        system_instruction=SHORT_ANSWER_EVALUATION_SYSTEM_PROMPT,
        client=client,
        temperature=0.2,
    )
    if not isinstance(ai_eval, dict):
        raise QuizResponseError("AI evaluation response must be a JSON object.")

    try:
        raw_marks = float(ai_eval.get("marks_obtained", 0.0))
    except (TypeError, ValueError) as exc:
        raise QuizResponseError("AI evaluation returned a non-numeric marks_obtained value.") from exc

    marks = round(max(0.0, min(max_marks, raw_marks)), 2)

    # Keep status strictly consistent with numeric marks
    if marks <= 0.0:
        status = "incorrect"
        marks = 0.0
    elif marks >= max_marks:
        status = "correct"
        marks = round(max_marks, 2)
    else:
        status = "partially_correct"

    raw_missed = ai_eval.get("missed", [])
    if isinstance(raw_missed, str):
        missed = [m.strip() for m in raw_missed.split(",") if m.strip()]
    elif isinstance(raw_missed, list):
        missed = [str(m).strip() for m in raw_missed if str(m).strip()]
    else:
        missed = []

    raw_incorrect = ai_eval.get("incorrect_points", [])
    if isinstance(raw_incorrect, list):
        incorrect_points = [str(p).strip() for p in raw_incorrect if str(p).strip()]
    elif isinstance(raw_incorrect, str) and raw_incorrect.strip():
        incorrect_points = [raw_incorrect.strip()]
    else:
        incorrect_points = []

    # Merge any factual inaccuracies into missed/feedback clarity
    combined_missed = list(missed)
    for point in incorrect_points:
        note = f"Incorrect point: {point}"
        if note not in combined_missed:
            combined_missed.append(note)

    explanation = str(ai_eval.get("explanation", "")).strip() or expected_answer
    suggestions = str(ai_eval.get("suggestions", "")).strip()

    evaluation = {
        "question_id": question.get("id"),
        "question_type": "short_answer",
        "question": q_text,
        "student_answer": cleaned_answer,
        "correct_answer": expected_answer,
        "status": status,
        "is_correct": status == "correct",
        "marks_obtained": marks,
        "max_marks": max_marks,
        "explanation": explanation,
        "missed": combined_missed,
        "incorrect_points": incorrect_points,
        "suggestions": suggestions,
    }
    return generate_feedback(evaluation)


def get_performance_level(percentage: float) -> str:
    """
    Map a percentage score to the LearnMate performance category.

    Categories:
    - 90–100%: Excellent
    - 75–89%: Very Good
    - 50–74%: Good
    - Below 50%: Needs Improvement
    """
    if percentage >= 90.0:
        return "Excellent"
    if percentage >= 75.0:
        return "Very Good"
    if percentage >= 50.0:
        return "Good"
    return "Needs Improvement"


def calculate_score(evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate quiz score summary across evaluated MCQ and short-answer questions.

    Args:
        evaluations: Sequence of evaluation dictionaries returned by evaluate_mcq / evaluate_short_answer.

    Returns:
        Dict containing total_questions, correct_answers, incorrect_answers,
        partially_correct_answers, total_marks_obtained, max_possible_marks,
        percentage, and performance_level.
    """
    if not evaluations:
        return {
            "total_questions": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "partially_correct_answers": 0,
            "total_marks_obtained": 0.0,
            "max_possible_marks": 0.0,
            "percentage": 0.0,
            "performance_level": "Needs Improvement",
        }

    total_questions = len(evaluations)
    correct_answers = 0
    incorrect_answers = 0
    partially_correct_answers = 0
    total_marks_obtained = 0.0
    max_possible_marks = 0.0

    for item in evaluations:
        status = str(item.get("status", "incorrect")).strip().lower()
        if status == "correct":
            correct_answers += 1
        elif status == "partially_correct":
            partially_correct_answers += 1
        else:
            incorrect_answers += 1

        max_m = max(0.0, float(item.get("max_marks", 0.0)))
        obtained_m = max(0.0, min(max_m, float(item.get("marks_obtained", 0.0))))
        total_marks_obtained += obtained_m
        max_possible_marks += max_m

    total_marks_obtained = round(total_marks_obtained, 2)
    max_possible_marks = round(max_possible_marks, 2)
    percentage = (
        round((total_marks_obtained / max_possible_marks) * 100.0, 2)
        if max_possible_marks > 0
        else 0.0
    )
    performance_level = get_performance_level(percentage)

    return {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "partially_correct_answers": partially_correct_answers,
        "total_marks_obtained": total_marks_obtained,
        "max_possible_marks": max_possible_marks,
        "percentage": percentage,
        "performance_level": performance_level,
    }


def evaluate_quiz(
    quiz: Dict[str, Any],
    student_answers: Union[Dict[Any, Any], Sequence[Any]],
    client: Optional[genai.Client] = None,
) -> Dict[str, Any]:
    """
    Evaluate all answers for a generated quiz and compute the final score summary.

    Args:
        quiz: Quiz dictionary produced by `generate_quiz()`.
        student_answers: Either a dict mapping question id (int or str) to student answer,
                         or a sequence aligned with `quiz["questions"]`.
        client: Optional pre-configured genai.Client.

    Returns:
        Dict with keys: 'topic', 'difficulty', 'question_type', 'evaluations', and 'summary'.

    Raises:
        QuizInputError: If `quiz` structure is invalid.
    """
    if not isinstance(quiz, dict) or not isinstance(quiz.get("questions"), list):
        raise QuizInputError("Invalid quiz object: expected a dict with a 'questions' list.")

    questions = quiz["questions"]
    evaluations: List[Dict[str, Any]] = []

    for index, question in enumerate(questions):
        q_id = question.get("id", index + 1)

        if isinstance(student_answers, dict):
            if q_id in student_answers:
                answer = student_answers[q_id]
            elif str(q_id) in student_answers:
                answer = student_answers[str(q_id)]
            else:
                answer = student_answers.get(index, "")
        elif isinstance(student_answers, (list, tuple)):
            answer = student_answers[index] if index < len(student_answers) else ""
        else:
            answer = ""

        q_type = str(question.get("type", "")).strip().lower()
        if q_type == "mcq":
            result = evaluate_mcq(question, answer)
        elif q_type == "short_answer":
            result = evaluate_short_answer(question, answer, client=client)
        else:
            raise QuizInputError(f"Unsupported question type '{q_type}' at question {q_id}.")

        evaluations.append(result)

    summary = calculate_score(evaluations)

    return {
        "topic": quiz.get("topic", ""),
        "difficulty": quiz.get("difficulty", ""),
        "question_type": quiz.get("question_type", ""),
        "evaluations": evaluations,
        "summary": summary,
    }


def get_sample_quiz() -> Dict[str, Any]:
    """
    Return a ready-to-use offline sample quiz demonstrating both MCQs and Short Answer questions.
    Useful for demos, UI previews, and quick testing without an API key.
    """
    questions = [
        {
            "id": 1,
            "type": "mcq",
            "question": "In Python, which data structure stores items in key-value pairs?",
            "options": {
                "A": "List",
                "B": "Tuple",
                "C": "Dictionary",
                "D": "Set",
            },
            "correct_answer": "C",
            "explanation": (
                "A Python dictionary stores data as key-value pairs (e.g., {'name': 'Selvarani'}), "
                "allowing fast lookups by key."
            ),
            "max_marks": DEFAULT_MCQ_MARKS,
        },
        {
            "id": 2,
            "type": "mcq",
            "question": "What is the time complexity of Binary Search on a sorted list of n elements?",
            "options": {
                "A": "O(n)",
                "B": "O(log n)",
                "C": "O(n^2)",
                "D": "O(1)",
            },
            "correct_answer": "B",
            "explanation": (
                "Binary Search cuts the search range in half on every step, resulting in O(log n) time complexity."
            ),
            "max_marks": DEFAULT_MCQ_MARKS,
        },
        {
            "id": 3,
            "type": "short_answer",
            "question": "Explain the main difference between a Python list and a tuple.",
            "expected_answer": (
                "A Python list is mutable, meaning its elements can be changed, added, or removed after creation. "
                "A tuple is immutable, meaning its contents cannot be modified once created."
            ),
            "keywords": ["mutable", "immutable", "modify", "list", "tuple"],
            "rubric": (
                "5 marks: Correctly explains that lists are mutable and tuples are immutable with clear meaning. "
                "2-4 marks: Mentions one side or gives an incomplete explanation. 0 marks: Incorrect."
            ),
            "max_marks": DEFAULT_SHORT_ANSWER_MARKS,
        },
    ]

    return {
        "topic": "Python & Data Structures",
        "difficulty": "Beginner",
        "question_type": "Mixed",
        "num_questions": len(questions),
        "total_max_marks": round(sum(float(q["max_marks"]) for q in questions), 2),
        "questions": questions,
    }
