"""
tutor.py - LearnMate AI Tutor Module
------------------------------------
Author: Shrimathi (Team Lead + LLM/API Integration)
Project: LearnMate – AI-Powered Personal Tutor

This module provides the core AI Tutor functionality for LearnMate:
- Integrates with the official Google GenAI Python SDK (`google-genai`).
- Manages secure API key loading via `python-dotenv`.
- Powers multi-turn academic tutoring across 3 distinct learning modes:
  Beginner, Intermediate, and Advanced.
- Provides an alternative explanation engine (`explain_differently`)
  for students struggling with specific concepts.
- Offers robust, graceful error handling for missing keys, network issues,
  and API limits so teammate interfaces (e.g., Streamlit) never crash.
"""

import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables from .env file
load_dotenv()

# Default Gemini model (Gemini 2.5 Flash is recommended, fast, and highly capable)
DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Learning mode prompt guidelines
SYSTEM_PROMPTS: Dict[str, str] = {
    "beginner": (
        "You are LearnMate, a friendly, patient, and encouraging AI tutor specialized in teaching beginners.\n"
        "Your pedagogical guidelines:\n"
        "1. Language: Use very simple, clear English suitable for someone new to the subject.\n"
        "2. Step-by-step: Break down the concept into intuitive, progressive steps.\n"
        "3. Everyday examples: Anchor explanations in relatable, real-world everyday analogies and familiar scenarios.\n"
        "4. Jargon-free: Avoid unnecessary technical jargon and complex formulas. If a technical term is essential, "
        "explain it in plain English first.\n"
        "5. Tone: Warm, engaging, and supportive. Inspire curiosity and build student confidence."
    ),
    "intermediate": (
        "You are LearnMate, a knowledgeable, structured, and practical AI tutor for intermediate students.\n"
        "Your pedagogical guidelines:\n"
        "1. Detailed explanations: Provide comprehensive explanations explaining both 'what' and 'why'.\n"
        "2. Technical terminology: Introduce and define standard domain terminology accurately.\n"
        "3. Practical examples: Include concrete practical examples, code snippets, or applied scenarios.\n"
        "4. Structure: Organize points with clear headings, bullet points, or sequential stages.\n"
        "5. Tone: Engaging, intellectually stimulating, and professional."
    ),
    "advanced": (
        "You are LearnMate, an expert academic AI tutor for advanced students, researchers, and practitioners.\n"
        "Your pedagogical guidelines:\n"
        "1. In-depth rigor: Provide deep, rigorous conceptual and theoretical analysis.\n"
        "2. Technical details: Cover underlying mechanisms, formal definitions, math/algorithms, and architectural trade-offs.\n"
        "3. Challenging examples: Use complex, real-world case studies and non-trivial edge cases.\n"
        "4. Applications & connections: Discuss practical applications, industry standards, and related advanced concepts.\n"
        "5. Tone: Scholarly, precise, and intellectually demanding."
    ),
}

EXPLAIN_DIFFERENTLY_SYSTEM_PROMPT: str = (
    "You are LearnMate, an adaptive AI tutor specialized in helping students overcome learning roadblocks.\n"
    "When a student does not understand an initial explanation, your goal is to explain the same concept "
    "from a completely new and fresh angle.\n"
    "Your guidelines:\n"
    "1. Completely Different Approach: Adopt an entirely distinct mental model, perspective, or pedagogical framework "
    "(e.g., visual imagery, narrative storytelling, physical machinery analogy, or reverse engineering).\n"
    "2. New Analogies & Examples: Use novel, vivid analogies and different concrete examples.\n"
    "3. No Repetition: Do NOT repeat the explanations, phrasing, structure, or analogies used in the previous explanation.\n"
    "4. Clarity First: Strip away confusion and make the core intuition immediately click.\n"
    "5. Tone: Encouraging, empathetic, and clear."
)


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initialize and return the official Google GenAI Client.

    Args:
        api_key: Optional API key. If not provided, it is read from the
                 GEMINI_API_KEY environment variable.

    Returns:
        genai.Client: Configured Gemini client instance.

    Raises:
        ValueError: If no API key is provided and GEMINI_API_KEY is not set.
    """
    # Ensure fresh check in case .env was updated
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key or not key.strip() or key.strip() == "your_api_key_here":
        raise ValueError(
            "Gemini API key is missing or not configured. "
            "Please set a valid GEMINI_API_KEY in your .env file."
        )

    return genai.Client(api_key=key.strip())


def _format_history(history: Optional[List[Any]]) -> List[types.Content]:
    """
    Normalize and convert various conversation history formats into a list of
    google.genai.types.Content objects.

    Supported formats:
    - List of types.Content
    - List of dicts: {"role": "user"|"assistant"|"model", "content": "..."}
      or {"role": "...", "text": "..."} or {"role": "...", "parts": [...]}
    - List of tuples: ("user"|"assistant", "message text")

    Args:
        history: Raw history list or None.

    Returns:
        List of types.Content objects ready for Gemini API.
    """
    if not history:
        return []

    formatted: List[types.Content] = []

    for item in history:
        if isinstance(item, types.Content):
            formatted.append(item)
            continue

        role = "user"
        text = ""

        if isinstance(item, dict):
            raw_role = str(item.get("role", "user")).lower()
            role = "model" if raw_role in ("assistant", "model", "tutor", "bot") else "user"

            if "content" in item:
                text = str(item["content"])
            elif "text" in item:
                text = str(item["text"])
            elif "parts" in item:
                parts = item["parts"]
                if isinstance(parts, list):
                    text = " ".join(
                        getattr(p, "text", str(p)) if hasattr(p, "text") else str(p)
                        for p in parts
                    )
                else:
                    text = str(parts)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            raw_role, raw_text = item
            role = "model" if str(raw_role).lower() in ("assistant", "model", "tutor", "bot") else "user"
            text = str(raw_text)
        elif isinstance(item, str):
            role = "user"
            text = item
        else:
            text = str(item)

        if text.strip():
            formatted.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text.strip())],
                )
            )

    return formatted


def ask_tutor(
    topic: str,
    level: str,
    question: str,
    history: Optional[List[Any]] = None,
    client: Optional[genai.Client] = None,
) -> str:
    """
    Answer student academic questions with a personalized explanation based on learning level.

    Args:
        topic: The academic topic or subject (e.g., 'Photosynthesis', 'Binary Search', 'Thermodynamics').
        level: Student's learning level ('Beginner', 'Intermediate', 'Advanced').
        question: The specific question or concept the student needs help with.
        history: Optional list representing previous conversation turns to support follow-up questions.
        client: Optional pre-configured genai.Client instance (useful for testing or custom configs).

    Returns:
        str: AI-generated personalized explanation, or a user-friendly error message if an issue occurs.
    """
    # Validate required inputs
    clean_topic = (topic or "").strip()
    clean_question = (question or "").strip()

    if not clean_question:
        return "Please ask a question so I can help you learn!"

    # Normalize level (default to beginner if unrecognized)
    normalized_level = (level or "beginner").strip().lower()
    system_instruction = SYSTEM_PROMPTS.get(normalized_level, SYSTEM_PROMPTS["beginner"])

    # Prepare Gemini client
    try:
        gemini_client = client or get_gemini_client()
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except Exception as e:
        return f"Initialization Error: Unable to configure Gemini client: {str(e)}"

    # Format conversation history
    contents = _format_history(history)

    # Build prompt for the current turn
    current_prompt = (
        f"Topic: {clean_topic}\n"
        f"Student Question: {clean_question}\n\n"
        "Please provide a clear, accurate, and student-friendly explanation tailored to the requested learning level. "
        "Include relevant examples where useful."
    )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=current_prompt)],
        )
    )

    # Configure generation
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.7,
    )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        if not response or not response.text or not response.text.strip():
            return (
                "I apologize, but I could not generate an explanation for this question. "
                "Please try rephrasing your question or asking about a specific part of the topic."
            )

        return response.text.strip()

    except APIError as e:
        error_msg = getattr(e, "message", str(e))
        return f"Gemini API Error: {error_msg}. Please verify your network connection and quota."
    except Exception as e:
        return f"Error: Unable to generate tutor response: {str(e)}"


def explain_differently(
    topic: str,
    question: str,
    previous_explanation: str,
    client: Optional[genai.Client] = None,
) -> str:
    """
    Explain a concept differently using a fresh approach, analogies, and alternative examples
    when a student does not understand the previous explanation.

    Args:
        topic: The academic topic or subject.
        question: The student's question or the concept to re-explain.
        previous_explanation: The explanation that the student found difficult or confusing.
        client: Optional pre-configured genai.Client instance.

    Returns:
        str: A newly framed explanation using different analogies and perspectives.
    """
    clean_topic = (topic or "").strip()
    clean_question = (question or "").strip()
    clean_previous = (previous_explanation or "").strip()

    if not clean_question and not clean_topic:
        return "Please specify the topic or question you would like me to explain differently."

    try:
        gemini_client = client or get_gemini_client()
    except ValueError as e:
        return f"Configuration Error: {str(e)}"
    except Exception as e:
        return f"Initialization Error: Unable to configure Gemini client: {str(e)}"

    prompt = (
        f"Topic: {clean_topic or 'General'}\n"
        f"Student Question: {clean_question}\n\n"
        "The student did not understand the following previous explanation:\n"
        f"\"\"\"\n{clean_previous}\n\"\"\"\n\n"
        "Task:\n"
        "1. Explain the same concept using a completely DIFFERENT mental model, approach, or perspective.\n"
        "2. Use fresh, intuitive analogies and alternative real-world examples.\n"
        "3. Do NOT repeat the explanations, structure, or analogies from the previous explanation above.\n"
        "4. Focus on making the tricky concept simple, intuitive, and crystal clear."
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=EXPLAIN_DIFFERENTLY_SYSTEM_PROMPT,
        temperature=0.8,
    )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        if not response or not response.text or not response.text.strip():
            return (
                "I apologize, but I was unable to generate an alternative explanation at this time. "
                "Please try specifying which part was confusing."
            )

        return response.text.strip()

    except APIError as e:
        error_msg = getattr(e, "message", str(e))
        return f"Gemini API Error: {error_msg}. Please check your connection or quota."
    except Exception as e:
        return f"Error: Unable to generate alternative explanation: {str(e)}"
