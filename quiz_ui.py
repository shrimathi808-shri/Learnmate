"""
quiz_ui.py - LearnMate Interactive Streamlit Quiz Interface
-----------------------------------------------------------
Author: Selvarani (CSE Student – Quiz Generation & Answer Evaluation)
Project: LearnMate – AI-Powered Personal Tutor

This module provides a clean, beginner-friendly Streamlit UI wrapper around `quiz.py`.
- Can be run standalone: `streamlit run quiz_ui.py`
- Can be embedded inside Zeba's main `app.py` via `render_quiz_ui()`
- Keeps all quiz business logic decoupled inside `quiz.py`
"""

from typing import Any, Dict, Optional

import streamlit as st

from quiz import (
    QuizError,
    VALID_DIFFICULTIES,
    VALID_QUESTION_TYPES,
    evaluate_quiz,
    generate_quiz,
    get_sample_quiz,
)

STATE_QUIZ = "lm_quiz_data"
STATE_ANSWERS = "lm_quiz_answers"
STATE_INDEX = "lm_quiz_index"
STATE_SUBMITTED = "lm_quiz_submitted"
STATE_RESULT = "lm_quiz_result"


def _init_quiz_state() -> None:
    """Initialize Streamlit session_state keys used by the quiz module."""
    st.session_state.setdefault(STATE_QUIZ, None)
    st.session_state.setdefault(STATE_ANSWERS, {})
    st.session_state.setdefault(STATE_INDEX, 0)
    st.session_state.setdefault(STATE_SUBMITTED, False)
    st.session_state.setdefault(STATE_RESULT, None)


def _clear_question_widget_keys(quiz: Optional[Dict[str, Any]]) -> None:
    """Remove per-question widget keys from session_state so fresh widgets render cleanly."""
    if not quiz:
        return
    for q in quiz.get("questions", []):
        q_id = q.get("id")
        for prefix in ("lm_widget_mcq_", "lm_widget_sa_"):
            st.session_state.pop(f"{prefix}{q_id}", None)


def _reset_attempt(keep_quiz: bool = True) -> None:
    """Reset student answers and submission status (either for Retake or New Quiz)."""
    current_quiz = st.session_state.get(STATE_QUIZ)
    _clear_question_widget_keys(current_quiz)
    st.session_state[STATE_ANSWERS] = {}
    st.session_state[STATE_INDEX] = 0
    st.session_state[STATE_SUBMITTED] = False
    st.session_state[STATE_RESULT] = None
    if not keep_quiz:
        st.session_state[STATE_QUIZ] = None


def _sync_widget_answer(q_id: Any, widget_key: str) -> None:
    """Persist the current widget value into the canonical answers dictionary."""
    value = st.session_state.get(widget_key)
    st.session_state[STATE_ANSWERS][q_id] = "" if value is None else value


def _count_answered(quiz: Dict[str, Any], answers: Dict[Any, Any]) -> int:
    """Count how many questions currently have a non-empty student answer."""
    count = 0
    for q in quiz.get("questions", []):
        val = answers.get(q.get("id"), "")
        if isinstance(val, str) and val.strip():
            count += 1
    return count


def _render_setup_screen(default_topic: str = "", default_difficulty: str = "Beginner") -> None:
    """Render Step 1–5: Quiz configuration form and generation trigger."""
    st.subheader("Create Your Practice Quiz")
    st.write("Choose a topic and preferences below, and LearnMate will generate an interactive quiz.")

    diff_index = (
        VALID_DIFFICULTIES.index(default_difficulty)
        if default_difficulty in VALID_DIFFICULTIES
        else 0
    )

    with st.form("learnmate_quiz_setup_form"):
        topic = st.text_input(
            "Topic or Subject",
            value=default_topic,
            placeholder="e.g., Photosynthesis, Binary Search, Python Lists, Thermodynamics",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            difficulty = st.selectbox("Difficulty Level", options=VALID_DIFFICULTIES, index=diff_index)
        with col2:
            question_type = st.selectbox("Question Type", options=VALID_QUESTION_TYPES, index=2)
        with col3:
            num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)

        generate_clicked = st.form_submit_button("Generate Quiz", type="primary", use_container_width=True)

    if generate_clicked:
        if not topic.strip():
            st.warning("Please enter a topic before generating a quiz.")
        else:
            with st.spinner("Generating your personalized quiz with Gemini AI..."):
                try:
                    quiz_data = generate_quiz(
                        topic=topic,
                        difficulty=difficulty,
                        num_questions=num_questions,
                        question_type=question_type,
                    )
                    _reset_attempt(keep_quiz=False)
                    st.session_state[STATE_QUIZ] = quiz_data
                    st.rerun()
                except QuizError as exc:
                    st.error(f"Could not generate quiz: {exc}")

    with st.expander("No API key right now? Try an offline sample quiz"):
        st.caption("Useful for testing the interactive quiz workflow and UI without calling Gemini.")
        if st.button("Load Sample Quiz (Python & Data Structures)"):
            _reset_attempt(keep_quiz=False)
            st.session_state[STATE_QUIZ] = get_sample_quiz()
            st.rerun()


def _render_question_screen(quiz: Dict[str, Any]) -> None:
    """Render Step 6–7: Question-by-question interface with answer preservation & Submit."""
    questions = quiz["questions"]
    total = len(questions)
    current_idx = max(0, min(total - 1, st.session_state[STATE_INDEX]))
    st.session_state[STATE_INDEX] = current_idx
    question = questions[current_idx]
    q_id = question["id"]
    answers = st.session_state[STATE_ANSWERS]

    st.subheader(f"Quiz: {quiz['topic']}")
    st.caption(
        f"Difficulty: **{quiz['difficulty']}** | Type: **{quiz['question_type']}** | "
        f"Total Marks: **{quiz['total_max_marks']:g}**"
    )

    answered_count = _count_answered(quiz, answers)
    st.progress((current_idx + 1) / total, text=f"Question {current_idx + 1} of {total}  •  Answered: {answered_count}/{total}")

    with st.container(border=True):
        q_type_label = "Multiple Choice" if question["type"] == "mcq" else "Short Answer"
        st.markdown(f"**Q{current_idx + 1}. ({q_type_label} — {question['max_marks']:g} mark{'s' if question['max_marks'] != 1 else ''})**")
        st.markdown(f"#### {question['question']}")

        if question["type"] == "mcq":
            widget_key = f"lm_widget_mcq_{q_id}"
            options = question["options"]
            option_keys = ["A", "B", "C", "D"]
            if widget_key not in st.session_state:
                saved = answers.get(q_id)
                st.session_state[widget_key] = saved if saved in option_keys else None

            st.radio(
                "Select one option:",
                options=option_keys,
                format_func=lambda opt: f"{opt}) {options[opt]}",
                key=widget_key,
                on_change=_sync_widget_answer,
                args=(q_id, widget_key),
            )
        else:
            widget_key = f"lm_widget_sa_{q_id}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = str(answers.get(q_id, ""))

            st.text_area(
                "Write your answer below:",
                height=140,
                placeholder="Explain the concept in your own words...",
                key=widget_key,
                on_change=_sync_widget_answer,
                args=(q_id, widget_key),
            )

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1.5])
    with nav_col1:
        if st.button("Previous", disabled=(current_idx == 0), use_container_width=True):
            st.session_state[STATE_INDEX] = current_idx - 1
            st.rerun()
    with nav_col2:
        if st.button("Next", disabled=(current_idx >= total - 1), use_container_width=True):
            st.session_state[STATE_INDEX] = current_idx + 1
            st.rerun()
    with nav_col3:
        submit_disabled = st.session_state[STATE_SUBMITTED]
        if st.button("Submit Quiz", type="primary", disabled=submit_disabled, use_container_width=True):
            with st.spinner("Evaluating your answers and preparing feedback..."):
                try:
                    result = evaluate_quiz(quiz, st.session_state[STATE_ANSWERS])
                    st.session_state[STATE_RESULT] = result
                    st.session_state[STATE_SUBMITTED] = True
                    st.rerun()
                except QuizError as exc:
                    st.error(
                        f"Evaluation could not be completed: {exc} "
                        "Your answers are still saved—please try submitting again."
                    )

    if answered_count < total:
        st.caption(f"Tip: You still have {total - answered_count} unanswered question(s). Use Previous/Next to review before submitting.")

    if st.button("Cancel & Start Over"):
        _reset_attempt(keep_quiz=False)
        st.rerun()


def _render_results_screen(result: Dict[str, Any]) -> None:
    """Render Step 8–10: Score summary, detailed per-question feedback, and Retake option."""
    summary = result["summary"]
    evaluations = result["evaluations"]

    st.subheader(f"Quiz Results: {result.get('topic', '')}")

    level = summary["performance_level"]
    banner_msg = (
        f"Performance Level: **{level}** — You scored "
        f"**{summary['total_marks_obtained']:g} / {summary['max_possible_marks']:g}** "
        f"(**{summary['percentage']:g}%**)"
    )
    if level == "Excellent":
        st.success(banner_msg)
    elif level == "Very Good":
        st.info(banner_msg)
    elif level == "Good":
        st.warning(banner_msg)
    else:
        st.error(banner_msg)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Questions", summary["total_questions"])
    m2.metric("Correct", summary["correct_answers"])
    m3.metric("Partially Correct", summary["partially_correct_answers"])
    m4.metric("Incorrect", summary["incorrect_answers"])

    m5, m6, m7 = st.columns(3)
    m5.metric("Marks Obtained", f"{summary['total_marks_obtained']:g} / {summary['max_possible_marks']:g}")
    m6.metric("Percentage", f"{summary['percentage']:g}%")
    m7.metric("Performance", summary["performance_level"])

    st.divider()
    st.subheader("Detailed Feedback & Explanations")

    for idx, item in enumerate(evaluations, start=1):
        status = item["status"]
        badge = {"correct": "✅ Correct", "partially_correct": "🟡 Partially Correct", "incorrect": "❌ Incorrect"}[status]
        with st.expander(
            f"Q{idx}: {item['question']}  —  {badge} ({item['marks_obtained']:g}/{item['max_marks']:g})",
            expanded=True,
        ):
            student_ans = item.get("student_answer") or "(No answer submitted)"
            st.markdown(f"**Your Answer:** {student_ans}")
            st.markdown(f"**Correct Answer:** {item['correct_answer']}")
            st.markdown(f"**Marks Obtained:** {item['marks_obtained']:g} / {item['max_marks']:g}")
            st.markdown(f"**Explanation:** {item['explanation']}")
            if item["missed"]:
                st.markdown("**What You Missed:**")
                for miss in item["missed"]:
                    st.markdown(f"- {miss}")
            else:
                st.markdown("**What You Missed:** Nothing — great coverage!")
            st.markdown(f"**Suggestions for Improvement:** {item['suggestions']}")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Retake Quiz", type="primary", use_container_width=True):
            _reset_attempt(keep_quiz=True)
            st.rerun()
    with btn_col2:
        if st.button("Create New Quiz", use_container_width=True):
            _reset_attempt(keep_quiz=False)
            st.rerun()


def render_quiz_ui(default_topic: str = "", default_difficulty: str = "Beginner") -> Optional[Dict[str, Any]]:
    """
    Render the complete interactive LearnMate Quiz experience inside Streamlit.

    Designed for direct integration into Zeba's `app.py`:
        from quiz_ui import render_quiz_ui
        quiz_result = render_quiz_ui(default_topic=selected_topic, default_difficulty=selected_level)

    Returns:
        The evaluation result dictionary once the quiz is submitted, or None while in progress.
    """
    _init_quiz_state()

    quiz = st.session_state[STATE_QUIZ]
    submitted = st.session_state[STATE_SUBMITTED]
    result = st.session_state[STATE_RESULT]

    if quiz is None:
        _render_setup_screen(default_topic=default_topic, default_difficulty=default_difficulty)
        return None

    if submitted and result is not None:
        _render_results_screen(result)
        return result

    _render_question_screen(quiz)
    return None


if __name__ == "__main__":
    st.set_page_config(page_title="LearnMate Quiz Module", page_icon="📝", layout="centered")
    st.title("📝 LearnMate Interactive Quiz")
    render_quiz_ui()
