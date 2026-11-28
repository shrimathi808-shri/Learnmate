# 📝 LearnMate – AI Quiz Generation & Answer Evaluation Module

**Author:** Selvarani (CSE Student)  
**Module Files:** `quiz.py`, `quiz_ui.py`, `test_quiz.py`

---

## 1. Overview

This module adds AI-powered quiz generation, automated answer evaluation (for both MCQs and conceptual Short Answer questions), score calculation, and student-friendly feedback to **LearnMate**.

### Key Features
- **Reuses `tutor.get_gemini_client()`** — no duplicate API key setup or separate client initialization.
- **Customizable Quiz Generation (`generate_quiz`)**:
  - Any topic or subject
  - Difficulty levels: `Beginner`, `Intermediate`, `Advanced`
  - Question types: `MCQ`, `Short Answer`, `Mixed`
  - Configurable number of questions (`1`–`20`)
- **Smart Answer Evaluation**:
  - `evaluate_mcq()` checks selected options deterministically and explains why the right option is correct.
  - `evaluate_short_answer()` uses Gemini to grade **conceptual understanding**, award **partial marks**, highlight **missing concepts / mistakes**, and reject incorrect or empty answers fairly.
- **Accurate Scoring (`calculate_score`)**:
  - Tracks total questions, correct, partially correct, incorrect counts, marks obtained, max possible marks, percentage, and performance category:
    - `90–100%`: **Excellent**
    - `75–89%`: **Very Good**
    - `50–74%`: **Good**
    - `Below 50%`: **Needs Improvement**
- **Interactive Streamlit UI (`quiz_ui.py`)**:
  - One question at a time with Previous/Next navigation
  - Preserves answers across navigation
  - Prevents accidental duplicate submissions
  - Shows detailed per-question feedback, final score summary, and a **Retake Quiz** button

---

## 2. Teammate Integration Guide

### For Zeba (Frontend – `app.py`)
Option A — **Embed the ready-made interactive quiz UI** directly inside a Streamlit tab or page:

```python
import streamlit as st
from quiz_ui import render_quiz_ui

st.header("Practice Quiz")
quiz_result = render_quiz_ui(
    default_topic=st.session_state.get("selected_topic", ""),
    default_difficulty=st.session_state.get("selected_level", "Beginner"),
)

# Optional: save quiz_result once the student submits
if quiz_result:
    summary = quiz_result["summary"]
    st.toast(f"Quiz saved! Score: {summary['percentage']}%")
```

Option B — **Call `quiz.py` functions directly** if building custom UI components:

```python
from quiz import generate_quiz, evaluate_quiz, QuizError

try:
    quiz = generate_quiz(
        topic="Photosynthesis",
        difficulty="Beginner",
        num_questions=4,
        question_type="Mixed",
    )
    result = evaluate_quiz(quiz, student_answers={1: "B", 2: "A", 3: "Plants make glucose using sunlight.", 4: "..."})
    print(result["summary"])
except QuizError as exc:
    st.error(str(exc))
```

### For Deepshika (Database – `database.py`)
When `evaluate_quiz()` completes (or when `render_quiz_ui()` returns a non-`None` `quiz_result`), `quiz_result["summary"]` provides all fields ready to insert into SQLite:

```python
summary = quiz_result["summary"]
# Available fields:
# quiz_result["topic"]
# quiz_result["difficulty"]
# summary["total_questions"]
# summary["correct_answers"]
# summary["partially_correct_answers"]
# summary["incorrect_answers"]
# summary["total_marks_obtained"]
# summary["max_possible_marks"]
# summary["percentage"]
# summary["performance_level"]
```

---

## 3. Running & Testing on Windows (PowerShell)

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run all offline unit tests (tutor + quiz)
.\venv\Scripts\python.exe -m pytest test_quiz.py test_tutor.py -v

# 3. Launch the standalone interactive Quiz UI
.\venv\Scripts\python.exe -m streamlit run quiz_ui.py
```

*(Note: Ensure `.env` has your `GEMINI_API_KEY` for live AI quiz generation, or use the **"Load Sample Quiz"** button in `quiz_ui.py` for offline demos.)*
