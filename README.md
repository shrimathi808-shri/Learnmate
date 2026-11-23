# 🎓 LearnMate – AI-Powered Personal Tutor

> **HOMINGO HACK DAYS Hackathon Project**  
> An intelligent, multi-level personal AI tutoring system designed to make complex academic concepts simple, engaging, and accessible for students at every stage of learning.

---

## 👥 Team & Responsibilities

| Member | Role | Assigned Module | File |
| :--- | :--- | :--- | :--- |
| **Shrimathi** | **Team Lead** | LLM & Gemini API Integration | `tutor.py` |
| **Zeba** | Member | Streamlit Frontend & UI | `app.py` |
| **Selvarani** | Member | AI Quiz Generation & Evaluation | `quiz.py` |
| **Deepshika** | Member | SQLite Database & Progress Tracking | `database.py` |

---

## 🚀 Shrimathi's Module: AI Tutor & Gemini Integration (`tutor.py`)

This module provides the core intelligence for LearnMate, connecting Python with Google's modern Gemini models via the official `google-genai` SDK.

### Key Capabilities:
1. **Three Adaptive Learning Modes**:
   - 🟢 **Beginner**: Step-by-step explanations, everyday real-world analogies, simple English, and zero unnecessary jargon.
   - 🟡 **Intermediate**: Structured, detailed explanations balancing core principles, technical terminology, and practical examples.
   - 🔴 **Advanced**: Rigorous academic analysis, formal definitions, underlying mechanisms, trade-offs, and complex real-world use cases.
2. **Context-Aware Follow-Up Chat**:
   - Multi-turn conversation history support for natural question-and-answer tutoring sessions.
3. **"Explain Differently" Engine**:
   - When a student doesn't grasp an initial explanation, generates an alternative perspective using fresh analogies and mental models without repeating previous phrasing.
4. **Resilient Error Handling**:
   - Graceful handling of missing API keys, network hiccups, rate limits, and empty responses without crashing teammate applications.

---

## 📂 Project Structure

```text
LearnMate/
├── tutor.py           # Core AI Tutor logic & Gemini API integration
├── test_tutor.py      # Unit tests (15 offline tests with mocked Gemini responses)
├── requirements.txt   # Python project dependencies
├── .env.example       # Template for environment variables
├── .gitignore         # Ignores .env, virtualenvs, and python caches
└── README.md          # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shrimathi808-shri/LearnMate.git
cd LearnMate
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key
1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and insert your Gemini API key from [Google AI Studio](https://aistudio.google.com/):
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

---

## 🧪 Running Unit Tests

Run the complete test suite completely offline (no API key required):

```bash
# Using pytest
pytest test_tutor.py -v

# Or using standard unittest
python test_tutor.py -v
```

All 15 tests verify:
- Beginner, Intermediate, and Advanced learning personas
- Multi-turn conversation history formatting
- "Explain Differently" prompt construction & alternative explanations
- Missing API key detection & graceful error messages
- Quota/API error recovery and empty response handling

---

## 💡 Quick Code Example

```python
from tutor import ask_tutor, explain_differently

# 1. Ask a question tailored to student's level
explanation = ask_tutor(
    topic="Photosynthesis",
    level="Beginner",
    question="How do plants make their food?"
)
print("Tutor Explanation:\n", explanation)

# 2. Re-explain with a fresh approach if confusing
alt_explanation = explain_differently(
    topic="Photosynthesis",
    question="How do plants make their food?",
    previous_explanation=explanation
)
print("\nAlternative Explanation:\n", alt_explanation)
```

---

## 🤝 Teammate Integration Guide

### For Zeba (Frontend – `app.py`)
Import the tutor functions directly into your Streamlit interface:
```python
from tutor import ask_tutor, explain_differently

# In your chat loop:
response = ask_tutor(
    topic=selected_topic,
    level=selected_level,
    question=user_input,
    history=st.session_state.messages
)
```

### For Selvarani (Quiz Generator – `quiz.py`)
Reuse the pre-configured Gemini client helper:
```python
from tutor import get_gemini_client

client = get_gemini_client()
```

### For Deepshika (Database – `database.py`)
Log tutoring sessions into SQLite:
```python
# Save topic, level, question, and response to student history table
```

---

## 📄 License
Created for HOMINGO HACK DAYS.
