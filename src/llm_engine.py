import os
import re
import time
import json
import random
import pandas as pd

MODEL = "llama-3.3-70b-versatile"

# ── Groq Client Setup ──────────────────────────────────────────────────────
try:
    from groq import Groq

    GROQ_API_KEY = "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"
    client = Groq(api_key=GROQ_API_KEY)
    GROQ_AVAILABLE = True

except Exception as e:
    client = None
    GROQ_AVAILABLE = False
    print(f"[llm_engine] Groq init failed: {e}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
questions_path = os.path.join(BASE_DIR, "data", "Software Questions.csv")


# ── Core LLM call ──────────────────────────────────────────────────────────
def ask_groq(prompt: str) -> str:
    if not GROQ_AVAILABLE or client is None:
        raise RuntimeError("Groq client is not available.")

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict but helpful mock interview assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=900,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if "429" in str(e):
                time.sleep((attempt + 1) * 3)
                continue
            raise RuntimeError(str(e))

    raise RuntimeError("Groq API failed after 3 retries.")


# ── Load questions from CSV ────────────────────────────────────────────────
def load_questions(category: str = None) -> list:
    try:
        df = pd.read_csv(questions_path, encoding="latin1")
    except Exception:
        return default_questions(category or "General", 5)

    q_col = None
    for col in df.columns:
        if "question" in col.lower():
            q_col = col
            break
    if q_col is None:
        q_col = df.columns[0]

    questions = df[q_col].dropna().astype(str).tolist()

    if category:
        for col in df.columns:
            cl = col.lower()
            if "category" in cl or "topic" in cl or "role" in cl:
                mask = df[col].astype(str).str.lower().str.contains(category.lower(), na=False)
                filtered = df.loc[mask, q_col].dropna().astype(str).tolist()
                if filtered:
                    questions = filtered
                break

    return [q.strip() for q in questions if len(q.strip()) > 8]


# ── Fallback question bank ─────────────────────────────────────────────────
def default_questions(category: str, num: int = 5) -> list:
    bank = {
        "Data Science": [
            "What is overfitting in machine learning?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain bias and variance trade-off.",
            "What is feature engineering?",
            "What is cross-validation and why is it used?",
            "Explain the difference between bagging and boosting.",
            "What is a confusion matrix?",
            "How do you handle missing data in a dataset?",
            "What is the purpose of regularization?",
            "Explain the difference between classification and regression."
        ],
        "Python Developer": [
            "What are Python decorators and how do they work?",
            "What is the difference between a list and a tuple?",
            "Explain *args and **kwargs in Python.",
            "What is exception handling in Python?",
            "What are generators in Python?",
            "What is the difference between deepcopy and shallow copy?",
            "How does Python manage memory?",
            "What are Python's built-in data structures?",
            "What is a lambda function?",
            "Explain the GIL in Python."
        ],
        "Web Designing": [
            "What is responsive design?",
            "What is the CSS box model?",
            "What is the difference between HTML, CSS, and JavaScript?",
            "What is semantic HTML?",
            "How does JavaScript interact with the DOM?",
            "What is Flexbox and when would you use it?",
            "What is the difference between inline and block elements?",
            "What are CSS media queries?",
            "What is the difference between GET and POST requests?",
            "What is CORS and why does it matter?"
        ],
        "DevOps Engineer": [
            "What is Docker and how does it work?",
            "What is CI/CD and why is it important?",
            "What is Kubernetes and what problems does it solve?",
            "What is infrastructure as code?",
            "What is the difference between Git pull and Git fetch?",
            "How do you monitor a production system?",
            "What is a load balancer?",
            "Explain blue-green deployment.",
            "What is Ansible and how is it used?",
            "What is the difference between a VM and a container?"
        ],
        "Java Developer": [
            "What is OOP and what are its four pillars?",
            "What is inheritance in Java?",
            "What is the difference between JDK, JRE, and JVM?",
            "What is exception handling in Java?",
            "What is method overriding vs method overloading?",
            "What is the difference between an interface and an abstract class?",
            "What are Java Streams?",
            "What is the Collections framework in Java?",
            "What is multithreading in Java?",
            "What is the difference between HashMap and HashTable?"
        ],
        "Database": [
            "What is normalization and why is it important?",
            "What is indexing in databases?",
            "What is a primary key vs a foreign key?",
            "What is the difference between SQL and NoSQL?",
            "What is a JOIN in SQL? Explain types.",
            "What is a stored procedure?",
            "What is ACID in databases?",
            "What is the difference between WHERE and HAVING?",
            "What is database sharding?",
            "What is a transaction in SQL?"
        ],
        "General": [
            "Tell me about yourself.",
            "What are your strengths and weaknesses?",
            "Describe a challenging project you worked on.",
            "Why do you want this role?",
            "What technologies are you most comfortable with?",
            "Where do you see yourself in 5 years?",
            "How do you handle tight deadlines?",
            "Describe a time you worked in a team.",
            "What motivates you?",
            "How do you keep your technical skills up to date?"
        ]
    }
    pool = bank.get(category, bank["General"])
    return pool[:num]


# ── Generate questions via LLM ─────────────────────────────────────────────
def generate_questions(category: str, skills: list, num: int = 5) -> list:
    prompt = f"""You are a technical interviewer conducting an interview for a {category} role.

Candidate's skills: {', '.join(skills[:10]) if skills else 'Not specified'}

Generate exactly {num} interview questions tailored to this candidate.

Rules:
- One question per line
- Number each question: 1. 2. 3. etc.
- Each must be a complete, meaningful interview question
- Mix technical and conceptual questions
- No explanations, no headings, no code blocks
- Questions must end with a question mark
"""

    try:
        raw = ask_groq(prompt)
        questions = []

        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering like "1." or "1)"
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if (
                len(cleaned) > 10
                and (
                    "?" in cleaned
                    or cleaned.lower().startswith(
                        ("tell", "describe", "explain", "what", "how",
                         "why", "can you", "have you", "do you")
                    )
                )
            ):
                questions.append(cleaned)

        if len(questions) >= num:
            return questions[:num]

        # If LLM returned fewer than needed, pad with defaults
        if questions:
            extras = default_questions(category, num)
            for q in extras:
                if q not in questions:
                    questions.append(q)
                if len(questions) >= num:
                    break
            return questions[:num]

    except Exception as e:
        print(f"[generate_questions] LLM failed: {e}")

    # Full fallback to CSV then default bank
    fallback = load_questions(category)
    if len(fallback) >= num:
        return random.sample(fallback, num)
    if fallback:
        return fallback[:num]

    return default_questions(category, num)


# ── Evaluate a single answer ───────────────────────────────────────────────
def evaluate_answer(question: str, user_answer: str) -> dict:
    prompt = f"""You are a strict mock interviewer evaluating a candidate's answer.

Question: {question}
Candidate Answer: {user_answer}

Evaluate the answer and return ONLY valid JSON with no extra text:
{{
  "score": 7,
  "strengths": "one concise sentence about what was good",
  "weaknesses": "one concise sentence about what was lacking",
  "improved_answer": "a better 2-3 sentence model answer",
  "tip": "one actionable tip to improve future answers"
}}
"""

    fallback = {
        "score": 6,
        "strengths": "Your answer addresses the question directly.",
        "weaknesses": "It needs more structure and a concrete example.",
        "improved_answer": (
            user_answer if user_answer.strip()
            else "Provide a clear definition, a brief explanation, and one real-world example."
        ),
        "tip": "Use the structure: Definition → Explanation → Example."
    }

    try:
        raw = ask_groq(prompt)
        raw = raw.strip()

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "score":           int(data.get("score", 6)),
                "strengths":       str(data.get("strengths",       fallback["strengths"])),
                "weaknesses":      str(data.get("weaknesses",      fallback["weaknesses"])),
                "improved_answer": str(data.get("improved_answer", fallback["improved_answer"])),
                "tip":             str(data.get("tip",             fallback["tip"]))
            }
    except Exception as e:
        print(f"[evaluate_answer] LLM failed: {e}")

    # Score by length if LLM unavailable
    words = len(user_answer.split())
    if words < 8:
        fallback["score"] = 3
        fallback["weaknesses"] = "Your answer is far too short."
        fallback["tip"] = "Write at least 3-4 sentences with a definition and example."
    elif words < 20:
        fallback["score"] = 5
    else:
        fallback["score"] = 7

    return fallback


# ── Resume improvement suggestions ────────────────────────────────────────
def get_resume_suggestions(skills: list, missing_skills: list, category: str) -> str:
    prompt = f"""You are an expert career counselor helping a candidate improve their resume.

Target role: {category}
Skills they have: {', '.join(skills) if skills else 'None listed'}
Skills they are missing: {', '.join(missing_skills) if missing_skills else 'None'}

Give exactly 5 short, specific, actionable suggestions to improve their profile for this role.
Number each suggestion. Be direct and practical.
"""

    try:
        return ask_groq(prompt)
    except Exception:
        tips = []
        if missing_skills:
            tips.append(f"1. Learn and build projects using: {', '.join(missing_skills[:3])}.")
        else:
            tips.append("1. Strengthen your existing project descriptions with measurable outcomes.")
        tips.append("2. Tailor your resume summary to match the target role's keywords.")
        tips.append("3. Add GitHub links or live demos to your projects.")
        tips.append("4. Include certifications relevant to the role.")
        tips.append("5. Quantify your achievements (e.g. 'improved performance by 30%').")
        return "\n".join(tips)


# ── Interview summary ──────────────────────────────────────────────────────
def get_interview_summary(qa_pairs: list, category: str) -> dict:
    qa_text = ""
    scores  = []

    for i, qa in enumerate(qa_pairs, 1):
        score_val = qa.get("score", 0)
        try:
            scores.append(float(str(score_val).replace("/10", "").strip()))
        except Exception:
            pass
        qa_text += f"Q{i}: {qa.get('question', '')}\nA{i}: {qa.get('answer', '')}\nScore: {qa.get('score', '')}\n\n"

    avg = round(sum(scores) / len(scores), 1) if scores else 6.0

    prompt = f"""You are reviewing a completed mock interview for a {category} role.

{qa_text}

Return ONLY valid JSON with no extra text:
{{
  "overall_score": {avg},
  "overall_feedback": "two sentences summarising overall performance",
  "top_strength": "one sentence about the candidate's best quality",
  "top_weakness": "one sentence about the main area to improve",
  "recommendation": "one specific action the candidate should take"
}}
"""

    fallback = {
        "overall_score":    avg,
        "overall_feedback": "You showed a reasonable understanding of the topics. Focus on adding more depth and concrete examples.",
        "top_strength":     "You attempted all questions with relevant answers.",
        "top_weakness":     "Your answers need better structure and more specific examples.",
        "recommendation":   "Practice the STAR method: Situation, Task, Action, Result."
    }

    try:
        raw = ask_groq(prompt)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "overall_score":    data.get("overall_score",    fallback["overall_score"]),
                "overall_feedback": data.get("overall_feedback", fallback["overall_feedback"]),
                "top_strength":     data.get("top_strength",     fallback["top_strength"]),
                "top_weakness":     data.get("top_weakness",     fallback["top_weakness"]),
                "recommendation":   data.get("recommendation",   fallback["recommendation"]),
            }
    except Exception as e:
        print(f"[get_interview_summary] LLM failed: {e}")

    return fallback