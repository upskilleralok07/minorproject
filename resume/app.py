import streamlit as st
import sys
import os
import random
import sqlite3
import bcrypt
from datetime import datetime

from src.resume_parser import parse_resume, clean_text, SKILLS_LIST
from src.placement_model import predict_placement
from src.llm_engine import (
    evaluate_answer,
    get_resume_suggestions,
    generate_questions,
    get_interview_summary,
    load_questions
)
from src.internship_recommender import InternshipRecommender

# ── App Identity ───────────────────────────────────────────────────────────
APP_NAME = "PlacePilot AI"

# ── Database / Data Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "placepilot.db")
INTERNSHIP_CSV_PATH = os.path.join(BASE_DIR, "data", "structured_internships_madhya_pradesh.csv")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            resume_score REAL,
            interview_score REAL,
            resume_label TEXT,
            target_role TEXT,
            skills_found INTEGER,
            missing_skills INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS internship_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            company_name TEXT,
            internship_role TEXT,
            functional_area TEXT,
            industry TEXT,
            state TEXT,
            distance_km REAL,
            recommendation_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def signup_user(name: str, email: str, password: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name.strip(), email.strip().lower(), hash_password(password))
        )
        conn.commit()
        return True, "Account created successfully. Please login."
    except sqlite3.IntegrityError:
        return False, "This email is already registered."
    finally:
        conn.close()


def login_user(email: str, password: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    user = cur.fetchone()
    conn.close()

    if user and verify_password(password, user["password"]):
        return True, dict(user)
    return False, "Invalid email or password."


def save_resume_result(user_id, placement_result, parsed_resume, target_role="General"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results (
            user_id, resume_score, interview_score, resume_label,
            target_role, skills_found, missing_skills, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        placement_result.get("score", 0),
        None,
        placement_result.get("label", ""),
        target_role,
        len(placement_result.get("skills_found", [])),
        len(placement_result.get("missing_skills", [])),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def save_interview_result(user_id, interview_score, target_role="General"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results (
            user_id, resume_score, interview_score, resume_label,
            target_role, skills_found, missing_skills, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        None,
        interview_score,
        "",
        target_role,
        None,
        None,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def save_internship_recommendations(user_id, recommendations_df):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM internship_recommendations WHERE user_id = ?", (user_id,))

    for _, row in recommendations_df.iterrows():
        cur.execute("""
            INSERT INTO internship_recommendations (
                user_id, company_name, internship_role, functional_area,
                industry, state, distance_km, recommendation_score, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            str(row.get("company_name", "")),
            str(row.get("internship_role", "")),
            str(row.get("functional_area", "")),
            str(row.get("industry", "")),
            str(row.get("state", "")),
            float(row.get("distance_km", 0)),
            float(row.get("recommendation_score", 0)),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    conn.commit()
    conn.close()


def fetch_user_results(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM results
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def fetch_latest_resume_result(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM results
        WHERE user_id = ? AND resume_score IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def fetch_latest_interview_result(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM results
        WHERE user_id = ? AND interview_score IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def fetch_latest_internship_recommendations(user_id, limit=5):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM internship_recommendations
        WHERE user_id = ?
        ORDER BY created_at DESC, recommendation_score DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows


# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚀",
    layout="wide"
)

init_db()

# ── Dark Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }

    .score-big {
        font-size: 64px;
        font-weight: 800;
        text-align: center;
    }

    .skill-tag {
        display: inline-block;
        background-color: #1f6feb;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
    }

    .missing-tag {
        display: inline-block;
        background-color: #b91c1c;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
    }

    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid #21262d;
    }

    .metric-box {
        background-color: #21262d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #30363d;
        min-height: 150px;
    }

    .answer-box {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
    }

    .feedback-good {
        background-color: #0d4429;
        border: 1px solid #238636;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }

    .feedback-bad {
        background-color: #3d1c1c;
        border: 1px solid #b91c1c;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }

    .stButton > button {
        background-color: #1f6feb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #388bfd;
    }

    .stTextArea textarea, .stTextInput input {
        background-color: #21262d !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    .stProgress > div > div {
        background-color: #1f6feb;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────────────────────
defaults = {
    "user": None,
    "parsed_resume": None,
    "placement_result": None,
    "questions": [],
    "current_q": 0,
    "qa_pairs": [],
    "interview_done": False,
    "interview_summary": None,
    "audio_mode": False,
    "internship_recommendations": None,
    "internship_recommender": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.internship_recommender is None:
    try:
        st.session_state.internship_recommender = InternshipRecommender(INTERNSHIP_CSV_PATH)
    except Exception:
        st.session_state.internship_recommender = None


def logout():
    for key in list(defaults.keys()):
        st.session_state[key] = defaults[key]
    st.rerun()


# ── Auth Screen ────────────────────────────────────────────────────────────
if st.session_state.user is None:
    st.markdown(f"""
    <div style='text-align:center; padding: 40px 0 10px 0;'>
        <h1 style='font-size:48px; font-weight:800; color:#58a6ff;'>
            🚀 {APP_NAME}
        </h1>
        <p style='font-size:18px; color:#8b949e;'>
            Resume analysis · job matching · mock interviews · AI guidance
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Signup"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            success, result = login_user(login_email, login_password)
            if success:
                st.session_state.user = result
                st.success("Login successful.")
                st.rerun()
            else:
                st.error(result)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        signup_name = st.text_input("Full Name", key="signup_name")
        signup_email = st.text_input("Email Address", key="signup_email")
        signup_password = st.text_input("Create Password", type="password", key="signup_password")

        if st.button("Create Account"):
            if not signup_name.strip() or not signup_email.strip() or not signup_password.strip():
                st.warning("Please fill all fields.")
            elif len(signup_password) < 6:
                st.warning("Password must be at least 6 characters.")
            else:
                success, msg = signup_user(signup_name, signup_email, signup_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ── Sidebar Navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🚀 {APP_NAME}")
    st.caption(f"Welcome, {st.session_state.user['name']}")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🏠 Home",
            "📄 Resume Analyzer",
            "💼 Job Match Score",
            "🎤 Mock Interview",
            "🎯 Internship Recommendation",
            "💡 AI Suggestions",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if st.session_state.parsed_resume:
        st.markdown("### ✅ Resume Loaded")
        skills = st.session_state.parsed_resume.get("skills", [])
        st.markdown(f"**Skills found:** {len(skills)}")
        if st.session_state.placement_result:
            score = st.session_state.placement_result.get("score", 0)
            st.markdown(f"**Placement Score:** {score}%")
    else:
        st.markdown("### ⚠️ No Resume Loaded")
        st.caption("Go to Resume Analyzer first")

    st.markdown("---")
    if st.button("Logout"):
        logout()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("<div class='section-header'>📊 Your Dashboard</div>", unsafe_allow_html=True)

    user_id = st.session_state.user["id"]
    latest_resume = fetch_latest_resume_result(user_id)
    latest_interview = fetch_latest_interview_result(user_id)
    latest_recommendations = fetch_latest_internship_recommendations(user_id, limit=1)
    all_results = fetch_user_results(user_id)

    resume_score = int(latest_resume["resume_score"]) if latest_resume and latest_resume["resume_score"] is not None else 0
    interview_score = latest_interview["interview_score"] if latest_interview and latest_interview["interview_score"] is not None else "N/A"
    target_role = latest_resume["target_role"] if latest_resume and latest_resume["target_role"] else "General"
    missing_skills = latest_resume["missing_skills"] if latest_resume and latest_resume["missing_skills"] is not None else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <div style='font-size:14px; color:#8b949e;'>Resume Score</div>
            <div style='font-size:42px; font-weight:800; color:#58a6ff;'>{resume_score}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <div style='font-size:14px; color:#8b949e;'>Interview Score</div>
            <div style='font-size:42px; font-weight:800; color:#58a6ff;'>{interview_score}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <div style='font-size:14px; color:#8b949e;'>Target Role</div>
            <div style='font-size:28px; font-weight:800; color:#58a6ff; margin-top:18px;'>{target_role}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <div style='font-size:14px; color:#8b949e;'>Missing Skills</div>
            <div style='font-size:42px; font-weight:800; color:#b91c1c;'>{missing_skills}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>Latest Resume Result</h3>", unsafe_allow_html=True)
        if latest_resume:
            st.write(f"**Score:** {int(latest_resume['resume_score'])}%")
            st.write(f"**Label:** {latest_resume['resume_label']}")
            st.write(f"**Skills Found:** {latest_resume['skills_found']}")
            st.write(f"**Missing Skills:** {latest_resume['missing_skills']}")
            st.write(f"**Created At:** {latest_resume['created_at']}")
        else:
            st.info("No resume analysis found yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>Latest Interview Result</h3>", unsafe_allow_html=True)
        if latest_interview:
            st.write(f"**Interview Score:** {latest_interview['interview_score']}/10")
            st.write(f"**Role:** {latest_interview['target_role']}")
            st.write(f"**Created At:** {latest_interview['created_at']}")
        else:
            st.info("No interview attempt found yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Recent Activity</h3>", unsafe_allow_html=True)
    if all_results:
        for row in all_results[:10]:
            resume_text = f"Resume: {int(row['resume_score'])}%" if row["resume_score"] is not None else "Resume: —"
            interview_text = f"Interview: {row['interview_score']}/10" if row["interview_score"] is not None else "Interview: —"
            st.write(f"- {row['created_at']} | {resume_text} | {interview_text} | Role: {row['target_role']}")
    else:
        st.info("No activity yet. Start by analyzing your resume.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Latest Internship Recommendation</h3>", unsafe_allow_html=True)
    if latest_recommendations:
        row = latest_recommendations[0]
        st.write(f"**Company:** {row['company_name']}")
        st.write(f"**Role:** {row['internship_role']}")
        st.write(f"**Industry:** {row['industry']}")
        st.write(f"**Distance:** {row['distance_km']} km")
        st.write(f"**Recommendation Score:** {round(float(row['recommendation_score']), 4)}")
    else:
        st.info("No internship recommendations found yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HOME
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Home":
    st.markdown(f"""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <h1 style='font-size:48px; font-weight:800; color:#58a6ff;'>
            🚀 {APP_NAME}
        </h1>
        <p style='font-size:18px; color:#8b949e;'>
            Analyze your resume · Match jobs · Practice interviews · Get AI tips
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='card' style='text-align:center;'>
            <div style='font-size:36px;'>📄</div>
            <h3 style='color:#58a6ff;'>Resume Analyzer</h3>
            <p style='color:#8b949e; font-size:14px;'>
                Paste your resume and get an instant placement score
                with skill breakdown.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card' style='text-align:center;'>
            <div style='font-size:36px;'>💼</div>
            <h3 style='color:#58a6ff;'>Job Match Score</h3>
            <p style='color:#8b949e; font-size:14px;'>
                See how well your resume matches
                different job categories.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card' style='text-align:center;'>
            <div style='font-size:36px;'>🎤</div>
            <h3 style='color:#58a6ff;'>Mock Interview</h3>
            <p style='color:#8b949e; font-size:14px;'>
                Practice with AI-generated questions
                and get instant feedback.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='card' style='text-align:center;'>
            <div style='font-size:36px;'>💡</div>
            <h3 style='color:#58a6ff;'>AI Suggestions</h3>
            <p style='color:#8b949e; font-size:14px;'>
                Get personalized tips to improve
                your resume and skills.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Start by going to Resume Analyzer in the sidebar!")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESUME ANALYZER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📄 Resume Analyzer":
    st.markdown("<div class='section-header'>📄 Resume Analyzer</div>", unsafe_allow_html=True)

    resume_text = st.text_area(
        "Paste your resume text here",
        height=300,
        placeholder="Paste your full resume content here..."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("🔍 Analyze Resume")

    if analyze_btn:
        if not resume_text.strip():
            st.warning("Please paste your resume text first!")
        else:
            with st.spinner("Analyzing your resume..."):
                parsed = parse_resume(resume_text)
                result = predict_placement(parsed)
                st.session_state.parsed_resume = parsed
                st.session_state.placement_result = result
                st.session_state.internship_recommendations = None

                save_resume_result(
                    user_id=st.session_state.user["id"],
                    placement_result=result,
                    parsed_resume=parsed,
                    target_role=parsed.get("category", "General")
                )

            st.success("✅ Resume analyzed successfully!")

    if st.session_state.placement_result:
        result = st.session_state.placement_result
        parsed = st.session_state.parsed_resume

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        score = result["score"]
        color = "#238636" if score >= 60 else "#b91c1c"

        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size:14px; color:#8b949e;'>Placement Score</div>
                <div style='font-size:52px; font-weight:800; color:{color};'>{score}%</div>
                <div style='font-size:16px;'>{result['label']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size:14px; color:#8b949e;'>Skills Found</div>
                <div style='font-size:52px; font-weight:800; color:#58a6ff;'>{parsed['num_skills']}</div>
                <div style='font-size:16px; color:#8b949e;'>out of {len(SKILLS_LIST)} tracked</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size:14px; color:#8b949e;'>Resume Length</div>
                <div style='font-size:52px; font-weight:800; color:#58a6ff;'>{parsed['word_count']}</div>
                <div style='font-size:16px; color:#8b949e;'>words</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Skills Found")
            if result["skills_found"]:
                tags = "".join([f"<span class='skill-tag'>{s}</span>" for s in result["skills_found"]])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.warning("No matching skills found")

        with col2:
            st.markdown("#### ❌ Missing Key Skills")
            if result["missing_skills"]:
                tags = "".join([f"<span class='missing-tag'>{s}</span>" for s in result["missing_skills"]])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.success("You have all key skills! 🎉")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Skill Breakdown")
        breakdown = parsed.get("skill_breakdown", {})
        if breakdown:
            for category, skills in breakdown.items():
                st.markdown(f"**{category}**")
                st.progress(min(len(skills) / 5, 1.0))
                tags = "".join([f"<span class='skill-tag'>{s}</span>" for s in skills])
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — JOB MATCH SCORE
# ════════════════════════════════════════════════════════════════════════════
elif page == "💼 Job Match Score":
    st.markdown("<div class='section-header'>💼 Job Match Score</div>", unsafe_allow_html=True)

    if not st.session_state.parsed_resume:
        st.warning("⚠️ Please analyze your resume first in the Resume Analyzer!")
    else:
        parsed = st.session_state.parsed_resume
        skills = set(parsed.get("skills", []))

        JOB_ROLES = {
            "Data Scientist": [
                "python", "machine learning", "deep learning", "pandas",
                "numpy", "sql", "tensorflow", "scikit-learn", "statistics"
            ],
            "Python Developer": [
                "python", "flask", "django", "fastapi", "sql",
                "git", "rest api", "docker"
            ],
            "Web Developer": [
                "html", "css", "javascript", "react", "node",
                "git", "rest api", "sql"
            ],
            "DevOps Engineer": [
                "docker", "kubernetes", "aws", "linux", "git",
                "jenkins", "ansible", "ci/cd"
            ],
            "Data Engineer": [
                "python", "sql", "spark", "hadoop", "kafka",
                "airflow", "etl", "aws"
            ],
            "Java Developer": [
                "java", "spring boot", "sql", "git",
                "rest api", "docker", "maven"
            ],
        }

        st.markdown("### How well does your resume match each role?")
        st.markdown("<br>", unsafe_allow_html=True)

        for role, required in JOB_ROLES.items():
            matched = skills.intersection(set(required))
            score = round(len(matched) / len(required) * 100)
            missing = set(required) - skills
            color = "#238636" if score >= 70 else "#d29922" if score >= 40 else "#b91c1c"

            with st.expander(f"{role}  —  {score}% Match"):
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.markdown(f"""
                    <div style='text-align:center;'>
                        <div style='font-size:42px; font-weight:800; color:{color};'>{score}%</div>
                        <div style='color:#8b949e;'>match</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("**✅ You have:**")
                    if matched:
                        tags = "".join([f"<span class='skill-tag'>{s}</span>" for s in matched])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None matched")

                    st.markdown("<br>**❌ You need:**", unsafe_allow_html=True)
                    if missing:
                        tags = "".join([f"<span class='missing-tag'>{s}</span>" for s in missing])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.success("Perfect match! 🎉")

                st.progress(score / 100)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MOCK INTERVIEW
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎤 Mock Interview":
    st.markdown("<div class='section-header'>🎤 Mock Interview</div>", unsafe_allow_html=True)

    try:
        from audiorecorder import audiorecorder
        import speech_recognition as sr
        from pydub import AudioSegment
        import io
        AUDIO_AVAILABLE = True
    except Exception:
        AUDIO_AVAILABLE = False

    def transcribe_audio(audio_bytes: bytes) -> str:
        if not AUDIO_AVAILABLE:
            return ""

        recognizer = sr.Recognizer()
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            with sr.AudioFile(wav_io) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)

            return recognizer.recognize_google(audio_data)
        except Exception:
            return ""

    if not st.session_state.parsed_resume:
        st.warning("⚠️ Please analyze your resume first in Resume Analyzer!")
    else:
        parsed = st.session_state.parsed_resume
        skills = parsed.get("skills", [])
        category = parsed.get("category", "Software Engineering")

        if not st.session_state.questions:
            col1, col2, col3 = st.columns(3)

            with col1:
                num_q = st.slider("Number of questions", 3, 10, 5)

            with col2:
                use_ai = st.checkbox("Use AI-generated questions", value=True)

            with col3:
                role = st.selectbox(
                    "Interview role",
                    ["Data Science", "Python Developer", "Web Designing",
                     "DevOps Engineer", "Java Developer", "Database"]
                )

            if AUDIO_AVAILABLE:
                st.toggle(
                    "🎙️ Enable Audio Answers",
                    key="audio_mode",
                    value=True,
                    help="Record your answers using microphone"
                )
            else:
                st.session_state.audio_mode = False
                st.info("Audio mode unavailable. Install: streamlit-audiorecorder, SpeechRecognition, pydub")

            if st.button("🚀 Start Interview"):
                with st.spinner("Preparing your interview questions..."):
                    if use_ai:
                        questions = generate_questions(role, skills, num_q)
                    else:
                        all_q = load_questions(role)
                        if len(all_q) >= num_q:
                            questions = random.sample(all_q, num_q)
                        else:
                            questions = all_q

                if not questions:
                    st.error("No questions could be generated.")
                else:
                    st.session_state.questions = questions
                    st.session_state.current_q = 0
                    st.session_state.qa_pairs = []
                    st.session_state.interview_done = False
                    st.session_state.interview_summary = None
                    st.rerun()

        elif not st.session_state.interview_done:
            questions = st.session_state.questions
            current = st.session_state.current_q
            total = len(questions)
            audio_mode = st.session_state.get("audio_mode", False)

            st.markdown(f"**Question {current + 1} of {total}**")
            st.progress((current + 1) / total)

            question_text = str(questions[current]).strip()
            if not question_text or len(question_text) < 5:
                question_text = "Please explain one project from your resume."

            st.markdown(f"""
            <div class='card'>
                <div style='color:#8b949e; font-size:13px; margin-bottom:8px;'>
                    Question {current + 1}
                </div>
                <div style='font-size:18px; font-weight:600;'>
                    {question_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            final_answer = ""

            if audio_mode and AUDIO_AVAILABLE:
                tab1, tab2 = st.tabs(["🎙️ Audio Answer", "⌨️ Text Answer"])

                with tab1:
                    audio = audiorecorder(
                        start_prompt="🎙️ Click to Record",
                        stop_prompt="⏹️ Stop Recording",
                        pause_prompt="",
                        key=f"audio_{current}"
                    )

                    if len(audio) > 0:
                        audio_bytes = audio.export().read()
                        st.audio(audio_bytes, format="audio/wav")

                        with st.spinner("Transcribing audio..."):
                            transcribed = transcribe_audio(audio_bytes)

                        if transcribed:
                            final_answer = st.text_area(
                                "Edit transcription if needed:",
                                value=transcribed,
                                height=120,
                                key=f"audio_edit_{current}"
                            )

                with tab2:
                    typed_answer = st.text_area(
                        "Your Answer",
                        height=150,
                        key=f"text_answer_{current}"
                    )
                    if typed_answer.strip():
                        final_answer = typed_answer
            else:
                final_answer = st.text_area(
                    "Your Answer",
                    height=150,
                    key=f"answer_{current}"
                )

            if st.button("Submit Answer ➡️"):
                if not final_answer.strip():
                    st.warning("Please provide an answer first.")
                else:
                    with st.spinner("Evaluating your answer..."):
                        feedback = evaluate_answer(question_text, final_answer)

                    st.session_state.qa_pairs.append({
                        "question": question_text,
                        "answer": final_answer,
                        "score": feedback.get("score", "6"),
                        "feedback": feedback,
                    })

                    st.success(f"Score: {feedback.get('score', '6')}/10")
                    st.write("**Strength:**", feedback.get("strengths", "N/A"))
                    st.write("**Weakness:**", feedback.get("weaknesses", "N/A"))
                    st.write("**Better Answer:**", feedback.get("improved_answer", "N/A"))
                    st.write("**Tip:**", feedback.get("tip", "N/A"))

                    if current + 1 < total:
                        if st.button("Next Question"):
                            st.session_state.current_q += 1
                            st.rerun()
                    else:
                        if st.button("Finish Interview"):
                            summary = get_interview_summary(st.session_state.qa_pairs, category)
                            st.session_state.interview_summary = summary
                            st.session_state.interview_done = True

                            overall_score = summary.get("overall_score", 0)
                            try:
                                overall_score = float(str(overall_score).replace("/10", "").strip())
                            except Exception:
                                overall_score = 0

                            save_interview_result(
                                user_id=st.session_state.user["id"],
                                interview_score=overall_score,
                                target_role=category
                            )

                            st.rerun()

        else:
            summary = st.session_state.interview_summary or {}
            st.markdown("## 🏁 Interview Complete!")
            st.write("**Overall Score:**", summary.get("overall_score", "N/A"))
            st.write("**Overall Feedback:**", summary.get("overall_feedback", "N/A"))
            st.write("**Top Strength:**", summary.get("top_strength", "N/A"))
            st.write("**Top Weakness:**", summary.get("top_weakness", "N/A"))
            st.write("**Recommendation:**", summary.get("recommendation", "N/A"))

            st.markdown("### 📋 Question-by-Question Review")
            for i, qa in enumerate(st.session_state.qa_pairs, 1):
                with st.expander(f"Q{i}: {qa['question'][:60]}..."):
                    st.markdown(f"**Your Answer:** {qa['answer']}")
                    st.markdown(f"**Score:** {qa['score']}/10")
                    fb = qa.get("feedback", {})
                    st.markdown(f"✅ **Strength:** {fb.get('strengths', 'N/A')}")
                    st.markdown(f"⚠️ **Weakness:** {fb.get('weaknesses', 'N/A')}")
                    st.markdown(f"💡 **Better Answer:** {fb.get('improved_answer', 'N/A')}")

            if st.button("🔄 Start New Interview"):
                st.session_state.questions = []
                st.session_state.current_q = 0
                st.session_state.qa_pairs = []
                st.session_state.interview_done = False
                st.session_state.interview_summary = None
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INTERNSHIP RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Internship Recommendation":
    st.markdown("<div class='section-header'>🎯 AI Internship Recommendation</div>", unsafe_allow_html=True)

    if not st.session_state.parsed_resume:
        st.warning("⚠️ Please analyze your resume first in the Resume Analyzer!")
    elif st.session_state.internship_recommender is None:
        st.error("Internship recommender could not be loaded. Check your CSV path and recommender file.")
        st.caption(f"Expected CSV path: {INTERNSHIP_CSV_PATH}")
    else:
        parsed = st.session_state.parsed_resume
        user_skills = parsed.get("skills", [])
        detected_role = parsed.get("category", "General")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"**Detected Resume Category:** {detected_role}")
        st.write(f"**Skills Found in Resume:** {len(user_skills)}")
        if user_skills:
            tags = "".join([f"<span class='skill-tag'>{s}</span>" for s in user_skills[:15]])
            st.markdown(tags, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            preferred_role = st.text_input("Preferred Role", value=detected_role)
            preferred_functional_area = st.text_input("Preferred Functional Area", value="")
            preferred_company = st.text_input("Preferred Company", value="")

        with col2:
            preferred_industry = st.text_input("Preferred Industry", value="")
            max_distance = st.slider("Maximum Distance (km)", 1, 600, 200)
            top_n = st.slider("Number of Recommendations", 1, 20, 5)

        if st.button("🔍 Recommend Internships"):
            with st.spinner("Finding the best internships for you..."):
                recommendations = st.session_state.internship_recommender.recommend(
                    user_skills=user_skills,
                    preferred_role=preferred_role,
                    preferred_functional_area=preferred_functional_area,
                    preferred_industry=preferred_industry,
                    preferred_company=preferred_company,
                    preferred_state="madhya pradesh",
                    max_distance=max_distance,
                    top_n=top_n
                )

                st.session_state.internship_recommendations = recommendations

                if recommendations is not None and not recommendations.empty:
                    save_internship_recommendations(
                        st.session_state.user["id"],
                        recommendations
                    )
                    st.success("Internship recommendations generated successfully.")
                else:
                    st.warning("No matching internships found for your criteria.")

        if st.session_state.internship_recommendations is not None and not st.session_state.internship_recommendations.empty:
            st.markdown("### 🎯 Top Internship Recommendations")
            recommendations = st.session_state.internship_recommendations

            for _, row in recommendations.iterrows():
                explanation = st.session_state.internship_recommender.explain_recommendation(
                    internship_row=row,
                    user_skills=user_skills,
                    preferred_role=preferred_role
                )

                st.markdown(f"""
                <div class='card'>
                    <h3 style='color:#58a6ff; margin-bottom:10px;'>{str(row['company_name']).title()}</h3>
                    <p><b>Role:</b> {str(row['internship_role']).title()}</p>
                    <p><b>Functional Area:</b> {str(row['functional_area']).title()}</p>
                    <p><b>Industry:</b> {str(row['industry']).title()}</p>
                    <p><b>State:</b> {str(row['state']).title()}</p>
                    <p><b>Distance:</b> {row['distance_km']} km</p>
                    <p><b>Recommendation Score:</b> {round(float(row['recommendation_score']), 4)}</p>
                    <p><b>Why recommended:</b> {explanation}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📋 Tabular View")
            st.dataframe(recommendations, use_container_width=True)
        else:
            previous_rows = fetch_latest_internship_recommendations(st.session_state.user["id"], limit=5)
            if previous_rows:
                st.markdown("### 🕘 Latest Saved Recommendations")
                for row in previous_rows:
                    st.markdown(f"""
                    <div class='card'>
                        <h3 style='color:#58a6ff;'>{row['company_name']}</h3>
                        <p><b>Role:</b> {row['internship_role']}</p>
                        <p><b>Industry:</b> {row['industry']}</p>
                        <p><b>Distance:</b> {row['distance_km']} km</p>
                        <p><b>Score:</b> {round(float(row['recommendation_score']), 4)}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI SUGGESTIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 AI Suggestions":
    st.markdown("<div class='section-header'>💡 AI Resume Suggestions</div>", unsafe_allow_html=True)

    if not st.session_state.parsed_resume:
        st.warning("⚠️ Please analyze your resume first in the Resume Analyzer!")
    else:
        parsed = st.session_state.parsed_resume
        result = st.session_state.placement_result

        col1, col2 = st.columns(2)

        with col1:
            target_role = st.selectbox(
                "Target Role",
                ["Data Science", "Python Developer", "Web Designing",
                 "DevOps Engineer", "Java Developer", "Database",
                 "Blockchain", "Network Security Engineer"]
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            get_tips = st.button("✨ Get AI Suggestions")

        if get_tips:
            with st.spinner("Generating personalized suggestions..."):
                tips = get_resume_suggestions(
                    skills=parsed.get("skills", []),
                    missing_skills=result.get("missing_skills", []),
                    category=target_role
                )

            st.markdown("### 🎯 Your Personalized Action Plan")
            st.markdown(f"""
            <div class='card'>
                {tips.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Your Current Profile")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Skills You Have:**")
            skills = parsed.get("skills", [])
            if skills:
                tags = "".join([f"<span class='skill-tag'>{s}</span>" for s in skills])
                st.markdown(tags, unsafe_allow_html=True)

        with col2:
            st.markdown("**❌ Key Skills Missing:**")
            missing = result.get("missing_skills", [])
            if missing:
                tags = "".join([f"<span class='missing-tag'>{s}</span>" for s in missing])
                st.markdown(tags, unsafe_allow_html=True)
            else:
                st.success("No critical skills missing! 🎉")