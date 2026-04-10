import re
import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
resume_path = os.path.join(BASE_DIR, "data", "Resume_tech.csv")

# ── Tech Categories to KEEP ────────────────────────────────────────────────
TECH_CATEGORIES = [
    "Data Science", "Web Designing", "Java Developer", "Python Developer",
    "DevOps Engineer", "Network Security Engineer", "Database", "Hadoop",
    "ETL Developer", "DotNet Developer", "Blockchain", "Testing",
    "Automation Testing", "SAP Developer", "Business Analyst"
]

# ── Master Skills List ─────────────────────────────────────────────────────
SKILLS_LIST = [
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript",
    "r", "golang", "kotlin", "swift", "php", "ruby", "scala",

    # Web Development
    "html", "css", "react", "angular", "vue", "node", "flask",
    "django", "fastapi", "bootstrap", "rest api", "spring boot",

    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "data analysis", "data science", "pandas", "numpy", "scipy",
    "tensorflow", "keras", "pytorch", "scikit-learn", "opencv",
    "statistics", "data visualization", "feature engineering",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite",
    "firebase", "redis", "oracle", "cassandra",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes",
    "git", "github", "linux", "ci/cd", "jenkins", "ansible",

    # Data Engineering
    "hadoop", "spark", "kafka", "hive", "airflow", "etl",

    # Other Tech
    "blockchain", "selenium", "sap", "tableau",
    "power bi", "matplotlib", "seaborn", "excel",
]

# ── Skill Categories for Breakdown ────────────────────────────────────────
SKILL_CATEGORIES = {
    "Programming":    ["python", "java", "c++", "c#", "javascript",
                       "typescript", "r", "golang", "scala", "php"],
    "Web Dev":        ["html", "css", "react", "angular", "vue",
                       "node", "flask", "django", "fastapi", "spring boot"],
    "Data & ML":      ["machine learning", "deep learning", "nlp",
                       "data analysis", "data science", "pandas", "numpy",
                       "tensorflow", "pytorch", "scikit-learn", "statistics"],
    "Database":       ["sql", "mysql", "postgresql", "mongodb",
                       "sqlite", "oracle", "cassandra", "firebase"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes",
                       "git", "linux", "jenkins", "ansible"],
    "Data Engineering": ["hadoop", "spark", "kafka", "hive", "airflow", "etl"],
}

# ── Text Cleaning ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\r\n|\n|\r', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 /#+.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── Skill Extraction ───────────────────────────────────────────────────────
def extract_skills(text: str) -> list:
    found = []
    for skill in SKILLS_LIST:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return found

# ── Skill Breakdown by Category ────────────────────────────────────────────
def categorize_skills(skills: list) -> dict:
    breakdown = {}
    for category, cat_skills in SKILL_CATEGORIES.items():
        matched = [s for s in skills if s in cat_skills]
        if matched:
            breakdown[category] = matched
    return breakdown

# ── Parse Single Resume ────────────────────────────────────────────────────
def parse_resume(resume_text: str) -> dict:
    cleaned   = clean_text(resume_text)
    skills    = extract_skills(cleaned)
    breakdown = categorize_skills(skills)

    return {
        "cleaned_text":    cleaned,
        "skills":          skills,
        "num_skills":      len(skills),
        "skill_breakdown": breakdown,
        "word_count":      len(cleaned.split()),
        "has_python":      int("python" in skills),
        "has_ml":          int("machine learning" in skills),
        "has_sql":         int("sql" in skills),
        "has_dl":          int("deep learning" in skills),
        "has_web":         int(any(s in skills for s in
                               ["html", "react", "flask", "django"])),
        "has_cloud":       int(any(s in skills for s in
                               ["aws", "azure", "gcp", "docker"])),
    }

# ── Load & Parse Entire Dataset ────────────────────────────────────────────
def load_and_parse_all() -> pd.DataFrame:
    df = pd.read_csv(resume_path, encoding='latin1')

    # Filter tech categories only
    df = df[df['Category'].isin(TECH_CATEGORIES)].reset_index(drop=True)
    print(f"✅ Tech resumes found: {len(df)}")
    print(f"Categories: {df['Category'].unique()}\n")

    # Parse each resume
    parsed_rows = []
    for _, row in df.iterrows():
        parsed = parse_resume(str(row['Resume']))
        parsed['category'] = row['Category']
        parsed_rows.append(parsed)

    parsed_df = pd.DataFrame(parsed_rows)

    print(f"Average skills per resume : {parsed_df['num_skills'].mean():.1f}")
    print(f"Resumes with Python       : {parsed_df['has_python'].sum()}")
    print(f"Resumes with ML           : {parsed_df['has_ml'].sum()}")
    print(f"Resumes with SQL          : {parsed_df['has_sql'].sum()}")

    print(f"\n--- Sample: First Resume ---")
    print(f"Category : {parsed_df['category'][0]}")
    print(f"Skills   : {parsed_df['skills'][0]}")
    print(f"Breakdown: {parsed_df['skill_breakdown'][0]}")

    return parsed_df

# ── Run directly to test ───────────────────────────────────────────────────
if __name__ == "__main__":
    parsed_df = load_and_parse_all()
    print("\n✅ Resume parsing complete!")
    print(parsed_df[['category', 'num_skills', 'has_python',
                      'has_ml', 'has_sql']].head(10))