import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Helper mappings
# -----------------------------
ROLE_SKILL_MAP = {
    "data scientist": ["python", "machine learning", "pandas", "numpy", "sql", "statistics"],
    "python developer": ["python", "flask", "django", "fastapi", "sql", "git"],
    "web developer": ["html", "css", "javascript", "react", "node", "sql", "git"],
    "devops engineer": ["docker", "kubernetes", "aws", "linux", "git", "ci/cd"],
    "java developer": ["java", "spring boot", "sql", "git", "rest api"],
    "database": ["sql", "mysql", "postgresql", "database"],
    "it intern": ["python", "sql", "git", "computer networks"],
    "information technology": ["python", "sql", "git", "api"],
    "sales & marketing": ["communication", "sales", "marketing", "negotiation"],
    "operations management": ["excel", "analysis", "coordination", "reporting"],
    "finance & accounting": ["excel", "finance", "accounting", "analysis"],
    "human resources": ["communication", "recruitment", "coordination"],
    "maintenance": ["technical", "safety", "equipment"],
    "customer care / service": ["communication", "customer service", "problem solving"],
}


# -----------------------------
# Text cleaning
# -----------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# -----------------------------
# Main recommender class
# -----------------------------
class InternshipRecommender:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None
        self.vectorizer = None
        self.text_matrix = None

        self.load_data()
        self.prepare_features()

    def load_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = [
            "company_name",
            "internship_role",
            "functional_area",
            "industry",
            "state",
            "distance_km",
        ]

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        for col in ["company_name", "internship_role", "functional_area", "industry", "state"]:
            self.df[col] = self.df[col].apply(normalize_text)

        self.df["distance_km"] = pd.to_numeric(self.df["distance_km"], errors="coerce")
        self.df["distance_km"] = self.df["distance_km"].fillna(self.df["distance_km"].median())

        self.df = self.df.drop_duplicates().reset_index(drop=True)

    def prepare_features(self):
        self.df["combined_text"] = (
            self.df["company_name"] + " " +
            self.df["internship_role"] + " " +
            self.df["functional_area"] + " " +
            self.df["industry"] + " " +
            self.df["state"]
        )

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.text_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

    # -----------------------------
    # Infer skills for internship row
    # -----------------------------
    def infer_required_skills(self, internship_role, functional_area):
        internship_role = normalize_text(internship_role)
        functional_area = normalize_text(functional_area)

        inferred_skills = set()

        for key, skills in ROLE_SKILL_MAP.items():
            if key in internship_role or key in functional_area:
                inferred_skills.update(skills)

        return inferred_skills

    # -----------------------------
    # Build user query from filters
    # -----------------------------
    def build_user_query(
        self,
        preferred_role="",
        preferred_functional_area="",
        preferred_industry="",
        preferred_company="",
        preferred_state="madhya pradesh",
    ):
        parts = [
            normalize_text(preferred_role),
            normalize_text(preferred_functional_area),
            normalize_text(preferred_industry),
            normalize_text(preferred_company),
            normalize_text(preferred_state),
        ]
        return " ".join([p for p in parts if p]).strip()

    # -----------------------------
    # Recommend internships
    # -----------------------------
    def recommend(
        self,
        user_skills=None,
        preferred_role="",
        preferred_functional_area="",
        preferred_industry="",
        preferred_company="",
        preferred_state="madhya pradesh",
        max_distance=None,
        top_n=5,
    ):
        if user_skills is None:
            user_skills = []

        user_skills = {normalize_text(skill) for skill in user_skills if str(skill).strip()}

        user_query = self.build_user_query(
            preferred_role=preferred_role,
            preferred_functional_area=preferred_functional_area,
            preferred_industry=preferred_industry,
            preferred_company=preferred_company,
            preferred_state=preferred_state,
        )

        # If no query from filters, still use role + skills fallback
        if not user_query:
            fallback_query = " ".join(list(user_skills)[:10])
            user_query = fallback_query if fallback_query else "internship madhya pradesh"

        user_vector = self.vectorizer.transform([user_query])
        text_scores = cosine_similarity(user_vector, self.text_matrix).flatten()

        results = self.df.copy()
        results["text_score"] = text_scores

        # Exact / partial matching boosts
        role_boost = []
        function_boost = []
        industry_boost = []
        company_boost = []
        state_boost = []
        skill_overlap_score = []
        distance_score = []

        for _, row in results.iterrows():
            row_role = row["internship_role"]
            row_function = row["functional_area"]
            row_industry = row["industry"]
            row_company = row["company_name"]
            row_state = row["state"]
            row_distance = safe_float(row["distance_km"], default=9999)

            # 1. role boost
            rb = 1.0 if preferred_role and normalize_text(preferred_role) in row_role else 0.0
            role_boost.append(rb)

            # 2. functional area boost
            fb = 1.0 if preferred_functional_area and normalize_text(preferred_functional_area) in row_function else 0.0
            function_boost.append(fb)

            # 3. industry boost
            ib = 1.0 if preferred_industry and normalize_text(preferred_industry) in row_industry else 0.0
            industry_boost.append(ib)

            # 4. company boost
            cb = 1.0 if preferred_company and normalize_text(preferred_company) in row_company else 0.0
            company_boost.append(cb)

            # 5. state boost
            sb = 1.0 if preferred_state and normalize_text(preferred_state) in row_state else 0.0
            state_boost.append(sb)

            # 6. skill overlap score
            inferred_skills = self.infer_required_skills(row_role, row_function)
            if inferred_skills:
                overlap = len(user_skills.intersection(inferred_skills)) / len(inferred_skills)
            else:
                overlap = 0.0
            skill_overlap_score.append(overlap)

            # 7. distance score
            if max_distance is not None:
                if row_distance <= max_distance:
                    ds = 1 - (row_distance / max_distance) if max_distance > 0 else 1.0
                    ds = max(ds, 0.0)
                else:
                    ds = -0.5
            else:
                # soft preference for nearer internships
                ds = 1 / (1 + row_distance)
            distance_score.append(ds)

        results["role_boost"] = role_boost
        results["function_boost"] = function_boost
        results["industry_boost"] = industry_boost
        results["company_boost"] = company_boost
        results["state_boost"] = state_boost
        results["skill_overlap_score"] = skill_overlap_score
        results["distance_score"] = distance_score

        # Final weighted score
        results["recommendation_score"] = (
            0.40 * results["text_score"] +
            0.20 * results["skill_overlap_score"] +
            0.12 * results["role_boost"] +
            0.08 * results["function_boost"] +
            0.07 * results["industry_boost"] +
            0.05 * results["company_boost"] +
            0.03 * results["state_boost"] +
            0.05 * results["distance_score"]
        )

        # Strict filtering by distance if provided
        if max_distance is not None:
            results = results[results["distance_km"] <= max_distance]

        results = results.sort_values(
            by=["recommendation_score", "distance_km"],
            ascending=[False, True]
        ).reset_index(drop=True)

        return results.head(top_n)[[
            "company_name",
            "internship_role",
            "functional_area",
            "industry",
            "state",
            "distance_km",
            "recommendation_score",
            "skill_overlap_score",
        ]]

    # -----------------------------
    # Explain why a row was recommended
    # -----------------------------
    def explain_recommendation(self, internship_row, user_skills=None, preferred_role=""):
        if user_skills is None:
            user_skills = []

        user_skills = {normalize_text(skill) for skill in user_skills if str(skill).strip()}

        inferred_skills = self.infer_required_skills(
            internship_row.get("internship_role", ""),
            internship_row.get("functional_area", "")
        )

        matched_skills = sorted(user_skills.intersection(inferred_skills))

        reasons = []

        if preferred_role and normalize_text(preferred_role) in normalize_text(internship_row.get("internship_role", "")):
            reasons.append("role matches your preferred role")

        if matched_skills:
            reasons.append(f"matched skills: {', '.join(matched_skills[:5])}")

        reasons.append(f"distance is {internship_row.get('distance_km', 'N/A')} km")

        return " | ".join(reasons)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    recommender = InternshipRecommender("data/structured_internships_madhya_pradesh.csv")

    user_skills = ["python", "sql", "git", "communication"]

    top_results = recommender.recommend(
        user_skills=user_skills,
        preferred_role="it intern",
        preferred_functional_area="information technology",
        preferred_industry="oil, gas & energy",
        preferred_company="gail",
        preferred_state="madhya pradesh",
        max_distance=200,
        top_n=10,
    )

    print(top_results.to_string(index=False))