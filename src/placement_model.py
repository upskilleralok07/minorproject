import os
import pickle
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def predict_placement(parsed_resume: dict) -> dict:
    """
    Input  : output dict from parse_resume()
    Output : placement score, label, missing skills, suggestions
    """
    models_dir = os.path.join(BASE_DIR, "models")

    with open(os.path.join(models_dir, "placement_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    # Patch sklearn compatibility issue for older/newer pickles
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"

    # Build feature row
    skill_diversity = len(parsed_resume.get("skill_breakdown", {}))
    word_count = parsed_resume.get("word_count", 0)
    resume_quality = min(word_count / 500, 1.0)

    row = {
        "num_skills": parsed_resume.get("num_skills", 0),
        "has_python": parsed_resume.get("has_python", 0),
        "has_ml": parsed_resume.get("has_ml", 0),
        "has_sql": parsed_resume.get("has_sql", 0),
        "has_dl": parsed_resume.get("has_dl", 0),
        "has_web": parsed_resume.get("has_web", 0),
        "has_cloud": parsed_resume.get("has_cloud", 0),
        "skill_diversity": skill_diversity,
        "resume_quality": resume_quality,
        "word_count": word_count,
    }

    X = pd.DataFrame([row])

    # Ensure all expected feature columns exist
    for feature in features:
        if feature not in X.columns:
            X[feature] = 0

    X = X[features]
    X_scaled = scaler.transform(X)

    # Safe prediction
    try:
        prob = model.predict_proba(X_scaled)[0][1]
    except Exception:
        pred = model.predict(X_scaled)[0]
        prob = 0.75 if pred == 1 else 0.35

    label = "✅ Likely to be Placed" if prob > 0.5 else "❌ Needs Improvement"

    important = ["python", "sql", "machine learning", "docker", "git", "deep learning", "aws"]
    skills = parsed_resume.get("skills", [])
    missing = [s for s in important if s not in skills]

    return {
        "score": round(prob * 100, 1),
        "label": label,
        "skills_found": skills,
        "missing_skills": missing,
        "skill_breakdown": parsed_resume.get("skill_breakdown", {}),
    }