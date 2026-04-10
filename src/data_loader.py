import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

resume_path = os.path.join(BASE_DIR, "data", "Resume.csv")
job_path = os.path.join(BASE_DIR, "data", "job_descriptions.csv")
question_path = os.path.join(BASE_DIR, "data", "Software Questions.csv")

resume_df = pd.read_csv(resume_path, encoding='latin1')
job_df = pd.read_csv(job_path, encoding='latin1')
questions_df = pd.read_csv(question_path, encoding='latin1')

print("Resume Data:\n", resume_df.head())
print("Job Data:\n", job_df.head())
print("Questions Data:\n", questions_df.head())
print(resume_df.columns)
print(job_df.columns)
print(questions_df.columns)