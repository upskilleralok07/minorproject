import bcrypt
from src.db import get_connection


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def signup_user(name: str, email: str, password: str):
    conn = get_connection()
    cur = conn.cursor()

    try:
        hashed = hash_password(password)
        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, hashed)
        )
        conn.commit()
        return True, "Account created successfully."
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            return False, "Email already exists."
        return False, str(e)
    finally:
        conn.close()


def login_user(email: str, password: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()

    if user and verify_password(password, user["password"]):
        return True, dict(user)

    return False, "Invalid email or password."