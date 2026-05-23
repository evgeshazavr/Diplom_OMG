"""
app.py — Flask API для EduPath AI
===================================
Запуск:
    python app.py  →  http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import os
import threading
import webbrowser

from recommender import (
    load_data,
    build_index,
    get_recommendation,
    SentenceTransformer,
    EMBED_MODEL,
)
from database import (
    init_db,
    create_user,
    verify_user,
    get_user_by_id,
    save_recommendation,
    get_user_recommendations,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "edupath-dev-secret-2026")
CORS(app, supports_credentials=True)

# ── Инициализация при старте ──────────────────────────────────────
print("Инициализирую базу данных...")
init_db()
print("Загружаю данные и модель эмбеддингов...")
chunks, directions = load_data()
embed_model = SentenceTransformer(EMBED_MODEL)
vector_index = build_index(chunks, embed_model)
print("Готово! API доступен на http://localhost:5000")


# ── Фронтенд ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "edupath.html"))


@app.route("/logo.svg")
def logo():
    return send_file(os.path.join(BASE_DIR, "logo.svg"), mimetype="image/svg+xml")


# ── Авторизация ───────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True) or {}
    name  = (data.get("name")  or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Заполните все поля"}), 400
    if len(password) < 6:
        return jsonify({"error": "Пароль минимум 6 символов"}), 400

    user_id = create_user(name, email, password)
    if user_id is None:
        return jsonify({"error": "Пользователь с таким email уже существует"}), 409

    session["user_id"] = user_id
    return jsonify({"id": user_id, "name": name, "email": email, "status": "ok"})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True) or {}
    email    = (data.get("email")    or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Заполните все поля"}), 400

    user = verify_user(email, password)
    if not user:
        return jsonify({"error": "Неверный email или пароль"}), 401

    session["user_id"] = user["id"]
    return jsonify({"id": user["id"], "name": user["name"], "email": user["email"], "status": "ok"})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/api/me", methods=["GET"])
def api_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Не авторизован"}), 401
    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"error": "Пользователь не найден"}), 404
    return jsonify({"id": user["id"], "name": user["name"], "email": user["email"], "status": "ok"})


# ── Рекомендации ──────────────────────────────────────────────────

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Пустое тело запроса"}), 400

    applicant = {
        "full_name":          data.get("full_name", "Абитуриент"),
        "birth_date":         data.get("birth_date", ""),
        "applicant_type":     data.get("applicant_type", "Бакалавриат (ЕГЭ)"),
        "ege_scores":         data.get("ege_scores", {}),
        "previous_education": data.get("previous_education", ""),
        "work_format":        data.get("work_format", ""),
        "it_level":           data.get("it_level", ""),
        "interests":          data.get("interests", ""),
    }

    try:
        result = get_recommendation(applicant, chunks, vector_index, embed_model, directions)
        if isinstance(result, dict):
            rec_text = result["text"]
            top_dirs = result.get("top_directions", [])
        else:
            rec_text = result
            top_dirs = []

        # Сохраняем в БД если пользователь авторизован
        user_id = session.get("user_id")
        if user_id:
            save_recommendation(user_id, applicant, rec_text, top_dirs)

        return jsonify({"recommendation": rec_text, "top_directions": top_dirs, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Не авторизован"}), 401
    recs = get_user_recommendations(user_id)
    return jsonify({"history": recs, "status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": EMBED_MODEL})


@app.route("/directions", methods=["GET"])
def get_directions():
    return jsonify(directions)


if __name__ == "__main__":
    port = 5000
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(debug=False, host="0.0.0.0", port=port)
