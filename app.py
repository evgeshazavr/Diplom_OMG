"""
app.py — Flask API для EduPath AI
===================================
Оборачивает recommender.py в REST API, которое фронтенд вызывает через fetch().

Установка зависимостей:
    pip install flask flask-cors sentence-transformers ollama numpy

Запуск:
    python app.py
    → http://localhost:5000

После запуска открой edupath.html в браузере.
Фронтенд будет слать POST /recommend и получать JSON с рекомендациями.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

# Импортируем функции из recommender.py
from recommender import (
    load_data,
    build_index,
    get_recommendation,
    SentenceTransformer,
    EMBED_MODEL,
)

app = Flask(__name__)
CORS(app)  # разрешаем запросы с фронтенда (localhost / file://)

# ── Инициализация при старте ──────────────────────────────────────
print("Загружаю данные и модель эмбеддингов...")
chunks, directions = load_data()
embed_model = SentenceTransformer(EMBED_MODEL)
index = build_index(chunks, embed_model)
print("Готово! API доступен на http://localhost:5000")


# ── Эндпоинты ─────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Проверка работоспособности API."""
    return jsonify({"status": "ok", "model": EMBED_MODEL})


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Принимает анкету абитуриента, возвращает рекомендации.

    Тело запроса (JSON):
    {
        "full_name":       "Иван Иванов",
        "birth_date":      "01.01.2006",        // опционально
        "applicant_type":  "Бакалавриат (ЕГЭ)", // или "Магистратура"
        "ege_scores": {                           // только для бакалавриата
            "Русский язык": 80,
            "Математика (профильная)": 75,
            "Информатика и ИКТ": 90
        },
        "previous_education": "...",             // для магистратуры
        "work_format":  "За компьютером (разработка, аналитика, дизайн)",
        "it_level":     "Уверенно пишу код на одном или нескольких языках",
        "interests":    "хочу заниматься машинным обучением и анализом данных"
    }

    Возвращает (JSON):
    {
        "recommendation": "текст от LLM",
        "status": "ok"
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Пустое тело запроса"}), 400

    # Нормализуем тип поступления
    app_type = data.get("applicant_type", "Бакалавриат (ЕГЭ)")
    applicant = {
        "full_name":          data.get("full_name", "Абитуриент"),
        "birth_date":         data.get("birth_date", ""),
        "applicant_type":     app_type,
        "ege_scores":         data.get("ege_scores", {}),
        "previous_education": data.get("previous_education", ""),
        "work_format":        data.get("work_format", ""),
        "it_level":           data.get("it_level", ""),
        "interests":          data.get("interests", ""),
    }

    try:
        result = get_recommendation(applicant, chunks, index, embed_model, directions)
        return jsonify({"recommendation": result, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/directions", methods=["GET"])
def get_directions():
    """Возвращает весь каталог направлений (без RAG-чанков)."""
    return jsonify(directions)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
