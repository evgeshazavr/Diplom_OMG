"""
Рекомендательная система направлений для абитуриентов
======================================================
Архитектура: RAG (векторный поиск + LLM через Ollama)

Зависимости:
    pip install sentence-transformers ollama numpy

Запуск Ollama (предварительно):
    ollama pull mistral
    (Ollama запускается автоматически после установки)

Запуск системы:
    python recommender.py          -- интерактивный режим
    python recommender.py --demo   -- тест на примерах из датасета
"""

import json
import os
import re
import sys
import numpy as np

# Ollama работает локально — не пускать через системный прокси
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import ollama
from sentence_transformers import SentenceTransformer

# ================================================================
# КОНФИГУРАЦИЯ
# ================================================================

OLLAMA_MODEL = "mistral"          # или "llama3", "llama3.1" и т.д.
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"  # поддерживает русский
TOP_K        = 5                  # кандидатов из векторного поиска
TOP_N        = 3                  # направлений в итоговом ответе

RAG_CHUNKS_PATH  = "rag_chunks.json"
DIRECTIONS_PATH  = "directions_catalog.json"
DATASET_PATH     = "dataset_full.json"

# Официальные минимальные баллы ЕГЭ 2026
EGE_MIN_2026 = {
    "Русский язык":            40,
    "Математика (профильная)": 40,
    "Физика":                  41,
    "Информатика и ИКТ":       46,
    "Обществознание":          45,
    "История":                 40,
    "Литература":              40,
    "География":               40,
    "Иностранный язык":        40,
}

WORK_FORMAT_OPTIONS = {
    "1": "За компьютером (разработка, аналитика, дизайн)",
    "2": "Руками / в поле (инженер, монтаж, оборудование)",
    "3": "С людьми (менеджмент, коммуникации, продажи)",
    "4": "Ещё не определился(ась)",
}

IT_LEVEL_OPTIONS = {
    "1": "Никогда не программировал(а)",
    "2": "Пробовал(а) — писал(а) простые скрипты, делал(а) сайты",
    "3": "Уверенно пишу код на одном или нескольких языках",
    "4": "Не связываю себя с программированием",
}

# ================================================================
# ЗАГРУЗКА ДАННЫХ И ПОСТРОЕНИЕ ИНДЕКСА
# ================================================================

def load_data():
    with open(RAG_CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    with open(DIRECTIONS_PATH, encoding="utf-8") as f:
        directions = json.load(f)
    return chunks, directions

def build_index(chunks, model: SentenceTransformer):
    """Строим матрицу эмбеддингов для всех чанков."""
    print("Строю векторный индекс направлений...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)

def vector_search(query: str, index: np.ndarray, chunks: list,
                  model: SentenceTransformer, top_k: int = TOP_K,
                  level_filter: str = None) -> list:
    """Косинусный поиск по индексу."""
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = index @ q_emb
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])

    results = []
    for idx, score in ranked:
        chunk = chunks[idx]
        if level_filter and chunk.get("level") != level_filter:
            continue
        results.append({**chunk, "_score": float(score)})
        if len(results) >= top_k:
            break
    return results

# ================================================================
# ФИЛЬТР ЕГЭ
# ================================================================

def parse_exam_groups(raw_list: list) -> list:
    """Парсит вступительные испытания в группы OR-условий."""
    aliases = {
        'русский язык':      'Русский язык',
        'математика':        'Математика (профильная)',
        'физика':            'Физика',
        'информатика':       'Информатика и ИКТ',
        'информатика и икт': 'Информатика и ИКТ',
        'обществознание':    'Обществознание',
        'история':           'История',
        'литература':        'Литература',
        'география':         'География',
        'иностранный язык':  'Иностранный язык',
    }
    groups = []
    for item in raw_list:
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', item)
        text = re.sub(r'[-•*]', '', text).strip().lower().replace('\xa0', ' ')
        if not text:
            continue
        if any(x in text for x in ['собеседован', 'профессиональн', 'эссе', 'билет', 'испытан']):
            continue
        parts = re.split(r'\s+или\s+|/', text)
        group = [aliases[p.strip()] for p in parts if p.strip() in aliases]
        if group:
            groups.append(group)
    return groups

def check_ege_eligibility(direction: dict, ege_scores: dict) -> tuple:
    """Проверяет, проходит ли абитуриент по баллам ЕГЭ."""
    if direction.get("level") == "master":
        return True, []

    exam_groups = parse_exam_groups(direction.get("entrance_exams", []))
    if not exam_groups:
        return True, []

    reasons = []
    for group in exam_groups:
        group_ok = False
        min_score = 40
        for subj in group:
            score = ege_scores.get(subj, 0)
            min_score = EGE_MIN_2026.get(subj, 40)
            if score >= min_score:
                group_ok = True
                break
        if not group_ok:
            reasons.append(f"Нет балла по: {' / '.join(group)} (мин. {min_score})")

    return len(reasons) == 0, reasons

# ================================================================
# ПРОМПТ И ГЕНЕРАЦИЯ
# ================================================================

SYSTEM_PROMPT = """Ты — помощник приёмной комиссии университета.
На основе анкеты абитуриента и списка подходящих направлений выбери топ-3 и дай персональные рекомендации.

Правила:
- Отвечай строго на русском языке
- Для каждого направления укажи: код направления, официальное название, профиль, факультет
- Объясняй связь между интересами абитуриента и направлением
- Учитывай предпочтительный формат работы и уровень IT-подготовки
- Упоминай конкретные профессии и дисциплины
- Будь дружелюбным и мотивирующим
- Структурируй ответ: 1. ... 2. ... 3. ...
"""

def build_rag_prompt(applicant: dict, candidates: list) -> str:
    """Собирает промпт для LLM из анкеты и найденных кандидатов."""
    lines = ["## Анкета абитуриента\n"]
    lines.append(f"Имя: {applicant.get('full_name', 'не указано')}")
    lines.append(f"Дата рождения: {applicant.get('birth_date', 'не указана')}")
    lines.append(f"Тип поступления: {applicant.get('applicant_type', 'Бакалавриат')}")

    if applicant.get("ege_scores"):
        lines.append("\nБаллы ЕГЭ:")
        for subj, score in applicant["ege_scores"].items():
            min_s = EGE_MIN_2026.get(subj, 40)
            status = "✓" if score >= min_s else "✗ ниже минимума"
            lines.append(f"  • {subj}: {score} ({status})")

    if applicant.get("previous_education"):
        lines.append(f"\nПредыдущее образование: {applicant['previous_education']}")

    if applicant.get("work_format"):
        lines.append(f"\nПредпочтительный формат работы: {applicant['work_format']}")

    if applicant.get("it_level"):
        lines.append(f"Уровень IT-подготовки: {applicant['it_level']}")

    lines.append(f"\nИнтересы и цели: {applicant.get('interests', 'не указано')}")

    lines.append("\n## Подходящие направления (выбери топ-3)\n")
    for i, c in enumerate(candidates, 1):
        lvl = "Бакалавриат" if c["level"] == "bachelor" else "Магистратура"
        lines.append(f"### Вариант {i}: {c['direction_title']} ({lvl}, {c['faculty']})")
        lines.append(c["text"])
        lines.append("")

    lines.append("## Задача")
    lines.append(
        "Выбери топ-3 наиболее подходящих направления из списка выше. "
        "Для каждого укажи код, официальное название направления, профиль и объясни "
        "почему оно подходит этому абитуриенту с учётом его интересов, формата работы и уровня подготовки."
    )

    return "\n".join(lines)

def get_recommendation(applicant: dict, chunks: list, index: np.ndarray,
                       embed_model: SentenceTransformer, directions: list) -> str:
    """Полный пайплайн: поиск → фильтр ЕГЭ → генерация ответа."""
    app_type = applicant.get("applicant_type", "")
    is_master = "магистр" in app_type.lower()
    level_filter = "master" if is_master else "bachelor"

    # Строим поисковый запрос из интересов + формата работы
    query_parts = []
    if applicant.get("previous_education"):
        query_parts.append(applicant["previous_education"])
    if applicant.get("interests"):
        query_parts.append(applicant["interests"])
    if applicant.get("work_format"):
        query_parts.append(applicant["work_format"])
    query = " ".join(query_parts)

    # Векторный поиск
    candidates = vector_search(query, index, chunks, embed_model,
                               top_k=TOP_K * 2, level_filter=level_filter)

    # Фильтр ЕГЭ (только бакалавриат)
    if not is_master and applicant.get("ege_scores"):
        ege_scores = applicant["ege_scores"]
        dir_map = {d["title"]: d for d in directions}
        filtered, rejected = [], []
        for c in candidates:
            d = dir_map.get(c["direction_title"])
            if d is None:
                filtered.append(c)
                continue
            ok, reasons = check_ege_eligibility(d, ege_scores)
            if ok:
                filtered.append(c)
            else:
                rejected.append((c["direction_title"], reasons))

        if rejected:
            print(f"\n[Фильтр ЕГЭ] Исключено: {len(rejected)} направлений")
            for name, reasons in rejected[:3]:
                print(f"  - {name}: {'; '.join(reasons)}")

        candidates = filtered

    candidates = candidates[:TOP_K]

    if not candidates:
        return (
            "К сожалению, по вашим баллам ЕГЭ не удалось подобрать подходящих направлений. "
            "Рекомендуем обратиться в приёмную комиссию для консультации."
        )

    prompt = build_rag_prompt(applicant, candidates)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": 0.3, "num_predict": 1024},
    )

    top_directions = [
        {"title": c["direction_title"], "level": c["level"], "faculty": c["faculty"]}
        for c in candidates[:3]
    ]

    return {"text": response["message"]["content"], "top_directions": top_directions}

# ================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ================================================================

def ask(prompt, options=None):
    """Задаёт вопрос и возвращает ответ. Если options — показывает меню."""
    if options:
        for key, val in options.items():
            print(f"  {key}. {val}")
        while True:
            choice = input(prompt).strip()
            if choice in options:
                return options[choice]
            print("  Введите цифру из списка.")
    return input(prompt).strip()

def interactive_mode(chunks, index, embed_model, directions):
    print("\n" + "=" * 60)
    print("   Помощник по выбору образовательного направления")
    print("=" * 60)

    print("\nТип поступления:")
    app_type_choice = ask("  Введите цифру (1 или 2): ", {"1": "Бакалавриат", "2": "Магистратура"})
    is_master = app_type_choice == "Магистратура"

    applicant = {}
    applicant["full_name"]      = input("\nФИО: ").strip()
    applicant["birth_date"]     = input("Дата рождения (ДД.ММ.ГГГГ): ").strip()
    applicant["applicant_type"] = "Магистратура" if is_master else "Бакалавриат (ЕГЭ)"

    if is_master:
        applicant["previous_education"] = input("\nСпециальность / направление бакалавриата: ").strip()
    else:
        print("\nБаллы ЕГЭ (Enter — пропустить предмет):")
        ege = {}
        for subj, min_score in EGE_MIN_2026.items():
            val = input(f"  {subj} (мин. {min_score}): ").strip()
            if val.isdigit():
                ege[subj] = int(val)
        applicant["ege_scores"] = ege

    print("\nПредпочтительный формат работы:")
    applicant["work_format"] = ask("  Введите цифру: ", WORK_FORMAT_OPTIONS)

    print("\nУровень IT-подготовки:")
    applicant["it_level"] = ask("  Введите цифру: ", IT_LEVEL_OPTIONS)

    applicant["interests"] = input(
        "\nОпишите свои интересы и чем хотите заниматься\n"
        "(пишите свободно, например: «люблю математику, хочу в IT»): "
    ).strip()

    print("\n⏳ Подбираю направления...")
    result = get_recommendation(applicant, chunks, index, embed_model, directions)

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 60)
    print(result["text"] if isinstance(result, dict) else result)

# ================================================================
# DEMO
# ================================================================

def run_demo(chunks, index, embed_model, directions):
    """Прогон на примерах из датасета."""
    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"{DATASET_PATH} не найден, пропускаю demo.")
        return

    print("\n" + "=" * 60)
    print("DEMO: тест на примерах из датасета")
    print("=" * 60)

    examples = (
        [ex for ex in dataset if ex["meta"]["type"] == "bachelor"][:1] +
        [ex for ex in dataset if ex["meta"]["type"] == "master"][:1]
    )

    for ex in examples:
        applicant = json.loads(ex["input"])
        print(f"\n--- {applicant['full_name']} ({applicant['applicant_type']}) ---")
        print(f"Интересы: {applicant.get('interests', '')}")
        if applicant.get("work_format"):
            print(f"Формат работы: {applicant['work_format']}")
        if applicant.get("it_level"):
            print(f"IT-уровень: {applicant['it_level']}")

        result = get_recommendation(applicant, chunks, index, embed_model, directions)
        print("\nОтвет модели:")
        print(result["text"] if isinstance(result, dict) else result)
        print("\nОжидаемый (из датасета):")
        print(ex["output"][:500] + "...")
        print()

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    chunks, directions = load_data()

    print(f"Загружаю модель эмбеддингов: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    index = build_index(chunks, embed_model)

    if "--demo" in sys.argv:
        run_demo(chunks, index, embed_model, directions)
    else:
        interactive_mode(chunks, index, embed_model, directions)
