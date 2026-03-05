# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib

# ────────────────────────────────────────────────
# Импорты расширений и моделей — в правильном порядке
# ────────────────────────────────────────────────
from extensions import db  # db отсюда
from models import User, Answer  # модели после db

app = Flask(__name__)

app.config['SECRET_KEY'] = 'super-secret-key-please-change-me-987654321'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///proforient.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

# Привязываем db к приложению
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Пожалуйста, войдите'
login_manager.login_message_category = 'info'

# Загрузка модели (если есть)
try:
    model = joblib.load('proforient_model.joblib')
except FileNotFoundError:
    model = None
    print("Модель proforient_model.joblib не найдена → запустите model_ml.py")


# ────────────────────────────────────────────────
# user_loader
# ────────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))


# ────────────────────────────────────────────────
# Маршруты (register, login, logout, quiz и т.д.)
# ────────────────────────────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Имя пользователя уже занято', 'danger')
            return redirect(url_for('register'))

        user = User(
            username=username,
            password=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash('Регистрация успешна', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            flash('Вход выполнен', 'success')
            return redirect(url_for('quiz'))
        flash('Неверные данные', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/quiz', methods=['GET', 'POST'])
@login_required
def quiz():
    if model is None:
        flash('Модель не загружена', 'danger')
        return render_template('quiz.html')

    if request.method == 'POST':
        answers = [
            request.form.get('answer1', '').strip(),
            request.form.get('answer2', '').strip(),
            request.form.get('answer3', '').strip()
        ]
        input_text = ' '.join(answers)

        probs = model.predict_proba([input_text])[0]
        programs = model.classes_
        sorted_idx = probs.argsort()[::-1]

        top1 = programs[sorted_idx[0]]
        similar1 = programs[sorted_idx[1]] if len(programs) > 1 else ''
        similar2 = programs[sorted_idx[2]] if len(programs) > 2 else ''

        new_answer = Answer(
            user_id=current_user.id,
            answers=' || '.join(answers),
            recommendation=top1,
            similars=f"{similar1},{similar2}"
        )
        db.session.add(new_answer)
        db.session.commit()

        return render_template('result.html',
                               top1=top1,
                               similar1=similar1,
                               similar2=similar2)

    return render_template('quiz.html')

@app.route('/')
def index():
    # Вариант 1: сразу перенаправляем на страницу входа (самый логичный)
    return redirect(url_for('login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # создаёт таблицы при первом запуске
    app.run(debug=True)