import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

# Загрузка
data = pd.read_csv('dataset/train.csv')
X = data['answer1'] + ' ' + data['answer2'] + ' ' + data['answer3']  # Объединим тексты
y = data['program']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + MLP
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500)),  # Лимит фич для простоты
    ('clf', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])
model.fit(X_train, y_train)

# Тестирование (для ВКР)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Сохранение
joblib.dump(model, 'proforient_model.joblib')