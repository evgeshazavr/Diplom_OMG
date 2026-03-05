import pandas as pd
import numpy as np

# Пример данных: 100 сэмплов
np.random.seed(42)
programs = ['Информатика', 'Экономика', 'Медицина']
data = {
    'answer1': [f"Люблю {np.random.choice(['кодить', 'считать деньги', 'помогать людям'])} и {np.random.choice(['логику', 'финансы', 'биологию'])}" for _ in range(100)],
    'answer2': [f"Мои способности: {np.random.choice(['программирование', 'анализ', 'эмпатия'])}" for _ in range(100)],
    'answer3': [f"Интересы: {np.random.choice(['IT', 'бизнес', 'здравоохранение'])}" for _ in range(100)],
    'program': np.random.choice(programs, 100)
}
df = pd.DataFrame(data)
df.to_csv('dataset/train.csv', index=False)
print("Датасет создан: dataset/train.csv")