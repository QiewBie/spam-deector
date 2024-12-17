from flask import Flask, request, render_template, jsonify
from pyngrok import ngrok
import tensorflow as tf
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Створення Flask-додатку
app = Flask(__name__)

MODEL_PATH = 'spam_classifier_model.h5'
VECTORIZER_PATH = 'vectorizer.pkl'  # Додано шлях до векторизатора

# Завантажуємо модель
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Модель не знайдено за шляхом {MODEL_PATH}")

# Завантажуємо векторизатор
if os.path.exists(VECTORIZER_PATH):
    with open(VECTORIZER_PATH, 'rb') as file:
        vectorizer = pickle.load(file)
else:
    raise FileNotFoundError(f"Векторизатор не знайдено за шляхом {VECTORIZER_PATH}")

# Обробка введених даних у Flask-маршруті
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")  # Це повідомлення для дебагу
        
        data = request.get_json()

        if 'input' not in data:
            return jsonify({"error": "Ключ 'input' відсутній у даних"})

        input_data = data['input']
        print(f"Input received: {input_data}")  # Це для перевірки введеного тексту

        if not input_data:
            return jsonify({"error": "Введення порожнє!"})

        # Перетворення введеного тексту через векторизатор
        input_vector = vectorizer.transform([input_data]).toarray()
        print(f"Input vector shape: {input_vector.shape}")  # Перевірка розміру вектора

        # Прогнозування результату (отримання ймовірності)
        prediction_prob = model.predict(input_vector)
        print(f"Prediction probability: {prediction_prob}")  # Перевірка ймовірності

        if prediction_prob is not None and len(prediction_prob) > 0:
            prediction_prob_value = prediction_prob[0][0]
            print(f"Prediction probability value: {prediction_prob_value}")  # Значення ймовірності

            # Повертаємо ймовірність як результат
            return jsonify({"prediction": float(prediction_prob_value)})

        return jsonify({"error": "Не вдалося зробити прогноз"})

    except KeyError:
        return jsonify({"error": "Ключ 'input' у JSON-відсутній"})
    except Exception as ex:
        print(f"Error: {str(ex)}")  # Додано для виведення помилок у консоль
        return jsonify({"error": str(ex)})


# Підключаємо Ngrok для віддаленого доступу
public_url = ngrok.connect(5000)
print(f"Сервіс Flask доступний за посиланням: {public_url}")

if __name__ == '__main__':
    app.run(port=5000)
