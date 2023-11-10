import cv2
import numpy as np
from keras.models import load_model

# Загрузка ранее сохраненной модели нейронной сети
model = load_model('my_nerone_set.keras')

# Загрузка и предобработка изображения с использованием OpenCV
img_path = '5.jpg'  # Путь к изображению, которое вы хотите распознать
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Загрузка изображения в оттенках серого

img = cv2.resize(img, (28, 28))  # Изменение размера изображения до 28x28 пикселей (аналогично размеру MNIST изображений)
img = img / 255.0  # Нормализация значений пикселей из диапазона 0-255 в диапазон 0-1

# Преобразование изображения в формат, ожидаемый моделью (1, 28, 28, 1)
img = img.reshape(1, 28, 28, 1)

# Предсказание с использованием загруженной модели
predictions = model.predict(img)
predicted_digit = np.argmax(predictions)  # Получение индекса с максимальной вероятностью, который представляет предсказанную цифру

print(f"Предсказанная цифра: {predicted_digit}")
