from keras.models import load_model
import cv2
import numpy as np

# загружаем модель
model = load_model("my_model.keras")

# Загрузите изображение в оттенках серого и измените его размер на 28x28
image_path ="2.jpg"
img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_cv = cv2.resize(img_cv, (28, 28))

# Печать размера изображения (должен быть (28, 28))
print(img_cv.shape)

# Нормализуйте изображение, аналагично как делали с обучащими данными
image = img_cv / 255.0

# Разверните изображение в одномерный вектор
image = image.reshape(1, 784)

# Получите предсказание модели для изображения
predictions = model.predict(image)
# Печать предсказанных вероятностей для каждого класса
print(predictions)

# Получите индекс класса с наибольшей вероятностью (предполагается, что классы от 0 до 9)
predicted_class = np.argmax(predictions)

# Выведите предсказанный класс
print("Predicted class:", predicted_class)
