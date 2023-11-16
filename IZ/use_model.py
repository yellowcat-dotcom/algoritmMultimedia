import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def predict_and_visualize(model, image_path, height=256, width=256):
    # преобразуем изображение
    img = load_img(image_path, target_size=(height, width), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Делайте прогнозы, используя загруженную модель
    prediction = model.predict(img_array)
    # преобразуем полученные значения в бинарные
    prediction = (prediction > 0.455).astype(np.uint8)

    # показываем изображения
    plt.figure()
    plt.imshow(prediction[0, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.show()


loaded_model = load_model('trained_model.h5')
predict_and_visualize(loaded_model, 'evil_tumor.jpg')
