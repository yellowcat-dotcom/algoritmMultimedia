import tensorflow as tf
from keras import layers, models

def create_object_detection_model(input_shape, num_classes):
    model = models.Sequential()

    # Определение архитектуры модели
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Пример использования модели
input_shape = (224, 224, 3)  # Размер входного изображения (пример)
num_classes = 20  # Замените на количество классов, соответствующее вашей задаче

model = create_object_detection_model(input_shape, num_classes)

# Вывод структуры модели
model.save("model_1")
