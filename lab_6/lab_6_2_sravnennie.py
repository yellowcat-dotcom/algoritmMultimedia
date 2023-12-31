import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

# Загрузка сохраненной модели
model = load_model("my_model.keras")

# Параметры для сравнения
epochs_list = [1, 2, 3]
learning_rates = [0.001, 0.01, 0.1]
num_layers_list = [1, 2, 3]  # Список для количества слоев
accuracies = []

# Загрузка и предобработка данных MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Предобработка данных
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Преобразование изображений в одномерные массивы
train_images_flat = train_images.reshape((60000, 784))
test_images_flat = test_images.reshape((10000, 784))

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Перебор различных значений эпох, скоростей обучения и количества слоев
for epochs in epochs_list:
    for lr in learning_rates:
        for num_layers in num_layers_list:
            # Создание модели с определенным количеством слоев
            layered_model = tf.keras.Sequential()
            for _ in range(num_layers):
                # 128 нейронов в слоях нейронной сети
                layered_model.add(tf.keras.layers.Dense(128, activation='relu'))
            layered_model.add(tf.keras.layers.Dense(10, activation='softmax'))

            # Компиляция модели
            layered_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

            # Обучение модели
            layered_model.fit(train_images_flat, train_labels, epochs=epochs, batch_size=128, verbose=0)

            # Оценка модели на тестовых данных
            test_loss, test_accuracy = layered_model.evaluate(test_images_flat, test_labels, verbose=0)

            # Добавление параметров в список accuracies
            accuracies.append((epochs, lr, num_layers, test_accuracy))

# Вывод результатов
for epochs, lr, num_layers, accuracy in accuracies:
    print(f'Эпохи: {epochs}, Скорость обучения: {lr}, Количество слоев: {num_layers}, Точность на тесте: {accuracy}')
