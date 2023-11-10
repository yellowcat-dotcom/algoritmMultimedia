from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential

# Загрузка данных MNIST и предобработка
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка изображений и меток
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255  # Размерности изменяются для использования сверточных слоев
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  # А также значения пикселей масштабируются от 0 до 1
y_train = to_categorical(y_train, 10)  # Преобразование меток в векторы с однократным кодированием (one-hot encoding)
y_test = to_categorical(y_test, 10)

# Создание модели сверточной нейронной сети
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))  # Первый сверточный слой с 32 фильтрами размером 3x3 и функцией активации ReLU
model.add(MaxPooling2D(pool_size=(2, 2)))  # Первый слой субдискретизации (пулинга) с размером пула 2x2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))  # Второй сверточный слой с 64 фильтрами размером 3x3 и функцией активации ReLU
model.add(MaxPooling2D(pool_size=(2, 2)))  # Второй слой субдискретизации (пулинга) с размером пула 2x2
model.add(Flatten())  # Плоский слой, который преобразует данные в одномерный массив перед подачей их на полносвязные слои
model.add(Dense(128, activation='relu'))  # Полносвязный слой с 128 нейронами и функцией активации ReLU
model.add(Dense(10, activation='softmax'))  # Выходной слой с 10 нейронами (по одному для каждой цифры от 0 до 9) и функцией активации softmax

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Используется оптимизатор Adam и функция потери категориальная перекрестная энтропия

# Обучение модели
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))  # Модель обучается на тренировочных данных в течение 3 эпох с батч-размером 128

# Оценка модели на тестовых данных
score = model.evaluate(x_test, y_test, verbose=0)  # Модель оценивается на тестовых данных, и результаты сохраняются в переменной score

# Сохранение модели
model.save("my_nerone_set.keras")

# Вывод результатов
print('Test loss:', score[0])  # Вывод потери на тестовых данных
print('Test accuracy:', score[1])  # Вывод точности на тестовых данных
