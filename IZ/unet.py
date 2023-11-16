from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Генерация синтетических данных для примера
def generate_synthetic_data(num_samples=100, image_size=(256, 256), noise_level=0.2):
    x_train = np.random.randn(num_samples, *image_size, 1)
    y_train = np.random.randn(num_samples, *image_size, 1)

    # Добавление шума и создание маски
    y_train[x_train < noise_level] = 1.0
    return x_train, y_train

def unet_model(input_size=(256, 256, 1)):
    inputs = keras.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom layer
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Гиперпараметры
epochs = 2
batch_size = 32

# Генерация данных
x_train, y_train = generate_synthetic_data(num_samples=1000)

# Создание и компиляция модели
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Визуализация результатов обучения
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Предсказания модели
predictions = model.predict(x_train[:5])

# Визуализация результатов (пример для первых 5 изображений)
for i in range(5):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(x_train[i, :, :, 0], cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(y_train[i, :, :, 0], cmap='gray')
    plt.title('Target Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.title('Predicted Mask')

    plt.show()
