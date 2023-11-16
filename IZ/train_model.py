import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models



def unet_model(input_channels=1, height=256, width=256, num_classes=1):
    # Входной слой
    inputs = layers.Input(shape=(height, width, input_channels))

    # Сверточные слои и объединение в пул для захвата объектов
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Слои для восстановления изображения
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    up1 = layers.UpSampling2D(size=(2, 2))(conv2)

    # Выходной слой бинарной сегментации
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(up1)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# генерируем изображения и маски
train_images = np.random.rand(100, 256, 256, 1)
train_masks = np.random.randint(0, 2, size=(100, 256, 256, 1), dtype=np.uint8)

height, width, channels = 256, 256, 1


num_epochs = 2
batch_size = 20
num_classes = 1

# создаем
model = unet_model(input_channels=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# обучаем
model.fit(train_images, train_masks, epochs=num_epochs, batch_size=batch_size)

#сохраняем модель
model.save('trained_model.h5')
