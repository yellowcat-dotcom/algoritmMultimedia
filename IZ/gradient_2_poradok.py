import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'insult.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# применение оператора Лапласа
def laplacian_operator(image):
    height, width = image.shape
    laplacian_result = np.zeros((height, width), dtype=np.float64)

    # Ядро второй производной по оси X
    kernel_x = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])

    # Ядро второй производной по оси Y
    kernel_y = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])

    # Применение оператора Лапласа
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            laplacian_result[i, j] = np.abs(np.sum(kernel_x * image[i-1:i+2, j-1:j+2])) + \
                                     np.abs(np.sum(kernel_y * image[i-1:i+2, j-1:j+2]))

    return laplacian_result

# Применение оператора Лапласа
laplacian_result = laplacian_operator(image)

# Пороговая обработка для выделения контуров
threshold_value = 30
binary_laplacian = np.where(laplacian_result > threshold_value, 255, 0).astype(np.uint8)

# Визуализация результатов
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(laplacian_result), cmap='gray')
plt.title('Laplacian')

plt.subplot(1, 3, 3)
plt.imshow(binary_laplacian, cmap='gray')
plt.title('Binary Laplacian')

plt.show()
