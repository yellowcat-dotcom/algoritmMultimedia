# Импорт необходимых библиотек
import sys
import cv2
import numpy as np

# лимит рекурсий для предотвращения ошибок
sys.setrecursionlimit(100000)

# поиск в глубину
def dfs(x, y, contour_points, binary_image):
    # Добавляем текущий пиксель в стек
    stack = [(x, y)]
    while stack:
        # достаем пиксель
        x, y = stack.pop()
        # Проверяем он в изображении/он белый?
        if 0 <= x < binary_image.shape[0] and 0 <= y < binary_image.shape[1] and binary_image[x, y] == 255:

            # Добавление точки контура
            contour_points.append((x, y))
            #этот пиксель посещен (пиксель черный)
            binary_image[x, y] = 0

            # Добавляем соседние пиксели в стек для дальнейшего исследования
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

# !!!!!!!
def findContours(binary_image):
    # Список контуров
    contours = []

    #идем по изображению
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            # Если пиксель белый, начинается поиск контура
            if binary_image[i, j] == 255:

                # Точки контура
                contour_points = []

                #Начинаем поиск контура (поиск в глубину)
                dfs(i, j, contour_points, binary_image)

                #добавляем контур в список контуров
                contours.append(contour_points)
    return contours

# Загрузка изображения и преобразование его в двоичное
image = cv2.imread('metastases.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('шашаша', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Наш findContours
contours = findContours(binary_image.copy())

# создаем пустое изображение
output_image = np.zeros_like(image)

# Рисование контуров на новом изображении
for contour_points in contours:
    for x, y in contour_points:
        # красим пиксели границы белым
        output_image[x, y] = 255


cv2.imshow('Contours', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
