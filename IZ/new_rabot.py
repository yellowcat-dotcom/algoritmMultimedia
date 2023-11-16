import sys
import cv2
import numpy as np

sys.setrecursionlimit(1000)

def dfs(x, y, contour_points):
    stack = [(x, y)]
    while stack:
        x, y = stack.pop()
        if 0 <= x < binary_image.shape[0] and 0 <= y < binary_image.shape[1] and binary_image[x][y] == 255:
            contour_points.append((x, y))
            binary_image[x][y] = 0  # Помечаем пиксель как посещенный
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

def findContours(binary_image):
    contours = []
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i][j] == 255:
                contour_points = []
                dfs(i, j, contour_points)
                contours.append(contour_points)
    return contours

# переводим изображение в двоичное
image = cv2.imread('evil_tumor.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

contours = findContours(binary_image.copy())

# Рисование контуров на оригинальном изображении
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
for contour_points in contours:
    for x, y in contour_points:
        output_image[x, y] = [255, 0, 225]



cv2.imshow('Contours', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
