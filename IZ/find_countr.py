import cv2
import numpy as np

def contour_detection(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Применение бинаризации для выделения объектов
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Поиск контуров в бинарном изображении
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание копии изображения для отображения контуров
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Рисование контуров на изображении
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    cv2.imshow("image_with_contours",image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Сохранение изображения с контурами


# Пример использования функции
input_image_path = 'insult.jpg'
contour_detection(input_image_path)
