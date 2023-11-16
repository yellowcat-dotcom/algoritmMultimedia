import cv2

def canny_edge_detection(image_path, low_threshold, high_threshold):
    # Загрузите изображение с диска
    original_image = cv2.imread(image_path, 0)  # Чтение изображения в оттенках серого

    # Применить алгоритм Кенни для обнаружения границ
    edges = cv2.Canny(original_image, low_threshold, high_threshold)

    # Вывести изображение с обнаруженными границами
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Путь к изображению
image_path = 'insult.jpg'

# Нижний и верхний пороги для алгоритма Кенни
low_threshold = 50
high_threshold = 150

# Вызов функции для обнаружения границ на изображении
canny_edge_detection(image_path, low_threshold, high_threshold)
