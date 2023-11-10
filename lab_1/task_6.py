import cv2
import numpy as np

# Инициализация камеры (в данном случае, используется встроенная камера)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка при открытии камеры.")
    exit()

while True:
    # Чтение кадра с камеры
    ret, frame = cap.read()

    if not ret:
        print("Ошибка при чтении кадра с камеры.")
        break

    # Получение размеров кадра
    height, width, _ = frame.shape

    # Создание изображения с крестом
    color = (0, 0, 255)  # Красный цвет в формате BGR
    thickness = 2  # Толщина

    # Вертикальный прямоугольник
    rect_width_1 = 50
    rect_height_1 = 350

    x1_1 = width // 2 - rect_width_1 // 2 #210
    y1_1 = height // 2 - rect_height_1 // 2 #360
    x2_1 = width // 2 + rect_width_1 // 2  #300
    y2_1 = height // 2 + rect_height_1 //2 #120

    # Горизонтальный прямоугольник
    rect_width_2 = 50
    rect_height_2 = 350

    x1_2 = width // 2 - rect_height_2 // 2
    y1_2 = height // 2 - rect_width_2 // 2
    x2_2 = width // 2 + rect_height_2 // 2
    y2_2 = height // 2 + rect_width_2 // 2

    # Отрисовка прямоугольников
    cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), color, thickness)
    cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), color, thickness)

    # Размер ядра для размытия
    kernel_size = (21, 21)

    # Часть изображения, соответствующая горизонтальному прямоугольнику
    img_part = frame[y1_2:y2_2, x1_2:x2_2]

    # Размытие части изображения
    img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 0)

    # Замена части изображения размытой версией
    frame[y1_2:y2_2, x1_2:x2_2] = img_part_blur

    # Отображение изображения с крестом
    cv2.imshow('Camera with Cross', frame)

    # Ожидание нажатия клавиши "q" для выхода
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Закрытие камеры и окна
cap.release()
cv2.destroyAllWindows()