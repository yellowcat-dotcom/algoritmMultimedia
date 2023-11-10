import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if not ret:
        break

    # Получаем высоту, ширину и количество каналов изображения.
    height, width, _ = frame.shape

    # Создаем пустое изображение (черно-белое) с такими же размерами как кадр.
    cross_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Определяем параметры для вертикальной линии.
    vertical_line_width = 60
    vertical_line_height = 300
    cv2.rectangle(cross_image,
                  (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2),
                  (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2),
                  (0, 0, 255), 2)
    rect_start_v = (width // 2 - vertical_line_width // 2, height // 2 - vertical_line_height // 2)
    rect_end_v = (width // 2 + vertical_line_width // 2, height // 2 + vertical_line_height // 2)

    # Определяем параметры для горизонтальной линии.
    horizontal_line_width = 250
    horizontal_line_height = 55
    cv2.rectangle(cross_image,
                  (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2),
                  (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2),
                  (0, 0, 255), 2)
    rect_start_h = (width // 2 - horizontal_line_width // 2, height // 2 - horizontal_line_height // 2)
    rect_end_h = (width // 2 + horizontal_line_width // 2, height // 2 + horizontal_line_height // 2)

    # Определяем цвет пикселя в центре кадра.
    central_pixel_color = hsv[height // 2, width // 2]
    hue_value = central_pixel_color[0]
    result_color = None
    if (hue_value < 30 or hue_value > 145):
        result_color = (0, 0, 255)
    elif (hue_value <= 90 and hue_value >= 30):
        result_color = (0, 255, 0)
    else:
        result_color = (255, 0, 0)

    print(hue_value)
    cv2.rectangle(cross_image, rect_start_h, rect_end_h, result_color, -1)
    cv2.rectangle(cross_image, rect_start_v, rect_end_v, result_color, -1)

    # Объединяем оригинальное изображение с изображением с крестиками с помощью взвешенной суммы.
    result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

    cv2.imshow("Colored Cross", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()