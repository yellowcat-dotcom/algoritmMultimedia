import cv2
import numpy as np

i = 0

def main(kernel_size, standard_deviation, delta_tresh, min_area):
    global i
    i += 1

    # Открываем видеофайл для чтения
    video = cv2.VideoCapture(r'.\LR5.mp4', cv2.CAP_ANY)

    # Читаем первый кадр и преобразуем его в оттенки серого
    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    # Получаем ширину и высоту видеокадра
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Устанавливаем кодек и создаем объект для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(r'.\output ' + str(i) + '.mp4', fourcc, 144, (w, h))

    while True:
        # Сохраняем старый кадр для вычисления разницы между кадрами
        old_img = img.copy()
        ok, frame = video.read()
        if not ok:
            break

        # Преобразуем текущий кадр в оттенки серого и применяем гауссово
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        # Вычисляем разницу между текущим и предыдущим кадрами
        diff = cv2.absdiff(img, old_img)
        # Бинаризируем разницу: пиксели, превышающие порог delta_tresh, становятся белыми, остальные - черными
        thresh = cv2.threshold(diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        # Находим контуры на бинарном изображении
        (contors, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если площадь контура больше заданного значения min_area, записываем текущий кадр
        for contr in contors:
            area = cv2.contourArea(contr)
            if area < min_area:
                continue
            video_writer.write(frame)

    # Закрываем видеозапись
    video_writer.release()


# Первый набор параметров
kernel_size = 3
standard_deviation = 50
delta_tresh = 60
min_area = 20
main(kernel_size, standard_deviation, delta_tresh, min_area)

# Второй набор параметров
kernel_size = 11
standard_deviation = 70
delta_tresh = 60
min_area = 20
main(kernel_size, standard_deviation, delta_tresh, min_area)

# Третий набор параметров
kernel_size = 3
standard_deviation = 50
delta_tresh = 20
min_area = 20
main(kernel_size, standard_deviation, delta_tresh, min_area)

# Четвертый набор параметров
kernel_size = 3
standard_deviation = 50
delta_tresh = 60
min_area = 10
main(kernel_size, standard_deviation, delta_tresh, min_area)
