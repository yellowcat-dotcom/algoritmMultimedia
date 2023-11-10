import cv2
import numpy as np


# Функция для создания ядра Гаусса
def gauss(x, y, omega, a, b):
    omega2 = 2 * (omega ** 2)
    m1 = 1 / (np.pi * omega2)
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / omega2)
    return m1 * m2


# Функция для применения гауссова размытия к изображению
def GaussianBlur(img, kernel_size, standard_deviation):
    # Создаем пустое ядро
    kernel = np.ones((kernel_size, kernel_size))
    a = b = (kernel_size + 1) // 2

    # Заполняем ядро значениями, вычисленными с помощью функции gauss
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)

    print('before normalization')
    print(kernel)

    # Нормализуем ядро (сумма всех элементов должна быть равна 1)
    sum = np.sum(kernel)
    kernel /= sum

    print('after normalization')
    print(kernel)

    # Создаем копию изображения
    imgBlur = img.copy()

    # Операция свертки между изображением и ядром
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    for i in range(x_start, imgBlur.shape[0] - x_start):
        for j in range(y_start, imgBlur.shape[1] - y_start):
            # Операция свертки
            val = 0
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    val += img[i + k, j + l] * kernel[k + (kernel_size // 2), l + (kernel_size // 2)]
            imgBlur[i, j] = val

    return imgBlur


# Основная функция для вызова гауссова размытия
def BlurFuss():
    # Считываем изображение
    img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

    # Стандартное отклонение и размер ядра фильтра
    standard_deviation = 1
    kernel_size = 5

    # Применяем гауссово размытие
    imgBlur1 = GaussianBlur(img, kernel_size, standard_deviation)
    cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur1)

    # Другие параметры размытия
    standard_deviation = 1
    kernel_size = 7

    # Применяем гауссово размытие с новыми параметрами
    imgBlur2 = GaussianBlur(img, kernel_size, standard_deviation)
    cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur2)

    # Применяем гауссово размытие с использованием OpenCV
    imgBlurOpenCV = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    cv2.imshow('img', img)
    cv2.imshow('OpenCV_blur', imgBlurOpenCV)
    cv2.waitKey(0)


# Вызываем основную функцию
BlurFuss()
