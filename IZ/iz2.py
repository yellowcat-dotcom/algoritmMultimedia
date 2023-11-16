import cv2
import numpy as np
import time


def svertka(img, kernel):
    kernel_size = len(kernel)
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    matr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]

    for i in range(x_start, len(matr)-x_start):
        for j in range(y_start, len(matr[i])-y_start):

            # операция свёртки
            val = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    val += img[i + k][j + l] * kernel[k +
                                                      (kernel_size//2)][l + (kernel_size//2)]
            matr[i][j] = val

    return matr


def get_angle_number(x, y):
    tg = y/x if x != 0 else 999

    if (x < 0):
        if (y < 0):
            if (tg > 2.414):
                return 0
            elif (tg < 0.414):
                return 6
            elif (tg <= 2.414):
                return 7
        else:
            if (tg < -2.414):
                return 4
            elif (tg < -0.414):
                return 5
            elif (tg >= -0.414):
                return 6
    else:
        if (y < 0):
            if (tg < -2.414):
                return 0
            elif (tg < -0.414):
                return 1
            elif (tg >= -0.414):
                return 2
        else:
            if (tg < 0.414):
                return 2
            elif (tg < 2.414):
                return 3
            elif (tg >= 2.414):
                return 4


i = 0

def task(path, standard_deviation, kernel_size, bound_path):
    start_time = time.time()
    global i
    i += 1
    # Задание 1
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # считывание изображения
    imgBlurByCV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation) # применение фильтрации для подавления шумов

    #cv2.imshow("task_1", imgBlurByCV2)

    # Задание 2

    # # 1) Зададим матрицы оператора Собеля
    Gx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

    Gy = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]


    #Оператор Прюитта
    # Gx = [[-1, 0, 1],
    #       [-1, 0, 1],
    #       [-1, 0, 1]]
    #
    # Gy = [[-1, -1, -1],
    #       [0, 0, 0],
    #       [1, 1, 1]]

    # Оператор Щарра:
    # Gx = [[-3, 0, 3],
    #      [-10, 0, 10],
    #      [-3, 0, 3]]
    #
    # Gy = [[-3, -10, -3],
    #      [0,   0,  0],
    #      [3,  10,  3]]

    # 2 а) Применяем операторы свертки
    img_Gx = svertka(img, Gx)
    img_Gy = svertka(img, Gy)

    # Создаем копию матрицы
    matr_gradient = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_gradient[i][j] = img[i][j]

    # 2 b) Находим длину вектора градиента
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_gradient[i][j] = np.sqrt(img_Gx[i][j] ** 2 + img_Gy[i][j] ** 2)

# находим матрицу углов
    img_angles = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_angles[i][j] = get_angle_number(img_Gx[i][j], img_Gy[i][j])

    # вывод матриц конец 2 задания
    img_gradient_to_print = img.copy()
    max_gradient = np.max(matr_gradient)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gradient_to_print[i][j] = (
                float(matr_gradient[i][j])/max_gradient)*255
    #print("Матрица значений длинн\n",img_gradient_to_print, '\n')
    #показывает как быстро меняется
    #cv2.imshow('img_gradient_to_print ' + str(i), img_gradient_to_print)

    img_angles_to_print = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_angles_to_print[i][j] = img_angles[i][j]/7*255
    #print("Матрица значений углов\n", img_angles_to_print)
    #направление изменения яркости
    #cv2.imshow('img_angles_to_print ' + str(i), img_angles_to_print)

    # 3 подавление немаксимумов
    img_border_not_filtered = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            angle = img_angles[i][j]
            gradient = matr_gradient[i][j]
            if (i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1):
                img_border_not_filtered[i][j] = 0
            else:
                x_shift = 0
                y_shift = 0
                if (angle == 0 or angle == 4):
                    x_shift = 0
                elif (angle > 0 and angle < 4):
                    x_shift = 1
                else:
                    x_shift = -1

                if (angle == 2 or angle == 6):
                    y_shift = 0
                elif (angle > 2 and angle < 6):
                    y_shift = -1
                else:
                    y_shift = 1

                is_max = gradient >= matr_gradient[i+y_shift][j + x_shift] and gradient >= matr_gradient[i-y_shift][j-x_shift]
                img_border_not_filtered[i][j] = 255 if is_max else 0
    #cv2.imshow('после подавления немаксимумов ' + str(i), img_border_not_filtered)

    # 4
    # находим верхние и нижние границы
    # если длина больше максимума - граница, если меньше минимума - не граница
    # то что посередине - область неоднозначности. рядом с границей - должна быть граница
    lower_bound = max_gradient/bound_path
    upper_bound = max_gradient - max_gradient/bound_path

    print('lower_bound', lower_bound, " ", "upper_bound", upper_bound)


    img_border_filtered = np.zeros(img.shape)


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradient = matr_gradient[i][j]
            if (img_border_not_filtered[i][j] == 255):
                if (gradient >= lower_bound and gradient <= upper_bound):
                    flag = False

                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (flag):
                                break
                            if (img_border_not_filtered[i+k][j+l] == 255 and matr_gradient[i+k][j+l] >= lower_bound):
                                flag = True
                                break
                    if (flag):
                        img_border_filtered[i][j] = 255
                elif (gradient > upper_bound):
                    img_border_filtered[i][j] = 255
    end_time = time.time()  # Запоминаем текущее время после окончания выполнения операций
    print(f"Время выполнения для изображения {i}: {end_time - start_time:.2f} секунд")

    cv2.imshow('Keny' + str(i), img_border_filtered)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# 5
task('insult.jpg', 1, 3, 15)