import cv2
import numpy as np
import time

def roberts(path, standard_deviation, kernel_size, threshold):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # считывание изображения
    # Применение фильтра Гаусса для сглаживания изображения
    smoothed_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    # Размеры изображения
    height, width = smoothed_image.shape

    # Оператор Робертса для вычисления градиента
    roberts_x = np.array([[1, 0],
                          [0, -1]])

    roberts_y = np.array([[0, 1],
                           [-1, 0]])

    # Создание пустого изображения для сохранения результата градиента
    gradient_magnitude = np.zeros_like(smoothed_image)

    # Измерение времени начала выполнения операций
    start_time = time.time()

    # Применение оператора Робертса вручную
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gradient_x = np.sum(smoothed_image[i-1:i+1, j-1:j+1] * roberts_x)
            gradient_y = np.sum(smoothed_image[i-1:i+1, j-1:j+1] * roberts_y)
            gradient_magnitude[i, j] = np.sqrt(gradient_x**2 + gradient_y**2)

    # Измерение времени окончания выполнения операций
    end_time = time.time()

    # Применение пороговой фильтрации для выделения границ
    threshold = threshold
    edges = (gradient_magnitude > threshold) * 255

    edges = edges.astype(np.uint8)

    # Вывод времени выполнения
    print("Время выполнения: {} секунд".format(end_time - start_time))

    # Вывод результата
    cv2.imshow('output_edges_roberts.jpg', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

roberts('multiple_sclerosis.jpg',1,3,10)

