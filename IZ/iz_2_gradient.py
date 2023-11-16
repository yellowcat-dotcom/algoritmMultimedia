import cv2
import numpy as np

def gradient_edge_detection(image):
    # Вычисление градиента в горизонтальном и вертикальном направлениях
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)

    # Вычисление градиента в горизонтальном направлении
    gradient_x[:, 1:-1] = image[:, 2:] - image[:, :-2]

    # Вычисление градиента в вертикальном направлении
    gradient_y[1:-1, :] = image[2:, :] - image[:-2, :]

    # Вычисление амплитуды градиента
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Применение пороговой фильтрации для выделения контуров
    _, edges = cv2.threshold(gradient_magnitude, 146, 200, cv2.THRESH_BINARY)

    return edges.astype(np.uint8)

# Загрузка изображения в оттенках серого
image = cv2.imread('insult.jpg', cv2.IMREAD_GRAYSCALE)

# Применение градиентного метода выделения контуров
edges = gradient_edge_detection(image)

cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохранение изображения с выделенными контурами
#cv2.imwrite('output_image_with_edges.jpg', edges)
