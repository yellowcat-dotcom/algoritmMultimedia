import cv2

# Путь к изображению
image_path = r"C:\Users\valen\PycharmProjects\algoritmMultimedia\img\imgg.png"

# Чтение изображения
img = cv2.imread(image_path)

if img is None:
    print("Ошибка при чтении изображения.")
    exit()

# Преобразование изображения в формат HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Создание окон с одинаковыми размерами
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 400, 400)

cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Image", 400, 400)

# Отображение исходного изображения
cv2.imshow("Original Image", img)

# Отображение изображения в формате HSV
cv2.imshow("HSV Image", img_hsv)

# Ожидание нажатия клавиши и закрытие окон при нажатии
cv2.waitKey(0)
cv2.destroyAllWindows()
