import cv2

img1_png = cv2.imread(r"C:\Users\valen\PycharmProjects\algoritmMultimedia\img\imgg.png", cv2.IMREAD_COLOR)  #цветное изображение
img2_jpg = cv2.imread(r"C:\Users\valen\PycharmProjects\algoritmMultimedia\img\img_1.jpg", cv2.IMREAD_GRAYSCALE) #оттенки серого
img3_jpeg = cv2.imread(r"C:\Users\valen\PycharmProjects\algoritmMultimedia\img\img.jpeg", cv2.IMREAD_REDUCED_COLOR_8) #8-битным цветовым пространством.

cv2.imshow("1", img1_png)  # с цветным изображением

cv2.namedWindow("2", cv2.WINDOW_NORMAL)  # с возможностью изменения размера
cv2.imshow("2", img2_jpg)

cv2.namedWindow("3", cv2.WINDOW_AUTOSIZE)  # без изменения размера
cv2.imshow("3", img3_jpeg)


cv2.waitKey(0)
cv2.destroyAllWindows()
