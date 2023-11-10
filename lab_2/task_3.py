import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    # открытия (размытие расширение)
    opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

    # замыкание (расширение размытие)
    closing = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    # эрозия (размытие) нужны все единицы
    erosion = cv2.erode(mask1, kernel, iterations=1)

    # расширение хотябы одна
    dilation = cv2.dilate(mask1, kernel, iterations=1)

    cv2.imshow("Erosion", erosion)
    cv2.imshow("Dilation", dilation)
    cv2.imshow("Opening", opening)
    cv2.imshow("Closing", closing)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()