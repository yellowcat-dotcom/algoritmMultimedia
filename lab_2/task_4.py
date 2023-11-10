import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    # открытия (размытие расширение)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # замыкание (расширение размытие)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    moments = cv2.moments(mask)
    area = moments['m00']/200
    #print("Area: ", area)

    if area > 0:
        # Вычисляем размеры и центр масс красной области
        width = height = int(np.sqrt(area))
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        # Рисуем прямоугольник вокруг красной области
        cv2.rectangle(frame,
                      (c_x - (width // 2), c_y - (height // 2)),
                      (c_x + (width // 2), c_y + (height // 2)),
                      (0, 0, 0), 2)

    cv2.imshow('Rectanle_frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


#прежде нахождения моментов открытие, потом закрытие