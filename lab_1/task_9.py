import cv2

# Создаем объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture("http://172.20.10.2:8080/video")

# Задаем размер окна
cv2.namedWindow("Phone's camera", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Phone's camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Ошибка чтения видео")
        break

cap.release()
cv2.destroyAllWindows()
