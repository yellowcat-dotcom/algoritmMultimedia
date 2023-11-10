import cv2

video_path = r"C:\Users\valen\PycharmProjects\algoritmMultimedia\video\video.mp4"

# Открытие видеопотока
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Ошибка при открытии видеофайла или камеры.")
    exit()

while True:
    # ret - это bool, был ли успешно прочитан кадр из видеопотока.
    # frame - кадр видеопотока
    ret, frame = cap.read()

    if not ret:
        print("Конец видеопотока.")
        break

    frame = cv2.resize(frame, (400, 600))
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Отображение кадра в окне
    cv2.imshow("Gray Video", gray_frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

# Закрытие видеопотока и окон
cap.release()
cv2.destroyAllWindows()