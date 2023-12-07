import cv2

# Загрузка предварительно обученного каскада для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Захват видео
cap = cv2.VideoCapture(0)

while True:
    # Считывание кадра
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого (эффективнее для обработки)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Рисование прямоугольников вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Отображение результата
    cv2.imshow('Video', frame)

    # Выход из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
