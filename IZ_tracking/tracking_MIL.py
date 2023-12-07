import cv2
import time
# Открываем видеофайл
video_path = 'car_sourses/5.mp4'
cap = cv2.VideoCapture(video_path)

# Читаем первый кадр
ret, frame = cap.read()

# Select ROI
bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)

# Ensure a proper bounding box is selected
if bbox[2] > 0 and bbox[3] > 0:
    # Initialize the tracker

    tracker = cv2.TrackerMIL_create()

    tracker.init(frame, bbox)

    # Создаем видеофайл для сохранения результатов
    result_video_path = "MIL_5.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_video = cv2.VideoWriter(result_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    while True:
        # Читаем следующий кадр
        ret, frame = cap.read()
        if not ret:
            break

        # Обновляем трекер и получаем новый bounding box
        success, bbox = tracker.update(frame)

        # Рисуем bounding box на кадре
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Записываем кадр в результаты
        result_video.write(frame)

        # Отображаем результат
        cv2.imshow("Tracking", frame)

        # Выход из цикла, если пользователь нажимает клавишу 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
        print(f"Время работы метода CSRT: {end_time - start_time:.5f} секунд")
        print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров/секунду")
        print(f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")
    else:
        print("Видеофайл не содержит кадров.")

    # Освобождаем ресурсы
    cap.release()
    result_video.release()
    cv2.destroyAllWindows()
else:
    print("Invalid bounding box. Please select a proper ROI.")

