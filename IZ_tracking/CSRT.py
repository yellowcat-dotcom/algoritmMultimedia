import cv2
import time
import cv2
print(cv2.__version__)
# загрузка исходного видеофайла
cap = cv2.VideoCapture('video_sources/example_1.mp4')

# читаем первый кадр из видеофайла
ret, frame = cap.read()

# выбор области интереса (ROI) для отслеживания
bbox = cv2.selectROI(frame, False)

# проверка, что пользователь выбрал ROI
if bbox[2] > 0 and bbox[3] > 0:
    # получение ширины и высоты кадра
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # создание объекта cv2.VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_results/output_csrt__4.avi', fourcc, 20.0, (width, height))

    start_time = time.time()
    # чтение видеопотока и отслеживание объектов
    while True:
        # читаем кадр из видеофайла
        ret, frame = cap.read()

        # инициализация трекера CSRT для текущего кадра и ROI
        tracker = cv2.TrackerMIL_create()
        success = tracker.init(frame, bbox)

        if success:
            # обновление трекера на текущем кадре
            success, bbox = tracker.update(frame)

            # отображение ROI на текущем кадре
            if success:
                x, y, w, h = [int(i) for i in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print("Object tracked successfully!")
            else:
                cv2.putText(frame, "Tracking failed!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                print("Tracking failed.")

            # запись текущего кадра в файл
            out.write(frame)

            # отображение текущего кадра
            cv2.imshow('Tracking', frame)

        # выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    # вывод сравнительных характеристик
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
        print(f"Время работы метода CSRT: {end_time - start_time:.5f} секунд")
        print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров/секунду")
        print(f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")
    else:
        print("Видеофайл не содержит кадров.")
    # освобождение ресурсов и закрытие всех окон
    cap.release()
    out.release()
    cv2.destroyAllWindows()
else:
    print("ROI selection canceled. Exiting.")
