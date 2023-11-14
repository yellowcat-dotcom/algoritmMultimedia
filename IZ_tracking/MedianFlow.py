import cv2
import time

# загрузка исходного видеофайла
cap = cv2.VideoCapture('video_sources/example_4.mp4')

# инициализация MedianFlow Tracker
tracker = cv2.TrackerMedianFlow_create()

# получение ширины и высоты кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# задание начального прямоугольника для отслеживания
_, frame = cap.read()
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

# создание объекта cv2.VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video_results/output_medianflow_4.avi', fourcc, 20.0, (width, height))

start_time = time.time()
# чтение видеопотока и отслеживание объектов
while True:
    # чтение кадра из видеофайла
    ret, frame = cap.read()

    if not ret:
        break

    # обновление трекера и получение нового прямоугольника
    success, bbox = tracker.update(frame)

    # отображение прямоугольника вокруг объекта
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # запись текущего кадра в файл
    out.write(frame)

    # отображение кадра с выделенными объектами
