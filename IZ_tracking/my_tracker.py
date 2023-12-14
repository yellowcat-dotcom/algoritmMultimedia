import cv2
import numpy as np
import time


class CAMShiftTracker(object):

    def __init__(self, curWindowRoi, imgBGR):
        self.updateCurrentWindow(curWindowRoi)
        self.updateHistograms(imgBGR)

        # установите критерии завершения
        self.term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def updateCurrentWindow(self, curWindowRoi):
        self.curWindow = curWindowRoi

    def updateHistograms(self, imgBGR):
        # Извлекаем BGR-подобласть
        self.bgrObjectRoi = imgBGR[self.curWindow[1]: self.curWindow[1] + self.curWindow[3], self.curWindow[0]: self.curWindow[0] + self.curWindow[2]]

        # BGR в HSV
        self.hsvObjectRoi = cv2.cvtColor(self.bgrObjectRoi, cv2.COLOR_BGR2HSV)

        # Создаем маску для определения области объекта
        self.mask = cv2.inRange(self.hsvObjectRoi, np.array((0., 50., 50.)), np.array((180, 255., 255.)))

        # Вычисляем гистограмму
        self.histObjectRoi = cv2.calcHist([self.hsvObjectRoi], [0], self.mask, [180], [0, 180])

        # Нормализуем
        cv2.normalize(self.histObjectRoi, self.histObjectRoi, 0, 255, cv2.NORM_MINMAX)

    def getBackProjectedImage(self, imgBGR):

        # BGR в HSV
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

        # Создаем обратно проецированное изображение с использованием гистограммы

        backProjectedImg = cv2.calcBackProject([imgHSV], [0], self.histObjectRoi, [0, 180], 1)

        self.backProjectedImg = backProjectedImg
        return backProjectedImg.copy()

    def computeNewWindow(self, imgBGR):

        self.getBackProjectedImage(imgBGR)

        # для получения нового положения отслеживаемого окна
        self.rotatedWindow, curWindow = cv2.CamShift(self.backProjectedImg, self.curWindow, self.term_criteria)

        # Получаем вершины нового окна
        self.rotatedWindow = cv2.boxPoints(self.rotatedWindow)
        self.rotatedWindow = np.int0(self.rotatedWindow)

        self.updateCurrentWindow(curWindow)

    # получение текущего окна
    def getCurWindow(self):
        return self.curWindow


video_path = 'car_sourses/5.mp4'
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(r"MyShift_5.mp4", fourcc, 90, (w, h))

ok, frame = cap.read()

# начальное окно от пользователя
bbox = cv2.selectROI(frame, False)

camShifTracker = CAMShiftTracker(bbox, frame)

start_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    timer = cv2.getTickCount()

    # вычисляем новое окно
    camShifTracker.computeNewWindow(frame)

    # кадры в секунду
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # координаты нового
    x, y, w, h = camShifTracker.getCurWindow()

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # показ и запись
    cv2.imshow("CAMShift Face Tracking", frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода MOG2: {end_time - start_time:.5f} секунд")
    print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров/секунду")
    print(
        f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")
else:
    print("Видеофайл не содержит кадров.")
writer.release()
