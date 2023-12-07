import cv2
import numpy as np
class CAMShiftTracker(object):

    def __init__(self, curWindowRoi, imgBGR):
        '''
        curWindow =[x,y, w,h] // initialize the window to be tracked by the tracker
        '''
        self.updateCurrentWindow(curWindowRoi)
        self.updateHistograms(imgBGR)

        # set up the termination criteria for meanshift, either 10 iterations or move by at least 1 pt
        self.term_criteria = (cv2.TERM_CRITERIA_EPS |
                              cv2.TERM_CRITERIA_COUNT, 10, 1)

    def updateCurrentWindow(self, curWindowRoi):
        self.curWindow = curWindowRoi

    def updateHistograms(self, imgBGR):
        '''
          update the histogram and rois according to the current object in the current image

        '''

        self.bgrObjectRoi = imgBGR[self.curWindow[1]: self.curWindow[1] + self.curWindow[3],
                            self.curWindow[0]: self.curWindow[0] + self.curWindow[2]]
        self.hsvObjectRoi = cv2.cvtColor(self.bgrObjectRoi, cv2.COLOR_BGR2HSV)

        # get the mask for calculating histogram and also remove some noise
        self.mask = cv2.inRange(self.hsvObjectRoi, np.array(
            (0., 50., 50.)), np.array((180, 255., 255.)))

        # use 180 bins for each H value, and normalize the histogram to lie b/w [0, 255]
        self.histObjectRoi = cv2.calcHist(
            [self.hsvObjectRoi], [0], self.mask, [180], [0, 180])
        cv2.normalize(self.histObjectRoi, self.histObjectRoi,
                      0, 255, cv2.NORM_MINMAX)

    def getBackProjectedImage(self, imgBGR):
        '''
           convert the current BGR image, imgBGR, to HSV color space
           and return the backProjectedImg
        '''
        # print("[info] getBackprjectImage calls", imgBGR.shape)
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

        # obtained the back projected image using the histogram obtained earlier

        backProjectedImg = cv2.calcBackProject(
            [imgHSV], [0], self.histObjectRoi, [0, 180], 1)

        self.backProjectedImg = backProjectedImg

        return backProjectedImg.copy()

    def computeNewWindow(self, imgBGR):
        '''
            Track the window enclosing the object of interest using CAMShift function of openCV for the
            current frame imgBGR
        '''

        self.getBackProjectedImage(imgBGR)

        self.rotatedWindow, curWindow = cv2.CamShift(
            self.backProjectedImg, self.curWindow, self.term_criteria)

        # get the rotated windo vertices

        self.rotatedWindow = cv2.boxPoints(self.rotatedWindow)
        self.rotatedWindow = np.int0(self.rotatedWindow)

        self.updateCurrentWindow(curWindow)

    def getCurWindow(self):
        return self.curWindow

    def getRotatedWindow(self):
        return self.rotatedWindow


def iz_part2(file, bbox):
    video_path = 'my_car/2.mp4'
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"IZ_tracking" + file +
                             "MeanShift.mp4", fourcc, 90, (w, h))

    ok, frame = cap.read()

    bbox = cv2.selectROI(frame, False)
    camShifTracker = CAMShiftTracker(bbox, frame)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        camShifTracker.computeNewWindow(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        x, y, w, h = camShifTracker.getCurWindow()

        # display the current window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)

        rotatedWindow = camShifTracker.getRotatedWindow()
        # display rotated window
        cv2.polylines(frame, [rotatedWindow], True,
                      (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # show the frame and update the FPS counter
        cv2.imshow("CAMShift Face Tracking", frame)
        writer.write(frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    writer.release()

files = ['my_car/1.mp4', 'my_car/3.mp4']  #
bboxs = [(250, 123, 156, 218), (587, 338, 352, 502)]



for file in files:
    bbox = bboxs[files.index(file)]
    iz_part2(file, bbox)