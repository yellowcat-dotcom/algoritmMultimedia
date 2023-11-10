import cv2
import numpy as np


def task1():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cv2.imshow('hsv_frame', hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task2():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define the range of red color in HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask1)

        # Display the result
        cv2.imshow('image', res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task3():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define the range of red color in HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

        erosion = cv2.erode(mask1, kernel, iterations=1)
        dilation = cv2.dilate(mask1, kernel, iterations=1)

        cv2.imshow("Erosion", erosion)
        cv2.imshow("Dilation", dilation)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task4():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        moments = cv2.moments(mask)
        area = moments['m00']
        print("Area: ", area)

        if area > 0:
            width = height = int(np.sqrt(area))
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])
            cv2.rectangle(frame,
                          (c_x - (width // 8), c_y - (height // 8)),
                          (c_x + (width // 8), c_y + (height // 8)),
                          (0, 0, 0), 2)

        cv2.imshow('Rectanle_frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# task1()
# task2()
# task3()
task4()
