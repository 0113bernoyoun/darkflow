#-*-coding: utf-8

import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE, COLOR_RED

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from datetime import datetime
import socketserver
import socket
import sys
import threading

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.15,
    'gpu': 0.75
}

tfnet = TFNet(option)
f = open("CarData.txt", 'w')
f.close()


rbuff = ''


#result = None
#frame = None
#capture = None
class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []

    def detect_motion(self):
        parking_index = list()
        #video_url = "C:/Users/YHJ/PycharmProjects/parkingLot/DetectParking-develop/ParkingLot-master/parking_lot/videos/parking_lot_1.mp4"
        #capture = open_cv.VideoCapture(video_url)
        #capture = open_cv.VideoCapture(self.video)
        capture = open_cv.VideoCapture(self.video)
        print(self.video)
       # capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)
        #capture.set(open_cv.CAP_PROP_FRAME_WIDTH, 640)
        #capture.set(open_cv.CAP_PROP_FRAME_HEIGHT, 480)
        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        capture = cv2.VideoCapture('sample1.mp4')
        colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
        print("colors : ",colors)
        cnt = 0
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 9876))
        while capture.isOpened():


            #cnt+=1
            cnt = 1
            stime = time.time()
         #   now1 = datetime.now()
           # past_sec = now1.second()
          #  print("ooooo")
            result, frame = capture.read()
            if result:
                results = tfnet.return_predict(frame)
                for color, result in zip(colors, results):
                #for result in results :
                    label = result['label']
                    if cnt == 1:
                    #if now1.second
                        if label == "car" or label == "truck" :
                            tl = (result['topleft']['x'], result['topleft']['y'])
                            br = (result['bottomright']['x'], result['bottomright']['y'])
                            x = (result['bottomright']['x'] - result['topleft']['x']) / 2
                            y = (result['bottomright']['y'] - result['topleft']['y']) / 2
                            x_y = (x,y)
                            now = datetime.now()
                            time_str = "종류 : " + str(label) + "\n시간 : "+ str(now.hour) +"시 " + str(now.minute) + "분 " + str(now.second) + "초\n" +"객체좌표 : " + str(x_y) + "\n\n\n"
                            f = open("CarData.txt", 'a',encoding='utf-8')
                            f.write(time_str)
                            f.close()
                #cnt = 0
            if frame is None:
                break
            if not result:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))

            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.__apply(grayed, index, c)

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds
            global rbuff
            #temp_str = ""
            rbuff = ""
            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)
                if statuses[index]:
                    color = COLOR_GREEN
                    #temp_str = temp_str + str(int(index)+1) + "1"
                    #rbuff = rbuff + str(int(index)+1) + "1"
                    rbuff = rbuff + "0"
                else :
                    color = COLOR_RED
                    #temp_str = temp_str + str(int(index)+1) + "0"
                    #rbuff = rbuff + str(int(index) + 1) + "0"
                    rbuff = rbuff +"1"
                #color = COLOR_GREEN if statuses[index] else COLOR_RED
                sock.send(rbuff.encode(encoding='utf-8'))
                draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)




            open_cv.imshow("hello guys", new_frame)

            print(rbuff)
            k = open_cv.waitKey(20)
            if k == ord("q"):

                break

        capture.release()
        open_cv.destroyAllWindows()


    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        logging.debug("status: %s", status)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass