import argparse
import yaml
from coordinates_generator import CoordinatesGenerator
from motion_detector import MotionDetector
from colors import COLOR_RED
import logging
import cv2

def main():
    logging.basicConfig(level=logging.INFO)

    cap = cv2.VideoCapture("./videos/sample1.mp4")

    ret, frame = cap.read()
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.imwrite('test111.jpg', frame)


    start_frame = "4000"
    image_file = "./images/parking_lot_1.png"
    test_image = "test111.jpg"
    data_file = "./data/coordinates_1.yml"
    video_file = "./videos/sample1.mp4"
    #video_url = "parking_lot_1.mp4"
    video_url = "C:/Users/YHJ/PycharmProjects/parkingLot/DetectParking-develop/ParkingLot-master/parking_lot/videos/sample1.mp4"



    if image_file is not None:
        with open(data_file, "w+") as points:

            generator = CoordinatesGenerator(test_image, points, COLOR_RED)
            generator.generate()

    with open(data_file, "r") as data:
        points = yaml.load(data , Loader=yaml.FullLoader)
        print(points)
        detector = MotionDetector(video_file, points, int(start_frame))
        detector.detect_motion()
        print("sdfs")

if __name__ == '__main__':
    main()
