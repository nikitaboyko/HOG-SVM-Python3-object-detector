from detector import ObjectDetector
import numpy as np
import cv2
import argparse

detector_path = r'd:\Users\boykni\Desktop\object_detection\Object-Detector-master\detector.svm'
#annotations_path = r"d:\Users\boykni\Desktop\object_detection\Object-Detector-master\annot.npy"
images_path = r"\\Mos-srv1\Petroview\NAWAT\Completion Data\LA_COMPLETION_REPORTS\imageclassifier\training_dataset\titlepage\3_242294_2_1.jpg"

detector = ObjectDetector(loadPath=detector_path)

imagePath = images_path
image = cv2.imread(imagePath)
detector.detect(image,annotate="LOGO")
