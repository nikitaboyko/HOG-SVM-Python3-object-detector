from detector import ObjectDetector
import numpy as np
import argparse

detector_path = r'd:\Users\boykni\Desktop\object_detection\Object-Detector-master\detector.svm'
annotations_path = r"d:\Users\boykni\Desktop\object_detection\Object-Detector-master\annot.npy"
images_path = r"d:\Users\boykni\Desktop\object_detection\Object-Detector-master\images.npy"

print ("[INFO] loading annotations and images")
annots = np.load(annotations_path)
imagePaths = np.load(images_path)

detector = ObjectDetector()
print ("[INFO] creating & saving object detector")

detector.fit(imagePaths,annots,visualize=True,savePath=detector_path)
