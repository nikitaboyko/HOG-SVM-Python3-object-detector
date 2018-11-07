import numpy as np
import cv2
import argparse
from imutils.paths import list_images
import os
from tqdm import tqdm

class BoxSelector(object):
    def __init__(self, image, window_name,color=(0,0,255)):
        #store image and an original copy
        self.image = image
        self.orig = image.copy()

        #capture start and end point co-ordinates
        self.start = None
        self.end = None

        #flag to indicate tracking
        self.track = False
        self.color = color
        self.window_name = window_name

        #hook callback to the named window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,self.mouseCallBack)

    def mouseCallBack(self, event, x, y, flags, params):
        #start tracking if left-button-clicked down
        if event==cv2.EVENT_LBUTTONDOWN:
            self.start = (x,y)
            self.track = True

        #capture/end tracking while mouse-move or left-button-click released
        elif self.track and (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONUP):
            self.end = (x,y)
            if not self.start==self.end:
                self.image = self.orig.copy()
                #draw rectangle on the image
                cv2.rectangle(self.image, self.start, self.end, self.color, 2)
                if event==cv2.EVENT_LBUTTONUP:
                    self.track=False

            #in case of clicked accidently, reset tracking
            else:
                self.image = self.orig.copy()
                self.start = None
                self.track = False
            cv2.imshow(self.window_name,self.image)

    @property
    def roiPts(self):
        if self.start and self.end:
            pts = np.array([self.start,self.end])
            s = np.sum(pts,axis=1)
            (x,y) = pts[np.argmin(s)]
            (xb,yb) = pts[np.argmax(s)]
            return [(x,y),(xb,yb)]
        else:
            return []


#parse arguments

dataset_path = r'd:\Users\boykni\Desktop\object_detection\Object-Detector-master\dataset'
annotations_path = r"d:\Users\boykni\Desktop\object_detection\Object-Detector-master\annot.npy"
images_path = r"d:\Users\boykni\Desktop\object_detection\Object-Detector-master\images.npy"

# ap = argparse.ArgumentParser()
# ap.add_argument("-d","--dataset",required=True,help="path to images dataset...")
# ap.add_argument("-a","--annotations",required=True,help="path to save annotations...")
# ap.add_argument("-i","--images",required=True,help="path to save images")
# args = vars(ap.parse_args())
print('hi')
#annotations and image paths
annotations = []
imPaths = []
#loop through each image and collect annotations

for im in tqdm(os.listdir(dataset_path)):
    imagePath = os.path.join(dataset_path,im)
    #load image and create a BoxSelector instance
    image = cv2.imread(imagePath)
    bs = BoxSelector(image,"Image")
    cv2.imshow("Image",image)
    cv2.waitKey(0)

    #order the points suitable for the Object detector
    pt1,pt2 = bs.roiPts
    (x,y,xb,yb) = [pt1[0],pt1[1],pt2[0],pt2[1]]
    annotations.append([int(x),int(y),int(xb),int(yb)])
    imPaths.append(imagePath)

#save annotations and image paths to disk
annotations = np.array(annotations)
imPaths = np.array(imPaths,dtype="unicode")
np.save(annotations_path, annotations)
np.save(images_path,imPaths)
