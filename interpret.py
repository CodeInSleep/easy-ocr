import argparse
import os
import cv2
import numpy as np
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="Path to the image to be interpreted")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
boxes = np.load('boxes.npy')

params = None
with open('params.pickle', 'rb') as handle:
    params = pickle.load(handle)

rW = params["rW"]
rH = params["rH"]

for idx, (startX, startY, endX, endY) in enumerate(boxes):
   startX = int(startX*rW)
   startY = int(startY*rH)
   endX = int(endX*rW)
   endY = int(endY*rH)

   cv2.rectangle(image, (startX, startY), (endX, endY),
           (0, 255, 0), 2)

   cv2.putText(image, str(idx), (startX, startY), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

cv2.imwrite(os.path.join('images', 'marked.jpg'), image)
