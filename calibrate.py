import os
import sys
import cv2
import numpy as np

rows = 9
cols = 6
patternSize = (rows, cols)

img_dir = 'images'
# calibrate picamera, finding the intrinsic parameters of camera 
cam = cv2.VideoCapture(0)

ret, frame = cam.read()
if not ret:
    print('camera did not open..')
    sys.exit(1)

retval, corners = cv2.findChessboardCorners(frame, patternSize)

if retval != 0:
    print('detected corners..')
    print(np.array(corners).shape)

cv2.drawChessboardCorners(frame, patternSize, corners, retval)
img_name = os.path.join(img_dir, "chess.png")
cv2.imwrite(img_name, frame)
print("{} written!".format(img_name))




cam.release()

cv2.destroyAllWindows()
