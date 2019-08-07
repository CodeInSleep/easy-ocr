import os
import cv2
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils

image = cv2.imread(os.path.join('images', 'warped.jpg'))

def show_image(winname, img, keep=False, wait=True):
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(winname, img)
    if wait:
        cv2.waitKey(0)
    if not keep:
        cv2.destroyWindow(winname)

res = image.copy()
winname = 'image'
show_image(winname, image, keep=True, wait=False)

ref_point = []

class BoxMarker:
    def __init__(self):
        self.areaCnt = 0
        self.areas = []
        self.namesTaken = []

    def addArea(self, start, end, name):
        print('created area %s' % name)
        self.areas.append(DraggedArea(start, end, name))
        self.areaCnt += 1
        self.namesTaken.append(name)
        print(self.areas)

class DraggedArea:
    def __init__(self, start, end, name):
        self.start = start
        self.end = end
        self.name = name

    def draw(self, img):
        cv2.rectangle(img, self.start, \
                self.end, (0, 255, 0))
        return img

    def textSeg(self, img):
        print('[INFO] performing text segmentation...')
        # bouding box for the dragged area
        roi = img[self.start[1]:self.end[1], self.start[0]:self.end[0]]

        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,roi)
        contours, hier = cv2.findContours(roi, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        d = 0
        print('num contours: ', len(contours))
        for ctr in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            print('roi shape: ', roi.shape)
            char_roi = roi[y:y+h, x:x+w]

            winname = 'character: {}'.format(d)

            print(char_roi)
            show_image(winname, char_roi)

            cv2.imwrite('%s_char_%d.png' % (self.name, d), char_roi)

            d += 1
        """
        # extract the Value component from the HSV color space and apply adaptive thresholding
        # to reveal the characters on the license plate
        V = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)

        # resize the license plate region to a canonical size
        roi = imutils.resize(roi, width=400)
        thresh = imutils.resize(thresh, width=400)

        winname = 'text segmentation'
        show_image(winname, thresh)
        # cv2.imwrite('%s_char_%d.png' % (self.name, d), thresh)

        # perform a connected components analysis and initialize the mask to store the locations
        # of the character candidates
        labels = measure.label(thresh, neighbors=8, background=0)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")

        print('num labels: ', len(np.unique(labels)))
        print('thresh shape: ', thresh.shape)
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
 
            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # TODO: figure out how to handle different APIs with
            # different versions of OpenCV
            cnts = cnts[0]


            # ensure at least one contour was found in the mask
            if len(cnts) > 0:
                # grab the largest contour which corresponds to the component in the mask, then
                # grab the bounding box for the contour
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
 
                # compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(roi.shape[0])
 
                # determine if the aspect ratio, solidity, and height of the contour pass
                # the rules tests
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95
 
                # check to see if the component passes all the tests
                if keepAspectRatio and keepSolidity and keepHeight:
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        # clear pixels that touch the borders of the character candidates mask and detect
        # contours in the candidates mask
        charCandidates = segmentation.clear_border(charCandidates)
 
        # TODO:
        # There will be times when we detect more than the desired number of characters --
        # it would be wise to apply a method to 'prune' the unwanted characters
 
        # return the license plate region object containing the license plate, the thresholded
        # license plate, and the character candidates
        """
        return contours
        
        

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def onMouse(event, x, y, flags, bm):
    global ref_point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        areaName = input('name for dragged area: ')
        while areaName in bm.namesTaken:
            areaName = input('name taken.. try another.')
        ref_point.append((x, y))
        bm.addArea(ref_point[0], ref_point[1], areaName) 

bm = BoxMarker()
# draw areas to and label them 
cv2.setMouseCallback(winname, onMouse, bm)

cv2.waitKey(0)

print('created total of %d areas' % bm.areaCnt)
for area in bm.areas:
    res = area.draw(res)

charCands = bm.areas[0].textSeg(image)
winname = 'charCands'
show_image(winname, charCands)

winname = 'result'
show_image(winname, res)
cv2.destroyAllWindows()
