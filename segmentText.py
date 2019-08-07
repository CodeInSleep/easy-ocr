import os
import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, RIL
from utils import show_image
from PIL import Image

def getContours(fname):
    img = cv2.imread(fname, 0)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 75, 10)

    orig = img.copy()

    print('orig image size: ', img.shape)

    img = Image.fromarray(img)

    with PyTessBaseAPI() as api:
        api.SetImage(img)
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        
        for i, (im, box, _, _) in enumerate(boxes):
            # im is a PIL image object
            # box is a dict with x, y, w and h keys

            # api.SetRectangle(box['x'], box['y'], box['w'], box['h'])cv2.rectangle(img, self.start, \
                # self.end, (0, 255, 0))
            print(box)
            cv2.rectangle(orig, (box['x'], box['y']), \
                (box['x']+box['w'], box['y']+box['h']), (0, 255, 0))
            # ocrResult = api.GetUTF8Text()
            # conf = api.MeanTextConf()
            # print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
            #       "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))

            # turn PIL image type to openCV image (numpy array)
            im = np.array(im)

            # get canny edges
            im = cv2.Canny(im, 30, 200)
            show_image('canny', im)

            print('im shape: ', im.shape)
            contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

            print('num contours: ', len(contours))
            epsilon = 3
            contour_poly = []
            for cont in contours:
                contour_poly.append(cv2.approxPolyDP(cont, epsilon, True))

            bounding_rects = []
            for poly in contour_poly:
                bounding_rects.append(cv2.boundingRect(poly))

            filtered_rects = []
            for rect in bounding_rects:
                x, y, w, h = rect
                if w > 5 and h > 5:
                    filtered_rects.append(rect)
                    cv2.rectangle(orig, (box['x']+x, box['y']+y), 
                        (box['x']+x+w, box['y']+y+h), (0, 0, 255))
            
            print('rects: ', filtered_rects)
    show_image('orig', orig)    
    

getContours(os.path.join('images', 'textseg1.png'))

