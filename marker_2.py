# load the image and resize it to a smaller factor so that
# the shapes can be approximated better

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import imutils

class ShapeDetector:
	def __init__(self):
		print("initialized shaped detector")

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.03 * peri, True);print(approx)
        # if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
	# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape
class ColorLabeler:
    def __init__(self):
        print("intialzied color labeler")
        colors = OrderedDict({
			"red": (255, 0, 0),
			"green": (0, 255, 0),
			"blue": (0, 0, 255)})
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self,image,c):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        minDist = (np.inf, None)
        for (i, row) in enumerate(self.lab):
			d=dist.euclidean(row[0],mean)
			if d < minDist[0]:
				minDist=(d,i)
		return self.colorNames[minDist[1]]


def pick_image():
    global image, resized, ratio;
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
    ret,frame = cap.read() # return a single frame in variable `frame`

    while(True):
        cv2.imshow('img1',frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
            cv2.imwrite('images/c1.png',frame)
            cv2.destroyAllWindows()
            break

    cap.release()
    image = cv2.imread('images/shapes.jpg')
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

def startmarking():
    # blur the resized image slightly, then convert it to both
    # grayscale and the L*a*b* color spaces
    pick_image()
    global image, resized, ratio;
    print(ratio)
    canned = cv2.canny(resized, (5, 5), 0)
    cv2.imshow("Image_1",canned)
    gray = cv2.cvtColor(canned, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image_2",gray)
    # lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    # cv2.imshow("Image_3",lab)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    cl = ColorLabeler()
    # compute the center of the contour, then detect the name of the
	# shape using only the contour
    for c in cnts:
        M = cv2.moments(c)
        print(M)
        if(M["m00"]!='' and M["m00"]!= None and M["m00"]!=0.0):
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = sd.detect(c)
            color = cl.label(lab, c)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

startmarking();
