import cv2
import numpy as np
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.cv.CV_EVENT_LBUTTONUP:
        clicked = True

cv2.namedWindow('image capture', cv2.WINDOW_NORMAL)
template=cv2.imread('hospital.png',cv2.IMREAD_GRAYSCALE)
template=cv2.resize(template,(50,100))
#initialize the camera object with VideoCapture
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 10)
sucess, frame = camera.read()

while sucess and cv2.waitKey(1) == -1 and not clicked:
    cv2.imwrite('snapshot.png', frame)
    gray = cv2.imread('snapshot.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image capture', gray)
    gray=cv2.resize(gray,(50,100))
    res=cv2.matchTemplate(gray,template,cv2.TM_CCORR_NORMED)
    #print(res)
    threshold = 0.89
    #print(res)
    loc = np.where( abs(res) >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(gray, pt, (pt[0] + 50, pt[1] + 100), (0,255,0))
        print("found hospital. click to proceed")
        cv2.waitKey()
    sucess, frame = camera.read()


print( 'photo taken press any key to exit')
cv2.waitKey()
cv2.destroyAllWindows()
