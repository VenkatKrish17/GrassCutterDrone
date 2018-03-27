import cv2
import numpy as np
import imutils
def pick_image():
    global image;
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
    ret,frame = cap.read() # return a single frame in variable `frame`

    while(True):
        cv2.imshow('img1',frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
            cv2.imwrite('images/c1.png',frame)
            cv2.destroyAllWindows()
            break

    cap.release()

pick_image()

img = cv2.imread('images/c1.png')

cv2.imshow('original',img)
blurred=cv2.medianBlur(img,5)
canned = cv2.Canny(blurred, 200, 300)
cv2.imshow("canned",canned)
image,contours,h = cv2.findContours(canned.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
num_of_triangles=0
num_of_rect=0
for cnt in cntsSorted:
    #print(cnt)
    approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True)
    M = cv2.moments(cnt)

    if(M["m10"]!=0.0 and M["m00"]!=0.0 and M["m01"]!=0.0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        print(len(approx))
        if len(approx)==3:
            num_of_triangles+=1
            cv2.drawContours(img,[cnt],0,(0,255,0),-1)
            cv2.putText(img, "triangle", (cX - 20, cY - 20),
        		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if(num_of_triangles==3):
                break;
        if len(approx)==4:
            num_of_rect+=1
            cv2.drawContours(img,[cnt],0,(0,255,255),-1)
            cv2.putText(img, "rectangle", (cX - 20, cY - 20),
        		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if(num_of_rect==3):
                break;

# circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
