import numpy as np
import cv2


def select_black(image):
    # white color mask
    lower = np.uint8([0, 0, 0])
    upper = np.uint8([50, 50, 50])
    black_mask = cv2.inRange(image, lower, upper)
    return black_mask
def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)
def select_region(image,camera):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    if(camera=='front'):
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.1, rows*0.95]
        top_left     = [cols*0.4, rows*0.6]
        bottom_right = [cols*0.9, rows*0.95]
        top_right    = [cols*0.6, rows*0.6]
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return filter_region(image, vertices)
    if(camera=='bottom'):
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.3, rows]
        top_left     = [cols*0.3, 0]
        bottom_right = [cols*0.7, rows]
        top_right    = [cols*0.7, 0]
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return filter_region(image, vertices)
def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #w,h=frame.shape[::-1]
    #gray = cv2.Canny(frame,100,100)

    # Display the resulting frame
    black=select_black(frame)
    blurred=cv2.GaussianBlur(black, (5, 5), 0)
    edges = detect_edges(blurred)
    selected = select_region(edges,'front')
    hough=hough_lines(selected)
    if(len(hough)!=0):
        for x1,y1,x2,y2 in hough[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),4)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
