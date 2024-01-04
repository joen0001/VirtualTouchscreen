import cv2
import numpy as np

# setup variables
lower_thresh = 170
upper_thresh = 255
min_contour_area = 40
dist_thres = 200

capture = cv2.VideoCapture(1)

while True:
    ret, frame = capture.read()
    
    if not ret:
        break
    
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # apply binary thresholding
    ret, img_thres = cv2.threshold(blur, lower_thresh, upper_thresh, cv2.THRESH_BINARY_INV)
    
    # find contours
    contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over all detected contours
    for contour in contours:
        # calculate area of contour
        area = cv2.contourArea(contour)
        
        # minimum size of contour to avoid picking up random contours
        if area > min_contour_area:
            # calculate centroid of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # calculate distance to furthest point
            dist_max = 0
            max_point = None
            for point in contour:
                x, y = point[0]
                dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                if dist > dist_max and dist < dist_thres: # furthest point has to be a minimum distance to be counted as a outstretched finger
                    if x > 0 and x < frame.shape[1] and y > 0 and y < frame.shape[0]:
                        dist_max = dist
                        max_point = point
            
            # draw contour and centroid
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # draw furthest point if found
            if max_point is not None:
                x, y = max_point[0]
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
    # display output
    cv2.imshow("Output", frame)
    # cv2.imshow("Binary", img_thres)
    
    # check for exit 'esc' key
    if cv2.waitKey(1) == 27:
        break
        
# release resources
capture.release()
cv2.destroyAllWindows()

# TO DO
# Use section of contour that is on edge as reference, shouldn't be included as the furthest point from centroid
# Right now, its removing the last edge of the contour but only by one pixel 'x < frame.shape[1]'
# Thus still sometimes showing as the furthest from centroid at the edge of image

# Can't assume that the finger is always pointing inwards

# Assume the arm/body is always blocking one edge of screen
# Make it so the fingertip point cannot be within ?? pixels of that edge?