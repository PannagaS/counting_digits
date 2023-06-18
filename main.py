import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise #used for distance calculation

#defining global vars
background = None
accumulated_weight = 0.5

#roi corners
roi_left = 600 #x1
roi_top = 60 #y1
roi_right = 300 #x2
roi_bottom = 360 #y2

#function to find avg background value
def calc_accum_avg(frame, accumulated_weight):
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)


#Segmeting hand frame in ROI
def segment(frame,threshold=25):
    #calculate the absolute difference between the bg and the passed frame
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    #apply threshold to the diff image
    ret, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY) #image,min_threshold, max_threshold,type of thresholding
    
    #grabbing the external contours from the image
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    else:
        #assuming the largest external contour in ROI is the hand
        hand_segment = max(contours,key=cv2.contourArea)
        
        return (thresholded, hand_segment)
    

#counting the number of fingers
def count_fingers(thresholded, hand_segment):
    
    conv_hull = cv2.convexHull(hand_segment)
    
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    #center
    cX = (left[0] + right[0])//2 #indexed as 0 because we want x coordinate
    cY = (top[1] + bottom[1])//2 #indexed as 1 because we want y coordinate
    
    #using pairwise to calculate distance
    distance = pairwise.euclidean_distances([(cX,cY)], Y=[left, right, top, bottom])[0]
    
    max_distance = distance.max()
    
    radius = int(0.78*max_distance)
    circumference = 2*np.pi*radius
    
    #creating ROI for that circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
    
    cv2.circle(circular_roi, (cX,cY), radius, 255, 10)
    
    #using bitwise-and with the circular roi as a mask
    circular_roi =  cv2.bitwise_and(thresholded,thresholded, mask=circular_roi)
    
    #now, we'll find all the contours from the above image
    image, contours, heirarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #actually counting the number of points outside the circle
    count = 0
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        
        #we want to make sure the contour region is not at the very bottom of the hand
        #i.e., we dont want to count wrist contour
        
        out_of_wrist = (cY + (cY*0.25)) > (y+h)
        
        #we should also make sure the number of points belonging to contour does not exceed 25% of the circumference of the circular_roi
        #otherwise we're counting points outside of the hand itself! this is not the goal
        
        limit_points = ((circumference*0.25) > cnt.shape[0])
        
        if out_of_wrist and limit_points:
            #if this condition is satisfied that means we're counting fingers
            count+=1
    return count


#putting it altogether, i.e., couting fingers by running video stream!
cam = cv2.VideoCapture(1)

num_frames = 0

#for the first 30 to 60 frames we'll calculate background avg
while True:
    ret, frame = cam.read()
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7),0) #to average out the noise
    if num_frames <60:
        calc_accum_avg(gray,accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy, 'Wait, getting background now!', (200,300),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
            cv2.imshow('Finger Count', frame_copy)
    else:
        #waiting for someone to enter the hand in the roi
        hand = segment(gray)
        
        
        if hand is not None:
            #ie if we are detecting the hand
            
            thresholded, hand_segment = hand
            
            #draws contours around real hand in live stream
            cv2.drawContours(frame_copy, [hand_segment+(roi_right,roi_top)],-1,(0,0,255),3)
            
            fingers = count_fingers(thresholded, hand_segment)
            
            cv2.putText(frame_copy, str(fingers), (70,50),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),2)
            
            cv2.imshow('Thresholded', thresholded) #this helps in debugging, basically like adjusting bg, thresholds, etc.
            
    
    cv2.rectangle(frame_copy,(roi_left, roi_top), (roi_right, roi_bottom), (0,255,0),2)
    
    num_frames +=1
    
    cv2.imshow('Finger Count', frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27: 
        break
        
cam.release()
cv2.destroyAllWindows()
            

