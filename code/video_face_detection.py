# -*- coding: utf-8 -*-
"""

@author: esinb
"""
# import required libraries
import cv2 
import face_recognition

# capture video from default camera (use 1, 2 for additional ones)
video_stream = cv2.VideoCapture('images/testing/biden.mp4')

# initialize the variable to hold all face locations in the frame
all_face_locations = []

while True:
    # get current frame
    ret, current_frame = video_stream.read()
    
    # resize the current frame to 1/4 so that the computer can process faster
    current_frame_small = cv2.resize(current_frame, 
                                     (0, 0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(
                                        current_frame_small,
                                        number_of_times_to_upsample=2,
                                        model='hog')
    #loop through faces
    for index, current_face_location in enumerate(all_face_locations):
        # splitting the tuple to get the four position values
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        # cgange the position magnitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        # draw a rectangle around the face detected
        cv2.rectangle(current_frame, (left_pos,top_pos),
                      (right_pos,bottom_pos),(0,0,255), 2) # 2 is for thickness
        
    # show the current face with rectangle drawn
    cv2.imshow("Webcam Video", current_frame)
        
    # Press 'q' on the keyboard to break the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# release the webcam resource
video_stream.release()
# close all opencv windows open
cv2.destroyAllWindows()
        
        
        
        
        
        