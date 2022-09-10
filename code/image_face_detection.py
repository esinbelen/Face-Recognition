# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 15:43:30 2022

@author: esinb
"""
# import required libraries
import cv2 
import face_recognition

# load an image to detect
image_to_detect = cv2.imread('images/two_people.jpg')

# show image
cv2.imshow('image', image_to_detect) # Show the image, note that the name of the output window must be same
cv2.waitKey(0) # To load and hold the image
cv2.destroyAllWindows() # To close the window after the required kill value was provided

# find all face locations, model can be cnn or hog
# number_of_faces=1 higher and detect more faces
all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

# print the numer of faces detected
print('There are {} face(s) in this image'.format(len(all_face_locations)))


#loop through faces
for index, current_face_location in enumerate(all_face_locations):
    # splitting the tuple to get the four position values
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
    # slice the image by positions
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow("Face No: "+str(index), current_face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 