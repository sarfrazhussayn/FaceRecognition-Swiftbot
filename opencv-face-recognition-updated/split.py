'''
Uses OpenCV, takes an mp4 video and imagify.

----
Open the main.py and edit the path to the video. Then run:

Which will produce a folder called data with the data. There will be 100 images for video input.

'''


import cv2
import numpy as np
import os
import time
import argparse
#import urllib.request

vid = r"https://www.techslides.com/demos/sample-videos/small.mp4"
# "https://raw.githubusercontent.com/ahsansn/workspace/master/small.mp4"
# r"C:\Users\Ahsan Ahmed\Videos\small.mp4"
# "https://techslides.com/demos/sample-videos/small.mp4"

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", default="hello",
    help="path to input directory of faces + images")
ap.add_argument("-v", "--video", default=vid,
    help="path to input directory of faces + images")
args = vars(ap.parse_args())


#urllib.request.urlretrieve(vid, 'video_name.mp4')

# Playing video from file:
FrameSkip = 10 #1 picture per 30 frames

cap = cv2.VideoCapture(args["video"])
while(cap.isOpened()==False):
    cap = cv2.VideoCapture(args["video"])
    print(1, cap.isOpened())

    
print(cap)
print(cap.isOpened())
#-----------------------------------------------
#Creating /directory

try:
    if not os.path.exists('dataset/'+str(args["name"])):
        os.makedirs('dataset/'+str(args["name"]))
except OSError:
    print ('Error: Creating directory of data')

#-----------------------------------------------
path = 'dataset/'+str(args["name"])+"/"
#-----------------------------------------------
#Calculations
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 (version 2) used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count,fps)
duration = frame_count/fps
#-----------------------------------------------


currentFrame = 0
while(currentFrame<frame_count):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #then above print('Creating...' + name)  add 
    if(currentFrame//FrameSkip==currentFrame/FrameSkip):
        # Saves image of the current frame in jpg file
        name = str(path) + "Picture" + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
    else:
        pass
        #print("control here")
        # To stop duplicate images
    currentFrame += 1


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
