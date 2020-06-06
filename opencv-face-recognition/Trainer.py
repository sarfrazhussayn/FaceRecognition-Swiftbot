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
import urllib.request
import pickle
from imutils import paths
import imutils
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def extract_embeddings():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", default='dataset',
    	help="path to input directory of faces + images")
    ap.add_argument("-e", "--embeddings", default='output/embeddings.pickle',
    	help="path to output serialized db of facial embeddings")
    ap.add_argument("-d", "--detector", default='face_detection_model',
    	help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", default='openface_nn4.small2.v1.t7',
    	help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-c", "--confidence", type=float, default=0.65,
    	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector from disk
    #print("[INFO] loading face detector...")
    protoPath = "./face_detection_model/deploy.prototxt"
    modelPath = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    #print("[INFO] loading face recognizer...")
    #embedder = cv2.dnn.readNetFromTorch(os.path.abspath(__file__) + "/" +args["embedding_model"])
    #return ("/home/output/embeddings.pickle")
    embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")

    # grab the paths to the input images in our dataset
    #print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("./" + args["dataset"]))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):

    	# extract the person name from the image path
    	#print("[INFO] processing image {}/{}".format(i + 1,	len(imagePaths)))
    	name = imagePath.split(os.path.sep)[-2]
    	#print(imagePath)

    	# load the image, resize it to have a width of 600 pixels (while
    	# maintaining the aspect ratio), and then grab the image
    	# dimensions
    	image = cv2.imread(imagePath)
    	image = imutils.resize(image, width=600)
    	(h, w) = image.shape[:2]

    	# construct a blob from the image
    	imageBlob = cv2.dnn.blobFromImage(
    		cv2.resize(image, (300, 300)), 1.0, (300, 300),
    		(104.0, 177.0, 123.0), swapRB=False, crop=False)

    	# apply OpenCV's deep learning-based face detector to localize
    	# faces in the input image
    	detector.setInput(imageBlob)
    	detections = detector.forward()

    	# ensure at least one face was found
    	if len(detections) > 0:
    		# we're making the assumption that each image has only ONE
    		# face, so find the bounding box with the largest probability
    		i = np.argmax(detections[0, 0, :, 2])
    		confidence = detections[0, 0, i, 2]

    		# ensure that the detection with the largest probability also
    		# means our minimum probability test (thus helping filter out
    		# weak detections)
    		if confidence > args["confidence"]:
    			# compute the (x, y)-coordinates of the bounding box for
    			# the face
    			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    			(startX, startY, endX, endY) = box.astype("int")

    			# extract the face ROI and grab the ROI dimensions
    			face = image[startY:endY, startX:endX]
    			(fH, fW) = face.shape[:2]

    			# ensure the face width and height are sufficiently large
    			if fW < 20 or fH < 20:
    				continue

    			# construct a blob for the face ROI, then pass the blob
    			# through our face embedding model to obtain the 128-d
    			# quantification of the face
    			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
    				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    			embedder.setInput(faceBlob)
    			vec = embedder.forward()

    			# add the name of the person + corresponding face
    			# embedding to their respective lists
    			knownNames.append(name)
    			knownEmbeddings.append(vec.flatten())
    			total += 1

    # dump the facial embeddings + names to disk
    #print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    #print(data)
    f = open("./" + args["embeddings"], "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("[INFO] Process Complete...")
    return ("completed")

def train_model():

    # load the face embeddings
    #print("[INFO] loading face embeddings...")
    data = pickle.loads(open("./output/embeddings.pickle", "rb").read())

    # encode the labels
    #print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    #print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open("./output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("./output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()
    print("[INFO] Process Complete...")
    return True



def index(_url,_id):

    hdr = {'User-agent': 'Mozilla/5.0'}
    req =  _url #"https://raw.githubusercontent.com/ahsansn/workspace/master/VID_20191126_100350848.mp4"
    userId = _id  #"Minhaj"

    #return "yaaa"

    # construct the argument parser and parse the arguments
    if((req == None) or (userId == None)):
        return "nothing provided"
    vid = req

    #vid = "https://raw.githubusercontent.com/ahsansn/workspace/master/small.mp4"
    # r"C:\Users\Ahsan Ahmed\Videos\small.mp4"
    #"https://techslides.com/demos/sample-videos/small.mp4"

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", default=userId,
        help="path to input directory of faces + images")
    ap.add_argument("-v", "--video", default=vid,
        help="path to input directory of faces + images")
    args = vars(ap.parse_args())



    urllib.request.urlretrieve(vid, './video_name.mp4')

    # Playing video from file:
    FrameSkip = 10 #1 picture per 30 frames
    cap = cv2.VideoCapture("./video_name.mp4")#args["video"]


    #return (cap)
    #return (cap.isOpened())
    #print(cap)
    #-----------------------------------------------
    #Creating /directory
    path = "C:/Users/UN/Desktop/opencv-face-recognition/FaceRecognition-Swiftbot/opencv-face-recognition/dataset/Minhaj/"
    try:
        if not os.path.exists('./dataset/'+str(args["name"])):
            os.makedirs('./dataset/'+str(args["name"]))
    except OSError:
        print ('Error: Creating directory of data')

    #-----------------------------------------------dataset/'+str(args["name"])+"/"
    #-----------------------------------------------
    #Calculations
    print(cap)
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
            #print ('Creating...' + name)
            cv2.imwrite(name, frame)
        else:
            pass
            #print("control here")
            # To stop duplicate images
        currentFrame += 1


    # When everything done, release the capture
    cap.release()
    #cv2.destroyAllWindows()

    #return "aa";
    extract_embeddings()
    train_model()
    #return "completed"
    return json.dumps({"status": "successfull" })



#train_model()


index("https://raw.githubusercontent.com/ahsansn/workspace/master/VID_20191126_100350848.mp4","Minhaj")