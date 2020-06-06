# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from PIL import Image
import requests
from io import BytesIO
from skimage import io
import urllib.request


# construct the argument parser and parse the arguments
img_o= "./UPLOAD_FOLDER/Gray.jpg"#"https://images.pexels.com/photos/736716/pexels-photo-736716.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"
#img_ = '.\images\mukesh.jpg'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to input image")
ap.add_argument("-d", "--detector", default='face_detection_model',
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", default='openface_nn4.small2.v1.t7',
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", default='output/recognizer.pickle',
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", default='output/le.pickle',
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.65,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
'''
print(img_o)
img = Image.open(BytesIO((requests.get(img_o)).content))
print("--", img_)
'''

#image = cv2.imread(img)

image = io.imread(img_o)

'''
with urllib.request.urlopen(img_o) as url:
	req = url.read()

#req = urllib.urlopen(img_o)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1)
'''
#urllib.request.urlretrieve(img_o, "hello.jpg")
#print("-here")
#imaasdsdge = cv2.imread("hello.jpg")


#print(image)

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

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
print(name)
import json

#import sklearn

def index(inter):
    # hdr = {'User-agent': 'Mozilla/5.0'}
    # req = request.vars.picUrl

    #return "yaaa"

    # construct the argument parser and parse the arguments
    # if(req == None):
    #     return "nothing provided"

    #img_o= args["images"]
    #"https://ik.imagekit.io/demo/img/newPages/dl_SysFHc2jm.png"
    #"https://api.anomoz.com/testing/pic.jpg"#"https://images.pexels.com/photos/736716/pexels-photo-736716.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"
    img_o = inter

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=img_o,
    	help="path to input image")
    ap.add_argument("-d", "--detector", default='face_detection_model',
    	help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", default='openface_nn4.small2.v1.t7',
    	help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", default='output/recognizer.pickle',
    	help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", default='output/le.pickle',
    	help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.65,
    	help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.dirname(os.path.abspath(__file__)) + "/face_detection_model/deploy.prototxt" #os.path.sep.join([args["detector"], "deploy.prototxt"])
    #return (protoPath)
    #r"face_detection_model/deploy.prototxt"#os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.dirname(os.path.abspath(__file__)) + "/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    #os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(os.path.dirname(os.path.abspath(__file__)) + "/"+ args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(os.path.dirname(os.path.abspath(__file__)) + "/" + args["recognizer"], "rb").read())
    le = pickle.loads(open(os.path.dirname(os.path.abspath(__file__)) + "/" + args["le"], "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    '''
    print(img_o)
    img = Image.open(BytesIO((requests.get(img_o)).content))
    print("--", img_)
    '''

    #image = cv2.imread(img)

    image = io.imread(img_o)

    '''
    with urllib.request.urlopen(img_o) as url:
    	req = url.read()

    #req = urllib.urlopen(img_o)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    '''
    #urllib.request.urlretrieve(img_o, "hello.jpg")
    #print("-here")
    #imaasdsdge = cv2.imread("hello.jpg")


    #print(image)

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

    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]

    	# filter out weak detections
    	if confidence > args["confidence"]:
    		# compute the (x, y)-coordinates of the bounding box for the
    		# face
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")

    		# extract the face ROI
    		face = image[startY:endY, startX:endX]
    		(fH, fW) = face.shape[:2]

    		# ensure the face width and height are sufficiently large
    		if fW < 20 or fH < 20:
    			continue

    		# construct a blob for the face ROI, then pass the blob
    		# through our face embedding model to obtain the 128-d
    		# quantification of the face
    		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
    			(0, 0, 0), swapRB=True, crop=False)
    		embedder.setInput(faceBlob)
    		vec = embedder.forward()

    		# perform classification to recognize the face
    		preds = recognizer.predict_proba(vec)[0]
    		j = np.argmax(preds)
    		proba = preds[j]
    		name = le.classes_[j]

    		# draw the bounding box of the face along with the associated
    		# probability
    		text = "{}: {:.2f}%".format(name, proba * 100)
    		y = startY - 10 if startY - 10 > 10 else startY + 10
    		cv2.rectangle(image, (startX, startY), (endX, endY),
    			(0, 0, 255), 2)
    		cv2.putText(image, text, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    print(name)
    return json.dumps({"personName": name })
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
index("./hello.jpg")