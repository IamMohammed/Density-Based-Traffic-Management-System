import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)  # index for number of cameras
whT = 320 # width, height, Target
confThreshold = 0.5 # threshold value of the confidence parameter
nmsThreshold = 0.2

#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n') # Open and extract info from text file
print(classNames)
## Model Files
modelConfiguration = "yolov3-320.cfg"   # importing config file
modelWeights = "yolov3.weights"         # importing weight file
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)  # create network
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)                # declare that opencv is the backend
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)                      # declare that cpu is target


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    # network only accepts image in a particular format i.e blob
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)  # convert the image to blob format
    net.setInput(blob) # set the input == blob
    layersNames = net.getLayerNames()  # to get names of the convolutional layers/ 3-outputs
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames) # forward pass the image to our network
    findObjects(outputs, img)

    count = 0
    for i in range(79):
        if int(classNames[0][i] == 0):
            count += 1
    print('Number of people:', count)

    cv.imshow('Image', img)
    cv.waitKey(1)