import logging

import numpy as np
import cv2
import os
import imutils
import pyttsx3
import math
coordinates={}
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def calculateDistance(idx,dict):
    distList=[]
    dist=0
    listap=[]
    #try only to change one object rect color by the time
    xc,yc=dict[idx]
    #listap.append(idx)

    for j, (xc1, yc1) in dict.items():
        if j != idx:
            # print(int(xc),yc)
            dist= math.sqrt((xc1 - xc) ** 2 + (yc1 - yc) ** 2)
            print(idx, j, dist)
            distList.append(dist)
            # print(dist)
            # listidx.append((i,j))
            # distList.append(dist)

    return distList
def calculateCenter(startX, startY,endX, endY):
    centerX=(startX+endX)/2
    centerY = (startY+endY)/2
    return centerX,centerY
# def drawLine(frame,detected_persons):
#     listap = []
#     for i,(xc,yc) in detected_persons.items():
#         listap.append(i)
#         for j,(xc1,yc1) in detected_persons.items():
#             if j not in listap:
#                 #print(int(xc),yc)
#                 cv2.line(frame,(int(xc),int(yc)),(int(xc1),int(yc1)),(255,0,0),2)
#                 #print((int(xc),int(yc)),(int(xc1),int(yc1)))
def loopLine(detections):
    close_dict=[]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # first dimintion 0 and then the first dim inside our first dim 0 then the dimentions which have confidence >0.7 #then select the lable element wich in this case is 1
        idx = int(detections[0, 0, i, 1])
        if idx == 15:
            if confidence > 0.85:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (centerX,centerY)= calculateCenter(startX, startY, endX, endY)
                detected_persons[i]={centerX,centerY}
                coordinates[i]=(startX, startY, endX, endY)
                distList=calculateDistance(i,detected_persons)
                if len(distList) !=0:
                    #print(distList)
                    dist=min(distList)
                    if dist !=0 and dist <200:
                        close_dict.append(i)
                        logging.info(f'{i} has social distance')
                # display the prediction
                #label = "{}:{}: {:.2f}%".format(i,AVAILABLE_CLASSES[idx], confidence * 100)
    return close_dict
                #y = startY - 15 if startY - 15 > 15 else startY + 15  # padding
               # cv2.putText(frame, label, (startX, y),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# all object that model can predect
AVAILABLE_CLASSES = \
    ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# open cam
camera = cv2.VideoCapture(r"D:\ML\testvideo2.mp4")
engine = pyttsx3.init()
# load model
net = cv2.dnn.readNetFromCaffe(r"D:\ML\group_study\object-master\object-master\MobileNetSSD_deploy.prototxt.txt", r"D:\ML\group_study\object-master\object-master\MobileNetSSD_deploy.caffemodel")

# main loop of the prog
while True:
    # read 1st frame
    ret, colored_image = camera.read()
    if not ret:
        break
    frame = imutils.resize(colored_image, width=800)

    # stack of object
    detected_persons = {}

    # define hight and width of the frame 
    (h, w) = frame.shape[0:2]

    # preprocessing of the frame before  classifying "numbers are standard"
    blob = cv2.dnn.blobFromImage(frame,
                                 0.007843, (900, 900), 127.5,swapRB=True)

    net.setInput(blob)

    # forward-propagate
    detections = net.forward()
    persons_list_social=loopLine(detections)
    for key  in detected_persons.keys():

        if key in persons_list_social:
            print(key)
            bounding_box_color=(0,0,255)

        else:
            bounding_box_color=(0,255,0)

        (startX, startY, endX, endY)= coordinates[key]
        print(coordinates)

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                bounding_box_color, 2)
        label = "{}".format(key)
        y = startY - 15 if startY - 15 > 15 else startY + 15  # padding
        cv2.putText(frame, label, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# print(detections)

    cv2.imshow("_", frame)
    if cv2.waitKey(10) == ord('q'):
        break
camera.release()






