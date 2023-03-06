import cv2
width = 320
hight = 320
confidince_thrushold = 0.5
import numpy as np


classNames = ['CAR', 'car_plate']
model_configration = "G:\AI\grage.cfg"
model_weghits = "G:\\AI\\grage.weights"

net = cv2.dnn.readNetFromDarknet(model_configration, model_weghits)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.imread('G:\\AI\\car3.jpg')

def find_objects(outputs, img):
    ht, wt, ct = cap.shape
    bounding_box = []
    calssids = []
    confidince_value = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classid = np.argmax(scores)
            conf = scores[classid]
            if conf > confidince_thrushold:
                w, h = int(det[2]* wt) , int(det[3]* ht)
                x, y = int((det[0]*wt) - w/2), int((det[1]*ht) - h/2)
                bounding_box.append([x, y, w, h])
                confidince_value.append(float(conf))
                calssids.append(classid)
    #print(len(bounding_box))
    incidies = cv2.dnn.NMSBoxes(bounding_box, confidince_value, confidince_thrushold, nms_threshold=0.3)
    for i in incidies:
        #i = i[0]
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (w+x, h+y),(255, 0, 0), 3)
        cv2.putText(img, f'{classNames[calssids[i]].upper()}, {int(confidince_value[i]*100)}%',
                    (x, y-10), cv2.FONT_ITALIC, 0.6, (0, 0, 200), 2)


blob = cv2.dnn.blobFromImage(cap, 1/255, (width, hight), [0, 0, 0], 1, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_names)
find_objects(outputs, cap)
cv2.imshow("YOLO", cap)
cv2.waitKey(0)