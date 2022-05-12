import cv2
import numpy as np
import glob
import random

class FireDetection():

    def __init__(self, path = "dataset_prep", conf_thresh = 0.3):
        self.net = cv2.dnn.readNet("yolov3_custom1_1000.weights", "yolov3.cfg")
        self.classes = ['fire']
        self.images_path = glob.glob(path+ "/images//*.jpg")
        # print(self.images_path)
        self.colors = np.random.uniform(0,255,size = (len(self.classes), 3))
        self.confidence_threshold = conf_thresh
        self.font = cv2.FONT_HERSHEY_PLAIN

    def run(self):
        layer_names = self.net.getLayerNames()
        # INSERT HERE THE PATH OF YOUR IMAGES
        # random.shuffle(images_path)
        # loop through all the images
        count = 0
        for img_path in self.images_path:
            if count > 10:
                break
            count += 1
            # loading image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (250,250), interpolation=cv2.INTER_AREA)
            height, width, channels = img.shape

            # detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)
            
            self.net.setInput(blob)
            output_layers = self.net.getUnconnectedOutLayersNames()
            outs = self.net.forward(output_layers)

            # showing informations on screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.confidence_threshold:
                        # object detected
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        
                        # rectangle coordinates
                        x = int(center_x - w /2)
                        y = int(center_y - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            if len(indexes) != 0:
                for i in indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = self.classes[class_ids[i]]
                    confidence_value = str(round(confidences[i],2))
                    color = (0,255,0)
                    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
                    cv2.putText(img, label+ " "+confidence_value, (x,y+20), self.font, 2, (255,255,255), 2)


            cv2.imshow("Image", img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()




