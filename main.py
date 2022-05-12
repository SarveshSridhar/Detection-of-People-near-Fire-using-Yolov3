import fire_detection
import normal_coco
import random
import glob
import cv2
import numpy as np
from shapely.geometry import Polygon

class Detector():
    def __init__(self, dataSet_path = "dataset_prep", 
                cocoClassNames_path = 'coco.names', 
                fire_ConfThresh = 0.3, Normal_ConfThresh = 0.3):

        if dataSet_path != 'dataset_prep':
            self.images_path = ["dataset_prep/images/images220.jpg"]
        else:
            self.images_path = glob.glob(dataSet_path+"/images//*.jpg")
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.VehicleClasses = ['person','bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
        self.VideoPath = 'dataset_prep/video.mp4'

        self.Normalnet = cv2.dnn.readNet('yolov3 (1).weights','coco_yolov3.cfg')
        self.NormalClasses = []
        with open(cocoClassNames_path, 'r') as f:
            self.NormalClasses = f.read().splitlines()
        self.NormalThresh = Normal_ConfThresh
        self.NormalColors = np.random.uniform(0,255,size = (len(self.NormalClasses), 3))

        self.Firenet = cv2.dnn.readNet("yolov3_custom1_1000.weights", "yolov3.cfg")
        self.FireClasses = ['fire']
        self.FireColors = np.random.uniform(0,255,size = (len(self.FireClasses), 3))
        self.FireThresh = fire_ConfThresh

    # def runVideo(self):
    #     cap = cv2.VideoCapture(self.VideoPath)
    #     layer_names = self.Firenet.getLayerNames()
    #     while True:
    #         _, img = cap.read()


    def runPic(self):
        '''
        Fire first
        '''
        layer_names = self.Firenet.getLayerNames()
        
        random.shuffle(self.images_path)
        
        count = 0
        for img_path in self.images_path:
            if count > 10:
                break
            count += 1
            # loading image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (600,600), interpolation=cv2.INTER_AREA)
            height, width, channels = img.shape

            # detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)
            
            # FIRE BLOB
            self.Firenet.setInput(blob)
            output_layers1 = self.Firenet.getUnconnectedOutLayersNames()
            Fireouts = self.Firenet.forward(output_layers1)

            # NORMAL BLOB
            self.Normalnet.setInput(blob)
            output_layers2 = self.Normalnet.getUnconnectedOutLayersNames()
            Normalouts = self.Normalnet.forward(output_layers2)

            # showing informations on screen - FIRE
            class_ids1 = []
            confidences1 = []
            boxes1 = []
            for out in Fireouts:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.FireThresh:
                        # object detected
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        
                        # rectangle coordinates
                        x = int(center_x - w /2)
                        y = int(center_y - h/2)

                        boxes1.append([x,y,w,h])
                        confidences1.append(float(confidence))
                        class_ids1.append(class_id)

            Fireindexes = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.5, 0.4)
            
            # SHOW INFORMATION - VEHICLES
            class_ids2 = []
            confidences2 = []
            boxes2 = []
            for out in Normalouts:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.NormalThresh:
                        # object detected
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        
                        # rectangle coordinates
                        x = int(center_x - w /2)
                        y = int(center_y - h/2)

                        boxes2.append([x,y,w,h])
                        confidences2.append(float(confidence))
                        class_ids2.append(class_id)

            Normalindexes = cv2.dnn.NMSBoxes(boxes2, confidences2, 0.5, 0.4)

            f1 = 0 #Flag1
            Fire_arr = []
            if len(Fireindexes) != 0:
                for i in Fireindexes.flatten():
                    x1,y1,w1,h1 = boxes1[i]
                    Fire_arr.append([x1,y1,w1,h1])
                    label1 = self.FireClasses[class_ids1[i]]
                    print(label1, (x1,y1,w1,h1))
                    confidence_value1 = str(round(confidences1[i],2))
                    color1 = (0,255,0)
                    cv2.rectangle(img, (x1,y1), (x1+w1,y1+h1), color1, 2)
                    img1 = cv2.putText(img, label1+ " "+confidence_value1, (x1,y1+20), self.font, 2, (255,255,255), 2)
                    f1 = 1

            f2 = 0 #Flag2
            Vehicle_arr = []
            if len(Normalindexes) != 0:
                for i in Normalindexes.flatten():
                    x2,y2,w2,h2 = boxes2[i]
                    # LABEL ONLY VEHICLES
                    if self.NormalClasses[class_ids2[i]] in self.VehicleClasses:
                        Vehicle_arr.append([x2,y2,w2,h2])
                        print(self.NormalClasses[class_ids2[i]], (x2,y2,w2,h2))
                        label2 = self.NormalClasses[class_ids2[i]]
                        confidence_value2 = str(round(confidences2[i],2))
                        color2 = (255,255,0)
                        cv2.rectangle(img, (x2,y2), (x2+w2,y2+h2), color2, 2)
                        img2 = cv2.putText(img, label2+ " "+confidence_value2, (x,y+20), self.font, 2, (0,255,255), 2)
                        f2 = 1

            if f1 == 1 and f2 == 1:
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                return Fire_arr, Vehicle_arr

        cv2.destroyAllWindows()

    def calculate_iou(self, x1,y1,w1,h1,x2,y2,w2,h2):
        box_1 = [[x1, y1], [x1+w1, y1], [x1+w1, y1+h1], [x1, y1+h1]]
        box_2 = [[x2, y2], [x2+w2, y2], [x2+w2, y2+h2], [x2, y2+h2]]
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou

    def findIOU(self, Fire_arr, Vehicle_arr):
        print(Fire_arr)
        for num_fire in range(len(Fire_arr)):
            x1,y1,w1,h1 = Fire_arr[num_fire][0], Fire_arr[num_fire][1], Fire_arr[num_fire][2], Fire_arr[num_fire][3]
            # find distance
            for num_normal in range(len(Vehicle_arr)):
                x2,y2,w2,h2 = Vehicle_arr[num_fire][0], Vehicle_arr[num_fire][1], Vehicle_arr[num_fire][2], Vehicle_arr[num_fire][3]
                iou_value = self.calculate_iou(x1, y1, w1, h1, x2, y2, w2, h2)
                print("IOU value: ",iou_value)

        return Fire_arr


def main():
        detector = Detector()
        Fire_arr, Vehicle_arr = detector.runPic()
        distance = detector.findIOU(Fire_arr, Vehicle_arr)
        print(distance)

if __name__ == '__main__':
    main()