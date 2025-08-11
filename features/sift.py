import numpy as np
import cv2
from .embd import embdModel

test = embdModel()

class Sift:
    def __init__(self , threshold):
        self.threshold = threshold
    
    def collect_descriptors(self,mesur_list,img):
        des_list = []
        sift = cv2.SIFT_create()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in mesur_list:
            h = i[3]
            w = i[2]*h
            x1 = int(i[0]-w/2)
            y1 = int(i[1]-h/2)
            x2 = int(i[0] + w/2)
            y2 = int(i[1] + h/2)
            template = img[y1:y2, x1:x2,:]
            template = cv2.GaussianBlur(template, (5, 5), 0)
            _, des = sift.detectAndCompute(template,None)
            des_list.append(des)

        return des_list
    
    def percent_matching(self,des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1 , des2 , k=2)

        good_dist = []
        i=0
        for m,n in matches:
            if m.distance < 0.95*n.distance:
                good_dist.append(m.distance)
            i=i+1
        
        good_dist = np.array(good_dist)
        thres_dist = good_dist[good_dist < self.threshold]
        if len(good_dist)==0:
            return 0
        
        return (len(thres_dist)/len(good_dist))*100
