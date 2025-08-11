import torch
import numpy as np
from torchvision.models import resnet34 , ResNet34_Weights
from torch import nn
from torchvision import transforms
import cv2
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class embdModel():
    def __init__(self):
        modules = list(resnet34(weights=ResNet34_Weights).children())[:-1]
        self.embdModel = nn.Sequential(*modules)
        self.embdModel.to(DEVICE)
        self.embdModel.eval()

    def getEmbedding(self,img):
        img = torch.from_numpy(img).float().permute(2,0,1)
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = self.embdModel(img_tensor)

        emb = emb.view(emb.shape[0] , -1)

        return emb

    def collectEmbedding(self,mesur_list,frame):
        emb_list = []

        for i in mesur_list:
            h = i[3]
            w = i[2]*h
            x1 = abs(int(i[0]-w/2))
            y1 = abs(int(i[1]-h/2))
            x2 = abs(int(i[0] + w/2))
            y2 = abs(int(i[1] + h/2))


            template = frame[y1:y2, x1:x2,:]
            #print(template.shape)
            template = cv2.GaussianBlur(template, (3, 3), 0)
            emb = self.getEmbedding(template)
            emb_list.append(emb)

        return emb_list

