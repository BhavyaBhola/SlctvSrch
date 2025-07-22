import torch
import numpy as np
from torchvision.models import resnet34 , ResNet34_Weights
from torch import nn
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class embdModel():
    def __init__(self):
        modules = list(resnet34(weights=ResNet34_Weights).children())[:-1]
        self.embdModel = nn.Sequential(*modules)
        self.embdModel.to(DEVICE)
        self.embdModel.eval()

    def getEmbedding(self,img):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = self.embdModel(img_tensor)

        emd = emd.view(emd.shape[0] , -1)

        return emb


