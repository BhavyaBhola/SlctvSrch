import numpy as np
from ultralytics.models.sam import SAM
from ultralytics import FastSAM
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class genMask():

    def __init__(self, model_path="models/FastSAM-s.pt"):
        print("Init Sam")
        self.segModel = FastSAM(model_path).to(DEVICE)
        print("SAM initialized")

    def generateMask(self, image , box):
        if box is None:
            return None
        
        results = self.segModel.predict(image,bboxes=[box],verbose=False,retina_masks=True)

        if results[0].masks is not None and results[0].masks.data.numel()>0:
            mask_data = results[0].masks.data[0].cpu().numpy()
            boolean_mask = (mask_data > 0.5)
            return boolean_mask
        
        return None
    
