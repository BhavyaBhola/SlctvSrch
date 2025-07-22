from ultralytics import YOLO
from ultralytics.models.sam import SAM
import numpy as np
import cv2

def yoloDetect(img):
    model = YOLO('models/yolov8n.pt')
    
    res = model(img , classes=[2])

    return res


def getMask(img):
    detects = yoloDetect(img)
    seg_model = SAM('models/mobile_sam.pt')
    boxes = detects[0].boxes.xyxy.cpu().numpy().astype(int)
    foregroundMask = np.zeros((img.shape[0] , img.shape[1]))

    for box in boxes:
        result = seg_model.predict(img , bboxes=[box])
        mask = result[0].masks.data[0].cpu().numpy()
        mask = (mask > 0.5).astype(bool)
        foregroundMask[mask] = 255
        img[mask] = (0.5 * img[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

    foreground = cv2.bitwise_and(img , img , mask=foregroundMask)
    cv2.imwrite('test.jpg', img)

if __name__ == "__main__":
    img = cv2.imread("frames/950.jpg")
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    getMask(img)
