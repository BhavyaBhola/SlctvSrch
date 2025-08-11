import cv2
import os

def getFrames(output_dir , video_path):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_skip = 0
    i = 0 
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            save_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            frame_count += 1
            cv2.imwrite(save_path, frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = "frames"
    
    getFrames(output_dir)

