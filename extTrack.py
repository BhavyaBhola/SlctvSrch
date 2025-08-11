import numpy as np
import os
from utils.encode import decodeMask
import cv2
import random
import argparse

def generate_overlay_video(track_id, alpha=0.85, fps=24, output_path="output_video.mp4"):
    
    validTracks = np.load("validTracks.npy", allow_pickle=True)
    trail_maps = np.load("trails_map.npy", allow_pickle=True)
    background = cv2.imread("background.jpg")

    vd = trail_maps.item()
    first_frame = cv2.imread(os.path.join("frames", os.listdir("frames")[0]))

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for valid in validTracks:
        if track_id == valid.id:
            color = tuple(random.randint(0, 255) for _ in range(3))
            frame_files = sorted(os.listdir("output_frames"), key=lambda x: int(x.split("_")[1].split(".")[0]))
            for i in range(len(valid.frame)):
                for o_frames in frame_files:
                    frame_id = int(o_frames.split("_")[1].split(".")[0])
                    if frame_id == valid.frame[i]:
                        img = cv2.imread(os.path.join("frames", o_frames))
                        mask = decodeMask(valid.mask_history[frame_id]).astype(np.uint8)
                        mask_3ch = cv2.merge([mask, mask, mask])
                        blended = background.copy().astype(np.float32)
                        blended[mask_3ch == 1] = (
                            alpha * img.astype(np.float32)[mask_3ch == 1] +
                            (1 - alpha) * background.astype(np.float32)[mask_3ch == 1]
                        )
                        blended = blended.astype(np.uint8)
                        for j in range(1, len(vd[track_id])):
                            pt1 = tuple(map(int, vd[track_id][j-1]))
                            pt2 = tuple(map(int, vd[track_id][j]))
                            cv2.line(blended, pt1, pt2, color, 5)
                        out.write(blended)
                        

    out.release()


def main():
    parser = argparse.ArgumentParser(
        description="Extract object based on id"
    )

    parser.add_argument("--track_id" , type=int , required=True,help="Id of the object you want to extract")
    parser.add_argument("--alpha" , type=float , default=0.85,help="degree of blending")
    parser.add_argument("--fps" , type=int, default=30 , help="Fps of the output video")
    parser.add_argument("--output_path" , type=str,default="extTrack.mp4" , help="output video path")

    args = parser.parse_args()

    generate_overlay_video(
        track_id=args.track_id,
        alpha=args.alpha,
        fps=args.fps,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
