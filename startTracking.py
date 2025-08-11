import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import os
import argparse

from tqdm import tqdm

import result
import initTrack
import matching
import tracker
from features.sift import Sift
from features.embd import embdModel
from kalmanFilter import KalmanFilter
from segments.getMask import genMask
from utils.encode import decodeMask
from utils.backgndExt import extBck
from utils.extractFrames import getFrames

def run(data_path, model, detection_conf, visualize , Save , getDetectFile):

    images = sorted([img for img in os.listdir(data_path)])
    model = YOLO("models/" + model + ".pt")
    kf = KalmanFilter()
    emb_model = embdModel()
    mask_generator = genMask()

    offline_all_tracks = []
    all_tracks = []
    validTracks = []
    uniq_id = 1
    color_map = {}
    trails_map = {}

    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    for frame_no , image in (enumerate(tqdm(images , total=len(images) , ascii=" >="))):

        image_path = os.path.join(data_path, image)
        frame = cv2.imread(image_path)

        results = model(frame, verbose=False, conf=detection_conf, classes=[2])
        mesur_list = initTrack.collect_measurments(results)
        emb_list = emb_model.collectEmbedding(mesur_list, frame)
        yolo_boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        current_frame_masks = [mask_generator.generateMask(frame, box) for box in yolo_boxes_xyxy]

        C = B = C3 = B3 = None
        if all_tracks:
            C, B = matching.maha_dist_matrix(mesur_list, all_tracks, kf)
            C3, B3 = matching.embMatching(emb_list=emb_list, all_tracks=all_tracks)

        unmatches = mesur_list.copy()
        if all_tracks:
            all_tracks, unmatches, emb_list, remaining_masks = matching.matching_assignment(
                B, C, B3, C3, all_tracks, unmatches, frame_no, kf, current_frame_masks, emb_list=emb_list)
        else:
            remaining_masks = current_frame_masks

        unmatches_track, uniq_id = initTrack.new_track(unmatches, uniq_id, frame_no, kf, remaining_masks, emb_list=emb_list)
        all_tracks, offline_all_tracks = tracker.update_tracks(all_tracks, offline_all_tracks, unmatches_track, kf)

        frame, color_map, trails_map = result.draw_current_tracks(frame, all_tracks, color_map, trails_map=trails_map)

        combined_mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        for track in all_tracks:
            if track.mask_history:
                last_frame_no = max(track.mask_history.keys())
                last_mask = track.mask_history[last_frame_no]
                track_color = color_map.get(track.id, (0, 255, 0))
                dec_mask = decodeMask(last_mask)
                combined_mask_overlay[dec_mask] = track_color
        frame = cv2.addWeighted(frame, 1.0, combined_mask_overlay, 0.5, 0)

        save_path = os.path.join(output_dir, f"frame_{frame_no:05d}.jpg")
        cv2.imwrite(save_path, frame)

        
        if visualize:
            cv2.imshow("Tracking Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if getDetectFile:
        result.det_file(offline_all_tracks, all_tracks, output_dir)
    print(f"Tracking data saved in '{output_dir}/det.txt'")
    create_video_from_frames(output_dir, "output_video.mp4", fps=25)
    total_tracks = np.concatenate((offline_all_tracks , all_tracks))
    
    for track in total_tracks:
        if len(track.frame)>10:
            validTracks.append(track)

    if Save:
        np.save("validTracks.npy" , validTracks)
        np.save("trails_map.npy" , trails_map)


def create_video_from_frames(folder, output_path, fps=25):
    images = sorted([img for img in os.listdir(folder) if img.endswith(".jpg")])
    if not images:
        print("No frames found for video creation.")
        return

    sample_img = cv2.imread(os.path.join(folder, images[0]))
    height, width, _ = sample_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(folder, img_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Main file used for Tracking and creating Tracks")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--model", type=str, default="yolov8n", help="YOLO model name or path.")
    parser.add_argument("--detection_conf", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during processing.")
    parser.add_argument("--no-save", action="store_true", help="Disable saving results.")
    parser.add_argument("--get-detect-file", action="store_true", help="Generate detection file.")
    parser.add_argument("--output_dir", type=str, default="frames", help="Directory to store extracted frames.")

    args = parser.parse_args()

    extBck(video_path=args.video_path)
    getFrames(video_path=args.video_path, output_dir=args.output_dir)
    
    run(
        data_path=args.output_dir,
        model=args.model,
        detection_conf=args.detection_conf,
        visualize=args.visualize,
        Save=not args.no_save,
        getDetectFile=args.get_detect_file
    )

if __name__ == "__main__":
    main()


