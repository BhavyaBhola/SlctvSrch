import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import os

import result
import initTrack
import matching
import tracker
from features.sift import Sift
from features.embd import embdModel
from kalmanFilter import KalmanFilter


def run(video_source, model, detection_conf, sift_good_dist, min_sift_score, accumulate_sift, visualize):

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    model = YOLO("models/" + model + ".pt")

    kf = KalmanFilter()
    sift = Sift(sift_good_dist)
    emb_model = embdModel()

    offline_all_tracks = []
    all_tracks = []
    uniq_id = 1
    frame_no = 0
    color_map = {}
    trails_map = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        results = model(frame, verbose=False, conf=detection_conf, classes=[2])

        mesur_list = initTrack.collect_measurments(results)
        #des_list = sift.collect_descriptors(mesur_list, frame)
        emb_list = emb_model.collectEmbedding(mesur_list,frame)

        C = B = C2 = B2 = C3 = B3 = None
        if all_tracks:
            C, B = matching.maha_dist_matrix(mesur_list, all_tracks, kf)
            #C2, B2 = matching.sift_dist_matrix(des_list, all_tracks, sift, min_sift_score, accumulate_sift)
            C3 , B3 = matching.embMatching(emb_list=emb_list , all_tracks=all_tracks)
            #C4 , B4 = matching.centroid_distance(all_tracks=all_tracks , mesur_list=mesur_list)

        unmatches = mesur_list.copy()
        
        if all_tracks:
            all_tracks, unmatches , emb_list = matching.matching_assignment(B, C, B3, C3, all_tracks, unmatches, frame_no, kf,emb_list=emb_list)

        unmatches_track, uniq_id = initTrack.new_track(unmatches, uniq_id, frame_no, kf,emb_list=emb_list)
        all_tracks, offline_all_tracks = tracker.update_tracks(all_tracks, offline_all_tracks, unmatches_track, kf)

        if visualize:
            frame, color_map ,trails_map= result.draw_current_tracks(frame, all_tracks, color_map,trails_map=trails_map)
            cv2.imshow("Tracking Window", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_no += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    det_array = result.det_file(offline_all_tracks, all_tracks, output_dir)
    print(f"Tracking data saved in '{output_dir}/det.txt'")


if __name__ == "__main__":

    run(video_source="/home/bhavyab/Desktop/dev/SlctvSrch/videos/2431853-hd_1920_1080_25fps.mp4",
        model="yolov8n",
        detection_conf=0.4,
        sift_good_dist=300,
        min_sift_score=20,
        accumulate_sift=3,
        visualize=True)

# def run(data_path, model, detection_conf, sift_good_dist, min_sift_score, accumulate_sift, visualize):

#     image_folder = data_path
#     images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

#     model = YOLO("models/" + model +".pt")
#     kf = KalmanFilter()
#     sift = Sift(sift_good_dist)

#     offline_all_tracks = []
#     all_tracks=[]
#     uniq_id = 0
#     color_map = {} 

#     for frame_no, image in enumerate(images):
#         print(frame_no)
#         image_path = os.path.join(image_folder, image)
#         frame = cv2.imread(image_path)
#         if frame is None:
#             continue
            
#         results = model(frame, verbose=True, conf=detection_conf , classes=[2])

#         mesur_list = initTrack.collect_measurments(results)
#         des_list = sift.collect_descriptors(mesur_list, frame)

#         C = B = C2 = B2 = None
#         if all_tracks:
#             C,B = matching.maha_dist_matrix(mesur_list, all_tracks, kf)
#             C2, B2 = matching.sift_dist_matrix(des_list, all_tracks, sift, min_sift_score, accumulate_sift)

#         unmatches = mesur_list.copy()
#         if all_tracks:
#             all_tracks, unmatches, des_list = matching.matching_assignment(C, B, C2, B2, all_tracks, unmatches, des_list, frame_no, kf)

#         unmatches_track, uniq_id = initTrack.new_track(unmatches, des_list, uniq_id, frame_no, kf)
        
#         # The update_tracks function in the prompt had a different signature
#         # Assuming the correct one based on the logic is as follows:
#         all_tracks, offline_all_tracks = tracker.update_tracks(all_tracks, offline_all_tracks, unmatches_track, kf)

#         # --- REAL-TIME VISUALIZATION ---
#         if visualize:
#             # Use the new function to draw current tracks on the frame
#             frame, color_map = result.draw_current_tracks(frame, all_tracks, color_map)
            
#             # Display the frame in a window
#             cv2.imshow("Real-time Tracking", frame)

#             # Press 'q' to exit the loop
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         # --- END OF VISUALIZATION BLOCK ---

#     # After the loop, clean up the window
#     if visualize:
#         cv2.destroyAllWindows()

#     # The logic to save the final detection file remains unchanged
#     det_array = result.det_file(offline_all_tracks, all_tracks, data_path)

# if __name__ == "__main__":
#     run(data_path="frames",
#         model="yolov8n",
#         detection_conf=0.4,
#         sift_good_dist=300,
#         min_sift_score=20,
#         accumulate_sift=3,
#         visualize=True)