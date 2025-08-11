import numpy as np
from scipy.optimize import linear_sum_assignment

def sift_dist_matrix(des_list, all_tracks, sift, min_score, compare_number):
    C = np.zeros((len(des_list) , len(all_tracks)))
    B = np.zeros((len(des_list) , len(all_tracks)))
    
    for i in range(len(des_list)):
        if des_list[i] is None or des_list[i].shape[0]< 11:
            continue
        for j in range(len(all_tracks)):
            max_val=0
            for k in all_tracks[j].descriptor[-compare_number:]:
                if k is None:
                    continue
                val = sift.percent_matching(des_list[i], k)
                if val>max_val:
                    max_val = val
            C[i,j] = max_val
            if C[i,j] >= min_score:
                B[i,j] = 1

    return C,B

# def centroid_distance(all_tracks , mesur_list, min_dist=0.4):
#     C = np.zeros((len(mesur_list) , len(all_tracks)))
#     B = np.zeros((len(mesur_list) , len(all_tracks)))

#     for i in range(len(mesur_list)):
#         for j in range(len(all_tracks)):
#             x_new , x_prev = mesur_list[i][0] , all_tracks[j].mean[-1][0]
#             y_new , y_prev = mesur_list[i][1] , all_tracks[j].mean[-1][1]

#             centroid_distance = np.sqrt((x_new-x_prev)**2 + (y_new-y_prev)**2)
#             C[i][j] = centroid_distance

#             if centroid_distance<=min_dist:
#                 B[i][j] = 1
#             else:
#                 B[i][j] = 0

#     return C,B


def maha_dist_matrix(mesur_list, all_tracks, kf):

    C = np.zeros((len(mesur_list), len(all_tracks)))
    B = np.zeros((len(mesur_list), len(all_tracks)))
    for i in range(len(mesur_list)):
        for j in range(len(all_tracks)):
            C[i][j] = kf.mahalanobis_dist(all_tracks[j].mean[-1], all_tracks[j].covariance, mesur_list[i])
            if C[i][j] <= 9:
                B[i][j] = 1
            else:
                B[i][j] = 0
    return C,B

def embMatching(emb_list, all_tracks,similarity=0.993):
    C = np.zeros((len(emb_list) , len(all_tracks)))
    B = np.zeros((len(emb_list) , len(all_tracks)))

    
    for i in range(len(emb_list)):
        for j in range(len(all_tracks)):
            if all_tracks[j].embedding[0] == None:
                continue
            prod = np.dot(emb_list[i][0] , all_tracks[j].embedding[0])
            magA = np.linalg.norm(emb_list[i][0])
            magB = np.linalg.norm(all_tracks[j].embedding[0])

            C[i][j] = prod / (magA*magB)
            if C[i][j]>=similarity:
                B[i][j] = 1
            else:
                B[i][j] = 0
    
    return C,B


def matching_assignment(B , C, B2 , C2 , all_tracks , unmatches, frame_no, kf, current_frame_masks ,emb_list):
    l1 = 0.5
    l2 = 0.5
    
    C = C / 10
    cost = l1*C + l2*C2
    row_idx , col_idx = linear_sum_assignment(cost)
    #print(f" cost-- {cost}")
    del_idx = []

    for k in range(len(row_idx)):
        if B[row_idx[k]][col_idx[k]]>0 and B2[row_idx[k]][col_idx[k]]>0:
            # print("matched")
            obj = all_tracks[col_idx[k]]
            obj.measurement.append(unmatches[row_idx[k]])  
            # obj.descriptor.append(des_list[row_idx[k]])  
            obj.frame.append(frame_no)
            obj.embedding = emb_list[row_idx[k]]
            obj.status = 'matched'
            obj.reset()
            mask_to_add = current_frame_masks[row_idx[k]]
            obj.addMask(frame_no,mask_to_add)

            new_m, new_c = kf.update(obj.mean[-1], obj.covariance, obj.measurement[-1])
            obj.mean[-1] = new_m
            obj.covariance = new_c

            del_idx.append(row_idx[k])

    unmatches = [ele for idx, ele in enumerate(unmatches) if idx not in del_idx]
    emb_list = [ele for idx , ele in enumerate(emb_list) if idx not in del_idx]
    remaining_masks = [ele for idx, ele in enumerate(current_frame_masks) if idx not in del_idx]

    return all_tracks, unmatches ,emb_list , remaining_masks 




