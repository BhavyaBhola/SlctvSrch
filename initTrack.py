import numpy as np
from track import Track

def collect_measurments(results):
    mesur_list = []

    for obj in results[0].boxes:

        bbx = obj.xywh[0]
        x_cen = float(bbx[0])
        y_cen = float(bbx[1])
        ar = float(bbx[3] / bbx[2])
        h = float(bbx[3])

        msmt = np.array([x_cen , y_cen , ar , h])
        mesur_list.append(msmt)

    return mesur_list

def new_track(unmatches, unique_id , frame_no ,kf,emb_list):
    unmatches_track = []
    for j,measur in enumerate(unmatches):
        mean, covariance = kf.initialize(measur)
        
        emb = emb_list[j]
        unmatches_track.append(Track(unique_id,'new',[measur], [mean] ,[frame_no], covariance, embedding=emb))
        unique_id+=1

    
    return unmatches_track, unique_id