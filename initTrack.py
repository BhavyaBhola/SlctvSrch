import numpy as np
from track import Track

def collect_measurments(results):
    mesur_list = []

    for obj in results[0].boxes:

        bbx = obj.xywh[0]
        x_cen = float(bbx[0])
        y_cen = float(bbx[1])
        ar = float(bbx[3] / bbx[2]) if bbx[2] != 0 else 0
        h = float(bbx[3])

        msmt = np.array([x_cen , y_cen , ar , h])
        mesur_list.append(msmt)

    return mesur_list

def new_track(unmatches, unique_id , frame_no ,kf,current_frame_masks,emb_list):
    unmatches_track = []
    for j,measur in enumerate(unmatches):
        mean, covariance = kf.initialize(measur)
        
        emb = emb_list[j]
        mask = current_frame_masks[j]

        new_obj_track = Track(
            id=unique_id,
            status="new",
            measurment=[measur],
            mean=[mean],
            frame=[frame_no],
            covariance=covariance,
            embedding=emb
        )

        new_obj_track.addMask(frame_no,mask)
        unmatches_track.append(new_obj_track)

        unique_id+=1

    
    return unmatches_track, unique_id