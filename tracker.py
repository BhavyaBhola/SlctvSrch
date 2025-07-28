def update_tracks(all_tracks , offline_all_tracks , unmatched_tracks , kf):

    all_tracks = all_tracks + unmatched_tracks

    for n , obj in enumerate(all_tracks):
        if obj.status == "new" and obj.counter>3:
            del all_tracks[n]
        
        if obj.status == "matched" and obj.counter>10:
            offline_all_tracks.append(all_tracks[n])
            del all_tracks[n]

        pred_m, pred_c = kf.predict(obj.mean[-1], obj.covariance)
        obj.mean.append(pred_m)
        obj.covariance = pred_c
        obj.inc_count()

    return all_tracks, offline_all_tracks
