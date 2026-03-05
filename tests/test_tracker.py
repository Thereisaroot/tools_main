from isac_labelr.models import Detection
from isac_labelr.vision.tracker import ByteTrackLite


def test_tracker_keeps_id_for_close_boxes():
    tracker = ByteTrackLite(iou_threshold=0.1, max_missed=2)

    frame1 = [Detection((10, 10, 50, 80), 0.9)]
    tracks1 = tracker.update(frame1)
    assert len(tracks1) == 1
    tid = tracks1[0].track_id

    frame2 = [Detection((12, 11, 52, 81), 0.88)]
    tracks2 = tracker.update(frame2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == tid
