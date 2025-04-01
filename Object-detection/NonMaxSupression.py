import torch
from Intersection_over_union import intersection_over_union
def nonmax_suppression(bboxes,iou_treshold,probability_treshold, box_format='corners'):

    #bboxes = [[class,bb_probability,x1,y1,x2,y2]]
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > probability_treshold]
    bboxes_after_suppression = []
    bboxes = sorted(bboxes, key=lambda box: box[1], reverse=True)
    while bboxes:
        chosen_bbox = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_bbox[0]
            or intersection_over_union(torch.tensor(chosen_bbox[2:]),
                                          torch.tensor(box[2:]),
                                          box_format=box_format
                                          ) < iou_treshold
        ]
        bboxes_after_suppression.append(chosen_bbox)
    return bboxes_after_suppression

