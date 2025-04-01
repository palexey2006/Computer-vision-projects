import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self,S=7, B=2,C=20):
        super(YoloLoss,self).__init__()
        self.mse =nn.MSELoss(reduction='sum')
        self.S = S # split size of image
        self.B = B # number of boxes
        self.C = C # number of classes

        # from YOLO paper, means how much we should pay loss
        # for no object and the box coordinates (noobj and coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, pred, target):
        pred = pred.reshape(-1,self.S,self.S,self.C + self.B * 5) # Batch_size x (grid_size x grid_size) x Num_classes * (BB * (x,y,h,w,confidence_score))
        iou_b1 = intersection_over_union(pred[...,21:25], target[...,21:25])
        iou_b2 = intersection_over_union(pred[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[...,20].unsqueeze(3) #Identity object i(basically gonna say whether there is an object)


        # For box coordinates

        box_predictions = exists_box * ((bestbox + pred[...,26:30] + (1 - bestbox) * pred[...,21:25]))
        box_targets = exists_box * target[...,21:25]
        #(N,S,S,4)
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # (N,S,S,4) -> (N*S*S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets,end_dim=-2)
                            )

        # For object loss
        # pred_box is the confidence score for the bbox with the highest IoU
        pred_box = (bestbox * pred[...,25:26] + (1 - bestbox) * pred[...,20:21])

        #(N*S*S)
        object_loss = self.mse(torch.flatten(exists_box * pred_box),
                               torch.flatten(exists_box * target[...,20:21])
                               )

        #For NO object loss
        # (N,S,S,1) -> (N,S*S)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * pred[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * pred[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # for Class Loss
        # (N,S,S,20) -> (N*S*S,20)
        class_loss = self.mse(torch.flatten(exists_box * pred[...,:20], end_dim=-2),
                              torch.flatten(exists_box * target[...,:20], end_dim=-2)
                              )


        loss = (self.lambda_coord * box_loss # first two rows of loss in the paper
                + object_loss # third row in the paper
                + self.lambda_noobj * no_object_loss
                + class_loss
                )
        return loss



