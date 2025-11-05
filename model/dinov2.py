import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align, nms
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import pickle
import cv2
import pdb

class DINOv2Backbone(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base'): # facebook/dinov3-vit7b16-pretrain-lvd1689m , facebook/dinov2-base
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.out_channels = self.model.config.hidden_size
    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state
        
        # Remove the CLS token (first token) and reshape to spatial dimensions
        features = features[:, 1:, :]
        B, N, C = features.shape
        
        H = W = int(N ** 0.5) 
        
        if H * W != N:
            target_size = H*W
            if N > target_size:
                features = features[:, :target_size, :]
            else:
                padding = target_size - N
                features = torch.cat([features, torch.zeros(B, padding, C, device=features.device)], dim=1)
        features = features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        return features

class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.output = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    def forward(self, x):
        x = self.lateral(x)
        x = self.output(x)
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, 4)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


# 5. Simple DINOv2 Detector for Object Detection
class DINOv2Detector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = DINOv2Backbone()
        self.fpn = SimpleFPN(self.backbone.out_channels)
        self.rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.rpn_head = RPNHead(256, self.rpn_anchor_generator.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
            self.rpn_anchor_generator,
            self.rpn_head,
            fg_iou_thresh=0.6,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7
        )
        self.head = DetectionHead(256, num_classes)
        self.nms_thresh = 0.5
        self.score_thresh = 0.3

    def forward(self, images, targets=None, mode='train'):
        features = self.backbone(images)
        features = self.fpn(features)

        feature_dict = {"0": features}
        image_sizes = [(images.shape[-2], images.shape[-1]) for _ in range(images.shape[0])]
        image_list = ImageList(images, image_sizes)
        
        if mode == 'train':
            # Convert targets to the format expected by torchvision RPN
            # The RPN expects targets as a list of dictionaries with specific keys
            rpn_targets = []
            for i, target in enumerate(targets):
                boxes = target["boxes"]
                labels = target["labels"]
                if len(boxes.shape) == 3 and boxes.shape[0] == 1:
                    boxes = boxes.squeeze(0)
                    
                if len(labels.shape) == 2 and labels.shape[0] == 1:
                    labels = labels.squeeze(0)  # Remove first dimension
                
                rpn_target = {
                    "boxes": boxes,
                    "labels": labels
                }
                if not torch.isfinite(boxes).all() or (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any():
                    print("Invalid GT boxes detected:", boxes)
                    pdb.set_trace()
                rpn_targets.append(rpn_target)
            
            if not rpn_targets:
                print("Warning: No valid targets found, returning empty outputs")
                return {"scores": [], "bbox_deltas": [], "proposals": []}, {}

            proposals, rpn_losses = self.rpn(image_list, feature_dict, rpn_targets)
        else:
            proposals, _ = self.rpn(image_list, feature_dict)
            rpn_losses = {}

        # Process each image separately instead of concatenating
        all_scores = []
        all_bbox_deltas = []
        all_roi_features = []
        
        for i, props in enumerate(proposals):
            if props.shape[0] == 0:
                # No proposals for this image
                all_scores.append(torch.empty((0, self.head.cls_score.out_features), device=features.device))
                all_bbox_deltas.append(torch.empty((0, 4), device=features.device))
                all_roi_features.append(torch.empty((0, 256, 7, 7), device=features.device))
                continue
            
            # Process proposals for this specific image
            batch_idx = torch.full((props.shape[0], 1), i, device=props.device)
            rois = torch.cat([batch_idx, props], dim=1)
            
            # ROI Align for this image's proposals with proper spatial scale
            # Map image-space boxes to feature map coordinates
            img_h, img_w = image_list.image_sizes[i]
            feat_h, feat_w = features.shape[-2], features.shape[-1]
            spatial_scale = float(feat_w) / float(img_w)
            roi_feats = roi_align(features, rois, output_size=(7, 7), spatial_scale=spatial_scale, aligned = True)
            
            # Get scores and bbox deltas for this image
            scores, bbox_deltas = self.head(roi_feats)
            
            all_scores.append(scores)
            all_bbox_deltas.append(bbox_deltas)
            all_roi_features.append(roi_feats)
        
        if mode == 'train':
            # Return per-image outputs for training
            return {
                "scores": all_scores,  # List of tensors, one per image
                "bbox_deltas": all_bbox_deltas,  # List of tensors, one per image
                "proposals": proposals,  # List of tensors, one per image
                "roi_features": all_roi_features,  # List of tensors, one per image
                "pred_labels": [torch.argmax(s, dim=1) if s.shape[0] > 0 else torch.empty(0, dtype=torch.long, device=features.device) for s in all_scores]  # List of tensors, one per image
            }, rpn_losses
        else:
            # Apply NMS and thresholding for inference
            output = {"boxes": [], "labels": [], "scores": [], "roi_features": []}
            
            for i, (props, scores, bbox_deltas, roi_feats) in enumerate(zip(proposals, all_scores, all_bbox_deltas, all_roi_features)):
                if props.shape[0] == 0:
                    # Maintain alignment by appending empty predictions for this image
                    output["boxes"].append(torch.empty((0, 4), device=features.device))
                    output["labels"].append(torch.empty(0, dtype=torch.long, device=features.device))
                    output["scores"].append(torch.empty(0, device=features.device))
                    output["roi_features"].append(torch.empty((0, 256, 7, 7), device=features.device))
                    continue
                
                #Decode bbox deltas (dx, dy, dw, dh) relative to proposals
                #Convert proposals to center-size
                widths = props[:, 2] - props[:, 0]
                heights = props[:, 3] - props[:, 1]
                ctr_x = props[:, 0] + 0.5 * widths
                ctr_y = props[:, 1] + 0.5 * heights

                dx = bbox_deltas[:, 0]
                dy = bbox_deltas[:, 1]
                dw = bbox_deltas[:, 2]
                dh = bbox_deltas[:, 3]

                max_log_scale = 10.0   # e.g. exp(10) ~ 22026, choose smaller if necessary
                dw = torch.clamp(dw, min=-max_log_scale, max=max_log_scale)
                dh = torch.clamp(dh, min=-max_log_scale, max=max_log_scale)

                # Apply deltas
                pred_ctr_x = ctr_x + dx * widths
                pred_ctr_y = ctr_y + dy * heights
                # Clamp dw, dh for numerical stability
                # dw = torch.clamp(dw, max=4.0)
                # dh = torch.clamp(dh, max=4.0)
                pred_w = widths * torch.exp(dw)
                pred_h = heights * torch.exp(dh)

                x1 = pred_ctr_x - 0.5 * pred_w
                y1 = pred_ctr_y - 0.5 * pred_h
                x2 = pred_ctr_x + 0.5 * pred_w
                y2 = pred_ctr_y + 0.5 * pred_h

                b = torch.stack([x1, y1, x2, y2], dim=1)

                # Clip boxes to image size
                b[:, 0::2] = b[:, 0::2].clamp(min=0, max=img_w)
                b[:, 1::2] = b[:, 1::2].clamp(min=0, max=img_h)
                p = F.softmax(scores, dim=1)
                conf, lab = p.max(dim=1)
                
                # Apply score threshold
                # Filter low-confidence and background (class 0)
                keep = (conf > self.score_thresh) & (lab != 0)
                b, conf, lab, rf = b[keep], conf[keep], lab[keep], roi_feats[keep]
                
                if b.shape[0] > 0:
                    # Apply NMS
                    keep_nms = nms(b, conf, self.nms_thresh)
                    output["boxes"].append(b[keep_nms])
                    output["labels"].append(lab[keep_nms])
                    output["scores"].append(conf[keep_nms])
                    output["roi_features"].append(rf[keep_nms])
                else:
                    # No detections after thresholding
                    output["boxes"].append(torch.empty((0, 4), device=features.device))
                    output["labels"].append(torch.empty(0, dtype=torch.long, device=features.device))
                    output["scores"].append(torch.empty(0, device=features.device))
                    output["roi_features"].append(torch.empty((0, 256, 7, 7), device=features.device))
            
            return output, rpn_losses