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
    def __init__(self, model_name='facebook/dinov2-base'):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.out_channels = self.model.config.hidden_size
    def forward(self, x):
        # Get the last hidden state from DINOv2
        outputs = self.model(x)
        features = outputs.last_hidden_state  # (B, N, C)
        
        # Remove the CLS token (first token) and reshape to spatial dimensions
        features = features[:, 1:, :]  # (B, N-1, C)
        B, N, C = features.shape
        
        # Calculate spatial dimensions (DINOv2 outputs square features)
        H = W = int(N ** 0.5)
        
        # Ensure we have the right number of tokens for a square feature map
        if H * W != N:
            # If not a perfect square, pad or truncate
            target_size = H * W
            if N > target_size:
                features = features[:, :target_size, :]
            else:
                # Pad with zeros if needed
                padding = target_size - N
                features = torch.cat([features, torch.zeros(B, padding, C, device=features.device)], dim=1)
        
        # Reshape to spatial format (B, C, H, W)
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
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        self.rpn_head = RPNHead(256, self.rpn_anchor_generator.num_anchors_per_location()[0])
        self.rpn = RegionProposalNetwork(
            self.rpn_anchor_generator,
            self.rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7
        )
        self.head = DetectionHead(256, num_classes)
        self.nms_thresh = 0.4
        self.score_thresh = 0.1

    def forward(self, images, targets=None, mode='train'):
        features = self.backbone(images)
        features = self.fpn(features)
        
        # Create proper feature dictionary format for RPN
        # Based on the error, the RPN expects features as a single tensor, not a list
        # The RPN head applies conv2d directly to the feature tensor
        feature_dict = {"0": features}
        
        # Create ImageList object for RPN
        image_list = ImageList(images, [tuple(img.shape[-2:]) for img in images])
        
        if mode == 'train':
            # Convert targets to the format expected by torchvision RPN
            # The RPN expects targets as a list of dictionaries with specific keys
            rpn_targets = []
            for i, target in enumerate(targets):
                boxes = target["boxes"]
                labels = target["labels"]

                # Fix dimension issue: remove extra batch dimension if present
                if len(boxes.shape) == 3 and boxes.shape[0] == 1:
                    boxes = boxes.squeeze(0)  # Remove first dimension
                
                # Check if boxes are in correct format (should be xyxy with 4 values)
                if boxes.shape[-1] != 4:
                    print(f"Warning: Image {i} boxes have {boxes.shape[-1]} values, expected 4")
                    print(f"Boxes content: {boxes}")
                    
                if len(labels.shape) == 2 and labels.shape[0] == 1:
                    labels = labels.squeeze(0)  # Remove first dimension
                
                rpn_target = {
                    "boxes": boxes,
                    "labels": labels
                }
                rpn_targets.append(rpn_target)
            
            if not rpn_targets:
                print("Warning: No valid targets found, returning empty outputs")
                return {"scores": [], "bbox_deltas": [], "proposals": []}, {}

            
            proposals, rpn_losses = self.rpn(image_list, feature_dict, rpn_targets)
        else:
            proposals, _ = self.rpn(image_list, feature_dict)
            rpn_losses = {}

        # ROI Align
        roi_features = []
        proposal_list = []
        for i, props in enumerate(proposals):
            if props.shape[0] == 0:
                continue
            batch_idx = torch.full((props.shape[0], 1), i, device=props.device)
            rois = torch.cat([batch_idx, props], dim=1)
            proposal_list.append(rois)
        
        if len(proposal_list) == 0:
            if mode == 'train':
                return {"scores": [], "bbox_deltas": [], "proposals": proposals}, rpn_losses
            else:
                return {}, rpn_losses
        
        proposal_cat = torch.cat(proposal_list, dim=0)
        roi_feats = roi_align(features, proposal_cat, output_size=(7, 7))
        scores, bbox_deltas = self.head(roi_feats)
        
        if mode == 'train':
            # Return raw outputs for training
            return {
                "scores": scores,
                "bbox_deltas": bbox_deltas,
                "proposals": proposals,
                "roi_features": roi_feats,
                "pred_labels": torch.argmax(scores, dim=1)  # Add predicted class labels
            }, rpn_losses
        else:
            # Apply NMS and thresholding for inference
            output = {"boxes": [], "labels": [], "scores": [], "roi_features": []}
            start = 0
            for i, props in enumerate(proposals):
                num = props.shape[0]
                if num == 0:
                    continue
                s = scores[start:start+num]
                b = props + bbox_deltas[start:start+num]
                p = F.softmax(s, dim=1)
                conf, lab = p.max(dim=1)
                keep = conf > self.score_thresh
                b, conf, lab, rf = b[keep], conf[keep], lab[keep], roi_feats[start:start+num][keep]
                
                if b.shape[0] > 0:
                    keep_nms = nms(b, conf, self.nms_thresh)
                    output["boxes"].append(b[keep_nms])
                    output["labels"].append(lab[keep_nms])
                    output["scores"].append(conf[keep_nms])
                    output["roi_features"].append(rf[keep_nms])
                start += num
            
            return output, rpn_losses