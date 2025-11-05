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

class DINOv2BackboneMultiLayer(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base', out_indices=[3, 6, 9, 11]):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.out_indices = out_indices
        self.out_channels = self.model.config.hidden_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.final_ln = getattr(self.model, "layernorm", None)
        self.model.eval()
        
    def forward(self, x):
        """
        Extract features from multiple layers of DINOv2 backbone
        Returns a dictionary with keys '0', '1', '2', '3' for each layer
        """
        outputs = self.model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Extract features from specified layers
        multi_layer_features = {}
        B, _, H, W = x.shape
        
        for i, idx in enumerate(self.out_indices):
            # Get hidden states from the specified layer
            hs = hidden_states[idx + 1]  # +1 because hidden_states includes input
            
            # Apply final layer norm if available
            if self.final_ln is not None:
                hs = self.final_ln(hs)
            
            # Remove CLS token and register tokens
            patch_tokens = hs[:, 1 + self.num_register_tokens:, :]  # [B, H_p*W_p, C]
            
            # Calculate patch dimensions
            patch_size = self.model.config.patch_size
            Hp, Wp = H // patch_size, W // patch_size
            
            # Reshape to spatial dimensions
            features = patch_tokens.unflatten(1, (Hp, Wp)).permute(0, 3, 1, 2).contiguous()
            
            # Store in dictionary with string key
            multi_layer_features[str(i)] = features
            
        return multi_layer_features

class MultiLayerFPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # Create lateral and output convolutions for each layer
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) for _ in range(4)
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)
        ])
        
    def forward(self, multi_layer_features):
        """
        Apply FPN processing to each feature map
        """
        processed_features = {}
        for i, (key, features) in enumerate(multi_layer_features.items()):
            x = self.lateral_convs[i](features)
            x = self.output_convs[i](x)
            processed_features[key] = x
        return processed_features

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

class MultiLayerDINOv2Detector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = DINOv2BackboneMultiLayer(out_indices=[3, 6, 9, 11])
        self.fpn = MultiLayerFPN(self.backbone.out_channels)

        for name, params in self.backbone.named_parameters():
            params.requires_grad_(False)
        self.backbone.eval()
        
        # Multi-scale anchor generator for 4 different feature maps
        self.rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128), (64, 128, 256), (128, 256, 512), (256, 512, 1024)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )
        
        # RPN head for multi-scale features
        self.rpn_head = RPNHead(256, self.rpn_anchor_generator.num_anchors_per_location()[0])
        
        # Region Proposal Network
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

    def select_feature_map(self, proposals, multi_layer_features):
        """
        Select the most appropriate feature map for each proposal based on size
        """
        selected_features = []
        selected_rois = []
        
        for i, props in enumerate(proposals):
            if props.shape[0] == 0:
                selected_features.append(torch.empty((0, 256, 7, 7), device=props.device))
                selected_rois.append(torch.empty((0, 5), device=props.device))
                continue
                
            # Calculate proposal areas
            areas = (props[:, 2] - props[:, 0]) * (props[:, 3] - props[:, 1])
            
            # Select feature map based on proposal size
            # Smaller proposals -> finer features, larger proposals -> coarser features
            feature_map_indices = []
            for area in areas:
                if area < 32 * 32:
                    feature_map_indices.append(0)  # Finest features
                elif area < 64 * 64:
                    feature_map_indices.append(1)
                elif area < 128 * 128:
                    feature_map_indices.append(2)
                else:
                    feature_map_indices.append(3)  # Coarsest features
            
            # Process each proposal with its selected feature map
            batch_rois = []
            batch_features = []
            
            for j, (prop, feat_idx) in enumerate(zip(props, feature_map_indices)):
                # Get the appropriate feature map
                feat_map = multi_layer_features[str(feat_idx)][i:i+1]  # [1, C, H, W]
                
                # Create ROI tensor [batch_idx, x1, y1, x2, y2]
                roi = torch.cat([torch.tensor([[0]], device=prop.device), prop.unsqueeze(0)], dim=1)
                batch_rois.append(roi)
                
                # ROI align
                roi_feat = roi_align(feat_map, roi, output_size=(7, 7), spatial_scale=1.0, aligned=True)
                batch_features.append(roi_feat)
            
            if batch_rois:
                selected_rois.append(torch.cat(batch_rois, dim=0))
                selected_features.append(torch.cat(batch_features, dim=0))
            else:
                selected_rois.append(torch.empty((0, 5), device=props.device))
                selected_features.append(torch.empty((0, 256, 7, 7), device=props.device))
        
        return selected_features, selected_rois

    def forward(self, images, targets=None, mode='train'):
        # Extract multi-layer features
        multi_layer_features = self.backbone(images)
        
        # Apply FPN to each feature map
        processed_features = self.fpn(multi_layer_features)
        
        # Create image list for RPN
        image_sizes = [(images.shape[-2], images.shape[-1]) for _ in range(images.shape[0])]
        image_list = ImageList(images, image_sizes)
        
        if mode == 'train':
            # Convert targets to the format expected by torchvision RPN
            rpn_targets = []
            for i, target in enumerate(targets):
                boxes = target["boxes"]
                labels = target["labels"]
                if len(boxes.shape) == 3 and boxes.shape[0] == 1:
                    boxes = boxes.squeeze(0)
                    
                if len(labels.shape) == 2 and labels.shape[0] == 1:
                    labels = labels.squeeze(0)
                
                rpn_target = {
                    "boxes": boxes,
                    "labels": labels
                }
                rpn_targets.append(rpn_target)
            
            if not rpn_targets:
                print("Warning: No valid targets found, returning empty outputs")
                return {"scores": [], "bbox_deltas": [], "proposals": []}, {}

            proposals, rpn_losses = self.rpn(image_list, processed_features, rpn_targets)
        else:
            proposals, _ = self.rpn(image_list, processed_features)
            rpn_losses = {}

        # Select appropriate feature maps for each proposal
        selected_features, selected_rois = self.select_feature_map(proposals, processed_features)
        
        # Process each image separately
        all_scores = []
        all_bbox_deltas = []
        all_roi_features = []
        
        for i, (props, roi_feats) in enumerate(zip(proposals, selected_features)):
            if props.shape[0] == 0:
                all_scores.append(torch.empty((0, self.head.cls_score.out_features), device=images.device))
                all_bbox_deltas.append(torch.empty((0, 4), device=images.device))
                all_roi_features.append(torch.empty((0, 256, 7, 7), device=images.device))
                continue
            
            # Get scores and bbox deltas
            scores, bbox_deltas = self.head(roi_feats)
            
            all_scores.append(scores)
            all_bbox_deltas.append(bbox_deltas)
            all_roi_features.append(roi_feats)
        
        if mode == 'train':
            return {
                "scores": all_scores,
                "bbox_deltas": all_bbox_deltas,
                "proposals": proposals,
                "roi_features": all_roi_features,
                "pred_labels": [torch.argmax(s, dim=1) if s.shape[0] > 0 else torch.empty(0, dtype=torch.long, device=images.device) for s in all_scores]
            }, rpn_losses
        else:
            # Apply NMS and thresholding for inference
            output = {"boxes": [], "labels": [], "scores": [], "roi_features": []}
            
            for i, (props, scores, bbox_deltas, roi_feats) in enumerate(zip(proposals, all_scores, all_bbox_deltas, all_roi_features)):
                if props.shape[0] == 0:
                    output["boxes"].append(torch.empty((0, 4), device=images.device))
                    output["labels"].append(torch.empty(0, dtype=torch.long, device=images.device))
                    output["scores"].append(torch.empty(0, device=images.device))
                    output["roi_features"].append(torch.empty((0, 256, 7, 7), device=images.device))
                    continue
                
                # Decode bbox deltas
                widths = props[:, 2] - props[:, 0]
                heights = props[:, 3] - props[:, 1]
                ctr_x = props[:, 0] + 0.5 * widths
                ctr_y = props[:, 1] + 0.5 * heights

                dx = bbox_deltas[:, 0]
                dy = bbox_deltas[:, 1]
                dw = bbox_deltas[:, 2]
                dh = bbox_deltas[:, 3]

                max_log_scale = 10.0
                dw = torch.clamp(dw, min=-max_log_scale, max=max_log_scale)
                dh = torch.clamp(dh, min=-max_log_scale, max=max_log_scale)

                pred_ctr_x = ctr_x + dx * widths
                pred_ctr_y = ctr_y + dy * heights
                pred_w = widths * torch.exp(dw)
                pred_h = heights * torch.exp(dh)

                x1 = pred_ctr_x - 0.5 * pred_w
                y1 = pred_ctr_y - 0.5 * pred_h
                x2 = pred_ctr_x + 0.5 * pred_w
                y2 = pred_ctr_y + 0.5 * pred_h

                b = torch.stack([x1, y1, x2, y2], dim=1)

                # Clip boxes to image size
                img_h, img_w = image_sizes[i]
                b[:, 0::2] = b[:, 0::2].clamp(min=0, max=img_w)
                b[:, 1::2] = b[:, 1::2].clamp(min=0, max=img_h)
                
                p = F.softmax(scores, dim=1)
                conf, lab = p.max(dim=1)
                
                # Apply score threshold
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
                    output["boxes"].append(torch.empty((0, 4), device=images.device))
                    output["labels"].append(torch.empty(0, dtype=torch.long, device=images.device))
                    output["scores"].append(torch.empty(0, device=images.device))
                    output["roi_features"].append(torch.empty((0, 256, 7, 7), device=images.device))
            
            return output, rpn_losses