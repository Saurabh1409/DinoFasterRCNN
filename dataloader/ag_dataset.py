import torch
import torch.nn as nn
import pickle
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from constant import const
from transformers import AutoImageProcessor, AutoModel
import pdb

class ActionGenomeDataset(Dataset):
    def __init__(self, data_path, phase = 'train', datasize = 'full', filter_nonperson_box_frame = True, filter_small_box = False):
        self.data_path = data_path
        self.phase = phase
        self.datasize = datasize
        self.filter_nonperson_box_frame = filter_nonperson_box_frame
        self.filter_small_box = filter_small_box
        self.frames_path = os.path.join(self.data_path,const.FRAMES)

        print("---------       Loading Annotation Files       ---------")
        self._fetch_object_classes()
        self.person_bbox, self.object_bbox = self._fetch_object_person_bboxes(filter_small_box)

        print("---------       Building Dataset       ---------")
        self._build_dataset()

        print("---------       Initializing Processor       ---------")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base") #facebook/dinov3-vit7b16-pretrain-lvd1689m , facebook/dinov2-base

        print(f"Dataset initialized with {len(self.samples)} frames")
        print(f"Object classes: {len(self.object_classes)}")
    
    def _fetch_object_classes(self):
        """Load object class names from file"""
        self.object_classes = [const.BACKGROUND]
        object_classes_path = os.path.join(self.data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE)
        
        with open(object_classes_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)

        # Apply class name corrections as in reference
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def _fetch_object_person_bboxes(self, filter_small_box = False):
        """Load person and object bounding box annotations"""
        annotations_path = os.path.join(self.data_path, const.ANNOTATIONS)
        with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
            person_bbox = pickle.load(f)
        with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
            object_bbox = pickle.load(f)

        return person_bbox, object_bbox

    def _build_dataset(self):
        self.samples = []
        self.valid_nums = 0
        self.non_gt_human_nums = 0
        
        for frame_name in self.person_bbox.keys():
            if self.object_bbox[frame_name][0][const.METADATA][const.SET] != self.phase:
                continue

            person_boxes = self.person_bbox[frame_name][const.BOUNDING_BOX]

            # Filter frames without person boxes if required
            if self.filter_nonperson_box_frame and len(person_boxes) == 0:
                self.non_gt_human_nums += 1
                continue

            # person_boxes= np.array([
            #             person_boxes[0,0]-person_boxes[0,2],
            #             person_boxes[0,1]-person_boxes[0,3],
            #             person_boxes[0,0]+person_boxes[0,2],
            #             person_boxes[0,1]+person_boxes[0,3]
            #         ])

            objects = []
            for obj in self.object_bbox[frame_name]:
                if obj[const.VISIBLE] and obj[const.BOUNDING_BOX] is not None:
                    class_idx = self.object_classes.index(obj[const.CLASS])

                    bbox = obj[const.BOUNDING_BOX]
                    bbox_xyxy = np.array([
                        bbox[0],bbox[1],
                        bbox[0]+bbox[2],
                        bbox[1]+bbox[3]
                    ])

                    objects.append({
                        'bbox':bbox_xyxy,
                        'class':class_idx
                    })

            # Add sample if it has valid annotations
            if len(objects) > 0 or len(person_boxes) > 0:
                self.samples.append({
                    'filename': frame_name,
                    'person_boxes': person_boxes,
                    'objects': objects
                })
                self.valid_nums += 1   

        print(f"Built dataset with {self.valid_nums} valid frames\n")
        print(f"Removed {self.non_gt_human_nums} frames without person boxes\n")     
    def __len__(self):
        return(len(self.samples))
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.frames_path,sample['filename'])

        # Load original image and keep original size for box scaling
        pil_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        if orig_w<orig_h:
            new_w=224
            new_h = int(orig_h * 224/orig_w)
        else:
            new_h = 224
            new_w = int(orig_w * 224/orig_h)
        # Process image with default DINOv2 processor settings

        image = self.processor(images = pil_image, return_tensors="pt")["pixel_values"].squeeze(0)


        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        offset_x = (new_w - 224)//2
        offset_y = (new_h - 224)//2
        boxes = []
        labels = []
        sample['person_boxes'] = sample['person_boxes'].reshape(1,4)
        for person_box in sample['person_boxes']:
            # Assume person_box is in xyxy; scale to processed image size
            #pb = np.array(person_box, dtype=np.float32)
            pb_scaled = np.array([person_box[0]*scale_x, person_box[1]*scale_y, person_box[2]*scale_x, person_box[3]*scale_y], dtype=np.float32)
            pb_scaled[0::2]-=offset_x
            pb_scaled[1::2]-=offset_y
            pb_scaled[0::2] = np.clip(pb_scaled[0::2],0,224)
            pb_scaled[1::2] = np.clip(pb_scaled[1::2],0,224)
            if pb_scaled[2]-pb_scaled[0]<1 or pb_scaled[3]-pb_scaled[1]<1:
                continue
            boxes.append(pb_scaled)
            labels.append(1)

        for obj in sample["objects"]:
            ob = np.array(obj["bbox"], dtype=np.float32)
            ob_scaled = np.array([ob[0]*scale_x, ob[1]*scale_y, ob[2]*scale_x, ob[3]*scale_y], dtype=np.float32)
            ob_scaled[0::2]-=offset_x
            ob_scaled[1::2]-=offset_y
            ob_scaled[0::2] = np.clip(ob_scaled[0::2],0,224)
            ob_scaled[1::2] = np.clip(ob_scaled[1::2],0,224)
            if ob_scaled[2]-ob_scaled[0]<1 or ob_scaled[3]-ob_scaled[1]<1:
                continue
            boxes.append(ob_scaled)
            labels.append(obj["class"])


        target = {
            "boxes":torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0,4),dtype=torch.float32),
            "labels":torch.tensor(labels,dtype=torch.int64) if labels else torch.empty((0,),dtype=torch.int64)
        }
        # if len(boxes) > 0:
        #     target["boxes"] = target["boxes"] / 224.0  
        return image, target

def collate_fn(batch):
    image = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return image, target
# AG_Dataset = ActionGenomeDataset(const.DATA_PATH)

# train_dataloader = DataLoader(AG_Dataset,batch_size = 2, shuffle=False)
# for batch_idx, (image,taget) in enumerate(train_dataloader):
#     print(f" BATCH NUM : {batch_idx} : \n")
#     pdb.set_trace()
   
