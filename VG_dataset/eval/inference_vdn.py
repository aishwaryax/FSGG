import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath('.'))

import time
from vdn.VDN import visual_rel_model
import cv2
import torch
import numpy as np
import json
import copy
from tqdm import tqdm
from ckn.CKN import FC_Net
from yolo_dataset import create_dataloader
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, colorstr
from utils.torch_utils import select_device

from evaluation.bounding_box import BoxList
from evaluation.general import simi_direction_suppression,relation_search,objects_search_relation
import torchvision.transforms as transforms
from evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall
import torchtext
from datapath import image_file,test_object_yolo_label_dir,test_relation_label_dir,names,relations_names

# Define the paths and model weights
image_path = 'eval/110.jpg'
weights = 'vdn/VDN+EPBS.pt'
object_weights = 'yolol_object_test_28.pt'

mode = 'sgdet' #predcls sgdet

opt_device = '0' if torch.cuda.is_available() else 'cpu'
nc = 151
opt_imgsz = 640

source = 'image'
opt_augment = True

conf_thres = 0.01
iou_thres = 0.45
agnostic_nms = True
obj_count_limit = 80

batch_size = 1
single_cls = False
save_hybrid = False
opt_line_thickness =3

iou_types = ['bbox','relations']

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the image and preprocess it
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to the model input size
    transforms.ToTensor(),  # Convert to tensor
])

image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

VDN_model = visual_rel_model()
model_dict = VDN_model.state_dict()
pretrained_dict = torch.load(VDN_weight, map_location=device)
pretrained_dict = pretrained_dict['model']
model_dict.update(pretrained_dict)
VDN_model.load_state_dict(model_dict)
VDN_model.to(device)

object_model = torch.load(object_weights, map_location=device)['model']
object_model.to(device)
object_model.eval()


word2vec = torchtext.vocab.GloVe(name='6B', dim=50)
word_vec_list = []
for name_index in range(len(names)):
    word_vec = word2vec.get_vecs_by_tokens(names[name_index], lower_case_backup=True)
    word_vec_list.append(word_vec)
word_vec_list = torch.stack(word_vec_list).to(device)


# Perform inference
with torch.no_grad():
    nb, _, height, width = image.shape  # batch size, channels, height, width
    object_model = object_model.to(torch.float32)
    object_pred = object_model(image, augment=False)[0]
    obj_pred = non_max_suppression(object_pred, conf_thres=0.01, iou_thres=0.45, classes=None, agnostic=False)
    for i, det in enumerate(obj_pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image.shape).round()

            det = det[det[:, 4].sort(descending=True)[1]] #sorted based on confidence
            obj_pred = det

    predited_obj = torch.cat((obj_pred[:,5].unsqueeze(1),(obj_pred[:,0]/width).unsqueeze(1),(obj_pred[:,1]/height).unsqueeze(1),(obj_pred[:,2]/width).unsqueeze(1),(obj_pred[:,3]/height).unsqueeze(1)),1)
    obj_location_feature =  torch.cat(( ((predited_obj[:,3]-predited_obj[:,1])/2).unsqueeze(1),((predited_obj[:,4]-predited_obj[:,2])/2).unsqueeze(1),predited_obj[:,1:]),1)
    obj_word_feature = word_vec_list[predited_obj[:, 0].long()]
    predited_obj_confidence = obj_pred[:,4:5]
    #obj = torch.cat((obj_word_feature, obj_location_feature), 1)
    obj = torch.cat((obj_word_feature, obj_location_feature.repeat((1, 5))), 1)
    obj_id = torch.arange((len(predited_obj)))
    a_id = obj_id.repeat(len(obj_id),1).view(-1,)
    b_id = torch.repeat_interleave(obj_id,len(obj_id),dim=0)
    obj_pair = torch.cat((a_id.unsqueeze(1),b_id.unsqueeze(1)),dim=1)
    obj_pair = obj_pair[obj_pair[:,0]!=obj_pair[:,1]]

    #[word_vector,c_x,c_y,x1,y1,x2,y2 ,word_vector,c_x,c_y,x1,y1,x2,y2, dc_x, dc_y, dx1,dx2,dy2,dy2]
    rel_feature = torch.cat((obj[obj_pair[:, 0].long()], obj[obj_pair[:, 1].long()], obj[obj_pair[:, 1].long()][:, 50:] - obj[obj_pair[:, 0].long()][:, 50:]),1)
    obj_pair_confidence =  torch.cat((predited_obj_confidence[obj_pair[:, 0].long()], predited_obj_confidence[obj_pair[:, 1].long()]),1)
    #obj_pred[:,:4] = xyxy2xywh(obj_pred[:,:4])

    pred_relations = model(rel_feature)
    pred_relations = torch.sigmoid(pred_relations)
    pred_relations_conf, pred_relations = pred_relations[:,0:1],pred_relations[:,1:]

    pred_relations = torch.cat([torch.tensor([[0] for i in range(len(pred_relations))], device=device), pred_relations], dim=1)


    #ingore later 40 predicate
    pred_relations_values, pred_relation_indices = torch.sort(pred_relations, 1, descending=True)
    indices = torch.arange(0,len(pred_relations_values),device=device)
    indices = torch.repeat_interleave(indices, len(pred_relation_indices[:, 10:][0]), dim=0)
    _pred_relation_indices = torch.cat((indices.reshape(-1,1),pred_relation_indices[:, 10:].reshape(-1,1)),1)
    pred_relations[_pred_relation_indices[:,0],_pred_relation_indices[:,1]] = 0.0


    pred_relations = pred_relations * pred_relations_conf
    relation_max, relation_argmax = torch.max(pred_relations, dim=1)

    obj_pair_confidence = torch.cat((obj_pair_confidence, relation_max.unsqueeze(1)),1)
    obj_pair_confidence = obj_pair_confidence[:,0]*obj_pair_confidence[:,1]*obj_pair_confidence[:,2]

    head_semantic = torch.cat([obj_pair[:,:2].to(device),relation_argmax.view(-1, 1), relation_max.view(-1, 1)], dim=1)
    
    head_semantic = head_semantic[torch.argsort(head_semantic[:,3],descending=True)]
    head_semantic = head_semantic[torch.argsort(obj_pair_confidence, descending=True)]

    # Assuming obj_pred contains the detected objects
    detected_obj_ids = set()  # Store IDs of detected objects
    for obj in obj_pred:
        detected_obj_ids.add(int(obj[5]))  # Assuming the label index is at index 5

    # Filter relations for objects that appear in the image
    filtered_head_semantic = []
    for triplet in head_semantic:
        obj1_idx, rel_idx, obj2_idx = int(triplet[0].item()), int(triplet[2].item()), int(triplet[1].item())
        if obj1_idx in detected_obj_ids and obj2_idx in detected_obj_ids:
            obj1_label = names[obj1_idx]
            obj2_label = names[obj2_idx]
            rel_label = relations_names[rel_idx]
            filtered_head_semantic.append((obj1_label, rel_label, obj2_label, triplet[3].item()))

    # Print the filtered triplets and their confidences
    for triplet in filtered_head_semantic:  # Show top 10 relations
        obj1_label, rel_label, obj2_label, confidence = triplet
        print(f"{obj1_label} --> {rel_label} --> {obj2_label}, Confidence: {confidence}")
