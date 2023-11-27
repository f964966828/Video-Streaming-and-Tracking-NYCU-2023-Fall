import numpy as np
import torch
import torchreid
from torchvision import transforms

def get_reid_model(device):
    reid_model = torchreid.models.osnet_ain.osnet_ain_x1_0()
    reid_model = reid_model.to(device)
    reid_model.classifier = None

    state_dict = torch.load('osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  
        if not 'classifier' in new_key:
            new_state_dict[new_key] = value
    reid_model.load_state_dict(new_state_dict)

    reid_model.classifier = torch.nn.Identity()
    reid_model.eval()
    return reid_model

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def giou(bbox1, bbox2):
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area

    # Find the smallest enclosing box
    enclose_x_min = min(x1_min, x2_min)
    enclose_y_min = min(y1_min, y2_min)
    enclose_x_max = max(x1_max, x2_max)
    enclose_y_max = max(y1_max, y2_max)
    enclose_area = (enclose_x_max - enclose_x_min) * (enclose_y_max - enclose_y_min)

    # Calculate GIoU
    giou = inter_area / union_area - (enclose_area - union_area) / enclose_area

    return giou

def cost_matrix(instances, embeddings, bboxes, current_frame, width, height):
    cost = np.zeros((len(instances), len(embeddings)))
    for i in range(len(instances)):
        for j in range(len(embeddings)):
            cost[i][j] -= cosine_similarity(instances[i].embedding, embeddings[j])
            cost[i][j] -= giou(instances[i].bbox, bboxes[j])
        
        near_boundary = False
        x1, y1, x2, y2 = instances[i].bbox
        if x1 < 10 or x2 > width - 10:
            near_boundary = True
        if y1 < 10 or y2 > height - 10:
            near_boundary = True
        if current_frame - instances[i].last_frame > 5 and near_boundary:
            cost[i] = np.zeros(len(embeddings))
    return cost
