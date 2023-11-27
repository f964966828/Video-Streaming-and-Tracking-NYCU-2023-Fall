import os
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from instance import Instance
from utils import *

# Define input video object
input_video_name = 'hard_9.mp4'
cap = cv2.VideoCapture(input_video_name)

# Define output video object
output_video_name = input_video_name[:-4] + '_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

# Define object detection detect_model
detect_model = YOLO("yolov8s.pt")

# Define reid model and load pretrained weight
device = ('cuda' if torch.cuda.is_available() else 'cpu')
reid_model = get_reid_model(device)
transform = get_transform()

# Run for each frame
instances = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(frame_count):
    _, frame = cap.read()
    
    result = detect_model.predict(frame, conf=0.6, iou=0.6, classes=0)[0]
    xyxy = result.boxes.xyxy.cpu().detach().numpy()

    bboxes = []
    embeddings = []
    for x1, y1, x2, y2 in xyxy:
        instance_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])

        embedding = reid_model(transform(instance_crop).unsqueeze(0).to(device))[0]
        embeddings.append(embedding.cpu().detach().numpy())
    
    current_instances = []
    if len(instances) == 0 and len(embeddings) != 0:
        for bbox, embedding in zip(bboxes, embeddings):
            instance = Instance(id_=len(instances)+1, embedding=embedding, bbox=bbox, last_frame=i)
            instances.append(instance)
            current_instances.append(instance)
    elif len(embeddings) != 0:
        cost = cost_matrix(instances, embeddings, bboxes, i, width, height)
        rows, cols = linear_sum_assignment(cost)

        for r, c in zip(rows, cols):
            if cost[r, c] < 0.0:
                instances[r].bbox = bboxes[c]
                instances[r].embedding = embeddings[c]
                instances[r].last_frame = i
                current_instances.append(instances[r])
            else:
                instance = Instance(id_=len(instances)+1, embedding=embeddings[c], bbox=bboxes[c], last_frame=i)
                instances.append(instance)
                current_instances.append(instance)
    
    for instance in current_instances:
        x1, y1, x2, y2 = instance.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), instance.color, 2)
        cv2.putText(frame, str(instance.id_), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(frame, f"count: {len(instances)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    out.write(frame)

print(f"count: {len(instances)}")

cap.release()
out.release()
