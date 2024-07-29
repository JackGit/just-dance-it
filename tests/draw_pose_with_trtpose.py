import json
import trt_pose.coco
import trt_pose.models
import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import numpy as np
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from torch2trt import TRTModule

# 加载人类骨架数据
with open('./models/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

# 加载预训练模型
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, num_links).cuda().eval()
MODEL_WEIGHTS = './models/densenet121_baseline_att_256x256_B_epoch_160.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

# 优化模型为 TensorRT
from torch2trt import torch2trt
data = torch.zeros((1, 3, 224, 224)).cuda()
model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

# 解析和绘制对象
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

# 打开视频文件
video_path = '../videos/demo_01.mp4'
cap = cv2.VideoCapture(video_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    data = transform(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        cmap, paf = model_trt(data)
    
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)

    draw_objects(frame, counts, objects, peaks)
    
    cv2.imshow('TRT Pose', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
