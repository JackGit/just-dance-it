import sys
import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np

# 添加仓库路径到 Python 路径

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

# 加载预训练模型
model = PoseEstimationWithMobileNet()
checkpoint = torch.load('./models/checkpoint_iter_370000.pth', map_location='cpu')
load_state(model, checkpoint)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# 预处理函数
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 后处理函数
def postprocess(heatmaps):
    keypoints = []
    for heatmap in heatmaps:
        _, _, h, w = heatmap.shape
        heatmap = heatmap.reshape(h, w)
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        keypoints.append((x, y))
    return keypoints

# 打开视频文件
video_path = 'path_to_your_dance_video.mp4'
cap = cv2.VideoCapture(video_path)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将 BGR 图像转换为 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # 预处理图像并移动到 GPU
    input_tensor = preprocess(image_pil)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    # 进行推理
    with torch.no_grad():
        heatmaps = model(input_tensor)
    
    # 后处理以获取关键点
    keypoints = postprocess(heatmaps.cpu().numpy())
    
    # 在帧上绘制关键点
    for x, y in keypoints:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # 计算并显示 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # 显示帧
    cv2.imshow('Dance Pose', frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
