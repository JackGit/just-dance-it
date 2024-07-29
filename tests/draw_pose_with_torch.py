import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
from torchvision import models  # 导入 models 模块

# 定义 HRNet 模型架构
class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(in_features=2048, out_features=34)  # 17 个关键点，每个关键点包含 x 和 y 坐标

    def forward(self, x):
        return self.model(x)

# 加载 HRNet 模型和权重
model = HRNet()
state_dict = torch.load('./models/hrnetv2_w32_imagenet_pretrained.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)  # strict=False 忽略缺失的 key
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# 预处理函数
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 192)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 后处理函数
def postprocess(output):
    keypoints = output.view(-1, 2).detach().cpu().numpy()
    return keypoints

# 打开视频文件
video_path = '../videos/demo_01.mp4'
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
        output = model(input_tensor)
    
    # 后处理以获取关键点
    keypoints = postprocess(output)
    
    # 在帧上绘制关键点
    for i in range(keypoints.shape[0]):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
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
