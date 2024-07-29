import cv2
import torch
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 设置配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试时的阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(cfg.MODEL.DEVICE)

predictor = DefaultPredictor(cfg)

cap = cv2.VideoCapture('../videos/demo_01.mp4')

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    outputs = predictor(frame)
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    frame = v.get_image()[:, :, ::-1].copy()  # 确保 frame 是一个 NumPy 数组
    
    # 在帧上显示 FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Detectron2 Pose', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
