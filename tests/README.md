## draw pose

1. mediapipe: is slow, not able to use GPU;
2. hrnetv2_w32_imagenet_pretrained, failed to draw anything;
3. PoseEstimationWithMobileNet not able to run due to module import issue;
4. openpose, installing
5. trt_pose, install tensorrt, torch2trt, then trt_pose
    installing torch2trt but failed with cuda var
6. Detectron2, by default is slow


20240729 测试结果：
1. mediapipe 效果最稳，但是帧率不足 20，而且不支持 GPU
2. detectron2 应该是效果最好的，但是使用了 GPU，帧率不足 20，而且效果不稳定（抖动i很强烈）
3. trt_pose 未安装成功
4. openpose 未安装成功