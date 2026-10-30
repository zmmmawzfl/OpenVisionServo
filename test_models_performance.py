#!/usr/bin/env python3
"""
YOLO-World 模型性能对比测试 (640 vs 320)
自适应处理带 txt_feats 和不带 txt_feats 的模型
"""

import time
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# ============ 配置 ============
MODEL_640 = "yolov8s-worldv2.onnx"        # 原始 640x640 模型
MODEL_320 = "yolov8s-worldv2_320.onnx"    # 新导出的 320x320 模型
WARMUP_ITERATIONS = 5
BENCHMARK_ITERATIONS = 30
CONF_THRESHOLD = 0.3

# COCO 类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# ============ 预处理函数 ============
def preprocess(frame, target_size):
    """将摄像头帧 (RGB) 转换为模型输入 (1,3,target_size,target_size) 归一化到 [0,1]"""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))        # HWC -> CHW
    img = img[np.newaxis, ...]                # batch
    return img

def create_dummy_input(inp):
    """根据输入信息生成 dummy 数据"""
    shape = inp.shape
    resolved = []
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
        elif dim == 'batch':
            resolved.append(1)
        elif dim == 'num_classes':
            resolved.append(80)   # 默认 COCO 80 类
        elif dim == 'height' or dim == 'width':
            resolved.append(640)  # fallback，但 images 不在这里处理
        else:
            resolved.append(1)
    # 如果是 txt_feats，生成归一化的随机特征
    if inp.name == 'txt_feats':
        num_classes = resolved[1] if len(resolved) > 1 else 80
        feat_dim = resolved[2] if len(resolved) > 2 else 512
        data = np.random.randn(1, num_classes, feat_dim).astype(np.float32)
        norms = np.linalg.norm(data, axis=2, keepdims=True)
        data = data / (norms + 1e-8)
        return data
    else:
        # 其他未知输入，用随机数据
        return np.random.randn(*resolved).astype(np.float32)

# ============ 后处理函数（简化） ============
def postprocess(outputs, conf_threshold=0.3):
    """解析 ONNX 输出，提取边界框和得分（假设输出为 scores, boxes）"""
    if len(outputs) < 2:
        return []
    scores = np.squeeze(outputs[0], axis=0)   # (num_classes, num_anchors)
    boxes = np.squeeze(outputs[1], axis=0)    # (4, num_anchors)
    scores = scores.T                         # (num_anchors, num_classes)
    boxes = boxes.T                           # (num_anchors, 4)
    detections = []
    for i in range(scores.shape[0]):
        class_id = np.argmax(scores[i])
        confidence = scores[i][class_id]
        if confidence < conf_threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        detections.append((class_id, confidence, (x1, y1, x2, y2)))
    return detections

# ============ 测试单个模型 ============
def test_model(sess, inputs, frame, target_size, iterations, warmup):
    """测试模型推理时间，自动构建 feed_dict"""
    feed_dict = {}
    for inp in inputs:
        if inp.name == 'images':
            feed_dict[inp.name] = preprocess(frame, target_size)
        elif inp.name == 'txt_feats':
            feed_dict[inp.name] = create_dummy_input(inp)
        else:
            print(f"⚠️ 未知输入名称: {inp.name}，使用随机数据")
            feed_dict[inp.name] = create_dummy_input(inp)

    # 预热
    print(f"  预热 {warmup} 次...")
    for _ in range(warmup):
        sess.run(None, feed_dict)

    # 正式测试
    print(f"  测试 {iterations} 次...")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        outputs = sess.run(None, feed_dict)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if (i+1) % 10 == 0:
            print(f"    进度: {i+1}/{iterations}")

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time

    # 解析检测结果
    detections = postprocess(outputs, CONF_THRESHOLD)
    print(f"  检测到 {len(detections)} 个物体（置信度 > {CONF_THRESHOLD}）")
    if len(detections) > 0:
        for d in detections[:3]:
            class_id, conf, box = d
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            print(f"    - {class_name}: {conf:.2f}, 框={box}")

    return avg_time, std_time, fps

# ============ 主程序 ============
def main():
    print("=== YOLO-World 模型性能对比测试 ===")

    import os
    if not os.path.exists(MODEL_640):
        print(f"❌ 模型文件 {MODEL_640} 不存在，请检查路径。")
        return
    if not os.path.exists(MODEL_320):
        print(f"❌ 模型文件 {MODEL_320} 不存在，请检查路径。")
        return

    # 初始化摄像头
    print("📷 初始化摄像头...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 640)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    frame = picam2.capture_array()
    picam2.stop()
    picam2.close()
    print("✅ 摄像头就绪")

    # 加载 640 模型
    print(f"📦 加载模型 {MODEL_640} ...")
    sess_640 = ort.InferenceSession(MODEL_640, providers=['CPUExecutionProvider'])
    inputs_640 = sess_640.get_inputs()
    for inp in inputs_640:
        print(f"   输入: {inp.name}, 形状: {inp.shape}")

    # 加载 320 模型
    print(f"📦 加载模型 {MODEL_320} ...")
    sess_320 = ort.InferenceSession(MODEL_320, providers=['CPUExecutionProvider'])
    inputs_320 = sess_320.get_inputs()
    for inp in inputs_320:
        print(f"   输入: {inp.name}, 形状: {inp.shape}")

    # 测试 640 模型
    print(f"\n🖥️ 测试 {MODEL_640} (输入尺寸 640x640):")
    avg640, std640, fps640 = test_model(
        sess_640, inputs_640, frame, 640,
        BENCHMARK_ITERATIONS, WARMUP_ITERATIONS
    )

    # 测试 320 模型
    print(f"\n🖥️ 测试 {MODEL_320} (输入尺寸 320x320):")
    avg320, std320, fps320 = test_model(
        sess_320, inputs_320, frame, 320,
        BENCHMARK_ITERATIONS, WARMUP_ITERATIONS
    )

    # 结果汇总
    print("\n" + "="*50)
    print("📊 性能对比结果")
    print("="*50)
    print(f"🟦 {MODEL_640} (640x640):")
    print(f"   平均耗时: {avg640:.2f} ± {std640:.2f} ms")
    print(f"   帧率: {fps640:.2f} FPS")
    print(f"🟩 {MODEL_320} (320x320):")
    print(f"   平均耗时: {avg320:.2f} ± {std320:.2f} ms")
    print(f"   帧率: {fps320:.2f} FPS")
    speedup = avg640 / avg320
    print(f"🚀 加速比: {speedup:.2f}x")
    print("="*50)

    if speedup > 1.5:
        print("✅ 新模型提速明显，建议替换为 320 模型！")
    elif speedup > 1.2:
        print("✅ 新模型有一定提速，可以考虑替换。")
    else:
        print("⚠️ 提速不明显，可能需要评估精度损失。")

if __name__ == "__main__":
    main()
