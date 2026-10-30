#!/usr/bin/env python3
"""
YOLO-World ONNX 推理封装（支持动态文本输入）
"""

import numpy as np
import cv2
import onnxruntime as ort

class YOLOWORLD:
    def __init__(self, model_path, device="cpu", conf=0.3, imgsz=320):
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # 创建 ONNX Runtime 会话
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # 获取输入输出信息
        self.inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in self.inputs]
        self.output_names = [out.name for out in self.session.get_outputs()]
        # 检查是否有 txt_feats 输入
        self.has_txt_feats = 'txt_feats' in self.input_names
        # 保存当前类别描述（用于生成 txt_feats）
        self.classes = None
        self.txt_feats = None  # 缓存文本特征

    def set_classes(self, classes, txt_feats=None):
        """
        设置当前检测的类别描述。
        classes: list of strings, e.g., ["person", "car", "dog"]
        txt_feats: 可选，如果提供则直接使用，否则需要外部生成（通常由调用者提供）
        """
        self.classes = classes
        if txt_feats is not None:
            self.txt_feats = txt_feats
        else:
            # 如果没有提供，则生成一个占位符（需要外部提供）
            # 实际使用时，应传入 CLIP 编码的文本特征
            self.txt_feats = None

    def __call__(self, frame, conf=None, imgsz=None, iou=0.5):
        """推理函数，返回 boxes, scores, class_ids"""
        if conf is None:
            conf = self.conf
        if imgsz is None:
            imgsz = self.imgsz

        # 预处理图像
        h, w = frame.shape[:2]
        # 保持宽高比缩放并填充（可选，但模型输入固定为正方形，我们直接 resize）
        resized = cv2.resize(frame, (imgsz, imgsz))
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = input_tensor[np.newaxis, ...]          # (1, 3, imgsz, imgsz)

        # 构建输入字典
        feed_dict = {'images': input_tensor}
        if self.has_txt_feats:
            # 如果没有 txt_feats，则使用默认（需要保证存在）
            if self.txt_feats is None:
                # 如果未设置，使用随机特征（仅用于演示，实际应使用 CLIP 编码）
                raise ValueError("txt_feats 未设置，请先调用 set_classes 并传入文本特征！")
            feed_dict['txt_feats'] = self.txt_feats

        # 推理
        outputs = self.session.run(self.output_names, feed_dict)

        # 解析输出（假设 outputs[0] = scores, outputs[1] = boxes）
        # 具体格式取决于模型导出方式，可能需要调整
        scores = outputs[0]   # shape: (1, num_classes, num_anchors)
        boxes = outputs[1]    # shape: (1, 4, num_anchors)

        # 后处理：NMS 等（简化版，只取最高分）
        scores_np = np.squeeze(scores)  # (num_classes, num_anchors)
        boxes_np = np.squeeze(boxes)    # (4, num_anchors)

        # 阈值过滤
        max_score = np.max(scores_np)
        if max_score < conf:
            return [], [], []

        # 获取最高分对应的框和类别
        max_idx = np.argmax(scores_np)
        class_id = max_idx // scores_np.shape[1]   # 类别索引
        anchor_idx = max_idx % scores_np.shape[1]  # 锚点索引
        box = boxes_np[:, anchor_idx]  # (4,)
        # 转换格式：假设输出为 cx, cy, w, h -> x1, y1, x2, y2
        cx, cy, w_box, h_box = box
        x1 = cx - w_box/2
        y1 = cy - h_box/2
        x2 = cx + w_box/2
        y2 = cy + h_box/2

        # 返回列表（兼容原有接口）
        return [[x1, y1, x2-x1, y2-y1]], [float(max_score)], [class_id]

    # 可选：从 CLIP 生成 txt_feats 的方法（需要外部模型）
    # 这里不实现，由调用者提供
