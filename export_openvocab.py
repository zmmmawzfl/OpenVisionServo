from ultralytics import YOLOWorld

# 加载模型
model = YOLOWorld('YOLO-World/yolov8s-worldv2.pt')

# 设置类别占位（启用开放词汇功能，最大类别数设为 10）
model.set_classes([""] * 10)

# 导出 ONNX，输入尺寸 320，opset 12，简化模型
model.export(format='onnx', imgsz=320, opset=12, simplify=True)
