# OpenVisionServo

OpenVisionServo 是一个基于开放词汇目标检测、视觉跟踪与双轴伺服控制的智能视觉系统，面向机器人巡检、目标锁定、智能监控和视觉伺服应用场景。

项目结合了 YOLO-World（开放词汇检测）、OpenCV 跟踪器、语音识别、文本翻译与舵机控制，能够在摄像头视野中持续锁定指定目标，并通过机械结构进行自动对焦和跟踪。

## 项目亮点

- 开放词汇目标检测：支持动态文本类别输入，适用于未知目标与场景扩展
- 视觉目标跟踪：采用 OpenCV CSRT/KCF 跟踪器，增强目标连续性
- 双舵机视觉伺服：实现云台式自动转向控制，保持目标处于画面中心
- 语音交互：接入 Vosk 中文语音识别，支持口头指定目标类别
- 多语言支持：结合翻译模块，帮助实现中英文目标描述处理
- ONNX 推理：支持轻量化部署，适合边缘设备或低功耗平台

## 技术栈

- Python 3.10+
- OpenCV
- PyTorch
- ONNX Runtime
- YOLO-World / CLIP
- Vosk
- Raspberry Pi / Linux + GPIO / 舵机驱动

## 系统架构

- `visual_servo.py`：主控制程序，负责目标检测、跟踪、状态管理和伺服控制
- `yolo_world_onnx.py`：YOLO-World ONNX 推理封装
- `audio_manager.py`：语音识别与音频输入管理
- `translator.py`：中文/英文翻译接口
- `servo_driver.py`：双舵机驱动控制
- `export_openvocab.py`：开放词汇模型导出与适配工具
- `test_models_speed.py` / `test_models_performance.py`：模型速度与性能测试脚本

## 适用场景

- 智能监控与目标锁定
- 机器人视觉跟踪
- 自动对焦云台系统
- 智能交互端视觉控制
- 低成本开源视觉伺服平台

## 硬件要求

- 摄像头（例如 CSI 摄像头、USB 摄像头）
- 双轴舵机云台
- Raspberry Pi / Ubuntu/Linux 主机
- USB 麦克风或音频输入设备
- 适配的 GPIO 或 PWM 驱动板

## 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动主程序

```bash
python visual_servo.py
```

3. 在程序中指定目标关键词，例如：

- 人
- 手机
- 书
- 杯子
- 纸箱

系统会通过视觉检测 + 目标跟踪 + 云台控制自动跟随目标。

## GitHub 发布建议

该项目适合发布到 GitHub，并建议使用以下方式进行仓库管理：

- 使用清晰的仓库名：`OpenVisionServo`
- 建议保留 `README.md`、`LICENSE`、`requirements.txt` 等基础文件
- 把大模型权重（`.pt`、`.onnx`）放入 Git LFS 或单独发布为 Release 包
- 若用于实验室或竞赛项目，可配套增加示例视频、硬件连接图与使用说明

## 版权说明

本项目采用 MIT License 开源协议，适合二次开发、教学研究和工程原型验证。

## 未来规划

- 增加多目标跟踪支持
- 提高低功耗部署性能
- 支持更稳定的视觉伺服闭环控制
- 扩展增强实时语音命令交互
- 提供更完整的硬件接线文档和演示视频

---

如果你希望，我还可以继续为你补充：

- 一个更正式的英文版 README
- 一个适合 GitHub 首页展示的项目介绍页
- 一个 `requirements.txt` 依赖文件
- 一套 GitHub 发布命令（创建仓库、添加远程、提交与推送）

