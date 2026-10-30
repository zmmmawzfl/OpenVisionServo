#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import json
import threading
import queue
import gc
import random
import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import Transform

# 新增：CLIP 支持
import torch
from transformers import CLIPProcessor, CLIPModel

from servo_driver import DualServoPanTilt

try:
    from translator import translator
except ImportError:
    print("⚠️ translator 模块未找到，将使用默认翻译（直接返回原文）")
    class DummyTranslator:
        def translate(self, text, target_lang="en"):
            return text
    translator = DummyTranslator()

try:
    from audio_manager import audio_manager
except ImportError:
    print("⚠️ audio_manager 模块未找到，语音功能禁用")
    audio_manager = None

# 导入新的 ONNX 推理类（支持开放词汇）
from yolo_world_onnx import YOLOWORLD

# ========================= User Config =========================
PAN_MIN, PAN_MAX = 15, 165
TILT_MIN, TILT_MAX = 30, 150
TILT_SOFT_MIN = 30
TILT_SOFT_MAX = 150

KP_PAN = 0.10
KP_TILT = 0.12
DEAD_ZONE = 3
MAX_DELTA = 15

TRACKER_TYPE = 'CSRT'
MAX_LOST_FRAMES = 5
CONTROL_INTERVAL = 2
TRACKER_UPDATE_INTERVAL = 2

STABLE_FRAMES = 12

DETECT_TIMEOUT = 3.0
SMART_SEARCH_TIMEOUT = 20.0
RANDOM_SEARCH_INTERVAL = 1.0
SMART_DITHER_AMPLITUDE = 20.0
SMART_TILT_SCALE = 1.0
YOLO_DETECT_INTERVAL = 30

SHOW_WINDOW = True
WINDOW_NAME = "Visual Servo"

CALIB_FILE = "motion_compensation_config.json"
# ===============================================================

STATE_WAIT_TARGET = 0
STATE_TRACK = 1
STATE_LOST_DETECT = 2
STATE_LOST_SMART = 3
STATE_LOST_RANDOM = 4

def is_chinese(text):
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

# ---------- 文本特征生成（使用 CLIP） ----------
_clip_model = None
_clip_processor = None

def get_text_features(texts):
    """
    将文本列表编码为 YOLO-World 所需的 txt_feats。
    返回: numpy array, shape (1, num_classes, 512)
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        # 首次调用加载 CLIP 模型（约 600MB 内存）
        model_id = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_id)
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model.eval()
        print("✅ CLIP 模型加载完成")
    if isinstance(texts, str):
        texts = [texts]
    inputs = _clip_processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_feats = _clip_model.get_text_features(**inputs)  # (num_classes, 512)
    # L2 归一化（YOLO-World 要求）
    txt_feats = txt_feats / torch.norm(txt_feats, dim=1, keepdim=True)
    # 增加 batch 维度并转为 numpy
    txt_feats = txt_feats.unsqueeze(0).cpu().numpy()  # (1, num_classes, 512)
    return txt_feats

# ---------- 状态管理 ----------
class AppState:
    def __init__(self):
        self.tracker = None
        self.tracker_initialized = False
        self.current_bbox = None
        self.track_lost_counter = 0
        self.lost_start_time = None
        self.detect_start_time = None
        self.last_random_move_time = 0
        self.random_search_start_time = 0
        self.frame_count = 0
        self.state = STATE_WAIT_TARGET
        self.target_desc_cn = None
        self.target_desc_en = None

        # 固定参数卡尔曼
        self.kf_pan = SimpleKalman(90, q=0.003, r=0.4)
        self.kf_tilt = SimpleKalman(90, q=0.003, r=0.4)

        self.stable_counter = 0
        self.drift_counter = 0
        self.last_relocate_time = 0

        self.last_track_bbox = None
        self.last_track_success = False

        self.detect_pending = False
        self.detect_submit_time = 0.0
        self.need_relocate = False
        self.timeout_printed = False

        self.last_known_position = (90, 90)

class SimpleKalman:
    __slots__ = ('x', 'p', 'q', 'r')
    def __init__(self, init_val=90, q=0.02, r=0.2):
        self.x = init_val
        self.p = 1.0
        self.q = q
        self.r = r

    def update(self, measurement):
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

def compute_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = w1 * h1
    box2Area = w2 * h2
    union = box1Area + box2Area - interArea
    if union == 0:
        return 0
    return interArea / union

def create_tracker(tracker_type='KCF'):
    try:
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError("Unsupported tracker")
    except Exception as e:
        print("Tracker creation error:", e)
        return None

def release_tracker(tracker_obj):
    if tracker_obj is not None:
        try:
            tracker_obj.clear()
        except Exception:
            pass

def is_x11_available():
    display = os.environ.get('DISPLAY')
    if not display:
        return False
    try:
        if display.startswith(':'):
            display_num = display.split(':')[1].split('.')[0]
            socket_path = f'/tmp/.X11-unix/X{display_num}'
            return os.path.exists(socket_path)
        elif ':' in display:
            _, disp = display.split(':')
            display_num = disp.split('.')[0]
            socket_path = f'/tmp/.X11-unix/X{display_num}'
            return os.path.exists(socket_path)
        return False
    except:
        return False

def set_process_priority(nice_value=-10):
    try:
        os.nice(nice_value)
        print(f"✅ 主进程优先级已提升 (nice={nice_value})")
    except PermissionError:
        print("⚠️ 无法提升优先级，请使用 sudo 运行")
    except Exception as e:
        print(f"⚠️ 设置优先级失败: {e}")

# ---------- 舵机控制线程 ----------
class ServoController(threading.Thread):
    def __init__(self, pan_tilt, cmd_queue):
        super().__init__()
        self.pt = pan_tilt
        self.cmd_queue = cmd_queue
        self.running = True
        self.daemon = True
        self.max_retries = 3

    def run(self):
        while self.running:
            try:
                pan, tilt = self.cmd_queue.get(timeout=0.1)
                for _ in range(self.max_retries):
                    try:
                        self.pt.set_position(pan, tilt, smooth=False)
                        break
                    except Exception as e:
                        print(f"Servo set_position error, retry: {e}")
                        time.sleep(0.05)
            except queue.Empty:
                continue
            except Exception as e:
                print("Servo thread error:", e)

    def stop(self):
        self.running = False

# ---------- 键盘输入线程 ----------
class KeyboardInputThread(threading.Thread):
    def __init__(self, target_queue):
        super().__init__()
        self.target_queue = target_queue
        self.daemon = True

    def run(self):
        print("\n💡 请在终端输入要追踪的目标描述（例如：'人' 或 '穿黑色衣服的人'），然后按回车")
        print("💡 也支持语音指令（如果已配置）")
        while True:
            try:
                line = sys.stdin.readline()
                if line:
                    line = line.strip()
                    if line:
                        print(f"⌨️ 键盘输入: {line}")
                        self.target_queue.put(line)
            except:
                break

# ---------- 多进程检测器（使用 YOLOWORLD 类，支持 txt_feats） ----------
class AsyncDetector:
    def __init__(self, model_path='yolov8s-worldv2_320_full.onnx', conf=0.30, imgsz=320):
        import multiprocessing as mp
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.input_queue = mp.Queue(maxsize=1)
        self.output_queue = mp.Queue(maxsize=1)
        self.ready_event = mp.Event()
        self.process = mp.Process(target=self._worker, daemon=True)
        self.process.start()
        if not self.ready_event.wait(timeout=15.0):
            print("⚠️ 检测子进程加载超时，继续运行")

    def _worker(self):
        import multiprocessing as mp
        import os
        import numpy as np
        from yolo_world_onnx import YOLOWORLD

        try:
            os.nice(10)
        except:
            pass

        model = YOLOWORLD(self.model_path, device="cpu", conf=self.conf, imgsz=self.imgsz)
        # 预设置默认类别
        model.set_classes(["person"], np.random.randn(1, 1, 512).astype(np.float32))
        self.ready_event.set()
        print("✅ 检测子进程就绪 (ONNX, 开放词汇)")

        while True:
            try:
                frame, w, h, desc, txt_feats = self.input_queue.get(timeout=0.1)
                if frame is None:
                    continue
                model.set_classes([desc], txt_feats)
                boxes, scores, class_ids = model(frame, conf=self.conf, imgsz=self.imgsz, iou=0.7)

                if len(boxes) > 0:
                    scores_np = np.array(scores)
                    idx = scores_np.argmax()
                    best_box = boxes[idx]
                    x, y, bw, bh = best_box
                    x1 = max(0, min(x, w))
                    y1 = max(0, min(y, h))
                    x2 = max(0, min(x + bw, w))
                    y2 = max(0, min(y + bh, h))
                    if x2 > x1 and y2 > y1:
                        detected_bbox = (x1, y1, x2 - x1, y2 - y1)
                        confidence = scores_np[idx].item()
                        try:
                            self.output_queue.get_nowait()
                        except:
                            pass
                        self.output_queue.put((detected_bbox, confidence))
                    else:
                        self._put_none()
                else:
                    self._put_none()
            except mp.queues.Empty:
                continue
            except Exception as e:
                print("检测子进程错误:", e)
                self._put_none()

    def _put_none(self):
        try:
            self.output_queue.get_nowait()
        except:
            pass
        self.output_queue.put((None, 0))

    def submit(self, frame, w, h, desc, txt_feats):
        try:
            self.input_queue.get_nowait()
        except:
            pass
        try:
            self.input_queue.put_nowait((frame.copy(), w, h, desc, txt_feats))
        except:
            pass

    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except:
            return None

    def shutdown(self):
        self.process.terminate()
        self.process.join(timeout=1.0)

def handle_detection_result(state, detected_bbox, confidence, frame, need_relocate=False):
    if frame is None:
        print("⚠️ handle_detection_result: frame 为空，跳过")
        return

    if detected_bbox is None:
        print(f"⚠️ 未检测到目标: {state.target_desc_en}")
        if state.state == STATE_WAIT_TARGET or state.state == STATE_LOST_DETECT:
            state.state = STATE_LOST_DETECT
            state.detect_start_time = time.time()
        return

    x1, y1, bw, bh = detected_bbox
    if bw <= 0 or bh <= 0:
        print("⚠️ 目标框无效")
        if state.state == STATE_WAIT_TARGET or state.state == STATE_LOST_DETECT:
            state.state = STATE_LOST_DETECT
            state.detect_start_time = time.time()
        return

    new_bbox = (x1, y1, bw, bh)

    if need_relocate or state.state == STATE_TRACK:
        if state.tracker_initialized and state.current_bbox is not None:
            iou = compute_iou(state.current_bbox, new_bbox)
            if iou >= 0.5:
                return
            print(f"🔄 重新定位目标 (iou={iou:.2f})")
        new_tracker = create_tracker(TRACKER_TYPE)
        if new_tracker is None:
            print("无法创建新跟踪器")
            return
        new_tracker.init(frame, new_bbox)
        ok, bbox = new_tracker.update(frame)
        if ok:
            if state.tracker is not None:
                release_tracker(state.tracker)
            state.tracker = new_tracker
            state.tracker_initialized = True
            state.current_bbox = bbox
            state.last_track_bbox = bbox
            state.last_track_success = True
            state.state = STATE_TRACK
            state.lost_start_time = None
            state.detect_start_time = None
            state.drift_counter = 0
            state.last_relocate_time = time.time()
            print(f"🎯 目标已锁定: {state.target_desc_en} (置信度: {confidence:.2f})")
        else:
            release_tracker(new_tracker)
            print("跟踪器初始化不稳定，保持原状态")
    else:
        new_tracker = create_tracker(TRACKER_TYPE)
        if new_tracker is None:
            print("无法创建跟踪器")
            if state.state == STATE_WAIT_TARGET or state.state == STATE_LOST_DETECT:
                state.state = STATE_LOST_DETECT
                state.detect_start_time = time.time()
            return
        new_tracker.init(frame, new_bbox)
        ok, bbox = new_tracker.update(frame)
        if ok:
            if state.tracker is not None:
                release_tracker(state.tracker)
            state.tracker = new_tracker
            state.tracker_initialized = True
            state.current_bbox = bbox
            state.last_track_bbox = bbox
            state.last_track_success = True
            state.state = STATE_TRACK
            state.lost_start_time = None
            state.detect_start_time = None
            state.drift_counter = 0
            print(f"🎯 目标已锁定: {state.target_desc_en} (置信度: {confidence:.2f})")
        else:
            release_tracker(new_tracker)
            print("跟踪器初始化不稳定")
            if state.state == STATE_WAIT_TARGET or state.state == STATE_LOST_DETECT:
                state.state = STATE_LOST_DETECT
                state.detect_start_time = time.time()

def main():
    set_process_priority(-10)
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    headless = os.getenv('HEADLESS', '0') == '1'
    show_window_effective = SHOW_WINDOW and not headless
    if show_window_effective:
        if not is_x11_available():
            print("⚠️ X11 不可用，将禁用图形窗口")
            show_window_effective = False
    if not show_window_effective:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print("OpenCL acceleration enabled.")
    else:
        print("OpenCL not available, using CPU only.")

    start_time = time.time()
    _time = time.time
    _hypot = math.hypot
    _abs = abs

    # 舵机
    try:
        pt = DualServoPanTilt(pan_pin=12, tilt_pin=13,
                              pan_min=PAN_MIN, pan_max=PAN_MAX,
                              tilt_min=TILT_MIN, tilt_max=TILT_MAX)
        pt.center()
        time.sleep(0.5)
    except Exception as e:
        print("Servo init failed:", e)
        sys.exit(1)

    print("Calibrating servos...")
    for angle in [PAN_MIN, PAN_MAX, 90]:
        pt.pan.set_angle(angle, smooth=False)
        time.sleep(0.1)
    for angle in [TILT_MIN, TILT_MAX, 90]:
        pt.tilt.set_angle(angle, smooth=False)
        time.sleep(0.1)
    pt.center()
    time.sleep(0.3)
    print("Calibration done")

    # ---------- 摄像头初始化 ----------
    picam2 = None
    for attempt in range(3):
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (320, 240)},
                transform=Transform(hflip=1),
                controls={
                    "FrameRate": 30,
                    "AeEnable": True,
                    "AwbEnable": True,
                    "AeMeteringMode": 0,
                    "AnalogueGain": 1.5,
                    "Brightness": 0.2,
                    "Contrast": 1.0,
                    "Saturation": 1.0
                }
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)
            print("Camera ready (320x240, 30fps, hardware flip, AWB enabled)")
            break
        except Exception as e:
            print("Camera init attempt", attempt+1, "failed:", e)
            if picam2 is not None:
                picam2.close()
                picam2 = None
            time.sleep(0.5)
    if picam2 is None:
        print("Camera init failed")
        pt.cleanup()
        sys.exit(1)

    # ---------- 多进程检测器（使用新模型，开放词汇） ----------
    print("⏳ 启动多进程检测器 (ONNX, 320x320, 开放词汇)...")
    async_detector = AsyncDetector('yolov8s-worldv2_320_full.onnx', conf=0.30, imgsz=320)
    print("✅ 检测器已启动")

    # 预热（使用一张假图）
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    dummy_desc = "person"
    dummy_txt_feats = get_text_features([dummy_desc])
    async_detector.submit(dummy_frame, 320, 320, dummy_desc, dummy_txt_feats)
    warmup_start = _time()
    while _time() - warmup_start < 10.0:
        res = async_detector.get_result()
        if res is not None:
            print("✅ 预热完成")
            break
        time.sleep(0.05)
    else:
        print("⚠️ 预热超时，继续运行")

    # 音频
    if audio_manager is not None:
        for _ in range(20):
            if audio_manager.is_ready():
                print("🎤 语音模块已就绪")
                break
            time.sleep(0.1)
    else:
        print("⚠️ 语音模块不可用")

    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, "r") as f:
            cfg = json.load(f)
        print("Loaded compensation: pan", cfg.get('MOTION_COMPENSATION_SCALE_PAN',0), "tilt", cfg.get('MOTION_COMPENSATION_SCALE_TILT',0))

    # 跟踪器
    tracker = create_tracker(TRACKER_TYPE)
    if tracker is None:
        tracker = create_tracker('KCF')
        if tracker is None:
            print("No tracker available! Exiting.")
            picam2.close()
            pt.cleanup()
            sys.exit(1)

    # 键盘输入
    target_queue = queue.Queue()
    keyboard_thread = KeyboardInputThread(target_queue)
    keyboard_thread.start()

    state = AppState()
    state.tracker = tracker

    servo_queue = queue.Queue(maxsize=5)
    servo_thread = ServoController(pt, servo_queue)
    servo_thread.start()

    last_log_angles = (None, None)
    log_angle_threshold = 1.0
    prev_gray = None
    NOISE_UPDATE_INTERVAL = 20

    print("Start tracking with YOLO-World ONNX (320x320) + CSRT")
    print("Tracker:", TRACKER_TYPE, "control interval:", CONTROL_INTERVAL)
    print("Tracker update interval:", TRACKER_UPDATE_INTERVAL, "frames")
    print("Search limits: Pan [%d, %d], Tilt [%d, %d]" % (PAN_MIN, PAN_MAX, TILT_SOFT_MIN, TILT_SOFT_MAX))
    print("\n▶️ 请通过语音或键盘输入目标描述")
    print("   键盘输入方法：在终端输入文字后按回车")

    if show_window_effective:
        print("🖥️ 图形窗口已启用")
    else:
        print("🖥️ 无头模式运行（不显示图像）")

    try:
        while True:
            # ---------- 采集图像 ----------
            raw = picam2.capture_array()
            if raw is None:
                print("⚠️ 摄像头采集失败，尝试重新初始化...")
                try:
                    picam2.stop()
                    picam2.close()
                except:
                    pass
                picam2 = None
                for attempt in range(3):
                    try:
                        picam2 = Picamera2()
                        config = picam2.create_preview_configuration(
                            main={"size": (320, 240)},
                            transform=Transform(hflip=1),
                            controls={
                                "FrameRate": 30,
                                "AeEnable": True,
                                "AwbEnable": True,
                                "AeMeteringMode": 0,
                                "AnalogueGain": 1.5,
                                "Brightness": 0.2,
                                "Contrast": 1.0,
                                "Saturation": 1.0
                            }
                        )
                        picam2.configure(config)
                        picam2.start()
                        time.sleep(0.5)
                        print("✅ 摄像头重新初始化成功")
                        raw = picam2.capture_array()
                        if raw is not None:
                            break
                    except Exception as e:
                        print(f"Re-init failed: {e}")
                        time.sleep(0.5)
                if raw is None:
                    print("⚠️ 摄像头恢复失败，跳过帧")
                    time.sleep(0.02)
                    continue

            # 颜色转换
            frame_umat = cv2.UMat(raw)
            frame_umat = cv2.cvtColor(frame_umat, cv2.COLOR_RGB2BGR)
            frame = frame_umat.get()
            gray = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2GRAY).get() if prev_gray is not None else None
            if prev_gray is not None and state.frame_count % NOISE_UPDATE_INTERVAL == 0:
                diff = cv2.absdiff(gray, prev_gray)
                std = np.std(diff)
                r_new = 0.05 + (std / 50.0) * 0.95
                r_new = max(0.05, min(1.0, r_new))
                state.kf_pan.r = r_new
                state.kf_tilt.r = r_new
            if state.frame_count % NOISE_UPDATE_INTERVAL == 0:
                prev_gray = gray.copy() if gray is not None else None

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            # ---------- 处理检测结果 ----------
            result = async_detector.get_result()
            if result is not None and state.target_desc_en is not None:
                detected_bbox, confidence = result
                state.detect_pending = False
                state.timeout_printed = False

                if confidence >= 0.25:
                    if state.state == STATE_WAIT_TARGET or state.state == STATE_LOST_DETECT:
                        handle_detection_result(state, detected_bbox, confidence, frame, need_relocate=False)
                    elif state.state == STATE_TRACK:
                        if state.need_relocate:
                            handle_detection_result(state, detected_bbox, confidence, frame, need_relocate=True)
                            state.need_relocate = False
                        else:
                            if detected_bbox is not None and state.tracker_initialized and state.current_bbox is not None:
                                iou = compute_iou(state.current_bbox, detected_bbox)
                                tx, ty, tw, th = state.current_bbox
                                dx, dy, dw, dh = detected_bbox
                                cx_cur = tx + tw//2
                                cy_cur = ty + th//2
                                cx_det = dx + dw//2
                                cy_det = dy + dh//2
                                dist = _hypot(cx_cur - cx_det, cy_cur - cy_det)
                                if iou < 0.5 or dist > w * 0.25:
                                    state.drift_counter += 1
                                else:
                                    state.drift_counter = 0
                                if state.drift_counter >= 8 and (_time() - state.last_relocate_time > 3.5):
                                    print(f"⚠️ 漂移严重 (iou={iou:.2f}, dist={dist:.1f})，紧急重定位")
                                    state.need_relocate = True
                                    handle_detection_result(state, detected_bbox, confidence, frame, need_relocate=True)
                                    state.need_relocate = False
                    elif state.state == STATE_LOST_SMART or state.state == STATE_LOST_RANDOM:
                        handle_detection_result(state, detected_bbox, confidence, frame, need_relocate=False)

            # ---------- 超时处理 ----------
            if state.detect_pending:
                elapsed = _time() - state.detect_submit_time
                if elapsed > DETECT_TIMEOUT:
                    if not state.timeout_printed:
                        print("⏰ 检测超时，自动取消等待并进入检测状态")
                        state.timeout_printed = True
                    state.detect_pending = False
                    if state.state == STATE_WAIT_TARGET or state.state == STATE_LOST_DETECT:
                        state.state = STATE_LOST_DETECT
                        state.detect_start_time = _time()

            # ---------- 键盘输入 ----------
            try:
                key_text = target_queue.get_nowait()
                if key_text:
                    state.target_desc_cn = key_text
                    if is_chinese(key_text):
                        state.target_desc_en = translator.translate(key_text)
                    else:
                        state.target_desc_en = key_text
                    state.state = STATE_WAIT_TARGET
                    state.detect_start_time = None
                    state.lost_start_time = None
                    print(f"⏳ 正在检测目标: {key_text} -> {state.target_desc_en} ...")
                    # 生成文本特征
                    txt_feats = get_text_features([state.target_desc_en])
                    while async_detector.get_result() is not None:
                        pass
                    async_detector.submit(frame, w, h, state.target_desc_en, txt_feats)
                    state.detect_pending = True
                    state.detect_submit_time = _time()
                    state.timeout_printed = False
            except queue.Empty:
                pass

            # ---------- 语音输入 ----------
            if audio_manager is not None and audio_manager.is_ready():
                try:
                    text = audio_manager.get_result()
                    if text:
                        print(f"🗣️ 识别: {text}")
                        chinese_desc = text.replace("追踪", "").replace("找", "").replace("看", "").strip()
                        if chinese_desc:
                            if is_chinese(chinese_desc):
                                state.target_desc_en = translator.translate(chinese_desc)
                            else:
                                state.target_desc_en = chinese_desc
                            state.target_desc_cn = chinese_desc
                            state.state = STATE_WAIT_TARGET
                            state.detect_start_time = None
                            state.lost_start_time = None
                            print(f"⏳ 正在检测目标: {chinese_desc} -> {state.target_desc_en} ...")
                            txt_feats = get_text_features([state.target_desc_en])
                            while async_detector.get_result() is not None:
                                pass
                            async_detector.submit(frame, w, h, state.target_desc_en, txt_feats)
                            state.detect_pending = True
                            state.detect_submit_time = _time()
                            state.timeout_printed = False
                except Exception as e:
                    pass

            # ---------- 跟踪器更新 ----------
            track_success = False
            track_bbox = None
            if state.tracker_initialized:
                if state.frame_count % TRACKER_UPDATE_INTERVAL == 0:
                    try:
                        track_success, track_bbox = state.tracker.update(frame)
                    except Exception:
                        track_success = False
                    state.last_track_success = track_success
                    if track_success:
                        state.last_track_bbox = track_bbox
                else:
                    track_success = state.last_track_success
                    track_bbox = state.last_track_bbox

                if not track_success:
                    print("⚠️ CSRT 追踪失败，立即进入检测状态")
                    state.tracker_initialized = False
                    state.state = STATE_LOST_DETECT
                    state.detect_start_time = _time()
                    state.track_lost_counter = 0
                else:
                    state.track_lost_counter = 0

            # ---------- 状态机 ----------
            if state.state == STATE_WAIT_TARGET:
                try:
                    servo_queue.put_nowait((90, 90))
                except queue.Full:
                    pass

            elif state.state == STATE_TRACK and state.tracker_initialized and track_success:
                state.current_bbox = track_bbox
                state.lost_start_time = None
                state.detect_start_time = None

                if state.frame_count % 100 == 0 and state.target_desc_en is not None and not state.detect_pending:
                    # 定期刷新检测
                    txt_feats = get_text_features([state.target_desc_en])
                    while async_detector.get_result() is not None:
                        pass
                    async_detector.submit(frame, w, h, state.target_desc_en, txt_feats)
                    state.detect_pending = True
                    state.detect_submit_time = _time()
                    state.timeout_printed = False

            elif state.state == STATE_LOST_DETECT:
                if state.detect_start_time is None:
                    state.detect_start_time = _time()
                elapsed = _time() - state.detect_start_time
                if elapsed > DETECT_TIMEOUT:
                    print("静默检测超时，进入智能搜索")
                    state.state = STATE_LOST_SMART
                    state.lost_start_time = _time()
                    state.detect_start_time = None
                else:
                    if state.frame_count % YOLO_DETECT_INTERVAL == 0 and not state.detect_pending:
                        if state.target_desc_en is not None:
                            txt_feats = get_text_features([state.target_desc_en])
                            while async_detector.get_result() is not None:
                                pass
                            async_detector.submit(frame, w, h, state.target_desc_en, txt_feats)
                            state.detect_pending = True
                            state.detect_submit_time = _time()
                            state.timeout_printed = False

            elif state.state == STATE_LOST_SMART:
                elapsed = _time() - state.lost_start_time if state.lost_start_time is not None else 0
                if elapsed > SMART_SEARCH_TIMEOUT:
                    state.state = STATE_LOST_RANDOM
                    state.last_random_move_time = _time()
                    state.random_search_start_time = _time()
                    state.last_known_position = (state.kf_pan.x, state.kf_tilt.x)
                    print("智能搜索超时，进入随机搜索")
                else:
                    state.last_known_position = (state.kf_pan.x, state.kf_tilt.x)
                    radius = min(20.0, elapsed * 1.2)
                    angle = elapsed * 0.6
                    dither_pan = radius * math.sin(angle)
                    dither_tilt = radius * math.cos(angle) * 1.0
                    target_pan = max(PAN_MIN, min(PAN_MAX, state.kf_pan.x + dither_pan))
                    target_tilt = max(TILT_SOFT_MIN, min(TILT_SOFT_MAX, state.kf_tilt.x + dither_tilt))
                    try:
                        servo_queue.put_nowait((target_pan, target_tilt))
                    except queue.Full:
                        pass

                    if state.frame_count % YOLO_DETECT_INTERVAL == 0 and not state.detect_pending:
                        if state.target_desc_en is not None:
                            txt_feats = get_text_features([state.target_desc_en])
                            while async_detector.get_result() is not None:
                                pass
                            async_detector.submit(frame, w, h, state.target_desc_en, txt_feats)
                            state.detect_pending = True
                            state.detect_submit_time = _time()
                            state.timeout_printed = False

            elif state.state == STATE_LOST_RANDOM:
                t = _time() - state.random_search_start_time
                radius = min(40.0, t * 2.0)
                angle = t * 0.5
                pan_angle = state.last_known_position[0] + radius * math.sin(angle)
                tilt_angle = state.last_known_position[1] + radius * math.cos(angle)
                pan_angle = max(PAN_MIN, min(PAN_MAX, pan_angle))
                tilt_angle = max(TILT_SOFT_MIN, min(TILT_SOFT_MAX, tilt_angle))
                try:
                    servo_queue.put_nowait((pan_angle, tilt_angle))
                except queue.Full:
                    pass
                state.last_random_move_time = _time()

                if state.frame_count % YOLO_DETECT_INTERVAL == 0 and not state.detect_pending:
                    if state.target_desc_en is not None:
                        txt_feats = get_text_features([state.target_desc_en])
                        while async_detector.get_result() is not None:
                            pass
                        async_detector.submit(frame, w, h, state.target_desc_en, txt_feats)
                        state.detect_pending = True
                        state.detect_submit_time = _time()
                        state.timeout_printed = False

            # ---------- 舵机控制 ----------
            if state.state == STATE_TRACK and state.tracker_initialized and state.current_bbox is not None:
                if state.frame_count % CONTROL_INTERVAL == 0:
                    x, y, bw, bh = [int(v) for v in state.current_bbox]
                    target_cx = x + bw // 2
                    target_cy = y + bh // 2
                    err_x = target_cx - cx
                    err_y = target_cy - cy

                    if _abs(err_x) < DEAD_ZONE and _abs(err_y) < DEAD_ZONE:
                        state.stable_counter += 1
                    else:
                        state.stable_counter = 0

                    if state.stable_counter >= STABLE_FRAMES:
                        continue

                    if _abs(err_x) < DEAD_ZONE:
                        err_x = 0
                    if _abs(err_y) < DEAD_ZONE:
                        err_y = 0

                    delta_pan = err_x * KP_PAN
                    delta_tilt = -err_y * KP_TILT
                    delta_pan = max(-MAX_DELTA, min(MAX_DELTA, delta_pan))
                    delta_tilt = max(-MAX_DELTA, min(MAX_DELTA, delta_tilt))

                    cur_pan = pt.pan.current_angle if pt.pan.current_angle is not None else 90
                    cur_tilt = pt.tilt.current_angle if pt.tilt.current_angle is not None else 90

                    raw_pan = max(PAN_MIN, min(PAN_MAX, cur_pan + delta_pan))
                    raw_tilt = max(TILT_SOFT_MIN, min(TILT_SOFT_MAX, cur_tilt + delta_tilt))

                    smooth_pan = state.kf_pan.update(raw_pan)
                    smooth_tilt = state.kf_tilt.update(raw_tilt)

                    smooth_pan = max(PAN_MIN, min(PAN_MAX, smooth_pan))
                    smooth_tilt = max(TILT_SOFT_MIN, min(TILT_SOFT_MAX, smooth_tilt))

                    try:
                        servo_queue.put_nowait((smooth_pan, smooth_tilt))
                    except queue.Full:
                        pass

                    if (last_log_angles[0] is None or 
                        _abs(smooth_pan - last_log_angles[0]) > log_angle_threshold or 
                        _abs(smooth_tilt - last_log_angles[1]) > log_angle_threshold or
                        _abs(err_x) > 10 or _abs(err_y) > 10):
                        print(f"Track: Pan={smooth_pan:.1f} Tilt={smooth_tilt:.1f} err=({err_x},{err_y})")
                        last_log_angles = (smooth_pan, smooth_tilt)

            # ---------- 显示窗口 ----------
            if show_window_effective:
                if state.current_bbox is not None and state.state == STATE_TRACK:
                    x, y, bw, bh = [int(v) for v in state.current_bbox]
                    cv2.rectangle(frame_umat, (x,y), (x+bw, y+bh), (0,255,0), 2)
                    cv2.circle(frame_umat, (x+bw//2, y+bh//2), 4, (0,255,0), -1)
                if state.state == STATE_WAIT_TARGET:
                    mode_text = "WAIT_TARGET"
                elif state.state == STATE_TRACK:
                    mode_text = "TRACK"
                elif state.state == STATE_LOST_DETECT:
                    mode_text = "DETECTING (no move)"
                elif state.state == STATE_LOST_SMART:
                    mode_text = "SMART SEARCH"
                else:
                    mode_text = "RANDOM SEARCH"
                cv2.putText(frame_umat, "Mode: " + mode_text, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                status = "OK" if state.tracker_initialized and track_success else "LOST"
                cv2.putText(frame_umat, "Tracker: " + status, (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                try:
                    cv2.imshow(WINDOW_NAME, frame_umat)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print("⚠️ 显示窗口错误，已自动关闭显示:", e)
                    show_window_effective = False
                    cv2.destroyAllWindows()

            state.frame_count += 1
            if state.frame_count % 500 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print("Runtime error:", e)
        import traceback
        traceback.print_exc()
    finally:
        async_detector.shutdown()
        servo_thread.stop()
        servo_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        if picam2 is not None:
            picam2.stop()
            picam2.close()
            time.sleep(0.5)
        pt.cleanup()
        if state.tracker is not None:
            release_tracker(state.tracker)
        if audio_manager is not None:
            audio_manager.cleanup()

        elapsed = _time() - start_time
        if state.frame_count > 0:
            print("Avg FPS:", round(state.frame_count / elapsed, 2), "(frames", state.frame_count, "time", round(elapsed,2), "s)")
        else:
            print("No frames processed")
        print("Exit")

if __name__ == "__main__":
    main()
