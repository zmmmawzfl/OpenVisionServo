#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import queue
import json
import pyaudio
from vosk import Model, KaldiRecognizer

# 尝试抑制 ALSA 和 PulseAudio 的错误输出
os.environ['PYAUDIO_DEBUG'] = '0'
os.environ['ALSA_CARD'] = ''
os.environ['PULSE_RUNTIME_PATH'] = '/run/user/1000/pulse'

class AudioManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = False
        self.result_queue = queue.Queue(maxsize=5)
        self.recognizer = None
        self.mic = None
        self.stream = None
        self.device_index = None
        self.model_loaded = False
        self.running = True
        self.thread = None
        self.error = None
        self._start_initialization()

    def _start_initialization(self):
        def init_task():
            try:
                self.device_index = self._detect_device_with_timeout(2.0)
                if self.device_index is None:
                    self.error = "未找到音频输入设备"
                    return
                model_path = "vosk-model-small-cn-0.22"
                if not os.path.exists(model_path):
                    self.error = f"模型目录 {model_path} 不存在"
                    return
                vosk_model = Model(model_path)
                self.recognizer = KaldiRecognizer(vosk_model, 16000)
                self.mic = pyaudio.PyAudio()
                self.stream = self.mic.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4096,
                    input_device_index=self.device_index
                )
                self.model_loaded = True
                print(f"✅ 音频初始化成功 (设备索引 {self.device_index})")
                self.thread = threading.Thread(target=self._reader_thread, daemon=True)
                self.thread.start()
            except Exception as e:
                self.error = str(e)
                print(f"❌ 音频初始化失败: {e}")

        threading.Thread(target=init_task, daemon=True).start()

    def _detect_device_with_timeout(self, timeout=2.0):
        result = None
        done = False

        def detect():
            nonlocal result, done
            try:
                p = pyaudio.PyAudio()
                try:
                    host_info = p.get_host_api_info_by_index(0)
                    device_count = host_info['deviceCount']
                    preferred = None
                    fallback = None
                    for i in range(device_count):
                        dev_info = p.get_device_info_by_host_api_device_index(0, i)
                        if dev_info['maxInputChannels'] > 0:
                            if fallback is None:
                                fallback = i
                            name = dev_info['name'].lower()
                            if 'usb' in name or 'mic' in name:
                                preferred = i
                                break
                    result = preferred if preferred is not None else fallback
                finally:
                    p.terminate()
            except Exception as e:
                print(f"设备检测异常: {e}")
            finally:
                done = True

        thread = threading.Thread(target=detect, daemon=True)
        thread.start()
        start = time.time()
        while not done and (time.time() - start) < timeout:
            time.sleep(0.05)
        if not done:
            print("⚠️ 设备检测超时，禁用语音")
            return None
        return result

    def _reader_thread(self):
        while self.running and self.stream is not None:
            try:
                data = self.stream.read(4096, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        try:
                            self.result_queue.put_nowait(text)
                        except queue.Full:
                            try:
                                self.result_queue.get_nowait()
                            except queue.Empty:
                                pass
                            self.result_queue.put_nowait(text)
            except Exception as e:
                print("音频读取线程错误:", e)
                time.sleep(0.1)

    def is_ready(self):
        return self.model_loaded and self.stream is not None

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def cleanup(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.mic:
            try:
                self.mic.terminate()
            except:
                pass
        self.model_loaded = False

audio_manager = AudioManager()
