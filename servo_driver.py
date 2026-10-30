#!/usr/bin/env python3
# servo_driver.py
# 树莓派5 硬件PWM驱动 (sysfs) - 带缓存与线程安全

import os
import time
import threading

class MG995Servo:
    # 引脚到PWM通道的映射（树莓派5 实测）
    PIN_TO_CHANNEL = {12: 0, 13: 1}

    def __init__(self, pin, angle_min=0, angle_max=180):
        self.pin = pin
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.channel = self.PIN_TO_CHANNEL.get(pin)
        if self.channel is None:
            raise ValueError(f"引脚 {pin} 未配置PWM通道，请修改 PIN_TO_CHANNEL 映射")
        self.chip = "/sys/class/pwm/pwmchip0"
        self.pwm_path = None
        # 线程锁必须在任何使用它的方法之前初始化
        self._lock = threading.Lock()
        self._export()
        self._set_period(20000000)  # 20ms
        self.current_angle = None
        self._last_write_angle = None
        self.set_angle(90, smooth=False)

    def _export(self):
        try:
            with open(f"{self.chip}/unexport", "w") as f:
                f.write(str(self.channel))
            time.sleep(0.05)
        except:
            pass
        try:
            with open(f"{self.chip}/export", "w") as f:
                f.write(str(self.channel))
        except FileExistsError:
            pass
        self.pwm_path = f"{self.chip}/pwm{self.channel}"

    def _unexport(self):
        try:
            with open(f"{self.chip}/unexport", "w") as f:
                f.write(str(self.channel))
        except:
            pass

    def _set_period(self, period_ns):
        with self._lock:
            with open(f"{self.pwm_path}/period", "w") as f:
                f.write(str(period_ns))

    def _set_duty(self, duty_ns):
        with self._lock:
            with open(f"{self.pwm_path}/duty_cycle", "w") as f:
                f.write(str(duty_ns))

    def _enable(self):
        with self._lock:
            with open(f"{self.pwm_path}/enable", "w") as f:
                f.write("1")

    def _disable(self):
        with self._lock:
            with open(f"{self.pwm_path}/enable", "w") as f:
                f.write("0")

    def _angle_to_duty_ns(self, angle):
        duty_us = 500 + (angle / 180.0) * 2000
        return int(duty_us * 1000)

    def set_angle(self, angle, smooth=False, duration=0.5):
        """
        设置舵机角度。
        :param angle: 目标角度（度）
        :param smooth: 是否平滑过渡（本驱动已去除循环睡眠，由上层控制步进）
        :param duration: 平滑时长（保留兼容，实际未使用）
        """
        angle = max(self.angle_min, min(self.angle_max, angle))
        target_duty = self._angle_to_duty_ns(angle)

        # 缓存检查：如果角度与上次写入的差值小于0.5度，跳过写入
        if self._last_write_angle is not None and abs(angle - self._last_write_angle) < 0.5:
            self.current_angle = angle
            return True

        # 直接写入目标占空比
        self._set_duty(target_duty)
        self._enable()
        self.current_angle = angle
        self._last_write_angle = angle
        # 若需调试可取消下面注释
        # print(f"GPIO{self.pin} -> {angle:.1f} deg")
        return True

    def center(self):
        return self.set_angle(90)

    def sweep(self, delay=0.5):
        print("Sweep test...")
        for a in [0, 90, 180, 90, 0]:
            self.set_angle(a, smooth=False)
            time.sleep(delay)
        print("Sweep done")

    def cleanup(self):
        self._disable()
        self._unexport()


class DualServoPanTilt:
    def __init__(self, pan_pin=12, tilt_pin=13,
                 pan_min=0, pan_max=180, tilt_min=0, tilt_max=180):
        self.pan = MG995Servo(pan_pin, pan_min, pan_max)
        self.tilt = MG995Servo(tilt_pin, tilt_min, tilt_max)

    def set_position(self, pan_angle, tilt_angle, smooth=False, duration=0.2):
        self.pan.set_angle(pan_angle, smooth, duration)
        self.tilt.set_angle(tilt_angle, smooth, duration)

    def center(self):
        self.pan.center()
        self.tilt.center()

    def cleanup(self):
        self.pan.cleanup()
        self.tilt.cleanup()
