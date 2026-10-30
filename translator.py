#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argostranslate.package
import argostranslate.translate
import threading
import time
import os

class ArgosTranslator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ArgosTranslator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = False
        self.cache = {}
        self._load_existing_packages()  # 只加载已有包，不联网
        # 如果没找到包，则在后台尝试下载（不阻塞）
        if not self._initialized:
            threading.Thread(target=self._init_translator_background, daemon=True).start()

    def _load_existing_packages(self):
        """尝试加载已安装的翻译包，不联网"""
        try:
            installed = argostranslate.package.get_installed_packages()
            for pkg in installed:
                if pkg.from_code == "zh" and pkg.to_code == "en":
                    self._initialized = True
                    print("✅ Argos Translate 已就绪（使用已安装包）")
                    return
            print("⚠️ 未找到中英翻译包，将尝试后台下载")
        except Exception as e:
            print(f"加载翻译包失败: {e}")

    def _init_translator_background(self):
        """后台线程尝试下载并安装翻译包"""
        try:
            print("⏳ 后台下载 Argos 翻译包...")
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(
                    lambda x: x.from_code == "zh" and x.to_code == "en",
                    available_packages
                )
            )
            argostranslate.package.install_from_path(package_to_install.download())
            self._initialized = True
            print("✅ Argos Translate 后台下载完成")
        except Exception as e:
            print(f"⚠️ Argos 后台下载失败: {e}")

    def translate(self, text, target_lang="en"):
        if not text or not text.strip():
            return text
        if text in self.cache:
            print(f"🔤 缓存: '{text}' -> '{self.cache[text]}'")
            return self.cache[text]
        # 如果尚未初始化，直接返回原文（避免阻塞）
        if not self._initialized:
            print("⚠️ 翻译未就绪，返回原文")
            return text
        try:
            result = argostranslate.translate.translate(text, "zh", "en")
            if result:
                self.cache[text] = result
                print(f"🔤 翻译: '{text}' -> '{result}'")
                return result
            return text
        except Exception as e:
            print(f"⚠️ 翻译失败: {e}")
            return text

    def load(self):
        pass

    def is_loaded(self):
        return self._initialized


translator = ArgosTranslator()
translate_to_english = translator.translate
