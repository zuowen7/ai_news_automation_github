#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI新闻自动化系统 - 快速启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    main()
