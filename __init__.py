"""
YOLOv8 交通检测系统工具模块
提供检测、追踪、越线计数等核心功能
"""

from .detection import YOLODetector
from .tracking import MultiObjectTracker
from .line_crossing import LineCrossingCounter
from .area_counting import AreaCounter
from .analysis import TrafficAnalyzer
from .export import DataExporter
from .data_saver import AutoDataSaver

__all__ = [
    'YOLODetector',
    'MultiObjectTracker', 
    'LineCrossingCounter',
    'AreaCounter',
    'TrafficAnalyzer',
    'DataExporter'
]

__version__ = "1.0.0"
__author__ = "YOLOv8交通检测系统"