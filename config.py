"""
YOLOv8 交通检测系统配置文件
包含模型参数、类别标签、路径配置等
"""

import os
from pathlib import Path

class Config:
    """系统配置类"""
    
    # 基础路径配置
    BASE_DIR = Path(__file__).parent
    ASSETS_DIR = BASE_DIR / "assets"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    DATA_DIR = BASE_DIR / "data"
    
    # 模型配置
    MODEL_CONFIGS = {
        "yolov8n": {
            "name": "YOLOv8 Nano",
            "description": "最快速度，较低精度",
            "model_size": "6MB"
        },
        "yolov8s": {
            "name": "YOLOv8 Small", 
            "description": "平衡速度和精度",
            "model_size": "22MB"
        },
        "yolov8m": {
            "name": "YOLOv8 Medium",
            "description": "中等速度，较高精度", 
            "model_size": "52MB"
        },
        "yolov8l": {
            "name": "YOLOv8 Large",
            "description": "较慢速度，高精度",
            "model_size": "88MB"  
        },
        "yolov8x": {
            "name": "YOLOv8 Extra Large",
            "description": "最高精度，最慢速度",
            "model_size": "136MB"
        }
    }
    
    # 默认模型设置
    DEFAULT_MODEL = "yolov8s.pt"
    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_IOU_THRESHOLD = 0.45
    
    # COCO数据集类别 (完整80类，带中文翻译)
    COCO_CLASSES = {
        0: {'en': 'person', 'zh': '行人', 'color': (255, 0, 0)},
        1: {'en': 'bicycle', 'zh': '自行车', 'color': (0, 255, 0)},
        2: {'en': 'car', 'zh': '汽车', 'color': (0, 0, 255)},
        3: {'en': 'motorcycle', 'zh': '摩托车', 'color': (255, 255, 0)},
        4: {'en': 'airplane', 'zh': '飞机', 'color': (255, 0, 255)},
        5: {'en': 'bus', 'zh': '公交车', 'color': (0, 255, 255)},
        6: {'en': 'train', 'zh': '火车', 'color': (128, 0, 0)},
        7: {'en': 'truck', 'zh': '卡车', 'color': (0, 128, 0)},
        8: {'en': 'boat', 'zh': '船', 'color': (0, 0, 128)},
        9: {'en': 'traffic light', 'zh': '交通灯', 'color': (128, 128, 0)},
        10: {'en': 'fire hydrant', 'zh': '消防栓', 'color': (128, 0, 128)},
        11: {'en': 'stop sign', 'zh': '停车标志', 'color': (0, 128, 128)},
        12: {'en': 'parking meter', 'zh': '停车计时器', 'color': (192, 192, 192)},
        13: {'en': 'bench', 'zh': '长椅', 'color': (128, 128, 128)},
        14: {'en': 'bird', 'zh': '鸟', 'color': (64, 64, 64)},
        15: {'en': 'cat', 'zh': '猫', 'color': (255, 165, 0)},
        16: {'en': 'dog', 'zh': '狗', 'color': (255, 192, 203)},
        17: {'en': 'horse', 'zh': '马', 'color': (160, 82, 45)},
        18: {'en': 'sheep', 'zh': '羊', 'color': (255, 255, 255)},
        19: {'en': 'cow', 'zh': '牛', 'color': (0, 0, 0)},
        20: {'en': 'elephant', 'zh': '大象', 'color': (169, 169, 169)},
        21: {'en': 'bear', 'zh': '熊', 'color': (139, 69, 19)},
        22: {'en': 'zebra', 'zh': '斑马', 'color': (255, 248, 220)},
        23: {'en': 'giraffe', 'zh': '长颈鹿', 'color': (255, 215, 0)},
        24: {'en': 'backpack', 'zh': '背包', 'color': (75, 0, 130)},
        25: {'en': 'umbrella', 'zh': '雨伞', 'color': (148, 0, 211)},
        26: {'en': 'handbag', 'zh': '手提包', 'color': (255, 20, 147)},
        27: {'en': 'tie', 'zh': '领带', 'color': (30, 144, 255)},
        28: {'en': 'suitcase', 'zh': '行李箱', 'color': (50, 205, 50)},
        29: {'en': 'frisbee', 'zh': '飞盘', 'color': (255, 69, 0)},
        30: {'en': 'skis', 'zh': '滑雪板', 'color': (0, 191, 255)},
        31: {'en': 'snowboard', 'zh': '雪板', 'color': (240, 230, 140)},
        32: {'en': 'sports ball', 'zh': '运动球', 'color': (255, 105, 180)},
        33: {'en': 'kite', 'zh': '风筝', 'color': (34, 139, 34)},
        34: {'en': 'baseball bat', 'zh': '棒球棒', 'color': (210, 180, 140)},
        35: {'en': 'baseball glove', 'zh': '棒球手套', 'color': (165, 42, 42)},
        36: {'en': 'skateboard', 'zh': '滑板', 'color': (244, 164, 96)},
        37: {'en': 'surfboard', 'zh': '冲浪板', 'color': (0, 206, 209)},
        38: {'en': 'tennis racket', 'zh': '网球拍', 'color': (255, 228, 181)},
        39: {'en': 'bottle', 'zh': '瓶子', 'color': (173, 216, 230)},
        40: {'en': 'wine glass', 'zh': '酒杯', 'color': (221, 160, 221)},
        41: {'en': 'cup', 'zh': '杯子', 'color': (238, 130, 238)},
        42: {'en': 'fork', 'zh': '叉子', 'color': (218, 165, 32)},
        43: {'en': 'knife', 'zh': '刀', 'color': (188, 143, 143)},
        44: {'en': 'spoon', 'zh': '勺子', 'color': (176, 196, 222)},
        45: {'en': 'bowl', 'zh': '碗', 'color': (205, 92, 92)},
        46: {'en': 'banana', 'zh': '香蕉', 'color': (255, 255, 0)},
        47: {'en': 'apple', 'zh': '苹果', 'color': (255, 0, 0)},
        48: {'en': 'sandwich', 'zh': '三明治', 'color': (210, 180, 140)},
        49: {'en': 'orange', 'zh': '橙子', 'color': (255, 165, 0)},
        50: {'en': 'broccoli', 'zh': '西兰花', 'color': (0, 128, 0)},
        51: {'en': 'carrot', 'zh': '胡萝卜', 'color': (255, 140, 0)},
        52: {'en': 'hot dog', 'zh': '热狗', 'color': (205, 92, 92)},
        53: {'en': 'pizza', 'zh': '披萨', 'color': (255, 228, 196)},
        54: {'en': 'donut', 'zh': '甜甜圈', 'color': (255, 192, 203)},
        55: {'en': 'cake', 'zh': '蛋糕', 'color': (255, 228, 225)},
        56: {'en': 'chair', 'zh': '椅子', 'color': (139, 69, 19)},
        57: {'en': 'couch', 'zh': '沙发', 'color': (160, 82, 45)},
        58: {'en': 'potted plant', 'zh': '盆栽', 'color': (0, 128, 0)},
        59: {'en': 'bed', 'zh': '床', 'color': (255, 255, 240)},
        60: {'en': 'dining table', 'zh': '餐桌', 'color': (210, 180, 140)},
        61: {'en': 'toilet', 'zh': '马桶', 'color': (255, 255, 255)},
        62: {'en': 'tv', 'zh': '电视', 'color': (0, 0, 0)},
        63: {'en': 'laptop', 'zh': '笔记本电脑', 'color': (169, 169, 169)},
        64: {'en': 'mouse', 'zh': '鼠标', 'color': (128, 128, 128)},
        65: {'en': 'remote', 'zh': '遥控器', 'color': (105, 105, 105)},
        66: {'en': 'keyboard', 'zh': '键盘', 'color': (0, 0, 0)},
        67: {'en': 'cell phone', 'zh': '手机', 'color': (255, 255, 255)},
        68: {'en': 'microwave', 'zh': '微波炉', 'color': (192, 192, 192)},
        69: {'en': 'oven', 'zh': '烤箱', 'color': (128, 128, 128)},
        70: {'en': 'toaster', 'zh': '烤面包机', 'color': (211, 211, 211)},
        71: {'en': 'sink', 'zh': '水槽', 'color': (220, 220, 220)},
        72: {'en': 'refrigerator', 'zh': '冰箱', 'color': (255, 255, 255)},
        73: {'en': 'book', 'zh': '书', 'color': (255, 228, 196)},
        74: {'en': 'clock', 'zh': '时钟', 'color': (245, 245, 220)},
        75: {'en': 'vase', 'zh': '花瓶', 'color': (176, 224, 230)},
        76: {'en': 'scissors', 'zh': '剪刀', 'color': (192, 192, 192)},
        77: {'en': 'teddy bear', 'zh': '泰迪熊', 'color': (139, 69, 19)},
        78: {'en': 'hair drier', 'zh': '吹风机', 'color': (255, 255, 255)},
        79: {'en': 'toothbrush', 'zh': '牙刷', 'color': (0, 255, 255)}
    }
    
    # 交通相关类别 (重点关注)
    TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]  # 行人、自行车、汽车、摩托车、公交车、卡车、交通灯、停车标志
    
    # 视频设置
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # UI设置
    WINDOW_SIZE = (1400, 900)
    VIDEO_DISPLAY_SIZE = (800, 600)
    
    # 追踪设置
    MAX_DISAPPEARED = 30
    MAX_DISTANCE = 100
    
    # 越线检测设置
    LINE_THICKNESS = 3
    COUNTING_LINE_COLOR = (0, 255, 0)  # 绿色
    
    # 数据导出设置
    EXPORT_FORMATS = ['CSV', 'Excel', 'JSON']
    
    @classmethod
    def get_class_name_zh(cls, class_id):
        """获取中文类别名称"""
        if class_id in cls.COCO_CLASSES:
            return cls.COCO_CLASSES[class_id]['zh']
        return f"未知类别_{class_id}"
    
    @classmethod
    def get_class_name_en(cls, class_id):
        """获取英文类别名称"""
        if class_id in cls.COCO_CLASSES:
            return cls.COCO_CLASSES[class_id]['en']
        return f"unknown_class_{class_id}"
    
    @classmethod
    def get_class_color(cls, class_id):
        """获取类别对应颜色"""
        if class_id in cls.COCO_CLASSES:
            return cls.COCO_CLASSES[class_id]['color']
        return (128, 128, 128)  # 默认灰色
    
    @classmethod
    def is_traffic_class(cls, class_id):
        """判断是否为交通相关类别"""
        return class_id in cls.TRAFFIC_CLASSES
    
    @classmethod
    def get_model_info(cls, model_name):
        """获取模型信息"""
        model_key = model_name.replace('.pt', '').lower()
        return cls.MODEL_CONFIGS.get(model_key, {
            "name": model_name,
            "description": "自定义模型",
            "model_size": "未知"
        })