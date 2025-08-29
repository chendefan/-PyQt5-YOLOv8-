"""
YOLOv8 检测模块
提供目标检测和实例分割功能
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

from config import Config

class YOLODetector:
    """YOLOv8 检测器类"""
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，默认使用配置中的默认模型
            device: 推理设备 ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path or Config.DEFAULT_MODEL
        self.device = self._get_device(device)
        self.model = None
        self.confidence = Config.DEFAULT_CONFIDENCE
        self.iou_threshold = Config.DEFAULT_IOU_THRESHOLD
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 加载模型
        self.load_model()
    
    def _setup_logging(self):
        """设置日志配置"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _get_device(self, device: str) -> str:
        """获取推理设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        elif device == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                self.logger.warning("CUDA 不可用，回退到 CPU")
                return 'cpu'
        return device
    
    def set_device(self, device: str) -> bool:
        """
        设置推理设备
        
        Args:
            device: 设备类型 ('cpu', 'cuda', 'auto')
            
        Returns:
            bool: 是否设置成功
        """
        old_device = self.device
        new_device = self._get_device(device)
        
        if new_device != old_device:
            self.device = new_device
            if self.model is not None:
                try:
                    # 重新加载模型到新设备
                    if hasattr(self.model, 'to'):
                        self.model.to(self.device)
                    self.logger.info(f"设备已从 {old_device} 切换到 {self.device}")
                    return True
                except Exception as e:
                    self.logger.error(f"设备切换失败: {str(e)}")
                    self.device = old_device  # 恢复原设备
                    return False
            else:
                self.logger.info(f"设备已设置为: {self.device}")
                return True
        return True
    
    def get_device_info(self) -> Dict:
        """
        获取设备信息
        
        Returns:
            Dict: 设备信息
        """
        info = {
            'current_device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': 0,
            'cuda_device_name': None
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        
        return info
    
    def load_model(self) -> bool:
        """
        加载YOLOv8模型
        
        Returns:
            bool: 是否加载成功
        """
        try:
            self.logger.info(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置设备
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            self.logger.info(f"模型加载成功，设备: {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def set_parameters(self, confidence: float = None, iou_threshold: float = None):
        """
        设置检测参数
        
        Args:
            confidence: 置信度阈值 (0-1)
            iou_threshold: IoU阈值 (0-1)
        """
        if confidence is not None:
            self.confidence = max(0.0, min(1.0, confidence))
        if iou_threshold is not None:
            self.iou_threshold = max(0.0, min(1.0, iou_threshold))
        
        self.logger.info(f"检测参数更新 - 置信度: {self.confidence}, IoU: {self.iou_threshold}")
    
    def detect(self, image: np.ndarray, return_annotated: bool = True) -> Dict:
        """
        执行目标检测
        
        Args:
            image: 输入图像 (BGR格式)
            return_annotated: 是否返回标注后的图像
            
        Returns:
            Dict: 检测结果字典
        """
        if self.model is None:
            self.logger.error("模型未加载")
            return self._empty_result()
        
        try:
            # 执行推理
            results = self.model(
                image,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # 解析结果
            detections = self._parse_results(results[0])
            
            # 生成标注图像
            annotated_image = None
            if return_annotated:
                annotated_image = self._annotate_image(image.copy(), detections)
            
            return {
                'detections': detections,
                'annotated_image': annotated_image,
                'total_objects': len(detections),
                'traffic_objects': sum(1 for d in detections if Config.is_traffic_class(d['class_id'])),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"检测过程出错: {str(e)}")
            return self._empty_result()
    
    def _parse_results(self, result) -> List[Dict]:
        """
        解析YOLO检测结果
        
        Args:
            result: YOLO结果对象
            
        Returns:
            List[Dict]: 检测对象列表
        """
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(scores[i]),
                    'class_id': int(classes[i]),
                    'class_name_zh': Config.get_class_name_zh(classes[i]),
                    'class_name_en': Config.get_class_name_en(classes[i]),
                    'color': Config.get_class_color(classes[i]),
                    'is_traffic': Config.is_traffic_class(classes[i])
                }
                detections.append(detection)
        
        # 处理分割掩码（如果存在）
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for i, detection in enumerate(detections):
                if i < len(masks):
                    detection['mask'] = masks[i]
        
        return detections
    
    def _annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            np.ndarray: 标注后的图像
        """
        if not detections:
            return image
        
        # 转换为PIL图像以支持中文文本
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体
        try:
            # Windows系统中文字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
                "C:/Windows/Fonts/arial.ttf"   # Arial备用
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, 20)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except Exception as e:
            font = ImageFont.load_default()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            confidence = detection['confidence']
            class_name = detection['class_name_zh']
            color = detection['color']
            
            # 使用PIL绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 准备标签文本
            label = f"{class_name} {confidence:.2f}"
            
            # 使用PIL绘制中文文本
            try:
                # 获取文本尺寸
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 绘制文本背景
                bg_coords = [
                    x1, y1 - text_height - 10,
                    x1 + text_width + 10, y1
                ]
                draw.rectangle(bg_coords, fill=color)
                
                # 绘制文本
                draw.text((x1 + 5, y1 - text_height - 5), label, fill=(255, 255, 255), font=font)
                
            except Exception as e:
                # 如果PIL文本绘制失败，使用简单的英文标签
                label_en = f"{detection['class_name_en']} {confidence:.2f}"
                
                # 使用默认字体绘制
                try:
                    bbox = draw.textbbox((0, 0), label_en, font=ImageFont.load_default())
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # 绘制文本背景
                    bg_coords = [
                        x1, y1 - text_height - 10,
                        x1 + text_width + 10, y1
                    ]
                    draw.rectangle(bg_coords, fill=color)
                    
                    # 绘制文本
                    draw.text((x1 + 5, y1 - text_height - 5), label_en, fill=(255, 255, 255), font=ImageFont.load_default())
                except:
                    # 最后的备选方案：只绘制置信度
                    simple_label = f"{confidence:.2f}"
                    draw.text((x1 + 5, y1 - 20), simple_label, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 处理分割掩码（如果存在）
        for detection in detections:
            if 'mask' in detection:
                mask = detection['mask']
                color = detection['color']
                
                # 调整掩码大小以匹配图像
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # 创建彩色掩码
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0.5] = color
                
                # 混合掩码和原图像
                image = cv2.addWeighted(image, 0.8, colored_mask, 0.2, 0)
        
        return image
    
    def _empty_result(self) -> Dict:
        """返回空检测结果"""
        return {
            'detections': [],
            'annotated_image': None,
            'total_objects': 0,
            'traffic_objects': 0,
            'success': False
        }
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = Config.get_model_info(self.model_path)
        info.update({
            'device': self.device,
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'loaded': self.model is not None
        })
        return info
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        批量检测
        
        Args:
            images: 图像列表
            
        Returns:
            List[Dict]: 检测结果列表
        """
        results = []
        for image in images:
            result = self.detect(image, return_annotated=False)
            results.append(result)
        return results