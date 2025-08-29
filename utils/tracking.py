"""
多目标追踪模块
提供车辆追踪功能
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import time
import uuid
from PIL import Image, ImageDraw, ImageFont
import os

from config import Config

class TrackedObject:
    """被追踪的对象类"""
    
    def __init__(self, object_id: str, detection: Dict, frame_id: int):
        """
        初始化追踪对象
        
        Args:
            object_id: 唯一标识符
            detection: 检测结果字典
            frame_id: 帧ID
        """
        self.id = object_id
        self.class_id = detection['class_id']
        self.class_name = detection['class_name_zh']
        self.color = detection['color']
        
        # 位置信息
        self.current_position = None
        self.bboxes = deque(maxlen=10)      # 保存最近10个边界框
        
        # 初始位置
        bbox = detection['bbox']
        center = self._bbox_to_center(bbox)
        self.current_position = center
        self.bboxes.append(bbox)
        
        # 状态信息
        self.last_seen = frame_id
        self.disappeared_count = 0
        self.is_active = True
        self.confidence_history = deque(maxlen=10)
        self.confidence_history.append(detection['confidence'])
        
        # 运动信息
        self.velocity = (0, 0)
        self.direction = 0  # 角度
        self.distance_traveled = 0.0
        
        # 创建时间
        self.created_time = time.time()
        self.last_update_time = time.time()
    
    def update(self, detection: Dict, frame_id: int):
        """
        更新追踪对象
        
        Args:
            detection: 新的检测结果
            frame_id: 当前帧ID
        """
        bbox = detection['bbox']
        center = self._bbox_to_center(bbox)
        current_time = time.time()
        
        # 更新位置
        if self.current_position:
            last_center = self.current_position
            # 计算速度和方向
            dt = current_time - self.last_update_time
            if dt > 0:
                dx = center[0] - last_center[0]
                dy = center[1] - last_center[1]
                self.velocity = (dx / dt, dy / dt)
                
                # 计算方向角度
                if dx != 0 or dy != 0:
                    self.direction = np.arctan2(dy, dx) * 180 / np.pi
                
                # 累计移动距离
                distance = np.sqrt(dx**2 + dy**2)
                self.distance_traveled += distance
        
        self.current_position = center
        self.bboxes.append(bbox)
        
        # 更新状态
        self.last_seen = frame_id
        self.disappeared_count = 0
        self.is_active = True
        self.confidence_history.append(detection['confidence'])
        self.last_update_time = current_time
    
    def mark_disappeared(self):
        """标记对象消失"""
        self.disappeared_count += 1
        if self.disappeared_count > Config.MAX_DISAPPEARED:
            self.is_active = False
    
    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """获取当前位置"""
        return self.current_position
    
    def get_current_bbox(self) -> Optional[List[float]]:
        """获取当前边界框"""
        if self.bboxes:
            return list(self.bboxes[-1])
        return None

    
    def get_average_confidence(self) -> float:
        """获取平均置信度"""
        if self.confidence_history:
            return sum(self.confidence_history) / len(self.confidence_history)
        return 0.0
    
    def get_lifetime(self) -> float:
        """获取生存时间（秒）"""
        return time.time() - self.created_time
    
    def _bbox_to_center(self, bbox: List[float]) -> Tuple[float, float]:
        """将边界框转换为中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class MultiObjectTracker:
    """多目标追踪器"""
    
    def __init__(self):
        """初始化追踪器"""
        self.tracked_objects = {}  # 活跃的追踪对象
        self.inactive_objects = {}  # 非活跃的追踪对象
        self.next_object_id = 0
        self.frame_count = 0
        
        # 追踪参数
        self.max_distance = Config.MAX_DISTANCE
        self.max_disappeared = Config.MAX_DISAPPEARED
        
        # 统计信息
        self.total_tracked = 0
        self.class_counts = defaultdict(int)
    
    def update(self, detections: List[Dict]) -> Dict:
        """
        更新追踪器
        
        Args:
            detections: 当前帧的检测结果列表
            
        Returns:
            Dict: 追踪结果
        """
        self.frame_count += 1
        
        # 如果没有检测结果，标记所有对象为消失
        if len(detections) == 0:
            for obj in self.tracked_objects.values():
                obj.mark_disappeared()
            self._remove_inactive_objects()
            return self._get_tracking_results()
        
        # 如果没有正在追踪的对象，创建新的追踪对象
        if len(self.tracked_objects) == 0:
            for detection in detections:
                self._create_new_object(detection)
        else:
            # 执行数据关联
            self._associate_detections(detections)
        
        # 清理非活跃对象
        self._remove_inactive_objects()
        
        return self._get_tracking_results()
    
    def _associate_detections(self, detections: List[Dict]):
        """
        关联检测结果和追踪对象
        
        Args:
            detections: 检测结果列表
        """
        # 计算距离矩阵
        object_centers = []
        object_ids = []
        
        for obj_id, obj in self.tracked_objects.items():
            center = obj.get_current_position()
            if center:
                object_centers.append(center)
                object_ids.append(obj_id)
        
        detection_centers = []
        for detection in detections:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            detection_centers.append(center)
        
        # 计算距离矩阵
        if len(object_centers) > 0 and len(detection_centers) > 0:
            distances = np.zeros((len(object_centers), len(detection_centers)))
            
            for i, obj_center in enumerate(object_centers):
                for j, det_center in enumerate(detection_centers):
                    distances[i, j] = np.sqrt(
                        (obj_center[0] - det_center[0]) ** 2 +
                        (obj_center[1] - det_center[1]) ** 2
                    )
            
            # 简单的贪心匹配算法
            used_detections = set()
            used_objects = set()
            
            # 按距离从小到大排序
            matches = []
            for i in range(len(object_centers)):
                for j in range(len(detection_centers)):
                    if distances[i, j] < self.max_distance:
                        matches.append((distances[i, j], i, j))
            
            matches.sort(key=lambda x: x[0])
            
            # 执行匹配
            for distance, obj_idx, det_idx in matches:
                if obj_idx not in used_objects and det_idx not in used_detections:
                    obj_id = object_ids[obj_idx]
                    self.tracked_objects[obj_id].update(detections[det_idx], self.frame_count)
                    used_objects.add(obj_idx)
                    used_detections.add(det_idx)
            
            # 标记未匹配的对象为消失
            for i, obj_id in enumerate(object_ids):
                if i not in used_objects:
                    self.tracked_objects[obj_id].mark_disappeared()
            
            # 为未匹配的检测结果创建新对象
            for j in range(len(detections)):
                if j not in used_detections:
                    self._create_new_object(detections[j])
        else:
            # 如果没有已存在的对象，为所有检测创建新对象
            for detection in detections:
                self._create_new_object(detection)
    
    def _create_new_object(self, detection: Dict):
        """
        创建新的追踪对象
        
        Args:
            detection: 检测结果
        """
        object_id = str(uuid.uuid4())[:8]  # 生成短UUID
        new_object = TrackedObject(object_id, detection, self.frame_count)
        self.tracked_objects[object_id] = new_object
        
        # 更新统计
        self.total_tracked += 1
        self.class_counts[detection['class_name_zh']] += 1
    
    def _remove_inactive_objects(self):
        """移除非活跃对象"""
        inactive_ids = []
        for obj_id, obj in self.tracked_objects.items():
            if not obj.is_active:
                inactive_ids.append(obj_id)
        
        for obj_id in inactive_ids:
            obj = self.tracked_objects.pop(obj_id)
            self.inactive_objects[obj_id] = obj
    
    def _get_tracking_results(self) -> Dict:
        """获取追踪结果"""
        active_objects = []
        for obj in self.tracked_objects.values():
            if obj.is_active:
                obj_info = {
                    'id': obj.id,
                    'class_id': obj.class_id,
                    'class_name': obj.class_name,
                    'bbox': obj.get_current_bbox(),
                    'center': obj.get_current_position(),
                    'velocity': obj.velocity,
                    'direction': obj.direction,
                    'confidence': obj.get_average_confidence(),
                    'lifetime': obj.get_lifetime(),
                    'distance_traveled': obj.distance_traveled,
                    'color': obj.color
                }
                active_objects.append(obj_info)
        
        return {
            'tracked_objects': active_objects,
            'active_count': len(active_objects),
            'total_tracked': self.total_tracked,
            'class_counts': dict(self.class_counts),
            'frame_count': self.frame_count
        }
    
    def draw_tracks(self, image: np.ndarray, show_ids: bool = True, 
                   show_info: bool = False) -> np.ndarray:
        """
        在图像上绘制追踪结果
        
        Args:
            image: 输入图像
            show_ids: 是否显示ID
            show_info: 是否显示详细信息
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        # 如果有中文文本需要显示，使用PIL
        has_chinese_text = any(obj.is_active for obj in self.tracked_objects.values())
        
        if has_chinese_text and (show_ids or show_info):
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 加载中文字体
            try:
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/simsun.ttc",
                    "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/arial.ttf"
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, 16)
                            break
                        except:
                            continue
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except Exception:
                font = ImageFont.load_default()
        
        for obj in self.tracked_objects.values():
            if not obj.is_active:
                continue
                
            bbox = obj.get_current_bbox()
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            color = obj.color
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 使用PIL绘制中文文本
            if has_chinese_text and (show_ids or show_info):
                try:
                    # 绘制ID
                    if show_ids:
                        id_text = f"ID: {obj.id}"
                        draw.text((x1, y1 - 50), id_text, fill=(255, 255, 255), font=font)
                    
                    # 绘制类别名称
                    class_text = obj.class_name
                    draw.text((x1, y1 - 30), class_text, fill=(255, 255, 255), font=font)
                    
                    # 绘制详细信息
                    if show_info:
                        info_text = f"速度: {obj.velocity[0]:.1f},{obj.velocity[1]:.1f}"
                        draw.text((x1, y2 + 5), info_text, fill=(255, 255, 255), font=font)
                        
                except Exception:
                    # 如果PIL绘制失败，回退到OpenCV英文文本
                    if show_ids:
                        id_text = f"ID: {obj.id}"
                        cv2.putText(image, id_text, (x1, y1 - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 使用英文类别名称
                    from config import Config
                    class_text_en = Config.get_class_name_en(obj.class_id)
                    cv2.putText(image, class_text_en, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if show_info:
                        info_text = f"Speed: {obj.velocity[0]:.1f},{obj.velocity[1]:.1f}"
                        cv2.putText(image, info_text, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # 不需要中文时直接使用OpenCV
                if show_ids:
                    id_text = f"ID: {obj.id}"
                    cv2.putText(image, id_text, (x1, y1 - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if show_info:
                    info_text = f"Speed: {obj.velocity[0]:.1f},{obj.velocity[1]:.1f}"
                    cv2.putText(image, info_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 如果使用了PIL，转换回OpenCV格式
        if has_chinese_text and (show_ids or show_info):
            try:
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except:
                pass  # 如果转换失败，使用原图像
        
        return image
    
    def get_statistics(self) -> Dict:
        """获取追踪统计信息"""
        return {
            'total_objects': len(self.tracked_objects),
            'active_objects': len([obj for obj in self.tracked_objects.values() if obj.is_active]),
            'total_tracked_ever': self.total_tracked,
            'class_distribution': dict(self.class_counts),
            'frame_count': self.frame_count,
            'inactive_objects': len(self.inactive_objects)
        }
    
    def reset(self):
        """重置追踪器"""
        self.tracked_objects.clear()
        self.inactive_objects.clear()
        self.next_object_id = 0
        self.frame_count = 0
        self.total_tracked = 0
        self.class_counts.clear()