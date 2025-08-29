"""
区域计数模块
提供区域内目标计数功能
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import time
import json

from config import Config

class CountingArea:
    """计数区域类"""
    
    def __init__(self, area_id: str, points: List[Tuple[int, int]], 
                 count_mode: str = "enter"):
        """
        初始化计数区域
        
        Args:
            area_id: 区域ID
            points: 区域多边形顶点列表 [(x1, y1), (x2, y2), ...]
            count_mode: 计数模式 ("enter" - 进入计数, "exit" - 离开计数, "both" - 双向计数)
        """
        self.id = area_id
        self.points = points
        self.count_mode = count_mode
        
        # 计数统计
        self.counts = defaultdict(int)  # 按类别统计
        self.total_count = 0
        self.enter_count = 0
        self.exit_count = 0
        
        # 区域内对象状态跟踪
        self.objects_in_area = {}  # 当前在区域内的对象
        self.object_history = {}   # 对象状态历史
        
        # 计数事件历史
        self.counting_events = []
        
        # 区域属性
        self.color = Config.COUNTING_LINE_COLOR  # 使用配置中的颜色
        self.thickness = Config.LINE_THICKNESS
        self.is_active = True
        
        # 创建时间
        self.created_time = time.time()
        
        # 计算区域属性
        self._calculate_area_properties()
    
    def _calculate_area_properties(self):
        """计算区域属性"""
        if len(self.points) >= 3:
            # 计算区域面积
            self.area_size = cv2.contourArea(np.array(self.points, dtype=np.int32))
            
            # 计算中心点
            moments = cv2.moments(np.array(self.points, dtype=np.int32))
            if moments['m00'] != 0:
                self.center = (
                    int(moments['m10'] / moments['m00']),
                    int(moments['m01'] / moments['m00'])
                )
            else:
                # 如果无法计算重心，使用几何中心
                x_coords = [p[0] for p in self.points]
                y_coords = [p[1] for p in self.points]
                self.center = (
                    int(sum(x_coords) / len(x_coords)),
                    int(sum(y_coords) / len(y_coords))
                )
        else:
            self.area_size = 0
            self.center = (0, 0)
    
    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        """
        判断点是否在区域内
        
        Args:
            point: 点坐标 (x, y)
            
        Returns:
            bool: 是否在区域内
        """
        if len(self.points) < 3:
            return False
        
        return cv2.pointPolygonTest(
            np.array(self.points, dtype=np.int32), 
            point, 
            False
        ) >= 0
    
    def update_object_status(self, obj_id: str, center: Tuple[float, float], 
                           class_info: Dict, timestamp: float = None):
        """
        更新对象状态
        
        Args:
            obj_id: 对象ID
            center: 对象中心点
            class_info: 类别信息
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        is_inside = self.is_point_inside(center)
        was_inside = obj_id in self.objects_in_area
        
        # 记录状态变化
        if obj_id not in self.object_history:
            self.object_history[obj_id] = []
        
        self.object_history[obj_id].append({
            'timestamp': timestamp,
            'position': center,
            'inside': is_inside
        })
        
        # 保持历史记录不超过100条
        if len(self.object_history[obj_id]) > 100:
            self.object_history[obj_id] = self.object_history[obj_id][-100:]
        
        # 检测进入或离开事件
        if is_inside and not was_inside:
            # 对象进入区域
            self.objects_in_area[obj_id] = {
                'enter_time': timestamp,
                'class_info': class_info,
                'positions': deque(maxlen=50)
            }
            self.objects_in_area[obj_id]['positions'].append(center)
            
            if self.count_mode in ["enter", "both"]:
                self._record_counting_event(obj_id, "enter", center, class_info, timestamp)
                
        elif not is_inside and was_inside:
            # 对象离开区域
            if obj_id in self.objects_in_area:
                stay_duration = timestamp - self.objects_in_area[obj_id]['enter_time']
                
                if self.count_mode in ["exit", "both"]:
                    self._record_counting_event(obj_id, "exit", center, class_info, timestamp, stay_duration)
                
                del self.objects_in_area[obj_id]
                
        elif is_inside and was_inside:
            # 对象仍在区域内，更新位置
            if obj_id in self.objects_in_area:
                self.objects_in_area[obj_id]['positions'].append(center)
    
    def _record_counting_event(self, obj_id: str, event_type: str, 
                             position: Tuple[float, float], class_info: Dict, 
                             timestamp: float, stay_duration: float = 0):
        """
        记录计数事件
        
        Args:
            obj_id: 对象ID
            event_type: 事件类型 ("enter" 或 "exit")
            position: 位置
            class_info: 类别信息
            timestamp: 时间戳
            stay_duration: 停留时间（仅对exit事件有效）
        """
        class_name = class_info.get('class_name', 'unknown')
        
        # 更新计数
        if event_type == "enter":
            self.enter_count += 1
            if self.count_mode == "enter":
                self.counts[class_name] += 1
                self.total_count += 1
        elif event_type == "exit":
            self.exit_count += 1
            if self.count_mode == "exit":
                self.counts[class_name] += 1
                self.total_count += 1
        
        if self.count_mode == "both":
            self.counts[class_name] += 1
            self.total_count += 1
        
        # 记录事件
        event = {
            'timestamp': timestamp,
            'area_id': self.id,
            'object_id': obj_id,
            'event_type': event_type,
            'class_name': class_name,
            'class_id': class_info.get('class_id', -1),
            'position': position,
            'confidence': class_info.get('confidence', 0.0),
            'stay_duration': stay_duration
        }
        
        self.counting_events.append(event)
        
        print(f"区域计数事件: {class_name} {event_type} 区域 {self.id}")

class AreaCounter:
    """区域计数器"""
    
    def __init__(self):
        """初始化计数器"""
        self.counting_areas = {}
        self.global_counts = defaultdict(int)
        self.total_events = 0
        
        # 事件历史
        self.event_history = deque(maxlen=1000)
        
        # 统计信息
        self.statistics = {
            'start_time': time.time(),
            'total_events': 0,
            'class_statistics': defaultdict(int),
            'area_statistics': defaultdict(int)
        }
    
    def add_area(self, area_id: str, points: List[Tuple[int, int]], 
                count_mode: str = "enter") -> bool:
        """
        添加计数区域
        
        Args:
            area_id: 区域ID
            points: 区域顶点列表
            count_mode: 计数模式
            
        Returns:
            bool: 是否添加成功
        """
        try:
            if len(points) < 3:
                print(f"区域 {area_id} 至少需要3个点")
                return False
            
            area = CountingArea(area_id, points, count_mode)
            self.counting_areas[area_id] = area
            return True
        except Exception as e:
            print(f"添加计数区域失败: {e}")
            return False
    
    def remove_area(self, area_id: str) -> bool:
        """
        移除计数区域
        
        Args:
            area_id: 区域ID
            
        Returns:
            bool: 是否移除成功
        """
        if area_id in self.counting_areas:
            del self.counting_areas[area_id]
            return True
        return False
    
    def update_tracking(self, tracking_results: Dict):
        """
        更新追踪结果并进行区域计数
        
        Args:
            tracking_results: 追踪结果字典
        """
        tracked_objects = tracking_results.get('tracked_objects', [])
        current_time = time.time()
        
        for obj in tracked_objects:
            obj_id = obj['id']
            center = obj['center']
            
            if center is None:
                continue
            
            class_info = {
                'class_id': obj['class_id'],
                'class_name': obj['class_name'],
                'confidence': obj.get('confidence', 0.0)
            }
            
            # 更新每个区域的对象状态
            for area in self.counting_areas.values():
                if area.is_active:
                    area.update_object_status(obj_id, center, class_info, current_time)
        
        # 清理过期的对象历史
        self._cleanup_expired_history()
    
    def _cleanup_expired_history(self, timeout: float = 60.0):
        """
        清理过期的对象历史
        
        Args:
            timeout: 超时时间（秒）
        """
        current_time = time.time()
        
        for area in self.counting_areas.values():
            # 清理不再活跃的对象
            expired_objects = []
            for obj_id, obj_info in area.objects_in_area.items():
                if current_time - obj_info['enter_time'] > timeout:
                    expired_objects.append(obj_id)
            
            for obj_id in expired_objects:
                del area.objects_in_area[obj_id]
    
    def draw_areas(self, image: np.ndarray, show_counts: bool = True,
                  show_objects: bool = True, enabled: bool = True) -> np.ndarray:
        """
        在图像上绘制计数区域
        
        Args:
            image: 输入图像
            show_counts: 是否显示计数信息
            show_objects: 是否显示区域内对象
            enabled: 是否启用区域显示
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        # 如果未启用或没有区域，直接返回原图像
        if not enabled or not self.counting_areas:
            return image
            
        # 转换为PIL图像以支持中文文本
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import os
        
        for area in self.counting_areas.values():
            if not area.is_active:
                continue
            
            # 绘制区域轮廓
            if len(area.points) >= 3:
                points = np.array(area.points, dtype=np.int32)
                cv2.polylines(image, [points], True, area.color, area.thickness)
                
                # 填充半透明区域
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], area.color)
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
            
            # 使用PIL绘制中文文本以避免问号问题
            if show_counts:
                # 转换图像格式
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                # 尝试加载中文字体
                try:
                    font_paths = [
                        "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
                        "C:/Windows/Fonts/simhei.ttf", # 黑体
                        "C:/Windows/Fonts/simsun.ttc", # 宋体
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
                        
                except Exception:
                    font = ImageFont.load_default()
                
                # 绘制区域标签
                label_text = f"区域 {area.id}"
                label_pos = (area.center[0] - 50, area.center[1] - 30)
                
                # 绘制文本背景
                bbox = draw.textbbox(label_pos, label_text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=area.color)
                
                # 绘制文本
                draw.text(label_pos, label_text, fill=(255, 255, 255), font=font)
                
                # 显示计数信息
                count_text = f"总数: {area.total_count}"
                count_pos = (area.center[0] - 50, area.center[1] - 5)
                
                # 绘制计数背景和文本
                bbox = draw.textbbox(count_pos, count_text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=area.color)
                draw.text(count_pos, count_text, fill=(255, 255, 255), font=font)
                
                # 显示详细计数
                if area.count_mode == "both":
                    detail_text = f"进: {area.enter_count} 出: {area.exit_count}"
                    detail_pos = (area.center[0] - 50, area.center[1] + 20)
                    
                    bbox = draw.textbbox(detail_pos, detail_text, font=font)
                    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=area.color)
                    draw.text(detail_pos, detail_text, fill=(255, 255, 255), font=font)
                
                # 转换回OpenCV格式
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 显示区域内对象
            if show_objects:
                for obj_id, obj_info in area.objects_in_area.items():
                    if obj_info['positions']:
                        pos = obj_info['positions'][-1]
                        cv2.circle(image, (int(pos[0]), int(pos[1])), 5, (0, 255, 255), -1)
        
        return image
    
    def get_area_statistics(self, area_id: str = None) -> Dict:
        """
        获取区域统计信息
        
        Args:
            area_id: 区域ID，为None时返回所有区域统计
            
        Returns:
            Dict: 统计信息
        """
        if area_id and area_id in self.counting_areas:
            area = self.counting_areas[area_id]
            return {
                'area_id': area.id,
                'total_count': area.total_count,
                'enter_count': area.enter_count,
                'exit_count': area.exit_count,
                'class_counts': dict(area.counts),
                'current_objects': len(area.objects_in_area),
                'recent_events': area.counting_events[-10:],
                'area_size': area.area_size,
                'active_duration': time.time() - area.created_time
            }
        else:
            # 返回所有区域的汇总统计
            all_stats = {}
            total_all_count = 0
            
            for aid, area in self.counting_areas.items():
                all_stats[aid] = {
                    'total_count': area.total_count,
                    'enter_count': area.enter_count,
                    'exit_count': area.exit_count,
                    'class_counts': dict(area.counts),
                    'current_objects': len(area.objects_in_area)
                }
                total_all_count += area.total_count
            
            return {
                'areas': all_stats,
                'total_all_count': total_all_count,
                'system_statistics': dict(self.statistics)
            }
    
    def export_area_data(self, filename: str = None) -> str:
        """
        导出区域计数数据
        
        Args:
            filename: 文件名，为None时自动生成
            
        Returns:
            str: 导出的文件路径
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"area_counting_data_{timestamp}.json"
        
        export_data = {
            'export_time': time.time(),
            'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'areas': {},
            'global_statistics': self.get_area_statistics(),
            'event_history': list(self.event_history)
        }
        
        # 导出每个区域的详细数据
        for area_id, area in self.counting_areas.items():
            export_data['areas'][area_id] = {
                'id': area.id,
                'points': area.points,
                'count_mode': area.count_mode,
                'total_count': area.total_count,
                'enter_count': area.enter_count,
                'exit_count': area.exit_count,
                'class_counts': dict(area.counts),
                'counting_events': area.counting_events,
                'area_size': area.area_size,
                'created_time': area.created_time
            }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return filename
        except Exception as e:
            print(f"导出数据失败: {e}")
            return ""
    
    def reset(self):
        """重置所有计数器"""
        for area in self.counting_areas.values():
            area.counts.clear()
            area.total_count = 0
            area.enter_count = 0
            area.exit_count = 0
            area.counting_events.clear()
            area.objects_in_area.clear()
            area.object_history.clear()
        
        self.global_counts.clear()
        self.total_events = 0
        self.event_history.clear()
        
        # 重置统计信息
        self.statistics = {
            'start_time': time.time(),
            'total_events': 0,
            'class_statistics': defaultdict(int),
            'area_statistics': defaultdict(int)
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict]:
        """
        获取最近的计数事件
        
        Args:
            count: 事件数量
            
        Returns:
            List[Dict]: 事件列表
        """
        return list(self.event_history)[-count:]