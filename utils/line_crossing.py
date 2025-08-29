"""
越线检测模块
提供车辆越线计数功能
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import time
import json

from config import Config

class CountingLine:
    """计数线类"""
    
    def __init__(self, line_id: str, start_point: Tuple[int, int], 
                 end_point: Tuple[int, int], direction: str = "both"):
        """
        初始化计数线
        
        Args:
            line_id: 线条ID
            start_point: 起始点 (x, y)
            end_point: 结束点 (x, y)
            direction: 计数方向 ("up", "down", "left", "right", "both")
        """
        self.id = line_id
        self.start_point = start_point
        self.end_point = end_point
        self.direction = direction
        
        # 计算线条参数
        self._calculate_line_params()
        
        # 计数统计
        self.counts = defaultdict(int)  # 按类别统计
        self.total_count = 0
        self.direction_counts = {"positive": 0, "negative": 0}
        
        # 越线记录
        self.crossing_events = []
        self.object_states = {}  # 记录对象相对于线条的状态
        
        # 线条属性
        self.color = Config.COUNTING_LINE_COLOR
        self.thickness = Config.LINE_THICKNESS
        self.is_active = True
        
        # 创建时间
        self.created_time = time.time()
    
    def _calculate_line_params(self):
        """计算线条的数学参数"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # 线条向量
        self.vector = (x2 - x1, y2 - y1)
        
        # 线条长度
        self.length = np.sqrt(self.vector[0]**2 + self.vector[1]**2)
        
        # 法向量（用于判断点在线条的哪一侧）
        if self.length > 0:
            self.normal = (-self.vector[1] / self.length, self.vector[0] / self.length)
        else:
            self.normal = (0, 1)
        
        # 线条方程 ax + by + c = 0
        if x2 - x1 != 0:
            self.a = (y2 - y1) / (x2 - x1)
            self.b = -1
            self.c = y1 - self.a * x1
        else:
            # 垂直线
            self.a = 1
            self.b = 0
            self.c = -x1
    
    def point_side(self, point: Tuple[float, float]) -> float:
        """
        判断点相对于线条的位置
        
        Args:
            point: 点坐标 (x, y)
            
        Returns:
            float: >0 在一侧，<0 在另一侧，=0 在线上
        """
        x, y = point
        return self.a * x + self.b * y + self.c
    
    def is_crossing(self, old_point: Tuple[float, float], 
                   new_point: Tuple[float, float]) -> Optional[str]:
        """
        检测是否越线
        
        Args:
            old_point: 旧位置
            new_point: 新位置
            
        Returns:
            Optional[str]: "positive" 或 "negative" 或 None
        """
        old_side = self.point_side(old_point)
        new_side = self.point_side(new_point)
        
        # 检查是否跨越了线条
        if old_side * new_side < 0:  # 符号相反，说明越线了
            if old_side < 0 < new_side:
                return "positive"
            elif old_side > 0 > new_side:
                return "negative"
        
        return None
    
    def check_intersection(self, old_point: Tuple[float, float], 
                         new_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        计算轨迹与线条的交点
        
        Args:
            old_point: 旧位置
            new_point: 新位置
            
        Returns:
            Optional[Tuple[float, float]]: 交点坐标或None
        """
        x1, y1 = old_point
        x2, y2 = new_point
        x3, y3 = self.start_point
        x4, y4 = self.end_point
        
        # 计算两条线的交点
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # 平行线
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # 检查交点是否在两条线段上
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None

class LineCrossingCounter:
    """越线计数器"""
    
    def __init__(self):
        """初始化计数器"""
        self.counting_lines = {}
        self.global_counts = defaultdict(int)
        self.total_crossings = 0
        
        # 越线事件历史
        self.crossing_history = deque(maxlen=1000)
        
        # 统计信息
        self.statistics = {
            'start_time': time.time(),
            'total_events': 0,
            'class_statistics': defaultdict(int),
            'direction_statistics': defaultdict(int)
        }
    
    def add_line(self, line_id: str, start_point: Tuple[int, int], 
                end_point: Tuple[int, int], direction: str = "both") -> bool:
        """
        添加计数线
        
        Args:
            line_id: 线条ID
            start_point: 起始点
            end_point: 结束点
            direction: 计数方向
            
        Returns:
            bool: 是否添加成功
        """
        try:
            line = CountingLine(line_id, start_point, end_point, direction)
            self.counting_lines[line_id] = line
            return True
        except Exception as e:
            print(f"添加计数线失败: {e}")
            return False
    
    def remove_line(self, line_id: str) -> bool:
        """
        移除计数线
        
        Args:
            line_id: 线条ID
            
        Returns:
            bool: 是否移除成功
        """
        if line_id in self.counting_lines:
            del self.counting_lines[line_id]
            return True
        return False
    
    def update_tracking(self, tracking_results: Dict):
        """
        更新追踪结果并检测越线
        
        Args:
            tracking_results: 追踪结果字典
        """
        tracked_objects = tracking_results.get('tracked_objects', [])
        
        for obj in tracked_objects:
            obj_id = obj['id']
            current_pos = obj['center']
            
            if current_pos is None:
                continue
            
            # 检查每条计数线
            for line_id, line in self.counting_lines.items():
                if not line.is_active:
                    continue
                
                # 获取对象在该线条上的历史状态
                key = f"{obj_id}_{line_id}"
                
                if key in line.object_states:
                    old_pos = line.object_states[key]['position']
                    crossing_direction = line.is_crossing(old_pos, current_pos)
                    
                    if crossing_direction is not None:
                        # 检测到越线
                        self._record_crossing(line, obj, crossing_direction, current_pos)
                
                # 更新对象状态
                line.object_states[key] = {
                    'position': current_pos,
                    'last_update': time.time()
                }
        
        # 清理过期的对象状态
        self._cleanup_expired_states()
    
    def _record_crossing(self, line: CountingLine, obj: Dict, 
                        direction: str, crossing_point: Tuple[float, float]):
        """
        记录越线事件
        
        Args:
            line: 计数线对象
            obj: 追踪对象
            direction: 越线方向
            crossing_point: 越线点
        """
        # 检查方向是否符合计数要求
        if line.direction != "both" and line.direction != direction:
            return
        
        class_name = obj['class_name']
        
        # 更新计数
        line.counts[class_name] += 1
        line.total_count += 1
        line.direction_counts[direction] += 1
        
        # 更新全局计数
        self.global_counts[class_name] += 1
        self.total_crossings += 1
        
        # 记录越线事件
        crossing_event = {
            'timestamp': time.time(),
            'line_id': line.id,
            'object_id': obj['id'],
            'class_name': class_name,
            'class_id': obj['class_id'],
            'direction': direction,
            'crossing_point': crossing_point,
            'confidence': obj.get('confidence', 0.0),
            'velocity': obj.get('velocity', (0, 0))
        }
        
        line.crossing_events.append(crossing_event)
        self.crossing_history.append(crossing_event)
        
        # 更新统计信息
        self.statistics['total_events'] += 1
        self.statistics['class_statistics'][class_name] += 1
        self.statistics['direction_statistics'][direction] += 1
        
        print(f"检测到越线: {class_name} 通过线条 {line.id} ({direction}方向)")
    
    def _cleanup_expired_states(self, timeout: float = 30.0):
        """
        清理过期的对象状态
        
        Args:
            timeout: 超时时间（秒）
        """
        current_time = time.time()
        
        for line in self.counting_lines.values():
            expired_keys = []
            for key, state in line.object_states.items():
                if current_time - state['last_update'] > timeout:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del line.object_states[key]
    
    def draw_lines(self, image: np.ndarray, show_counts: bool = True) -> np.ndarray:
        """
        在图像上绘制计数线
        
        Args:
            image: 输入图像
            show_counts: 是否显示计数
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        if not self.counting_lines:
            return image
            
        # 首先绘制所有线条
        for line in self.counting_lines.values():
            if not line.is_active:
                continue
            # 绘制线条
            cv2.line(image, line.start_point, line.end_point, 
                    line.color, line.thickness)
        
        # 如果需要显示计数信息，转换为PIL图像处理中文文本
        if show_counts:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            import os
            
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
                            font = ImageFont.truetype(font_path, 18)
                            break
                        except:
                            continue
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except Exception:
                font = ImageFont.load_default()
            
            # 为每条线绘制文本信息
            for line in self.counting_lines.values():
                if not line.is_active:
                    continue
                    
                # 计算线条中点
                mid_point = (
                    (line.start_point[0] + line.end_point[0]) // 2,
                    (line.start_point[1] + line.end_point[1]) // 2
                )
                
                # 绘制线条ID
                line_text = f"线条 {line.id}"
                line_pos = (mid_point[0] - 30, mid_point[1] - 30)
                
                # 绘制文本背景
                bbox = draw.textbbox(line_pos, line_text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=line.color)
                
                # 绘制文本
                draw.text(line_pos, line_text, fill=(255, 255, 255), font=font)
                
                # 显示计数信息
                count_text = f"总数: {line.total_count}"
                count_pos = (mid_point[0] - 30, mid_point[1] - 5)
                
                # 绘制计数背景和文本
                bbox = draw.textbbox(count_pos, count_text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=line.color)
                draw.text(count_pos, count_text, fill=(255, 255, 255), font=font)
                
                # 显示方向计数
                if line.direction == "both":
                    dir_text = f"上{line.direction_counts['positive']} 下{line.direction_counts['negative']}"
                    dir_pos = (mid_point[0] - 30, mid_point[1] + 20)
                    
                    bbox = draw.textbbox(dir_pos, dir_text, font=font)
                    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=line.color)
                    draw.text(dir_pos, dir_text, fill=(255, 255, 255), font=font)
            
            # 转换回OpenCV格式
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    
    def get_line_statistics(self, line_id: str = None) -> Dict:
        """
        获取线条统计信息
        
        Args:
            line_id: 线条ID，为None时返回所有线条统计
            
        Returns:
            Dict: 统计信息
        """
        if line_id and line_id in self.counting_lines:
            line = self.counting_lines[line_id]
            return {
                'line_id': line.id,
                'total_count': line.total_count,
                'class_counts': dict(line.counts),
                'direction_counts': dict(line.direction_counts),
                'recent_events': line.crossing_events[-10:],  # 最近10个事件
                'active_duration': time.time() - line.created_time
            }
        else:
            # 返回所有线条的汇总统计
            all_stats = {}
            for lid, line in self.counting_lines.items():
                all_stats[lid] = {
                    'total_count': line.total_count,
                    'class_counts': dict(line.counts),
                    'direction_counts': dict(line.direction_counts)
                }
            
            return {
                'lines': all_stats,
                'global_counts': dict(self.global_counts),
                'total_crossings': self.total_crossings,
                'system_statistics': dict(self.statistics)
            }
    
    def export_crossing_data(self, filename: str = None) -> str:
        """
        导出越线数据
        
        Args:
            filename: 文件名，为None时自动生成
            
        Returns:
            str: 导出的文件路径
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"crossing_data_{timestamp}.json"
        
        export_data = {
            'export_time': time.time(),
            'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'lines': {},
            'global_statistics': self.get_line_statistics(),
            'crossing_history': list(self.crossing_history)
        }
        
        # 导出每条线的详细数据
        for line_id, line in self.counting_lines.items():
            export_data['lines'][line_id] = {
                'id': line.id,
                'start_point': line.start_point,
                'end_point': line.end_point,
                'direction': line.direction,
                'total_count': line.total_count,
                'class_counts': dict(line.counts),
                'direction_counts': dict(line.direction_counts),
                'crossing_events': line.crossing_events,
                'created_time': line.created_time
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
        for line in self.counting_lines.values():
            line.counts.clear()
            line.total_count = 0
            line.direction_counts = {"positive": 0, "negative": 0}
            line.crossing_events.clear()
            line.object_states.clear()
        
        self.global_counts.clear()
        self.total_crossings = 0
        self.crossing_history.clear()
        
        # 重置统计信息
        self.statistics = {
            'start_time': time.time(),
            'total_events': 0,
            'class_statistics': defaultdict(int),
            'direction_statistics': defaultdict(int)
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict]:
        """
        获取最近的越线事件
        
        Args:
            count: 事件数量
            
        Returns:
            List[Dict]: 事件列表
        """
        return list(self.crossing_history)[-count:]