"""
自动数据保存器
在检测开始时自动创建数据文件，检测过程中实时保存数据
"""

import os
import csv
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional


class AutoDataSaver:
    """自动数据保存器 - 实时保存检测数据到CSV文件"""
    
    def __init__(self, output_dir: str = "saved_data"):
        """
        初始化自动数据保存器
        
        Args:
            output_dir: 数据保存目录
        """
        self.output_dir = output_dir
        self.session_id = None
        self.files = {}
        self.is_active = False
        self.lock = threading.Lock()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_session(self) -> str:
        """
        开始新的数据保存会话，立即创建数据文件
        
        Returns:
            str: 会话ID
        """
        with self.lock:
            # 生成会话ID（基于时间戳）
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.is_active = True
            
            # 立即创建数据文件
            self._create_data_files()
            
            print(f"数据保存会话已开始，会话ID: {self.session_id}")
            print(f"数据文件已创建在: {self.output_dir}")
            
            return self.session_id
    
    def _create_data_files(self):
        """创建数据保存文件"""
        self.files = {
            'detection': os.path.join(self.output_dir, f"检测数据_{self.session_id}.csv"),
            'tracking': os.path.join(self.output_dir, f"追踪数据_{self.session_id}.csv"),
            'crossing': os.path.join(self.output_dir, f"越线数据_{self.session_id}.csv"),
            'area': os.path.join(self.output_dir, f"区域计数_{self.session_id}.csv"),
        }
        
        # 创建检测数据文件并写入表头
        with open(self.files['detection'], 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间戳', '帧数', '类别', '置信度', 'X坐标', 'Y坐标', '宽度', '高度'])
        
        # 创建追踪数据文件并写入表头
        with open(self.files['tracking'], 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间戳', '帧数', '追踪ID', '类别', '置信度', 'X坐标', 'Y坐标', '宽度', '高度', '中心X', '中心Y'])
        
        # 创建越线数据文件并写入表头
        with open(self.files['crossing'], 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间戳', '帧数', '追踪ID', '类别', '线条ID', '越线方向', '越线次数'])
        
        # 创建区域计数文件并写入表头
        with open(self.files['area'], 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['时间戳', '帧数', '追踪ID', '类别', '区域ID', '事件类型', '区域内总数'])
    
    def add_detection_data(self, frame_count: int, detections: List[Dict]):
        """
        添加检测数据
        
        Args:
            frame_count: 帧数
            detections: 检测结果列表
        """
        if not self.is_active:
            return
        
        with self.lock:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                with open(self.files['detection'], 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    for det in detections:
                        writer.writerow([
                            timestamp,
                            frame_count,
                            det.get('class_name', ''),
                            f"{det.get('confidence', 0):.3f}",
                            int(det.get('bbox', [0, 0, 0, 0])[0]),
                            int(det.get('bbox', [0, 0, 0, 0])[1]),
                            int(det.get('bbox', [0, 0, 0, 0])[2]),
                            int(det.get('bbox', [0, 0, 0, 0])[3])
                        ])
            except Exception as e:
                print(f"保存检测数据时出错: {e}")
    
    def add_tracking_data(self, frame_count: int, tracked_objects: List[Dict]):
        """
        添加追踪数据
        
        Args:
            frame_count: 帧数
            tracked_objects: 追踪对象列表
        """
        if not self.is_active:
            return
        
        with self.lock:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                with open(self.files['tracking'], 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    for obj in tracked_objects:
                        bbox = obj.get('bbox', [0, 0, 0, 0])
                        center_x = int(bbox[0] + bbox[2] / 2)
                        center_y = int(bbox[1] + bbox[3] / 2)
                        
                        writer.writerow([
                            timestamp,
                            frame_count,
                            obj.get('track_id', ''),
                            obj.get('class_name', ''),
                            f"{obj.get('confidence', 0):.3f}",
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2]),
                            int(bbox[3]),
                            center_x,
                            center_y
                        ])
            except Exception as e:
                print(f"保存追踪数据时出错: {e}")
    
    def add_crossing_event(self, frame_count: int, event_data: Dict):
        """
        添加越线事件数据
        
        Args:
            frame_count: 帧数
            event_data: 越线事件数据
        """
        if not self.is_active:
            return
        
        with self.lock:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                with open(self.files['crossing'], 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        frame_count,
                        event_data.get('track_id', ''),
                        event_data.get('class_name', ''),
                        event_data.get('line_id', ''),
                        event_data.get('direction', ''),
                        event_data.get('crossing_count', 0)
                    ])
            except Exception as e:
                print(f"保存越线数据时出错: {e}")
    
    def add_area_event(self, frame_count: int, event_data: Dict):
        """
        添加区域计数事件数据
        
        Args:
            frame_count: 帧数
            event_data: 区域事件数据
        """
        if not self.is_active:
            return
        
        with self.lock:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                with open(self.files['area'], 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        frame_count,
                        event_data.get('track_id', ''),
                        event_data.get('class_name', ''),
                        event_data.get('area_id', ''),
                        event_data.get('event_type', ''),
                        event_data.get('area_count', 0)
                    ])
            except Exception as e:
                print(f"保存区域数据时出错: {e}")
    
    def stop_session(self):
        """停止数据保存会话"""
        with self.lock:
            if self.is_active:
                self.is_active = False
                print(f"数据保存会话已停止，会话ID: {self.session_id}")
                print(f"数据已保存到: {self.output_dir}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        获取当前会话信息
        
        Returns:
            Dict: 会话信息
        """
        return {
            'session_id': self.session_id,
            'is_active': self.is_active,
            'output_dir': self.output_dir,
            'files': self.files.copy() if self.files else {}
        }
    
    def get_saved_files(self) -> List[str]:
        """
        获取已保存的文件列表
        
        Returns:
            List[str]: 文件路径列表
        """
        if not self.files:
            return []
        
        existing_files = []
        for file_path in self.files.values():
            if os.path.exists(file_path):
                existing_files.append(file_path)
        
        return existing_files