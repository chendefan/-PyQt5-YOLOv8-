"""
PyQt5 YOLOv8 交通检测系统主界面
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import threading
from pathlib import Path
import subprocess
import platform
import os

from config import Config
from utils import YOLODetector, MultiObjectTracker, LineCrossingCounter, AreaCounter, TrafficAnalyzer, DataExporter
from utils import AutoDataSaver

class VideoDisplayWidget(QLabel):
    """视频显示控件"""
    
    # 定义信号用于线条绘制和区域绘制
    line_drawn = pyqtSignal(tuple, tuple)  # (start_point, end_point)
    area_drawn = pyqtSignal(list)  # [point1, point2, point3, ...]
    
    def __init__(self):
        super().__init__()
        # 设置固定的最小和推荐尺寸
        self.setMinimumSize(800, 600)
        self.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("请加载视频文件或开启摄像头")
        self.setScaledContents(True)
        
        # 固定宽高比例 (4:3)
        self.aspect_ratio = 4.0 / 3.0
        
        # 设置尺寸策略以保持比例
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 鼠标事件相关
        self.drawing_mode = None  # 'line', 'area', 'rectangle', or None
        self.line_start = None
        self.line_end = None
        self.area_points = []  # 区域绘制点列表
        self.temp_area_points = []  # 临时区域点
        self.drawing_line = False
        self.drawing_area = False
        
        # 矩形绘制相关
        self.rect_start = None
        self.rect_end = None
        self.drawing_rect = False
        
        # 视频帧信息（用于坐标转换）
        self.video_frame_size = None  # (width, height) 原始视频帧尺寸
        self.display_rect = None      # 实际显示图像在控件中的位置和大小
        
    def enable_drawing(self, mode: str = None):
        """启用或禁用绘制模式"""
        self.drawing_mode = mode  # 'line', 'area', 'rectangle', or None
        if mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.area_points.clear()
            self.temp_area_points.clear()
            self.drawing_line = False
            self.drawing_area = False
            self.drawing_rect = False
    
    def enable_line_drawing(self, enabled=True):
        """启用或禁用线条绘制（保持向后兼容）"""
        if enabled:
            self.enable_drawing('line')
        else:
            self.enable_drawing(None)
    
    def enable_area_drawing(self, enabled=True):
        """启用或禁用区域绘制（点击模式）"""
        if enabled:
            self.enable_drawing('area')
        else:
            self.enable_drawing(None)
    
    def enable_rectangle_drawing(self, enabled=True):
        """启用或禁用矩形绘制（拖拽模式）"""
        if enabled:
            self.enable_drawing('rectangle')
        else:
            self.enable_drawing(None)
    
    def set_video_frame_info(self, frame_width: int, frame_height: int):
        """
        设置视频帧信息，用于坐标转换
        
        Args:
            frame_width: 视频帧宽度
            frame_height: 视频帧高度
        """
        self.video_frame_size = (frame_width, frame_height)
        self.update_display_rect()
    
    def update_display_rect(self):
        """
        更新显示区域信息（在控件中的实际显示位置和大小）
        """
        if not self.video_frame_size:
            return
            
        pixmap = self.pixmap()
        if not pixmap:
            return
            
        # 获取控件尺寸
        widget_size = self.size()
        widget_width = widget_size.width()
        widget_height = widget_size.height()
        
        # 获取显示图像尺寸
        pixmap_size = pixmap.size()
        pixmap_width = pixmap_size.width()
        pixmap_height = pixmap_size.height()
        
        # 计算居中显示的位置
        x_offset = (widget_width - pixmap_width) // 2
        y_offset = (widget_height - pixmap_height) // 2
        
        self.display_rect = QRect(x_offset, y_offset, pixmap_width, pixmap_height)
    
    def widget_to_frame_coordinates(self, widget_x: int, widget_y: int) -> tuple:
        """
        将控件坐标转换为视频帧坐标
        
        Args:
            widget_x: 控件X坐标
            widget_y: 控件Y坐标
            
        Returns:
            tuple: (视频帧X坐标, 视频帧Y坐标)
        """
        if not self.video_frame_size or not self.display_rect:
            # 如果没有视频信息，直接返回原坐标
            return (widget_x, widget_y)
        
        # 转换为相对于显示区域的坐标
        relative_x = widget_x - self.display_rect.x()
        relative_y = widget_y - self.display_rect.y()
        
        # 检查是否在显示区域内
        if (relative_x < 0 or relative_x >= self.display_rect.width() or 
            relative_y < 0 or relative_y >= self.display_rect.height()):
            # 如果在显示区域外，返回原坐标
            return (widget_x, widget_y)
        
        # 计算缩放比例
        display_width = self.display_rect.width()
        display_height = self.display_rect.height()
        frame_width, frame_height = self.video_frame_size
        
        scale_x = frame_width / display_width
        scale_y = frame_height / display_height
        
        # 转换到视频帧坐标
        frame_x = int(relative_x * scale_x)
        frame_y = int(relative_y * scale_y)
        
        # 确保坐标在合理范围内
        frame_x = max(0, min(frame_x, frame_width - 1))
        frame_y = max(0, min(frame_y, frame_height - 1))
        
        return (frame_x, frame_y)
    
    def frame_to_widget_coordinates(self, frame_x: int, frame_y: int) -> tuple:
        """
        将视频帧坐标转换为控件坐标
        
        Args:
            frame_x: 视频帧X坐标
            frame_y: 视频帧Y坐标
            
        Returns:
            tuple: (控件X坐标, 控件Y坐标)
        """
        if not self.video_frame_size or not self.display_rect:
            # 如果没有视频信息，直接返回原坐标
            return (frame_x, frame_y)
        
        # 计算缩放比例
        display_width = self.display_rect.width()
        display_height = self.display_rect.height()
        frame_width, frame_height = self.video_frame_size
        
        scale_x = display_width / frame_width
        scale_y = display_height / frame_height
        
        # 转换到显示区域坐标
        relative_x = int(frame_x * scale_x)
        relative_y = int(frame_y * scale_y)
        
        # 转换为控件坐标
        widget_x = relative_x + self.display_rect.x()
        widget_y = relative_y + self.display_rect.y()
        
        return (widget_x, widget_y)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_mode:
            # 获取控件坐标
            widget_pos = (event.x(), event.y())
            
            if self.drawing_mode == 'line':
                self.drawing_line = True
                self.line_start = widget_pos
            elif self.drawing_mode == 'area':
                self.area_points.append(widget_pos)
                self.temp_area_points = self.area_points.copy()
                self.drawing_area = True
                self.update()
            elif self.drawing_mode == 'rectangle':
                self.drawing_rect = True
                self.rect_start = widget_pos
                self.rect_end = widget_pos
        elif event.button() == Qt.RightButton and self.drawing_mode == 'area':
            # 右键完成区域绘制
            if len(self.area_points) >= 3:
                # 转换坐标后发送信号
                frame_points = [self.widget_to_frame_coordinates(x, y) for x, y in self.area_points]
                self.area_drawn.emit(frame_points)
                self.enable_drawing(None)
            else:
                print("区域至少需要3个点")
    
    def mouseMoveEvent(self, event):
        if self.drawing_mode == 'line' and self.drawing_line and self.line_start:
            self.update()
        elif self.drawing_mode == 'area' and self.drawing_area:
            # 更新临时区域点列表，显示预览
            if self.area_points:
                cursor_pos = (event.x(), event.y())
                self.temp_area_points = self.area_points.copy() + [cursor_pos]
                self.update()
        elif self.drawing_mode == 'rectangle' and self.drawing_rect:
            # 更新矩形终点
            self.rect_end = (event.x(), event.y())
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing_mode == 'line' and self.drawing_line:
                self.line_end = (event.x(), event.y())
                self.drawing_line = False
                
                # 转换坐标后发送线条绘制完成信号
                frame_start = self.widget_to_frame_coordinates(*self.line_start)
                frame_end = self.widget_to_frame_coordinates(*self.line_end)
                self.line_drawn.emit(frame_start, frame_end)
                
                # 禁用线条绘制模式
                self.enable_drawing(None)
            elif self.drawing_mode == 'rectangle' and self.drawing_rect:
                self.rect_end = (event.x(), event.y())
                self.drawing_rect = False
                
                # 将矩形转换为区域点列表（四个角点）
                x1, y1 = self.rect_start
                x2, y2 = self.rect_end
                
                # 确保矩形有一定大小
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    # 创建矩形的四个角点（控件坐标）
                    rectangle_points = [
                        (min(x1, x2), min(y1, y2)),  # 左上
                        (max(x1, x2), min(y1, y2)),  # 右上
                        (max(x1, x2), max(y1, y2)),  # 右下
                        (min(x1, x2), max(y1, y2))   # 左下
                    ]
                    
                    # 转换为视频帧坐标后发送区域绘制完成信号
                    frame_points = [self.widget_to_frame_coordinates(x, y) for x, y in rectangle_points]
                    self.area_drawn.emit(frame_points)
                
                # 禁用矩形绘制模式
                self.enable_drawing(None)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制临时线条
        if self.drawing_mode == 'line' and self.drawing_line and self.line_start:
            painter.setPen(QPen(Qt.green, 3, Qt.SolidLine))
            # 获取当前鼠标位置
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            painter.drawLine(QPoint(*self.line_start), cursor_pos)
        
        # 绘制临时区域（点击模式）
        elif self.drawing_mode == 'area' and len(self.temp_area_points) >= 2:
            painter.setPen(QPen(Qt.blue, 3, Qt.SolidLine))
            
            # 绘制区域边界
            for i in range(len(self.temp_area_points)):
                start_point = QPoint(*self.temp_area_points[i])
                end_point = QPoint(*self.temp_area_points[(i + 1) % len(self.temp_area_points)])
                painter.drawLine(start_point, end_point)
            
            # 绘制顶点
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.setBrush(Qt.red)
            for point in self.temp_area_points:
                painter.drawEllipse(QPoint(*point), 4, 4)
        
        # 绘制临时矩形（拖拽模式）
        elif self.drawing_mode == 'rectangle' and self.drawing_rect and self.rect_start and self.rect_end:
            painter.setPen(QPen(Qt.cyan, 3, Qt.SolidLine))
            
            # 绘制矩形
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            
            rect = QRect(
                min(x1, x2), min(y1, y2),
                abs(x2 - x1), abs(y2 - y1)
            )
            painter.drawRect(rect)
            
            # 绘制角点
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.setBrush(Qt.red)
            painter.drawEllipse(QPoint(x1, y1), 4, 4)  # 起点
            painter.drawEllipse(QPoint(x2, y2), 4, 4)  # 终点
        
        painter.end()
    
    def setPixmap(self, pixmap):
        """重写setPixmap方法以自动更新显示信息"""
        super().setPixmap(pixmap)
        # 更新显示区域信息
        self.update_display_rect()
    
    def resizeEvent(self, event):
        """重写resize事件以保持固定宽高比"""
        super().resizeEvent(event)
        # 根据窗口大小调整视频显示区域的比例
        new_size = event.size()
        width = new_size.width()
        height = new_size.height()
        
        # 计算保持宽高比的新尺寸
        if width / height > self.aspect_ratio:
            # 窗口太宽，以高度为准
            new_width = int(height * self.aspect_ratio)
            new_height = height
        else:
            # 窗口太高，以宽度为准
            new_width = width
            new_height = int(width / self.aspect_ratio)
        
        # 应用新尺寸（通过stylesheet保持居中）
        self.setFixedSize(new_width, new_height)
        
        # 更新显示区域信息
        self.update_display_rect()
    
    def sizeHint(self):
        """返回推荐大小"""
        return QSize(800, 600)
    
    def minimumSizeHint(self):
        """返回最小大小"""
        return QSize(400, 300)

class ControlPanelWidget(QWidget):
    """控制面板控件"""
    
    def __init__(self):
        super().__init__()
        # 设置固定宽度以保持布局比例
        self.setMinimumWidth(350)
        self.setMaximumWidth(450)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 文件控制组
        file_group = QGroupBox("文件控制")
        file_layout = QVBoxLayout()
        
        self.load_video_btn = QPushButton("加载视频文件")
        self.load_camera_btn = QPushButton("开启摄像头")
        self.start_btn = QPushButton("开始检测")  # 新增开始按钮
        self.stop_btn = QPushButton("停止")
        
        # 初始状态：开始按钮禁用
        self.start_btn.setEnabled(False)
        
        file_layout.addWidget(self.load_video_btn)
        file_layout.addWidget(self.load_camera_btn)
        file_layout.addWidget(self.start_btn)
        file_layout.addWidget(self.stop_btn)
        file_group.setLayout(file_layout)
        
        # 检测控制组
        detection_group = QGroupBox("检测控制")
        detection_layout = QVBoxLayout()
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(['auto', 'cpu', 'cuda'])
        self.device_combo.setCurrentText('auto')
        device_layout.addWidget(self.device_combo)
        detection_layout.addLayout(device_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
        self.model_combo.setCurrentText('yolov8s.pt')
        model_layout.addWidget(self.model_combo)
        detection_layout.addLayout(model_layout)
        
        # 置信度设置
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        detection_layout.addLayout(conf_layout)
        
        # IoU设置
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU阈值:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 90)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        detection_layout.addLayout(iou_layout)
        
        detection_group.setLayout(detection_layout)
        
        # 追踪控制组
        tracking_group = QGroupBox("追踪控制")
        tracking_layout = QVBoxLayout()
        
        self.tracking_enabled = QCheckBox("启用多目标追踪")
        self.tracking_enabled.setChecked(True)
        self.show_ids = QCheckBox("显示ID")
        self.show_ids.setChecked(True)
        
        tracking_layout.addWidget(self.tracking_enabled)
        tracking_layout.addWidget(self.show_ids)
        tracking_group.setLayout(tracking_layout)
        
        # 越线检测组
        crossing_group = QGroupBox("越线检测")
        crossing_layout = QVBoxLayout()
        
        self.crossing_enabled = QCheckBox("启用越线检测")
        self.crossing_enabled.setChecked(True)
        self.add_line_btn = QPushButton("添加计数线")
        self.clear_lines_btn = QPushButton("清除所有线条")
        self.reset_counts_btn = QPushButton("重置计数")
        
        crossing_layout.addWidget(self.crossing_enabled)
        crossing_layout.addWidget(self.add_line_btn)
        crossing_layout.addWidget(self.clear_lines_btn)
        crossing_layout.addWidget(self.reset_counts_btn)
        crossing_group.setLayout(crossing_layout)
        
        # 区域计数组
        area_group = QGroupBox("区域计数")
        area_layout = QVBoxLayout()
        
        self.area_enabled = QCheckBox("启用区域计数")
        self.area_enabled.setChecked(True)
        
        # 计数模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("计数模式:"))
        self.area_mode_combo = QComboBox()
        self.area_mode_combo.addItems(['进入计数', '离开计数', '双向计数'])
        self.area_mode_combo.setCurrentText('进入计数')
        mode_layout.addWidget(self.area_mode_combo)
        area_layout.addLayout(mode_layout)
        
        # 简化为只有矩形绘制模式
        self.add_area_btn = QPushButton("添加计数区域")
        self.clear_areas_btn = QPushButton("清除所有区域")
        self.reset_area_counts_btn = QPushButton("重置区域计数")
        
        area_layout.addWidget(self.area_enabled)
        area_layout.addWidget(self.add_area_btn)
        area_layout.addWidget(self.clear_areas_btn)
        area_layout.addWidget(self.reset_area_counts_btn)
        area_group.setLayout(area_layout)
        
        # 数据保存组
        save_group = QGroupBox("数据保存")
        save_layout = QVBoxLayout()
        
        # 保存状态显示
        self.save_status = QLabel("未开始检测")
        self.save_status.setStyleSheet("color: gray; font-size: 11px; padding: 5px; border: 1px solid #ddd; border-radius: 3px;")
        
        # 查看保存文件按钮
        self.view_files_btn = QPushButton("查看保存的文件")
        self.view_files_btn.setEnabled(False)
        
        # 保存目录设置
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("保存目录:"))
        self.save_dir_label = QLabel("saved_data")
        self.save_dir_label.setStyleSheet("font-size: 9px; color: #666;")
        dir_layout.addWidget(self.save_dir_label)
        
        save_layout.addWidget(QLabel("保存状态:"))
        save_layout.addWidget(self.save_status)
        save_layout.addLayout(dir_layout)
        save_layout.addWidget(self.view_files_btn)
        save_group.setLayout(save_layout)
        
        # 添加所有组到主布局
        layout.addWidget(file_group)
        layout.addWidget(detection_group)
        layout.addWidget(tracking_group)
        layout.addWidget(crossing_group)
        layout.addWidget(area_group)
        layout.addWidget(save_group)
        layout.addStretch()
        
        self.setLayout(layout)

class StatisticsWidget(QWidget):
    """统计信息控件"""
    
    def __init__(self):
        super().__init__()
        # 设置固定大小以保持布局比例
        self.setMinimumHeight(120)
        self.setMaximumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 实时统计
        title_label = QLabel("实时统计信息:")
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(self.stats_text)
        
        self.setLayout(layout)
    
    def update_statistics(self, stats_text):
        """更新统计信息"""
        self.stats_text.setPlainText(stats_text)

class VideoProcessor(QObject):
    """视频处理器 - 在单独线程中运行"""
    
    # 定义信号用于线程间通信
    frame_ready = pyqtSignal(np.ndarray)  # 新帧准备好
    statistics_ready = pyqtSignal(str)    # 统计信息准备好
    processing_finished = pyqtSignal()    # 处理完成
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.tracker = None
        self.line_counter = None
        self.area_counter = None  # 新增区域计数器
        self.data_saver = None  # 新增数据保存器
        self.video_capture = None
        self.is_processing = False
        self.frame_count = 0
        self.control_settings = {}
        
    def set_components(self, detector, tracker, line_counter, area_counter, data_saver):
        """设置处理组件"""
        self.detector = detector
        self.tracker = tracker
        self.line_counter = line_counter
        self.area_counter = area_counter
        self.data_saver = data_saver
        
    def set_video_source(self, video_capture):
        """设置视频源"""
        self.video_capture = video_capture
        
    def update_settings(self, settings):
        """更新控制设置"""
        self.control_settings = settings
        
    def start_processing(self):
        """开始处理"""
        self.is_processing = True
        # 开始数据保存会话
        if self.data_saver:
            self.data_saver.start_session()
        self.process_video()
        
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        # 停止数据保存会话
        if self.data_saver:
            self.data_saver.stop_session()
        
    def process_video(self):
        """视频处理主循环"""
        while self.is_processing and self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            try:
                # 执行检测
                detection_results = self.detector.detect(frame)
                
                if detection_results['success']:
                    detections = detection_results['detections']
                    annotated_frame = detection_results['annotated_image']
                    
                    # 保存检测数据
                    if self.data_saver:
                        self.data_saver.add_detection_data(self.frame_count, detections)
                    
                    # 执行追踪
                    if self.control_settings.get('tracking_enabled', True):
                        tracking_results = self.tracker.update(detections)
                        
                        # 绘制追踪结果
                        if tracking_results['tracked_objects']:
                            annotated_frame = self.tracker.draw_tracks(
                                annotated_frame,
                                show_ids=self.control_settings.get('show_ids', True)
                            )
                            
                            # 保存追踪数据
                            if self.data_saver:
                                self.data_saver.add_tracking_data(self.frame_count, tracking_results['tracked_objects'])
                            
                            # 越线检测
                            if self.control_settings.get('crossing_enabled', True):
                                self.line_counter.update_tracking(tracking_results)
                            
                            # 区域计数
                            if self.control_settings.get('area_enabled', True):
                                self.area_counter.update_tracking(tracking_results)
                    
                    # 绘制计数线
                    if self.control_settings.get('crossing_enabled', True):
                        annotated_frame = self.line_counter.draw_lines(annotated_frame, show_counts=True)
                    
                    # 绘制计数区域
                    if self.control_settings.get('area_enabled', True):
                        annotated_frame = self.area_counter.draw_areas(
                            annotated_frame, 
                            show_counts=True, 
                            enabled=True
                        )
                    
                    # 发送帧信号
                    self.frame_ready.emit(annotated_frame)
                    
                    # 准备统计信息
                    stats_text = f"帧数: {self.frame_count}\n"
                    
                    if self.control_settings.get('tracking_enabled', True):
                        tracking_stats = self.tracker.get_statistics()
                        stats_text += f"当前追踪对象: {tracking_stats['active_objects']}\n"
                        stats_text += f"总追踪对象: {tracking_stats['total_tracked_ever']}\n"
                    
                    if self.control_settings.get('crossing_enabled', True):
                        crossing_stats = self.line_counter.get_line_statistics()
                        stats_text += f"总越线次数: {crossing_stats.get('total_crossings', 0)}\n"
                    
                    if self.control_settings.get('area_enabled', True):
                        area_stats = self.area_counter.get_area_statistics()
                        stats_text += f"区域计数总数: {area_stats.get('total_all_count', 0)}\n"
                    
                    # 发送统计信息信号
                    self.statistics_ready.emit(stats_text)
                    
            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue
            
            # 控制帧率
            time.sleep(0.03)  # 约30fps
            
        self.processing_finished.emit()

class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_system()
        self.init_ui()
        self.setup_connections()
        self.setup_video_processor()
        
    def init_system(self):
        """初始化系统组件"""
        self.detector = YOLODetector()
        self.tracker = MultiObjectTracker()
        self.line_counter = LineCrossingCounter()
        self.area_counter = AreaCounter()  # 新增区域计数器
        self.analyzer = TrafficAnalyzer()
        self.exporter = DataExporter()
        self.data_saver = AutoDataSaver()
        
        # 视频处理相关
        self.video_capture = None
        self.video_thread = None
        self.video_processor = None
        self.is_processing = False
        self.frame_count = 0
        
    def setup_video_processor(self):
        """设置视频处理器和线程"""
        # 创建视频处理器
        self.video_processor = VideoProcessor()
        self.video_processor.set_components(self.detector, self.tracker, self.line_counter, self.area_counter,self.data_saver)
        
        # 连接信号
        self.video_processor.frame_ready.connect(self.update_display_safe)
        self.video_processor.statistics_ready.connect(self.update_statistics_safe)
        self.video_processor.processing_finished.connect(self.on_processing_finished)
        
        # 创建线程
        self.video_thread = QThread()
        self.video_processor.moveToThread(self.video_thread)
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("YOLOv8 交通检测系统")
        
        # 设置固定窗口大小和最小大小
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        
        # 设置窗口的宽高比约束
        self.window_aspect_ratio = 1400.0 / 900.0
        
        # 创建中央控件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 使用QSplitter来实现固定比例的布局
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        
        # 左侧视频显示区域容器
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 视频显示控件
        self.video_display = VideoDisplayWidget()
        
        # 连接视频显示控件的信号
        self.video_display.line_drawn.connect(self.on_line_drawn)
        self.video_display.area_drawn.connect(self.on_area_drawn)
        
        # 创建视频显示容器以保持居中
        video_container = QWidget()
        video_container_layout = QHBoxLayout(video_container)
        video_container_layout.addWidget(self.video_display)
        video_container_layout.setAlignment(Qt.AlignCenter)
        
        left_layout.addWidget(video_container, 4)  # 视频区域占4/5
        
        # 统计信息区域
        self.statistics_widget = StatisticsWidget()
        self.statistics_widget.setMaximumHeight(150)
        left_layout.addWidget(self.statistics_widget, 1)  # 统计区域占1/5
        
        # 右侧控制面板
        self.control_panel = ControlPanelWidget()
        
        # 添加到分割器
        main_splitter.addWidget(left_container)
        main_splitter.addWidget(self.control_panel)
        
        # 设置分割器比例 (75% : 25%)
        main_splitter.setSizes([1050, 350])  # 总计1400
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_splitter)
        
        central_widget.setLayout(main_layout)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar().showMessage("系统就绪")
    
    def resizeEvent(self, event):
        """重写resize事件以保持窗口比例"""
        super().resizeEvent(event)
        
        # 获取新窗口大小
        new_size = event.size()
        width = new_size.width()
        height = new_size.height()
        
        # 检查是否需要调整比例
        current_ratio = width / height
        
        # 如果比例偏差太大，调整窗口大小
        if abs(current_ratio - self.window_aspect_ratio) > 0.1:
            if current_ratio > self.window_aspect_ratio:
                # 窗口太宽，调整宽度
                new_width = int(height * self.window_aspect_ratio)
                self.resize(new_width, height)
            else:
                # 窗口太高，调整高度
                new_height = int(width / self.window_aspect_ratio)
                self.resize(width, new_height)
    
    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        # 确保初始窗口大小符合比例
        self.adjustSize()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        load_video_action = QAction('加载视频文件', self)
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)
        
        load_camera_action = QAction('开启摄像头', self)
        load_camera_action.triggered.connect(self.start_camera)
        file_menu.addAction(load_camera_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('导出数据', self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """设置信号连接"""
        # 控制按钮连接
        self.control_panel.load_video_btn.clicked.connect(self.load_video)
        self.control_panel.load_camera_btn.clicked.connect(self.start_camera)
        self.control_panel.start_btn.clicked.connect(self.start_detection)  # 新增开始按钮连接
        self.control_panel.stop_btn.clicked.connect(self.stop_processing)
        
        # 参数滑块连接
        self.control_panel.conf_slider.valueChanged.connect(self.update_confidence)
        self.control_panel.iou_slider.valueChanged.connect(self.update_iou)
        
        # 越线检测按钮连接
        self.control_panel.add_line_btn.clicked.connect(self.add_counting_line)
        self.control_panel.clear_lines_btn.clicked.connect(self.clear_counting_lines)
        self.control_panel.reset_counts_btn.clicked.connect(self.reset_counts)
        
        
        # 模型选择连接
        self.control_panel.model_combo.currentTextChanged.connect(self.change_model)
        self.control_panel.device_combo.currentTextChanged.connect(self.change_device)
        
        # 设置变更连接
        self.control_panel.tracking_enabled.stateChanged.connect(self.update_processor_settings)
        self.control_panel.show_ids.stateChanged.connect(self.update_processor_settings)
        self.control_panel.crossing_enabled.stateChanged.connect(self.update_processor_settings)
        self.control_panel.area_enabled.stateChanged.connect(self.update_processor_settings)
        
        # 区域计数按钮连接
        self.control_panel.add_area_btn.clicked.connect(self.add_rectangle_counting_area)
        self.control_panel.clear_areas_btn.clicked.connect(self.clear_counting_areas)
        self.control_panel.reset_area_counts_btn.clicked.connect(self.reset_area_counts)
        
        # 数据保存按钮连接
        self.control_panel.view_files_btn.clicked.connect(self.view_saved_files)
    
    def load_video(self):
        """加载视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)"
        )
        
        if file_path:
            self.stop_processing()  # 停止当前处理
            
            # 加载视频文件
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                # 显示第一帧
                ret, frame = self.video_capture.read()
                if ret:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
                    # 显示静态帧
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    
                    # 设置视频帧信息用于坐标转换
                    self.video_display.set_video_frame_info(w, h)
                    
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.video_display.setPixmap(scaled_pixmap)
                
                # 启用开始按钮
                self.control_panel.start_btn.setEnabled(True)
                self.statusBar().showMessage(f"已加载视频: {Path(file_path).name}，点击'开始检测'按钮开始处理")
            else:
                QMessageBox.warning(self, "错误", "无法打开视频文件")
    
    def start_camera(self):
        """开启摄像头"""
        self.stop_processing()
        self.video_capture = cv2.VideoCapture(0)
        if self.video_capture.isOpened():
            # 显示第一帧
            ret, frame = self.video_capture.read()
            if ret:
                # 显示静态帧
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                
                # 设置视频帧信息用于坐标转换
                self.video_display.set_video_frame_info(w, h)
                
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_display.setPixmap(scaled_pixmap)
            
            # 启用开始按钮
            self.control_panel.start_btn.setEnabled(True)
            self.statusBar().showMessage("摄像头已开启，点击'开始检测'按钮开始处理")
        else:
            QMessageBox.warning(self, "错误", "无法打开摄像头")
    
    def start_detection(self):
        """开始检测处理"""
        if self.video_capture and self.video_capture.isOpened():
            self.start_processing()
            # 禁用开始按钮，启用停止按钮
            self.control_panel.start_btn.setEnabled(False)
            self.control_panel.stop_btn.setEnabled(True)
            self.statusBar().showMessage("正在进行检测处理...")
        else:
            QMessageBox.warning(self, "错误", "请先加载视频文件或开启摄像头")
    
    def start_processing(self):
        """开始视频处理"""
        if not self.is_processing and self.video_capture:
            self.is_processing = True
            
            # 开始数据保存会话
            if self.data_saver:
                session_id = self.data_saver.start_session()
                self.update_data_save_status(f"正在保存数据 (会话: {session_id})", True)
            
            # 设置视频源和设置
            self.video_processor.set_video_source(self.video_capture)
            self.update_processor_settings()
            
            # 启动线程
            if not self.video_thread.isRunning():
                self.video_thread.started.connect(self.video_processor.start_processing)
                self.video_thread.start()
            else:
                self.video_processor.start_processing()
    
    def stop_processing(self):
        """停止视频处理"""
        if self.is_processing:
            self.is_processing = False
            
            # 停止数据保存会话
            if self.data_saver:
                self.data_saver.stop_session()
                saved_files = self.data_saver.get_saved_files()
                file_count = len(saved_files)
                self.update_data_save_status(f"数据保存完成 (共{file_count}个文件)", True)
            
            # 停止处理器
            if self.video_processor:
                self.video_processor.stop_processing()
            
            # 停止线程
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.quit()
                self.video_thread.wait()
        
        # 重置按钮状态
        if hasattr(self, 'control_panel'):
            if self.video_capture and self.video_capture.isOpened():
                # 如果有视频源，启用开始按钮
                self.control_panel.start_btn.setEnabled(True)
            else:
                # 如果没有视频源，禁用开始按钮
                self.control_panel.start_btn.setEnabled(False)
                
                # 更新数据保存状态
                if hasattr(self, 'data_saver'):
                    self.update_data_save_status("未开始检测", False)
                
                # 释放视频源
                if self.video_capture:
                    self.video_capture.release()
                    self.video_capture = None
                
                # 清空显示
                self.video_display.clear()
                self.video_display.setText("请加载视频文件或开启摄像头")
        
        self.statusBar().showMessage("已停止处理")
    
    def update_processor_settings(self):
        """更新处理器设置"""
        if self.video_processor:
            settings = {
                'tracking_enabled': self.control_panel.tracking_enabled.isChecked(),
                'show_ids': self.control_panel.show_ids.isChecked(),
                'crossing_enabled': self.control_panel.crossing_enabled.isChecked(),
                'area_enabled': self.control_panel.area_enabled.isChecked()
            }
            self.video_processor.update_settings(settings)
    
    @pyqtSlot(np.ndarray)
    def update_display_safe(self, frame):
        """线程安全的显示更新"""
        if frame is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # 设置视频帧信息用于坐标转换
                self.video_display.set_video_frame_info(w, h)
                
                # 创建QPixmap并进行缩放，保持宽高比
                pixmap = QPixmap.fromImage(qt_image)
                
                # 获取当前显示区域大小
                display_size = self.video_display.size()
                
                # 缩放图像以适应显示区域，保持宽高比
                scaled_pixmap = pixmap.scaled(
                    display_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                # 设置到显示控件
                self.video_display.setPixmap(scaled_pixmap)
                
            except Exception as e:
                print(f"显示更新错误: {e}")
    
    @pyqtSlot(str)
    def update_statistics_safe(self, stats_text):
        """线程安全的统计更新"""
        try:
            self.statistics_widget.update_statistics(stats_text)
        except Exception as e:
            print(f"统计更新错误: {e}")
    
    @pyqtSlot()
    def on_processing_finished(self):
        """处理完成回调"""
        self.is_processing = False
        
        # 更新按钮状态
        if self.video_capture and self.video_capture.isOpened():
            self.control_panel.start_btn.setEnabled(True)
        else:
            self.control_panel.start_btn.setEnabled(False)
        
        self.statusBar().showMessage("视频处理完成")
    
    def update_confidence(self, value):
        """更新置信度阈值"""
        confidence = value / 100.0
        self.control_panel.conf_label.setText(f"{confidence:.2f}")
        self.detector.set_parameters(confidence=confidence)
    
    def update_iou(self, value):
        """更新IoU阈值"""
        iou = value / 100.0
        self.control_panel.iou_label.setText(f"{iou:.2f}")
        self.detector.set_parameters(iou_threshold=iou)
    
    def change_model(self, model_name):
        """更换模型"""
        self.detector.model_path = model_name
        if self.detector.load_model():
            self.statusBar().showMessage(f"已切换到模型: {model_name}")
        else:
            QMessageBox.warning(self, "错误", f"无法加载模型: {model_name}")
    
    def change_device(self, device):
        """更换设备"""
        # 转换中文设备名称到英文
        device_map = {
            'auto': 'auto',
            'cpu': 'cpu', 
            'cuda': 'cuda'
        }
        
        device_key = device_map.get(device, device)
        
        if self.detector.set_device(device_key):
            device_info = self.detector.get_device_info()
            self.statusBar().showMessage(f"设备已切换到: {device_info['current_device']}")
        else:
            QMessageBox.warning(self, "错误", f"无法切换到设备: {device}")
    
    def add_counting_line(self):
        """添加计数线"""
        # 启用线条绘制模式
        self.video_display.enable_line_drawing(True)
        self.statusBar().showMessage("请在视频画面上点击两个点来绘制计数线")
        QMessageBox.information(self, "提示", "请在视频画面上点击两个点来绘制计数线")
    
    @pyqtSlot(tuple, tuple)
    def on_line_drawn(self, start_point, end_point):
        """处理绘制的线条"""
        try:
            # 计算线条ID
            line_id = f"line_{len(self.line_counter.counting_lines) + 1}"
            
            # 添加计数线
            success = self.line_counter.add_line(line_id, start_point, end_point)
            
            if success:
                self.statusBar().showMessage(f"已添加计数线: {line_id}")
                QMessageBox.information(self, "成功", f"已添加计数线: {line_id}")
                
                # 更新处理器设置
                self.update_processor_settings()
            else:
                self.statusBar().showMessage("添加计数线失败")
                QMessageBox.warning(self, "错误", "添加计数线失败")
                
        except Exception as e:
            error_msg = f"添加计数线时出错: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            print(f"Line drawing error: {e}")
    
    def clear_counting_lines(self):
        """清除所有计数线"""
        try:
            self.line_counter.counting_lines.clear()
            self.statusBar().showMessage("已清除所有计数线")
            QMessageBox.information(self, "成功", "已清除所有计数线")
        except Exception as e:
            error_msg = f"清除计数线时出错: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.warning(self, "错误", error_msg)
    
    def reset_counts(self):
        """重置计数"""
        try:
            self.line_counter.reset()
            self.statusBar().showMessage("已重置所有计数")
            QMessageBox.information(self, "成功", "已重置所有计数")
        except Exception as e:
            error_msg = f"重置计数时出错: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.warning(self, "错误", error_msg)
    
    def add_rectangle_counting_area(self):
        """添加计数区域（画框模式）"""
        # 启用矩形绘制模式
        self.video_display.enable_rectangle_drawing(True)
        self.statusBar().showMessage("请在视频画面上拖拽绘制矩形计数区域")
        QMessageBox.information(self, "提示", "请在视频画面上拖拽绘制矩形计数区域")
    
    @pyqtSlot(list)
    def on_area_drawn(self, points):
        """处理绘制的区域"""
        try:
            # 计算区域 ID
            area_id = f"area_{len(self.area_counter.counting_areas) + 1}"
            
            # 获取计数模式
            mode_map = {
                '进入计数': 'enter',
                '离开计数': 'exit',
                '双向计数': 'both'
            }
            mode_text = self.control_panel.area_mode_combo.currentText()
            count_mode = mode_map.get(mode_text, 'enter')
            
            # 添加计数区域
            success = self.area_counter.add_area(area_id, points, count_mode)
            
            if success:
                self.statusBar().showMessage(f"已添加计数区域: {area_id}")
                QMessageBox.information(self, "成功", f"已添加计数区域: {area_id}")
                
                # 更新处理器设置
                self.update_processor_settings()
            else:
                self.statusBar().showMessage("添加计数区域失败")
                QMessageBox.warning(self, "错误", "添加计数区域失败")
                
        except Exception as e:
            error_msg = f"添加计数区域时出错: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            print(f"Area drawing error: {e}")
    
    def clear_counting_areas(self):
        """清除所有计数区域"""
        try:
            self.area_counter.counting_areas.clear()
            
            # 强制更新处理器设置以反映清除操作
            self.update_processor_settings()
            
            self.statusBar().showMessage("已清除所有计数区域")
            QMessageBox.information(self, "成功", "已清除所有计数区域")
        except Exception as e:
            error_msg = f"清除计数区域时出错: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.warning(self, "错误", error_msg)
    
    def reset_area_counts(self):
        """重置区域计数"""
        try:
            self.area_counter.reset()
            self.statusBar().showMessage("已重置所有区域计数")
            QMessageBox.information(self, "成功", "已重置所有区域计数")
        except Exception as e:
            error_msg = f"重置区域计数时出错: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.warning(self, "错误", error_msg)
    
    def export_data(self):
        """导出数据"""
        format_type = self.control_panel.export_format.currentText().lower()
        
        try:
            # 获取当前数据
            tracking_results = self.tracker.get_statistics() if self.tracker else {}
            crossing_results = self.line_counter.get_line_statistics() if self.line_counter else {}
            area_results = self.area_counter.get_area_statistics() if self.area_counter else {}
            
            # 合并所有结果
            combined_results = {
                'tracking': tracking_results,
                'line_crossing': crossing_results,
                'area_counting': area_results
            }
            
            # 导出综合报告
            filepath = self.exporter.export_comprehensive_report(
                [], tracking_results, combined_results, format_type
            )
            
            if filepath:
                QMessageBox.information(self, "导出成功", f"数据已导出到: {filepath}")
            else:
                QMessageBox.warning(self, "导出失败", "数据导出失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出过程中发生错误: {str(e)}")
    
    def view_saved_files(self):
        """查看保存的文件"""
        if self.data_saver:
            saved_files = self.data_saver.get_saved_files()
            if saved_files:
                file_list = "\n".join([os.path.basename(f) for f in saved_files])
                QMessageBox.information(self, "已保存的文件", f"保存的数据文件:\n\n{file_list}")
                
                # 询问是否打开保存目录
                reply = QMessageBox.question(self, "打开目录", "是否打开保存目录?", 
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    try:
                        
                        save_dir = self.data_saver.output_dir
                        
                        # 确保目录存在
                        if os.path.exists(save_dir):
                            if platform.system() == "Windows":
                                # 使用 Popen 非阻塞调用
                                subprocess.Popen(f'explorer "{save_dir}"', shell=True)
                            elif platform.system() == "Darwin":  # macOS
                                subprocess.Popen(['open', save_dir])
                            else:  # Linux
                                subprocess.Popen(['xdg-open', save_dir])
                        else:
                            QMessageBox.warning(self, "错误", f"保存目录不存在: {save_dir}")
                    except Exception as e:
                        QMessageBox.warning(self, "错误", f"无法打开保存目录: {str(e)}")
            else:
                QMessageBox.information(self, "提示", "当前没有保存的数据文件")
        else:
            QMessageBox.warning(self, "错误", "数据保存器未初始化")
    
    def update_data_save_status(self, status_text: str, enable_view_btn: bool = True):
        """更新数据保存状态显示"""
        if hasattr(self.control_panel, 'save_status'):
            self.control_panel.save_status.setText(status_text)
            if hasattr(self.control_panel, 'view_files_btn'):
                self.control_panel.view_files_btn.setEnabled(enable_view_btn)
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                         "YOLOv8 交通检测系统\n\n"
                         "基于PyQt5和YOLOv8的智能交通分析系统\n"
                         "支持目标检测、多目标追踪、越线计数等功能\n\n"
                         "版本: 1.0.0")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_processing()
        
        # 确保线程正常停止
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.quit()
            self.video_thread.wait(3000)  # 等待3秒
            
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("YOLOv8交通检测系统")
    app.setApplicationVersion("1.0.0")
    
    # 设置中文字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()