"""
交通数据分析模块
提供数据统计、分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class TrafficAnalyzer:
    """交通数据分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.detection_data = []
        self.tracking_data = []
        self.crossing_data = []
        
        # 分析结果缓存
        self.analysis_cache = {}
        self.last_analysis_time = 0
        
        # 统计窗口设置
        self.time_windows = {
            '1分钟': 60,
            '5分钟': 300,
            '15分钟': 900,
            '1小时': 3600
        }
    
    def add_detection_data(self, detections: List[Dict], timestamp: float = None):
        """
        添加检测数据
        
        Args:
            detections: 检测结果列表
            timestamp: 时间戳，默认使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        for detection in detections:
            data_entry = {
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp),
                'class_id': detection['class_id'],
                'class_name': detection['class_name_zh'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'is_traffic': detection['is_traffic']
            }
            self.detection_data.append(data_entry)
        
        # 清理过期数据（保留最近24小时）
        self._cleanup_old_data()
    
    def add_tracking_data(self, tracking_results: Dict, timestamp: float = None):
        """
        添加追踪数据
        
        Args:
            tracking_results: 追踪结果
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        data_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp),
            'active_objects': tracking_results.get('active_count', 0),
            'total_tracked': tracking_results.get('total_tracked', 0),
            'class_counts': tracking_results.get('class_counts', {}),
            'tracked_objects': tracking_results.get('tracked_objects', [])
        }
        self.tracking_data.append(data_entry)
    
    def add_crossing_data(self, crossing_events: List[Dict]):
        """
        添加越线数据
        
        Args:
            crossing_events: 越线事件列表
        """
        for event in crossing_events:
            data_entry = {
                'timestamp': event['timestamp'],
                'datetime': datetime.fromtimestamp(event['timestamp']),
                'line_id': event['line_id'],
                'object_id': event['object_id'],
                'class_name': event['class_name'],
                'direction': event['direction'],
                'crossing_point': event['crossing_point']
            }
            self.crossing_data.append(data_entry)
    
    def _cleanup_old_data(self, hours: int = 24):
        """
        清理过期数据
        
        Args:
            hours: 保留时间（小时）
        """
        cutoff_time = time.time() - (hours * 3600)
        
        self.detection_data = [d for d in self.detection_data if d['timestamp'] > cutoff_time]
        self.tracking_data = [d for d in self.tracking_data if d['timestamp'] > cutoff_time]
        self.crossing_data = [d for d in self.crossing_data if d['timestamp'] > cutoff_time]
    
    def get_detection_statistics(self, time_window: str = '1小时') -> Dict:
        """
        获取检测统计信息
        
        Args:
            time_window: 时间窗口
            
        Returns:
            Dict: 统计结果
        """
        if not self.detection_data:
            return self._empty_statistics()
        
        # 时间过滤
        window_seconds = self.time_windows.get(time_window, 3600)
        cutoff_time = time.time() - window_seconds
        
        filtered_data = [d for d in self.detection_data if d['timestamp'] > cutoff_time]
        
        if not filtered_data:
            return self._empty_statistics()
        
        # 基础统计
        total_detections = len(filtered_data)
        
        # 按类别统计
        class_counts = Counter(d['class_name'] for d in filtered_data)
        
        # 置信度统计
        confidences = [d['confidence'] for d in filtered_data]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # 交通相关目标统计
        traffic_detections = [d for d in filtered_data if d['is_traffic']]
        traffic_ratio = len(traffic_detections) / total_detections if total_detections > 0 else 0
        
        # 时间分布统计
        time_distribution = self._calculate_time_distribution(filtered_data)
        
        return {
            'time_window': time_window,
            'total_detections': total_detections,
            'class_counts': dict(class_counts),
            'average_confidence': avg_confidence,
            'traffic_objects': len(traffic_detections),
            'traffic_ratio': traffic_ratio,
            'time_distribution': time_distribution,
            'detection_rate': total_detections / (window_seconds / 60)  # 每分钟检测数
        }
    
    def get_tracking_statistics(self, time_window: str = '1小时') -> Dict:
        """
        获取追踪统计信息
        
        Args:
            time_window: 时间窗口
            
        Returns:
            Dict: 统计结果
        """
        if not self.tracking_data:
            return self._empty_statistics()
        
        window_seconds = self.time_windows.get(time_window, 3600)
        cutoff_time = time.time() - window_seconds
        
        filtered_data = [d for d in self.tracking_data if d['timestamp'] > cutoff_time]
        
        if not filtered_data:
            return self._empty_statistics()
        
        # 当前活跃对象数
        current_active = filtered_data[-1]['active_objects'] if filtered_data else 0
        
        # 平均活跃对象数
        avg_active = np.mean([d['active_objects'] for d in filtered_data])
        
        # 最大活跃对象数
        max_active = max([d['active_objects'] for d in filtered_data])
        
        # 总追踪对象数
        total_tracked = filtered_data[-1]['total_tracked'] if filtered_data else 0
        
        return {
            'time_window': time_window,
            'current_active_objects': current_active,
            'average_active_objects': avg_active,
            'max_active_objects': max_active,
            'total_tracked_objects': total_tracked,
            'tracking_efficiency': avg_active / max_active if max_active > 0 else 0
        }
    
    def get_crossing_statistics(self, time_window: str = '1小时') -> Dict:
        """
        获取越线统计信息
        
        Args:
            time_window: 时间窗口
            
        Returns:
            Dict: 统计结果
        """
        if not self.crossing_data:
            return self._empty_statistics()
        
        window_seconds = self.time_windows.get(time_window, 3600)
        cutoff_time = time.time() - window_seconds
        
        filtered_data = [d for d in self.crossing_data if d['timestamp'] > cutoff_time]
        
        if not filtered_data:
            return self._empty_statistics()
        
        # 基础统计
        total_crossings = len(filtered_data)
        
        # 按线条统计
        line_counts = Counter(d['line_id'] for d in filtered_data)
        
        # 按方向统计
        direction_counts = Counter(d['direction'] for d in filtered_data)
        
        # 按类别统计
        class_counts = Counter(d['class_name'] for d in filtered_data)
        
        # 越线率
        crossing_rate = total_crossings / (window_seconds / 60)  # 每分钟越线数
        
        return {
            'time_window': time_window,
            'total_crossings': total_crossings,
            'line_counts': dict(line_counts),
            'direction_counts': dict(direction_counts),
            'class_counts': dict(class_counts),
            'crossing_rate': crossing_rate
        }
    
    def _calculate_time_distribution(self, data: List[Dict]) -> Dict:
        """
        计算时间分布
        
        Args:
            data: 数据列表
            
        Returns:
            Dict: 时间分布统计
        """
        if not data:
            return {}
        
        # 按小时分组
        hour_counts = defaultdict(int)
        for item in data:
            hour = item['datetime'].hour
            hour_counts[hour] += 1
        
        return dict(hour_counts)
    
    def generate_traffic_report(self, time_window: str = '1小时') -> Dict:
        """
        生成综合交通报告
        
        Args:
            time_window: 时间窗口
            
        Returns:
            Dict: 综合报告
        """
        detection_stats = self.get_detection_statistics(time_window)
        tracking_stats = self.get_tracking_statistics(time_window)
        crossing_stats = self.get_crossing_statistics(time_window)
        
        # 计算交通流量趋势
        traffic_trend = self._calculate_traffic_trend(time_window)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            detection_stats, tracking_stats, crossing_stats
        )
        
        report = {
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'time_window': time_window,
            'detection_statistics': detection_stats,
            'tracking_statistics': tracking_stats,
            'crossing_statistics': crossing_stats,
            'traffic_trend': traffic_trend,
            'recommendations': recommendations
        }
        
        return report
    
    def _calculate_traffic_trend(self, time_window: str) -> Dict:
        """
        计算交通流量趋势
        
        Args:
            time_window: 时间窗口
            
        Returns:
            Dict: 趋势分析结果
        """
        if not self.crossing_data:
            return {'trend': 'no_data', 'change_rate': 0}
        
        window_seconds = self.time_windows.get(time_window, 3600)
        current_time = time.time()
        
        # 当前时间窗口的数据
        current_data = [
            d for d in self.crossing_data 
            if current_time - window_seconds < d['timestamp'] <= current_time
        ]
        
        # 前一个时间窗口的数据
        previous_data = [
            d for d in self.crossing_data
            if current_time - 2 * window_seconds < d['timestamp'] <= current_time - window_seconds
        ]
        
        current_count = len(current_data)
        previous_count = len(previous_data)
        
        if previous_count == 0:
            return {'trend': 'insufficient_data', 'change_rate': 0}
        
        change_rate = (current_count - previous_count) / previous_count * 100
        
        if change_rate > 10:
            trend = 'increasing'
        elif change_rate < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_rate': change_rate,
            'current_count': current_count,
            'previous_count': previous_count
        }
    
    def _generate_recommendations(self, detection_stats: Dict, 
                                tracking_stats: Dict, crossing_stats: Dict) -> List[str]:
        """
        生成建议
        
        Args:
            detection_stats: 检测统计
            tracking_stats: 追踪统计
            crossing_stats: 越线统计
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 检测相关建议
        if detection_stats.get('average_confidence', 0) < 0.7:
            recommendations.append("检测置信度较低，建议调整模型参数或改善图像质量")
        
        if detection_stats.get('traffic_ratio', 0) < 0.3:
            recommendations.append("交通目标占比较低，可能需要调整摄像头角度")
        
        # 追踪相关建议
        tracking_efficiency = tracking_stats.get('tracking_efficiency', 0)
        if tracking_efficiency < 0.7:
            recommendations.append("追踪效率偏低，建议优化追踪参数")
        
        # 越线相关建议
        crossing_rate = crossing_stats.get('crossing_rate', 0)
        if crossing_rate > 100:  # 每分钟超过100次越线
            recommendations.append("越线频率过高，可能存在误检，建议调整计数线位置")
        elif crossing_rate < 1:
            recommendations.append("越线频率较低，建议检查计数线设置是否合理")
        
        if not recommendations:
            recommendations.append("系统运行正常，各项指标良好")
        
        return recommendations
    
    def _empty_statistics(self) -> Dict:
        """返回空统计结果"""
        return {
            'total_detections': 0,
            'class_counts': {},
            'average_confidence': 0,
            'traffic_objects': 0,
            'traffic_ratio': 0,
            'time_distribution': {},
            'detection_rate': 0
        }
    
    def create_visualization(self, stat_type: str = 'detection', 
                           time_window: str = '1小时', 
                           save_path: str = None) -> str:
        """
        创建可视化图表
        
        Args:
            stat_type: 统计类型 ('detection', 'tracking', 'crossing')
            time_window: 时间窗口
            save_path: 保存路径
            
        Returns:
            str: 图表文件路径
        """
        plt.figure(figsize=(12, 8))
        
        if stat_type == 'detection':
            self._plot_detection_stats(time_window)
        elif stat_type == 'tracking':
            self._plot_tracking_stats(time_window)
        elif stat_type == 'crossing':
            self._plot_crossing_stats(time_window)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"traffic_analysis_{stat_type}_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_detection_stats(self, time_window: str):
        """绘制检测统计图表"""
        stats = self.get_detection_statistics(time_window)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 类别分布饼图
        if stats['class_counts']:
            ax1.pie(stats['class_counts'].values(), labels=stats['class_counts'].keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title(f'目标类别分布 ({time_window})')
        
        # 时间分布柱状图
        if stats['time_distribution']:
            hours = list(stats['time_distribution'].keys())
            counts = list(stats['time_distribution'].values())
            ax2.bar(hours, counts)
            ax2.set_title('小时分布')
            ax2.set_xlabel('小时')
            ax2.set_ylabel('检测数量')
        
        # 检测率趋势
        ax3.text(0.5, 0.5, f"检测率: {stats['detection_rate']:.1f}/分钟\n"
                           f"平均置信度: {stats['average_confidence']:.3f}\n"
                           f"交通目标比例: {stats['traffic_ratio']:.2%}",
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('关键指标')
        ax3.axis('off')
        
        # 总体统计
        ax4.text(0.5, 0.5, f"总检测数: {stats['total_detections']}\n"
                           f"交通目标: {stats['traffic_objects']}\n"
                           f"时间窗口: {time_window}",
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('总体统计')
        ax4.axis('off')
        
        plt.suptitle('交通检测统计分析', fontsize=16, fontweight='bold')
    
    def _plot_tracking_stats(self, time_window: str):
        """绘制追踪统计图表"""
        stats = self.get_tracking_statistics(time_window)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 活跃对象数量趋势
        if self.tracking_data:
            timestamps = [d['datetime'] for d in self.tracking_data[-50:]]  # 最近50个点
            active_counts = [d['active_objects'] for d in self.tracking_data[-50:]]
            
            ax1.plot(timestamps, active_counts, marker='o')
            ax1.set_title('活跃对象数量趋势')
            ax1.set_ylabel('活跃对象数')
            ax1.tick_params(axis='x', rotation=45)
        
        # 追踪效率
        efficiency = stats.get('tracking_efficiency', 0)
        ax2.pie([efficiency, 1-efficiency], labels=['有效追踪', '追踪损失'], 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('追踪效率')
        
        # 关键指标
        ax3.text(0.5, 0.5, f"当前活跃: {stats['current_active_objects']}\n"
                           f"平均活跃: {stats['average_active_objects']:.1f}\n"
                           f"最大活跃: {stats['max_active_objects']}\n"
                           f"追踪效率: {efficiency:.2%}",
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('关键指标')
        ax3.axis('off')
        
        # 总体统计
        ax4.text(0.5, 0.5, f"总追踪对象: {stats['total_tracked_objects']}\n"
                           f"时间窗口: {time_window}",
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('总体统计')
        ax4.axis('off')
        
        plt.suptitle('目标追踪统计分析', fontsize=16, fontweight='bold')
    
    def _plot_crossing_stats(self, time_window: str):
        """绘制越线统计图表"""
        stats = self.get_crossing_statistics(time_window)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 线条越线分布
        if stats['line_counts']:
            ax1.bar(stats['line_counts'].keys(), stats['line_counts'].values())
            ax1.set_title('各线条越线统计')
            ax1.set_ylabel('越线次数')
        
        # 方向分布
        if stats['direction_counts']:
            ax2.pie(stats['direction_counts'].values(), labels=stats['direction_counts'].keys(),
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('越线方向分布')
        
        # 类别分布
        if stats['class_counts']:
            ax3.barh(list(stats['class_counts'].keys()), list(stats['class_counts'].values()))
            ax3.set_title('越线目标类别分布')
            ax3.set_xlabel('越线次数')
        
        # 越线率
        ax4.text(0.5, 0.5, f"总越线次数: {stats['total_crossings']}\n"
                           f"越线率: {stats['crossing_rate']:.1f}/分钟\n"
                           f"时间窗口: {time_window}",
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('越线统计')
        ax4.axis('off')
        
        plt.suptitle('越线检测统计分析', fontsize=16, fontweight='bold')
    
    def export_analysis_data(self, filename: str = None) -> str:
        """
        导出分析数据
        
        Args:
            filename: 文件名
            
        Returns:
            str: 导出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_analysis_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 检测数据
            if self.detection_data:
                df_detection = pd.DataFrame(self.detection_data)
                df_detection.to_excel(writer, sheet_name='检测数据', index=False)
            
            # 追踪数据
            if self.tracking_data:
                df_tracking = pd.DataFrame(self.tracking_data)
                df_tracking.to_excel(writer, sheet_name='追踪数据', index=False)
            
            # 越线数据
            if self.crossing_data:
                df_crossing = pd.DataFrame(self.crossing_data)
                df_crossing.to_excel(writer, sheet_name='越线数据', index=False)
            
            # 统计报告
            report = self.generate_traffic_report()
            df_report = pd.DataFrame([report])
            df_report.to_excel(writer, sheet_name='统计报告', index=False)
        
        return filename