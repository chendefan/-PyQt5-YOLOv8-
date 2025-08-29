
# YOLOv8 智能交通分析系统

基于PyQt5和YOLOv8的智能交通分析系统，支持实时目标检测、多目标追踪、越线计数、区域计数等功能。

## ✨ 主要功能

- **🚗 交通目标检测**: 基于YOLOv8的高精度车辆、行人等目标检测
- **🔄 多目标追踪**: 实时追踪多个移动目标
- **📏 越线计数**: 自定义计数线，统计车辆通过情况
- **🏠 区域计数**: 自定义计数区域，统计经过该区域的目标数量，支持点击和拖拽两种绘制模式
- **💾 自动数据保存**: 实时自动保存检测、追踪、越线和区域计数数据
- **⚙️ 设备选择**: 支持CPU/GPU设备选择，优化性能
- **🖥️ 中文界面**: 完全中文化的用户界面

## 🛠️ 技术栈

- **前端界面**: PyQt5
- **计算机视觉**: OpenCV, YOLOv8 (Ultralytics)
- **深度学习**: PyTorch
- **数据处理**: Pandas, NumPy
- **数据可视化**: Matplotlib, Seaborn
- **数据导出**: openpyxl, xlsxwriter

## 📋 系统要求

- Python 3.9+
- Windows 10/11 (推荐)
- 4GB+ RAM (推荐8GB+)
- 支持CUDA的显卡 (可选，用于GPU加速)

## 🚀 快速开始

### 1. 环境配置

**推荐方式A: 使用Conda**
```bash
# 创建新的conda环境
conda create -n yolov8-traffic python=3.9
conda activate yolov8-traffic

# 安装依赖包
pip install -r requirements.txt
```

**方式B: 使用系统Python**
```bash
# 确认pip可用
python -m pip --version

# 安装依赖包
python -m pip install -r requirements.txt
```

### 2. 获取项目文件

```bash
# 如果使用git
git clone <repository-url>
cd yolov8-traffic-analysis

# 或者下载并解压项目文件
```

### 3. 安装依赖包

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 4. 运行系统

```bash
# 方式1: 使用启动脚本 (推荐)
python run.py

# 方式2: 直接运行主程序
python main.py
```


## 📖 使用指南

### 主界面介绍

系统采用左右分栏布局：

- **左侧**: 视频显示区域和统计信息
- **右侧**: 控制面板，包含各项功能设置

### 基本操作流程

1. **加载视频源**
   - 点击"加载视频文件"选择本地视频
   - 或点击"开启摄像头"使用实时摄像头

2. **调整检测参数**
   - 选择设备 (CPU/GPU/自动)
   - 选择YOLOv8模型 (n/s/m/l/x)
   - 调整置信度阈值 (0.1-0.95)
   - 调整IoU阈值 (0.1-0.9)

3. **配置追踪设置**
   - 启用/禁用多目标追踪
   - 选择是否显示ID

4. **设置越线检测**
   - 启用越线检测功能
   - 在视频画面上点击两点绘制计数线
   - 查看实时计数统计

5. **设置区域计数** 
   - 启用区域计数功能
   - 选择计数模式（进入/离开/双向）
   - 选择绘制方式：
     - **画框模式**: 在视频画面上拖拽绘制矩形区域
   - 支持清除所有区域和禁用显示
   - 查看区域计数统计

6. **自动数据保存** 
   - 检测开始时自动创建数据文件
   - 实时保存检测、追踪、越线和区域计数数据
   - 数据保存在 `saved_data` 目录下
   - 文件名包含时间戳和中文标识


#### 区域控制功能

- **启用/禁用**: 取消勾选"启用区域计数"时，区域立即从视频中消失
- **清除区域**: 点击"清除所有区域"立即移除所有绘制的区域
- **实时显示**: 所有操作立即生效，无需重启检测

### 模型选择建议

| 模型 | 速度 | 精度 | 文件大小 | 适用场景 |
|------|------|------|---------|----------|
| YOLOv8n | 最快 | 较低 | ~6MB | 实时检测、资源受限 |
| YOLOv8s | 快 | 平衡 | ~22MB | 一般应用推荐 |
| YOLOv8m | 中等 | 较高 | ~52MB | 精度要求较高 |
| YOLOv8l | 较慢 | 高 | ~88MB | 高精度需求 |
| YOLOv8x | 最慢 | 最高 | ~136MB | 最高精度要求 |

### 设备选择建议

| 设备类型 | 适用场景 | 性能表现 |
|----------|----------|-----------|
| **自动** | 推荐选项，自动检测GPU可用性 | 最优选择 |
| **CUDA (GPU)** | 有NVIDIA显卡环境 | 5-20x 加速 |
| **CPU** | 无GPU或调试环境 | 基本性能 |

**注意**: 首次使用GPU时可能需要一些初始化时间

## 📁 项目结构

```
yolov8-traffic-analysis/
├── main.py                 # PyQt5主应用程序
├── run.py                  # 启动脚本
├── config.py               # 系统配置文件
├── requirements.txt        # 依赖清单
├── utils/                 # 工具模块目录
│   ├── __init__.py        # 模块初始化
│   ├── detection.py       # YOLOv8检测模块
│   ├── tracking.py        # 多目标追踪模块
│   ├── line_crossing.py   # 越线检测模块
│   ├── area_counting.py   # 区域计数模块 
│   └── data_saver.py      # 自动数据保存模块 
├── saved_data/            # 自动保存的数据文件目录 
├── outputs/               # 输出文件目录
└── data/                  # 数据文件目录
```

## ⚙️ 配置说明

### 模型配置

系统支持多种YOLOv8模型，首次使用时会自动下载：

- `yolov8n.pt` - Nano版本 (~6MB)
- [yolov8s.pt](file://d:\文件保存\就业\qoder\yolov8s.pt) - Small版本 (~22MB) 
- `yolov8m.pt` - Medium版本 (~52MB)
- `yolov8l.pt` - Large版本 (~88MB)
- `yolov8x.pt` - Extra Large版本 (~136MB)

### 检测类别

系统基于COCO数据集，支持80个类别的目标检测，重点关注交通相关目标：

- 🚶‍♂️ 行人 (person)
- 🚲 自行车 (bicycle)  
- 🚗 汽车 (car)
- 🏍️ 摩托车 (motorcycle)
- 🚌 公交车 (bus)
- 🚛 卡车 (truck)
- 🚥 交通灯 (traffic light)
- 🛑 停车标志 (stop sign)

### 数据保存配置 ⭐ 新功能

#### 自动保存文件

系统会在检测开始时自动创建以下数据文件：

- `检测数据_YYYYMMDD_HHMMSS.csv` - 目标检测数据
- `追踪数据_YYYYMMDD_HHMMSS.csv` - 目标追踪数据  
- `越线数据_YYYYMMDD_HHMMSS.csv` - 越线计数数据
- `区域计数数据_YYYYMMDD_HHMMSS.csv` - 区域计数数据

#### 数据格式

所有数据文件采用CSV格式，包含时间戳、目标ID、坐标、类别等详细信息，便于后续分析和处理。


## 🤝 开发指南

### 添加新功能

1. 在对应的utils模块中实现功能
2. 在main.py中集成UI控件
3. 更新config.py配置
4. 添加相应测试

### 自定义检测类别

修改 [config.py](file://d:\文件保存\就业\qoder\config.py) 中的类别配置：

```python
CUSTOM_CLASSES = {
    80: {'en': 'custom_object', 'zh': '自定义目标', 'color': (255, 0, 0)}
}
```

### 扩展数据保存功能

修改 `utils/data_saver.py` 中的 [AutoDataSaver](file://d:\文件保存\就业\qoder\utils\data_saver.py#L12-L244) 类来自定义数据保存格式和内容。


## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 强大的目标检测框架
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - 跨平台GUI工具包
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyTorch](https://pytorch.org/) - 深度学习框架


**注意**: 
- 首次运行时系统会自动下载YOLOv8模型文件，请确保网络连接正常
- 系统会自动创建必要的目录结构（如 `saved_data` 等）
```

