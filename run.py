#!/usr/bin/env python3
"""
YOLOv8 交通检测系统启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'PyQt5': 'PyQt5',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少以下依赖包，请运行以下命令安装：")
        print(f"\n**Option 1: Using pip directly**")
        print(f"pip install {' '.join(missing_packages)}")
        print(f"\n**Option 2: Using python -m pip**")
        print(f"python -m pip install {' '.join(missing_packages)}")
        print(f"\n**Option 3: Using conda (if available)**")
        print(f"conda install {' '.join(missing_packages)}")
        print(f"\n**Option 4: Install from requirements file**")
        print(f"pip install -r requirements.txt")
        print(f"# or")
        print(f"python -m pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    print("YOLOv8 交通检测系统启动中...")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺少的依赖包后再运行系统")
        return
    
    print("\n所有依赖已满足，启动系统...")
    print("=" * 50)
    
    try:
        # 导入并启动主应用
        from main import main as run_main
        run_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有必要的文件都在当前目录中")
        
    except Exception as e:
        print(f"启动过程中发生错误: {e}")
        print("请检查系统配置和依赖安装")

if __name__ == "__main__":
    main()