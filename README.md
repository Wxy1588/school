# 智能书架整理系统

## 项目简介

本项目旨在实现一个基于计算机视觉的智能书架整理系统，能够自动识别书架上的书籍，并按照颜色、尺寸或书名进行分类整理建议。

**用户场景**：图书馆、家庭书房等场景中，书架上的书籍摆放杂乱，用户希望能够通过摄像头拍摄书架照片，自动识别书本并按照一定规则进行分类整理。

**核心功能**：
- 📚 书本检测：自动识别书架上的书籍位置
- 🎨 颜色分类：基于K-means聚类的颜色识别
- 📏 尺寸分类：按书本大小进行分类
- 🏷️ 书名识别：使用OCR技术识别书名
- 🔖 智能分类：基于大模型的书名分类
- 📊 可视化输出：检测结果可视化展示

---

## 技术方案

### 方案A：传统CV方法（已完成）

```
图像采集 → OpenCV预处理 → Canny边缘检测 → 轮廓提取 → 书本检测 → 颜色/尺寸分类 → 可视化
```

- **预处理**：cv2.cvtColor()、cv2.GaussianBlur()、cv2.Canny()
- **检测**：基于轮廓过滤的书脊检测
- **分类**：K-means颜色聚类 + 尺寸分类
- **优点**：依赖少，运行快，无需训练
- **缺点**：密集书架场景效果一般

### 方案B：深度学习方法（待完善）

```
图像采集 → OpenCV预处理 → YOLOv8书脊检测 → EasyOCR文字识别 → 大模型分类 → 可视化
```

- **预处理**：OpenCV图像增强
- **检测**：YOLOv8n-OBB深度学习检测
- **OCR**：EasyOCR中文识别
- **分类**：基于关键词/大模型的智能分类
- **优点**：检测准确率高，支持旋转框
- **缺点**：需要GPU训练模型

---

## 项目结构

```
shujia/
├── shelf_organizer/src/          # 源代码
│   ├── __init__.py
│   ├── image_processor.py        # OpenCV图像预处理
│   ├── book_detector.py          # 传统CV检测（方案A）
│   ├── book_detector_yolo.py     # YOLOv8检测（方案B）
│   ├── book_ocr.py               # Tesseract OCR
│   ├── book_ocr_easyocr.py       # EasyOCR（方案B）
│   ├── book_classifier.py        # 颜色/尺寸分类
│   ├── book_classifier_llm.py    # 大模型分类（智谱AI）
│   ├── sorter.py                 # 排序逻辑
│   ├── visualizer.py             # 可视化模块
│   ├── main.py                   # 方案A主程序
│   └── main_integrated.py        # 整合主程序
├── Dataset_1/                    # 测试数据集（425张）
├── 书脊检测yolo数据集(3879张).v1i.yolov8/  # YOLO训练数据集
├── models/                       # 模型文件目录
│   └── easyocr/                  # EasyOCR模型
├── dataset_output/               # 输出结果目录
├── requirements.txt              # 依赖列表
├── run_integrated.py             # 快速测试脚本
├── README.md                     # 项目说明
└── 实践报告.md                   # 实践报告
```

---

## 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n bookshelf python=3.9
conda activate bookshelf

# 安装核心依赖
pip install opencv-python numpy scikit-learn Pillow zhipuai

# 安装Tesseract OCR（可选）
# Windows: 下载安装tesseract.exe并添加到PATH

# 安装YOLOv8（可选，方案B）
pip install ultralytics

# 安装EasyOCR（可选，方案B）
pip install easyocr
```

---

## 使用方法

### 快速测试（使用传统CV方法）

```bash
python run_integrated.py
```

### 使用整合系统

```python
from shelf_organizer.src.main_integrated import ShelfOrganizerIntegrated

# 创建整理器实例
organizer = ShelfOrganizerIntegrated(
    detector_type='traditional',  # 'traditional' 或 'yolo'
    ocr_type='tesseract',         # 'tesseract' 或 'easyocr'
    classifier_type='color'       # 'color' 或 'llm'
)

# 处理图像
result = organizer.process_image('Dataset_1/00002.jpg', extract_titles=True)

# 输出结果
print(f"检测到 {result['num_books']} 本书")
for book in result['books']:
    print(f"- {book.get('title', '未知书名')}: {book.get('category', '未分类')}")
```

---

## 技术栈

| 模块 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.9+ |
| 图像处理 | OpenCV | 4.8+ |
| 数值计算 | NumPy | 1.24+ |
| 聚类算法 | scikit-learn | 1.3+ |
| 深度学习 | Ultralytics YOLOv8 | 8.0+ |
| OCR | EasyOCR / Tesseract | - |
| 大模型 | 智谱AI GLM | - |

---

## 核心OpenCV函数使用清单

| 函数 | 用途 |
|------|------|
| cv2.imread() | 读取图像 |
| cv2.cvtColor() | 灰度化、颜色空间转换 |
| cv2.GaussianBlur() | 高斯模糊去噪 |
| cv2.Canny() | 边缘检测 |
| cv2.findContours() | 轮廓检测 |
| cv2.boundingRect() | 获取边界框 |
| cv2.drawContours() | 绘制轮廓 |
| cv2.rectangle() | 绘制矩形 |
| cv2.putText() | 添加文字 |
| cv2.adaptiveThreshold() | 自适应二值化 |
| cv2.morphologyEx() | 形态学操作 |

---

## 数据集说明

### 测试数据集（Dataset_1）
- **来源**：网络采集
- **数量**：约425张
- **内容**：不同光照、角度的书架照片
- **用途**：测试检测效果

### 训练数据集（书脊检测yolo数据集）
- **来源**：开源书脊数据集
- **数量**：3879张
- **格式**：YOLO OBB格式（旋转边界框）
- **标注**：已标注（book_spine类别）
- **用途**：训练YOLOv8检测模型

---

## 已有成果

| 功能 | 状态 | 说明 |
|------|------|------|
| 环境搭建 | ✅ | Python 3.9 + OpenCV 4.8 |
| 图像预处理 | ✅ | 灰度化、高斯模糊、边缘检测 |
| 传统CV检测 | ✅ | Canny边缘检测+轮廓过滤 |
| YOLOv8检测模块 | ✅ | 支持OBB旋转框检测 |
| 颜色分类 | ✅ | K-means聚类（8种颜色） |
| 尺寸分类 | ✅ | 小/中/大三个尺寸类别 |
| Tesseract OCR | ✅ | 基础OCR识别 |
| EasyOCR集成 | ✅ | 支持中文垂直文字识别 |
| 大模型分类 | ✅ | 智谱AI GLM-4-Flash |
| 可视化输出 | ✅ | 检测框、分类标签绘制 |

---

## 待开发任务

| 任务 | 描述 | 负责人 | 状态 |
|------|------|--------|------|
| YOLO模型训练 | 使用3879张数据集训练YOLOv8模型 | 算法优化组 | 待开始 |
| OCR优化 | 提升垂直书脊文字识别准确率 | 算法优化组 | 进行中 |
| 可视化平台 | 开发Web界面展示检测结果 | 应用实践组 | 待开始 |
| 模型部署 | 将模型部署到边缘设备 | 应用实践组 | 待开始 |

---

## 注意事项

1. **EasyOCR模型**：首次使用需下载模型文件到 `models/easyocr/` 目录
2. **大模型API**：使用智谱AI需要配置API Key（已内置）
3. **YOLO训练**：需要GPU环境，建议使用Google Colab或本地GPU
4. **数据路径**：确保数据集路径正确，避免中文路径问题

---

## 提交记录

### 版本历史
- v1.0：基础框架搭建完成
- v1.1：集成传统CV检测和颜色分类
- v1.2：集成YOLOv8检测模块
- v1.3：集成EasyOCR和大模型分类