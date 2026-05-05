"""
快速测试脚本 - 智能书架整理系统整合版
用于测试整合后的系统功能
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shelf_organizer.src.main_integrated import ShelfOrganizerIntegrated


def quick_test():
    """快速测试整合版系统"""
    
    print("="*60)
    print("智能书架整理系统 - 整合版快速测试")
    print("="*60)
    
    test_image_dir = r"D:\HELL\CV\shujia\Dataset_1"
    output_dir = r"D:\HELL\CV\shujia\output_integrated"
    
    if not os.path.exists(test_image_dir):
        print(f"错误: 测试图像目录不存在: {test_image_dir}")
        return
    
    test_images = [f for f in os.listdir(test_image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print(f"错误: 测试图像目录中没有图像文件")
        return
    
    print(f"\n找到 {len(test_images)} 张测试图像")
    print(f"使用前5张进行测试...")
    
    test_images = test_images[:5]
    
    print("\n" + "-"*60)
    print("方案1: YOLOv8 + EasyOCR + 大模型分类 (推荐)")
    print("-"*60)
    
    try:
        organizer = ShelfOrganizerIntegrated(
            detector_type='yolo',
            ocr_type='easyocr',
            classifier_type='llm',
            use_gpu=False
        )
        
        for img_name in test_images:
            img_path = os.path.join(test_image_dir, img_name)
            try:
                result = organizer.process_image(img_path, extract_titles=True, output_debug=True)
                print(f"  {img_name}: 检测到 {result['num_books']} 本书")
                for i, book in enumerate(result['books'][:3]):
                    title = book.get('title', '')[:20]
                    category = book.get('category', '')
                    if title:
                        print(f"    书{i+1}: {title}... -> {category}")
            except Exception as e:
                print(f"  {img_name}: 处理失败 - {e}")
        
        print("\n方案1测试完成!")
        
    except Exception as e:
        print(f"方案1初始化失败: {e}")
        print("可能是依赖未安装，请运行: pip install ultralytics easyocr")
    
    print("\n" + "-"*60)
    print("方案2: 传统CV + Tesseract (原有方案)")
    print("-"*60)
    
    try:
        organizer = ShelfOrganizerIntegrated(
            detector_type='traditional',
            ocr_type='tesseract',
            classifier_type='color'
        )
        
        for img_name in test_images[:2]:
            img_path = os.path.join(test_image_dir, img_name)
            try:
                result = organizer.process_image(img_path, extract_titles=True)
                print(f"  {img_name}: 检测到 {result['num_books']} 本书")
            except Exception as e:
                print(f"  {img_name}: 处理失败 - {e}")
        
        print("\n方案2测试完成!")
        
    except Exception as e:
        print(f"方案2初始化失败: {e}")
    
    print("\n" + "="*60)
    print("快速测试完成!")
    print("="*60)


def train_yolo_model():
    """训练YOLO模型"""
    
    print("="*60)
    print("YOLOv8 书脊检测模型训练")
    print("="*60)
    
    from shelf_organizer.src.book_detector_yolo import BookDetectorYOLO
    
    data_yaml = r"D:\HELL\CV\shujia\书脊检测yolo数据集(3879张).v1i.yolov8\data_fixed.yaml"
    
    if not os.path.exists(data_yaml):
        print(f"错误: 数据配置文件不存在: {data_yaml}")
        return
    
    detector = BookDetectorYOLO()
    
    print("\n开始训练...")
    print("注意: CPU训练可能需要2-4小时，建议使用GPU")
    
    results = detector.train_model(
        data_yaml=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device='cpu'
    )
    
    print("\n训练完成!")
    print(f"模型保存在: runs/obb/bookspine_detector/weights/best.pt")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='智能书架整理系统 - 快速测试')
    parser.add_argument('--train', action='store_true', help='训练YOLO模型')
    parser.add_argument('--test', action='store_true', help='运行快速测试')
    
    args = parser.parse_args()
    
    if args.train:
        train_yolo_model()
    elif args.test:
        quick_test()
    else:
        print("请选择运行模式:")
        print("  python run_integrated.py --test   # 快速测试")
        print("  python run_integrated.py --train  # 训练YOLO模型")
        print("\n正在运行快速测试...")
        quick_test()
