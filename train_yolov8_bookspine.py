"""
YOLOv8n 书脊检测训练脚本
使用3879张书脊数据集训练轻量级检测模型
"""

from ultralytics import YOLO
import os

def train_bookspine_detector():
    model = YOLO('yolov8n.pt')
    
    result = model.train(
        data=r'D:/HELL/CV/shujia/书脊检测yolo数据集(3879张).v1i.yolov8/data_fixed.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',
        project='runs/obb',
        name='bookspine_detector',
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )
    
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    
    return result

def evaluate_model():
    model = YOLO('runs/obb/bookspine_detector/weights/best.pt')
    
    results = model.val(
        data=r'D:/HELL/CV/shujia/书脊检测yolo数据集(3879张).v1i.yolov8/data_fixed.yaml',
    )
    
    print("\n" + "="*50)
    print("评估结果：")
    print(f"mAP@50: {results.box.map50:.4f}")
    print(f"mAP@50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print("="*50)
    
    return results

def predict_test_images():
    model = YOLO('runs/obb/bookspine_detector/weights/best.pt')
    
    test_dir = r'D:/HELL/CV/shujia/书脊检测yolo数据集(3879张).v1i.yolov8/test/images'
    output_dir = r'D:/HELL/CV/shujia/prediction_results'
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))][:10]
    
    for img_path in test_images:
        results = model.predict(img_path, save=True, project=output_dir, name='predict')
        print(f"已处理: {os.path.basename(img_path)}")
    
    print(f"\n预测结果保存在: {output_dir}/predict")

if __name__ == '__main__':
    print("="*50)
    print("YOLOv8n 书脊检测训练")
    print("="*50)
    
    print("\n[1/3] 开始训练...")
    train_bookspine_detector()
    
    print("\n[2/3] 评估模型...")
    evaluate_model()
    
    print("\n[3/3] 预测测试图片...")
    predict_test_images()
    
    print("\n全部完成！")
