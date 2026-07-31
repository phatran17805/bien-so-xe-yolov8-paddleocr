from ultralytics import YOLO

if __name__ == '__main__':
    # Load mô hình best
    model = YOLO('runs/detect/traffic_model/weights/best.pt')

    # Đánh giá trên tập test
    metrics = model.val(data='data.yaml', split='test')
    
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")