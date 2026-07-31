from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=30,             
        imgsz=640,             
        batch=8,               
        workers=0,          
        device='cpu',          
        name='traffic_model'
    )

    print("HUẤN LUYỆN HOÀN TẤT!")
    print("Mô hình đã được lưu tại: runs/detect/traffic_model/weights/best.pt")