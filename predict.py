from ultralytics import YOLO

# Load mô hình đã được huấn luyện thành công
model = YOLO('weights/best.pt')

# Chạy dự đoán trên tập ảnh test 
results = model.predict(
    source='test/images',  # Thư mục ảnh cần nhận diện
    save=True,             # Lưu ảnh kết quả có vẽ khung
    conf=0.25              # Ngưỡng tin cậy 
)

print("DỰ ĐOÁN HOÀN TẤT!")
print("Ảnh kết quả có vẽ khung đã được lưu trong thư mục: runs/detect/predict")
