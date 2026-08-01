import time
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

WEIGHTS_PATH = r"weights/best.pt"
TEST_IMAGES_DIR = r'test\images'  

# Khởi tạo mô hình
yolo = YOLO(WEIGHTS_PATH)
ocr = PaddleOCR(use_angle_cls=False, lang='en', ocr_version='PP-OCRv3', show_log=False)

# Lấy danh sách ảnh test
image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.[jJ][pP][gG]')) + \
              glob.glob(os.path.join(TEST_IMAGES_DIR, '*.[pP][nN][gG]'))

total_yolo_time = 0
total_ocr_time = 0
total_images = len(image_paths)

print(f"Bắt đầu đo đạc hiệu năng trên {total_images} ảnh test...\n")
print(f"{'STT':<5} | {'Tên ảnh':<25} | {'Biển số đọc được':<20}")
print("-" * 55)

for idx, img_path in enumerate(image_paths, 1):
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Đo thời gian YOLOv8 
    t0 = time.time()
    results = yolo.predict(img, verbose=False)
    t1 = time.time()
    yolo_time = (t1 - t0) * 1000 # Chuyển sang mili-giây (ms)
    total_yolo_time += yolo_time

    # Đo thời gian OCR 
    ocr_time = 0
    plate_text = "Không tìm thấy"
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                t2 = time.time()
                ocr_res = ocr.ocr(plate_crop, det=True, cls=False)
                t3 = time.time()
                ocr_time += (t3 - t2) * 1000
                
                texts = []
                if ocr_res and ocr_res[0]:
                    for line in ocr_res[0]:
                        clean_part = re.sub(r'[^A-Z0-9]', '', str(line[1][0]).upper())
                        if clean_part:
                            texts.append(clean_part)
                if texts:
                    plate_text = "-".join(texts)

    total_ocr_time += ocr_time
    print(f"{idx:<5} | {os.path.basename(img_path):<25} | {plate_text:<20}")

# TÍNH TOÁN KẾT QUẢ 
avg_yolo = total_yolo_time / total_images
avg_ocr = total_ocr_time / total_images
avg_total = avg_yolo + avg_ocr
fps = 1000 / avg_total if avg_total > 0 else 0

print("\n" + "="*50)
print("BẢNG TỔNG HỢP KẾT QUẢ ĐO ĐẠC THỰC TẾ")
print("="*50)
print(f"• Tổng số ảnh kiểm thử          : {total_images} ảnh")
print(f"• Tốc độ YOLOv8n trung bình     : {avg_yolo:.2f} ms")
print(f"• Tốc độ PaddleOCR trung bình   : {avg_ocr:.2f} ms")
print(f"• Tổng độ trễ End-to-End        : {avg_total:.2f} ms")
print(f"• Tốc độ xử lý tương đương (FPS): {fps:.2f} FPS")
print("="*50)
