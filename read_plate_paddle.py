import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

import cv2
import re
import glob
import sqlite3
from datetime import datetime

from ultralytics import YOLO
import paddle

paddle.set_flags({'FLAGS_use_mkldnn': False})
paddle.set_device('cpu')

from paddleocr import PaddleOCR

# CẤU HÌNH CƠ SỞ DỮ LIỆU SQLITE
DB_FILE = 'traffic_log.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            vehicle_type TEXT,
            license_plate TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(image_name, vehicle_type, plate_text):
    if not plate_text:
        return
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO vehicle_logs (timestamp, image_name, vehicle_type, license_plate)
        VALUES (?, ?, ?, ?)
    ''', (now, image_name, vehicle_type, plate_text))
    conn.commit()
    conn.close()

init_db()

# 1. Khởi tạo PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=False, 
    lang='en', 
    show_log=False, 
    enable_mkldnn=False, 
    cpu_threads=1,
    use_gpu=False,
    ocr_version='PP-OCRv3'
)

# 2. Load mô hình YOLOv8 
model = YOLO('weights/best.pt')

input_dir = 'test/images'
output_dir = 'runs/detect/paddle_results'
os.makedirs(output_dir, exist_ok=True)

img_paths = glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]'))

print(f"🚀 Bắt đầu nhận diện trên {len(img_paths)} ảnh...\n")

detected_count = 0

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    image = cv2.imread(img_path)
    if image is None:
        continue
        
    results = model(image, verbose=False)
    
    for r in results:
        boxes = r.boxes
        
        vehicle_boxes = []
        plate_boxes = []
        
        # Phân loại các box nhận diện được từ YOLO
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0])
            
            if conf > 0.2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_name in ['car', 'motorcycle']:
                    vehicle_boxes.append({'name': cls_name, 'box': (x1, y1, x2, y2)})
                elif 'plate' in cls_name or 'license' in cls_name:
                    plate_boxes.append({'box': (x1, y1, x2, y2)})

        # Ghép biển số với phương tiện tương ứng
        for plate in plate_boxes:
            px1, py1, px2, py2 = plate['box']
            plate_center_x = (px1 + px2) / 2
            plate_center_y = (py1 + py2) / 2
            
            # Tìm xem biển số nằm trong khung phương tiện nào
            vehicle_type = "unknown"
            for v in vehicle_boxes:
                vx1, vy1, vx2, vy2 = v['box']
                if vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2:
                    vehicle_type = v['name']
                    break
            
            # Nếu biển số nằm hơi lệch ngoài khung xe nhưng ảnh có xe, gán theo xe tìm thấy
            if vehicle_type == "unknown" and len(vehicle_boxes) > 0:
                vehicle_type = vehicle_boxes[0]['name']

            plate_crop = image[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            detected_count += 1
            clean_text = ""

            try:
                ocr_result = ocr.ocr(plate_crop, det=True, cls=False)
                if ocr_result and ocr_result[0]:
                    texts = []
                    for line in ocr_result[0]:
                        raw_text = str(line[1][0])
                        clean_part = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                        if clean_part:
                            texts.append(clean_part)
                    clean_text = "-".join(texts)
            except Exception as e:
                print(f"Lỗi OCR ở file {img_name}: {e}")

            # Lưu vào Database
            if clean_text:
                save_to_db(img_name, vehicle_type, clean_text)

            print(f"[{detected_count}] File: {img_name} | Loại xe: {vehicle_type.upper()} | Biển số: '{clean_text}'")
            
            # Vẽ khung màu cam & viết tên phương tiện + biển số lên ảnh
            cv2.rectangle(image, (px1, py1), (px2, py2), (0, 165, 255), 2)
            if clean_text:
                display_str = f"{vehicle_type}: {clean_text}"
                cv2.putText(image, display_str, (px1, max(py1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, img_name), image)

print(f"\nĐã xử lý xong {detected_count} biển số!")
print(f"Dữ liệu loại xe & biển số đã lưu vào: '{DB_FILE}'")
