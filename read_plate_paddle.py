import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

from ultralytics import YOLO

import cv2
import re
import glob
import paddle
from paddleocr import PaddleOCR

paddle.set_device('cpu')

# 1. Khởi tạo PaddleOCR (Tắt det để đọc chữ trực tiếp trên ảnh cắt từ YOLO)
ocr = PaddleOCR(
    use_angle_cls=False, 
    lang='en', 
    show_log=False, 
    enable_mkldnn=False, 
    cpu_threads=1,
    ocr_version='PP-OCRv3'
)

# 2. Load mô hình YOLOv8 
model = YOLO('runs/detect/traffic_model/weights/best.pt')

input_dir = 'test/images'
output_dir = 'runs/detect/paddle_results'
os.makedirs(output_dir, exist_ok=True)

img_paths = glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]'))

print(f"Bắt đầu nhận diện và đọc biển số trên {len(img_paths)} ảnh...\n")

detected_count = 0

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    image = cv2.imread(img_path)
    if image is None:
        continue
        
    results = model(image, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Chỉ lọc Class biển số xe (license_plate)
            if ('plate' in cls_name.lower() or 'license' in cls_name.lower()) and conf > 0.2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Cắt riêng vùng chứa biển số
                plate_crop = image[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                detected_count += 1
                clean_text = ""

                # 3. Đưa vùng ảnh biển số vào PaddleOCR để đọc chữ
                try:
                    # Bật det=True để Paddle nhận diện được biển số 2 dòng của xe máy
                    ocr_result = ocr.ocr(plate_crop, det=True, cls=False)
                    
                    if ocr_result and ocr_result[0]:
                        texts = []
                        # Duyệt qua từng dòng chữ đọc được 
                        for line in ocr_result[0]:
                            raw_text = str(line[1][0])
                            # Làm sạch: Chỉ giữ lại chữ cái và số
                            clean_part = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                            if clean_part:
                                texts.append(clean_part)
                        
                        # Ghép các dòng lại với nhau (thêm dấu '-' ở giữa cho giống biển VN)
                        clean_text = "-".join(texts)
                except Exception as e:
                    print(f"Lỗi OCR ở file {img_name}: {e}")

                # In kết quả nhận diện lên Terminal
                print(f"[{detected_count}] File: {img_name} | Biển số đọc được: '{clean_text}'")
                
                # 4. Vẽ khung màu cam và viết chữ biển số lên ảnh gốc
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                if clean_text:
                    cv2.putText(image, clean_text, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Lưu ảnh kết quả
    cv2.imwrite(os.path.join(output_dir, img_name), image)

print(f"\nĐã phát hiện {detected_count} biển số. Ảnh kết quả lưu tại: '{output_dir}'")