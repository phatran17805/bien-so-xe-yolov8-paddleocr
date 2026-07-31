# Hệ Thống Nhận Diện Biển Số Và Phân Loại Phương Tiện Tự Động 

> **Báo cáo Thực tập Niên luận**  
> **Đề tài:** Nghiên cứu, xây dựng ứng dụng nhận diện biển số và phân loại phương tiện tại bãi xe thông minh dựa trên YOLOv8 và PaddleOCR  
> **Tác giả:** [Trần Văn Pha] - [23T1020376]  

---

## Giới thiệu dự án
Dự án xây dựng chuỗi xử lý End-to-End thực hiện 2 nhiệm vụ chính phục vụ quản lý bãi giữ xe thông minh:
1. **Phân loại phương tiện & Định vị biển số (YOLOv8n):** Phân loại Ô tô (`car`), Xe máy (`motorcycle`) và định vị vùng Biển số (`license_plate`).
2. **Nhận dạng ký tự quang học (PP-OCRv3):** Trích xuất chuỗi ký tự trên biển số xe (hỗ trợ cả biển ô tô 1 dòng dài và biển xe máy 2 dòng vuông).

### Kết quả thực nghiệm chính xác (Tập kiểm thử 50 ảnh):
- **mAP@0.5 (Detection):** `99.5%`
- **Tỷ lệ nhận dạng đúng OCR (Exact Match):** `90.0%`
- **Tốc độ suy luận YOLOv8n (CPU):** `110.75 ms`
- **Thời gian xử lý OCR (CPU):** `206.09 ms`
- **Tổng độ trễ End-to-End:** `316.84 ms` (~`3.16 FPS` trên CPU Intel Core i7-13620H)

---

## 1. Hướng dẫn cài đặt môi trường 

### Yêu cầu hệ thống:
- Python >= 3.8 (Khuyên dùng Python 3.9 hoặc 3.10)

### Cài đặt từng bước:

```bash
# 1. Clone repository về máy
git clone https://github.com/phatran17805/bien-so-xe-yolov8-paddleocr.git
cd bien-so-xe-yolov8-paddleocr

# 2. Tạo và kích hoạt môi trường ảo 
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```
---

## 2. Hướng dẫn tải & Chuẩn bị Tập dữ liệu (Dataset)

- **Tải dữ liệu:** Tải file nén dữ liệu `data.zip` tại [Link Google Drive của bạn ở đây](https://drive.google.com/file/d/1FM3VJ9eHbHzdUD1j7oy04j_gSWrYmlXI/view?usp=sharing) và giải nén vào thư mục gốc của dự án.

- **Cấu trúc thư mục dự án:**

├── weights/
│   └── best.pt               # Trọng số YOLOv8n đã huấn luyện tốt nhất
├── train/
│   ├── images/           # Ảnh huấn luyện (400 ảnh)
│   └── labels/           # Nhãn huấn luyện (.txt format YOLO)
├── val/
│   ├── images/           # Ảnh kiểm định (50 ảnh)
│   └── labels/           # Nhãn kiểm định (.txt format YOLO)
└── test/
    ├── images/           # Ảnh kiểm thử (50 ảnh)
    └── labels/           # Nhãn kiểm thử (.txt format YOLO)
├── data.yaml                 # File khai báo tập dữ liệu và các lớp
├── read_plate_paddle.py      # Module trích xuất và đọc ký tự OCR
├── train.py                  # Script huấn luyện mô hình YOLOv8n
├── val.py                    # Script tính toán chỉ số mAP50, Precision, Recall
├── predict.py                # Script chạy thử nghiệm nhận diện & vẽ khung ảnh
├── benchmark.py              # Script đánh giá hiệu năng và bấm giờ End-to-End
├── requirements.txt          # Danh sách thư viện cần thiết
└── README.md                 # Tài liệu hướng dẫn

- **Quản lý Trọng số:**
Trọng số tốt nhất sau khi huấn luyện best.pt được lưu trữ tại thư mục weights/best.pt.
File trọng số mặc định yolov8n.pt sẽ tự động được tải về từ Ultralytics khi chạy lệnh huấn luyện lần đầu.

---

## Lệnh chạy huấn luyện (Train) và đánh giá (Evaluation):
- **Chạy script huấn luyện mô hình YOLOv8n trên tập dữ liệu đã chuẩn bị:**

python train.py

- **Đánh giá mô hình YOLOv8:**

python val.py

- **Thử nghiệm dự đoán và vẽ khung trên tập test:**

python predict.py

- **Đo độ trễ hệ thống:**

python benchmark.py

- **Chạy thử nghiệm nhận diện ký tự OCR trên tập test:**

python read_plate_paddle.py
