import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torch

# --- Cấu hình ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True) # Tạo thư mục outputs nếu chưa có

# Chọn một video để thử nghiệm
VIDEO_ID = "Person1_0" 
VIDEO_PATH = TRAIN_DATA_DIR / f"samples/{VIDEO_ID}/drone_video.mp4"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / f"{VIDEO_ID}_yolo11_output.mp4" # Đổi tên file output

# --- Chương trình chính ---
def main():
    # Kiểm tra xem có GPU không
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Không tìm thấy video tại: {VIDEO_PATH}")
        return

    # 1. Tải model YOLOv11 (NÂNG CẤP TỪ v8)
    # Thư viện ultralytics sẽ tự động tải về trọng số khi chạy lần đầu
    print("Đang tải model YOLOv11...")
    model = YOLO('yolo11m.pt') 
    model.to(device)
    print("Tải model thành công!")

    # 2. Mở video nguồn
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Chuẩn bị video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))
    
    print(f"Bắt đầu xử lý video: {VIDEO_PATH.name} với YOLOv11")
    # 4. Lặp qua từng frame
    with tqdm(total=total_frames, desc="Xử lý video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Chạy YOLOv11 inference trên frame
            # imgsz=1024 để xử lý vật thể nhỏ tốt hơn
            # conf=0.4 để lọc bớt các phát hiện có độ tin cậy thấp
            results = model(frame, imgsz=1024, conf=0.4, verbose=False)

            # Lấy kết quả và vẽ lên frame
            annotated_frame = results[0].plot()

            # Ghi frame đã xử lý vào video output
            out.write(annotated_frame)
            pbar.update(1)

    # 5. Giải phóng tài nguyên
    cap.release()
    out.release()
    print(f"Xử lý hoàn tất! Video kết quả được lưu tại: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
