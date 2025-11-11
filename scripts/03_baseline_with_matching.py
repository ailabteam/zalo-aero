import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import torchreid
import torchvision.transforms as T

# --- Cấu hình ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_ID = "Person1_0"
VIDEO_PATH = TRAIN_DATA_DIR / f"samples/{VIDEO_ID}/drone_video.mp4"
REF_IMG_DIR = TRAIN_DATA_DIR / f"samples/{VIDEO_ID}/object_images"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / f"{VIDEO_ID}_matched_output.mp4"

SIMILARITY_THRESHOLD = 0.5 # Ngưỡng tương đồng, có thể tinh chỉnh
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Lớp trích xuất đặc trưng ---
class FeatureExtractor:
    def __init__(self, model_name='osnet_x1_0'):
        # Sử dụng torchreid để tải model Re-ID đã được huấn luyện trên Market1501
        print(f"Đang tải model Re-ID: {model_name}...")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1, # Không quan trọng, sẽ bỏ lớp classifier
            pretrained=True
        )
        self.model.to(DEVICE)
        self.model.eval()
        print("Tải model Re-ID thành công!")

        # Chuẩn bị pipeline tiền xử lý ảnh cho model Re-ID
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)), # Kích thước input chuẩn của nhiều model Re-ID
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, image_bgr):
        # Ảnh đầu vào là ảnh BGR từ OpenCV
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(DEVICE)
        feature = self.model(image_tensor)
        # Chuẩn hóa vector đặc trưng để so sánh cosine tốt hơn
        return torch.nn.functional.normalize(feature, p=2, dim=1)

def cosine_similarity(v1, v2):
    # v1: (1, D), v2: (N, D)
    return (v1 @ v2.T).squeeze()

# --- Chương trình chính ---
def main():
    print(f"Sử dụng thiết bị: {DEVICE}")

    # 1. Tải các model
    detector = YOLO('yolo11m.pt')
    extractor = FeatureExtractor()

    # 2. Xử lý ảnh tham chiếu để tạo vector mục tiêu
    ref_image_paths = list(REF_IMG_DIR.glob("*.jpg"))
    if not ref_image_paths:
        print(f"[ERROR] Không tìm thấy ảnh tham chiếu trong {REF_IMG_DIR}")
        return

    print("Đang trích xuất đặc trưng từ ảnh tham chiếu...")
    ref_features = []
    for img_path in ref_image_paths:
        img = cv2.imread(str(img_path))
        feature = extractor.extract(img)
        ref_features.append(feature)

    # Lấy trung bình các vector đặc trưng làm vector mục tiêu
    target_vector = torch.mean(torch.cat(ref_features, dim=0), dim=0, keepdim=True)
    print("Tạo vector mục tiêu thành công!")

    # 3. Mở video nguồn và chuẩn bị video output
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Bắt đầu xử lý video: {VIDEO_PATH.name}")
    # 4. Lặp qua từng frame để xử lý
    for frame_idx in tqdm(range(total_frames), desc="Xử lý video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chạy Detector
        results = detector(frame, imgsz=1024, conf=0.4, classes=[0], verbose=False) # Chỉ phát hiện người (class 0)
        
        candidate_boxes = results[0].boxes.xyxy
        # CHỈ XỬ LÝ NẾU CÓ ỨNG VIÊN
        if len(candidate_boxes) > 0:
            cropped_images = []
            valid_boxes = [] # Lưu lại các box hợp lệ (không bị rỗng)
            for box in candidate_boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0: # Đảm bảo ảnh crop không rỗng
                    cropped_images.append(crop)
                    valid_boxes.append(box)
            
            if cropped_images:
                # Trích xuất đặc trưng cho tất cả các ứng viên cùng lúc
                candidate_features = torch.cat([extractor.extract(img) for img in cropped_images], dim=0)

                # Tính độ tương đồng
                similarities = cosine_similarity(target_vector, candidate_features)
                
                # --- SỬA LỖI Ở ĐÂY ---
                # Đảm bảo similarities luôn là 1D tensor để có thể duyệt qua
                if similarities.dim() == 0:
                    similarities = similarities.unsqueeze(0)
                
                # Lọc các box có độ tương đồng cao
                matched_indices = torch.where(similarities >= SIMILARITY_THRESHOLD)[0]
                
                for idx in matched_indices:
                    box = valid_boxes[idx] # Dùng valid_boxes thay vì candidate_boxes
                    x1, y1, x2, y2 = map(int, box)
                    sim = similarities[idx].item()
                    
                    # Vẽ box và điểm số tương đồng lên frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Match: {sim:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)

    # 5. Giải phóng tài nguyên
    cap.release()
    out.release()
    print(f"Xử lý hoàn tất! Video kết quả được lưu tại: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()

