import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import json
from tqdm import tqdm
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2

from ultralytics import YOLO
# Tạm thời chưa import DINOv2Extractor, chúng ta sẽ load model trực tiếp
from run_inference_dino import process_video # Vẫn dùng hàm này vì logic giống

# --- Cấu hình ---
DATA_DIR = Path("./data/train")
ANNOTATIONS_PATH = DATA_DIR / "annotations/annotations.json"
# --- ĐƯỜNG DẪN TỚI MODEL MỚI ---
FINETUNED_MODEL_PATH = Path("./weights/dinov2_finetuned.pth")

class FineTunedDINOv2Extractor:
    """Một class wrapper để load model fine-tuned và có cùng interface."""
    def __init__(self, model_path):
        print(f"Đang tải model DINOv2 FINE-TUNED từ: {model_path}...")
        # Load kiến trúc DINOv2 gốc
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False)
        # Load trọng số đã fine-tune của chúng ta
        self.model.load_state_dict(torch.load(model_path))
        
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        print("Tải model FINE-TUNED thành công!")

        self.transform = T.Compose([
            T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def extract(self, image_bgr):
        # Logic extract y hệt phiên bản gốc
        if image_bgr is None or image_bgr.size == 0: return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.model.device)
        features = self.model(image_tensor)
        return torch.nn.functional.normalize(features, p=2, dim=1)


def load_ground_truth(video_ids):
    # ... (Hàm này giữ nguyên)
    with open(ANNOTATIONS_PATH, 'r') as f: all_annotations = json.load(f)
    gt_data = {}
    for video_info in all_annotations:
        if video_info['video_id'] in video_ids:
            boxes = []
            for anno_group in video_info.get("annotations", []):
                for bbox in anno_group.get("bboxes", []):
                    boxes.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], bbox["frame"]])
            gt_data[video_info['video_id']] = np.array(boxes)
    return gt_data

def main():
    print("Bắt đầu quy trình Validation cho model FINE-TUNED...")
    
    if not FINETUNED_MODEL_PATH.exists():
        print(f"LỖI: Không tìm thấy model fine-tuned tại {FINETUNED_MODEL_PATH}")
        print("Vui lòng chạy script finetune.py trước.")
        return

    # 1. Tải model
    detector = YOLO('yolo11m.pt')
    # --- SỬ DỤNG EXTRACTOR MỚI ---
    extractor = FineTunedDINOv2Extractor(FINETUNED_MODEL_PATH)
    
    # 2. Chia dữ liệu
    all_video_dirs = [d for d in (DATA_DIR / "samples").iterdir() if d.is_dir()]
    all_video_ids = sorted([d.name for d in all_video_dirs])
    validation_ids = all_video_ids[-4:]
    validation_dirs = [d for d in all_video_dirs if d.name in validation_ids]
    print(f"Sử dụng {len(validation_ids)} video để validation: {validation_ids}")

    # 3. Tải Ground Truth
    gt_data = load_ground_truth(validation_ids)

    # 4. Vòng lặp tìm ngưỡng tối ưu
    best_threshold = 0
    best_map = -1
    
    # Với model fine-tuned, đặc trưng sẽ rõ ràng hơn.
    # Chúng ta có thể tìm kiếm ở một khoảng ngưỡng cao hơn và chi tiết hơn.
    search_range = np.arange(0.95, 0.79, -0.01)
    print(f"Bắt đầu tìm kiếm ngưỡng tốt nhất trong khoảng: {search_range.min():.2f} - {search_range.max():.2f}")

    for threshold in search_range:
        threshold = round(threshold, 2)
        print(f"\n--- Thử nghiệm với ngưỡng: {threshold} ---")
        
        metric = MeanAveragePrecision(box_format='xyxy')
        all_preds, all_targets = [], []

        for sample_dir in tqdm(validation_dirs, desc=f"Threshold {threshold}"):
            video_id = sample_dir.name
            # ... (Logic xử lý và tính metric giữ nguyên)
            video_path = sample_dir / "drone_video.mp4"
            ref_img_dir = sample_dir / "object_images"
            pred_bboxes = process_video(video_path, ref_img_dir, detector, extractor, similarity_threshold=threshold)
            gt_bboxes_with_frame = gt_data.get(video_id, np.array([]))
            
            num_frames = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in range(num_frames):
                preds_on_frame = [p for p in pred_bboxes if p["frame"] == frame_idx]
                targets_on_frame = gt_bboxes_with_frame[gt_bboxes_with_frame[:, 4] == frame_idx]
                all_preds.append({
                    "boxes": torch.tensor([list(p.values())[1:] for p in preds_on_frame]).float(),
                    "scores": torch.ones(len(preds_on_frame)).float(),
                    "labels": torch.zeros(len(preds_on_frame)).int(),
                })
                all_targets.append({
                    "boxes": torch.tensor(targets_on_frame[:, :4]).float(),
                    "labels": torch.zeros(len(targets_on_frame)).int(),
                })

        metric.update(all_preds, all_targets)
        results = metric.compute()
        map_score = results['map'].item()
        print(f"Kết quả với ngưỡng {threshold}: mAP = {map_score:.4f}")
        
        if map_score > best_map:
            best_map = map_score
            best_threshold = threshold

    print("\n--- KẾT QUẢ VALIDATION CUỐI CÙNG ---")
    print(f"Ngưỡng tốt nhất tìm được: {best_threshold}")
    print(f"Điểm mAP cao nhất trên tập validation: {best_map:.4f}")

if __name__ == "__main__":
    import torchvision.transforms as T
    main()
