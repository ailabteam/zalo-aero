import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import json
import argparse
import torchvision.transforms as T
from collections import defaultdict

# --- ĐỊNH NGHĨA EXTRACTOR CHO MODEL FINE-TUNED (Lấy từ validation.py) ---
class FineTunedDINOv2Extractor:
    def __init__(self, model_path):
        print(f"Đang tải model DINOv2 FINE-TUNED từ: {model_path}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Tải model FINE-TUNED thành công!")
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    @torch.no_grad()
    def extract(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0: return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        features = self.model(image_tensor)
        return torch.nn.functional.normalize(features, p=2, dim=1)

# --- CÁC HÀM TIỆN ÍCH ---
def cosine_similarity_ensemble(ref_features, cand_features):
    if cand_features is None or cand_features.nelement() == 0:
        return torch.tensor([]).to(ref_features.device)
    similarity_matrix = ref_features @ cand_features.T
    max_sims, _ = torch.max(similarity_matrix, dim=0)
    return max_sims

# --- HÀM XỬ LÝ VIDEO "TỐI THƯỢNG" ---
def process_video_final(video_path, ref_img_dir, detector, extractor, similarity_threshold):
    # 1. Chuẩn bị vector tham chiếu (ensemble)
    ref_image_paths = list(ref_img_dir.glob("*.jpg"))
    if not ref_image_paths: return []
    ref_features = [extractor.extract(cv2.imread(str(p))) for p in ref_image_paths]
    ref_features = [f for f in ref_features if f is not None]
    if not ref_features: return []
    ref_features_tensor = torch.cat(ref_features, dim=0)

    # 2. Xử lý video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_tracked_bboxes = defaultdict(list) # {track_id: [(frame, box), ...]}

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_path.parent.name}", leave=False):
        ret, frame = cap.read()
        if not ret: break
        
        # Chạy Detector + Tracker
        results = detector.track(frame, persist=True, tracker="botsort.yaml", imgsz=1024, conf=0.4, verbose=False)
        
        if results[0].boxes.id is None: continue
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu()
        
        # Lấy feature cho TẤT CẢ các track đang có trong frame
        cropped_images = [frame[max(0,int(b[1])):int(b[3]), max(0,int(b[0])):int(b[2])] for b in boxes]
        valid_crops_info = [(crop, tid) for crop, tid in zip(cropped_images, track_ids) if crop.shape[0]>0 and crop.shape[1]>0]
        if not valid_crops_info: continue
            
        candidate_features = torch.cat([extractor.extract(crop) for crop, tid in valid_crops_info], dim=0)
        current_track_ids = [tid for crop, tid in valid_crops_info]

        # So khớp
        similarities = cosine_similarity_ensemble(ref_features_tensor, candidate_features)
        
        # Lưu lại tất cả các box có độ tương đồng cao
        for i, sim in enumerate(similarities):
            if sim >= similarity_threshold:
                track_id = current_track_ids[i]
                box = boxes[i].numpy().tolist()
                all_tracked_bboxes[track_id].append((frame_idx, box))
    
    cap.release()
    
    # --- POST-PROCESSING ---
    final_detections = []
    MIN_TRACK_LENGTH = 10 # Chỉ giữ lại các track xuất hiện trong ít nhất 10 frame
    
    for track_id, track_data in all_tracked_bboxes.items():
        if len(track_data) >= MIN_TRACK_LENGTH:
            for frame_idx, box in track_data:
                final_detections.append({
                    "frame": frame_idx, "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                })
    
    # Sắp xếp lại theo frame cho đúng định dạng
    final_detections.sort(key=lambda x: x['frame'])
    return final_detections

# --- HÀM MAIN ---
def main(args):
    # 1. Tải model
    detector = YOLO('yolo11m.pt')
    extractor = FineTunedDINOv2Extractor(args.model_path)

    # 2. Xử lý tất cả video trong input_dir
    all_results = []
    video_sample_dirs = [d for d in (Path(args.input_dir) / "samples").iterdir() if d.is_dir()]

    for sample_dir in tqdm(video_sample_dirs, desc="Processing All Videos"):
        video_id = sample_dir.name
        # ...
        detections = process_video_final(
            sample_dir / "drone_video.mp4", 
            sample_dir / "object_images", 
            detector, 
            extractor, 
            args.similarity_threshold
        )
        all_results.append({ "video_id": video_id, "detections": [{"bboxes": detections}] if detections else [] })

    # 3. Ghi kết quả
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Hoàn tất! Submission được lưu tại {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FINAL inference for Zalo AeroEyes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the test data directory.")
    parser.add_argument("--output_file", type=str, default="final_submission.json", help="Path to the output submission file.")
    parser.add_argument("--model_path", type=str, default="./weights/dinov2_finetuned.pth", help="Path to the fine-tuned model weights.")
    parser.add_argument("--similarity_threshold", type=float, required=True, help="THE GOLDEN THRESHOLD found from validation.py.")
    args = parser.parse_args()
    main(args)
