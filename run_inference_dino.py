import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
import argparse

# --- THAY ĐỔI QUAN TRỌNG: IMPORT TỪ MODULE MỚI ---
from src.extractor_dino import DINOv2Extractor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def cosine_similarity_ensemble(ref_features, cand_features):
    if cand_features is None or cand_features.nelement() == 0:
        return torch.tensor([]).to(DEVICE)
    similarity_matrix = ref_features @ cand_features.T
    max_sims, _ = torch.max(similarity_matrix, dim=0)
    return max_sims

def process_video(video_path, ref_img_dir, detector, extractor, similarity_threshold):
    # Logic của hàm này gần như y hệt phiên bản ensemble trước đó
    ref_image_paths = list(ref_img_dir.glob("*.jpg"))
    if not ref_image_paths: return []
    
    ref_features = [extractor.extract(cv2.imread(str(p))) for p in ref_image_paths]
    ref_features = [f for f in ref_features if f is not None]
    if not ref_features: return []
    ref_features_tensor = torch.cat(ref_features, dim=0)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections_for_video = []

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_path.parent.name}", leave=False):
        ret, frame = cap.read()
        if not ret: break
        
        results = detector(frame, imgsz=1024, conf=0.4, verbose=False)
        candidate_boxes = results[0].boxes.xyxy
        if len(candidate_boxes) > 0:
            cropped_images, valid_boxes = [], []
            for box in candidate_boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    cropped_images.append(crop)
                    valid_boxes.append(box.cpu().numpy().tolist())
            
            if cropped_images:
                extracted_feats = [extractor.extract(img) for img in cropped_images]
                valid_feats = [f for f in extracted_feats if f is not None]
                if not valid_feats: continue
                candidate_features = torch.cat(valid_feats, dim=0)
                
                similarities = cosine_similarity_ensemble(ref_features_tensor, candidate_features)
                if similarities.dim() == 0: similarities = similarities.unsqueeze(0)
                
                matched_indices = torch.where(similarities >= similarity_threshold)[0]
                
                for idx in matched_indices:
                    box = valid_boxes[idx]
                    detections_for_video.append({
                        "frame": frame_idx, "x1": int(box[0]), "y1": int(box[1]),
                        "x2": int(box[2]), "y2": int(box[3])
                    })
    
    cap.release()
    return detections_for_video

def main(args):
    print(f"Sử dụng thiết bị: {DEVICE}")
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    similarity_threshold = args.similarity_threshold
    
    detector = YOLO('yolo11m.pt')
    # --- THAY ĐỔI QUAN TRỌNG: SỬ DỤNG DINOv2Extractor ---
    extractor = DINOv2Extractor()

    all_results = []
    video_sample_dirs = [d for d in (input_dir / "samples").iterdir() if d.is_dir()]

    for sample_dir in tqdm(video_sample_dirs, desc="Processing All Videos"):
        video_id = sample_dir.name
        video_path = sample_dir / "drone_video.mp4"
        ref_img_dir = sample_dir / "object_images"
        if not video_path.exists(): continue

        detections = process_video(video_path, ref_img_dir, detector, extractor, similarity_threshold)
        all_results.append({ "video_id": video_id, "detections": [{"bboxes": detections}] if detections else [] })

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Hoàn tất!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DINOv2 + ENSEMBLE inference.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the test data directory.")
    parser.add-argument("--output_file", type=str, default="submission.json", help="Path to the output submission file.")
    # Ngưỡng của DINOv2 có thể khác, bắt đầu với 0.8 hoặc 0.85 là hợp lý
    parser.add-argument("--similarity_threshold", type=float, default=0.85, help="Cosine similarity threshold for matching.")
    args = parser.parse_args()
    main(args)
