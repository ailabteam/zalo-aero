import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
import argparse
from collections import defaultdict

from src.extractor import FeatureExtractor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def cosine_similarity(v1, v2):
    # v1: (1, D), v2: (N, D)
    return (v1 @ v2.T).squeeze()

def process_video(video_path, ref_img_dir, detector, extractor, similarity_threshold):
    # 1. Chuẩn bị vector tham chiếu trung bình
    ref_image_paths = list(ref_img_dir.glob("*.jpg"))
    if not ref_image_paths: return []
    
    ref_features = [extractor.extract(cv2.imread(str(p))) for p in ref_image_paths]
    ref_features = [f for f in ref_features if f is not None]
    if not ref_features: return []
    target_vector = torch.mean(torch.cat(ref_features, dim=0), dim=0, keepdim=True)

    # 2. Xử lý video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detections_for_video = []
    confirmed_track_ids = set()
    
    # Reset tracker cho mỗi video mới

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_path.parent.name}", leave=False):
        ret, frame = cap.read()
        if not ret: break
        
        results = detector.track(frame, persist=True, tracker="botsort.yaml", imgsz=1024, conf=0.4, verbose=False)
        
        if results[0].boxes.id is None: continue

        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu()
        
        current_detections_in_frame = {} # {track_id: box}

        for i, track_id in enumerate(track_ids):
            current_detections_in_frame[track_id] = boxes[i].numpy().tolist()
            
            # Nếu track_id đã được xác nhận trước đó, thêm ngay vào kết quả
            if track_id in confirmed_track_ids:
                box = current_detections_in_frame[track_id]
                detections_for_video.append({
                    "frame": frame_idx, "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                })
                continue # Bỏ qua việc re-check tốn kém

        # Chỉ re-check các track_id mới
        new_track_ids = [tid for tid in track_ids if tid not in confirmed_track_ids]
        if not new_track_ids: continue

        # Lấy feature cho các track mới
        new_boxes = [current_detections_in_frame[tid] for tid in new_track_ids]
        cropped_images = [frame[max(0,int(b[1])):int(b[3]), max(0,int(b[0])):int(b[2])] for b in new_boxes]
        
        valid_crops_info = [(crop, tid) for crop, tid in zip(cropped_images, new_track_ids) if crop.shape[0]>0 and crop.shape[1]>0]
        if not valid_crops_info: continue

        candidate_features = torch.cat([extractor.extract(crop) for crop, tid in valid_crops_info], dim=0)
        
        similarities = cosine_similarity(target_vector, candidate_features)
        if similarities.dim() == 0: similarities = similarities.unsqueeze(0)
        
        for i, sim in enumerate(similarities):
            if sim >= similarity_threshold:
                track_id = valid_crops_info[i][1]
                confirmed_track_ids.add(track_id)
                
                # Thêm detection của frame hiện tại vào kết quả
                box = current_detections_in_frame[track_id]
                detections_for_video.append({
                    "frame": frame_idx, "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                })
    
    cap.release()
    return detections_for_video

# Hàm main và parser giống hệt như run_inference_ensemble.py
def main(args):
    print(f"Sử dụng thiết bị: {DEVICE}")
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    similarity_threshold = args.similarity_threshold

    # Chỉ cần 1 model duy nhất
    detector = YOLO('yolo11m.pt')
    extractor = FeatureExtractor()

    all_results = []
    video_sample_dirs = [d for d in (input_dir / "samples").iterdir() if d.is_dir()]

    for sample_dir in tqdm(video_sample_dirs, desc="Processing All Videos"):
        video_id = sample_dir.name
        video_path = sample_dir / "drone_video.mp4"
        ref_img_dir = sample_dir / "object_images"
        if not video_path.exists(): continue

        detections = process_video(video_path, ref_img_dir, detector, extractor, similarity_threshold)
        all_results.append({
            "video_id": video_id,
            "detections": [{"bboxes": detections}] if detections else []
        })

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Hoàn tất!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRACKER+ENSEMBLE inference for Zalo AeroEyes Challenge.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the test data directory.")
    parser.add_argument("--output_file", type=str, default="submission.json", help="Path to the output submission file.")
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="Cosine similarity threshold for matching.")
    args = parser.parse_args()
    main(args)

