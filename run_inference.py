import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
import argparse

# Import lớp FeatureExtractor từ module src
from src.extractor import FeatureExtractor

# --- Cấu hình ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIMILARITY_THRESHOLD = 0.35  # Ngưỡng tương đồng

def cosine_similarity(v1, v2):
    # v1: (1, D), v2: (N, D)
    return (v1 @ v2.T).squeeze()

def process_video(video_path, ref_img_dir, detector, extractor):
    """
    Xử lý một video duy nhất và trả về danh sách các detection.
    """
    # 1. Xử lý ảnh tham chiếu để tạo vector mục tiêu
    ref_image_paths = list(ref_img_dir.glob("*.jpg"))
    if not ref_image_paths:
        return [] # Trả về rỗng nếu không có ảnh tham chiếu

    ref_features = []
    for img_path in ref_image_paths:
        img = cv2.imread(str(img_path))
        feature = extractor.extract(img)
        if feature is not None:
            ref_features.append(feature)

    if not ref_features:
        return [] # Trả về rỗng nếu không trích xuất được đặc trưng

    target_vector = torch.mean(torch.cat(ref_features, dim=0), dim=0, keepdim=True)

    # 2. Xử lý video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections_for_video = []

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_path.parent.name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy detector, không giới hạn class để tổng quát hơn
        results = detector(frame, imgsz=1024, conf=0.4, verbose=False)

        candidate_boxes = results[0].boxes.xyxy
        if len(candidate_boxes) > 0:
            cropped_images = []
            valid_boxes = []
            for box in candidate_boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    cropped_images.append(crop)
                    valid_boxes.append(box.cpu().numpy().tolist())

            if cropped_images:
                candidate_features = torch.cat([extractor.extract(img) for img in cropped_images if img is not None], dim=0)
                if candidate_features.nelement() == 0: continue

                similarities = cosine_similarity(target_vector, candidate_features)
                if similarities.dim() == 0:
                    similarities = similarities.unsqueeze(0)

                matched_indices = torch.where(similarities >= SIMILARITY_THRESHOLD)[0]

                for idx in matched_indices:
                    box = valid_boxes[idx]
                    detection = {
                        "frame": frame_idx,
                        "x1": int(box[0]), "y1": int(box[1]),
                        "x2": int(box[2]), "y2": int(box[3])
                    }
                    detections_for_video.append(detection)

    cap.release()
    return detections_for_video

def main(args):
    print(f"Sử dụng thiết bị: {DEVICE}")
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    # 1. Tải các model
    detector = YOLO('yolo11m.pt')
    extractor = FeatureExtractor()

    all_results = []

    # Lấy danh sách các video sample trong thư mục input
    video_sample_dirs = [d for d in (input_dir / "samples").iterdir() if d.is_dir()]

    for sample_dir in tqdm(video_sample_dirs, desc="Processing All Videos"):
        video_id = sample_dir.name
        video_path = sample_dir / "drone_video.mp4"
        ref_img_dir = sample_dir / "object_images"

        if not video_path.exists():
            print(f"Warning: Video not found for {video_id}")
            continue

        # Xử lý video và lấy kết quả
        detections = process_video(video_path, ref_img_dir, detector, extractor)

        # Định dạng kết quả theo yêu cầu submission
        video_result = {
            "video_id": video_id,
            "detections": []
        }
        if detections:
            # Lưu ý: Định dạng submission yêu cầu nhóm các bboxes lại.
            # Đây là phiên bản đơn giản, có thể cần cải tiến với tracker.
            video_result["detections"].append({"bboxes": detections})

        all_results.append(video_result)

    # 3. Ghi kết quả ra file JSON
    print(f"Đang ghi kết quả vào {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Hoàn tất!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for Zalo AeroEyes Challenge.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the test data directory (e.g., public_test).")
    parser.add_argument("--output_file", type=str, default="submission.json", help="Path to the output submission file.")

    args = parser.parse_args()
    main(args)

