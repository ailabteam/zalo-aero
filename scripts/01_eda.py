import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# --- Cấu hình đường dẫn ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# !!! THAY ĐỔI QUAN TRỌNG Ở ĐÂY !!!
TRAIN_DATA_DIR = DATA_DIR / "train" # Dữ liệu training nằm trong thư mục 'train'
# ---------------------------

ANNOTATIONS_PATH = TRAIN_DATA_DIR / "annotations/annotations.json"
SAMPLES_DIR = TRAIN_DATA_DIR / "samples"

def analyze_annotations(annotations_data):
    print("--- Phân Tích File Annotation ---")
    num_videos = len(annotations_data)
    print(f"Tổng số video có trong file annotation: {num_videos}")
    all_bbox_widths, all_bbox_heights, annotated_frames_counts = [], [], []
    for video_info in tqdm(annotations_data, desc="Phân tích annotations"):
        total_bboxes_in_video = 0
        if "annotations" in video_info and video_info["annotations"]:
            for annotation_group in video_info["annotations"]:
                bboxes = annotation_group.get("bboxes", [])
                total_bboxes_in_video += len(bboxes)
                for bbox in bboxes:
                    all_bbox_widths.append(bbox['x2'] - bbox['x1'])
                    all_bbox_heights.append(bbox['y2'] - bbox['y1'])
        annotated_frames_counts.append(total_bboxes_in_video)
    if annotated_frames_counts:
        print("\n[Thống kê số lượng frame được gán nhãn mỗi video]")
        print(f"  - Nhiều nhất: {np.max(annotated_frames_counts)} frames")
        print(f"  - Ít nhất: {np.min(annotated_frames_counts)} frames")
        print(f"  - Trung bình: {np.mean(annotated_frames_counts):.2f} frames")
    if all_bbox_widths:
        print("\n[Thống kê kích thước Bounding Box (width x height)]")
        print(f"  - Chiều rộng (width): Lớn nhất={np.max(all_bbox_widths)}, Nhỏ nhất={np.min(all_bbox_widths)}, Trung bình={np.mean(all_bbox_widths):.2f}")
        print(f"  - Chiều cao (height): Lớn nhất={np.max(all_bbox_heights)}, Nhỏ nhất={np.min(all_bbox_heights)}, Trung bình={np.mean(all_bbox_heights):.2f}")
    print("-" * 30)

def analyze_video_properties(annotations_data):
    print("\n--- Phân Tích Thuộc Tính Video (sử dụng OpenCV) ---")
    video_ids = [item['video_id'] for item in annotations_data]
    # Lấy 5 video đầu tiên để phân tích
    for video_id in tqdm(video_ids[:5], desc="Phân tích 5 video đầu tiên"):
        video_path = SAMPLES_DIR / video_id / "drone_video.mp4"
        if not video_path.exists():
            print(f"\n[WARNING] Không tìm thấy file: {video_path}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        print(f"\nThông tin video: {video_id} | Độ phân giải: {width}x{height} | FPS: {fps:.2f} | Tổng số frame: {frame_count} | Thời lượng: {duration_sec:.2f}s")
        cap.release()
    print("-" * 30)


if __name__ == "__main__":
    print(f"Đường dẫn project gốc: {PROJECT_ROOT}")
    print(f"Đường dẫn dữ liệu training: {TRAIN_DATA_DIR}")
    
    if not ANNOTATIONS_PATH.exists():
        print(f"[ERROR] Không tìm thấy file annotation tại: {ANNOTATIONS_PATH}")
        print("Vui lòng kiểm tra lại cấu trúc thư mục và đường dẫn trong script.")
        sys.exit(1)
    
    with open(ANNOTATIONS_PATH, 'r') as f:
        data = json.load(f)

    analyze_annotations(data)
    analyze_video_properties(data)
