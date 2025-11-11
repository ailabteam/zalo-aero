import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import json
from tqdm import tqdm
import random
import pickle
import shutil

# --- Cấu hình ---
DATA_DIR = Path("./data/train")
ANNOTATIONS_PATH = DATA_DIR / "annotations/annotations.json"

# Thư mục mới để lưu các ảnh đã crop
FINETUNE_IMG_DIR = Path("./data/finetune_data/images")
OUTPUT_PKL_PATH = Path("./data/finetune_data/finetune_paths.pkl")

# Xóa thư mục cũ và tạo mới để đảm bảo sạch sẽ
if FINETUNE_IMG_DIR.exists():
    shutil.rmtree(FINETUNE_IMG_DIR)
FINETUNE_IMG_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_SPLIT_NUM = 10 

def main():
    print("Bắt đầu chuẩn bị dữ liệu cho fine-tuning (Memory Optimized)...")

    with open(ANNOTATIONS_PATH, 'r') as f:
        all_annotations = json.load(f)

    all_video_ids = sorted([d['video_id'] for d in all_annotations])
    train_ids = all_video_ids[:TRAIN_SPLIT_NUM]
    print(f"Sử dụng {len(train_ids)} video để tạo dữ liệu: {train_ids}")

    # 1. Trích xuất và LƯU các ảnh ground-truth ra đĩa
    # all_gt_paths sẽ lưu đường dẫn thay vì ảnh
    all_gt_paths = {} # {video_id: [path1, path2, ...]}
    train_annotations = [info for info in all_annotations if info['video_id'] in train_ids]

    for video_info in tqdm(train_annotations, desc="Extracting and Saving GT images"):
        video_id = video_info['video_id']
        video_path = DATA_DIR / f"samples/{video_id}/drone_video.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): continue

        # Tạo thư mục con cho mỗi video
        video_img_dir = FINETUNE_IMG_DIR / video_id
        video_img_dir.mkdir(exist_ok=True)
        
        paths_for_video = []
        
        frames_to_read = {}
        for anno_group in video_info.get("annotations", []):
            for bbox in anno_group.get("bboxes", []):
                frame_num = bbox["frame"]
                if frame_num not in frames_to_read:
                    frames_to_read[frame_num] = []
                frames_to_read[frame_num].append(bbox)

        for i, frame_num in enumerate(sorted(frames_to_read.keys())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                for j, bbox in enumerate(frames_to_read[frame_num]):
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        # Lưu ảnh crop ra file
                        img_path = video_img_dir / f"frame_{frame_num}_crop_{j}.jpg"
                        cv2.imwrite(str(img_path), crop)
                        paths_for_video.append(str(img_path))
        
        all_gt_paths[video_id] = paths_for_video
        cap.release()

    # 2. Tạo các cặp đường dẫn
    finetune_path_pairs = []
    for video_id in tqdm(train_ids, desc="Creating path pairs"):
        ref_img_dir = DATA_DIR / f"samples/{video_id}/object_images"
        ref_paths = [str(p) for p in ref_img_dir.glob("*.jpg")]
        gt_paths = all_gt_paths.get(video_id, [])

        if not ref_paths or not gt_paths: continue

        # Tạo cặp Positive
        num_pos_pairs = min(len(gt_paths), 200)
        for _ in range(num_pos_pairs):
            ref_path = random.choice(ref_paths)
            gt_path = random.choice(gt_paths)
            finetune_path_pairs.append((ref_path, gt_path, 1))
        
        # Tạo cặp Negative
        num_neg_pairs = num_pos_pairs
        other_video_ids = [vid for vid in train_ids if vid != video_id]
        if not other_video_ids: continue

        for _ in range(num_neg_pairs):
            ref_path = random.choice(ref_paths)
            other_id = random.choice(other_video_ids)
            if all_gt_paths.get(other_id):
                neg_gt_path = random.choice(all_gt_paths[other_id])
                finetune_path_pairs.append((ref_path, neg_gt_path, 0))
    
    random.shuffle(finetune_path_pairs)
    print(f"Tạo thành công {len(finetune_path_pairs)} cặp đường dẫn.")

    # 3. Lưu lại bộ dữ liệu đường dẫn
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        pickle.dump(finetune_path_pairs, f)
    
    print(f"Dữ liệu đường dẫn đã được lưu tại: {OUTPUT_PKL_PATH}")

if __name__ == "__main__":
    main()
