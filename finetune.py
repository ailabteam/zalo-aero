import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import cv2
import torchvision.transforms as T

# --- Cấu hình ---
# Sửa lại đường dẫn đến file pkl chứa ĐƯỜNG DẪN
DATA_PATH = Path("./data/finetune_data/finetune_paths.pkl")
MODEL_SAVE_PATH = Path("./weights/dinov2_finetuned.pth")
MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
MARGIN = 2.0

# --- Định nghĩa Dataset V2: Đọc từ đường dẫn ---
class SiamesePathDataset(Dataset):
    def __init__(self, data_path_pairs, transform):
        self.data_path_pairs = data_path_pairs
        self.transform = transform

    def __len__(self):
        return len(self.data_path_pairs)

    def __getitem__(self, index):
        path1, path2, label = self.data_path_pairs[index]
        
        # Đọc ảnh từ đường dẫn
        img1_bgr = cv2.imread(path1)
        img2_bgr = cv2.imread(path2)
        
        # Xử lý trường hợp ảnh không đọc được
        if img1_bgr is None or img2_bgr is None:
            print(f"Warning: Could not read images for pair ({path1}, {path2}). Skipping.")
            # Trả về một cặp ảnh rỗng, sẽ được lọc sau
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), torch.tensor(-1.0)

        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
        
        img1_tensor = self.transform(img1_rgb)
        img2_tensor = self.transform(img2_rgb)
        
        return img1_tensor, img2_tensor, torch.tensor(label, dtype=torch.float32)

# --- Contrastive Loss (Không đổi) ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- Chương trình chính ---
def main():
    print(f"Bắt đầu quá trình fine-tuning trên thiết bị: {DEVICE}")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    backbone.to(DEVICE)

    print(f"Đang tải dữ liệu từ {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data_path_pairs = pickle.load(f)
    
    transform = T.Compose([
        T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dataset = SiamesePathDataset(data_path_pairs, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(backbone.parameters(), lr=LEARNING_RATE)
    
    print("Bắt đầu training loop...")
    backbone.train()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for img1, img2, label in pbar:
            # Lọc ra các sample bị lỗi
            valid_indices = label != -1.0
            if not torch.any(valid_indices): continue
            img1, img2, label = img1[valid_indices], img2[valid_indices], label[valid_indices]
            
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            output1 = backbone(img1)
            output2 = backbone(img2)
            loss = criterion(output1, output2, label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}")
    
    print("Training hoàn tất!")

    torch.save(backbone.state_dict(), MODEL_SAVE_PATH)
    print(f"Model đã được lưu tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
