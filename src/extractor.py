# src/extractor.py
import cv2
import torch
import torchreid
import torchvision.transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeatureExtractor:
    def __init__(self, model_name='osnet_x1_0'):
        print(f"Đang tải model Re-ID: {model_name}...")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=True
        )
        self.model.to(DEVICE)
        self.model.eval()
        print("Tải model Re-ID thành công!")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(DEVICE)
        feature = self.model(image_tensor)
        return torch.nn.functional.normalize(feature, p=2, dim=1)
