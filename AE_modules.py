import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# --- for loading dataset ---
class ColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        # filtering images
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # transformation as target (RGB)
        self.target_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # transformation as input (grayscale)
        self.input_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img_rgb = Image.open(img_path).convert("RGB")

        input_gray = self.input_transform(img_rgb)  # grayscale
        target_rgb = self.target_transform(img_rgb)  # RGB

        return input_gray, target_rgb


# --- AutoEncoder module for 128*128 ---
class ColorizationAutoencoder128(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder128, self).__init__()

        # Encoder: 128 to 4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        # Decoder: reconstruct from 4 to 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 3x128x128
            nn.Sigmoid()  # output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x