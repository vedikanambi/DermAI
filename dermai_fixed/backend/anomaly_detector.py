"""
anomaly_detector.py
Convolutional Autoencoder for anomaly/rare condition detection.
"""

import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
MODEL_DIR = os.getenv("MODEL_DIR", "./models")


def anomaly_score(pil_image: Image.Image) -> float:
    """
    Compute anomaly score via reconstruction MSE.
    Returns float — threshold 0.05: above = potentially anomalous.
    Demo mode: uses pixel statistics proxy.
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T

        class ConvAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 3, padding=1), nn.Sigmoid(),
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConvAutoencoder().to(device)
        ckpt = os.path.join(MODEL_DIR, "autoencoder.pth")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed = model(tensor)
            mse = nn.functional.mse_loss(reconstructed, tensor).item()
        return round(mse, 4)
    except Exception as e:
        logger.warning(f"Autoencoder anomaly score failed: {e}. Using proxy.")
        arr = np.array(pil_image.resize((64, 64))).astype(np.float32) / 255.0
        return round(float(np.std(arr) * 0.25), 4)
