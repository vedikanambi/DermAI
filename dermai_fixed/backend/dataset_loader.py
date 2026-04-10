"""
dataset_loader.py
Handles all interaction with the ISIC Archive public API.
Base URL: https://api.isic-archive.com/api/v2
No API key required.
"""

import os
import time
import logging
import requests
from io import BytesIO
from PIL import Image
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ISIC_API_BASE = os.getenv("ISIC_API_BASE", "https://api.isic-archive.com/api/v2")

CLASS_MAP = {
    "melanoma": "Melanoma",
    "melanocytic nevi": "Melanocytic Nevi",
    "basal cell carcinoma": "Basal Cell Carcinoma",
    "actinic keratosis": "Actinic Keratosis",
    "benign keratosis": "Benign Keratosis",
    "dermatofibroma": "Dermatofibroma",
    "vascular lesion": "Vascular Lesion",
    "squamous cell carcinoma": "Squamous Cell Carcinoma",
    "unknown": "Unknown",
}

REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def _get(url: str, params: dict = None, retries: int = 2) -> Optional[dict]:
    for attempt in range(retries):
        try:
            logger.info(f"GET {url} params={params}")
            resp = requests.get(url, params=params, timeout=15)
            logger.info(f"Response status: {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return None


def fetch_image_metadata(diagnosis: str, limit: int = 50, offset: int = 0) -> list:
    """Fetch image metadata for a given diagnosis label from ISIC API."""
    data = _get(f"{ISIC_API_BASE}/images/", params={
        "diagnosis": diagnosis,
        "limit": limit,
        "offset": offset,
        "fields": "isic_id,files,metadata"
    })
    if data and "results" in data:
        return data["results"]
    return []


def download_image_pil(image_url: str) -> Optional[Image.Image]:
    """Fetch image binary from URL and return as PIL Image."""
    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to download image from {image_url}: {e}")
        return None


def build_dataset_for_training(limit_per_class: int = 200, save_dir: str = "./data") -> dict:
    """
    Download real ISIC images for all 9 classes.
    Saves to save_dir/{class_name}/{isic_id}.jpg
    Returns summary dict { class_name: image_count }
    """
    summary = {}
    for diagnosis, label in CLASS_MAP.items():
        class_dir = os.path.join(save_dir, label.replace(" ", "_"))
        os.makedirs(class_dir, exist_ok=True)
        logger.info(f"Fetching metadata for {diagnosis}...")
        items = fetch_image_metadata(diagnosis, limit=limit_per_class)
        count = 0
        for item in items:
            isic_id = item.get("isic_id", "unknown")
            save_path = os.path.join(class_dir, f"{isic_id}.jpg")
            if os.path.exists(save_path):
                count += 1
                continue
            try:
                url = item["files"]["full"]["url"]
            except (KeyError, TypeError):
                try:
                    url = item["files"]["thumbnail_256"]["url"]
                except (KeyError, TypeError):
                    continue
            img = download_image_pil(url)
            if img:
                img.save(save_path)
                count += 1
        summary[label] = count
        logger.info(f"{label}: saved {count} images")
    return summary


def get_demo_images(n_per_class: int = 3) -> dict:
    """
    Returns n_per_class thumbnail PIL images per class fetched live from ISIC API.
    Returns dict: { class_name: [PIL images] }
    """
    result = {}
    for diagnosis, label in CLASS_MAP.items():
        images = []
        items = fetch_image_metadata(diagnosis, limit=n_per_class)
        for item in items[:n_per_class]:
            try:
                url = item["files"]["thumbnail_256"]["url"]
                img = download_image_pil(url)
                if img:
                    images.append(img)
            except (KeyError, TypeError):
                pass
        result[label] = images
    return result


def get_single_demo_image(diagnosis: str) -> Optional[Image.Image]:
    """Fetch one real thumbnail for a given diagnosis."""
    items = fetch_image_metadata(diagnosis, limit=1)
    if not items:
        return None
    try:
        url = items[0]["files"]["thumbnail_256"]["url"]
        return download_image_pil(url)
    except (KeyError, TypeError):
        return None


def fetch_dataset_stats() -> dict:
    """
    Returns total available image count per class from ISIC API.
    """
    stats = {}
    for diagnosis, label in CLASS_MAP.items():
        data = _get(f"{ISIC_API_BASE}/images/", params={"diagnosis": diagnosis, "limit": 1})
        if data:
            stats[label] = data.get("count", 0)
        else:
            stats[label] = 0
    return stats
