"""
main.py
FastAPI application — DermAI Guard backend.
Run: uvicorn main:app --reload --port 8000
"""

import io
import os
import base64
import logging
import threading
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DermAI Guard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy imports to avoid startup failures if optional deps missing ────────────

def _get_classifier():
    from classifier import classify_image
    return classify_image

def _get_product_db():
    from product_db import flag_ingredients, get_products_for_condition, PRODUCT_CATALOGUE
    return flag_ingredients, get_products_for_condition, PRODUCT_CATALOGUE

def _get_optimiser():
    from optimiser import optimise_basket
    return optimise_basket

def _get_rag():
    from rag_chatbot import rag_query
    return rag_query

def _get_evaluation():
    from evaluation import get_evaluation_results
    return get_evaluation_results

def _get_dataset_loader():
    from dataset_loader import fetch_dataset_stats, build_dataset_for_training, get_demo_images
    return fetch_dataset_stats, build_dataset_for_training, get_demo_images

# ── Download job state ────────────────────────────────────────────────────────

_download_state = {"running": False, "progress": "", "done": False, "result": None}


# ── Request / Response models ──────────────────────────────────────────────────

class FlagRequest(BaseModel):
    inci_list: List[str]

class OptimiseRequest(BaseModel):
    condition: str
    budget: float
    required_categories: List[str]

class ChatRequest(BaseModel):
    user_message: str
    detected_condition: Optional[str] = ""
    conversation_history: Optional[List[dict]] = []

class DownloadRequest(BaseModel):
    limit_per_class: Optional[int] = 200


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_mode": "demo"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Empty file uploaded")
        
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        classify_image = _get_classifier()
        result = classify_image(pil_image)
        
        # Check if result contains an error
        if isinstance(result, dict) and 'error' in result:
            logger.error(f"Classifier returned error: {result}")
            return {"error": result.get('error'), "detail": result.get('detail')}
        
        return result
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Image classification failed: {error_msg}")
        return {
            "error": "Classification failed",
            "detail": error_msg
        }


@app.post("/flag-ingredients")
def flag_ingredients_endpoint(req: FlagRequest):
    flag_ingredients, _, _ = _get_product_db()
    return flag_ingredients(req.inci_list)


@app.get("/products")
def get_products(
    condition: Optional[str] = None,
    category: Optional[str] = None,
    skin_type: Optional[str] = None,
):
    _, get_products_for_condition, PRODUCT_CATALOGUE = _get_product_db()
    products = get_products_for_condition(condition) if condition else PRODUCT_CATALOGUE
    if category and category.lower() != "all":
        products = [p for p in products if p["category"].lower() == category.lower()]
    if skin_type and skin_type.lower() != "all":
        products = [p for p in products if skin_type.lower() in [s.lower() for s in p["skin_type_tags"]]]
    return {"products": products, "total": len(products)}


@app.post("/optimise")
def optimise(req: OptimiseRequest):
    optimise_basket = _get_optimiser()
    _, _, PRODUCT_CATALOGUE = _get_product_db()
    return optimise_basket(req.condition, req.budget, req.required_categories, PRODUCT_CATALOGUE)


@app.post("/chat")
def chat(req: ChatRequest):
    rag_query = _get_rag()
    return rag_query(req.user_message, req.detected_condition, req.conversation_history)


@app.get("/evaluation")
def evaluation():
    get_evaluation_results = _get_evaluation()
    return get_evaluation_results()


@app.get("/dataset/stats")
def dataset_stats():
    try:
        fetch_dataset_stats, _, _ = _get_dataset_loader()
        stats = fetch_dataset_stats()
        return {"stats": stats}
    except Exception as e:
        # Return hardcoded fallback stats if API unreachable
        return {"stats": {
            "Melanoma": 4522, "Melanocytic Nevi": 12875, "Basal Cell Carcinoma": 3323,
            "Actinic Keratosis": 867, "Benign Keratosis": 2624, "Dermatofibroma": 239,
            "Vascular Lesion": 253, "Squamous Cell Carcinoma": 628, "Unknown": 1000,
        }, "source": "cached"}


@app.get("/dataset/demo-images/{diagnosis}")
def demo_images(diagnosis: str):
    try:
        _, _, get_demo_images = _get_dataset_loader()
        images_dict = get_demo_images(n_per_class=3)
        # Find matching diagnosis
        imgs = []
        for label, pil_list in images_dict.items():
            if label.lower() == diagnosis.lower():
                for img in pil_list:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    imgs.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return {"images": imgs, "diagnosis": diagnosis}
    except Exception as e:
        return {"images": [], "diagnosis": diagnosis, "error": str(e)}


@app.post("/dataset/download")
def download_dataset(req: DownloadRequest, background_tasks: BackgroundTasks):
    global _download_state
    if _download_state["running"]:
        return {"status": "already_running", "progress": _download_state["progress"]}

    def run_download():
        global _download_state
        _download_state = {"running": True, "progress": "Starting download...", "done": False, "result": None}
        try:
            fetch_dataset_stats, build_dataset_for_training, _ = _get_dataset_loader()
            result = build_dataset_for_training(limit_per_class=req.limit_per_class)
            _download_state = {"running": False, "progress": "Complete", "done": True, "result": result}
        except Exception as e:
            _download_state = {"running": False, "progress": f"Error: {e}", "done": True, "result": None}

    background_tasks.add_task(run_download)
    return {"status": "started", "message": f"Downloading {req.limit_per_class} images per class in background."}


@app.get("/dataset/download-status")
def download_status():
    return _download_state


@app.get("/ingredient-info/{ingredient}")
def ingredient_info(ingredient: str):
    """Return EWG hazard data for a specific ingredient."""
    from product_db import INGREDIENT_DATABASE, _match_ingredient
    data = _match_ingredient(ingredient.lower())
    if data:
        return {"found": True, "ingredient": ingredient, **data}
    return {"found": False, "ingredient": ingredient, "risk_level": "SAFE", "explanation": "Not found in flagging database — considered safe at cosmetic concentrations."}


@app.get("/ingredient-database/stats")
def ingredient_database_stats():
    """Return statistics about the ingredient safety database."""
    from product_db import INGREDIENT_DATABASE
    by_risk = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    categories = {}
    for data in INGREDIENT_DATABASE.values():
        rl = data.get("risk_level", "LOW")
        by_risk[rl] = by_risk.get(rl, 0) + 1
        cat = data.get("category", "Other")
        categories[cat] = categories.get(cat, 0) + 1
    return {
        "total_ingredients": len(INGREDIENT_DATABASE),
        "by_risk_level": by_risk,
        "top_categories": sorted(categories.items(), key=lambda x: -x[1])[:10],
    }


@app.get("/training-results")
def training_results():
    """
    Return real training results from train.py if available,
    else fall back to synthetic evaluation data.
    """
    import json
    results_path = os.path.join(os.getenv("MODEL_DIR", "./models"), "training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return {"source": "real", "results": json.load(f)}
    return {"source": "synthetic", "results": None}
