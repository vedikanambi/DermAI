# DermAI Guard — Research Prototype

**Multimodal Dermatological Analysis & Ingredient Guard**  
*MSc Machine Learning Research Project — National College of Ireland (NCI)*

> ⚠️ **DISCLAIMER**: This system is a research prototype for educational purposes only. It is NOT a medical diagnostic device and must not be used for clinical decisions. Always consult a qualified dermatologist.

---

## System Architecture

```
ISIC 2019 Images → EfficientNet-B4 ─┐
                                      ├─ Deep Ensemble → Grad-CAM XAI
                  → ViT-B/16 ────────┘
Clinical Metadata Features → SVM ─────────┐
                           → Random Forest ┼─ Stacking Ensemble
                           → MLP ──────────┘
                                      ↓
                             Detected Condition
                                      ↓
Irish Product DB → ILP Optimiser (PuLP/CBC) → Optimal Basket
EWG Ingredient DB → Ingredient Guard (135 ingredients)
                                      ↓
Condition / Ingredient / Routine / Product docs → ChromaDB (RAG) → Azure OpenAI GPT-4o → AI Assistant
```

## ML Methods

| Category | Methods |
|---|---|
| Traditional ML | SVM (RBF kernel), Random Forest (500 trees), MLP, trained on clinical metadata features (age, sex, body site, lesion size, dermoscopic type, history, biopsy). A HOG + LBP pixel-feature script is included (`fast_train_real_features.py`) |
| Ensemble | Stacking (SVM + RF + MLP, Logistic Regression meta-learner), Deep Ensemble (EfficientNet + ViT) |
| Deep Learning | EfficientNet-B4 (ImageNet pretrained), ViT-B/16 (ImageNet pretrained). Fine-tuning code is in `training/deep_learning/train.py`; the prototype runs in demo mode with ImageNet weights unless fine-tuned checkpoints are provided |
| Anomaly Detection | Convolutional Autoencoder (reconstruction MSE) |
| XAI | Grad-CAM (CNN) |
| Optimisation | Binary Integer Linear Programming (PuLP/CBC) |
| RAG | ChromaDB + sentence-transformers + Azure OpenAI GPT-4o |

## Datasets

| Dataset | Source | Size |
|---|---|---|
| ISIC Archive 2019 | isic-archive.com (official API) | 25,331 dermoscopy images, 8 diagnostic classes (+ unknown) |
| EWG Skin Deep | ewg.org/skindeep (reference for hazard scores) | Used as a reference for the curated ingredient database |
| Irish Product Catalogue | Boots IE, LookFantastic IE, McCabes | 50 curated products |
| Ingredient Safety DB | EWG, EU Cosmetics Reg, SCCS | 135 ingredients with hazard data |
| RAG knowledge base | Curated condition, ingredient, routine and product documents (DermNet NZ used as a reference) | 86 documents (11 condition, 20 ingredient, 5 routine, 50 product) |

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
# Create a .env file with your Azure OpenAI credentials (see below)
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Environment Variables (.env)
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
MODEL_DIR=./models
CHROMA_DIR=./chroma_db
```

> Never commit your `.env` file. Add it to `.gitignore`.

## Features

### 🔬 Skin Condition Classifier
- Deep Ensemble: EfficientNet-B4 + ViT-B/16 (ImageNet weights, demo mode; not fine-tuned on ISIC in this prototype)
- Traditional ML: SVM, Random Forest, MLP and Stacking on clinical metadata features (macro F1: 0.38–0.41)
- Grad-CAM heatmap visualisation (explainable AI)
- Confidence calibration with HIGH/MEDIUM/LOW reliability labels

### 🛡️ Ingredient Guard
- 135-ingredient EWG/SCCS/EU safety database
- Per-ingredient: risk level, EWG score, category, explanation
- Pattern-matching for hazard classes (e.g. parabens, isothiazolinones, PEG compounds)
- Irish skincare product catalogue (50 products, INCI lists, EWG scores)

### 🛒 Product Recommender
- Binary Integer LP (PuLP/CBC) maximises aggregate safety score within budget
- Auto-populates from classifier result (detected condition, recommended categories)
- Budget relaxation fallback (up to +50%)
- Condition-specific skincare guidance per detected class

### 💬 AI Assistant (RAG Chatbot)
- ChromaDB vector store (condition docs + ingredient docs + routine docs + product summaries)
- sentence-transformers embeddings (all-MiniLM-L6-v2)
- Azure OpenAI GPT-4o with top-5 retrieved chunks as context
- Multi-turn conversation history
- Detected condition injected as RAG context

### 📊 Evaluation
- Model comparison (accuracy, macro F1, weighted F1, mean AUC, latency)
- Learning curves (SVM, Random Forest, MLP; 5 training-set sizes)
- Confusion matrices and per-class metrics
- Latency benchmarks (P50/P95/P99 ms + throughput)
- Error analysis: majority-class bias, calibration gap, minority-class failures
- ISIC dataset download via official API

## Evaluation Results

Test set: 3,800-sample stratified hold-out (traditional ML models, clinical metadata features).

| Model | Accuracy | Macro F1 | Mean AUC | Latency P50 |
|---|---|---|---|---|
| SVM | 0.604 | 0.381 | 0.932 | 0.21 ms |
| Random Forest | 0.640 | 0.412 | 0.928 | 268.69 ms |
| MLP | 0.720 | 0.404 | 0.934 | 0.19 ms |
| **Stacking Ensemble** | 0.639 | **0.413** | **0.942** | 108.52 ms |

The EfficientNet-B4 + ViT-B/16 ensemble runs with ImageNet weights only, so no ISIC performance metrics are reported for it.

## Project Structure

```
dermai/
├── backend/
│   ├── main.py              # FastAPI application (14 endpoints)
│   ├── product_db.py        # 50 products + 135-ingredient safety database
│   ├── optimiser.py         # PuLP binary ILP with budget relaxation
│   ├── rag_chatbot.py       # ChromaDB + sentence-transformers + Azure OpenAI RAG
│   ├── evaluation.py        # Evaluation metrics
│   ├── run_evaluation.py    # Runs evaluation for all models
│   ├── statistical_analysis.py
│   ├── train_all_models.py
│   ├── dataset_loader.py    # ISIC Archive API client
│   ├── training/
│   │   ├── deep_learning/
│   │   │   ├── classifier.py        # EfficientNet-B4 + ViT-B/16 inference + Grad-CAM
│   │   │   ├── train.py             # Deep learning training pipeline
│   │   │   └── anomaly_detector.py  # Convolutional Autoencoder
│   │   └── traditional_ml/          # SVM, RF, MLP, Stacking training scripts
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Classifier.jsx       # Image upload + ensemble inference
│       │   ├── IngredientGuard.jsx  # EWG ingredient checker + catalogue
│       │   ├── Recommender.jsx      # ILP optimiser + auto-populate
│       │   └── Chatbot.jsx          # RAG multi-turn chatbot
│       ├── components/index.jsx
│       ├── context/AppContext.jsx   # Global state (condition, results)
│       └── api.js
└── README.md
```
