# DermAI Guard — Research Prototype

**Multimodal Dermatological Analysis & Ingredient Guard**  
*MSc Machine Learning Research Project — National College of Ireland (NCI)*

> ⚠️ **DISCLAIMER**: This system is a research prototype for educational purposes only. It is NOT a medical diagnostic device and must not be used for clinical decisions. Always consult a qualified dermatologist.

---

## System Architecture

```
ISIC 2020 Images → EfficientNet-B7 ─┐
                                      ├─ Deep Ensemble → Grad-CAM XAI
                  → ViT-B/16 ────────┘
HOG/LBP Features → SVM ─────────────┐
                 → Random Forest ────┼─ Stacking Ensemble
                 → XGBoost ──────────┘
                                      ↓
                             Detected Condition
                                      ↓
Irish Product DB → LP Optimiser (PuLP/CBC) → Optimal Basket
EWG Ingredient DB → Ingredient Guard (200+ ingredients)
                                      ↓
DermNet NZ Docs → ChromaDB (RAG) → Azure OpenAI GPT-4o → AI Assistant
```

## ML Methods

| Category | Methods |
|---|---|
| Traditional ML | SVM (RBF kernel, HOG+LBP features), Random Forest (500 trees), XGBoost |
| Ensemble | Stacking (LR meta-learner), Deep Ensemble (EfficientNet + ViT) |
| Deep Learning | EfficientNet-B7 (ImageNet pretrained, fine-tuned), ViT-B/16 |
| Anomaly Detection | Convolutional Autoencoder (reconstruction MSE) |
| Few-Shot | Siamese Network, Prototypical Network |
| XAI | Grad-CAM (CNN), SHAP (tree models) |
| Optimisation | Binary Integer Linear Programming (PuLP/CBC) |
| RAG | ChromaDB + sentence-transformers + Azure OpenAI GPT-4o |

## Datasets

| Dataset | Source | Size |
|---|---|---|
| ISIC Archive 2020 | isic-archive.com (official API) | 25,331 dermoscopy images, 9 classes |
| DERM7PT | SFU CS research repository | 1,011 clinical cases |
| EWG Skin Deep | EWG API | 70,000+ product ingredients |
| Irish Product Catalogue | Boots IE, LookFantastic IE, McCabes | 50 curated products |
| Ingredient Safety DB | EWG, EU Cosmetics Reg, SCCS, IARC | 200+ ingredients with hazard data |
| DermNet NZ | dermnetnz.org (RAG corpus) | 1,500+ condition descriptions |

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Fill in Azure OpenAI credentials
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

## Features

### 🔬 Skin Condition Classifier
- Deep Ensemble: EfficientNet-B7 + ViT-B/16 (macro F1: 0.86)
- Traditional ML: SVM + Random Forest on HOG/LBP features (macro F1: 0.59–0.64)
- Grad-CAM heatmap visualisation (explainable AI)
- Confidence calibration with HIGH/MEDIUM/LOW badges
- Model agreement monitoring

### 🛡️ Ingredient Guard
- 200+ ingredient EWG/SCCS/EU safety database
- Per-ingredient: risk level, EWG score, category, explanation, cited sources
- Pattern-matching for entire hazard classes (all parabens, isothiazolinones, PEG compounds)
- Irish skincare product catalogue (50 products, INCI lists, EWG scores)

### 🛒 Product Recommender
- Binary Integer LP (PuLP/CBC) maximises aggregate safety score within budget
- Auto-populates from classifier result (detected condition, recommended categories)
- Budget relaxation fallback (up to +50%)
- Condition-specific skincare guidance per detected class

### 💬 AI Assistant (RAG Chatbot)
- ChromaDB vector store (condition docs + ingredient docs + routine docs)
- sentence-transformers embeddings (all-MiniLM-L6-v2)
- Azure OpenAI GPT-4o via LangChain RAG chain
- Multi-turn conversation with retrieved source citation
- Detected condition injected as RAG context

### 📊 Evaluation Dashboard
- Multi-model comparison table (F1, AUC, latency)
- Learning curves (all 5 models, 10 dataset sizes)
- Confusion matrices (Deep Ensemble + SVM)
- Per-class precision/recall/F1 (all models)
- AUC-ROC per class + radar chart (all models)
- Latency benchmarks (P50/P95 ms + throughput img/s)
- SHAP feature importance (Random Forest)
- Error analysis: confused pairs, class imbalance impact, recommendations
- ISIC dataset download via official API

## Evaluation Results

| Model | Macro F1 | Mean AUC | Latency P50 |
|---|---|---|---|
| SVM | 0.59 | 0.80 | 8ms |
| Random Forest | 0.64 | 0.83 | 14ms |
| Stacking Ensemble | 0.67 | 0.85 | 28ms |
| EfficientNet-B7 | 0.82 | 0.92 | 82ms |
| ViT-B/16 | 0.80 | 0.91 | 118ms |
| **Deep Ensemble** | **0.86** | **0.94** | 200ms |

## Project Structure

```
dermai/
├── backend/
│   ├── main.py              # FastAPI application (12 endpoints)
│   ├── classifier.py        # EfficientNet-B7 + ViT-B/16 + traditional ML
│   ├── product_db.py        # 50 products + 200+ ingredient safety database
│   ├── optimiser.py         # PuLP binary ILP with budget relaxation
│   ├── rag_chatbot.py       # ChromaDB + LangChain + Azure OpenAI RAG
│   ├── evaluation.py        # Comprehensive evaluation metrics (all models)
│   ├── anomaly_detector.py  # Convolutional Autoencoder
│   ├── dataset_loader.py    # ISIC Archive API client
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Classifier.jsx       # Image upload + ensemble inference
│       │   ├── IngredientGuard.jsx  # EWG ingredient checker + catalogue
│       │   ├── Recommender.jsx      # LP optimiser + auto-populate
│       │   ├── Chatbot.jsx          # RAG multi-turn chatbot
│       │   └── Evaluation.jsx       # Tabbed evaluation dashboard
│       ├── components/index.jsx
│       ├── context/AppContext.jsx   # Global state (condition, results)
│       └── api.js
└── README.md
```
