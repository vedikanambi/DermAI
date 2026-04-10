"""
rag_chatbot.py
ChromaDB vector store + Azure OpenAI gpt-4o RAG chatbot.
Auto-populates on first run. Multi-turn conversation support.
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Knowledge documents ───────────────────────────────────────────────────────

CONDITION_DOCS = [
    ("Melanoma", "Melanoma is the most serious form of skin cancer. It develops from melanocytes. Key signs include asymmetry, irregular border, multiple colours, diameter >6mm. Skincare advice: avoid sun exposure, use SPF50+ mineral sunscreen daily, avoid harsh exfoliants, use gentle fragrance-free formulations. Recommended ingredients: zinc oxide, titanium dioxide, ceramides, niacinamide. Avoid: alcohol denat., fragrances, retinol (may increase photosensitivity)."),
    ("Melanocytic Nevi", "Melanocytic nevi are common benign moles. They require monitoring for changes. Skincare: gentle daily SPF50+ is essential. Avoid picking or irritating moles. Use fragrance-free products. Recommended ingredients: ceramides, hyaluronic acid, niacinamide, SPF. Avoid: salicylic acid directly on moles, alcohol-based toners."),
    ("Basal Cell Carcinoma", "Basal cell carcinoma (BCC) is the most common skin cancer. Often appears as a pearly bump or flat lesion. Requires medical treatment. Skincare support: strict daily SPF50+, barrier repair with ceramides, gentle non-irritating cleansers. Avoid alcohol denat., fragrance, AHAs/BHAs on affected areas."),
    ("Actinic Keratosis", "Actinic keratoses are rough scaly patches caused by UV damage. Pre-cancerous lesions. Skincare: high SPF is critical. Use gentle non-abrasive cleansers, ceramide-rich moisturisers. Avoid: retinol, strong AHAs, fragrance, alcohol. Recommended: niacinamide, hyaluronic acid, peptides."),
    ("Benign Keratosis", "Benign keratoses (seborrhoeic keratoses) are non-cancerous growths. They are harmless but can be cosmetically bothersome. Skincare: gentle moisturisation, SPF protection. Avoid picking. Recommended: ceramides, shea butter, glycerin. Avoid: harsh exfoliants, fragrance."),
    ("Dermatofibroma", "Dermatofibromas are benign skin growths, usually on legs. They are firm, slightly raised, brownish. Skincare: general skin health maintenance. SPF daily, gentle moisturiser. No specific restrictions. Recommended: any well-tolerated fragrance-free moisturiser."),
    ("Vascular Lesion", "Vascular lesions include spider veins, port wine stains, haemangiomas. Skincare: avoid heat and strong exfoliants. Use gentle SPF, soothing anti-redness products. Recommended: azelaic acid, green tea extract, ceramides, niacinamide. Avoid: fragrance, alcohol, aggressive physical exfoliation."),
    ("Squamous Cell Carcinoma", "Squamous cell carcinoma (SCC) is the second most common skin cancer. Appears as a firm red nodule or flat lesion. Medical treatment required. Skincare support: strict SPF50+ daily, gentle barrier repair. Avoid: fragrance, alcohol denat., harsh actives. Recommended: ceramides, peptides, niacinamide."),
    ("Eczema", "Eczema (atopic dermatitis) causes dry, itchy, inflamed skin. Skincare: moisturise frequently with ceramide-rich creams, use gentle fragrance-free cleansers. Recommended: ceramide NP, ceramide AP, glycerin, colloidal oatmeal, shea butter, panthenol. Avoid: sodium lauryl sulfate, fragrance, alcohol denat., parabens."),
    ("Rosacea", "Rosacea causes redness, visible blood vessels, and skin sensitivity. Skincare: gentle non-abrasive routine. Recommended: azelaic acid, niacinamide, centella asiatica, ceramides, mineral SPF. Avoid: fragrance, alcohol, strong AHAs, physical scrubs, hot products."),
    ("Acne", "Acne vulgaris involves blocked pores, inflammation, bacteria. Skincare: use non-comedogenic oil-free products. Recommended: salicylic acid (BHA), niacinamide, zinc PCA, benzoyl peroxide, azelaic acid. Avoid: mineral oil, heavy occlusives, fragrance, alcohol denat. (drying then rebounds to more oil)."),
]

INGREDIENT_DOCS = [
    ("Niacinamide", "Niacinamide (Vitamin B3) EWG score 1. Brightens skin, reduces hyperpigmentation, minimises pores, strengthens skin barrier, reduces inflammation. Safe for all skin types including sensitive. Works well with ceramides, hyaluronic acid, peptides. Widely recommended for acne, rosacea, eczema."),
    ("Ceramide NP", "Ceramide NP EWG score 1. Key skin barrier lipid. Restores and maintains skin barrier integrity. Essential for eczema, psoriasis, and dry skin. Works synergistically with cholesterol and fatty acids. Very safe, recommended for all skin types."),
    ("Hyaluronic Acid", "Hyaluronic acid EWG score 1. Humectant that attracts and retains moisture. Can hold up to 1000x its weight in water. Safe for all skin types. Best applied to damp skin and sealed with a moisturiser. Low and high molecular weight forms provide surface and deeper hydration."),
    ("Glycerin", "Glycerin EWG score 1. Humectant, attracts moisture from environment. Strengthens skin barrier. Safe for all skin types. Used in most moisturisers and cleansers as a base ingredient."),
    ("Salicylic Acid", "Salicylic acid (BHA) EWG score 3. Oil-soluble exfoliant that penetrates pores. Effective for acne and blackheads. Use 0.5–2% concentration. Avoid on broken or sensitive skin. Can cause dryness at high concentrations. Do not use on moles or lesions."),
    ("Retinol", "Retinol EWG score 5. Vitamin A derivative. Accelerates cell turnover, reduces fine lines, treats acne. Can cause irritation, dryness, and purging initially. Start slowly (0.1%). Avoid in pregnancy. Increases photosensitivity — always use SPF. Not recommended for rosacea or eczema in early stages."),
    ("Zinc Oxide", "Zinc oxide EWG score 2. Physical UV filter. Broad-spectrum SPF. Anti-inflammatory. Safe for sensitive skin, rosacea, acne. Preferred over chemical filters for reactive skin. Also mild antimicrobial."),
    ("Titanium Dioxide", "Titanium dioxide EWG score 2. Physical UV filter. Broad-spectrum SPF. Very safe. Reef-safe. Preferred for sensitive skin. Can leave white cast on deeper skin tones."),
    ("Alcohol Denat.", "Alcohol Denat. (denatured alcohol) EWG score 4-6. Strips the skin barrier, disrupts lipid layer, causes dehydration. Short-term mattifying effect but long-term barrier damage. High risk for sensitive, dry, eczema-prone skin. Avoid in most skincare unless in low concentrations in a well-formulated serum."),
    ("Parfum/Fragrance", "Parfum or Fragrance EWG score 8. Collective term for hundreds of undisclosed chemicals. Leading cause of contact dermatitis and skin sensitisation. No skincare benefit. Avoid especially for sensitive, rosacea, eczema, or reactive skin types."),
    ("Methylparaben", "Methylparaben EWG score 4. Common preservative. Endocrine disruption concern at high doses. Considered safe in low concentrations by EU. Avoid in leave-on products for sensitive skin as precaution."),
    ("DMDM Hydantoin", "DMDM Hydantoin EWG score 7. Formaldehyde-releasing preservative. Prolonged exposure can cause sensitisation. Considered a potential carcinogen at high doses. Recommend avoiding, especially for daily-use products."),
    ("Panthenol", "Panthenol (Pro-Vitamin B5) EWG score 1. Skin-conditioning agent. Improves wound healing, hydration, and skin barrier. Anti-inflammatory. Safe for all skin types including sensitive. Widely used in eczema and rosacea-friendly formulations."),
    ("Squalane", "Squalane EWG score 1. Lightweight emollient. Non-comedogenic. Derived from sugarcane or olives. Excellent for dry, sensitive, and acne-prone skin. Antioxidant properties. Very safe."),
    ("Tocopherol", "Tocopherol (Vitamin E) EWG score 2. Antioxidant. Supports skin barrier, reduces UV-induced damage. Works synergistically with Vitamin C. Generally safe. Can be comedogenic for some acne-prone skin at high concentrations."),
    ("Sodium Lauryl Sulfate", "Sodium Lauryl Sulfate (SLS) EWG score 4. Strong surfactant and foaming agent. Disrupts skin barrier lipids. Common irritant, especially for sensitive, eczema, and rosacea-prone skin. Avoid in cleansers for sensitive skin. Prefer gentler alternatives like cocamidopropyl betaine."),
    ("Azelaic Acid", "Azelaic acid EWG score 1. Naturally occurring acid. Multi-benefit: reduces hyperpigmentation, treats acne, calms rosacea redness, mild exfoliation. Safe for pregnancy. Anti-inflammatory. Recommended for sensitive and rosacea-prone skin."),
    ("Allantoin", "Allantoin EWG score 1. Soothing and healing ingredient. Promotes skin cell regeneration. Anti-irritant. Safe for all skin types. Used in sensitive skin, eczema, and post-procedure formulations."),
    ("Colloidal Oatmeal", "Colloidal oatmeal EWG score 1. FDA-approved skin protectant. Reduces itching and irritation. Strengthens skin barrier. Specifically recommended for eczema and psoriasis. Very safe, excellent tolerability."),
    ("Mineral Oil", "Mineral oil (paraffinum liquidum) EWG score 1-2. Highly refined occlusive. Locks in moisture effectively. Very safe — cosmetic grade highly purified. Low comedogenicity in reality despite reputation. Good for very dry skin and eczema. Some acne-prone individuals prefer to avoid."),
]

ROUTINE_DOCS = [
    ("Morning Skincare Routine", "Optimal morning routine: 1) Gentle cleanser or water rinse. 2) Toner (optional, hydrating not astringent). 3) Serum — Vitamin C or niacinamide. 4) Moisturiser appropriate for your skin type. 5) SPF50+ — most critical step. Never skip SPF in the morning. Apply SPF as final step."),
    ("Evening Skincare Routine", "Optimal evening routine: 1) Double cleanse — oil cleanser then water-based cleanser to remove SPF and makeup. 2) Exfoliant 2-3x per week (AHA/BHA). 3) Treatment serum — retinol, azelaic acid, or niacinamide. 4) Moisturiser — richer than morning. 5) Facial oil or occlusive (optional, for dry skin)."),
    ("Acne Skincare Routine", "Acne-focused routine: AM — gentle gel cleanser, niacinamide serum, lightweight oil-free moisturiser, SPF50 non-comedogenic. PM — salicylic acid cleanser, BHA toner, azelaic acid or benzoyl peroxide spot treatment, oil-free moisturiser. Key ingredients: salicylic acid, niacinamide, zinc PCA, azelaic acid. Avoid: heavy occlusives, coconut oil, fragrance, alcohol."),
    ("Sensitive and Eczema Routine", "Sensitive/eczema routine: Minimalist approach. AM — gentle cream cleanser, ceramide moisturiser, mineral SPF50. PM — gentle cleanser (micellar or cream), heavy ceramide cream or emollient. Key: fragrance-free everything, SLS-free, paraben-free, alcohol-free. Patch test all new products. Add products one at a time."),
    ("Anti-Ageing Routine", "Anti-ageing routine: AM — gentle cleanser, Vitamin C serum, niacinamide, SPF50 (most important anti-ageing step). PM — cleanser, retinol (start 0.1%, build up), peptide serum, rich moisturiser. Key ingredients: retinol, peptides, Vitamin C, niacinamide, hyaluronic acid. Weekly: gentle AHA exfoliation."),
]


# ── ChromaDB setup ────────────────────────────────────────────────────────────

_collection = None


def _init_vector_store():
    global _collection
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=CHROMA_DIR)

        try:
            _collection = client.get_collection("skincare_knowledge")
            logger.info(f"Loaded existing ChromaDB collection ({_collection.count()} docs).")
            return
        except Exception:
            pass

        logger.info("Initialising ChromaDB — populating knowledge base...")
        _collection = client.create_collection("skincare_knowledge")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        docs, ids, metas = [], [], []

        for name, text in CONDITION_DOCS:
            docs.append(text)
            ids.append(f"condition_{name.replace(' ','_')}")
            metas.append({"type": "condition", "name": name})

        for name, text in INGREDIENT_DOCS:
            docs.append(text)
            ids.append(f"ingredient_{name.replace(' ','_').replace('/','_')}")
            metas.append({"type": "ingredient", "name": name})

        for name, text in ROUTINE_DOCS:
            docs.append(text)
            ids.append(f"routine_{name.replace(' ','_')}")
            metas.append({"type": "routine", "name": name})

        # Add product summaries
        from product_db import PRODUCT_CATALOGUE
        for p in PRODUCT_CATALOGUE:
            text = f"{p['name']} by {p['brand']}. Category: {p['category']}. Price: €{p['price_eur']}. EWG score: {p['ewg_score']}. Skin types: {', '.join(p['skin_type_tags'])}. Key ingredients: {', '.join(p['inci_ingredients'][:8])}."
            docs.append(text)
            ids.append(f"product_{p['id']}")
            metas.append({"type": "product", "name": p["name"]})

        embeddings = model.encode(docs).tolist()
        _collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
        logger.info(f"ChromaDB populated with {len(docs)} documents.")

    except Exception as e:
        logger.error(f"ChromaDB init failed: {e}")
        _collection = None


_init_vector_store()


def _retrieve_chunks(query: str, n: int = 5) -> list:
    if _collection is None:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([query]).tolist()
        results = _collection.query(query_embeddings=query_embedding, n_results=n)
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        return []


def _get_openai_client():
    """Return (client, model_name, error_message).

    Supports Azure OpenAI (preferred) and falls back to OpenAI if needed.
    """
    try:
        import openai
    except Exception as e:
        return None, None, f"OpenAI package not installed: {e}"

    # Prefer Azure OpenAI if endpoint/key/deployment are configured
    if AZURE_ENDPOINT and AZURE_API_KEY and AZURE_OPENAI_DEPLOYMENT:
        try:
            client = openai.AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            )
            return client, AZURE_OPENAI_DEPLOYMENT, None
        except Exception as e:
            logger.warning("Azure OpenAI client init failed: %s", e)

    # Fallback to standard OpenAI
    if AZURE_API_KEY:
        try:
            client = openai.OpenAI(api_key=AZURE_API_KEY)
            return client, OPENAI_MODEL, None
        except Exception as e:
            return None, None, f"OpenAI client init failed: {e}"

    return None, None, "No valid OpenAI credentials found."


def rag_query(user_message: str, detected_condition: str, conversation_history: List[Dict]) -> dict:
    chunks = _retrieve_chunks(user_message)

    context_str = "\n\n".join(
        [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
    ) if chunks else "No specific context retrieved."

    system_prompt = (
        "You are DermAI Guard — a skincare assistant powered by a dermatology knowledge base. "
        "You provide ingredient-aware, evidence-informed skincare guidance tailored to the user's detected skin condition. "
        "You are NOT a doctor or dermatologist. "
        "You must always end EVERY response with: "
        "'⚠️ Please consult a qualified dermatologist before making any skincare or medical decisions.' "
        f"\n\nUser's detected skin condition: {detected_condition or 'Not yet detected'}.\n\n"
        f"Relevant knowledge base context:\n{context_str}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    client, model, client_err = _get_openai_client()
    if client is None:
        assistant_message = (
            "I'm sorry, I couldn't connect to the AI service right now. "
            f"Error: {client_err}. "
            "⚠️ Please consult a qualified dermatologist before making any skincare or medical decisions."
        )
        return {
            "assistant_message": assistant_message,
            "retrieved_chunks": chunks,
            "sources_count": len(chunks),
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
            temperature=0.4,
        )
        assistant_message = response.choices[0].message.content
    except Exception as e:
        logger.error("AI completion failed: %s", e)
        assistant_message = (
            "I'm sorry, I couldn't connect to the AI service right now. "
            f"Error: {str(e)[:200]}. "
            "⚠️ Please consult a qualified dermatologist before making any skincare or medical decisions."
        )

    return {
        "assistant_message": assistant_message,
        "retrieved_chunks": chunks,
        "sources_count": len(chunks),
    }
 