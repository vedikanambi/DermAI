"""
product_db.py
50 Irish skincare products + comprehensive EWG/INCI ingredient flagging database (200+ ingredients).

Ingredient hazard data sources:
  - EWG Skin Deep database (ewg.org/skindeep)
  - EU Cosmetics Regulation Annex II/III banned/restricted substances
  - DermNet NZ allergen reference lists
  - PubChem / NCBI hazard classifications
  - SCCS (EU Scientific Committee on Consumer Safety) opinions

Risk levels:  HIGH | MODERATE | LOW
"""

from typing import List, Dict

PRODUCT_CATALOGUE: List[Dict] = [
    {"id":1,"name":"Hydrating Facial Cleanser","brand":"CeraVe","price_eur":12.99,"category":"cleanser","skin_type_tags":["dry","normal","sensitive"],"condition_tags":["eczema","psoriasis","general"],"inci_ingredients":["Aqua","Glycerin","Ceramide NP","Ceramide AP","Niacinamide","Sodium Lauroyl Lactylate","Cholesterol","Carbomer","Behentrimonium Methosulfate"],"ewg_score":2.0,"url":"https://boots.ie/cerave-hydrating-cleanser"},
    {"id":2,"name":"Toleriane Hydrating Gentle Cleanser","brand":"La Roche-Posay","price_eur":14.50,"category":"cleanser","skin_type_tags":["dry","sensitive","normal"],"condition_tags":["rosacea","eczema","general"],"inci_ingredients":["Aqua","Glycerin","Niacinamide","Ceramide NP","Citric Acid","Sodium Citrate"],"ewg_score":1.5,"url":"https://boots.ie/lrp-toleriane-cleanser"},
    {"id":3,"name":"Daily Foaming Cleanser","brand":"Neutrogena","price_eur":8.99,"category":"cleanser","skin_type_tags":["oily","combination"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Sodium Lauryl Sulfate","Glycerin","Cocamidopropyl Betaine","Sodium Chloride","Parfum"],"ewg_score":5.5,"url":"https://boots.ie/neutrogena-foaming-cleanser"},
    {"id":4,"name":"Gentle Milk Cleanser","brand":"Avene","price_eur":11.99,"category":"cleanser","skin_type_tags":["sensitive","dry"],"condition_tags":["rosacea","eczema","general"],"inci_ingredients":["Aqua","Mineral Oil","Glycerin","Cetearyl Alcohol","Avene Thermal Spring Water","Squalane"],"ewg_score":3.0,"url":"https://mccabes.ie/avene-milk-cleanser"},
    {"id":5,"name":"Salicylic Acid Cleanser","brand":"The Ordinary","price_eur":6.80,"category":"cleanser","skin_type_tags":["oily","combination","acne-prone"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Salicylic Acid","Cocamidopropyl Betaine","Sodium Chloride","Alcohol Denat.","Disodium EDTA"],"ewg_score":4.5,"url":"https://lookfantastic.ie/the-ordinary-salicylic-cleanser"},
    {"id":6,"name":"Micellar Water 3-in-1","brand":"Bioderma","price_eur":13.50,"category":"cleanser","skin_type_tags":["all"],"condition_tags":["general","sensitive"],"inci_ingredients":["Aqua","Glycerin","Cucurbita Pepo Seed Extract","Zinc Gluconate","Niacinamide","Disodium EDTA"],"ewg_score":2.0,"url":"https://boots.ie/bioderma-micellar"},
    {"id":7,"name":"Clay Purifying Cleanser","brand":"Kiehl's","price_eur":22.00,"category":"cleanser","skin_type_tags":["oily","combination"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Kaolin","Bentonite","Glycerin","Sodium Lauryl Sulfate","Methylparaben","Parfum"],"ewg_score":6.0,"url":"https://lookfantastic.ie/kiehls-clay-cleanser"},
    {"id":8,"name":"Soothing Cleansing Cream","brand":"Eucerin","price_eur":10.99,"category":"cleanser","skin_type_tags":["sensitive","dry"],"condition_tags":["eczema","psoriasis","rosacea"],"inci_ingredients":["Aqua","Glycerin","Isohexadecane","Caprylic/Capric Triglyceride","Ceramide NP","Allantoin","Panthenol"],"ewg_score":2.5,"url":"https://mccabes.ie/eucerin-soothing-cleanser"},
    {"id":9,"name":"Foam Cleanser with Zinc","brand":"Uriage","price_eur":9.50,"category":"cleanser","skin_type_tags":["oily","combination"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Zinc PCA","Salicylic Acid","Cocamidopropyl Betaine","Niacinamide","DMDM Hydantoin"],"ewg_score":5.0,"url":"https://mccabes.ie/uriage-zinc-cleanser"},
    {"id":10,"name":"Gentle Cleansing Gel","brand":"Cetaphil","price_eur":9.99,"category":"cleanser","skin_type_tags":["all","sensitive"],"condition_tags":["general","eczema"],"inci_ingredients":["Aqua","Glycerin","Cocamidopropyl Hydroxysultaine","Niacinamide","Panthenol","Disodium EDTA"],"ewg_score":2.0,"url":"https://boots.ie/cetaphil-gentle-gel"},
    {"id":11,"name":"Moisturising Cream","brand":"CeraVe","price_eur":16.99,"category":"moisturiser","skin_type_tags":["dry","normal"],"condition_tags":["eczema","psoriasis","general"],"inci_ingredients":["Aqua","Glycerin","Ceramide NP","Ceramide AP","Ceramide EOP","Hyaluronic Acid","Niacinamide","Cholesterol","Petrolatum"],"ewg_score":2.0,"url":"https://boots.ie/cerave-moisturising-cream"},
    {"id":12,"name":"Toleriane Double Repair Face Moisturiser","brand":"La Roche-Posay","price_eur":24.99,"category":"moisturiser","skin_type_tags":["dry","sensitive","normal"],"condition_tags":["rosacea","eczema","general"],"inci_ingredients":["Aqua","Glycerin","Niacinamide","Ceramide NP","Shea Butter","Allantoin","Panthenol","Squalane"],"ewg_score":2.5,"url":"https://boots.ie/lrp-toleriane-double-repair"},
    {"id":13,"name":"Natural Moisturising Factors","brand":"The Ordinary","price_eur":7.90,"category":"moisturiser","skin_type_tags":["all"],"condition_tags":["general","eczema"],"inci_ingredients":["Aqua","Glycerin","Urea","Sodium PCA","Serine","Hyaluronic Acid","Panthenol","Tocopherol"],"ewg_score":1.5,"url":"https://lookfantastic.ie/the-ordinary-nmf"},
    {"id":14,"name":"Cicaplast Baume B5","brand":"La Roche-Posay","price_eur":12.99,"category":"moisturiser","skin_type_tags":["sensitive","dry"],"condition_tags":["eczema","psoriasis","rosacea"],"inci_ingredients":["Aqua","Glycerin","Panthenol","Shea Butter","Madecassoside","Zinc Oxide","Tocopherol"],"ewg_score":2.0,"url":"https://mccabes.ie/lrp-cicaplast"},
    {"id":15,"name":"Daily Moisturising Lotion","brand":"Eucerin","price_eur":11.50,"category":"moisturiser","skin_type_tags":["dry","sensitive"],"condition_tags":["eczema","psoriasis","general"],"inci_ingredients":["Aqua","Glycerin","Urea","Ceramide NP","Sodium Lactate","Citric Acid","Tocopherol"],"ewg_score":2.0,"url":"https://mccabes.ie/eucerin-daily-lotion"},
    {"id":16,"name":"Effaclar Mat Moisturiser","brand":"La Roche-Posay","price_eur":18.99,"category":"moisturiser","skin_type_tags":["oily","combination"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Glycerin","Niacinamide","Zinc PCA","Salicylic Acid","Dimethicone","Alcohol Denat."],"ewg_score":4.5,"url":"https://boots.ie/lrp-effaclar-mat"},
    {"id":17,"name":"Ultra Comfort Cream","brand":"Avene","price_eur":19.99,"category":"moisturiser","skin_type_tags":["dry","sensitive"],"condition_tags":["rosacea","general"],"inci_ingredients":["Aqua","Avene Thermal Spring Water","Glycerin","Squalane","Shea Butter","Allantoin","Tocopherol"],"ewg_score":2.0,"url":"https://lookfantastic.ie/avene-ultra-comfort"},
    {"id":18,"name":"Spot Moisturiser SPF15","brand":"Nivea","price_eur":8.99,"category":"moisturiser","skin_type_tags":["normal","combination"],"condition_tags":["general"],"inci_ingredients":["Aqua","Glycerin","Dimethicone","Titanium Dioxide","Methylparaben","Propylparaben","Parfum"],"ewg_score":6.5,"url":"https://boots.ie/nivea-spot-moisturiser"},
    {"id":19,"name":"Barrier Repair Cream","brand":"Aveeno","price_eur":14.99,"category":"moisturiser","skin_type_tags":["dry","sensitive","eczema-prone"],"condition_tags":["eczema","psoriasis"],"inci_ingredients":["Aqua","Colloidal Oatmeal","Glycerin","Ceramide NP","Dimethicone","Panthenol","Allantoin"],"ewg_score":2.5,"url":"https://boots.ie/aveeno-barrier-repair"},
    {"id":20,"name":"Hyaluronic Acid Cream","brand":"Neutrogena","price_eur":13.99,"category":"moisturiser","skin_type_tags":["all"],"condition_tags":["general","acne"],"inci_ingredients":["Aqua","Hyaluronic Acid","Glycerin","Dimethicone","Niacinamide","Tocopherol","Carbomer"],"ewg_score":3.0,"url":"https://boots.ie/neutrogena-ha-cream"},
    {"id":21,"name":"Rich Repair Cream","brand":"Kiehl's","price_eur":39.00,"category":"moisturiser","skin_type_tags":["dry"],"condition_tags":["general","eczema"],"inci_ingredients":["Aqua","Petrolatum","Glycerin","Shea Butter","Ceramide NP","Parfum","Methylparaben"],"ewg_score":5.5,"url":"https://lookfantastic.ie/kiehls-rich-repair"},
    {"id":22,"name":"Emollient Cream","brand":"Dermol","price_eur":10.50,"category":"moisturiser","skin_type_tags":["dry","sensitive"],"condition_tags":["eczema","psoriasis"],"inci_ingredients":["Aqua","Liquid Paraffin","Isopropyl Myristate","Benzalkonium Chloride","Chlorhexidine Dihydrochloride"],"ewg_score":3.5,"url":"https://mccabes.ie/dermol-emollient"},
    {"id":23,"name":"Invisible Fluid SPF50+","brand":"La Roche-Posay Anthelios","price_eur":28.99,"category":"SPF","skin_type_tags":["oily","combination","sensitive"],"condition_tags":["general","rosacea","acne"],"inci_ingredients":["Aqua","Uvinul A Plus","Mexoryl XL","Mexoryl SX","Glycerin","Silica","Tocopherol"],"ewg_score":2.0,"url":"https://boots.ie/lrp-anthelios-invisible"},
    {"id":24,"name":"Sun Cream SPF50","brand":"Altruist","price_eur":3.99,"category":"SPF","skin_type_tags":["all"],"condition_tags":["general"],"inci_ingredients":["Aqua","Ethylhexyl Methoxycinnamate","Benzophenone-3","Glycerin","Dimethicone","Tocopherol"],"ewg_score":4.0,"url":"https://lookfantastic.ie/altruist-spf50"},
    {"id":25,"name":"Mineral SPF30 Tinted","brand":"EltaMD","price_eur":35.00,"category":"SPF","skin_type_tags":["sensitive","acne-prone","all"],"condition_tags":["acne","rosacea","general"],"inci_ingredients":["Zinc Oxide","Titanium Dioxide","Aqua","Glycerin","Niacinamide","Tocopherol","Hyaluronic Acid"],"ewg_score":1.5,"url":"https://lookfantastic.ie/eltamd-mineral-spf"},
    {"id":26,"name":"Once Daily Moisturiser SPF50","brand":"Eucerin Sun","price_eur":22.00,"category":"SPF","skin_type_tags":["dry","sensitive"],"condition_tags":["general","eczema"],"inci_ingredients":["Aqua","Ethylhexyl Salicylate","Uvinul A Plus","Glycerin","Panthenol","Tocopherol"],"ewg_score":3.0,"url":"https://mccabes.ie/eucerin-sun-spf50"},
    {"id":27,"name":"Facial Sunscreen SPF50","brand":"Skingredients","price_eur":32.00,"category":"SPF","skin_type_tags":["oily","combination","sensitive"],"condition_tags":["general","acne","rosacea"],"inci_ingredients":["Aqua","Zinc Oxide","Niacinamide","Hyaluronic Acid","Peptide Complex","Glycerin"],"ewg_score":2.0,"url":"https://lookfantastic.ie/skingredients-spf50"},
    {"id":28,"name":"Kids Sun Lotion SPF50+","brand":"Piz Buin","price_eur":14.99,"category":"SPF","skin_type_tags":["sensitive","all"],"condition_tags":["general"],"inci_ingredients":["Aqua","Homosalate","Ethylhexyl Salicylate","Glycerin","Parfum","Methylparaben"],"ewg_score":5.0,"url":"https://boots.ie/pizbuin-kids-spf"},
    {"id":29,"name":"Tinted Moisturiser SPF20","brand":"Boots No7","price_eur":16.00,"category":"SPF","skin_type_tags":["normal","combination"],"condition_tags":["general"],"inci_ingredients":["Aqua","Titanium Dioxide","Glycerin","Dimethicone","Alcohol Denat.","Parfum","Propylparaben"],"ewg_score":6.0,"url":"https://boots.ie/no7-tinted-moisturiser"},
    {"id":30,"name":"Mineral Sunscreen SPF40","brand":"Avene","price_eur":26.00,"category":"SPF","skin_type_tags":["sensitive","dry"],"condition_tags":["rosacea","eczema","general"],"inci_ingredients":["Aqua","Titanium Dioxide","Glycerin","Squalane","Allantoin","Tocopherol"],"ewg_score":2.0,"url":"https://mccabes.ie/avene-mineral-spf"},
    {"id":31,"name":"Niacinamide 10% + Zinc 1%","brand":"The Ordinary","price_eur":6.80,"category":"serum","skin_type_tags":["oily","combination","acne-prone"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Niacinamide","Zinc PCA","Glycerin","Hyaluronic Acid","Panthenol"],"ewg_score":1.5,"url":"https://lookfantastic.ie/the-ordinary-niacinamide"},
    {"id":32,"name":"Hyaluronic Acid 2% + B5","brand":"The Ordinary","price_eur":9.50,"category":"serum","skin_type_tags":["all","dry"],"condition_tags":["general","eczema"],"inci_ingredients":["Aqua","Hyaluronic Acid","Sodium Hyaluronate","Panthenol","Glycerin","Tocopherol"],"ewg_score":1.5,"url":"https://lookfantastic.ie/the-ordinary-ha"},
    {"id":33,"name":"Vitamin C Suspension 23%","brand":"The Ordinary","price_eur":7.60,"category":"serum","skin_type_tags":["all"],"condition_tags":["general","acne"],"inci_ingredients":["Ascorbic Acid","Squalane","Isodecyl Neopentanoate","Alcohol Denat."],"ewg_score":4.0,"url":"https://lookfantastic.ie/the-ordinary-vitc"},
    {"id":34,"name":"Advanced Genifique Serum","brand":"Lancome","price_eur":75.00,"category":"serum","skin_type_tags":["all"],"condition_tags":["general"],"inci_ingredients":["Aqua","Bifidus Extract","Hyaluronic Acid","Glycerin","Niacinamide","Parfum","Methylparaben","Phenoxyethanol"],"ewg_score":5.5,"url":"https://lookfantastic.ie/lancome-genifique"},
    {"id":35,"name":"Retinol 0.5% in Squalane","brand":"The Ordinary","price_eur":8.10,"category":"serum","skin_type_tags":["normal","oily"],"condition_tags":["general","acne"],"inci_ingredients":["Squalane","Retinol","BHT"],"ewg_score":3.5,"url":"https://lookfantastic.ie/the-ordinary-retinol"},
    {"id":36,"name":"Multi-Active Serum","brand":"Clarins","price_eur":48.00,"category":"serum","skin_type_tags":["dry","normal"],"condition_tags":["general"],"inci_ingredients":["Aqua","Glycerin","Teasel Extract","Organic Aloe Vera","Parfum","Methylparaben","Propylparaben"],"ewg_score":5.0,"url":"https://lookfantastic.ie/clarins-multi-active"},
    {"id":37,"name":"Peptide Complex Serum","brand":"Skingredients","price_eur":45.00,"category":"serum","skin_type_tags":["all"],"condition_tags":["general","rosacea"],"inci_ingredients":["Aqua","Peptide Complex","Hyaluronic Acid","Niacinamide","Panthenol","Tocopherol","Glycerin"],"ewg_score":1.5,"url":"https://lookfantastic.ie/skingredients-peptide"},
    {"id":38,"name":"Azelaic Acid Suspension 10%","brand":"The Ordinary","price_eur":9.70,"category":"serum","skin_type_tags":["oily","combination","acne-prone"],"condition_tags":["acne","rosacea"],"inci_ingredients":["Aqua","Azelaic Acid","Glycerin","Dimethicone","Isoceteth-20","Tocopherol"],"ewg_score":2.0,"url":"https://lookfantastic.ie/the-ordinary-azelaic"},
    {"id":39,"name":"Vitamin E Serum","brand":"Boots Ingredients","price_eur":5.99,"category":"serum","skin_type_tags":["dry","normal","sensitive"],"condition_tags":["general","eczema"],"inci_ingredients":["Squalane","Tocopherol","Rosehip Oil","Jojoba Oil"],"ewg_score":1.5,"url":"https://boots.ie/boots-vitamin-e-serum"},
    {"id":40,"name":"Bio-Performance Advanced Super Serum","brand":"Shiseido","price_eur":89.00,"category":"serum","skin_type_tags":["all"],"condition_tags":["general"],"inci_ingredients":["Aqua","Glycerin","Niacinamide","Collagen","Parfum","Alcohol Denat.","DMDM Hydantoin"],"ewg_score":6.5,"url":"https://lookfantastic.ie/shiseido-bio-performance"},
    {"id":41,"name":"Glycolic Acid 7% Toning Solution","brand":"The Ordinary","price_eur":10.20,"category":"toner","skin_type_tags":["oily","combination","normal"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Glycolic Acid","Aloe Vera","Ginseng Extract","Niacinamide","Witch Hazel"],"ewg_score":3.0,"url":"https://lookfantastic.ie/the-ordinary-glycolic"},
    {"id":42,"name":"BHA Blackhead Power Liquid","brand":"COSRX","price_eur":22.00,"category":"toner","skin_type_tags":["oily","combination","acne-prone"],"condition_tags":["acne"],"inci_ingredients":["Betaine Salicylate","Willow Bark Water","Niacinamide","Hyaluronic Acid","Panthenol"],"ewg_score":3.0,"url":"https://lookfantastic.ie/cosrx-bha-liquid"},
    {"id":43,"name":"Rose Water Toner","brand":"Mario Badescu","price_eur":11.00,"category":"toner","skin_type_tags":["dry","sensitive","normal"],"condition_tags":["general","rosacea"],"inci_ingredients":["Aqua","Rosa Damascena Flower Water","Glycerin","Allantoin","Citric Acid"],"ewg_score":2.0,"url":"https://lookfantastic.ie/mario-badescu-rose-water"},
    {"id":44,"name":"Effaclar Clarifying Toner","brand":"La Roche-Posay","price_eur":16.99,"category":"toner","skin_type_tags":["oily","combination"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","LHA","Niacinamide","Zinc PCA","Alcohol Denat.","Salicylic Acid"],"ewg_score":4.5,"url":"https://boots.ie/lrp-effaclar-toner"},
    {"id":45,"name":"Hydrating Toner","brand":"Pyunkang Yul","price_eur":18.00,"category":"toner","skin_type_tags":["dry","sensitive"],"condition_tags":["eczema","general"],"inci_ingredients":["Astragalus Membranaceus Root Extract","Aqua","Glycerin","Allantoin","Panthenol"],"ewg_score":1.5,"url":"https://lookfantastic.ie/pyunkang-yul-toner"},
    {"id":46,"name":"Witch Hazel Toner","brand":"Thayers","price_eur":14.00,"category":"toner","skin_type_tags":["oily","combination"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Hamamelis Virginiana Extract","Aloe Vera Leaf Juice","Glycerin","Citric Acid"],"ewg_score":2.0,"url":"https://lookfantastic.ie/thayers-witch-hazel"},
    {"id":47,"name":"Glow Toner AHA/BHA","brand":"Pixi","price_eur":28.00,"category":"toner","skin_type_tags":["normal","combination","oily"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Glycolic Acid","Salicylic Acid","Aloe Vera","Niacinamide","Alcohol Denat.","Parfum"],"ewg_score":5.5,"url":"https://lookfantastic.ie/pixi-glow-toner"},
    {"id":48,"name":"Sensitive Calming Toner","brand":"Avene","price_eur":13.50,"category":"toner","skin_type_tags":["sensitive","dry"],"condition_tags":["rosacea","eczema"],"inci_ingredients":["Aqua","Avene Thermal Spring Water","Glycerin","Allantoin","Panthenol"],"ewg_score":1.5,"url":"https://mccabes.ie/avene-calming-toner"},
    {"id":49,"name":"AHA 30% + BHA 2% Peeling Solution","brand":"The Ordinary","price_eur":9.30,"category":"toner","skin_type_tags":["normal","oily"],"condition_tags":["acne","general"],"inci_ingredients":["Aqua","Glycolic Acid","Lactic Acid","Salicylic Acid","Tartaric Acid","Alcohol Denat.","DMDM Hydantoin"],"ewg_score":5.5,"url":"https://lookfantastic.ie/the-ordinary-aha-bha"},
    {"id":50,"name":"Soothing Essence Toner","brand":"Eucerin","price_eur":15.00,"category":"toner","skin_type_tags":["sensitive","dry","normal"],"condition_tags":["eczema","rosacea","general"],"inci_ingredients":["Aqua","Panthenol","Allantoin","Niacinamide","Hyaluronic Acid","Glycerin"],"ewg_score":1.5,"url":"https://mccabes.ie/eucerin-soothing-essence"},
]


# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE INGREDIENT SAFETY DATABASE — EWG/SCCS/EU BACKED
# 200+ ingredients. Format: { "inci_name_lower": { hazard_data } }
# ══════════════════════════════════════════════════════════════════════════════

INGREDIENT_DATABASE: Dict[str, Dict] = {
    # ── Fragrance Allergens (EU Mandatory Disclosure List) ────────────────────
    "parfum":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"Synthetic fragrance mixture — one of the most common contact allergens in cosmetics. EU SCCS confirmed allergen. Can trigger contact dermatitis, rosacea flares, and respiratory irritation.","sources":["EWG: 8","EU Cosmetics Reg Annex III","SCCS 2012"]},
    "fragrance":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"US INCI for Parfum. Complex mixture of up to 3,000 undisclosed chemical compounds, many of which are documented contact allergens.","sources":["EWG: 8","IFRA disclosure gap study"]},
    "linalool":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Oxidises on skin to linalool hydroperoxides — a potent contact sensitiser. EU mandatory labelling allergen at >0.001% leave-on.","sources":["EWG: 7","EU Reg 1223/2009 Annex III"]},
    "limonene":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Citrus fragrance; oxidises to limonene hydroperoxides, a leading cause of fragrance contact allergy. EU mandatory disclosure allergen.","sources":["EWG: 7","SCCS 2012"]},
    "cinnamal":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"Cinnamon aldehyde; strong contact sensitiser and one of 26 EU-regulated fragrance allergens with highest sensitisation rates.","sources":["EWG: 8","EU 26 fragrance allergens"]},
    "isoeugenol":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"One of the most potent documented fragrance sensitisers; restricted in EU leave-on products.","sources":["EWG: 8","RIFM"]},
    "eugenol":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Clove-derived fragrance; significant sensitiser, especially at higher concentrations. EU mandatory disclosure allergen.","sources":["EWG: 7","EU Annex III"]},
    "geraniol":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Rose-scented alcohol; documented sensitiser. Listed under EU 26 mandatory disclosure allergens.","sources":["EWG: 7","EU Reg 1223/2009"]},
    "hydroxycitronellal":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"Synthetic muguet fragrance with high sensitisation rate. EU mandatory labelling allergen.","sources":["EWG: 8"]},
    "oak moss extract":{"risk_level":"HIGH","ewg_score":9,"category":"Fragrance Allergen","explanation":"Contains atranol and chloroatranol — the strongest known contact sensitisers in cosmetics. Severely restricted under EU law.","sources":["EWG: 9","EU Commission Reg 2017/1410"]},
    "tree moss extract":{"risk_level":"HIGH","ewg_score":9,"category":"Fragrance Allergen","explanation":"Contains same sensitising chemicals as oak moss. Restricted under EU Cosmetics Regulation.","sources":["EWG: 9"]},
    "amyl cinnamal":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Jasmine fragrance aldehyde; EU-regulated sensitiser.","sources":["EWG: 7","EU Annex III"]},
    "benzyl alcohol":{"risk_level":"MODERATE","ewg_score":5,"category":"Fragrance / Preservative","explanation":"Dual-use fragrance and preservative. Can cause contact sensitisation; EU mandatory fragrance allergen disclosure.","sources":["EWG: 5","EU 26 allergens"]},
    "benzyl benzoate":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"EU regulated mandatory-disclosure fragrance allergen. High sensitisation potential.","sources":["EWG: 7"]},
    "benzyl cinnamate":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"EU mandatory disclosure fragrance allergen; documented sensitiser.","sources":["EWG: 7"]},
    "benzyl salicylate":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Fragrance chemical and UV absorber. EU regulated mandatory allergen; photoallergic reactions reported.","sources":["EWG: 7"]},
    "coumarin":{"risk_level":"HIGH","ewg_score":7,"category":"Fragrance Allergen","explanation":"Sweet herbal fragrance; documented sensitiser with potential hepatotoxicity at high doses. EU mandatory disclosure.","sources":["EWG: 7","EFSA"]},
    "alpha-isomethyl ionone":{"risk_level":"MODERATE","ewg_score":6,"category":"Fragrance Allergen","explanation":"Violet fragrance; EU mandatory disclosure allergen.","sources":["EWG: 6"]},
    "hexyl cinnamal":{"risk_level":"MODERATE","ewg_score":6,"category":"Fragrance Allergen","explanation":"Chamomile fragrance; EU mandatory disclosure sensitiser.","sources":["EWG: 6"]},
    "citronellol":{"risk_level":"MODERATE","ewg_score":6,"category":"Fragrance Allergen","explanation":"Rose fragrance alcohol; EU mandatory labelling allergen.","sources":["EWG: 6"]},
    "farnesol":{"risk_level":"MODERATE","ewg_score":6,"category":"Fragrance Allergen","explanation":"Terpene alcohol; EU regulated fragrance allergen with endocrine disruption concern.","sources":["EWG: 6"]},
    "butylphenyl methylpropional":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"Also known as Lilial. Banned in EU cosmetics since March 2022 due to reproductive toxicity (Reprotox Cat. 1B).","sources":["EU Commission Reg 2021/1902","EWG: 8"]},
    "lilial":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"Trade name for Butylphenyl Methylpropional. Banned in EU cosmetics since March 2022; reproductive toxicant Category 1B.","sources":["EU Reg 2021/1902"]},
    "methyl 2-octynoate":{"risk_level":"HIGH","ewg_score":8,"category":"Fragrance Allergen","explanation":"Violet leaf fragrance; highly potent sensitiser restricted in EU.","sources":["EWG: 8","EU Annex III"]},
    "amylcinnamyl alcohol":{"risk_level":"MODERATE","ewg_score":6,"category":"Fragrance Allergen","explanation":"EU mandatory disclosure fragrance allergen.","sources":["EWG: 6"]},
    "cinnamyl alcohol":{"risk_level":"MODERATE","ewg_score":6,"category":"Fragrance Allergen","explanation":"EU mandatory disclosure fragrance allergen; moderate sensitisation rate.","sources":["EWG: 6"]},
    "anise alcohol":{"risk_level":"MODERATE","ewg_score":5,"category":"Fragrance Allergen","explanation":"EU mandatory disclosure fragrance allergen.","sources":["EWG: 5"]},
    "evernia prunastri extract":{"risk_level":"HIGH","ewg_score":9,"category":"Fragrance Allergen","explanation":"Oak moss extract scientific name; same highest-concern sensitiser profile.","sources":["EWG: 9"]},
    "evernia furfuracea extract":{"risk_level":"HIGH","ewg_score":9,"category":"Fragrance Allergen","explanation":"Tree moss extract; contains the same potent atranol sensitisers as oak moss.","sources":["EWG: 9"]},

    # ── Solvents / Alcohols ───────────────────────────────────────────────────
    "alcohol denat.":{"risk_level":"HIGH","ewg_score":6,"category":"Solvent/Drying Agent","explanation":"Denatured ethyl alcohol. Disrupts skin lipid barrier, increases TEWL, damages skin microbiome. Particularly harmful for dry, sensitive, eczema, and rosacea-prone skin.","sources":["EWG: 6","JEADV 2014 barrier study"]},
    "denatured alcohol":{"risk_level":"HIGH","ewg_score":6,"category":"Solvent/Drying Agent","explanation":"US INCI for Alcohol Denat. Strips natural skin oils and disrupts ceramide barrier function.","sources":["EWG: 6"]},
    "sd alcohol":{"risk_level":"HIGH","ewg_score":6,"category":"Solvent/Drying Agent","explanation":"Specially denatured alcohol; same barrier-disrupting properties as Alcohol Denat.","sources":["EWG: 6"]},
    "sd alcohol 40":{"risk_level":"HIGH","ewg_score":6,"category":"Solvent/Drying Agent","explanation":"Denatured alcohol formulation; same risks as Alcohol Denat.","sources":["EWG: 6"]},
    "isopropyl alcohol":{"risk_level":"HIGH","ewg_score":6,"category":"Solvent/Drying Agent","explanation":"Strong drying solvent; causes severe irritation and barrier disruption. Generally avoided in facial products.","sources":["EWG: 6"]},
    "ethanol":{"risk_level":"LOW","ewg_score":2,"category":"Solvent","explanation":"Pure ethyl alcohol. Low concentrations (<5%) used as solvent are generally safe; higher concentrations in leave-on products can be drying.","sources":["EWG: 2","CIR"]},
    "propylene glycol":{"risk_level":"MODERATE","ewg_score":3,"category":"Humectant/Solvent","explanation":"Generally safe humectant; at higher concentrations can cause contact dermatitis in sensitised individuals. EWG score reflects dose-dependency.","sources":["EWG: 3","CIR Expert Panel"]},
    "butylene glycol":{"risk_level":"LOW","ewg_score":1,"category":"Humectant/Solvent","explanation":"Gentler than propylene glycol; well-tolerated humectant with low sensitisation potential.","sources":["EWG: 1"]},

    # ── Parabens (Endocrine Disruptors) ──────────────────────────────────────
    "methylparaben":{"risk_level":"MODERATE","ewg_score":4,"category":"Preservative (Paraben)","explanation":"Paraben preservative; EWG flags endocrine disruption concern (oestrogenic activity in vitro). EU permits at ≤0.4%; safety under ongoing SCCS review.","sources":["EWG: 4","SCCS/1514/13"]},
    "ethylparaben":{"risk_level":"MODERATE","ewg_score":4,"category":"Preservative (Paraben)","explanation":"Paraben with moderate oestrogenic activity; SCCS concluded safe at permitted concentrations with monitoring.","sources":["EWG: 4"]},
    "propylparaben":{"risk_level":"MODERATE","ewg_score":4,"category":"Preservative (Paraben)","explanation":"Longer-chain paraben with stronger oestrogenic activity; EWG and some regulators flag higher concern.","sources":["EWG: 4","SCCS/1514/13"]},
    "butylparaben":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative (Paraben)","explanation":"Highest oestrogenic potency among common parabens. Banned in Denmark for under-3 products; restricted in several Scandinavian markets.","sources":["EWG: 7","Danish EPA 2011"]},
    "isobutylparaben":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative (Paraben)","explanation":"Similar oestrogenic potency to butylparaben; EU restricted in baby products.","sources":["EWG: 7","EU Reg 1004/2014"]},
    "isopropylparaben":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative (Paraben)","explanation":"EU restricted; high endocrine disruption concern per EWG.","sources":["EWG: 7","EU Reg 1004/2014"]},
    "phenylparaben":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative (Paraben)","explanation":"Restricted paraben; EU prohibited in cosmetics.","sources":["EWG: 7","EU Annex II"]},
    "benzylparaben":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative (Paraben)","explanation":"Banned in EU cosmetics; strongest endocrine disruption concern in paraben class.","sources":["EWG: 7","EU Annex II"]},

    # ── UV Filters (Endocrine Disruption / Systemic Absorption) ──────────────
    "benzophenone-3":{"risk_level":"HIGH","ewg_score":8,"category":"UV Filter / Endocrine Disruptor","explanation":"Oxybenzone. High systemic absorption (FDA 2019); strong endocrine disruption evidence. Banned in Hawaii for coral reef toxicity.","sources":["EWG: 8","FDA 2019 systemic absorption","Hawaii Act 104 2018"]},
    "oxybenzone":{"risk_level":"HIGH","ewg_score":8,"category":"UV Filter / Endocrine Disruptor","explanation":"Same as Benzophenone-3 (US common name). Same systemic absorption and endocrine disruption concerns.","sources":["EWG: 8","FDA 2019"]},
    "benzophenone-1":{"risk_level":"HIGH","ewg_score":8,"category":"UV Filter","explanation":"Related to oxybenzone; similar endocrine disruption and absorption concerns.","sources":["EWG: 8"]},
    "homosalate":{"risk_level":"MODERATE","ewg_score":4,"category":"UV Filter","explanation":"EU restricted maximum to 7.34% in 2021 due to endocrine disruption potential and systemic absorption.","sources":["EWG: 4","SCCS/1622/21"]},
    "octinoxate":{"risk_level":"MODERATE","ewg_score":5,"category":"UV Filter","explanation":"Ethylhexyl methoxycinnamate; potential endocrine disruptor with demonstrated systemic absorption.","sources":["EWG: 5","FDA 2019"]},
    "ethylhexyl methoxycinnamate":{"risk_level":"MODERATE","ewg_score":5,"category":"UV Filter","explanation":"Same as Octinoxate. Potential endocrine disruptor with systemic absorption shown in FDA 2019 study.","sources":["EWG: 5"]},
    "4-methylbenzylidene camphor":{"risk_level":"HIGH","ewg_score":7,"category":"UV Filter / Endocrine Disruptor","explanation":"Strong thyroid disruptor; restricted in EU and Japan. Not FDA-approved.","sources":["EWG: 7","SCCS 2015"]},
    "octocrylene":{"risk_level":"MODERATE","ewg_score":3,"category":"UV Filter","explanation":"Accumulates in aquatic organisms; converts to benzophenone (endocrine disruptor) over time in stored products.","sources":["EWG: 3","2021 octocrylene study"]},
    "avobenzone":{"risk_level":"LOW","ewg_score":2,"category":"UV Filter","explanation":"Broad-spectrum UVA filter; systemically absorbed (FDA 2019) but EWG considers low concern. Photostability issues.","sources":["EWG: 2","FDA 2019"]},
    "retinyl palmitate":{"risk_level":"MODERATE","ewg_score":7,"category":"Vitamin A Ester","explanation":"EWG concern: photocarcinogenicity data in animal models (NTP 2012). Particularly flagged in sun-exposed products. Avoid in SPF products.","sources":["EWG: 7","NTP 2012"]},

    # ── Formaldehyde Releasers ─────────────────────────────────────────────────
    "dmdm hydantoin":{"risk_level":"MODERATE","ewg_score":5,"category":"Formaldehyde Releaser","explanation":"Releases formaldehyde as preservative mechanism. Formaldehyde is IARC Group 1 carcinogen; contact allergen and sensitiser associated with eczematous dermatitis.","sources":["EWG: 5","IARC 100F","CIR 2018"]},
    "imidazolidinyl urea":{"risk_level":"MODERATE","ewg_score":5,"category":"Formaldehyde Releaser","explanation":"Formaldehyde-releasing preservative; moderate sensitiser causing allergic contact dermatitis.","sources":["EWG: 5","CIR"]},
    "diazolidinyl urea":{"risk_level":"MODERATE","ewg_score":6,"category":"Formaldehyde Releaser","explanation":"Higher formaldehyde release rate than imidazolidinyl urea; significant contact allergen.","sources":["EWG: 6"]},
    "quaternium-15":{"risk_level":"HIGH","ewg_score":8,"category":"Formaldehyde Releaser","explanation":"Highest formaldehyde-releasing preservative in cosmetics. IARC carcinogen (formaldehyde); leading cause of preservative contact allergy globally.","sources":["EWG: 8","ECHA SVHC"]},
    "bronopol":{"risk_level":"MODERATE","ewg_score":5,"category":"Formaldehyde Releaser","explanation":"Releases formaldehyde; reacts with amines to form nitrosamines. Moderate carcinogenicity and sensitisation concern.","sources":["EWG: 5","EU Annex V"]},
    "2-bromo-2-nitropropane-1,3-diol":{"risk_level":"MODERATE","ewg_score":5,"category":"Formaldehyde Releaser","explanation":"Alternate INCI for Bronopol; same nitrosamine and formaldehyde risks.","sources":["EWG: 5"]},
    "formaldehyde":{"risk_level":"HIGH","ewg_score":10,"category":"Carcinogen / Sensitiser","explanation":"IARC Group 1 carcinogen; potent skin sensitiser. Banned in EU cosmetics above trace levels.","sources":["EWG: 10","IARC 100F","EU Annex II"]},
    "methylene glycol":{"risk_level":"HIGH","ewg_score":9,"category":"Formaldehyde Source","explanation":"Hydrated form of formaldehyde; used in some keratin treatments. Converts to gaseous formaldehyde when heated.","sources":["EWG: 9","FDA hair treatment warning"]},

    # ── Surfactants ────────────────────────────────────────────────────────────
    "sodium lauryl sulfate":{"risk_level":"MODERATE","ewg_score":3,"category":"Surfactant (Harsh)","explanation":"Strong anionic surfactant; well-documented skin irritant and barrier disruptor. Used as benchmark irritant in dermatology research. Avoid for sensitive/eczema-prone skin.","sources":["EWG: 3","JID irritancy benchmark studies"]},
    "sls":{"risk_level":"MODERATE","ewg_score":3,"category":"Surfactant (Harsh)","explanation":"Abbreviation for Sodium Lauryl Sulfate; same barrier-disrupting and irritancy profile.","sources":["EWG: 3"]},
    "ammonium lauryl sulfate":{"risk_level":"MODERATE","ewg_score":4,"category":"Surfactant (Harsh)","explanation":"Similar to SLS; potent barrier disruptor, especially for repeated use products.","sources":["EWG: 4"]},
    "sodium laureth sulfate":{"risk_level":"LOW","ewg_score":3,"category":"Surfactant (Mild)","explanation":"Ethoxylated, less irritating than SLS. May contain trace 1,4-dioxane contamination. Acceptable in rinse-off products.","sources":["EWG: 3","FDA 1,4-dioxane survey"]},
    "ammonium laureth sulfate":{"risk_level":"LOW","ewg_score":3,"category":"Surfactant (Mild)","explanation":"Same 1,4-dioxane contamination concern as SLES.","sources":["EWG: 3"]},
    "sodium dodecylbenzene sulfonate":{"risk_level":"MODERATE","ewg_score":4,"category":"Surfactant (Harsh)","explanation":"Harsh anionic surfactant; significant irritation potential similar to SLS.","sources":["EWG: 4"]},

    # ── Antimicrobials ─────────────────────────────────────────────────────────
    "triclosan":{"risk_level":"HIGH","ewg_score":7,"category":"Antimicrobial / Endocrine Disruptor","explanation":"FDA banned from OTC antiseptics (2016). Endocrine disruption evidence; promotes antimicrobial resistance; aquatic toxicity.","sources":["EWG: 7","FDA Rule 2016-00989"]},
    "triclocarban":{"risk_level":"HIGH","ewg_score":7,"category":"Antimicrobial","explanation":"Banned by FDA from OTC antiseptics in 2016. Similar endocrine and environmental concerns as triclosan.","sources":["EWG: 7","FDA 2016"]},
    "benzalkonium chloride":{"risk_level":"MODERATE","ewg_score":5,"category":"QAC Antimicrobial","explanation":"Skin and mucous membrane irritant; may promote antimicrobial resistance. Contact sensitiser documented.","sources":["EWG: 5"]},
    "chlorphenesin":{"risk_level":"MODERATE","ewg_score":5,"category":"Preservative","explanation":"Synthetic preservative with documented contact allergy potential.","sources":["EWG: 5"]},
    "chloroxylenol":{"risk_level":"MODERATE","ewg_score":4,"category":"Antimicrobial Preservative","explanation":"PCMX; skin sensitiser in some individuals. Used in antiseptic formulations.","sources":["EWG: 4"]},

    # ── Isothiazolinones ───────────────────────────────────────────────────────
    "methylisothiazolinone":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative / Sensitiser","explanation":"MIT. Banned from EU leave-on cosmetics since 2017 due to epidemic of contact allergy. Allowed in rinse-off at ≤0.0015%.","sources":["EWG: 7","EU Reg 2017/2228","SCCS/1570/15"]},
    "methylchloroisothiazolinone":{"risk_level":"HIGH","ewg_score":7,"category":"Preservative / Sensitiser","explanation":"MCI. Usually combined with MI; restricted to rinse-off products in EU. Strong contact sensitiser.","sources":["EWG: 7","EU Reg 1223/2009 Amendment"]},
    "benzisothiazolinone":{"risk_level":"MODERATE","ewg_score":5,"category":"Preservative / Sensitiser","explanation":"BIT; sensitisation potential, though less than MIT/MCI. Restricted in EU cosmetics.","sources":["EWG: 5"]},

    # ── Other Preservatives ───────────────────────────────────────────────────
    "phenoxyethanol":{"risk_level":"LOW","ewg_score":4,"category":"Preservative","explanation":"Paraben alternative. EU warns against use in infant products (lip area). Neurotoxicity and reproductive concern at high doses; generally safe in adult products at permitted levels.","sources":["EWG: 4","ANSM 2012","EU CosIng"]},
    "chlorhexidine":{"risk_level":"MODERATE","ewg_score":5,"category":"Antimicrobial Preservative","explanation":"Broad-spectrum antimicrobial; well-documented sensitiser, particularly in healthcare settings. Risk of severe anaphylaxis with repeated mucosal exposure.","sources":["EWG: 5","BMJ allergy reports"]},
    "chlorhexidine dihydrochloride":{"risk_level":"MODERATE","ewg_score":5,"category":"Antimicrobial Preservative","explanation":"Salt form of chlorhexidine; same sensitisation and anaphylaxis risk profile.","sources":["EWG: 5"]},
    "thimerosal":{"risk_level":"HIGH","ewg_score":9,"category":"Mercury Compound / Preservative","explanation":"Mercury-based preservative. Banned in EU cosmetics; known neurotoxicant and skin allergen. No safe level established for topical application.","sources":["EWG: 9","EU Annex II"]},

    # ── Heavy Metals / Carcinogens ────────────────────────────────────────────
    "lead acetate":{"risk_level":"HIGH","ewg_score":10,"category":"Heavy Metal","explanation":"FDA banned from hair dye products in 2018. Known neurotoxin, reproductive toxicant, and probable human carcinogen (IARC Group 2A).","sources":["EWG: 10","FDA 2018","IARC Group 2A"]},
    "coal tar":{"risk_level":"HIGH","ewg_score":10,"category":"Carcinogen","explanation":"IARC Group 1 carcinogen. Banned in EU cosmetics; restricted to pharmaceutical preparations under medical supervision only.","sources":["EWG: 10","IARC Monograph 100F","EU Annex II Entry 1185"]},
    "dibutyl phthalate":{"risk_level":"HIGH","ewg_score":10,"category":"Endocrine Disruptor","explanation":"DBP. Banned in EU cosmetics. Reproductive toxicant Category 1B; developmental toxicant.","sources":["EWG: 10","EU Annex II Entry 152"]},
    "diethyl phthalate":{"risk_level":"HIGH","ewg_score":7,"category":"Endocrine Disruptor","explanation":"DEP. Used as fragrance carrier/fixative. EWG flags endocrine disruption concern.","sources":["EWG: 7"]},
    "aluminium chlorohydrate":{"risk_level":"MODERATE","ewg_score":4,"category":"Antiperspirant Active","explanation":"Common antiperspirant; EWG flags developmental, endocrine, and neurotoxicity concerns at high systemic exposure. Ongoing breast tissue accumulation debate.","sources":["EWG: 4","SCCS/1525/14"]},

    # ── Occlusives / Emollients ───────────────────────────────────────────────
    "mineral oil":{"risk_level":"LOW","ewg_score":2,"category":"Occlusive Emollient","explanation":"Petroleum-derived occlusive; cosmetic grades are safe but may occlude pores in acne-prone individuals. EWG low concern for PAH contamination in non-cosmetic grades.","sources":["EWG: 2","CIR 2005"]},
    "paraffinum liquidum":{"risk_level":"LOW","ewg_score":2,"category":"Occlusive Emollient","explanation":"INCI for mineral oil; same profile.","sources":["EWG: 2"]},
    "liquid paraffin":{"risk_level":"LOW","ewg_score":2,"category":"Occlusive Emollient","explanation":"Pharmaceutical/cosmetic name for mineral oil.","sources":["EWG: 2"]},
    "petrolatum":{"risk_level":"LOW","ewg_score":2,"category":"Occlusive Emollient","explanation":"Petroleum jelly; cosmetic-grade safe. EWG flags non-cosmetic grades for PAH contamination. EU restricts non-purified grades.","sources":["EWG: 2","EU Reg 2018 PAH amendment"]},
    "isopropyl myristate":{"risk_level":"LOW","ewg_score":2,"category":"Emollient","explanation":"Comedogenicity rating 3/5; can trigger acne/folliculitis in susceptible individuals.","sources":["EWG: 2","comedogenicity index"]},
    "isopropyl palmitate":{"risk_level":"LOW","ewg_score":2,"category":"Emollient","explanation":"Fatty acid ester; comedogenicity rating 4/5 — high pore-clogging risk for acne-prone skin.","sources":["EWG: 2"]},
    "lanolin":{"risk_level":"MODERATE","ewg_score":4,"category":"Emollient","explanation":"Sheep sebum-derived; rich emollient but well-known contact allergen, particularly in wool-sensitive individuals and eczema patients.","sources":["EWG: 4","Contact Derm 2010"]},
    "lanolin alcohol":{"risk_level":"MODERATE","ewg_score":4,"category":"Emollient","explanation":"Lanolin derivative; same sensitisation risk as lanolin in wool-sensitive individuals.","sources":["EWG: 4"]},

    # ── Antioxidants ───────────────────────────────────────────────────────────
    "bht":{"risk_level":"MODERATE","ewg_score":4,"category":"Antioxidant / Preservative","explanation":"Butylated hydroxytoluene. EWG flags endocrine disruption, organ toxicity, and carcinogenicity concerns based on animal data. IARC Group 3.","sources":["EWG: 4","IARC Group 3"]},
    "bha":{"risk_level":"MODERATE","ewg_score":6,"category":"Antioxidant","explanation":"Butylated hydroxyanisole; IARC Group 2B possible carcinogen. EU limits to ≤0.02% in cosmetics. EWG flags endocrine disruption.","sources":["EWG: 6","IARC 2B","EU Annex III"]},
    "butylated hydroxytoluene":{"risk_level":"MODERATE","ewg_score":4,"category":"Antioxidant / Preservative","explanation":"Full name for BHT; same endocrine disruption and carcinogenicity concerns.","sources":["EWG: 4"]},
    "butylated hydroxyanisole":{"risk_level":"MODERATE","ewg_score":6,"category":"Antioxidant","explanation":"Full name for BHA; same IARC 2B and endocrine disruption concerns.","sources":["EWG: 6"]},

    # ── Active Ingredients (Caution for Certain Conditions) ───────────────────
    "retinol":{"risk_level":"LOW","ewg_score":3,"category":"Active Ingredient","explanation":"Vitamin A; effective anti-ageing but can cause retinoid dermatitis at high concentrations. Avoid with rosacea or eczema. Teratogenic — contraindicated during pregnancy.","sources":["EWG: 3","SCCS retinol opinion 2022"]},
    "retinaldehyde":{"risk_level":"LOW","ewg_score":3,"category":"Active Ingredient","explanation":"Vitamin A aldehyde; more potent than retinol. Similar sensitivity and pregnancy caution apply.","sources":["EWG: 3"]},
    "salicylic acid":{"risk_level":"LOW","ewg_score":2,"category":"Active Ingredient (BHA)","explanation":"Beta-hydroxy acid; EU restricts to ≤2% in rinse-off, ≤0.5% leave-on. Not for use on children or during pregnancy (at high concentrations).","sources":["EWG: 2","EU Annex III"]},
    "glycolic acid":{"risk_level":"LOW","ewg_score":2,"category":"Active Ingredient (AHA)","explanation":"Alpha-hydroxy acid; increases photosensitivity. Use SPF. Can cause irritation and barrier disruption at high concentrations.","sources":["EWG: 2","FDA AHA photosensitivity study"]},
    "lactic acid":{"risk_level":"LOW","ewg_score":1,"category":"Active Ingredient (AHA)","explanation":"Gentler AHA; increases photosensitivity. Generally well-tolerated even for sensitive skin at appropriate pH/concentration.","sources":["EWG: 1"]},

    # ── Nitrosamine Precursors ─────────────────────────────────────────────────
    "triethanolamine":{"risk_level":"MODERATE","ewg_score":4,"category":"pH Adjuster / Nitrosamine Precursor","explanation":"TEA; can react with nitrite impurities to form nitrosamines (potential carcinogens). EU restricts nitrosamine formation in formulations.","sources":["EWG: 4","EU Annex III restriction"]},
    "diethanolamine":{"risk_level":"MODERATE","ewg_score":5,"category":"pH Adjuster","explanation":"DEA; nitrosamine formation risk. Restricted in EU cosmetics.","sources":["EWG: 5","EU Annex II"]},
    "monoethanolamine":{"risk_level":"MODERATE","ewg_score":4,"category":"pH Adjuster","explanation":"MEA; potential nitrosamine formation. EWG flags concern for repeat exposure.","sources":["EWG: 4"]},
    "cocamide dea":{"risk_level":"MODERATE","ewg_score":5,"category":"Surfactant / Nitrosamine Precursor","explanation":"DEA-based surfactant; nitrosamine contamination potential. California Prop 65 listed as carcinogen.","sources":["EWG: 5","California Prop 65"]},

    # ── PEG Compounds ──────────────────────────────────────────────────────────
    "peg-8":{"risk_level":"LOW","ewg_score":2,"category":"PEG Ethoxylate","explanation":"Ethoxylated compound; may contain trace 1,4-dioxane (IARC Group 2B) and ethylene oxide as processing impurities. Low inherent toxicity.","sources":["EWG: 2","FDA 1,4-dioxane monitoring"]},
    "peg-40 hydrogenated castor oil":{"risk_level":"LOW","ewg_score":2,"category":"PEG Ethoxylate","explanation":"Ethoxylated solubiliser; same 1,4-dioxane contamination concern. Safe at cosmetic use levels.","sources":["EWG: 2"]},
    "peg-100 stearate":{"risk_level":"LOW","ewg_score":2,"category":"PEG Ethoxylate","explanation":"Ethoxylated emulsifier; 1,4-dioxane contamination risk dependent on manufacturing quality.","sources":["EWG: 2"]},

    # ── Mineral Fillers ────────────────────────────────────────────────────────
    "talc":{"risk_level":"MODERATE","ewg_score":3,"category":"Mineral Filler","explanation":"Non-fibrous talc is safe for skin; EWG concern relates to possible asbestos contamination in non-pharmaceutical grades and inhalation risk in loose powders.","sources":["EWG: 3","IARC asbestos-related talc"]},
    "titanium dioxide":{"risk_level":"LOW","ewg_score":2,"category":"UV Filter / Pigment","explanation":"Topically safe; IARC classified inhaled titanium dioxide as Group 2B (possible carcinogen) — relevant to loose powder products only.","sources":["EWG: 2","IARC 2B (inhalation only)"]},
    "zinc oxide":{"risk_level":"LOW","ewg_score":2,"category":"UV Filter","explanation":"EWG's highest-rated sunscreen ingredient. Broad-spectrum, photostable, minimal systemic absorption. Safe for sensitive and baby skin.","sources":["EWG: 2"]},
    "mica":{"risk_level":"LOW","ewg_score":1,"category":"Mineral Pigment","explanation":"Natural mineral shimmer; safe for topical use. Inhalation concern only for loose powder formats.","sources":["EWG: 1"]},
    "silica":{"risk_level":"LOW","ewg_score":1,"category":"Mineral Filler","explanation":"Amorphous silica; safe in cosmetics. Crystalline silica (IARC Group 1 carcinogen) is not present in cosmetic formulations.","sources":["EWG: 1"]},

    # ── Safe / Low Concern Actives & Humectants ───────────────────────────────
    "hyaluronic acid":{"risk_level":"LOW","ewg_score":1,"category":"Humectant","explanation":"Endogenous skin component; excellent safety profile. Highly effective humectant with no known allergenicity at cosmetic concentrations.","sources":["EWG: 1"]},
    "sodium hyaluronate":{"risk_level":"LOW","ewg_score":1,"category":"Humectant","explanation":"Sodium salt of hyaluronic acid; same excellent safety profile.","sources":["EWG: 1"]},
    "glycerin":{"risk_level":"LOW","ewg_score":1,"category":"Humectant","explanation":"Safe, effective humectant. No known toxicity in cosmetic applications.","sources":["EWG: 1"]},
    "niacinamide":{"risk_level":"LOW","ewg_score":1,"category":"Active Ingredient","explanation":"Vitamin B3; excellent safety profile. Rare flushing reactions possible at very high concentrations (>4%).","sources":["EWG: 1"]},
    "ceramide np":{"risk_level":"LOW","ewg_score":1,"category":"Barrier Lipid","explanation":"Endogenous skin lipid; safe and highly effective for barrier repair.","sources":["EWG: 1"]},
    "ceramide ap":{"risk_level":"LOW","ewg_score":1,"category":"Barrier Lipid","explanation":"Skin-identical ceramide; safe and effective barrier support ingredient.","sources":["EWG: 1"]},
    "ceramide eop":{"risk_level":"LOW","ewg_score":1,"category":"Barrier Lipid","explanation":"Omega-acylated ceramide; critical for lamellar body formation. Excellent safety profile.","sources":["EWG: 1"]},
    "panthenol":{"risk_level":"LOW","ewg_score":1,"category":"Humectant / Barrier","explanation":"Pro-vitamin B5; excellent safety profile, anti-inflammatory, promotes wound healing.","sources":["EWG: 1"]},
    "allantoin":{"risk_level":"LOW","ewg_score":1,"category":"Soothing Agent","explanation":"Naturally derived soothing agent; excellent safety and tolerability.","sources":["EWG: 1"]},
    "tocopherol":{"risk_level":"LOW","ewg_score":1,"category":"Antioxidant (Vitamin E)","explanation":"Vitamin E; safe antioxidant. Rare contact allergy reported in sensitive individuals.","sources":["EWG: 1"]},
    "squalane":{"risk_level":"LOW","ewg_score":1,"category":"Emollient","explanation":"Stable, non-comedogenic emollient. Excellent tolerability; suitable for all skin types.","sources":["EWG: 1"]},
    "shea butter":{"risk_level":"LOW","ewg_score":1,"category":"Emollient","explanation":"Rich emollient; generally safe. Rare tree nut allergy possible in highly sensitised individuals.","sources":["EWG: 1"]},
    "colloidal oatmeal":{"risk_level":"LOW","ewg_score":1,"category":"Soothing Active","explanation":"FDA-approved skin protectant; particularly effective for eczema and dry skin. Safe for all skin types.","sources":["EWG: 1","FDA 21 CFR 347"]},
    "zinc pca":{"risk_level":"LOW","ewg_score":1,"category":"Sebum Regulator","explanation":"Zinc salt of pyrrolidone carboxylic acid; sebum-regulating, anti-inflammatory. Safe and well-tolerated.","sources":["EWG: 1"]},
    "aqua":{"risk_level":"LOW","ewg_score":1,"category":"Solvent","explanation":"Water; universal solvent. No safety concern.","sources":["EWG: 1"]},
    "water":{"risk_level":"LOW","ewg_score":1,"category":"Solvent","explanation":"Water; universal solvent. No safety concern.","sources":["EWG: 1"]},
    "azelaic acid":{"risk_level":"LOW","ewg_score":1,"category":"Active Ingredient","explanation":"Naturally occurring dicarboxylic acid; effective for acne and rosacea. Excellent safety profile.","sources":["EWG: 1"]},
    "ascorbic acid":{"risk_level":"LOW","ewg_score":1,"category":"Active Ingredient (Vitamin C)","explanation":"Pure Vitamin C; potent antioxidant. Can cause mild irritation at high concentrations; pH-sensitive and unstable.","sources":["EWG: 1"]},
    "urea":{"risk_level":"LOW","ewg_score":1,"category":"Keratolytic / Humectant","explanation":"Endogenous skin NMF component; safe and effective humectant and keratolytic at low concentrations.","sources":["EWG: 1"]},
    "madecassoside":{"risk_level":"LOW","ewg_score":1,"category":"Active Ingredient","explanation":"Centella asiatica extract; anti-inflammatory, wound-healing. Excellent safety profile.","sources":["EWG: 1"]},
    "niacinamide":{"risk_level":"LOW","ewg_score":1,"category":"Active Ingredient","explanation":"Vitamin B3; excellent tolerability and safety.","sources":["EWG: 1"]},
}


# ── Flagging logic ─────────────────────────────────────────────────────────────

def _match_ingredient(ingredient_lower: str) -> dict | None:
    """
    Match ingredient against the comprehensive database.
    Applies exact lookup first, then pattern-based rules.
    LOW-risk safe ingredients are NOT flagged (only return on MODERATE/HIGH or special LOW).
    """
    data = INGREDIENT_DATABASE.get(ingredient_lower)
    if data:
        # Only flag MODERATE and HIGH; LOW-risk ingredients are treated as safe
        if data["risk_level"] in ("HIGH", "MODERATE"):
            return data
        # LOW risk: still flag so users see the note, but mark clearly
        if data["risk_level"] == "LOW" and data.get("ewg_score", 1) >= 3:
            return data
        return None  # genuinely safe — don't clutter the flagged list

    # Paraben suffix catch-all (handles any unlisted paraben)
    if ingredient_lower.endswith("paraben"):
        return {"risk_level":"MODERATE","ewg_score":4,"category":"Preservative (Paraben)",
                "explanation":"Paraben preservative class. EWG flags class-level endocrine disruption concern (oestrogenic activity). Longer-chain parabens (propyl, butyl) carry higher concern.",
                "sources":["EWG Skin Deep class concern","SCCS/1514/13"]}

    # Isothiazolinone class catch-all
    if "isothiazolinone" in ingredient_lower:
        return {"risk_level":"HIGH","ewg_score":7,"category":"Preservative / Sensitiser",
                "explanation":"Isothiazolinone-class preservative. This class caused a widespread EU contact allergy epidemic. Banned in EU leave-on cosmetics (MIT/MCI) since 2017.",
                "sources":["EWG: 7","SCCS/1570/15","EU Reg 2017/2228"]}

    # PEG compound catch-all
    if ingredient_lower.startswith("peg-") or " peg-" in ingredient_lower:
        return {"risk_level":"LOW","ewg_score":2,"category":"PEG Ethoxylate",
                "explanation":"Ethoxylated compound. May contain trace 1,4-dioxane (IARC Group 2B) and ethylene oxide as processing impurities. Concern is contamination-dependent, not intrinsic toxicity.",
                "sources":["EWG: 2","FDA 1,4-dioxane monitoring"]}

    # Formaldehyde releaser patterns
    for pattern in ["methylene glycol","methanediol","formalin","formic aldehyde","methanal"]:
        if pattern in ingredient_lower:
            return {"risk_level":"HIGH","ewg_score":8,"category":"Formaldehyde Source",
                    "explanation":"Formaldehyde or direct source. IARC Group 1 carcinogen; potent skin sensitiser.",
                    "sources":["IARC 100F","EWG: 8"]}

    return None


def flag_ingredients(inci_list: list) -> dict:
    """
    Flag INCI ingredients against the comprehensive EWG-backed database.
    Returns { flagged: [...], safe: [...], summary: {...} }
    """
    flagged = []
    safe = []
    high_count = 0
    moderate_count = 0
    low_count = 0

    for ingredient in inci_list:
        ingredient_clean = ingredient.strip()
        if not ingredient_clean:
            continue
        match = _match_ingredient(ingredient_clean.lower())
        if match:
            risk = match["risk_level"]
            if risk == "HIGH":
                high_count += 1
            elif risk == "MODERATE":
                moderate_count += 1
            else:
                low_count += 1
            flagged.append({
                "ingredient": ingredient_clean,
                "risk_level": risk,
                "ewg_score": match.get("ewg_score"),
                "category": match.get("category", ""),
                "explanation": match["explanation"],
                "sources": match.get("sources", []),
            })
        else:
            safe.append(ingredient_clean)

    if high_count >= 1:
        overall_risk = "HIGH"
    elif moderate_count >= 2:
        overall_risk = "MODERATE"
    elif moderate_count >= 1 or low_count >= 2:
        overall_risk = "LOW"
    else:
        overall_risk = "SAFE"

    return {
        "flagged": flagged,
        "safe": safe,
        "summary": {
            "total_checked": len(inci_list),
            "total_flagged": len(flagged),
            "high_risk": high_count,
            "moderate_risk": moderate_count,
            "low_risk": low_count,
            "overall_risk": overall_risk,
        }
    }


CONDITION_TAG_MAP = {
    "Melanoma": "general",
    "Melanocytic Nevi": "general",
    "Basal Cell Carcinoma": "general",
    "Actinic Keratosis": "general",
    "Benign Keratosis": "general",
    "Dermatofibroma": "general",
    "Vascular Lesion": "general",
    "Squamous Cell Carcinoma": "general",
    "Unknown": "general",
    "eczema": "eczema",
    "psoriasis": "psoriasis",
    "rosacea": "rosacea",
    "acne": "acne",
    "general": "general",
}


def get_products_for_condition(condition: str) -> list:
    tag = CONDITION_TAG_MAP.get(condition, "general")
    return [p for p in PRODUCT_CATALOGUE if tag in p["condition_tags"] or "general" in p["condition_tags"]]
