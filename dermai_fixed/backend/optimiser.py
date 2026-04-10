"""
optimiser.py
Binary Integer Linear Programme using PuLP.
Maximises product safety score subject to budget and category constraints.
"""

import logging
from typing import List, Dict
from product_db import flag_ingredients

logger = logging.getLogger(__name__)


def optimise_basket(
    condition: str,
    budget_eur: float,
    required_categories: List[str],
    product_catalogue: List[Dict],
) -> dict:
    try:
        import pulp
    except ImportError:
        return {"solve_status": "Error", "message": "PuLP not installed"}

    # Filter eligible products for the condition
    from product_db import get_products_for_condition, PRODUCT_CATALOGUE
    eligible = get_products_for_condition(condition)

    if not eligible:
        eligible = product_catalogue

    # Build the problem
    prob = pulp.LpProblem("SkincareBudgetOptimisation", pulp.LpMaximize)

    # Decision variables: x_i ∈ {0, 1}
    x = {p["id"]: pulp.LpVariable(f"x_{p['id']}", cat="Binary") for p in eligible}

    # Objective: maximise sum of safety score = (10 - ewg_score)
    prob += pulp.lpSum((10 - p["ewg_score"]) * x[p["id"]] for p in eligible)

    # Budget constraint
    prob += pulp.lpSum(p["price_eur"] * x[p["id"]] for p in eligible) <= budget_eur

    # One product per required category
    for cat in required_categories:
        cat_products = [p for p in eligible if p["category"].lower() == cat.lower()]
        if cat_products:
            prob += pulp.lpSum(x[p["id"]] for p in cat_products) == 1

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)
    solve_status = pulp.LpStatus[prob.status]

    if solve_status == "Infeasible":
        # Relax budget by 10% increments up to 50%
        for pct in [0.1, 0.2, 0.3, 0.4, 0.5]:
            relaxed_budget = budget_eur * (1 + pct)
            prob2 = pulp.LpProblem("SkincareBudgetRelaxed", pulp.LpMaximize)
            x2 = {p["id"]: pulp.LpVariable(f"x_{p['id']}", cat="Binary") for p in eligible}
            prob2 += pulp.lpSum((10 - p["ewg_score"]) * x2[p["id"]] for p in eligible)
            prob2 += pulp.lpSum(p["price_eur"] * x2[p["id"]] for p in eligible) <= relaxed_budget
            for cat in required_categories:
                cat_products = [p for p in eligible if p["category"].lower() == cat.lower()]
                if cat_products:
                    prob2 += pulp.lpSum(x2[p["id"]] for p in cat_products) == 1
            prob2.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob2.status] == "Optimal":
                selected = [p for p in eligible if pulp.value(x2[p["id"]]) == 1]
                total_price = sum(p["price_eur"] for p in selected)
                total_safety = sum(10 - p["ewg_score"] for p in selected)
                all_inci = [i for p in selected for i in p["inci_ingredients"]]
                flagged = flag_ingredients(all_inci)["flagged"]
                return {
                    "selected_products": selected,
                    "total_price": round(total_price, 2),
                    "total_safety_score": round(total_safety, 1),
                    "solve_status": f"Optimal (budget relaxed +{int(pct*100)}%)",
                    "flagged_in_basket": flagged,
                    "message": f"Budget was too low. Minimum required: €{round(relaxed_budget, 2)}"
                }
        return {
            "selected_products": [],
            "total_price": 0,
            "total_safety_score": 0,
            "solve_status": "Infeasible",
            "flagged_in_basket": [],
            "message": "No feasible solution found even with 50% budget relaxation."
        }

    selected = [p for p in eligible if pulp.value(x[p["id"]]) == 1]
    total_price = sum(p["price_eur"] for p in selected)
    total_safety = sum(10 - p["ewg_score"] for p in selected)
    all_inci = [i for p in selected for i in p["inci_ingredients"]]
    flagged = flag_ingredients(all_inci)["flagged"]

    return {
        "selected_products": selected,
        "total_price": round(total_price, 2),
        "total_safety_score": round(total_safety, 1),
        "solve_status": solve_status,
        "flagged_in_basket": flagged,
        "message": None,
    }
