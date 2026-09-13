import numpy as np
import pandas as pd
from src.agent_executor import OptimizationExecutor

# ==========================================
# 1. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
# ==========================================
np.random.seed(42)
N = 15

tickers = [f"TICK{i}" for i in range(1, N+1)]
secteurs = ["Tech", "Finance", "Energy"] * 5
esg_scores = np.random.uniform(40, 95, N)
expected_returns = np.random.uniform(0.02, 0.15, N)

df_data = pd.DataFrame({
    "Ticker": tickers,
    "Sector": secteurs,
    "ESG_Score": esg_scores,
    "Expected_Return": expected_returns
})

# Matrice de covariance semi-définie positive (A * A.T)
A = np.random.randn(N, N)
sigma = np.dot(A, A.T) / 100 
matrix_inputs = {
    "Variance": sigma.tolist(),
    "w0": np.full(N, 1.0/N).tolist()
}

# Variables de décision standards
vars_cont = [{"name": "x", "size": "n_rows", "type": "continuous"}]
vars_mixte = [
    {"name": "x", "size": "n_rows", "type": "continuous"},
    {"name": "b", "size": "n_rows", "type": "binary"}
]

# Config MOO allégée pour des tests rapides
moo_fast = {"pop_size": 40, "n_gen": 40, "seed": 42, "verbose": False}

# ==========================================
# 2. SUITE DE TESTS (LES 6 SCÉNARIOS)
# ==========================================
test_cases = [
    {
        "title": "TEST 1 : Min Variance - Contraintes Continues Uniquement (Déterministe)",
        "config": {
            "decision_variables": vars_cont,
            "objective": {"type": "quadratic", "target_name": "Variance", "direction": "min"},
            "constraints": [
                {"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0}
            ]
        }
    },
    {
        "title": "TEST 2 : Min Variance - Mixtes (Continues + Binaires) (Déterministe)",
        "config": {
            "decision_variables": vars_mixte,
            "objective": {"type": "quadratic", "target_name": "Variance", "direction": "min"},
            "constraints": [
                {"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0},
                {"applied_to": "b", "attribute": "sum_all", "constraint_family": "cardinality", "is_strict": True, "bound_type": "eq", "value": 5.0},
                {"applied_to": "x", "attribute": "min_buy_in", "constraint_family": "min_buy_in", "is_strict": True, "bound_type": "min", "min_value": 0.05}
            ]
        }
    },
    {
        "title": "TEST 3 : Min Variance & Max Return - Contraintes Continues (MOO)",
        "config": {
            "decision_variables": vars_cont,
            "objectives": [
                {"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Risk"},
                {"type": "linear", "target_name": "Expected_Return", "direction": "max", "name": "Return"}
            ],
            "moo": moo_fast,
            "constraints": [
                {"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0},
                {"applied_to": "x", "attribute": "Sector", "targets": ["Tech"], "is_strict": True, "bound_type": "max", "value": 0.4}
            ]
        }
    },
    {
        "title": "TEST 4 : Min Variance & Max Return - Mixtes (MOO)",
        "config": {
            "decision_variables": vars_mixte,
            "objectives": [
                {"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Risk"},
                {"type": "linear", "target_name": "Expected_Return", "direction": "max", "name": "Return"}
            ],
            "moo": moo_fast,
            "constraints": [
                {"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0},
                {"applied_to": "b", "attribute": "sum_all", "constraint_family": "cardinality", "is_strict": True, "bound_type": "eq", "value": 5.0},
                {"applied_to": "x", "attribute": "min_buy_in", "constraint_family": "min_buy_in", "is_strict": True, "bound_type": "min", "min_value": 0.05}
            ]
        }
    },
    {
        "title": "TEST 5 : Min Variance & Pénalité Soft Continue (ESG) (MOO)",
        "config": {
            "decision_variables": vars_mixte,
            "objective": {"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Risk"},
            "moo": moo_fast,
            "constraints": [
                {"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0},
                {"applied_to": "b", "attribute": "sum_all", "constraint_family": "cardinality", "is_strict": True, "bound_type": "eq", "value": 5.0},
                {"applied_to": "x", "attribute": "ESG_Score", "is_strict": False, "bound_type": "min", "value": 85.0}
            ]
        }
    },
    {
        "title": "TEST 6 : Min Variance & Pénalité Soft Binaire (Secteur) (MOO)",
        "config": {
            "decision_variables": vars_mixte,
            "objective": {"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Risk"},
            "moo": moo_fast,
            "constraints": [
                {"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0},
                {"applied_to": "b", "attribute": "sum_all", "constraint_family": "cardinality", "is_strict": True, "bound_type": "eq", "value": 5.0},
                {"applied_to": "b", "attribute": "Sector", "targets": ["Energy"], "constraint_family": "cardinality", "is_strict": False, "bound_type": "max", "value": 1.0}
            ]
        }
    }
]

# ==========================================
# 3. EXÉCUTION
# ==========================================
if __name__ == "__main__":
    executor = OptimizationExecutor(verbose=False)
    for i, test in enumerate(test_cases):
        print(f"\n{'='*60}\n{test['title']}\n{'='*60}")
        try:
            res = executor.run(test["config"], matrix_inputs, df_data)
            
            print(f"Statut  : {res.get('status', 'Success')}")
            
            if "type" in res and res["type"] == "pareto":
                pts = res.get("pareto_points", [])
                print(f"Points Pareto trouvés : {len(pts)}")
                if pts:
                    print(f"Objectif 1 (Min/Max)  : [{pts[0]['objectives_minimised'][0]:.4f} ... {pts[-1]['objectives_minimised'][0]:.4f}]")
                    if res.get("n_soft_constraints", 0) > 0:
                        print(f"Pénalités Soft (Point 1) : {pts[0]['soft_losses_raw']}")
            else:
                x_vals = np.array(res.get("x_values", []))
                print(f"Objectif (Valeur)     : {res.get('objective', 0.0):.6f}")
                print(f"Somme des poids       : {np.sum(x_vals):.4f}")
                if "selection" in res:
                    print(f"Actifs sélectionnés   : {int(np.sum(res['selection']))}")
                    
        except Exception as e:
            print(f"ERREUR FATALE SUR LE TEST {i+1} : {str(e)}")