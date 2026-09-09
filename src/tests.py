import pandas as pd
import numpy as np
from src.agent_executor import OptimizationExecutor

def generate_mock_data(n_assets=100):
    """Génère de fausses données financières pour tester le moteur."""
    np.random.seed(42)
    
    # Données factorielles (Pandas)
    df = pd.DataFrame({
        "Ticker": [f"Asset_{i}" for i in range(n_assets)],
        "Expected_Return": np.random.uniform(0.01, 0.15, n_assets),
        "ESG_Score": np.random.uniform(40, 100, n_assets),
        "Sector": np.random.choice(["Tech", "Health", "Energy", "Finance"], n_assets)
    })
    
    # Matrice de Covariance (CasADi / Numpy)
    # Création d'une matrice définie positive
    random_matrix = np.random.randn(n_assets, n_assets)
    covariance_matrix = np.dot(random_matrix, random_matrix.T) * 0.001
    
    matrix_inputs = {
        "Sigma": covariance_matrix.tolist()
    }
    
    return df, matrix_inputs

def test_scenario_1_standard_cvar(df, matrix_inputs):
    """
    Test 1 : Minimiser le CVaR avec des contraintes continues standards.
    Pas de variables binaires.
    """
    print("\n" + "="*50)
    print("TEST 1 : Minimisation CVaR (100% Investi, Max 5% par actif)")
    
    # On simule 500 jours de rendements passés pour le CVaR
    n_assets = len(df)
    scenarios = np.random.normal(0.0005, 0.02, (500, n_assets)).tolist()
    matrix_inputs["Historical_Scenarios"] = scenarios
    
    config = {
        "decision_variables": [
            {"name": "x", "size": "n_rows", "type": "continuous"}
        ],
        "objective": {
            "type": "cvar",
            "target_name": "Historical_Scenarios",
            "direction": "min"
        },
        "constraints": [
            {"applied_to": "x", "attribute": "sum_all", "constraint_family": "standard", "bound_type": "eq", "value": 1.0, "is_strict": True},
            {"applied_to": "x", "attribute": "element", "constraint_family": "standard", "bound_type": "max", "value": 0.05, "is_strict": True}
        ]
    }
    
    executor = OptimizationExecutor()
    res = executor.run(config, matrix_inputs, df)
    
    print(f"Status: {res['status']}")
    if res['success']:
        print(f"Objectif final (CVaR): {res['objective']:.4f}")
        weights = np.array(res['x_values'])
        print(f"Somme des poids : {np.sum(weights):.2f} (Attendu: 1.0)")
        print(f"Poids Max : {np.max(weights):.4f} (Attendu: <= 0.05)")


def test_scenario_2_cardinality_and_min_buy_in(df, matrix_inputs):
    """
    Test 2 : Moteur Industriel.
    Maximiser le rendement avec exactement 15 actifs.
    Si un actif est sélectionné, il doit peser au moins 3%.
    """
    print("\n" + "="*50)
    print("TEST 2 : Solve & Polish (Exactement 15 actifs, Min Buy-in 3%)")
    
    config = {
        "decision_variables": [
            {"name": "x", "size": "n_rows", "type": "continuous"},
            {"name": "b", "size": "n_rows", "type": "binary"}
        ],
        "objective": {
            "type": "linear",
            "target_name": "Expected_Return",
            "direction": "max"
        },
        "constraints": [
            # 100% investi (CasADi)
            {"applied_to": "x", "attribute": "sum_all", "constraint_family": "standard", "bound_type": "eq", "value": 1.0, "is_strict": True},
            # Min Buy-in 3% (CasADi / ALM)
            {"applied_to": "x", "attribute": "min_buy_in", "constraint_family": "min_buy_in", "bound_type": "min", "min_value": 0.03, "is_strict": True},
            # Exactement 15 actifs (SciPy ILP)
            {"applied_to": "b", "attribute": "sum_all", "constraint_family": "cardinality", "bound_type": "eq", "value": 15.0, "is_strict": True}
        ]
    }
    
    executor = OptimizationExecutor()
    res = executor.run(config, matrix_inputs, df)
    
    print(f"Status: {res['status']}")
    if res['success']:
        weights = np.array(res['x_values'])
        active_assets = np.sum(weights > 1e-4) # Actifs non-nuls
        print(f"Nombre d'actifs sélectionnés : {active_assets} (Attendu: 15)")
        print(f"Somme des poids : {np.sum(weights):.2f} (Attendu: 1.0)")
        print(f"Plus petit poids investi : {np.min(weights[weights > 1e-4]):.4f} (Attendu: >= 0.03)")


def test_scenario_3_anticipation_soft_constraints(df, matrix_inputs):
    """
    Test 3 : Vérification du bypass Pymoo.
    Minimiser la variance.
    Contrainte dure : 100% investi.
    Contrainte douce : Secteur Tech = 40% (Doit être ignorée par CasADi).
    """
    print("\n" + "="*50)
    print("TEST 3 : Anticipation Multi-Objectif (Soft Constraints Filter)")
    
    config = {
        "decision_variables": [
            {"name": "x", "size": "n_rows", "type": "continuous"}
        ],
        "objective": {
            "type": "quadratic",
            "target_name": "Sigma",
            "direction": "min"
        },
        "constraints": [
            # Hard (Sera respectée)
            {"applied_to": "x", "attribute": "sum_all", "constraint_family": "standard", "bound_type": "eq", "value": 1.0, "is_strict": True},
            # Soft (Sera ignorée par le code actuel, en attente de Pymoo)
            {"applied_to": "x", "attribute": "Sector", "targets": ["Tech"], "constraint_family": "standard", "bound_type": "eq", "value": 0.40, "is_strict": False}
        ]
    }
    
    executor = OptimizationExecutor()
    res = executor.run(config, matrix_inputs, df)
    
    print(f"Status: {res['status']}")
    if res['success']:
        weights = np.array(res['x_values'])
        tech_mask = df["Sector"] == "Tech"
        tech_exposure = np.sum(weights[tech_mask])
        print(f"Somme des poids : {np.sum(weights):.2f} (Attendu: 1.0)")
        print(f"Exposition Tech réelle : {tech_exposure:.4f} (Preuve que la soft constraint a bien été esquivée)")


if __name__ == "__main__":
    df_market, matrices = generate_mock_data(n_assets=100)
    
    test_scenario_1_standard_cvar(df_market, matrices)
    test_scenario_2_cardinality_and_min_buy_in(df_market, matrices)
    test_scenario_3_anticipation_soft_constraints(df_market, matrices)