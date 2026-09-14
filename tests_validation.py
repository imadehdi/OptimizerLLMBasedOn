import numpy as np
import pandas as pd
from src.agent_executor import OptimizationExecutor

def create_mock_data():
    # Univers de 4 actifs risqués + 1 ligne CASH
    df_data = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "TSLA", "GOOG", "CASH"],
        "Expected_Return": [0.08, 0.07, 0.12, 0.06, 0.0],
        "Sector": ["Tech", "Tech", "Auto", "Tech", "Liquidity"]
    })
    
    # Matrice de covariance factice
    np.random.seed(42)
    A = np.random.randn(5, 5)
    cov = np.dot(A, A.T) / 100
    cov[4, :] = 0.0 # Le cash n'a pas de volatilité
    cov[:, 4] = 0.0
    
    # Portefeuille initial parfaitement investi (20% sur chaque actif)
    w0 = [0.20, 0.20, 0.20, 0.20, 0.20]
    
    matrix_inputs = {
        "Variance": cov.tolist(),
        "w0": w0
    }
    
    return df_data, matrix_inputs

def run_tests():
    df_data, matrix_inputs = create_mock_data()
    executor = OptimizationExecutor(verbose=True)
    
    nav0 = 1_000_000.0
    inflow = 200_000.0

    print("\n" + "="*60)
    print("TEST 1 : CASHFLOW CONTINU (CasADi)")
    print("Objectif : Déployer le numéraire pour maximiser le rendement.")
    print("="*60)
    
    cfg_continuous = {
        "cashflow": {"nav0": nav0, "amount": inflow},
        "objective": {"type": "linear", "target_name": "Expected_Return", "direction": "max"},
        "constraints": []
    }
    
    res_cont = executor.run(cfg_continuous, matrix_inputs, df_data)
    print(f"Statut : {res_cont.get('status')}")
    print(f"Poids Finaux : {np.round(res_cont.get('weights_final', []), 4)}")
    print(f"Trades Nominaux (€) : {np.round(res_cont.get('trades_full_vector', []), 0)}")
    
    
    print("\n" + "="*60)
    print("TEST 2 : CASHFLOW MIXED-INTEGER (CasADi + SciPy)")
    print("Objectif : Déployer le numéraire, mais concentrer le portefeuille sur 2 actifs maximum (Cardinalité).")
    print("="*60)
    
    cfg_mip = {
        "cashflow": {"nav0": nav0, "amount": inflow},
        "objective": {"type": "linear", "target_name": "Expected_Return", "direction": "max"},
        "constraints": [
            {"constraint_family": "cardinality", "applied_to": "b", "attribute": "sum_all", "bound_type": "max", "value": 2.0}
        ]
    }
    
    res_mip = executor.run(cfg_mip, matrix_inputs, df_data)
    print(f"Statut : {res_mip.get('status')}")
    print(f"Poids Finaux : {np.round(res_mip.get('weights_final', []), 4)}")
    print(f"Sélection Binaire : {res_mip.get('trade_active_binaries', [])}")
    
    
    print("\n" + "="*60)
    print("TEST 3 : CASHFLOW MULTI-OBJECTIF (Pymoo NSGA-II)")
    print("Objectif : Minimiser le Risque ET Maximiser le Rendement, sous apport de capital.")
    print("="*60)
    
    cfg_moo = {
        "cashflow": {"nav0": nav0, "amount": inflow},
        "objectives": [
            {"type": "quadratic", "target_name": "Variance", "direction": "min"},
            {"type": "linear", "target_name": "Expected_Return", "direction": "max"}
        ],
        "constraints": [],
        "moo": {"pop_size": 40, "n_gen": 50, "verbose": False}
    }
    
    res_moo = executor.run(cfg_moo, matrix_inputs, df_data)
    print(f"Statut : {res_moo.get('status')}")
    print(f"Portefeuilles Pareto trouvés : {len(res_moo.get('pareto_points', []))}")
    if res_moo.get('pareto_points'):
        pt = res_moo['pareto_points'][0]
        print(f"Exemple Portefeuille 1 - Poids Finaux : {np.round(pt['weights'], 4)}")

if __name__ == "__main__":
    run_tests()