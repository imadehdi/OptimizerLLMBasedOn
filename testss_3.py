import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings('ignore')

try:
    from src.agent_executor import OptimizationExecutor
except ImportError:
    print("Assurez-vous de lancer ce script depuis la racine du projet.")
    exit(1)

# =====================================================================
# 1. GÉNÉRATION DE L'UNIVERS DE RUPTURE (500 ACTIFS)
# =====================================================================
def generate_stress_universe(n_assets=500):
    np.random.seed(99) # Nouvelle seed
    
    n_standard = n_assets - 3
    tickers = [f"TICK_{i:03d}" for i in range(n_standard)]
    types = np.random.choice(["EQUITY", "BOND", "DERIVATIVE"], size=n_standard, p=[0.7, 0.2, 0.1])
    currencies = np.random.choice(["USD", "EUR", "GBP"], size=n_standard, p=[0.6, 0.3, 0.1])
    
    df = pd.DataFrame({
        "Ticker": tickers,
        "InstrumentType": types,
        "Currency": currencies,
        "Sector": np.random.choice(["Tech", "Health", "Fin"], size=n_standard),
        "Expected_Return": np.random.normal(0.06, 0.15, n_standard),
    })
    
    cash_data = pd.DataFrame({
        "Ticker": ["CASH_USD", "CASH_EUR", "CASH_GBP"],
        "InstrumentType": ["CASH", "CASH", "CASH"],
        "Currency": ["USD", "EUR", "GBP"],
        "Sector": ["Cash", "Cash", "Cash"],
        "Expected_Return": [0.03, 0.02, 0.04],
    })
    df = pd.concat([df, cash_data], ignore_index=True)
    
    df["Cash_Impact"] = np.where(df["InstrumentType"] == "DERIVATIVE", 0.0, 1.0)
    df["Exposure"] = 1.0 
    df["FXRateToBase"] = df["Currency"].map({"USD": 1.0, "EUR": 1.1, "GBP": 1.25})
    df["FX_Spread"] = 0.0005 
    
    n_total = len(df)
    
    factor = np.random.randn(n_total, 10)
    cov_matrix = np.dot(factor, factor.T) * 0.02
    np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + 0.05)
    
    # LE CHOC DE MARCHÉ : w0 est sur les 50 premiers actifs, Benchmark sur les 50 derniers.
    w0 = np.zeros(n_total)
    w0[:50] = np.random.uniform(0.1, 1.0, 50)
    w0 = w0 / np.sum(w0)
    
    benchmark = np.zeros(n_total)
    # Les 50 derniers avant le cash
    benchmark[n_standard-50:n_standard] = np.random.uniform(0.1, 1.0, 50) 
    benchmark = benchmark / np.sum(benchmark)
    
    matrix_inputs = {
        "Variance": cov_matrix.tolist(),
        "w0": w0.tolist(),
        "Benchmark": benchmark.tolist()
    }
    
    return df, matrix_inputs

# =====================================================================
# 2. FONCTION D'AFFICHAGE D'AUDIT AVEC CHRONO
# =====================================================================
def print_audit_report(res, title, exec_time):
    print(f"\n{'='*65}")
    print(f" RESULTATS : {title}")
    print(f"{'='*65}")
    print(f"Temps d'exécution : {exec_time:.4f} secondes")
    
    if not res.get("success"):
        print(f"[ECHEC] Raison : {res.get('status')}")
        return
        
    print(f"Status Solveur : {res.get('status')}")
    print(f"Moteur Utilisé : {res.get('type')}")
    
    if "n_trades_active" in res:
        print(f"Nb de Trades   : {res.get('n_trades_active')} actifs touchés")
        
    w_expo = np.sum(res.get("weights_final", []))
    w_bilan = np.sum(res.get("weights_bilan_final", []))
    print(f"\nSomme (Exposure): {w_expo:.6f}")
    print(f"Somme (Bilan)   : {w_bilan:.6f}")
    
    print("\n--- Top 5 des Transactions Nettes (USD) ---")
    trades = np.array(res.get("trades_net", []))
    active_indices = np.argsort(np.abs(trades))[::-1][:5]
    for idx in active_indices:
        t_val = trades[idx]
        if abs(t_val) > 1e-2:
            direction = "ACHAT" if t_val > 0 else "VENTE"
            print(f"Index {idx:03d} | {direction} : {abs(t_val):,.0f} $")

# =====================================================================
# 3. SCÉNARIOS DE STRESS
# =====================================================================
def run_stress_tests():
    print("Génération de l'univers de STRESS (500 actifs, Choc de Benchmark)...")
    df_data, matrix_inputs = generate_stress_universe(500)
    executor = OptimizationExecutor(verbose=False)
    
    nav_initiale = 100_000_000.0

    # -----------------------------------------------------------------
    # TEST 1 : REBALANCEMENT CONTINU MASSIF
    # Le solveur est obligé de vendre les 50 premiers actifs et d'acheter 
    # les 50 derniers pour coller au nouveau Benchmark.
    # -----------------------------------------------------------------
    print("\n[Lancement du Stress Test 1] - Rebalancement Continu Total")
    cfg_test_1 = {
        "cashflow": {
            "nav0": nav_initiale,
            "amount": 0.0, # Pas d'inflow, juste de la rotation
            "base_currency": "USD",
            "tc_enabled": True,
            "tc_rate": 0.0010,
            "use_foreign_cash": True
        },
        "objective": {
            "type": "tracking_error",
            "target_name": "Variance",
            "direction": "min"
        },
        "constraints": [
            {"is_strict": True, "applied_to": "x", "attribute": "element", "bound_type": "min", "value": 0.0}
        ]
    }
    
    t0 = time.time()
    res_1 = executor.run(cfg_test_1, matrix_inputs, df_data)
    t1 = time.time()
    print_audit_report(res_1, "STRESS 1 : Rotation Massive Continue", t1 - t0)


    # -----------------------------------------------------------------
    # TEST 2 : LE CAUCHEMAR COMBINATOIRE (MIP)
    # Même problème (vendre les 50 premiers pour acheter les 50 derniers),
    # MAIS on lui interdit de faire plus de 15 trades au total !
    # Le solveur DOIT choisir les 15 actifs qui répliquent le mieux les 100 actifs impliqués.
    # -----------------------------------------------------------------
    print("\n[Lancement du Stress Test 2] - Rotation MIP Restreinte (Max 15 trades)")
    cfg_test_2 = {
        "cashflow": {
            "nav0": nav_initiale,
            "amount": 0.0,
            "base_currency": "USD",
            "tc_enabled": True,
            "tc_rate": 0.0010,
            "use_foreign_cash": True
        },
        "objective": {
            "type": "tracking_error",
            "target_name": "Variance",
            "direction": "min"
        },
        "constraints": [
            {"is_strict": True, "applied_to": "x", "attribute": "element", "bound_type": "min", "value": 0.0},
            {"is_strict": True, "applied_to": "b", "constraint_family": "cardinality", 
             "attribute": "sum_all", "bound_type": "max", "value": 15},
        ]
    }
    
    t0 = time.time()
    res_2 = executor.run(cfg_test_2, matrix_inputs, df_data)
    t1 = time.time()
    print_audit_report(res_2, "STRESS 2 : Rotation MIP (Max 15 trades)", t1 - t0)

if __name__ == "__main__":
    run_stress_tests()