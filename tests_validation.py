import numpy as np
import pandas as pd
import warnings

# Suppression des warnings Pandas pour un affichage propre
warnings.filterwarnings('ignore')
from src.agent_executor import OptimizationExecutor


# =====================================================================
# 1. GÉNÉRATION DE L'UNIVERS (Données Synthétiques Rigoureuses)
# =====================================================================
def generate_institutional_universe(n_assets=250):
    np.random.seed(42)
    
    n_standard = n_assets - 3 # On garde 3 places pour le cash
    tickers = [f"TICKER_{i:03d}" for i in range(n_standard)]
    
    # Mix institutionnel : 60% Actions, 30% Obligations, 10% Dérivés
    types = np.random.choice(["EQUITY", "BOND", "DERIVATIVE"], size=n_standard, p=[0.6, 0.3, 0.1])
    currencies = np.random.choice(["USD", "EUR", "GBP"], size=n_standard, p=[0.7, 0.2, 0.1])
    
    df = pd.DataFrame({
        "Ticker": tickers,
        "InstrumentType": types,
        "Currency": currencies,
        "Sector": np.random.choice(["Tech", "Health", "Fin", "Energy", "Cons"], size=n_standard),
        "Expected_Return": np.random.normal(0.06, 0.15, n_standard),
    })
    
    # Ajout des poches de liquidité (Obligatoires pour le multi-devises)
    cash_data = pd.DataFrame({
        "Ticker": ["CASH_USD", "CASH_EUR", "CASH_GBP"],
        "InstrumentType": ["CASH", "CASH", "CASH"],
        "Currency": ["USD", "EUR", "GBP"],
        "Sector": ["Cash", "Cash", "Cash"],
        "Expected_Return": [0.03, 0.02, 0.04],
    })
    df = pd.concat([df, cash_data], ignore_index=True)
    
    # --- Modélisation Financière Avancée ---
    # Cash_Impact : 0.0 pour les dérivés (ils ne consomment pas de cash à l'achat)
    df["Cash_Impact"] = np.where(df["InstrumentType"] == "DERIVATIVE", 0.0, 1.0)
    df["Exposure"] = 1.0 
    
    fx_rates = {"USD": 1.0, "EUR": 1.1, "GBP": 1.25} # Base = USD
    df["FXRateToBase"] = df["Currency"].map(fx_rates)
    df["FX_Spread"] = 0.0005 # 5 bps de spread FX
    
    n_total = len(df)
    
    # Matrice de Covariance (Semi-définie positive stricte)
    factor = np.random.randn(n_total, 5)
    cov_matrix = np.dot(factor, factor.T) * 0.02
    np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + 0.05)
    
    # Portefeuille initial (w0) : 100% investi, 0% dérivés initialement
    w0 = np.random.uniform(0, 1, n_total)
    w0[df["InstrumentType"] == "DERIVATIVE"] = 0.0
    w0 = w0 / np.sum(w0)
    
    # Benchmark
    benchmark = np.random.uniform(0, 1, n_total)
    benchmark = benchmark / np.sum(benchmark)
    
    matrix_inputs = {
        "Variance": cov_matrix.tolist(),
        "w0": w0.tolist(),
        "Benchmark": benchmark.tolist()
    }
    
    return df, matrix_inputs

# =====================================================================
# 2. FONCTION D'AFFICHAGE D'AUDIT
# =====================================================================
def print_audit_report(res, title):
    print(f"\n{'='*60}")
    print(f" RESULTATS : {title}")
    print(f"{'='*60}")
    
    if not res.get("success"):
        print(f"[ECHEC] Raison : {res.get('status')}")
        return
        
    print(f"Status Solveur : {res.get('status')}")
    print(f"Moteur Utilisé : {res.get('type')}")
    print(f"NAV Initiale   : {res.get('nav0'):,.0f} $")
    print(f"Cashflow       : {res.get('cashflow_amount'):,.0f} $")
    print(f"NAV Finale     : {res.get('nav1'):,.0f} $")
    
    if "tc_total" in res:
        print(f"Frais Titres   : {res.get('tc_total'):,.0f} $")
        
    if "n_trades_active" in res:
        print(f"Nb de Trades   : {res.get('n_trades_active')} actifs touchés")
        
    # Analyse de la conservation du budget
    w_expo = np.sum(res.get("weights_final", []))
    w_bilan = np.sum(res.get("weights_bilan_final", []))
    print(f"\nSomme (Exposure): {w_expo:.6f} (Doit faire 1.0 ou s'en approcher)")
    print(f"Somme (Bilan)   : {w_bilan:.6f} (Doit faire 1.0 strictement)")
    
    print("\n--- Top 5 des Transactions Nettes (USD) ---")
    trades = np.array(res.get("trades_net", []))
    active_indices = np.argsort(np.abs(trades))[::-1][:5]
    for idx in active_indices:
        t_val = trades[idx]
        if abs(t_val) > 1e-2:
            direction = "ACHAT" if t_val > 0 else "VENTE"
            print(f"Index {idx:03d} | {direction} : {abs(t_val):,.0f} $")

# =====================================================================
# 3. SCÉNARIOS DE TESTS
# =====================================================================
def run_tests():
    print("Génération de l'univers de 250 actifs (Actions, Bonds, Dérivés, Cash FX)...")
    df_data, matrix_inputs = generate_institutional_universe(250)
    executor = OptimizationExecutor(verbose=False)
    
    nav_initiale = 100_000_000.0 # 100 Millions USD

    # -----------------------------------------------------------------
    # TEST 1 : INFLOW MASSIF + CONTINU + FX
    # Modélise une injection de 10M$ à réinvestir avec minimisation de la Tracking Error.
    # Aucun binaire utilisé, l'algo CasADi classique est appelé.
    # -----------------------------------------------------------------
    print("\n[Lancement du Test 1] - Inflow Continu (10M$)")
    cfg_test_1 = {
        "cashflow": {
            "nav0": nav_initiale,
            "amount": 10_000_000.0,
            "base_currency": "USD",
            "tc_enabled": True,
            "tc_rate": 0.0010, # 10 bps
            "use_foreign_cash": True
        },
        "objective": {
            "type": "tracking_error",
            "target_name": "Variance",
            "direction": "min"
        },
        "constraints": [
            {"is_strict": True, "applied_to": "x", "attribute": "element", "bound_type": "min", "value": 0.0} # Long Only
        ]
    }
    res_1 = executor.run(cfg_test_1, matrix_inputs, df_data)
    print_audit_report(res_1, "TEST 1 : Cashflow Continu (TE Minimisation)")


    # -----------------------------------------------------------------
    # TEST 2 : OUTFLOW + MIP (CARDINALITÉ & MIN TICKET)
    # Modélise un rachat de 5M$. Le gérant veut lever ce cash en liquidant 
    # ou réduisant un MAXIMUM de 10 lignes (Cardinalité), avec des ordres
    # d'au moins 200 000 $ par ligne (Min Ticket) pour éviter de saupoudrer.
    # -----------------------------------------------------------------
    print("\n[Lancement du Test 2] - Outflow MIP (Rachat 5M$, Max 10 trades, Ticket > 200k$)")
    cfg_test_2 = {
        "cashflow": {
            "nav0": nav_initiale,
            "amount": -5_000_000.0,
            "base_currency": "USD",
            "tc_enabled": True,
            "tc_rate": 0.0015, # 15 bps
            "use_foreign_cash": True
        },
        "objective": {
            "type": "tracking_error",
            "target_name": "Variance",
            "direction": "min"
        },
        "constraints": [
            {"is_strict": True, "applied_to": "x", "attribute": "element", "bound_type": "min", "value": 0.0},
            
            # Limite mathématique stricte sur la variable binaire d'action (z / b)
            {"is_strict": True, "applied_to": "b", "constraint_family": "cardinality", 
             "attribute": "sum_all", "bound_type": "max", "value": 10},
             
            # Seuil d'action (Min Ticket) : 200k$ divisé par la nouvelle NAV
            {"is_strict": True, "applied_to": "b", "constraint_family": "min_ticket", 
             "min_value": 200_000.0 / (nav_initiale - 5_000_000.0)}
        ]
    }
    # L'Aiguilleur (agent_executor.py) détectera le "b" et la "cardinality" et routra vers solve_cashflow_mip_optimization
    res_2 = executor.run(cfg_test_2, matrix_inputs, df_data)
    print_audit_report(res_2, "TEST 2 : Cashflow MIP (Rachats Concentrés)")

if __name__ == "__main__":
    run_tests()