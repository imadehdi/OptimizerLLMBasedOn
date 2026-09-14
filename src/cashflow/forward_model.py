import casadi as ca
import numpy as np
import pandas as pd

def get_cash_index(df_data: pd.DataFrame, cash_ticker: str = "CASH") -> int:
    """
    Identifie dynamiquement l'indice de l'actif numéraire dans l'univers.
    """
    if "Ticker" not in df_data.columns:
        raise ValueError("df_data doit contenir une colonne 'Ticker'.")
        
    tickers = df_data["Ticker"].astype(str).str.upper().tolist()
    cash_ticker_up = cash_ticker.upper()
    
    if cash_ticker_up not in tickers:
        raise ValueError(f"Ticker '{cash_ticker}' introuvable dans df_data. Requis pour le Cashflow.")
        
    return tickers.index(cash_ticker_up)

def build_cashflow_expressions(trades_t, w0, nav0, cashflow_amount, cash_idx):
    """
    Génère les expressions CasADi reliant les trades t aux poids finaux w1.
    Hypothèse actuelle : TC = 0 (Conservation parfaite de l'apport).
    """
    nav1 = float(nav0) + float(cashflow_amount)
    n_assets = len(w0)
    
    # Séparation des indices risqués vs numéraire
    non_cash_idx = [i for i in range(n_assets) if i != cash_idx]
    
    w1_expr = ca.MX.zeros(n_assets, 1)
    v1_expr = ca.MX.zeros(n_assets, 1)
    
    # 1. Évolution de la poche risquée (Montant = Ancien Montant + Trade)
    for k, idx in enumerate(non_cash_idx):
        v1_expr[idx] = w0[idx] * float(nav0) + trades_t[k]
        w1_expr[idx] = v1_expr[idx] / nav1
        
    # 2. Évolution de la poche numéraire (Équation de budget autofinancé)
    sum_trades = ca.sum1(trades_t)
    v1_expr[cash_idx] = w0[cash_idx] * float(nav0) + float(cashflow_amount) - sum_trades
    w1_expr[cash_idx] = v1_expr[cash_idx] / nav1
    
    return {
        "w1": w1_expr,
        "v1": v1_expr,
        "cash1": v1_expr[cash_idx],
        "non_cash_idx": non_cash_idx
    }

def cashflow_numpy(trades_t, w0, nav0, cashflow_amount, cash_idx):
    """
    Rétro-calcul déterministe pour le reporting final (Numpy).
    """
    nav1 = float(nav0) + float(cashflow_amount)
    n_assets = len(w0)
    trades_t = np.asarray(trades_t, dtype=float).flatten()
    
    non_cash_idx = [i for i in range(n_assets) if i != cash_idx]
    
    w1 = np.zeros(n_assets, dtype=float)
    v1 = np.zeros(n_assets, dtype=float)
    
    for k, idx in enumerate(non_cash_idx):
        v1[idx] = w0[idx] * float(nav0) + trades_t[k]
        w1[idx] = v1[idx] / nav1
        
    sum_trades = np.sum(trades_t)
    v1[cash_idx] = w0[cash_idx] * float(nav0) + float(cashflow_amount) - sum_trades
    w1[cash_idx] = v1[cash_idx] / nav1
    
    return {
        "w1": w1,
        "v1": v1,
        "nav1": nav1,
        "cash1": float(v1[cash_idx]),
        "non_cash_idx": non_cash_idx
    }