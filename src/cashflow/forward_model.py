# src/cashflow/forward_model.py
from __future__ import annotations

import numpy as np
import casadi as ca
import pandas as pd


def get_cash_indices_by_ccy(df_data: pd.DataFrame, cash_ticker_prefix: str = "CASH") -> dict[str, int]:
    if "InstrumentType" not in df_data.columns:
        raise ValueError("df_data must contain column 'InstrumentType' for multi-cash.")
    if "Currency" not in df_data.columns:
        raise ValueError("df_data must contain column 'Currency' for multi-cash.")

    itype = df_data["InstrumentType"].astype(str).str.upper()
    is_cash = itype.eq("CASH")

    if "Ticker" in df_data.columns:
        tick = df_data["Ticker"].astype(str)
        is_cash = is_cash & tick.str.upper().str.startswith(str(cash_ticker_prefix).upper())

    cash_df = df_data[is_cash].copy()
    if cash_df.empty:
        raise ValueError("No CASH rows found. Need at least one CASH_<CCY> row.")

    out: dict[str, int] = {}
    for idx, row in cash_df.iterrows():
        ccy = str(row["Currency"]).upper().strip()
        if not ccy:
            raise ValueError(f"CASH row at index={idx} has empty Currency.")
        if ccy in out:
            raise ValueError(f"Duplicate CASH bucket for currency={ccy}.")
        out[ccy] = int(idx)

    return out


def get_non_cash_indices(df_data: pd.DataFrame) -> list[int]:
    if "InstrumentType" not in df_data.columns:
        raise ValueError("df_data must contain column 'InstrumentType'.")
    itype = df_data["InstrumentType"].astype(str).str.upper()
    return df_data.index[~itype.eq("CASH")].astype(int).tolist()


def _get_fx_to_base_map(df_data: pd.DataFrame, base_currency: str) -> dict[str, float]:
    base_ccy = str(base_currency).upper().strip()
    if "FXRateToBase" not in df_data.columns:
        return {base_ccy: 1.0}

    fx_map: dict[str, float] = {}
    for _, row in df_data[["Currency", "FXRateToBase"]].dropna().iterrows():
        ccy = str(row["Currency"]).upper().strip()
        if not ccy:
            continue
        try:
            fx = float(row["FXRateToBase"])
        except Exception:
            continue
        if fx <= 0:
            raise ValueError(f"FXRateToBase must be >0. Got {fx} for currency={ccy}.")
        fx_map.setdefault(ccy, fx)

    fx_map.setdefault(base_ccy, 1.0)
    return fx_map


def build_cashflow_expressions_multi_cash(
    *,
    t_buy: ca.MX,             # size n_non_cash (in BASE currency notional)
    t_sell: ca.MX,            # size n_non_cash (in BASE currency notional)
    t_fx_out_by_ccy: dict = None, # Base -> CCY (MX vars)
    t_fx_in_by_ccy: dict = None,  # CCY -> Base (MX vars)
    w0: np.ndarray,           # size n_rows (incl all CASH buckets), weights in BASE NAV
    nav0: float,
    cashflow_amount: float,   # in BASE currency
    df_data: pd.DataFrame,
    base_currency: str,
    tc_enabled: bool = False,
    tc_rate: float = 0.0,
    tc_buy_rate=None,
    tc_sell_rate=None,
) -> dict:
    base_ccy = str(base_currency).upper().strip()
    if not base_ccy:
        raise ValueError("base_currency is required.")

    n_rows = len(df_data)
    cash_idx_by_ccy = get_cash_indices_by_ccy(df_data)
    if base_ccy not in cash_idx_by_ccy:
        raise ValueError(f"Missing CASH bucket for base currency {base_ccy}.")

    non_cash_idx = get_non_cash_indices(df_data)
    n_non_cash = len(non_cash_idx)
    if int(t_buy.size1()) != n_non_cash or int(t_sell.size1()) != n_non_cash:
        raise ValueError(f"t_buy/t_sell must be size n_non_cash={n_non_cash}.")

    t_fx_out_by_ccy = t_fx_out_by_ccy or {}
    t_fx_in_by_ccy = t_fx_in_by_ccy or {}

    w0 = np.asarray(w0, dtype=float).reshape(-1)
    nav1 = float(nav0 + cashflow_amount)
    
    # ---------------------------------------------------------
    # PARAMÈTRES DÉRIVÉS ET FX (Extraction)
    # ---------------------------------------------------------
    c_impact = np.asarray(df_data["Cash_Impact"].fillna(1.0).values, dtype=float) if "Cash_Impact" in df_data.columns else np.ones(n_rows)
    e_impact = np.asarray(df_data["Exposure"].fillna(1.0).values, dtype=float) if "Exposure" in df_data.columns else np.ones(n_rows)
    
    fx_spread_by_ccy = {}
    if "FX_Spread" in df_data.columns:
        for ccy, idx in cash_idx_by_ccy.items():
            val = df_data.loc[idx, "FX_Spread"]
            fx_spread_by_ccy[ccy] = float(val) if pd.notna(val) else 0.0

    v0 = ca.DM(w0.reshape(-1, 1)) * float(nav0)
    
    # ---------------------------------------------------------
    # VECTORISATION : Application instantanée des trades titres
    # ---------------------------------------------------------
    t_net = t_buy - t_sell
    e_impact_nc = ca.DM(e_impact[non_cash_idx])
    c_impact_nc = ca.DM(c_impact[non_cash_idx])

    # Création d'un vecteur plein de zéros pour garantir la propreté du graphe CasADi
    delta_expo = ca.MX.zeros(n_rows, 1)
    delta_expo[non_cash_idx] = t_net * e_impact_nc
    v1_expo = v0 + delta_expo

    delta_bilan = ca.MX.zeros(n_rows, 1)
    delta_bilan[non_cash_idx] = t_net * c_impact_nc
    v1_bilan = v0 + delta_bilan

    # ---------------------------------------------------------
    # VECTORISATION : Flux Cashflow et devises (Dot Product)
    # ---------------------------------------------------------
    base_cash_idx = cash_idx_by_ccy[base_ccy]
    v1_bilan[base_cash_idx] += float(cashflow_amount)

    ccy_series = df_data.loc[non_cash_idx, "Currency"].astype(str).str.upper().str.strip().values
    t_cash_impact = t_net * c_impact_nc
    
    for ccy, cidx in cash_idx_by_ccy.items():
        # Masque booléen transformé en Float, ultra rapide (ex: [0, 1, 1, 0, 0])
        mask = (ccy_series == ccy).astype(float).reshape(1, -1)
        
        # Produit matriciel scalaire : somme instantanée des drains de cash de cette devise
        ccy_cash_drain = ca.mtimes(ca.DM(mask), t_cash_impact)
        v1_bilan[cidx] -= ccy_cash_drain

    # ---------------------------------------------------------
    # Exécution des conversions FX Spot avec spread
    # ---------------------------------------------------------
    for ccy, idx in cash_idx_by_ccy.items():
        if ccy == base_ccy:
            continue
        spread = fx_spread_by_ccy.get(ccy, 0.0)
        fx_out = t_fx_out_by_ccy.get(ccy, 0.0)
        fx_in = t_fx_in_by_ccy.get(ccy, 0.0)

        # Poche Centrale (Base) : Débitée du montant brut, créditée du net
        v1_bilan[base_cash_idx] = v1_bilan[base_cash_idx] - fx_out + fx_in * (1.0 - spread)
        # Poche Étrangère (CCY) : Débitée du brut, créditée du net
        v1_bilan[idx] = v1_bilan[idx] + fx_out * (1.0 - spread) - fx_in

    # ---------------------------------------------------------
    # Synchronisation de l'exposition du cash (Cash Expo = Cash Bilan)
    # ---------------------------------------------------------
    for ccy, idx in cash_idx_by_ccy.items():
        v1_expo[idx] = v1_bilan[idx]

    # ---------------------------------------------------------
    # Coûts de transaction titres (appliqués sur le Notionnel total)
    # ---------------------------------------------------------
    if tc_enabled:
        buy_r = float(tc_rate if tc_buy_rate is None else tc_buy_rate)
        sell_r = float(tc_rate if tc_sell_rate is None else tc_sell_rate)
        tc_total = buy_r * ca.sum1(t_buy) + sell_r * ca.sum1(t_sell)
    else:
        tc_total = ca.MX(0)

    cash_pre_by_ccy = {ccy: v1_bilan[idx] for ccy, idx in cash_idx_by_ccy.items()}
    cash_post_by_ccy = dict(cash_pre_by_ccy)
    if tc_enabled:
        cash_post_by_ccy[base_ccy] = cash_post_by_ccy[base_ccy] - tc_total

    w1_expo = v1_expo / float(nav1)
    w1_bilan = v1_bilan / float(nav1)

    # Reporting local cash amounts
    fx_map = _get_fx_to_base_map(df_data, base_currency=base_ccy)
    cash_pre_local_by_ccy, cash_post_local_by_ccy = {}, {}
    for ccy in cash_idx_by_ccy.keys():
        fx = float(fx_map.get(ccy, 1.0))
        cash_pre_local_by_ccy[ccy] = cash_pre_by_ccy[ccy] / fx
        cash_post_local_by_ccy[ccy] = cash_post_by_ccy[ccy] / fx

    return {
        "nav1": float(nav1),
        "w1_expo": w1_expo,
        "w1_bilan": w1_bilan,
        "v1_expo": v1_expo,
        "v1_bilan": v1_bilan,
        "tc_total": tc_total,
        "t_net": t_net,
        "non_cash_idx": non_cash_idx,
        "cash_idx_by_ccy": cash_idx_by_ccy,
        "base_currency": base_ccy,
        "cash_pre_by_ccy": cash_pre_by_ccy,
        "cash_post_by_ccy": cash_post_by_ccy,
        "cash_pre": cash_pre_by_ccy[base_ccy],
        "cash_post": cash_post_by_ccy[base_ccy],
        "cash_pre_local_by_ccy": cash_pre_local_by_ccy,
        "cash_post_local_by_ccy": cash_post_local_by_ccy,
    }


def cashflow_numpy_multi_cash(
    *,
    t_buy: np.ndarray,
    t_sell: np.ndarray,
    t_fx_out_by_ccy: dict = None,
    t_fx_in_by_ccy: dict = None,
    w0: np.ndarray,
    nav0: float,
    cashflow_amount: float,
    df_data: pd.DataFrame,
    base_currency: str,
    tc_enabled: bool = False,
    tc_rate: float = 0.0,
    tc_buy_rate=None,
    tc_sell_rate=None,
) -> dict:
    base_ccy = str(base_currency).upper().strip()
    cash_idx_by_ccy = get_cash_indices_by_ccy(df_data)
    non_cash_idx = get_non_cash_indices(df_data)
    n_rows = len(df_data)

    t_fx_out_by_ccy = t_fx_out_by_ccy or {}
    t_fx_in_by_ccy = t_fx_in_by_ccy or {}

    c_impact = np.asarray(df_data["Cash_Impact"].fillna(1.0).values, dtype=float) if "Cash_Impact" in df_data.columns else np.ones(n_rows)
    e_impact = np.asarray(df_data["Exposure"].fillna(1.0).values, dtype=float) if "Exposure" in df_data.columns else np.ones(n_rows)

    fx_spread_by_ccy = {}
    if "FX_Spread" in df_data.columns:
        for ccy, idx in cash_idx_by_ccy.items():
            val = df_data.loc[idx, "FX_Spread"]
            fx_spread_by_ccy[ccy] = float(val) if pd.notna(val) else 0.0

    t_buy = np.asarray(t_buy, dtype=float).reshape(-1)
    t_sell = np.asarray(t_sell, dtype=float).reshape(-1)
    w0 = np.asarray(w0, dtype=float).reshape(-1)

    nav1 = float(nav0 + cashflow_amount)
    v0 = w0 * float(nav0)
    
    v1_expo = v0.copy()
    v1_bilan = v0.copy()

    # ---------------------------------------------------------
    # VECTORISATION NUMPY : Application instantanée
    # ---------------------------------------------------------
    t_net = t_buy - t_sell
    v1_expo[non_cash_idx] += t_net * e_impact[non_cash_idx]
    v1_bilan[non_cash_idx] += t_net * c_impact[non_cash_idx]

    base_cash_idx = cash_idx_by_ccy[base_ccy]
    v1_bilan[base_cash_idx] += float(cashflow_amount)

    ccy_series = df_data.loc[non_cash_idx, "Currency"].astype(str).str.upper().str.strip().values
    t_cash_impact = t_net * c_impact[non_cash_idx]

    for ccy, cidx in cash_idx_by_ccy.items():
        mask = (ccy_series == ccy).astype(float)
        v1_bilan[cidx] -= np.sum(t_cash_impact * mask)

    for ccy, idx in cash_idx_by_ccy.items():
        if ccy == base_ccy:
            continue
        spread = fx_spread_by_ccy.get(ccy, 0.0)
        fx_out = float(t_fx_out_by_ccy.get(ccy, 0.0))
        fx_in = float(t_fx_in_by_ccy.get(ccy, 0.0))

        v1_bilan[base_cash_idx] += fx_in * (1.0 - spread) - fx_out
        v1_bilan[idx] += fx_out * (1.0 - spread) - fx_in

    for ccy, idx in cash_idx_by_ccy.items():
        v1_expo[idx] = v1_bilan[idx]

    if tc_enabled:
        buy_r = float(tc_rate if tc_buy_rate is None else tc_buy_rate)
        sell_r = float(tc_rate if tc_sell_rate is None else tc_sell_rate)
        tc_total = buy_r * float(np.sum(t_buy)) + sell_r * float(np.sum(t_sell))
    else:
        tc_total = 0.0

    cash_pre_by_ccy = {ccy: float(v1_bilan[idx]) for ccy, idx in cash_idx_by_ccy.items()}
    cash_post_by_ccy = dict(cash_pre_by_ccy)
    cash_post_by_ccy[base_ccy] = float(cash_post_by_ccy[base_ccy] - tc_total)

    w1_expo = v1_expo / float(nav1)
    w1_bilan = v1_bilan / float(nav1)

    fx_map = _get_fx_to_base_map(df_data, base_currency=base_ccy)
    cash_pre_local_by_ccy, cash_post_local_by_ccy = {}, {}
    for ccy in cash_idx_by_ccy.keys():
        fx = float(fx_map.get(ccy, 1.0))
        cash_pre_local_by_ccy[ccy] = float(cash_pre_by_ccy[ccy] / fx)
        cash_post_local_by_ccy[ccy] = float(cash_post_by_ccy[ccy] / fx)

    return {
        "nav1": nav1,
        "w1_expo": w1_expo,
        "w1_bilan": w1_bilan,
        "v1_expo": v1_expo,
        "v1_bilan": v1_bilan,
        "t_net": t_net,
        "tc_total": tc_total,
        "cash_idx_by_ccy": cash_idx_by_ccy,
        "cash_pre_by_ccy": cash_pre_by_ccy,
        "cash_post_by_ccy": cash_post_by_ccy,
        "cash_pre_local_by_ccy": cash_pre_local_by_ccy,
        "cash_post_local_by_ccy": cash_post_local_by_ccy,
        "t_fx_out_by_ccy": t_fx_out_by_ccy,
        "t_fx_in_by_ccy": t_fx_in_by_ccy,
    }