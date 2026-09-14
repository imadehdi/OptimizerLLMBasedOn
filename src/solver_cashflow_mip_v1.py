import numpy as np
import pandas as pd
import casadi as ca

from src.casadi_builder_v3 import CasadiProblemBuilder
from src.solver_v3 import solve_optimization
from src.cashflow.forward_model import get_cash_index, build_cashflow_expressions, cashflow_numpy


def _ensure_w0_with_cash(matrix_inputs: dict, df_data: pd.DataFrame):
    n_rows = len(df_data)
    if "w0" not in matrix_inputs:
        raise ValueError("Cashflow MIP requires matrix_inputs['w0'] including CASH weight.")
    w0 = np.asarray(matrix_inputs["w0"], dtype=float).reshape(-1)
    if w0.size != n_rows:
        raise ValueError(f"w0 length mismatch: got {w0.size}, expected {n_rows} (incl CASH).")
    if not np.isclose(np.sum(w0), 1.0, atol=1e-6):
        raise ValueError(f"w0 must sum to 1 (incl CASH). Got sum={float(np.sum(w0))}")
    return w0


def _ensure_cov_with_cash(matrix_inputs: dict, df_data: pd.DataFrame, cov_key: str = "Variance"):
    n_rows = len(df_data)
    if cov_key not in matrix_inputs:
        return np.eye(n_rows, dtype=float)

    cov = np.asarray(matrix_inputs[cov_key], dtype=float)
    if cov.shape == (n_rows, n_rows):
        return cov

    raise ValueError(f"{cov_key} has invalid shape {cov.shape}. Expected ({n_rows}, {n_rows}).")


def _ensure_expected_return_cash(df_data: pd.DataFrame, cash_idx: int):
    if "Expected_Return" not in df_data.columns:
        return df_data
    df = df_data.copy()
    if pd.isna(df.loc[cash_idx, "Expected_Return"]):
        df.loc[cash_idx, "Expected_Return"] = 0.0
    return df


def _split_constraints(cfg_constraints: list):
    c_x, c_b = [], []
    for c in (cfg_constraints or []):
        applied_to = (c.get("applied_to") or "x").lower().strip()
        if applied_to in ("b", "z"):
            c_b.append(c)
        else:
            c_x.append(c)
    return c_x, c_b


def solve_cashflow_mip_optimization(cfg: dict, matrix_inputs: dict, df_data: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Cashflow MIP V1:
    - Variables: t (trades, continus) + b (trade active, binaire), sur non-cash.
    - Cardinalité/implication sur b réutilise SciPy MILP (build_ilp_constraints).
    - Solveur: relax+milp+polish via solve_optimization().
    """

    cashflow_cfg = cfg.get("cashflow") or {}
    if "amount" not in cashflow_cfg or "nav0" not in cashflow_cfg:
        raise ValueError("cfg['cashflow'] must contain keys: 'amount', 'nav0'.")

    amount = float(cashflow_cfg["amount"])
    nav0 = float(cashflow_cfg["nav0"])
    if nav0 <= 0:
        raise ValueError("nav0 must be > 0.")

    cash_idx = get_cash_index(df_data, cash_ticker="CASH")
    df_full = _ensure_expected_return_cash(df_data, cash_idx=cash_idx)

    # Inputs
    w0 = _ensure_w0_with_cash(matrix_inputs, df_full)

    cov_full = _ensure_cov_with_cash(matrix_inputs, df_full, cov_key="Variance")
    matrix_inputs = dict(matrix_inputs)
    matrix_inputs["Variance"] = cov_full.tolist()

    objectives = cfg.get("objectives", []) or []
    single_objective = cfg.get("objective")
    obj_cfg = objectives[0] if objectives else single_objective
    if obj_cfg is None:
        obj_cfg = {"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Min Variance"}

    n_rows = len(df_full)
    builder = CasadiProblemBuilder(n_rows=n_rows)

    builder.create_variables([
        {"name": "t", "size": "n_non_cash", "type": "continuous"},
        {"name": "b", "size": "n_non_cash", "type": "binary"},
    ])

    t_var = builder.vars["t"]
    b_var = builder.vars["b"]

    # =========================================================
    # LE BLOC MANQUANT : Construction des expressions Cashflow
    # =========================================================
    fm = build_cashflow_expressions(
        trades_t=t_var,
        w0=w0,
        nav0=nav0,
        cashflow_amount=amount,
        cash_idx=cash_idx
    )

    w1_expr = fm["w1"]
    v1_expr = fm["v1"]
    cash1_expr = fm["cash1"]
    non_cash_idx = fm["non_cash_idx"]

    builder.set_evaluation_expression("x", w1_expr)
    builder.set_evaluation_expression("w", w1_expr)
    builder.set_evaluation_expression("w_final", w1_expr)

    # =========================================================
    # Physical constraints (long-only)
    # =========================================================
    v1_non_cash = ca.vertcat(*[v1_expr[i] for i in non_cash_idx])
    builder.add_constraint(v1_non_cash, 0.0, ca.inf)
    builder.add_constraint(cash1_expr, 0.0, ca.inf)

    # =========================================================
    # Big-M linking constraints NORMALISÉES : (t / M) <= b
    # =========================================================
    nav1 = nav0 + amount 
    if nav1 <= 0:
        raise ValueError("NAV1 <= 0 (nav0 + amount). Check cashflow inputs.")
    
    # M est la valeur max théorique d'un trade (la NAV totale)
    M = float(nav1)  

    # t / M - b <= 0 
    builder.add_constraint(t_var / M - b_var, -ca.inf, 0.0)
    # -t / M - b <= 0 <=> t/M >= -b
    builder.add_constraint(-t_var / M - b_var, -ca.inf, 0.0)

    # =========================================================
    # Constraints: split into x (weights) vs b (binaries)
    # =========================================================
    constraints_all = cfg.get("constraints", []) or []
    constraints_x, constraints_b = _split_constraints(constraints_all)

    builder.apply_smart_constraints(constraints_x, df_full, matrix_inputs)

    df_trade = df_full.iloc[non_cash_idx].reset_index(drop=True)
    builder.apply_smart_constraints(constraints_b, df_trade, matrix_inputs=None)

    # =========================================================
    # Objective on final weights
    # =========================================================
    obj_cfg = dict(obj_cfg)
    obj_cfg.setdefault("variable_name", "x")  
    builder.build_objective(obj_cfg, matrix_inputs, df_full)

    # =========================================================
    # Solve
    # =========================================================
    x0 = np.zeros(len(builder.lbx), dtype=float)  
    res = solve_optimization(builder, cfg, df_data=df_trade, x0=x0)

    if not res.get("success", False):
        return res

    vars_out = res.get("variables", {}) or {}
    t_opt = np.asarray(vars_out.get("t", []), dtype=float)
    b_opt = np.asarray(vars_out.get("b", []), dtype=float)

    rep = cashflow_numpy(
        trades_t=t_opt,
        w0=w0,
        nav0=nav0,
        cashflow_amount=amount,
        cash_idx=cash_idx
    )

    w1 = rep["w1"]
    v1 = rep["v1"]
    nav1 = rep["nav1"]
    cash1 = rep["cash1"]

    full_trades = np.zeros(n_rows, dtype=float)
    for k, idx in enumerate(non_cash_idx):
        full_trades[idx] = t_opt[k]
    full_trades[cash_idx] = -float(np.sum(t_opt))

    return {
        "success": True,
        "status": res.get("status", "Cashflow MIP Completed"),
        "type": "cashflow_mip_single_objective",
        "objective": float(res.get("objective", np.nan)),
        "nav0": float(nav0),
        "cashflow_amount": float(amount),
        "nav1": float(nav1),
        "cash_index": int(cash_idx),
        "cash_final_value": float(cash1),
        "cash_final_weight": float(w1[cash_idx]),
        "weights_initial": w0.tolist(),
        "weights_final": w1.tolist(),
        "trades_non_cash": t_opt.tolist(),
        "trade_active_binaries": b_opt.tolist(),
        "trades_full_vector": full_trades.tolist(),
        "positions_final_value": v1.tolist(),
        "sum_trades_non_cash": float(np.sum(t_opt)),
        "n_trades_active": int(np.sum(b_opt > 0.5)),
        "tickers_full": df_full["Ticker"].astype(str).tolist(),
        "tickers_trade": df_trade["Ticker"].astype(str).tolist(),
        "selection": res.get("selection"),  
        "binary_var_name": res.get("binary_var_name", "b"),
    }