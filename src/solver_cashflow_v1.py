import numpy as np
import pandas as pd
import casadi as ca

from src.casadi_builder_v3 import CasadiProblemBuilder
from src.cashflow.forward_model import get_cash_index, build_cashflow_expressions, cashflow_numpy


def _ensure_w0_with_cash(matrix_inputs: dict, df_data: pd.DataFrame, cash_idx: int):
    n_rows = len(df_data)
    if "w0" not in matrix_inputs:
        raise ValueError("Cashflow mode requires matrix_inputs['w0'] including CASH weight.")
    
    w0 = np.asarray(matrix_inputs["w0"], dtype=float).reshape(-1)
    
    if w0.size == n_rows:
        return w0
        
    if w0.size == n_rows - 1:
        w0_full = np.zeros(n_rows, dtype=float)
        non_cash_idx = [i for i in range(n_rows) if i != cash_idx]
        w0_full[non_cash_idx] = w0
        w0_full[cash_idx] = 1.0 - float(np.sum(w0))
        return w0_full
        
    raise ValueError(f"w0 has invalid length: {w0.size}. Expected {n_rows} or {n_rows-1}.")


def _ensure_cov_with_cash(matrix_inputs: dict, df_data: pd.DataFrame, cov_key: str = "Variance"):
    n_rows = len(df_data)
    if cov_key not in matrix_inputs:
        return np.eye(n_rows, dtype=float)
        
    cov = np.asarray(matrix_inputs[cov_key], dtype=float)
    if cov.shape == (n_rows, n_rows):
        return cov
        
    if cov.shape == (n_rows - 1, n_rows - 1):
        cov_full = np.zeros((n_rows, n_rows), dtype=float)
        cov_full[:n_rows - 1, :n_rows - 1] = cov
        return cov_full
        
    raise ValueError(f"{cov_key} has invalid shape {cov.shape}.")


def _ensure_expected_return_with_cash(df_data: pd.DataFrame, cash_idx: int):
    if "Expected_Return" not in df_data.columns:
        return df_data
    df = df_data.copy()
    if pd.isna(df.loc[cash_idx, "Expected_Return"]):
        df.loc[cash_idx, "Expected_Return"] = 0.0
    return df


def solve_cashflow_optimization(cfg: dict, matrix_inputs: dict, df_data: pd.DataFrame, verbose: bool = False) -> dict:
    cashflow_cfg = cfg.get("cashflow") or {}
    if "amount" not in cashflow_cfg:
        raise ValueError("cfg['cashflow'] must contain key 'amount'.")
    if "nav0" not in cashflow_cfg:
        raise ValueError("cfg['cashflow'] must contain key 'nav0'.")
        
    amount = float(cashflow_cfg["amount"])
    nav0 = float(cashflow_cfg["nav0"])
    if nav0 <= 0:
        raise ValueError("nav0 must be > 0.")
        
    cash_idx = get_cash_index(df_data, cash_ticker="CASH")
    
    w0 = _ensure_w0_with_cash(matrix_inputs, df_data, cash_idx=cash_idx)
    if not np.isclose(np.sum(w0), 1.0, atol=1e-6):
        raise ValueError(f"w0 must sum to 1 (incl CASH). Got sum={float(np.sum(w0))}")
        
    cov_full = _ensure_cov_with_cash(matrix_inputs, df_data, cov_key="Variance")
    matrix_inputs = dict(matrix_inputs)
    matrix_inputs["Variance"] = cov_full.tolist()
    
    df_local = _ensure_expected_return_with_cash(df_data, cash_idx=cash_idx)
    
    objectives = cfg.get("objectives", []) or []
    single_objective = cfg.get("objective")
    obj_cfg = objectives[0] if objectives else single_objective
    if obj_cfg is None:
        obj_cfg = {"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Min Variance"}
        
    n_rows = len(df_local)
    builder = CasadiProblemBuilder(n_rows=n_rows)
    
    builder.create_variables([
        {"name": "t", "size": "n_non_cash", "type": "continuous"}
    ])
    
    t_var = builder.vars["t"]
    
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
    
    v1_non_cash = ca.vertcat(*[v1_expr[i] for i in non_cash_idx])
    builder.add_constraint(v1_non_cash, 0.0, ca.inf)
    builder.add_constraint(cash1_expr, 0.0, ca.inf)
    
    constraints = cfg.get("constraints", []) or []
    builder.apply_smart_constraints(constraints, df_local, matrix_inputs)
    
    obj_cfg = dict(obj_cfg)
    obj_cfg.setdefault("variable_name", "x")
    builder.build_objective(obj_cfg, matrix_inputs, df_local)
    
    nlp = builder.build_nlp()
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-7,
        "ipopt.max_iter": 2000
    }
    
    solver = ca.nlpsol("cashflow_solver", "ipopt", nlp, opts)
    
    x0 = np.zeros(int(t_var.size1()), dtype=float)
    
    sol = solver(
        x0=x0,
        lbx=builder.lbx,
        ubx=builder.ubx,
        lbg=builder.lbg,
        ubg=builder.ubg
    )
    
    stats = solver.stats()
    if not stats.get("success", False):
        return {
            "success": False,
            "status": stats.get("return_status", "Unknown"),
        }
        
    x_opt = np.array(sol["x"]).reshape(-1)
    t_s, t_e = builder.var_indices["t"]
    t_opt = x_opt[t_s:t_e].astype(float)
    
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
        "status": "Cashflow Optimisation Completed",
        "type": "cashflow_single_objective",
        "objective": float(sol["f"]),
        "nav0": float(nav0),
        "cashflow_amount": float(amount),
        "nav1": float(nav1),
        
        "cash_index": int(cash_idx),
        "cash_final_value": float(cash1),
        "cash_final_weight": float(w1[cash_idx]),
        
        "weights_initial": w0.tolist(),
        "weights_final": w1.tolist(),
        
        "trades_non_cash": t_opt.tolist(),
        "trades_full_vector": full_trades.tolist(),
        
        "positions_final_value": v1.tolist(),
        "sum_trades_non_cash": float(np.sum(t_opt)),
        "tickers": df_local["Ticker"].astype(str).tolist(),
    }