# src/solver_cashflow_v1.py
import numpy as np
import pandas as pd
import casadi as ca
from src.casadi_builder_v3 import CasadiProblemBuilder
from src.cashflow.forward_model import get_cash_indices_by_ccy, build_cashflow_expressions_multi_cash, cashflow_numpy_multi_cash

def _ensure_w0_full(matrix_inputs: dict, n_rows: int) -> np.ndarray:
    w0 = np.asarray(matrix_inputs["w0"], dtype=float).reshape(-1)
    if not np.isclose(np.sum(w0), 1.0, atol=1e-6): raise ValueError(f"w0 must sum to 1. Got sum={float(np.sum(w0))}")
    return w0

def _split_constraints_cashflow(cfg_constraints: list):
    c_x, c_trade = [], []
    for c in (cfg_constraints or []):
        applied_to = (c.get("applied_to") or "x").lower().strip()
        if applied_to in ("b", "z", "t", "t_buy", "t_sell", "tc_total", "cash_pre", "cash_post"): c_trade.append(c)
        else: c_x.append(c)
    return c_x, c_trade

def solve_cashflow_optimization(cfg: dict, matrix_inputs: dict, df_data: pd.DataFrame, verbose: bool = False) -> dict:
    cashflow_cfg = cfg.get("cashflow") or {}
    amount = float(cashflow_cfg["amount"])
    nav0 = float(cashflow_cfg["nav0"])
    base_currency = str(cashflow_cfg.get("base_currency", "USD")).upper().strip()
    use_foreign_cash = bool(cashflow_cfg.get("use_foreign_cash", True))
    tc_enabled = bool(cashflow_cfg.get("tc_enabled", False))
    tc_rate = float(cashflow_cfg.get("tc_rate", 0.0))

    nav1 = float(nav0 + amount)
    df_local = df_data.copy()
    n_rows = len(df_local)
    cash_idx_by_ccy = get_cash_indices_by_ccy(df_local)
    w0 = _ensure_w0_full(matrix_inputs, n_rows)
    
    matrix_inputs = dict(matrix_inputs)
    matrix_inputs["Variance"] = np.asarray(matrix_inputs.get("Variance", np.eye(n_rows)), dtype=float).tolist()

    obj_cfg = (cfg.get("objectives") or [cfg.get("objective")])[0] or {"type": "quadratic", "target_name": "Variance", "direction": "min"}
    builder = CasadiProblemBuilder(n_rows=n_rows)
    non_cash_idx = df_local.index[df_local["InstrumentType"].astype(str).str.upper().ne("CASH")].astype(int).tolist()
    
    builder.create_variables([
        {"name": "t_buy", "size": len(non_cash_idx), "type": "continuous"},
        {"name": "t_sell", "size": len(non_cash_idx), "type": "continuous"}
    ])
    builder.add_constraint(builder.vars["t_buy"], 0.0, ca.inf)
    builder.add_constraint(builder.vars["t_sell"], 0.0, ca.inf)

    t_fx_out_by_ccy, t_fx_in_by_ccy = {}, {}
    for ccy in cash_idx_by_ccy.keys():
        if ccy == base_currency: continue
        builder.create_variables([{"name": f"t_fx_out_{ccy}", "size": 1, "type": "continuous"}, {"name": f"t_fx_in_{ccy}", "size": 1, "type": "continuous"}])
        t_fx_out_by_ccy[ccy] = builder.vars[f"t_fx_out_{ccy}"]
        t_fx_in_by_ccy[ccy] = builder.vars[f"t_fx_in_{ccy}"]
        builder.add_constraint(t_fx_out_by_ccy[ccy], 0.0, ca.inf)
        builder.add_constraint(t_fx_in_by_ccy[ccy], 0.0, ca.inf)

    fm = build_cashflow_expressions_multi_cash(
        t_buy=builder.vars["t_buy"], t_sell=builder.vars["t_sell"],
        t_fx_out_by_ccy=t_fx_out_by_ccy, t_fx_in_by_ccy=t_fx_in_by_ccy,
        w0=w0, nav0=nav0, cashflow_amount=amount, df_data=df_local, base_currency=base_currency, tc_enabled=tc_enabled, tc_rate=tc_rate
    )

    for k, v in [("x", fm["w1_expo"]), ("w", fm["w1_expo"]), ("w_bilan", fm["w1_bilan"]), ("t", fm["t_net"]), ("t_buy", builder.vars["t_buy"]), ("t_sell", builder.vars["t_sell"])]:
        builder.set_evaluation_expression(k, v)

    df_trade = df_local.iloc[non_cash_idx].reset_index(drop=True)

    # BLOCAGE DES FLUX FX (WASH-TRADE)
    if not use_foreign_cash:
        idx_nb = np.where(df_trade["Currency"].astype(str).str.upper().str.strip().ne(base_currency).to_numpy())[0].tolist()
        if idx_nb:
            builder.add_constraint(builder.vars["t_buy"][idx_nb], 0.0, 0.0)
            builder.add_constraint(builder.vars["t_sell"][idx_nb], 0.0, 0.0)
        for ccy in cash_idx_by_ccy.keys():
            if ccy != base_currency:
                builder.add_constraint(builder.vars[f"t_fx_out_{ccy}"], 0.0, 0.0)
                builder.add_constraint(builder.vars[f"t_fx_in_{ccy}"], 0.0, 0.0)

    # INDEXATION DIRECTE AU LIEU DE VERTCAT
    builder.add_constraint(fm["v1_bilan"][non_cash_idx], 0.0, ca.inf)

    for ccy, expr_pre in fm["cash_pre_by_ccy"].items():
        builder.add_constraint(fm["cash_post_by_ccy"][ccy] if tc_enabled else expr_pre, 0.0, ca.inf)

    c_x, c_t = _split_constraints_cashflow(cfg.get("constraints", []))
    builder.apply_smart_constraints(c_x, df_local, matrix_inputs)
    builder.apply_smart_constraints(c_t, df_trade, None)

    obj_cfg = dict(obj_cfg)
    obj_cfg.setdefault("variable_name", "x")
    builder.build_objective(obj_cfg, matrix_inputs, df_local)

    # HESSIAN APPROXIMATION MAGIQUE (INSTANTANÉ)
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-7, "ipopt.max_iter": 2000, "ipopt.hessian_approximation": "limited-memory"}
    solver = ca.nlpsol("solver", "ipopt", builder.build_nlp(), opts)

    x0 = np.zeros(len(builder.lbx), dtype=float)
    sol = solver(x0=x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)

    if not solver.stats().get("success", False): return {"success": False, "status": solver.stats().get("return_status", "Unknown")}

    x_opt = np.array(sol["x"]).reshape(-1)
    t_buy_opt = x_opt[builder.var_indices["t_buy"][0]:builder.var_indices["t_buy"][1]]
    t_sell_opt = x_opt[builder.var_indices["t_sell"][0]:builder.var_indices["t_sell"][1]]
    
    t_fx_out_opt = {ccy: float(x_opt[builder.var_indices[f"t_fx_out_{ccy}"][0]]) for ccy in cash_idx_by_ccy if ccy != base_currency}
    t_fx_in_opt = {ccy: float(x_opt[builder.var_indices[f"t_fx_in_{ccy}"][0]]) for ccy in cash_idx_by_ccy if ccy != base_currency}

    rep = cashflow_numpy_multi_cash(t_buy=t_buy_opt, t_sell=t_sell_opt, t_fx_out_by_ccy=t_fx_out_opt, t_fx_in_by_ccy=t_fx_in_opt, w0=w0, nav0=nav0, cashflow_amount=amount, df_data=df_local, base_currency=base_currency, tc_enabled=tc_enabled, tc_rate=tc_rate)
    
    full_trades = np.zeros(n_rows, dtype=float)
    full_trades[non_cash_idx] = rep["t_net"]

    return {
        "success": True, "status": "Completed", "type": "cashflow_single",
        "objective": float(sol["f"]), "nav0": nav0, "cashflow_amount": amount, "nav1": rep["nav1"],
        "weights_final": rep["w1_expo"].tolist(), "weights_bilan_final": rep["w1_bilan"].tolist(),
        "trades_net": rep["t_net"].tolist(), "tc_total": float(rep["tc_total"]),
        "trades_full_vector_pre_cost": full_trades.tolist()
    }