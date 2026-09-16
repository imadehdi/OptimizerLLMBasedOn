# src/solver_cashflow_mip_v1.py
import numpy as np
import pandas as pd
import casadi as ca
from src.casadi_builder_v3 import CasadiProblemBuilder
from src.solver_v3 import solve_optimization
from src.cashflow.forward_model import get_cash_indices_by_ccy, build_cashflow_expressions_multi_cash, cashflow_numpy_multi_cash

def _split_constraints_cashflow(cfg_constraints: list):
    c_x, c_trade = [], []
    for c in (cfg_constraints or []):
        applied_to = (c.get("applied_to") or "x").lower().strip()
        if applied_to in ("b", "z", "t", "t_buy", "t_sell", "tc_total", "cash_pre", "cash_post"): c_trade.append(c)
        else: c_x.append(c)
    return c_x, c_trade

def solve_cashflow_mip_optimization(cfg: dict, matrix_inputs: dict, df_data: pd.DataFrame, verbose: bool = False) -> dict:
    cashflow_cfg = cfg.get("cashflow") or {}
    amount = float(cashflow_cfg["amount"])
    nav0 = float(cashflow_cfg["nav0"])
    base_currency = str(cashflow_cfg.get("base_currency", "USD")).upper().strip()
    use_foreign_cash = bool(cashflow_cfg.get("use_foreign_cash", True))
    tc_enabled = bool(cashflow_cfg.get("tc_enabled", False))
    tc_rate = float(cashflow_cfg.get("tc_rate", 0.0))

    nav1 = float(nav0 + amount)
    df_full = df_data.copy()
    n_rows = len(df_full)
    cash_idx_by_ccy = get_cash_indices_by_ccy(df_full)
    w0 = np.asarray(matrix_inputs["w0"], dtype=float).reshape(-1)
    
    matrix_inputs = dict(matrix_inputs)
    matrix_inputs["Variance"] = np.asarray(matrix_inputs.get("Variance", np.eye(n_rows)), dtype=float).tolist()

    obj_cfg = (cfg.get("objectives") or [cfg.get("objective")])[0]
    non_cash_idx = df_full.index[df_full["InstrumentType"].astype(str).str.upper().ne("CASH")].astype(int).tolist()
    
    builder = CasadiProblemBuilder(n_rows=n_rows)
    builder.create_variables([
        {"name": "t_buy", "size": len(non_cash_idx), "type": "continuous"},
        {"name": "t_sell", "size": len(non_cash_idx), "type": "continuous"},
        {"name": "b", "size": len(non_cash_idx), "type": "binary"}
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
        w0=w0, nav0=nav0, cashflow_amount=amount, df_data=df_full, base_currency=base_currency, tc_enabled=tc_enabled, tc_rate=tc_rate
    )

    for k, v in [("x", fm["w1_expo"]), ("w", fm["w1_expo"]), ("t", fm["t_net"]), ("t_buy", builder.vars["t_buy"]), ("t_sell", builder.vars["t_sell"])]:
        builder.set_evaluation_expression(k, v)

    df_trade = df_full.iloc[non_cash_idx].reset_index(drop=True)

    # BLOCAGE DES FLUX FX (WASH-TRADE)
    if not use_foreign_cash:
        idx_nb = np.where(df_trade["Currency"].astype(str).str.upper().str.strip().ne(base_currency).to_numpy())[0].tolist()
        if idx_nb:
            builder.add_constraint(builder.vars["t_buy"][idx_nb], 0.0, 0.0)
            builder.add_constraint(builder.vars["t_sell"][idx_nb], 0.0, 0.0)
            builder.add_constraint(builder.vars["b"][idx_nb], 0.0, 0.0)
        for ccy in cash_idx_by_ccy.keys():
            if ccy != base_currency:
                builder.add_constraint(builder.vars[f"t_fx_out_{ccy}"], 0.0, 0.0)
                builder.add_constraint(builder.vars[f"t_fx_in_{ccy}"], 0.0, 0.0)

    # INDEXATION DIRECTE AU LIEU DE VERTCAT
    builder.add_constraint(fm["v1_bilan"][non_cash_idx], 0.0, ca.inf)

    for ccy, expr_pre in fm["cash_pre_by_ccy"].items():
        builder.add_constraint(fm["cash_post_by_ccy"][ccy] if tc_enabled else expr_pre, 0.0, ca.inf)

    # NORMALISATION BIG-M
    M = float(nav1)
    builder.add_constraint((builder.vars["t_buy"] / M) - builder.vars["b"], -ca.inf, 0.0)
    builder.add_constraint((builder.vars["t_sell"] / M) - builder.vars["b"], -ca.inf, 0.0)

    # NORMALISATION MIN TICKET
    constraints_all = cfg.get("constraints", []) or []
    min_vals = [float(c.get("min_value")) for c in constraints_all if str(c.get("constraint_family") or "").lower() == "min_ticket" and c.get("min_value")]
    if min_vals:
        builder.add_constraint((builder.vars["t_buy"] + builder.vars["t_sell"]) / M - max(min_vals) * builder.vars["b"], 0.0, ca.inf)

    c_x, c_t = _split_constraints_cashflow([c for c in constraints_all if str(c.get("constraint_family") or "").lower() != "min_ticket"])
    builder.apply_smart_constraints(c_x, df_full, matrix_inputs)
    builder.apply_smart_constraints(c_t, df_trade, None)

    obj_cfg = dict(obj_cfg)
    obj_cfg.setdefault("variable_name", "x")
    builder.build_objective(obj_cfg, matrix_inputs, df_full)

    # solver_v3 (solve_optimization) a maintenant les bons opts L-BFGS
    res = solve_optimization(builder, cfg, df_data=df_trade, x0=np.zeros(len(builder.lbx)))
    if not res.get("success"): return res

    x_opt = res["variables"]
    t_buy_opt, t_sell_opt, b_opt = np.array(x_opt["t_buy"]), np.array(x_opt["t_sell"]), np.array(x_opt["b"])
    t_fx_out_opt = {ccy: float(x_opt[f"t_fx_out_{ccy}"][0]) for ccy in cash_idx_by_ccy if ccy != base_currency}
    t_fx_in_opt = {ccy: float(x_opt[f"t_fx_in_{ccy}"][0]) for ccy in cash_idx_by_ccy if ccy != base_currency}

    rep = cashflow_numpy_multi_cash(t_buy=t_buy_opt, t_sell=t_sell_opt, t_fx_out_by_ccy=t_fx_out_opt, t_fx_in_by_ccy=t_fx_in_opt, w0=w0, nav0=nav0, cashflow_amount=amount, df_data=df_full, base_currency=base_currency, tc_enabled=tc_enabled, tc_rate=tc_rate)
    full_trades = np.zeros(n_rows, dtype=float)
    full_trades[non_cash_idx] = rep["t_net"]

    return {
        "success": True, "status": "MIP Completed", "type": "cashflow_mip",
        "objective": float(res["objective"]), "nav0": nav0, "cashflow_amount": amount, "nav1": rep["nav1"],
        "weights_final": rep["w1_expo"].tolist(), "weights_bilan_final": rep["w1_bilan"].tolist(),
        "trades_net": rep["t_net"].tolist(), "tc_total": float(rep["tc_total"]),
        "n_trades_active": int(np.sum(b_opt > 0.5)), "trades_full_vector_pre_cost": full_trades.tolist()
    }