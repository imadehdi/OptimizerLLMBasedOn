# src/solver_v3.py
import casadi as ca
import numpy as np
import scipy.optimize as spo
import pandas as pd

def parse_ilp_bounds(cstr: dict):
    b_type = (cstr.get("bound_type") or "").lower().strip()
    v = float(cstr.get("value", 0.0))
    if b_type == "eq": return v, v
    if b_type == "max": return -np.inf, v
    if b_type == "min": return v, np.inf
    if b_type == "range": return float(cstr.get("min_value", -np.inf)), float(cstr.get("max_value", np.inf))
    return -np.inf, np.inf

def build_ticker_index_map(df_data) -> dict:
    return {t.strip().lower(): i for i, t in enumerate(df_data["Ticker"].astype(str).tolist())}

def _normalise_ticker_list(lst) -> list:
    return list(dict.fromkeys([str(x).strip().lower() for x in (lst or []) if x]))

def compile_z_rows(constraints_config: list, n_rows: int, df_data: pd.DataFrame):
    def _row_with_coeffs(n: int, coeffs: dict) -> list:
        r = np.zeros(n, dtype=float)
        for idx, val in coeffs.items(): r[int(idx)] += float(val)
        return r.tolist()

    A_rows, lb_list, ub_list = [], [], []
    col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}
    ticker_to_idx = None

    for c in constraints_config or []:
        if not c.get("is_strict", True): continue
        fam, applied_to = (c.get("constraint_family") or "").lower().strip(), (c.get("applied_to") or "b").lower().strip()
        if applied_to not in {"b", "z"}: continue

        if fam == "cardinality":
            lb, ub = parse_ilp_bounds(c)
            attr = str(c.get("attribute") or "").strip().lower()
            if attr in ["sum_all", "sum", "somme", "total"]:
                A_rows.append([1.0] * n_rows); lb_list.append(lb); ub_list.append(ub)
            elif df_data is not None and col_map_case.get(attr) in df_data.columns:
                mask = np.isin(df_data[col_map_case[attr]].astype(str).str.lower().values, [str(t).lower() for t in c.get("targets", [])]).astype(float)
                A_rows.append(mask.tolist()); lb_list.append(lb); ub_list.append(ub)
            continue

        if fam == "implication":
            if ticker_to_idx is None: ticker_to_idx = build_ticker_index_map(df_data)
            I, J = _normalise_ticker_list(c.get("if")), _normalise_ticker_list(c.get("then"))
            if not I or not J: raise ValueError("Implication constraint: empty lists.")
            I_idx, J_idx = [ticker_to_idx[t] for t in I], [ticker_to_idx[t] for t in J]
            if_mode, then_mode = c.get("if_mode", "any").lower(), c.get("then_mode", "all").lower()

            if if_mode == "any" and then_mode == "all":
                for i in I_idx:
                    for j in J_idx: A_rows.append(_row_with_coeffs(n_rows, {i: 1.0, j: -1.0})); lb_list.append(-np.inf); ub_list.append(0.0)
            elif if_mode == "any" and then_mode == "any":
                for i in I_idx: A_rows.append(_row_with_coeffs(n_rows, {**{i: -1.0}, **{j: 1.0 for j in J_idx}})); lb_list.append(0.0); ub_list.append(np.inf)
            elif if_mode == "all" and then_mode == "all":
                for j in J_idx: A_rows.append(_row_with_coeffs(n_rows, {**{j: 1.0}, **{i: -1.0 for i in I_idx}})); lb_list.append(1.0 - len(I_idx)); ub_list.append(np.inf)
            else:
                A_rows.append(_row_with_coeffs(n_rows, {**{j: 1.0 for j in J_idx}, **{i: -1.0 for i in I_idx}})); lb_list.append(1.0 - len(I_idx)); ub_list.append(np.inf)

    return (np.array(A_rows, dtype=float), np.array(lb_list, dtype=float), np.array(ub_list, dtype=float)) if A_rows else (None, None, None)

def build_ilp_constraints(constraints_config: list, n_rows: int, df_data: pd.DataFrame):
    A, lb, ub = compile_z_rows(constraints_config, n_rows, df_data)
    return spo.LinearConstraint(A, lb, ub) if A is not None else None

def solve_optimization(builder, optimization_config: dict, df_data: pd.DataFrame = None, x0=None) -> dict:
    constraints_config = optimization_config.get("constraints", [])
    has_binaries = len(builder.binary_vars_names) > 0

    # L'ACCÉLÉRATEUR MAGIQUE POUR LA MATRICE CASADI (L-BFGS + MAX ITER)
    opts = {
        "ipopt.print_level": 0, 
        "print_time": 0, 
        "ipopt.tol": 1e-5, 
        "ipopt.max_iter": 300,
        "ipopt.hessian_approximation": "limited-memory"
    }

    n_dec = len(builder.lbx)
    current_x0 = np.asarray(x0, dtype=float).reshape(-1) if x0 is not None else np.zeros(n_dec, dtype=float)
    if current_x0.size != n_dec: current_x0 = np.zeros(n_dec, dtype=float)

    if hasattr(builder, "apply_gradient_normalisation"):
        try:
            if has_binaries:
                builder.add_alm_penalty(lambda_mult=0.0)
                builder.apply_gradient_normalisation(x0=current_x0, include_soft=True, include_alm=True)
                builder.remove_alm_penalty()
            else:
                builder.apply_gradient_normalisation(x0=current_x0, include_soft=True, include_alm=False)
        except Exception: pass

    if has_binaries:
        lambda_mult = 1.0
        for _ in range(3):
            builder.add_alm_penalty(lambda_mult=lambda_mult)
            sol = ca.nlpsol('solver', 'ipopt', builder.build_nlp(), opts)(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)
            current_x0 = np.array(sol['x']).flatten()
            lambda_mult *= 5.0
    else:
        solver = ca.nlpsol('solver', 'ipopt', builder.build_nlp(), opts)
        sol = solver(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)

    stats = solver.stats() if not has_binaries else {"success": True}
    x_relaxed = np.array(sol['x']).flatten()

    def _extract_all_vars(x_vec): return {n: x_vec[s:e].astype(float).tolist() for n, (s, e) in builder.var_indices.items()}

    if not has_binaries or not stats.get("success", False):
        return {"success": stats.get("success", False), "status": stats.get("return_status", "Unknown"), "objective": float(sol['f']), "variables": _extract_all_vars(x_relaxed), "x_values": x_relaxed[builder.var_indices["x"][0]:builder.var_indices["x"][1]].astype(float).tolist() if "x" in builder.var_indices else None}

    b_name = builder.binary_vars_names[0]
    start_idx, end_idx = builder.var_indices[b_name]
    n_bin = int(end_idx - start_idx)

    res_ilp = spo.milp(c=-x_relaxed[start_idx:end_idx], integrality=np.ones(n_bin), bounds=spo.Bounds(0, 1), constraints=[build_ilp_constraints(constraints_config, n_bin, df_data)] if build_ilp_constraints(constraints_config, n_bin, df_data) else None)
    if not res_ilp.success: return {"success": False, "status": "ILP Projection Failed"}

    builder.remove_alm_penalty()
    z_final = np.round(res_ilp.x)
    for i in range(n_bin): builder.lbx[start_idx + i] = builder.ubx[start_idx + i] = float(z_final[i])

    final_sol = ca.nlpsol('final_solver', 'ipopt', builder.build_nlp(), opts)(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)
    return {"success": True, "status": "Polish Completed", "objective": float(final_sol['f']), "variables": _extract_all_vars(np.array(final_sol['x']).flatten()), "selection": z_final.tolist(), "binary_var_name": b_name}