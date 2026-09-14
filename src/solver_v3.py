import casadi as ca
import numpy as np
import scipy.optimize as spo
import pandas as pd


def parse_ilp_bounds(cstr: dict):
    b_type = (cstr.get("bound_type") or "").lower().strip()
    v = float(cstr.get("value", 0.0))
    if b_type == "eq":
        return v, v
    if b_type == "max":
        return -np.inf, v
    if b_type == "min":
        return v, np.inf
    if b_type == "range":
        return float(cstr.get("min_value", -np.inf)), float(cstr.get("max_value", np.inf))
    return -np.inf, np.inf


def _build_ticker_index_map(df_data) -> dict:
    if df_data is None or "Ticker" not in df_data.columns:
        raise ValueError("df_data must contain a column 'Ticker' to use implication constraints.")
    tickers = df_data["Ticker"].astype(str).tolist()
    return {t.strip().lower(): i for i, t in enumerate(tickers)}


def _normalise_ticker_list(lst) -> list:
    if lst is None:
        return []
    out = []
    for x in lst:
        if x is None:
            continue
        s = str(x).strip().lower()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


def compile_z_rows(constraints_config: list, n_rows: int, df_data: pd.DataFrame):
    """
    Bac A : Construit la matrice A_bin * z encodée sous forme:
            lb <= A @ z <= ub
    pour SciPy (LinearConstraint) avec z binaire.

    Supporte (strict only, applied_to in {"b","z"}):
    - constraint_family == "cardinality":
      * global sum_all / sum / total
      * group sum via df_data[attribute] + targets
    - constraint_family == "implication":
      * champs: if (list[ticker]), then (list[ticker])
      * options: if_mode in ("any","all"), then_mode in ("any","all")
      * nécessite df_data avec une colonne "Ticker"
    """

    def _row_with_coeffs(n: int, coeffs: dict) -> list:
        r = np.zeros(n, dtype=float)
        for idx, val in coeffs.items():
            r[int(idx)] += float(val)
        return r.tolist()

    A_rows, lb_list, ub_list = [], [], []

    col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}
    ticker_to_idx = None

    for c in constraints_config or []:
        if not c.get("is_strict", True):
            continue

        fam = (c.get("constraint_family") or "").lower().strip()
        applied_to = (c.get("applied_to") or "b").lower().strip()

        if applied_to not in {"b", "z"}:
            continue

        # #################################################========
        # 1) CARDINALITY
        # #################################################========
        if fam == "cardinality":
            lb, ub = parse_ilp_bounds(c)
            attr = (c.get("attribute") or "").strip()
            attr_l = attr.lower()

            if attr_l in ["sum_all", "sum", "somme", "total"]:
                A_rows.append([1.0] * n_rows)
                lb_list.append(lb)
                ub_list.append(ub)
                continue

            if df_data is not None:
                real_col = col_map_case.get(attr_l, attr)

                if real_col in df_data.columns:
                    targets = [str(t).lower() for t in c.get("targets", []) or []]
                    values = df_data[real_col].astype(str).str.lower().values
                    mask = np.isin(values, targets).astype(float)
                    A_rows.append(mask.tolist())
                    lb_list.append(lb)
                    ub_list.append(ub)

            continue

        # #################################################========
        # 2) IMPLICATION
        # #################################################========
        if fam == "implication":
            if ticker_to_idx is None:
                ticker_to_idx = _build_ticker_index_map(df_data)

            I = _normalise_ticker_list(c.get("if"))
            J = _normalise_ticker_list(c.get("then"))

            if len(I) == 0:
                raise ValueError("Implication constraint: empty 'if' list is not allowed.")
            if len(J) == 0:
                raise ValueError("Implication constraint: empty 'then' list is not allowed.")

            inter = set(I).intersection(set(J))
            if inter:
                raise ValueError(f"Implication constraint: if/then intersection not allowed: {sorted(inter)}")

            if_mode = str(c.get("if_mode", "any")).lower().strip()
            then_mode = str(c.get("then_mode", "all")).lower().strip()

            if if_mode not in {"any", "all"}:
                raise ValueError(f"Implication constraint: invalid if_mode={if_mode}")
            if then_mode not in {"any", "all"}:
                raise ValueError(f"Implication constraint: invalid then_mode={then_mode}")

            try:
                I_idx = [ticker_to_idx[t] for t in I]
                J_idx = [ticker_to_idx[t] for t in J]
            except KeyError as e:
                raise ValueError(f"Implication constraint: unknown ticker in df_data['Ticker']: {str(e)}")

            if if_mode == "any" and then_mode == "all":
                for i in I_idx:
                    for j in J_idx:
                        A_rows.append(_row_with_coeffs(n_rows, {i: 1.0, j: -1.0}))
                        lb_list.append(-np.inf)
                        ub_list.append(0.0)

            elif if_mode == "any" and then_mode == "any":
                for i in I_idx:
                    coeffs = {i: -1.0}
                    for j in J_idx:
                        coeffs[j] = coeffs.get(j, 0.0) + 1.0
                    A_rows.append(_row_with_coeffs(n_rows, coeffs))
                    lb_list.append(0.0)
                    ub_list.append(np.inf)

            elif if_mode == "all" and then_mode == "all":
                rhs_lb = 1.0 - float(len(I_idx))
                for j in J_idx:
                    coeffs = {j: 1.0}
                    for i in I_idx:
                        coeffs[i] = coeffs.get(i, 0.0) - 1.0
                    A_rows.append(_row_with_coeffs(n_rows, coeffs))
                    lb_list.append(rhs_lb)
                    ub_list.append(np.inf)

            else:
                rhs_lb = 1.0 - float(len(I_idx))
                coeffs = {}
                for j in J_idx:
                    coeffs[j] = coeffs.get(j, 0.0) + 1.0
                for i in I_idx:
                    coeffs[i] = coeffs.get(i, 0.0) - 1.0
                A_rows.append(_row_with_coeffs(n_rows, coeffs))
                lb_list.append(rhs_lb)
                ub_list.append(np.inf)

            continue

    if not A_rows:
        return None, None, None

    return (
        np.array(A_rows, dtype=float),
        np.array(lb_list, dtype=float),
        np.array(ub_list, dtype=float),
    )


def build_ilp_constraints(constraints_config: list, n_rows: int, df_data: pd.DataFrame):
    A, lb, ub = compile_z_rows(constraints_config, n_rows, df_data)
    if A is None:
        return None
    return spo.LinearConstraint(A, lb, ub)


def solve_optimization(builder, optimization_config: dict, df_data: pd.DataFrame = None, x0=None) -> dict:
    """
    Exécuteur déterministe (Solve & Polish)
    Phase 1 : Relaxation Continue avec ALM
    Phase 2 : Projection Binaire avec SciPy MILP
    Phase 3 : Polish Continu Final

    MODIF: supporte un binaire de taille arbitraire (pas forcément builder.n_rows)
           et retourne toutes les variables (pas seulement les 'x_values').
    """
    constraints_config = optimization_config.get("constraints", [])
    has_binaries = len(builder.binary_vars_names) > 0

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-6
    }

    n_dec = len(builder.lbx)
    current_x0 = np.asarray(x0, dtype=float).reshape(-1) if x0 is not None else None
    if current_x0 is None or current_x0.size != n_dec:
        current_x0 = np.zeros(n_dec, dtype=float)

    # #################################################========
    # PHASE 1 : RELAXATION (ALM LOOP)
    # #################################################========
    if has_binaries:
        lambda_mult = 1.0
        for _ in range(3):
            builder.add_alm_penalty(lambda_mult=lambda_mult)
            nlp = builder.build_nlp()
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            sol = solver(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)
            current_x0 = np.array(sol['x']).flatten()
            lambda_mult *= 5.0
    else:
        nlp = builder.build_nlp()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)

    stats = solver.stats()
    x_relaxed = np.array(sol['x']).flatten()

    def _extract_all_vars(x_vec: np.ndarray) -> dict:
        out = {}
        for name, (s, e) in builder.var_indices.items():
            out[name] = x_vec[s:e].astype(float).tolist()
        return out

    # Si pas de binaires, ou échec IPOPT => retour direct
    if (not has_binaries) or (not stats.get("success", False)):
        vars_out = _extract_all_vars(x_relaxed)

        # Backward compatibility: x_values si variable 'x' existe
        x_values = None
        if "x" in builder.var_indices:
            s, e = builder.var_indices["x"]
            x_values = x_relaxed[s:e].astype(float).tolist()

        return {
            "success": stats.get("success", False),
            "status": stats.get("return_status", "Unknown"),
            "objective": float(sol['f']),
            "variables": vars_out,
            "x_values": x_values
        }

    # #################################################========
    # PHASE 2 : HARDENING (PROJECTION ILP SCIPY)
    # #################################################========
    b_name = builder.binary_vars_names[0]
    start_idx, end_idx = builder.var_indices[b_name]
    n_bin = int(end_idx - start_idx)

    z_ref = x_relaxed[start_idx:end_idx]

    c = -z_ref
    integrality = np.ones(n_bin)
    bounds = spo.Bounds(0, 1)

    milp_constraints = build_ilp_constraints(constraints_config, n_bin, df_data)

    if milp_constraints:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds, constraints=[milp_constraints])
    else:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds)

    if not res_ilp.success:
        return {"success": False, "status": "ILP Projection Failed (Combinatoire irréalisable)"}

    z_final = np.round(res_ilp.x)

    # #################################################========
    # PHASE 3 : POLISH (IPOPT FINAL)
    # #################################################========
    builder.remove_alm_penalty()

    for i in range(n_bin):
        builder.lbx[start_idx + i] = float(z_final[i])
        builder.ubx[start_idx + i] = float(z_final[i])

    final_nlp = builder.build_nlp()
    final_solver = ca.nlpsol('final_solver', 'ipopt', final_nlp, opts)
    final_sol = final_solver(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)

    x_final = np.array(final_sol['x']).flatten()
    vars_out = _extract_all_vars(x_final)

    x_values = None
    if "x" in builder.var_indices:
        s, e = builder.var_indices["x"]
        x_values = x_final[s:e].astype(float).tolist()

    return {
        "success": final_solver.stats().get("success", False),
        "status": "Polish Completed",
        "objective": float(final_sol['f']),
        "variables": vars_out,
        "x_values": x_values,
        "selection": z_final.tolist(),
        "binary_var_name": b_name
    }