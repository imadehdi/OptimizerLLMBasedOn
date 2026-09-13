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

def compile_z_rows(constraints_config: list, n_rows: int, df_data: pd.DataFrame):
    """
    Bac A : Construit la matrice A_bin * z <= b_bin pour SciPy.
    Grâce à preprocess_v3, toutes les exclusions/inclusions sont déjà 
    transformées en contraintes de cardinalité de groupes (__grp_k).
    """
    A_rows, lb_list, ub_list = [], [], []
    
    for c in constraints_config:
        # Filtre absolu : On ne garde que les contraintes strictes
        if not c.get("is_strict", True):
            continue
            
        fam = (c.get("constraint_family") or "").lower().strip()
        applied_to = (c.get("applied_to") or "b").lower().strip()
        
        # SciPy ne gère QUE les variables binaires (la sélection)
        if fam == "cardinality" and applied_to in ["b", "z"]:
            lb, ub = parse_ilp_bounds(c)
            attr = (c.get("attribute") or "").strip()
            
            # 1. Cardinalité Globale (sum_all)
            if attr.lower() in ["sum_all", "sum", "somme", "total"]:
                A_rows.append([1.0] * n_rows)
                lb_list.append(lb)
                ub_list.append(ub)
                
            # 2. Cardinalité de Sous-Groupe (Secteurs ou exclusions prétraitées __grp_X)
            elif df_data is not None and attr in df_data.columns:
                targets = [str(t).lower() for t in c.get("targets", [])]
                # Construit un masque binaire pour la ligne de la matrice
                mask = df_data[attr].astype(str).str.lower().isin(targets).astype(float).values
                A_rows.append(mask.tolist())
                lb_list.append(lb)
                ub_list.append(ub)

    if not A_rows:
        return None, None, None
        
    return np.array(A_rows, dtype=float), np.array(lb_list, dtype=float), np.array(ub_list, dtype=float)

def build_ilp_constraints(constraints_config: list, n_rows: int, df_data: pd.DataFrame):
    A, lb, ub = compile_z_rows(constraints_config, n_rows, df_data)
    if A is None:
        return None
    return spo.LinearConstraint(A, lb, ub)

def solve_optimization(builder, optimization_config: dict, df_data: pd.DataFrame = None, x0=None) -> dict:
    """
    Exécuteur déterministe (Solve & Polish)
    Phase 1 : Relaxation Continue avec ALM
    Phase 2 : Projection Binaire avec SciPy
    Phase 3 : Polish Continu Final
    """
    constraints_config = optimization_config.get("constraints", [])
    has_binaries = len(builder.binary_vars_names) > 0
    
    opts = {
        "ipopt.print_level": 0, 
        "print_time": 0,
        "ipopt.tol": 1e-6
    }

    # Initialisation neutre
    current_x0 = x0 if x0 is not None else [1.0 / builder.n_rows] * len(builder.lbx)

    # ==========================================
    # PHASE 1 : RELAXATION (ALM LOOP)
    # ==========================================
    if has_binaries:
        lambda_mult = 1.0
        # Boucle ALM dynamique pour polariser les binaires vers 0 ou 1
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
    if not has_binaries or not stats.get("success", False):
        x_opt = np.array(sol['x']).flatten()
        return {
            "success": stats.get("success", False),
            "status": stats.get("return_status", "Unknown"),
            "objective": float(sol['f']),
            "x_values": x_opt[:builder.n_rows].tolist()
        }

    # ==========================================
    # PHASE 2 : HARDENING (PROJECTION ILP SCIPY)
    # ==========================================
    b_name = builder.binary_vars_names[0] 
    start_idx, end_idx = builder.var_indices[b_name]
    x_opt = np.array(sol['x']).flatten()
    z_ref = x_opt[start_idx:end_idx]
    
    # Distance euclidienne vers la solution relâchée
    c = -z_ref 
    integrality = np.ones(builder.n_rows)
    bounds = spo.Bounds(0, 1)
    
    milp_constraints = build_ilp_constraints(constraints_config, builder.n_rows, df_data)
    
    if milp_constraints:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds, constraints=[milp_constraints])
    else:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds)
        
    if not res_ilp.success:
        return {"success": False, "status": "ILP Projection Failed (Combinatoire irréalisable)"}
    
    z_final = np.round(res_ilp.x)

    # ==========================================
    # PHASE 3 : POLISH (IPOPT FINAL)
    # ==========================================
    builder.remove_alm_penalty()
    
    # Verrouillage absolu des variables binaires
    for i in range(builder.n_rows):
        builder.lbx[start_idx + i] = float(z_final[i])
        builder.ubx[start_idx + i] = float(z_final[i])
        
    final_nlp = builder.build_nlp()
    final_solver = ca.nlpsol('final_solver', 'ipopt', final_nlp, opts)
    final_sol = final_solver(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)
    
    x_final = np.array(final_sol['x']).flatten()
    w_final = x_final[:builder.n_rows].tolist()
    
    return {
        "success": final_solver.stats().get("success", False),
        "status": "Polish Completed",
        "objective": float(final_sol['f']),
        "x_values": w_final,
        "selection": z_final.tolist()
    }