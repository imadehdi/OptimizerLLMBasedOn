import casadi as ca
import numpy as np
import scipy.optimize as spo

def parse_ilp_bounds(cstr):
    b_type = cstr.get("bound_type")
    v = cstr.get("value", 0.0)
    min_v = cstr.get("min_value", 0.0)
    max_v = cstr.get("max_value", np.inf)
    
    if b_type == "eq": return v, v
    elif b_type == "max": return 0.0, v
    elif b_type == "min": return v, np.inf
    elif b_type == "range": return min_v, max_v
    return 0.0, np.inf

def build_ilp_constraints(constraints_config, n_rows, df_data):
    """Génère la matrice A_eq, lb_eq, ub_eq pour la projection ILP."""
    A, lb, ub = [], [], []
    col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}

    for cstr in constraints_config:
        # === ANTICIPATION PYMOO ===
        if not cstr.get("is_strict", True):
            continue 
        # ==========================
        
        if cstr.get("constraint_family") == "cardinality":
            attr = cstr.get("attribute", "").lower()
            min_bound, max_bound = parse_ilp_bounds(cstr)
            
            # Cardinalité Globale
            if attr in ["sum_all", "sum", "somme", "total"]:
                A.append([1.0] * n_rows)
                lb.append(min_bound)
                ub.append(max_bound)
            
            # Cardinalité Sectorielle / Sous-groupe
            elif df_data is not None and attr in col_map_case:
                real_col = col_map_case[attr]
                targets = [str(t).lower() for t in cstr.get("targets", [])]
                row_mask = np.zeros(n_rows)
                for i, val in enumerate(df_data[real_col]):
                    if str(val).lower() in targets:
                        row_mask[i] = 1.0
                
                A.append(row_mask.tolist())
                lb.append(min_bound)
                ub.append(max_bound)
    
    if not A: return None
    return spo.LinearConstraint(np.array(A), np.array(lb), np.array(ub))

def solve_optimization(builder, optimization_config: dict, df_data=None, x0=None) -> dict:
    constraints_config = optimization_config.get("constraints", [])
    has_binaries = len(builder.binary_vars_names) > 0
    
    opts = {
        'ipopt.print_level': 0, 
        'print_time': 0,
        'ipopt.hessian_approximation': 'limited-memory'
    }

    # ==========================================
    # PHASE 1 : RELAXATION (ALM LOOP)
    # ==========================================
    solver = ca.nlpsol('solver', 'ipopt', builder.build_nlp(), opts)
    current_x0 = x0 if x0 is not None else [0.0] * len(builder.lbx)
    
    if has_binaries:
        # Pseudo-Annealing : on commence avec une pénalité douce pour lisser le gradient
        builder.add_alm_penalty(lambda_mult=5.0, mu_val=0.0)
        solver = ca.nlpsol('solver', 'ipopt', builder.build_nlp(), opts)
        
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
    # PHASE 2 : HARDENING (PROJECTION ILP)
    # ==========================================
    b_name = builder.binary_vars_names[0] 
    start_idx, end_idx = builder.var_indices[b_name]
    x_opt = np.array(sol['x']).flatten()
    z_ref = x_opt[start_idx:end_idx]
    
    c = -z_ref 
    integrality = np.ones(builder.n_rows)
    bounds = spo.Bounds(0, 1)
    milp_constraints = build_ilp_constraints(constraints_config, builder.n_rows, df_data)
    
    if milp_constraints:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds, constraints=[milp_constraints])
    else:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds)
        
    if not res_ilp.success:
        return {"success": False, "status": "ILP Projection Failed"}
    
    z_final = np.round(res_ilp.x)

    # ==========================================
    # PHASE 3 : POLISH (IPOPT FINAL)
    # ==========================================
    builder.remove_alm_penalty()
    for i in range(builder.n_rows):
        builder.lbx[start_idx + i] = float(z_final[i])
        builder.ubx[start_idx + i] = float(z_final[i])
        
    final_solver = ca.nlpsol('final_solver', 'ipopt', builder.build_nlp(), opts)
    final_sol = final_solver(x0=current_x0, lbx=builder.lbx, ubx=builder.ubx, lbg=builder.lbg, ubg=builder.ubg)
    
    x_final = np.array(final_sol['x']).flatten()
    return {
        "success": final_solver.stats().get("success", False),
        "status": "Polish Completed",
        "objective": float(final_sol['f']),
        "x_values": x_final[:builder.n_rows].tolist()
    }