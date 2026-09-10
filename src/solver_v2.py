import casadi as ca
import numpy as np
import scipy.optimize as spo

def parse_ilp_bounds(cstr):
    b_type = cstr.get("bound_type")
    v = cstr.get("value", 0.0)
    min_v = cstr.get("min_value", 0.0)
    max_v = cstr.get("max_value", np.inf)
    
    if b_type == "eq": return v, v
    elif b_type == "max": return -np.inf, v
    elif b_type == "min": return v, np.inf
    elif b_type == "range": return min_v, max_v
    return -np.inf, np.inf

def build_ilp_constraints(constraints_config, n_rows, df_data):
    """
    Génère la matrice A, lb, ub pour la projection ILP (SciPy).
    Traduit la cardinalité et les règles logiques en algèbre linéaire.
    """
    A, lb, ub = [], [], []
    col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}

    for cstr in constraints_config:
        
        # --- FILTRE PYMOO (MULTI-OBJECTIF) ---
        if not cstr.get("is_strict", True):
            continue 
        # -------------------------------------
        
        family = cstr.get("constraint_family")
        attr = cstr.get("attribute", "").lower()
        
        # 1. CONTRAINTES DE CARDINALITÉ
        if family == "cardinality":
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

        # 2. CONTRAINTES LOGIQUES BINAIRES
        elif family == "logic":
            # On suppose ici que 'targets' contient les index [idx_A, idx_B] des actifs concernés
            # (À adapter si 'targets' contient des Tickers textuels à mapper)
            targets = cstr.get("targets", [])
            if len(targets) >= 2:
                idx_A, idx_B = int(targets[0]), int(targets[1])
                row = np.zeros(n_rows)
                
                if attr == "exclusion":
                    # Mutuellement exclusifs : z_A + z_B <= 1
                    row[idx_A] = 1.0
                    row[idx_B] = 1.0
                    A.append(row.tolist())
                    lb.append(-np.inf)
                    ub.append(1.0)
                    
                elif attr == "implication":
                    # A implique B : z_A <= z_B => z_A - z_B <= 0
                    row[idx_A] = 1.0
                    row[idx_B] = -1.0
                    A.append(row.tolist())
                    lb.append(-np.inf)
                    ub.append(0.0)
                    
                elif attr == "corequisite":
                    # Toujours ensemble : z_A == z_B => z_A - z_B == 0
                    row[idx_A] = 1.0
                    row[idx_B] = -1.0
                    A.append(row.tolist())
                    lb.append(0.0)
                    ub.append(0.0)
    
    if not A: return None
    return spo.LinearConstraint(np.array(A), np.array(lb), np.array(ub))

def solve_optimization(builder, optimization_config: dict, df_data=None, x0=None) -> dict:
    """Moteur industriel : Relaxation Continue -> Projection ILP -> Polish IPOPT"""
    constraints_config = optimization_config.get("constraints", [])
    has_binaries = len(builder.binary_vars_names) > 0
    
    opts = {
        'ipopt.print_level': 0, 
        'print_time': 0,
        'ipopt.hessian_approximation': 'limited-memory'
    }

    current_x0 = x0 if x0 is not None else [0.0] * len(builder.lbx)

    # ==========================================
    # PHASE 1 : RELAXATION (ALM LOOP)
    # ==========================================
    # À FUSIONNER : C'est ici que vient se loger ta boucle ALM avec le scaling des gradients
    # pour faire monter dynamiquement lambda_mult et assurer une belle binarisation de z_ref.
    if has_binaries:
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
    # PHASE 2 : HARDENING (PROJECTION ILP SCIPY)
    # ==========================================
    b_name = builder.binary_vars_names[0] 
    start_idx, end_idx = builder.var_indices[b_name]
    x_opt = np.array(sol['x']).flatten()
    z_ref = x_opt[start_idx:end_idx]
    
    # Objectif géométrique : trouver les 0 et 1 les plus proches de z_ref
    c = -z_ref 
    integrality = np.ones(builder.n_rows)
    bounds = spo.Bounds(0, 1)
    
    milp_constraints = build_ilp_constraints(constraints_config, builder.n_rows, df_data)
    
    if milp_constraints:
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds, constraints=[milp_constraints])
    else:
        # On passe quand même ici s'il n'y a que du min_buy_in
        res_ilp = spo.milp(c=c, integrality=integrality, bounds=bounds)
        
    if not res_ilp.success:
        return {"success": False, "status": "ILP Projection Failed (Combinatoire irréalisable)"}
    
    z_final = np.round(res_ilp.x)

    # ==========================================
    # PHASE 3 : POLISH (IPOPT FINAL)
    # ==========================================
    builder.remove_alm_penalty()
    
    # Verrouillage absolu des variables binaires sur les choix de SciPy
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