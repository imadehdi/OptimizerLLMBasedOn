import pandas as pd
import numpy as np
import copy
from src.casadi_builder import CasadiProblemBuilder
from src.solver import solve_optimization
from src.moo_solver import solve_moo  # Import du nouveau moteur génétique

class OptimizationExecutor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        obj_type = optimization_config.get("objective", {}).get("type")
        
        # 1. Cas spécifique : Frontière de Pareto par méthode Epsilon (Objectif vs Objectif purs)
        if obj_type == "pareto_frontier":
            if self.verbose:
                print("Exécution de la frontière de Pareto (Epsilon-Constraint)...")
            return self._run_pareto_epsilon(optimization_config, matrix_inputs, df_data)
            
        # --- AIGUILLAGE INTELLIGENT ---
        constraints = optimization_config.get("constraints", [])
        
        # Détection de la présence d'au moins une Soft Constraint (Compromis)
        has_soft_constraints = any(not cstr.get("is_strict", True) for cstr in constraints)
        
        if has_soft_constraints:
            if self.verbose:
                print("Soft Constraints détectées. Routage vers Pymoo (Algorithme Génétique NSGA-II)...")
            return solve_moo(optimization_config, matrix_inputs, df_data)
            
        if self.verbose:
            print("Uniquement des Hard Constraints. Routage vers CasADi/SciPy (Solve & Polish)...")
            
        # --- PIPELINE CLASSIQUE (SOLVE & POLISH) ---
        n_rows = len(df_data)
        builder = CasadiProblemBuilder(n_rows=n_rows)
        
        builder.create_variables(optimization_config.get("decision_variables", []))
        builder.build_objective(optimization_config.get("objective", {}), matrix_inputs, df_data)
        
        # Le Builder ignorera de lui-même les contraintes de combinatoire (qui iront dans SciPy)
        builder.apply_smart_constraints(constraints, df_data)
        
        return solve_optimization(builder, optimization_config, df_data=df_data, x0=None)

    def _run_pareto_epsilon(self, config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        obj_cfg = config["objective"]
        target_1 = obj_cfg["target_1"] 
        target_2 = obj_cfg["target_2"] 
        points = obj_cfg.get("points", 20)

        def _get_raw_val(t_cfg, w):
            if t_cfg.get("type") == "linear":
                return np.dot(w, df_data[t_cfg.get("target_name")].values)
            elif t_cfg.get("type") == "quadratic":
                mat = np.array(matrix_inputs[t_cfg.get("target_name")])
                return w.T @ mat @ w
            return 0.0

        cfg_min_t2 = copy.deepcopy(config)
        cfg_min_t2["objective"] = target_2
        res_min = self.run(cfg_min_t2, matrix_inputs, df_data)
        
        cfg_max_t1 = copy.deepcopy(config)
        cfg_max_t1["objective"] = target_1
        res_max = self.run(cfg_max_t1, matrix_inputs, df_data)

        if not res_min.get("success") or not res_max.get("success"):
            return {"success": False, "error": "Impossible de trouver les ancres de la frontière."}

        w_min = np.array(res_min["x_values"])
        w_max = np.array(res_max["x_values"])
        
        raw_min = _get_raw_val(target_2, w_min)
        raw_max = _get_raw_val(target_2, w_max)
        epsilons = np.linspace(raw_min, raw_max, points)
        
        pareto_results = []
        n_rows = len(df_data)
        
        for eps in epsilons:
            builder = CasadiProblemBuilder(n_rows=n_rows)
            builder.create_variables(config.get("decision_variables", []))
            
            # Objectif : Target 1
            builder.build_objective(target_1, matrix_inputs, df_data)
            
            # Application des contraintes standards
            constraints = config.get("constraints", [])
            builder.apply_smart_constraints(constraints, df_data)
            
            # Contrainte Epsilon pour forcer Target 2
            x_var = builder.vars.get(target_1.get("variable_name", "x"))
            if target_2.get("type") == "linear":
                vec = df_data[target_2.get("target_name")].values.reshape(1, -1)
                builder.add_constraint(ca.mtimes(vec, x_var), -ca.inf, eps)
            elif target_2.get("type") == "quadratic":
                mat = ca.DM(matrix_inputs[target_2.get("target_name")])
                builder.add_constraint(ca.mtimes(ca.mtimes(x_var.T, mat), x_var), -ca.inf, eps)

            res = solve_optimization(builder, config, df_data=df_data, x0=None)
            
            if res.get("success"):
                w_opt = np.array(res["x_values"])
                val_t1 = _get_raw_val(target_1, w_opt)
                val_t2 = _get_raw_val(target_2, w_opt)
                
                pareto_results.append({
                    "target_2_value": float(val_t2),
                    "target_1_value": float(val_t1),
                    "weights": w_opt.tolist()
                })

        highlight_metric = obj_cfg.get("highlight_metric", "")
        best_idx = None
        highlight_message = ""
        
        if pareto_results and highlight_metric == "ratio":
            ratios = [p["target_1_value"] / (p["target_2_value"] + 1e-9) for p in pareto_results]
            best_idx = int(np.argmax(ratios))
            highlight_message = f"AI Insight: Optimal Ratio at Portfolio #{best_idx + 1}"
            
        elif pareto_results and highlight_metric == "knee_point":
            x_vals = np.array([p["target_2_value"] for p in pareto_results])
            y_vals = np.array([p["target_1_value"] for p in pareto_results])
            x1, y1 = x_vals[0], y_vals[0]
            x2, y2 = x_vals[-1], y_vals[-1]
            max_dist = -1
            
            for i, (x0, y0) in enumerate(zip(x_vals, y_vals)):
                dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / (np.sqrt((x2 - x1)**2 + (y2 - y1)**2) + 1e-9)
                if dist > max_dist:
                    max_dist = dist
                    best_idx = i
            highlight_message = f"AI Insight: Best geometric compromise (Knee Point) at Portfolio #{best_idx + 1}"

        return {
            "success": True,
            "type": "pareto",
            "pareto_points": pareto_results,
            "target_1_name": target_1.get("target_name", "Target_1"),
            "target_2_name": target_2.get("target_name", "Target_2"),
            "best_portfolio_index": best_idx,
            "highlight_message": highlight_message
        }