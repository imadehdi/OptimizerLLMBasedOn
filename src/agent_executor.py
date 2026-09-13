import numpy as np
import pandas as pd
from src.constraints.preprocess_v3 import preprocess_config_and_df_v3
from src.moo.solver import solve_moo
from src.casadi_builder_v3 import CasadiProblemBuilder
from src.solver_v3 import solve_optimization

# Nouvel import
from src.black_litterman import compute_black_litterman

class OptimizationExecutor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        cfg, df_local = preprocess_config_and_df_v3(optimization_config, df_data)
        
        # ==========================================
        # INTERCEPTION BLACK-LITTERMAN (PRÉTRAITEMENT)
        # ==========================================
        if "black_litterman" in cfg:
            bl_cfg = cfg["black_litterman"]
            P = np.array(bl_cfg["P"])
            Q = np.array(bl_cfg["Q"])
            tau = bl_cfg.get("tau", 0.05)
            rf = bl_cfg.get("rf", 0.0)
            
            # Extraction des données historiques
            mu_hist = df_local["Expected_Return"].values
            cov_hist = np.array(matrix_inputs["Variance"])
            
            # Utilisation du Benchmark comme proxy du marché (w_mkt), sinon équipondéré
            n_assets = len(df_local)
            w_mkt = np.array(matrix_inputs.get("Benchmark", np.full(n_assets, 1.0 / n_assets)))
            
            if self.verbose:
                print("Intégration des vues LLM (Black-Litterman)...")
                
            # Calcul des moments a posteriori
            mu_bl, cov_bl = compute_black_litterman(mu_hist, cov_hist, w_mkt, P, Q, tau, rf)
            
            # Écrasement silencieux des inputs originaux
            df_local["Expected_Return"] = mu_bl
            matrix_inputs["Variance"] = cov_bl.tolist()

        # ==========================================
        # AIGUILLAGE CLASSIQUE
        # ==========================================
        constraints = cfg.get("constraints", [])
        objectives = cfg.get("objectives", [])
        single_objective = cfg.get("objective")
        
        has_soft_constraints = any(not cstr.get("is_strict", True) for cstr in constraints)
        is_multi_objective = has_soft_constraints or (len(objectives) > 1)
        
        if is_multi_objective:
            if self.verbose:
                print("Routage vers le Moteur Multi-Objectif (Pymoo/NSGA-II)...")
            return solve_moo(cfg, matrix_inputs, df_local)
            
        if self.verbose:
            print("Routage vers le Moteur Single-Objective (CasADi + SciPy)...")
            
        n_rows = len(df_local)
        builder = CasadiProblemBuilder(n_rows=n_rows)
        builder.create_variables(cfg.get("decision_variables", []))
        
        obj_to_build = objectives[0] if objectives else single_objective
        if obj_to_build:
            builder.build_objective(obj_to_build, matrix_inputs, df_local)
            
        if hasattr(builder, "apply_smart_constraints"):
            builder.apply_smart_constraints(constraints, df_local)
            
        return solve_optimization(builder, cfg, df_data=df_local, x0=None)