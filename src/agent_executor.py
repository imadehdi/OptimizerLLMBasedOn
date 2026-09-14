import numpy as np
import pandas as pd

from src.constraints.preprocess_v3 import preprocess_config_and_df_v3
from src.moo.solver import solve_moo
from src.casadi_builder_v3 import CasadiProblemBuilder
from src.solver_v3 import solve_optimization

from src.black_litterman import compute_black_litterman

from src.solver_cashflow_v1 import solve_cashflow_optimization
from src.solver_cashflow_mip_v1 import solve_cashflow_mip_optimization


def _cashflow_needs_binary(cfg: dict) -> bool:
    # Si des contraintes s'appliquent à b/z ou si des familles binaires existent
    for c in (cfg.get("constraints", []) or []):
        applied_to = (c.get("applied_to") or "").lower().strip()
        fam = (c.get("constraint_family") or "").lower().strip()
        if applied_to in ("b", "z"):
            return True
        if fam in ("cardinality", "implication", "min_buy_in"):
            return True

    # Si decision_variables déclare un binaire (optionnel)
    for v in (cfg.get("decision_variables", []) or []):
        if str(v.get("type", "")).lower().strip() in ("binary", "integer"):
            return True

    return False


class OptimizationExecutor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        cfg, df_local = preprocess_config_and_df_v3(optimization_config, df_data)

        # =========================================================
        # INTERCEPTION BLACK-LITTERMAN (PRÉTRAITEMENT)
        # =========================================================
        if "black_litterman" in cfg:
            bl_cfg = cfg["black_litterman"]
            P = np.array(bl_cfg["P"])
            Q = np.array(bl_cfg["Q"])
            tau = bl_cfg.get("tau", 0.05)
            rf = bl_cfg.get("rf", 0.0)

            mu_hist = df_local["Expected_Return"].values
            cov_hist = np.array(matrix_inputs["Variance"])

            n_assets = len(df_local)
            w_mkt = np.array(matrix_inputs.get("Benchmark", np.full(n_assets, 1.0 / n_assets)))

            if self.verbose:
                print("Intégration des vues LLM (Black-Litterman)...")

            mu_bl, cov_bl = compute_black_litterman(mu_hist, cov_hist, w_mkt, P, Q, tau, rf)

            df_local["Expected_Return"] = mu_bl
            matrix_inputs["Variance"] = cov_bl.tolist()

        # =========================================================
        # ANALYSE DE LA CONFIGURATION (MATRICE 2x2)
        # =========================================================
        constraints = cfg.get("constraints", [])
        objectives = cfg.get("objectives", [])
        single_objective = cfg.get("objective")

        has_soft_constraints = any(not cstr.get("is_strict", True) for cstr in constraints)
        is_multi_objective = has_soft_constraints or (len(objectives) > 1)
        has_cashflow = cfg.get("cashflow") is not None

        if self.verbose:
            print(f"[DEBUG ROUTER] Cashflow: {has_cashflow} | Multi-Obj: {is_multi_objective} (Nb Objs: {len(objectives)}, Soft Cstrs: {has_soft_constraints})")

        # =========================================================
        # AIGUILLAGE GLOBAL
        # =========================================================
        if has_cashflow:
            if is_multi_objective:
                if self.verbose:
                    print("Routage vers le Moteur Multi-Objectif (Pymoo/NSGA) en mode CASHFLOW...")
                return solve_moo(cfg, matrix_inputs, df_local)
            
            if _cashflow_needs_binary(cfg):
                if self.verbose:
                    print("Routage vers le moteur Cashflow MIP (t + b, cardinalité/implication)...")
                return solve_cashflow_mip_optimization(cfg, matrix_inputs, df_local, verbose=self.verbose)

            if self.verbose:
                print("Routage vers le moteur Cashflow (Single-Objective, trades continus)...")
            return solve_cashflow_optimization(cfg, matrix_inputs, df_local, verbose=self.verbose)

        else:
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
                builder.apply_smart_constraints(constraints, df_local, matrix_inputs)

            return solve_optimization(builder, cfg, df_data=df_local, x0=None)