# src/agent_executor.py
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
        if fam in ("cardinality", "implication", "min_buy_in", "min_ticket"):
            return True

    # Si decision_variables déclare un binaire (optionnel)
    for v in (cfg.get("decision_variables", []) or []):
        if str(v.get("type", "")).lower().strip() in ("binary", "integer"):
            return True

    return False


def _soft_mode(cfg: dict) -> str:
    """
    Gestion de la stratégie soft constraints en mono-objectif (CasADi).
    - "penalty" (défaut): soft -> pénalité via tolérance (reste mono-objectif)
    - "moo": soft -> multi-objectif (chaque soft devient un objectif via solve_moo)
    """
    mode = (cfg.get("soft_constraints_mode") or "penalty")
    mode = str(mode).lower().strip()
    if mode not in ("penalty", "moo"):
        mode = "penalty"
    return mode


class OptimizationExecutor:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run(self, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
        out = preprocess_config_and_df_v3(optimization_config, df_data, matrix_inputs=matrix_inputs)
        if len(out) == 2:
            cfg, df_local = out
            mi_local = matrix_inputs
        else:
            cfg, df_local, mi_local = out
        matrix_inputs = mi_local

        # =========================================================================
        # INTERCEPTION BLACK-LITTERMAN (PRÉTRAITEMENT)
        # =========================================================================
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
            # NB: on modifie matrix_inputs (comportement historique). Si tu veux éviter l'effet de bord,
            # fais une copie en amont.
            matrix_inputs["Variance"] = cov_bl.tolist()

        # =========================================================================
        # ANALYSE DE LA CONFIGURATION
        # =========================================================================
        constraints = cfg.get("constraints", []) or []
        objectives = cfg.get("objectives", []) or []
        single_objective = cfg.get("objective")

        has_soft_constraints = any(not cstr.get("is_strict", True) for cstr in constraints)
        has_cashflow = cfg.get("cashflow") is not None

        # IMPORTANT (NOUVEAU):
        # - Avant: soft constraints => on basculait en MOO automatiquement.
        # - Maintenant: en mono-objectif, on peut choisir de les traiter en pénalité CasADi
        #   via soft_constraints_mode="penalty" (défaut).
        multi_obj_by_objectives = len(objectives) > 1
        soft_mode = _soft_mode(cfg)
        is_multi_objective = multi_obj_by_objectives or (has_soft_constraints and soft_mode == "moo")

        if self.verbose:
            print(
                f"[DEBUG ROUTER] Cashflow: {has_cashflow} | Multi-Obj: {is_multi_objective} "
                f"(Nb Objs: {len(objectives)}), Soft: {has_soft_constraints}, SoftMode: {soft_mode}"
            )

        # =========================================================================
        # AIGUILLAGE GLOBAL
        # =========================================================================
        if has_cashflow:
            # CASHFLOW
            # Note: la pénalisation soft mono-objectif via tolérance n'est branchée ici
            # que si solve_cashflow_optimization / solve_cashflow_mip_optimization la supportent.
            # Pour l'instant, on garde la logique:
            # - multi-objectif => solve_moo
            # - sinon => solve_cashflow_optimization / MIP selon besoins binaires.
            if is_multi_objective:
                if self.verbose:
                    print("Routage vers le Moteur Multi-Objectif (Pymoo/NSGA) en mode CASHFLOW...")
                return solve_moo(cfg, matrix_inputs, df_local)

            if _cashflow_needs_binary(cfg):
                if self.verbose:
                    print("Routage vers le moteur Cashflow MIP (t+ t- + b, cardinalité/implication/min_ticket)...")
                return solve_cashflow_mip_optimization(cfg, matrix_inputs, df_local, verbose=self.verbose)

            if self.verbose:
                print("Routage vers le moteur Cashflow (Single-Objective, trades continus)...")
            return solve_cashflow_optimization(cfg, matrix_inputs, df_local, verbose=self.verbose)

        # =========================================================================
        # CLASSIC (NON CASHFLOW)
        # =========================================================================
        if is_multi_objective:
            if self.verbose:
                print("Routage vers le Moteur Multi-Objectif (Pymoo/NSGA)...")
            return solve_moo(cfg, matrix_inputs, df_local)

        if self.verbose:
            print("Routage vers le Moteur Single-Objective (CasADi + SciPy)...")

        n_rows = len(df_local)
        builder = CasadiProblemBuilder(n_rows=n_rows)
        builder.create_variables(cfg.get("decision_variables", []))

        # 1) Objectif
        obj_to_build = objectives[0] if objectives else single_objective
        if obj_to_build:
            builder.build_objective(obj_to_build, matrix_inputs, df_local)

        # 2) Contraintes hard
        if hasattr(builder, "apply_smart_constraints"):
            builder.apply_smart_constraints(constraints, df_local, matrix_inputs)

        # 3) Soft constraints en mono-objectif => pénalités (NOUVEAU)
        # Activé si soft_mode="penalty" et qu'il existe au moins une contrainte soft.
        if has_soft_constraints and soft_mode == "penalty":
            # Paramètres globaux optionnels (tu peux les exposer ensuite dans l'UI)
            sp_cfg = cfg.get("soft_penalty", {}) or {}
            lambda_soft = float(sp_cfg.get("lambda_soft", 1.0))
            default_tolerance = float(sp_cfg.get("default_tolerance", 1e-4))
            penalty_power = int(sp_cfg.get("penalty_power", 2))

            # Applique uniquement si la méthode existe (casadi_builder_v3 modifié)
            if hasattr(builder, "apply_soft_constraints_as_penalty"):
                builder.apply_soft_constraints_as_penalty(
                    config_constraints=constraints,
                    df_data=df_local,
                    matrix_inputs=matrix_inputs,
                    lambda_soft=lambda_soft,
                    default_tolerance=default_tolerance,
                    penalty_power=penalty_power,
                )
                if self.verbose:
                    print(
                        f"[DEBUG SOFT->PENALTY] Applied soft penalties "
                        f"(lambda_soft={lambda_soft}, default_tolerance={default_tolerance}, power={penalty_power})"
                    )
            else:
                if self.verbose:
                    print("[WARN] soft_constraints_mode='penalty' but builder has no apply_soft_constraints_as_penalty().")

        return solve_optimization(builder, cfg, df_data=df_local, x0=None)