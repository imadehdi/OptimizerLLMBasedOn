import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Imports alignés sur la nouvelle architecture
from src.moo.problem import PortfolioMOOProblem
from src.moo.problem import PortfolioMOOProblemOnly
from src.moo.repair import PortfolioRepairZandW
from src.moo.repair import PortfolioRepairWOnly


def get_reference_directions(n_obj: int, pop_size: int, seed: int = 1):
    """
    Génère les directions de référence pour NSGA-III si l'utilisateur
    définit plus de 2 dimensions d'exploration (ex: Variance + ESG + Rendement).
    """
    try:
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as e:
        raise ImportError("Impossible d'importer get_reference_directions. Vérifiez pymoo.") from e

    for n_partitions in range(1, 50):
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        if len(ref_dirs) >= pop_size:
            return ref_dirs

    return get_reference_directions("das-dennis", n_obj, n_partitions=49)


def _needs_binary_mode(cfg: dict) -> bool:
    # 1) Décision variables explicites
    dv = cfg.get("decision_variables", []) or []
    for v in dv:
        if str(v.get("type", "")).lower() == "binary":
            return True

    # 2) Contraintes qui impliquent une sélection / non-convexité
    for c in (cfg.get("constraints", []) or []):
        applied_to = (c.get("applied_to") or "x").lower()
        fam = str(c.get("constraint_family", "")).lower()
        if applied_to in ["b", "z"]:
            return True
        if fam in ["cardinality", "min_buy_in"]:
            return True

    return False


def solve_moo(cfg: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
    """
    Moteur Génétique Multi-Objectif et Soft Constraints.
    Implémente NSGA-II / NSGA-III avec le Repair Operator Hybride (MILP + SLSQP).
    """
    n_assets = len(df_data)
    constraints_config = cfg.get("constraints", []) or []

    # Instanciation de l'architecture 4 Bacs
    binary_mode = _needs_binary_mode(cfg)

    if binary_mode:
        problem = PortfolioMOOProblem(n_assets, cfg, matrix_inputs, df_data)
        repair = PortfolioRepairZandW(n_assets, constraints_config, df_data, eps_select=1e-4)
    else:
        problem = PortfolioMOOProblemOnly(n_assets, cfg, matrix_inputs, df_data)
        repair = PortfolioRepairWOnly(n_assets, constraints_config, df_data, eps_select=1e-4)

    # Paramétrage de la puissance de calcul
    moo_cfg = cfg.get("moo", {}) or {}
    pop_size = int(moo_cfg.get("pop_size", 100))
    n_gen = int(moo_cfg.get("n_gen", 150))
    seed = int(moo_cfg.get("seed", 42))
    verbose = bool(moo_cfg.get("verbose", True))

    # Le nombre d'objectifs pour NSGA (M objectifs purs + S soft constraints)
    n_obj = problem.n_obj

    if n_obj <= 2:
        algorithm = NSGA2(
            pop_size=pop_size,
            repair=repair,
            eliminate_duplicates=True
        )
        algo_name = "NSGA2"
        ref_dirs_used = None
    else:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        ref_dirs = get_reference_directions(n_obj=n_obj, pop_size=pop_size, seed=seed)
        pop_size_eff = len(ref_dirs)

        algorithm = NSGA3(
            pop_size=pop_size_eff,
            ref_dirs=ref_dirs,
            repair=repair,
            eliminate_duplicates=True
        )
        algo_name = "NSGA3"
        ref_dirs_used = {"n_obj": int(n_obj), "n_ref_dirs": int(len(ref_dirs))}

    # Lancement de l'évolution
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=True,
        copy_algorithm=False
    )

    if res.F is None:
        return {"success": False, "status": f"MOO Failed to converge ({algo_name})"}

    # ... [Ton code avant la ligne 120] ...
    
    F_vals = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
    X_vals = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)

    # Re-calcul silencieux du dernier front pour extraire les valeurs brutes et la matrice G
    tmp_out = {}
    problem._evaluate(X_vals, tmp_out)

    raw_soft = tmp_out.get("raw_soft_losses")
    raw_obj = tmp_out.get("raw_objectives")
    G_vals = tmp_out.get("G")

    # Correction de l'attribut : Utiliser problem.n_obj au lieu de problem.M
    # Attention, problem.n_obj inclut les soft constraints (S). 
    # Le nombre d'objectifs primaires purs est problem.n_obj - problem.n_s
    n_primary_objs = problem.n_obj - problem.n_s if hasattr(problem, "n_s") else problem.n_obj

    pareto_points = []
    for i in range(len(F_vals)):
        if binary_mode:
            z = X_vals[i, :n_assets].astype(float)
            w = X_vals[i, n_assets:].astype(float)
        else:
            w = X_vals[i, :n_assets].astype(float)
            z = (w > 1e-4).astype(float)

        pt = {
            "objective_names": problem.objective_names,
            "soft_constraint_names": problem.soft_names,

            # Métriques (Bac Principal & Bac D)
            "objectives_minimised": [float(x) for x in F_vals[i, :n_primary_objs]],
            "objectives_raw": [float(x) for x in (raw_obj[i, :] if raw_obj is not None else [])],
            
            # Si des soft constraints existent, elles sont stockées après les objectifs primaires
            "soft_losses_normalised": [float(x) for x in F_vals[i, n_primary_objs:]] if problem.n_s > 0 else [],
            "soft_losses_raw": [float(x) for x in (raw_soft[i, :] if raw_soft is not None else [])],

            "hard_nl_violations": [float(x) for x in (G_vals[i, :] if G_vals is not None else [])],

            "weights": w.tolist(),
            "selection": z.tolist(),
        }
        pareto_points.append(pt)

    pareto_points.sort(key=lambda d: d["objectives_minimised"][0] if d["objectives_minimised"] else 0.0)

    return {
        "success": True,
        "status": "MOO Completed",
        "type": "pareto",
        "algorithm": algo_name,
        "ref_dirs": ref_dirs_used,

        "n_objectives_total": int(problem.n_obj),
        "n_objectives_primary": int(n_primary_objs),
        "n_soft_constraints": int(problem.n_s if hasattr(problem, "n_s") else 0),
        "n_hard_nl_constraints": int(problem.n_ieq_constr),

        "pareto_points": pareto_points,
    }
