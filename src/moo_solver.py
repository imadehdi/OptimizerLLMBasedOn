import numpy as np
import scipy.optimize as spo
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# On importe le constructeur logique ILP de ton moteur exact
from src.solver import build_ilp_constraints

class PortfolioRepair(Repair):
    """
    Opérateur de réparation en deux séquences :
    1. Projection combinatoire (Hardening via ILP SciPy) pour le vecteur binaire (z).
    2. Recalibrage algébrique pour le vecteur continu (w), respectant le min_buy_in et le budget.
    """
    def __init__(self, n_assets: int, constraints_config: list, df_data: pd.DataFrame):
        super().__init__()
        self.n_assets = n_assets
        self.df_data = df_data
        
        # Le constructeur ne lira que les contraintes où is_strict == True
        self.milp_constraints = build_ilp_constraints(constraints_config, n_assets, df_data)
        
        # Extraction du seuil min_buy_in strict
        self.min_buy_in = 0.0
        for cstr in constraints_config:
            if cstr.get("constraint_family") == "min_buy_in" and cstr.get("is_strict", True):
                self.min_buy_in = cstr.get("min_value", 0.0)

    def _do(self, problem, X, **kwargs):
        X_repaired = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            individu_brut = X[i, :]
            y_sel = individu_brut[:self.n_assets]   
            y_poids = individu_brut[self.n_assets:] 
            
            # --- SÉQUENCE 1 : RÉPARATION BINAIRE (ILP) ---
            c = -y_sel
            integrality = np.ones(self.n_assets)
            bounds = spo.Bounds(0, 1)
            
            if self.milp_constraints:
                res = spo.milp(c=c, integrality=integrality, bounds=bounds, constraints=[self.milp_constraints])
            else:
                res = spo.milp(c=c, integrality=integrality, bounds=bounds)
                
            z_rep = np.round(res.x) if res.success else np.zeros(self.n_assets)
            
            # --- SÉQUENCE 2 : RÉPARATION CONTINUE ---
            w_rep = y_poids * z_rep 
            
            somme_brute = np.sum(w_rep)
            if somme_brute > 0:
                w_rep = w_rep / somme_brute
            elif np.sum(z_rep) > 0:
                # Si Pymoo a mis des poids nuls sur tous les sélectionnés, on répartit équitablement
                w_rep = z_rep / np.sum(z_rep)
                
            # Application rigoureuse du min_buy_in avec redistribution de l'excédent
            if self.min_buy_in > 0 and np.sum(z_rep) > 0:
                mask = (z_rep == 1)
                # Remontée au seuil minimum
                w_rep[mask] = np.maximum(w_rep[mask], self.min_buy_in)
                
                excess_total = np.sum(w_rep) - 1.0
                if excess_total > 0:
                    # On identifie la "réserve" disponible sur les actifs au-dessus du seuil
                    surplus = w_rep - self.min_buy_in
                    surplus[surplus < 0] = 0.0
                    sum_surplus = np.sum(surplus)
                    
                    if sum_surplus > 0:
                        # On ponctionne le surplus proportionnellement pour revenir à 100%
                        w_rep = w_rep - (surplus / sum_surplus) * excess_total
                    else:
                        # Infeasibilité mathématique (ex: min_buy_in = 5% sur 30 actifs = 150%)
                        # Le portefeuille sera lourdement pénalisé plus tard
                        pass 
            
            X_repaired[i, :self.n_assets] = z_rep
            X_repaired[i, self.n_assets:] = w_rep
            
        return X_repaired


class PortfolioProblem(Problem):
    """
    Évalue les individus 100% légaux. 
    Objectif 0 : La fonction financière cible (Variance, Rendement...)
    Objectif 1 à N : Les erreurs de violation des Soft Constraints.
    """
    def __init__(self, n_assets: int, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame):
        self.n_assets = n_assets
        self.obj_config = optimization_config.get("objective", {})
        self.matrix_inputs = matrix_inputs
        self.df_data = df_data
        
        # Filtrage des Soft Constraints
        all_constraints = optimization_config.get("constraints", [])
        self.soft_constraints = [c for c in all_constraints if not c.get("is_strict", True)]
        
        # Pymoo a besoin de connaître le nombre exact d'objectifs (Objectif principal + N pénalités)
        n_objectives = 1 + len(self.soft_constraints)
        
        super().__init__(
            n_var=2 * n_assets, 
            n_obj=n_objectives,
            n_ieq_constr=0, 
            xl=0.0, 
            xu=1.0
        )
        
        self.col_map_case = {str(c).lower(): c for c in self.df_data.columns}

    def _evaluate(self, x, out, *args, **kwargs):
        F = np.zeros((x.shape[0], self.n_obj))
        
        for i in range(x.shape[0]):
            w = x[i, self.n_assets:] 
            
            # --- 1. CALCUL DE L'OBJECTIF FINANCIER PRINCIPAL ---
            obj_type = self.obj_config.get("type")
            direction = self.obj_config.get("direction", "min")
            target = self.obj_config.get("target_name")
            
            val_fin = 0.0
            if obj_type == "linear":
                returns = self.df_data[target].values
                val_fin = np.dot(w, returns)
                if direction == "max": val_fin = -val_fin
                
            elif obj_type == "quadratic":
                cov_matrix = np.array(self.matrix_inputs[target])
                val_fin = w.T @ cov_matrix @ w
                if direction == "max": val_fin = -val_fin
                
            else:
                # Fallback pour d'autres objectifs évaluables analytiquement
                val_fin = np.sum(w)
                
            F[i, 0] = val_fin
            
            # --- 2. CALCUL DES PÉNALITÉS SOFT (OBJECTIFS CONCURRENTS) ---
            for j, cstr in enumerate(self.soft_constraints):
                attr = cstr.get("attribute", "").lower()
                b_type = cstr.get("bound_type")
                cible = cstr.get("value", 0.0)
                
                current_val = 0.0
                
                # Évaluation de l'état actuel du portefeuille pour cette contrainte
                if attr in ["sum_all", "sum", "somme", "total"]:
                    current_val = np.sum(w)
                elif attr in self.col_map_case:
                    real_col = self.col_map_case[attr]
                    targets = [str(t).lower() for t in cstr.get("targets", [])]
                    mask = self.df_data[real_col].astype(str).str.lower().isin(targets).values
                    current_val = np.sum(w[mask])
                    
                # Calcul de la distance/erreur selon le type de borne
                error = 0.0
                if b_type == "eq":
                    error = abs(current_val - cible)
                elif b_type == "max":
                    error = max(0.0, current_val - cible)
                elif b_type == "min":
                    error = max(0.0, cible - current_val)
                elif b_type == "range":
                    min_v = cstr.get("min_value", 0.0)
                    max_v = cstr.get("max_value", 0.0)
                    error = max(0.0, min_v - current_val) + max(0.0, current_val - max_v)
                    
                # Les pénalités occupent les colonnes 1 à N de la matrice des objectifs
                F[i, 1 + j] = error

        out["F"] = F


def solve_moo(optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame) -> dict:
    """
    Point d'entrée pour l'exécuteur. Instancie le problème, la réparation, 
    lance NSGA-II et formate la frontière de Pareto en dictionnaire JSON.
    """
    n_assets = len(df_data)
    constraints_config = optimization_config.get("constraints", [])
    
    problem = PortfolioProblem(n_assets, optimization_config, matrix_inputs, df_data)
    repair = PortfolioRepair(n_assets, constraints_config, df_data)
    
    # Paramétrage institutionnel standard pour NSGA-II
    algorithm = NSGA2(
        pop_size=100,
        repair=repair,
        eliminate_duplicates=True
    )
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        seed=42,
        verbose=False
    )
    
    if res.F is None:
        return {"success": False, "status": "MOO Failed to converge"}
        
    pareto_points = []
    
    # Selon le nombre de solutions, res.F et res.X peuvent être 1D ou 2D
    F_vals = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)
    X_vals = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
    
    for i in range(F_vals.shape[0]):
        # On n'extrait que les poids (seconde moitié de l'ADN) pour l'utilisateur
        w_opt = X_vals[i, n_assets:]
        
        # Constitution du résultat par point
        point_data = {
            "primary_objective_value": float(F_vals[i, 0]),
            "soft_penalties": [float(val) for val in F_vals[i, 1:]],
            "weights": w_opt.tolist()
        }
        pareto_points.append(point_data)
        
    # Tri par l'objectif principal pour un rendu propre
    pareto_points.sort(key=lambda x: x["primary_objective_value"])
        
    return {
        "success": True,
        "status": "MOO Completed",
        "type": "pareto_soft_constraints",
        "soft_constraints_count": len(problem.soft_constraints),
        "pareto_points": pareto_points
    }