import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

class PortfolioMOOProblem(Problem):
    def __init__(self, n_assets: int, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame):
        self.n_assets = n_assets
        self.cfg = optimization_config or {}
        self.matrix_inputs = matrix_inputs or {}
        self.df_data = df_data
        
        self.col_map_case = {str(c).lower(): c for c in df_data.columns}
        
        # --- 1. PARSING DES OBJECTIFS (M) ---
        obj_list = self.cfg.get("objectives", [])
        if not obj_list:
            single = self.cfg.get("objective")
            if single:
                obj_list = [single]
                
        self.objectives_config = []
        for k, o in enumerate(obj_list):
            o = dict(o)
            o["name"] = o.get("name", f"obj_{k}")
            o["direction"] = o.get("direction", "min")
            self.objectives_config.append(o)
            
        self.M = len(self.objectives_config)
        self.objective_names = [o["name"] for o in self.objectives_config]

        # --- 2. PARSING DU BAC D (Soft Constraints -> S) ---
        all_constraints = self.cfg.get("constraints", []) or []
        self.soft_constraints = [c for c in all_constraints if not c.get("is_strict", True)]
        self.S = len(self.soft_constraints)
        self.soft_names = [c.get("name", f"soft_{i}") for i, c in enumerate(self.soft_constraints)]

        # --- 3. PARSING DU BAC C (Hard Non-Linear Constraints -> G) ---
        self.hard_nl_constraints = []
        # Liste des attributs qui sont linéaires et traités par le Repair Operator (Bac B)
        linear_attributes = ["sum_all", "sum", "somme", "total", "element", "min_buy_in", "turnover"]
        
        for c in all_constraints:
            if c.get("is_strict", True):
                applied_to = (c.get("applied_to") or "x").lower()
                attr = str(c.get("attribute", "")).lower()
                fam = str(c.get("constraint_family", "")).lower()
                
                # On isole les contraintes sur w/x qui NE SONT PAS gérées par le solveur linéaire
                if applied_to in ["x", "w"] and fam != "cardinality" and fam != "min_buy_in":
                    is_linear_col = attr in self.col_map_case
                    is_linear_attr = attr in linear_attributes
                    
                    if not (is_linear_col or is_linear_attr):
                        self.hard_nl_constraints.append(c)
                        
        self.n_ieq = len(self.hard_nl_constraints)

        super().__init__(
            n_var=n_assets * 2, # x de taille 2N : [z (binaire), w (continu)]
            n_obj=self.M + self.S, # Objectifs + Soft Constraints
            n_ieq_constr=self.n_ieq, # Violations Dures Non-Linéaires
            xl=0.0,
            xu=1.0
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        
        F = np.zeros((n_pop, self.n_obj))
        G = np.zeros((n_pop, self.n_ieq_constr)) if self.n_ieq_constr > 0 else None
        
        raw_obj = np.zeros((n_pop, self.M))
        raw_soft = np.zeros((n_pop, self.S))

        for i in range(n_pop):
            # Extraction des deux génomes (Z et W)
            z = X[i, :self.n_assets]
            w = X[i, self.n_assets:]

            # --- CALCUL DES OBJECTIFS PRINCIPAUX ---
            for j, obj_cfg in enumerate(self.objectives_config):
                val = self._compute_metric(z, w, obj_cfg)
                raw_obj[i, j] = val
                F[i, j] = -val if obj_cfg.get("direction") == "max" else val

            # --- CALCUL DU BAC D (Soft Constraints) ---
            for j, cstr in enumerate(self.soft_constraints):
                val = self._compute_metric(z, w, cstr)
                loss = self._compute_violation(val, cstr)
                raw_soft[i, j] = loss
                F[i, self.M + j] = loss

            # --- CALCUL DU BAC C (Hard Non-Linear Constraints) ---
            if G is not None:
                for j, cstr in enumerate(self.hard_nl_constraints):
                    val = self._compute_metric(z, w, cstr)
                    violation = self._compute_violation(val, cstr)
                    G[i, j] = violation

        out["F"] = F
        if G is not None:
            out["G"] = G
            
        out["raw_objectives"] = raw_obj
        out["raw_soft_losses"] = raw_soft

    def _compute_metric(self, z, w, config):
        """Moteur d'évaluation agnostique : gère z, w, le numérique et le catégoriel."""
        target_name = config.get("target_name") or config.get("attribute")
        obj_type = (config.get("type") or config.get("attribute", "")).lower()
        applied_to = (config.get("applied_to") or "x").lower()
        
        # Le moteur choisit le bon chromosome selon le dictionnaire
        var_vector = z if applied_to in ["b", "z"] else w

        if obj_type in ["quadratic", "variance", "volatility"]:
            cov = np.array(self.matrix_inputs.get(target_name, np.eye(self.n_assets)))
            return float(var_vector.T @ cov @ var_vector)
            
        elif obj_type == "tracking_error":
            cov = np.array(self.matrix_inputs.get(target_name, np.eye(self.n_assets)))
            w_b = np.array(self.matrix_inputs.get("Benchmark", np.zeros(self.n_assets)))
            diff = var_vector - w_b
            return float(diff.T @ cov @ diff)
            
        elif obj_type == "turnover":
            w0 = np.array(self.matrix_inputs.get("w0", np.zeros(self.n_assets)))
            return float(np.sum(np.abs(var_vector - w0)))

        elif obj_type in ["sum_all", "sum", "somme", "total"]:
            return float(np.sum(var_vector))
            
        elif obj_type == "linear" or obj_type in self.col_map_case:
            real_col = self.col_map_case.get(obj_type, target_name)
            if real_col in self.df_data.columns:
                targets = config.get("targets", [])
                
                if targets:
                    # Cas Catégoriel (Ex: Sector == Tech) -> Création d'un masque 0/1
                    targets_norm = [str(t).lower() for t in targets]
                    mask = self.df_data[real_col].astype(str).str.lower().isin(targets_norm).astype(float).values
                    return float(np.dot(var_vector, mask))
                else:
                    # Cas Numérique pur (Ex: Rendement ou Score ESG)
                    if pd.api.types.is_numeric_dtype(self.df_data[real_col]):
                        ret = self.df_data[real_col].fillna(0.0).values
                        return float(np.dot(var_vector, ret))
                
        return 0.0

    def _compute_violation(self, val, cstr):
        """Calcule l'erreur absolue. Renvoie 0.0 si la contrainte est respectée."""
        b_type = (cstr.get("bound_type") or "").lower().strip()
        
        if b_type == "eq":
            target = float(cstr.get("value", 0.0))
            return abs(val - target) 
        if b_type == "max":
            ub = float(cstr.get("value", 0.0))
            return max(0.0, val - ub) 
        if b_type == "min":
            lb = float(cstr.get("value", 0.0))
            return max(0.0, lb - val) 
        if b_type == "range":
            lb = float(cstr.get("min_value", -np.inf))
            ub = float(cstr.get("max_value", np.inf))
            return max(0.0, lb - val) + max(0.0, val - ub)
        
        return 0.0