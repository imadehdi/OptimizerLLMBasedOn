import casadi as ca
import numpy as np
import pandas as pd

def parse_casadi_bounds(constraint_dict):
    b_type = (constraint_dict.get("bound_type") or "").lower().strip()
    v = constraint_dict.get("value", 0.0)
    if b_type == "eq":
        return v, v
    if b_type == "max":
        return -ca.inf, v
    if b_type == "min":
        return v, ca.inf
    if b_type == "range":
        min_v = constraint_dict.get("min_value")
        max_v = constraint_dict.get("max_value")
        return (-ca.inf if min_v is None else min_v, ca.inf if max_v is None else max_v)
    raise ValueError(f"Type de borne inconnu: {b_type}")

def build_ticker_index_map(df_data: pd.DataFrame) -> dict:
    if df_data is None or "Ticker" not in df_data.columns:
        raise ValueError("df_data doit contenir une colonne 'Ticker' pour utiliser les contraintes implication.")
    tickers = df_data["Ticker"].astype(str).tolist()
    return {t.strip().lower(): i for i, t in enumerate(tickers)}

def _normalise_ticker_list(lst) -> list:
    if lst is None:
        return []
    out = []
    for x in lst:
        if x is None:
            continue
        s = str(x).strip().lower()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))

class CasadiProblemBuilder:
    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        self.vars = {}
        self.binary_vars_names = []
        self.base_objective = 0
        self.objective = 0
        self.g_exprs, self.lbg, self.ubg = [], [], []
        self.lbx, self.ubx = [], []
        self.var_indices = {}
        self._current_var_idx = 0

    def create_variables(self, decision_variables_config: list):
        for var in decision_variables_config:
            name = var["name"]
            size = self.n_rows if var["size"] == "n_rows" else 1

            self.vars[name] = ca.MX.sym(name, size)
            self.var_indices[name] = (self._current_var_idx, self._current_var_idx + size)
            self._current_var_idx += size

            if var["type"] in ("binary", "integer"):
                # RELAXATION CONTINUE: On force les bornes à [0.0, 1.0] pour l'optimisation par gradient
                self.lbx.extend([0.0] * size)
                self.ubx.extend([1.0] * size)
                self.binary_vars_names.append(name)
            else:
                self.lbx.extend([0.0 if name == "x" else -ca.inf] * size)
                self.ubx.extend([1.0 if name == "x" else ca.inf] * size)

    def build_objective(self, obj_config: dict, matrix_inputs: dict, df_data: pd.DataFrame):
        obj_type = (obj_config.get("type") or "").lower().strip()
        target_name = obj_config.get("target_name")
        variable_name = obj_config.get("variable_name", "x")
        direction = obj_config.get("direction", "min")

        current_var = self.vars.get(variable_name, self.vars.get("x"))
        if current_var is None:
            raise ValueError(f"Variable {variable_name} introuvable.")

        obj_expr = 0
        if obj_type == "quadratic":
            matrix_dm = ca.DM(matrix_inputs[target_name])
            obj_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
        elif obj_type == "linear":
            objective_vector = df_data[target_name].values.reshape(1, -1)
            obj_expr = ca.mtimes(objective_vector, current_var)
        else:
            obj_expr = ca.sum1(current_var)

        if direction == "max":
            obj_expr = -obj_expr

        # On verrouille l'objectif dans la classe
        self.base_objective = obj_expr
        self.objective = self.base_objective

    def add_alm_penalty(self, lambda_mult: float, mu_val: float = 0.0):
        """Phase 1 : Ajoute la pénalité de binarisation z(1-z) pour forcer les décisions."""
        penalty_expr = 0
        for b_name in self.binary_vars_names:
            b_var = self.vars[b_name]
            penalty_expr += ca.sum1(b_var * (1.0 - b_var))
        
        self.objective = self.base_objective + lambda_mult * penalty_expr + mu_val * penalty_expr

    def remove_alm_penalty(self):
        """Phase 3 : Retire la pénalité pour le Polish final."""
        self.objective = self.base_objective

    def add_constraint(self, expr, lb, ub):
        self.g_exprs.append(expr)
        self.lbg.append(lb)
        self.ubg.append(ub)

    def apply_smart_constraints(self, config_constraints: list, df_data: pd.DataFrame):
        col_map_case = {str(c).lower(): c for c in df_data.columns}

        for cstr in config_constraints:
            # Sécurité de l'Aiguilleur : CasADi ne touche JAMAIS aux contraintes douces
            if not cstr.get("is_strict", True):
                continue

            lb, ub = parse_casadi_bounds(cstr)
            attribute_lower = (cstr.get("attribute") or "").lower().strip()
            applied_to = cstr.get("applied_to", "x")
            current_var = self.vars.get(applied_to)
            
            if current_var is None:
                continue
                
            family = cstr.get("constraint_family", "standard")

            # 1. Budget Total & Cardinalité Relâchée (sum(b) = K est vu par CasADi !)
            if attribute_lower in ["sum_all", "sum", "somme", "total"]:
                self.add_constraint(ca.sum1(current_var), lb, ub)
                continue
                
            # 2. Min Buy-in (Contrainte croisée reliant x et b)
            if family == "min_buy_in":
                b_var = self.vars.get("b")
                if b_var is None: 
                    continue
                min_val = cstr.get("min_value", 0.0) 
                for i in range(self.n_rows):
                    self.add_constraint(current_var[i] - b_var[i], -ca.inf, 0.0)
                    self.add_constraint(current_var[i] - min_val * b_var[i], 0.0, ca.inf)
                continue

            # 3. Limites individuelles par actif (ex: w_max = 5%)
            if attribute_lower == "element":
                self.g_exprs.append(current_var)
                self.lbg.extend([lb] * self.n_rows)
                self.ubg.extend([ub] * self.n_rows)
                continue

            # 4. Limites linéaires (Secteurs, ESG...)
            if attribute_lower in col_map_case:
                real_col = col_map_case[attribute_lower]
                targets = cstr.get("targets", [])
                
                if targets:
                    targets_norm = [str(t).lower() for t in targets]
                    mask = df_data[real_col].astype(str).str.lower().isin(targets_norm).astype(float).values
                    exposure_vector = mask.reshape(1, -1)
                else:
                    if pd.api.types.is_numeric_dtype(df_data[real_col]):
                        exposure_vector = df_data[real_col].fillna(0.0).values.reshape(1, -1)
                    else:
                        continue
                        
                self.add_constraint(ca.mtimes(exposure_vector, current_var), lb, ub)

    def build_nlp(self) -> dict:
        g = ca.vertcat(*self.g_exprs) if self.g_exprs else ca.MX([])
        x = ca.vertcat(*list(self.vars.values()))
        return {"x": x, "f": self.objective, "g": g}