import casadi as ca
import numpy as np
import pandas as pd

def parse_casadi_bounds(constraint_dict):
    b_type = constraint_dict["bound_type"]
    v = constraint_dict.get("value")
    if b_type == "eq": return v, v
    elif b_type == "max": return -ca.inf, v 
    elif b_type == "min": return v, ca.inf
    elif b_type == "range":
        min_v = constraint_dict.get("min_value")
        max_v = constraint_dict.get("max_value")
        return (-ca.inf if min_v is None else min_v), (ca.inf if max_v is None else max_v)
    else: raise ValueError(f"Type de borne inconnu : {b_type}")

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
            
            if var["type"] in ["binary", "integer"]:
                # RELAXATION CONTINUE : Les binaires sont déclarées entre 0.0 et 1.0
                self.lbx.extend([0.0] * size)
                self.ubx.extend([1.0] * size)
                self.binary_vars_names.append(name)
            else:
                self.lbx.extend([0.0] * size if name == "x" else [-ca.inf] * size)
                self.ubx.extend([1.0] * size if name == "x" else [ca.inf] * size)

    def build_objective(self, obj_config: dict, matrix_inputs: dict, df_data: pd.DataFrame):
        obj_type = obj_config.get("type")
        target_name = obj_config.get("target_name")         
        secondary_target_name = obj_config.get("secondary_target_name") 
        var_name = obj_config.get("variable_name", "x")     
        direction = obj_config.get("direction", "min")       
        
        current_var = self.vars.get(var_name, self.vars.get("x"))
        eps = 1e-5

        if obj_type == "quadratic":
            matrix_dm = ca.DM(matrix_inputs[target_name]) 
            obj_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            
        elif obj_type == "linear":
            objective_vector = df_data[target_name].values.reshape(1, -1)
            obj_expr = ca.mtimes(objective_vector, current_var)
            
        elif obj_type == "tracking_error":
            matrix_dm = ca.DM(matrix_inputs[target_name]) 
            xb_vector = ca.DM(matrix_inputs.get("Benchmark", [0.0] * self.n_rows))
            active_weights = current_var - xb_vector
            obj_expr = ca.mtimes(ca.mtimes(active_weights.T, matrix_dm), active_weights)

        elif obj_type == "ratio":
            return_vector = df_data[target_name].values.reshape(1, -1)
            lin_return_expr = ca.mtimes(return_vector, current_var)
            matrix_dm = ca.DM(matrix_inputs[secondary_target_name])
            variance_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            obj_expr = lin_return_expr / ca.sqrt(variance_expr + eps)

        elif obj_type == "minimax":
            scenarios_matrix = ca.DM(matrix_inputs[target_name])
            scenario_values = ca.mtimes(scenarios_matrix, current_var)
            obj_expr = 0.005 * ca.log(ca.sum1(ca.exp(scenario_values / 0.005)))

        elif obj_type == "mad":
            scenarios_np = np.array(matrix_inputs[target_name])
            T_scenarios = scenarios_np.shape[0]
            mean_vector = np.mean(scenarios_np, axis=0).reshape(1, -1)
            deviations_np = scenarios_np - mean_vector
            portfolio_deviations = ca.mtimes(ca.DM(deviations_np), current_var)
            obj_expr = ca.sum1(ca.sqrt(portfolio_deviations**2 + eps)) / T_scenarios

        elif obj_type == "turnover":
            x0_vector = ca.DM(matrix_inputs.get("x0", [0.0]*self.n_rows))
            diff = current_var - x0_vector
            obj_expr = ca.sum1(ca.sqrt(diff**2 + eps))

        elif obj_type == "risk_budgeting":
            matrix_dm = ca.DM(matrix_inputs[target_name])
            budgets = ca.DM(matrix_inputs.get("Budgets", [1.0 / self.n_rows] * self.n_rows)) 
            variance_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)
            marginal_risk = ca.mtimes(matrix_dm, current_var)
            risk_contribution = current_var * marginal_risk
            target_contribution = budgets * variance_expr
            obj_expr = ca.sum1((risk_contribution - target_contribution)**2)
            
        elif obj_type == "cvar":
            scenarios_matrix = ca.DM(matrix_inputs[target_name])
            T_scenarios = scenarios_matrix.shape[0]
            beta = 0.95
            
            alpha = ca.MX.sym('alpha', 1) 
            u = ca.MX.sym('u', T_scenarios) 
            
            self.vars['alpha'] = alpha
            self.vars['u'] = u
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
            self.lbx.extend([0.0] * T_scenarios) 
            self.ubx.extend([ca.inf] * T_scenarios)
            
            self.var_indices['alpha'] = (self._current_var_idx, self._current_var_idx + 1)
            self._current_var_idx += 1
            self.var_indices['u'] = (self._current_var_idx, self._current_var_idx + T_scenarios)
            self._current_var_idx += T_scenarios
            
            portfolio_losses = ca.mtimes(scenarios_matrix, current_var)
            for t in range(T_scenarios):
                self.add_constraint(u[t] + alpha - portfolio_losses[t], 0.0, ca.inf)
                
            obj_expr = alpha + (1.0 / (T_scenarios * (1.0 - beta))) * ca.sum1(u)

        else:
            obj_expr = ca.sum1(current_var) 

        if direction == "max": obj_expr = -obj_expr
        self.base_objective = obj_expr
        self.objective = self.base_objective

    def add_alm_penalty(self, lambda_mult: float, mu_val: float):
        """
        Base de l'ALM pour polariser les variables binaires vers 0 ou 1.
        (À fusionner avec ton code pour le scaling des gradients ||∇f||).
        """
        penalty_expr = 0
        for b_name in self.binary_vars_names:
            b_var = self.vars[b_name]
            penalty_expr += ca.sum1(b_var * (1.0 - b_var))
        self.objective = self.base_objective + lambda_mult * penalty_expr

    def remove_alm_penalty(self):
        self.objective = self.base_objective

    def add_constraint(self, expr, lb, ub):
        self.g_exprs.append(expr)
        self.lbg.append(lb)
        self.ubg.append(ub)

    def apply_smart_constraints(self, config_constraints: list, df_data: pd.DataFrame):
        col_map_case = {str(c).lower(): c for c in df_data.columns}

        for cstr in config_constraints:
            
            # --- FILTRE PYMOO (MULTI-OBJECTIF) ---
            if not cstr.get("is_strict", True):
                # Si c'est une Soft Constraint, on l'ignore totalement ici.
                # Elle sera convertie en objectif secondaire par le Repair Operator de Pymoo.
                continue
            # -------------------------------------

            lb, ub = parse_casadi_bounds(cstr)
            attribute_lower = cstr.get("attribute", "").lower()
            current_var = self.vars.get(cstr.get("applied_to", "x"), self.vars.get("x"))
            family = cstr.get("constraint_family", "standard")

            # NOTE: Nous avons retiré le 'if family == "cardinality": continue'
            # CasADi va désormais percevoir la cardinalité sous forme relâchée continue !

            if attribute_lower in ["sum_all", "sum", "somme", "total"]:
                # S'applique à 'x' (Budget 100%) OU à 'b' (Cardinalité relâchée)
                self.add_constraint(ca.sum1(current_var), lb, ub)
                continue
                
            if attribute_lower == "element":
                self.g_exprs.append(current_var)
                self.lbg.extend([lb] * self.n_rows)
                self.ubg.extend([ub] * self.n_rows)
                continue
            
            if family == "min_buy_in":
                b_var = self.vars.get("b")
                if b_var is None: continue
                min_val = cstr.get("min_value", 0.0) 
                for i in range(self.n_rows):
                    self.add_constraint(current_var[i] - b_var[i], -ca.inf, 0.0)
                    self.add_constraint(current_var[i] - min_val * b_var[i], 0.0, ca.inf)
                continue

            # Cas par défaut : Produit scalaire sur des facteurs continus (ESG, Secteurs...)
            # ou sur les binaires (ex: max 3 actifs dans la tech -> exposure_vector * b <= 3)
            if attribute_lower in col_map_case and pd.api.types.is_numeric_dtype(df_data[col_map_case[attribute_lower]]):
                exposure_vector = df_data[col_map_case[attribute_lower]].values.reshape(1, -1)
                self.add_constraint(ca.mtimes(exposure_vector, current_var), lb, ub)

    def build_nlp(self) -> dict:
        g = ca.vertcat(*self.g_exprs) if self.g_exprs else ca.MX()
        x = ca.vertcat(*list(self.vars.values()))
        return {"x": x, "f": self.objective, "g": g}