# src/casadi_builder_v3.py
import casadi as ca
import numpy as np
import pandas as pd

def parse_casadi_bounds(constraint_dict):
    b_type = (constraint_dict.get("bound_type") or "").lower().strip()
    v = constraint_dict.get("value", 0.0)
    if b_type == "eq": return v, v
    if b_type == "max": return -ca.inf, v
    if b_type == "min": return v, ca.inf
    if b_type == "range":
        min_v = constraint_dict.get("min_value")
        max_v = constraint_dict.get("max_value")
        return (-ca.inf if min_v is None else min_v, ca.inf if max_v is None else max_v)
    raise ValueError(f"Type de borne inconnu: {b_type}")

class CasadiProblemBuilder:
    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        self.vars = {}
        self.binary_vars_names = []
        self.base_objective = ca.MX(0)
        self.soft_penalty_raw = ca.MX(0)
        self.alm_penalty_raw = ca.MX(0)
        self.scale_base = 1.0
        self.scale_soft = 1.0
        self.scale_alm = 1.0
        self.alm_lambda = 0.0
        self.objective = ca.MX(0)
        self.g_exprs, self.lbg, self.ubg = [], [], []
        self.lbx, self.ubx = [], []
        self.var_indices = {}
        self.current_var_idx = 0
        self.eval_exprs = {}

    def set_evaluation_expression(self, alias: str, expr):
        if alias is not None: self.eval_exprs[str(alias)] = expr

    def _resolve_symbol_or_expr(self, name: str):
        if name in self.vars: return self.vars[name]
        if name in self.eval_exprs: return self.eval_exprs[name]
        return None

    def _resolve_vector(self, preferred_name: str = "x"):
        v = self._resolve_symbol_or_expr(preferred_name)
        if v is not None: return v
        v = self._resolve_symbol_or_expr("x")
        if v is not None: return v
        for _, vv in self.vars.items(): return vv
        return None

    def _resolve_size(self, size_spec):
        if isinstance(size_spec, int): return int(size_spec)
        if isinstance(size_spec, str):
            s = size_spec.strip().lower()
            if s == "n_rows": return int(self.n_rows)
            if s in ["n_non_cash", "n_assets_ex_cash", "n_ex_cash"]: return int(self.n_rows - 1)
        return 1

    def create_variables(self, decision_variables_config: list):
        for var in decision_variables_config:
            name = var["name"]
            size = self._resolve_size(var.get("size", "n_rows"))
            self.vars[name] = ca.MX.sym(name, size)
            self.var_indices[name] = (self.current_var_idx, self.current_var_idx + size)
            self.current_var_idx += size
            vtype = str(var.get("type", "continuous")).lower().strip()
            if vtype in ("binary", "integer"):
                self.lbx.extend([0.0] * size)
                self.ubx.extend([1.0] * size)
                self.binary_vars_names.append(name)
            else:
                if name == "x":
                    self.lbx.extend([0.0] * size)
                    self.ubx.extend([1.0] * size)
                else:
                    self.lbx.extend([-ca.inf] * size)
                    self.ubx.extend([ca.inf] * size)

    def build_objective(self, obj_config: dict, matrix_inputs: dict, df_data: pd.DataFrame):
        obj_type = (obj_config.get("type") or "").lower().strip()
        target_name = obj_config.get("target_name")
        variable_name = obj_config.get("variable_name", "x")
        direction = obj_config.get("direction", "min")
        current_var = self._resolve_vector(variable_name)
        if current_var is None: raise ValueError(f"Variable '{variable_name}' introuvable.")
        
        obj_expr = ca.MX(0)
        if obj_type == "quadratic":
            Sigma = ca.DM(matrix_inputs[target_name])
            obj_expr = ca.mtimes(ca.mtimes(current_var.T, Sigma), current_var)
        elif obj_type == "tracking_error":
            cov = ca.DM(matrix_inputs[target_name])
            w_b = ca.DM(np.asarray(matrix_inputs["Benchmark"], dtype=float).reshape(-1, 1))
            diff = current_var - w_b
            obj_expr = ca.mtimes(ca.mtimes(diff.T, cov), diff)
        elif obj_type == "linear":
            v = df_data[target_name].values.reshape(1, -1)
            obj_expr = ca.mtimes(v, current_var)
        else:
            obj_expr = ca.sum1(current_var)

        if direction == "max": obj_expr = -obj_expr
        self.base_objective = obj_expr
        self._rebuild_objective()

    @staticmethod
    def _is_finite_number(x) -> bool:
        try: return np.isfinite(float(x))
        except: return False

    @staticmethod
    def _to_scalar_float(x, default=None):
        try: return float(x)
        except: return default

    def _violation_expr_scalar(self, g_scalar, lb, ub):
        v = 0
        if self._is_finite_number(lb): v = v + ca.fmax(0, float(lb) - g_scalar)
        if self._is_finite_number(ub): v = v + ca.fmax(0, g_scalar - float(ub))
        return v

    def _violation_expr_vector_sum(self, g_vec, lb, ub):
        g_vec = ca.reshape(g_vec, int(g_vec.size1() * g_vec.size2()), 1)
        v = 0
        if self._is_finite_number(lb): v = v + ca.sum1(ca.fmax(0, float(lb) - g_vec))
        if self._is_finite_number(ub): v = v + ca.sum1(ca.fmax(0, g_vec - float(ub)))
        return v

    def _constraint_expr_from_config(self, cstr: dict, df_data: pd.DataFrame, matrix_inputs: dict):
        lb, ub = parse_casadi_bounds(cstr)
        attribute_lower = (cstr.get("attribute") or "").lower().strip()
        applied_to = (cstr.get("applied_to") or "x").strip()
        family = (cstr.get("constraint_family") or "standard").lower().strip()
        current_var = self._resolve_vector(applied_to)
        if current_var is None or family == "min_buy_in": return None, None, None, None
        col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}

        if attribute_lower in ("variance", "quadratic", "volatility"):
            target_name = cstr.get("target_name") or cstr.get("matrix_name") or "Variance"
            Sigma = ca.DM(matrix_inputs[target_name])
            g = ca.reshape(ca.mtimes(ca.mtimes(current_var.T, Sigma), current_var), 1, 1)
            return g, lb, ub, False
        if attribute_lower in ["sum_all", "sum", "somme", "total"]:
            return ca.reshape(ca.sum1(current_var), 1, 1), lb, ub, False
        if attribute_lower == "element": return current_var, lb, ub, True
        if attribute_lower in col_map_case:
            real_col = col_map_case[attribute_lower]
            targets = cstr.get("targets", [])
            if targets:
                mask = df_data[real_col].astype(str).str.lower().isin([str(t).lower() for t in targets]).astype(float).values
                exposure_vector = mask.reshape(1, -1)
            elif pd.api.types.is_numeric_dtype(df_data[real_col]):
                exposure_vector = df_data[real_col].fillna(0.0).values.reshape(1, -1)
            else: return None, None, None, None
            return ca.reshape(ca.mtimes(exposure_vector, current_var), 1, 1), lb, ub, False
        return None, None, None, None

    def apply_soft_constraints_as_penalty(self, config_constraints: list, df_data: pd.DataFrame, matrix_inputs: dict = None, lambda_soft: float = 1.0, default_tolerance: float = 1e-4, penalty_power: int = 2, eps: float = 1e-12):
        if not config_constraints: return
        p = max(int(penalty_power), 1)
        penalty_sum = ca.MX(0)
        for cstr in config_constraints:
            if cstr.get("is_strict", True): continue
            g, lb, ub, is_vec = self._constraint_expr_from_config(cstr, df_data=df_data, matrix_inputs=matrix_inputs)
            if g is None: continue
            tol = self._to_scalar_float(cstr.get("soft_tolerance", None), default=None)
            if tol is None or tol <= 0: tol = float(default_tolerance)
            viol = self._violation_expr_vector_sum(g, lb, ub) if is_vec else self._violation_expr_scalar(ca.reshape(g, 1, 1)[0, 0], lb, ub)
            norm_v = viol / (float(tol) + float(eps))
            penalty_sum = penalty_sum + (norm_v if p == 1 else norm_v * norm_v if p == 2 else ca.power(norm_v, p))
        self.soft_penalty_raw = float(lambda_soft) * penalty_sum
        self._rebuild_objective()

    def add_constraint(self, expr, lb, ub):
        expr_mx = expr
        n = int(expr_mx.size1() * expr_mx.size2())
        if n == 0: return
        if not (int(expr_mx.size2()) == 1 and int(expr_mx.size1()) == n):
            expr_mx = ca.reshape(expr_mx, n, 1)
        
        def _to_list(x, n_expected):
            arr = np.asarray(x, dtype=float).reshape(-1)
            return [float(arr[0])] * n_expected if arr.size == 1 else [float(v) for v in arr.tolist()]

        self.g_exprs.append(expr_mx)
        self.lbg.extend(_to_list(lb, n))
        self.ubg.extend(_to_list(ub, n))

    def apply_smart_constraints(self, config_constraints: list, df_data: pd.DataFrame, matrix_inputs: dict = None):
        if config_constraints is None: return
        col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}
        for cstr in config_constraints:
            if not cstr.get("is_strict", True): continue
            lb, ub = parse_casadi_bounds(cstr)
            attribute_lower = (cstr.get("attribute") or "").lower().strip()
            applied_to = (cstr.get("applied_to") or "x").strip()
            current_var = self._resolve_vector(applied_to)
            if current_var is None: continue
            family = (cstr.get("constraint_family") or "standard").lower().strip()

            if attribute_lower in ("variance", "quadratic", "volatility"):
                target_name = cstr.get("target_name") or cstr.get("matrix_name") or "Variance"
                Sigma = ca.DM(matrix_inputs[target_name])
                self.add_constraint(ca.reshape(ca.mtimes(ca.mtimes(current_var.T, Sigma), current_var), 1, 1), lb, ub)
                continue
            if attribute_lower in ["sum_all", "sum", "somme", "total"]:
                self.add_constraint(ca.sum1(current_var), lb, ub)
                continue
            if family == "min_buy_in":
                b_var = self.vars.get("b")
                if b_var is None: continue
                min_val = float(cstr.get("min_value", 0.0))
                for i in range(self.n_rows):
                    self.add_constraint(current_var[i] - b_var[i], -ca.inf, 0.0)
                    self.add_constraint(current_var[i] - min_val * b_var[i], 0.0, ca.inf)
                continue
            if attribute_lower == "element":
                self.g_exprs.append(current_var)
                self.lbg.extend([lb] * int(current_var.size1()))
                self.ubg.extend([ub] * int(current_var.size1()))
                continue
            if attribute_lower in col_map_case:
                real_col = col_map_case[attribute_lower]
                targets = cstr.get("targets", [])
                if targets:
                    mask = df_data[real_col].astype(str).str.lower().isin([str(t).lower() for t in targets]).astype(float).values
                    exposure_vector = mask.reshape(1, -1)
                elif pd.api.types.is_numeric_dtype(df_data[real_col]):
                    exposure_vector = df_data[real_col].fillna(0.0).values.reshape(1, -1)
                else: continue
                self.add_constraint(ca.mtimes(exposure_vector, current_var), lb, ub)

    def add_alm_penalty(self, lambda_mult: float, mu_val: float = 0.0):
        penalty_expr = ca.MX(0)
        for b_name in self.binary_vars_names:
            b_var = self.vars[b_name]
            penalty_expr = penalty_expr + ca.sum1(b_var * (1.0 - b_var))
        self.alm_penalty_raw = penalty_expr
        self.alm_lambda = float(lambda_mult) + float(mu_val)
        self._rebuild_objective()

    def remove_alm_penalty(self):
        self.alm_lambda = 0.0
        self._rebuild_objective()

    def apply_gradient_normalisation(self, x0: np.ndarray, eps: float = 1e-12, min_scale: float = 1e-8, max_scale: float = 100.0, include_soft: bool = True, include_alm: bool = True):
        # OPTIMISATION MAX SCALE : bridé à 100.0 pour éviter l'explosion de x0=0 sur des variances
        x_sym = ca.vertcat(*list(self.vars.values())) if self.vars else ca.MX([])
        if x_sym.is_empty(): return
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size != int(x_sym.size1()): return

        def _inf_norm_grad(expr):
            g = ca.gradient(expr, x_sym)
            gv = np.array(ca.Function("grad_tmp", [x_sym], [g])(x0)).reshape(-1)
            return float(np.max(np.abs(gv))) if gv.size else 0.0

        gb = _inf_norm_grad(self.base_objective) if not self.base_objective.is_empty() else 0.0
        self.scale_base = float(np.clip(1.0 / (gb + float(eps)) if gb > 0 else 1.0, float(min_scale), float(max_scale)))

        if include_soft:
            gs = _inf_norm_grad(self.soft_penalty_raw) if not self.soft_penalty_raw.is_empty() else 0.0
            self.scale_soft = float(np.clip(1.0 / (gs + float(eps)) if gs > 0 else 1.0, float(min_scale), float(max_scale)))
        else: self.scale_soft = 1.0

        if include_alm:
            ga = _inf_norm_grad(self.alm_penalty_raw) if not self.alm_penalty_raw.is_empty() else 0.0
            self.scale_alm = float(np.clip(1.0 / (ga + float(eps)) if ga > 0 else 1.0, float(min_scale), float(max_scale)))
        else: self.scale_alm = 1.0

        self._rebuild_objective()

    def _rebuild_objective(self):
        self.objective = float(self.scale_base) * self.base_objective + float(self.scale_soft) * self.soft_penalty_raw + float(self.alm_lambda) * float(self.scale_alm) * self.alm_penalty_raw

    def build_nlp(self) -> dict:
        g = ca.vertcat(*self.g_exprs) if self.g_exprs else ca.MX([])
        x = ca.vertcat(*list(self.vars.values())) if self.vars else ca.MX([])
        return {"x": x, "f": self.objective, "g": g}