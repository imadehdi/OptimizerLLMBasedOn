# src/moo/problem.py
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
        
        self.has_cashflow = self.cfg.get("cashflow") is not None
        self.w0 = np.array(self.matrix_inputs.get("w0", np.zeros(self.n_assets)))
        self.w_pre = self.w0.copy()
        self.nav0 = 1.0
        self.nav1 = 1.0
        self.cf_amount = 0.0
        
        if self.has_cashflow:
            cf = self.cfg["cashflow"]
            self.nav0 = float(cf["nav0"])
            self.cf_amount = float(cf["amount"])
            self.nav1 = self.nav0 + self.cf_amount
            base_ccy = str(cf.get("base_currency", "USD")).upper().strip()
            
            try:
                from src.cashflow.forward_model import get_cash_indices_by_ccy
                cash_idx_by_ccy = get_cash_indices_by_ccy(self.df_data)
                base_cash_idx = cash_idx_by_ccy.get(base_ccy, -1)
            except Exception:
                base_cash_idx = -1
                    
            for i in range(self.n_assets):
                if i == base_cash_idx:
                    self.w_pre[i] = (self.w0[i] * self.nav0 + self.cf_amount) / self.nav1
                else:
                    self.w_pre[i] = (self.w0[i] * self.nav0) / self.nav1

        obj_list = self.cfg.get("objectives", [])
        if not obj_list:
            single = self.cfg.get("objective")
            if single: obj_list = [single]

        self.objectives_config = []
        for k, o in enumerate(obj_list):
            o = dict(o)
            o["name"] = o.get("name", f"obj_{k}")
            o["direction"] = o.get("direction", "min")
            self.objectives_config.append(o)

        self.n_obj = len(self.objectives_config)
        self.objective_names = [o["name"] for o in self.objectives_config]

        all_constraints = self.cfg.get("constraints", []) or []
        self.soft_constraints = [c for c in all_constraints if not c.get("is_strict", True)]
        self.n_s = len(self.soft_constraints)
        self.soft_names = [c.get("name", f"soft_{i}") for i, c in enumerate(self.soft_constraints)]

        self.hard_nl_constraints = []
        linear_attributes = ["sum_all", "sum", "somme", "total", "element", "min_buy_in", "turnover"]

        for c in all_constraints:
            if c.get("is_strict", True):
                applied_to = (c.get("applied_to") or "x").lower()
                attr = str(c.get("attribute", "")).lower()
                fam = str(c.get("constraint_family", "")).lower()

                if applied_to in ["x", "w"] and fam != "cardinality" and fam != "min_buy_in":
                    is_linear_col = attr in self.col_map_case
                    is_linear_attr = attr in linear_attributes
                    if not (is_linear_col or is_linear_attr):
                        self.hard_nl_constraints.append(c)

        self.n_ieq_constr = len(self.hard_nl_constraints)

        super().__init__(n_var=n_assets * 2, n_obj=self.n_obj, n_ieq_constr=self.n_ieq_constr, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.zeros((n_pop, self.n_obj))
        G = np.zeros((n_pop, self.n_ieq_constr)) if self.n_ieq_constr > 0 else None
        raw_obj = np.zeros((n_pop, self.n_obj))
        raw_soft = np.zeros((n_pop, self.n_s))

        for i in range(n_pop):
            z = X[i, :self.n_assets]
            w = X[i, self.n_assets:]

            for j, obj_cfg in enumerate(self.objectives_config):
                val = self._compute_metric(z, w, obj_cfg)
                raw_obj[i, j] = val
                F[i, j] = -val if obj_cfg.get("direction") == "max" else val

            for j, cstr in enumerate(self.soft_constraints):
                val = self._compute_metric(z, w, cstr)
                loss = self._compute_violation(val, cstr)
                raw_soft[i, j] = loss
                F[i, self.n_obj + j] = loss

            if G is not None:
                for j, cstr in enumerate(self.hard_nl_constraints):
                    val = self._compute_metric(z, w, cstr)
                    violation = self._compute_violation(val, cstr)
                    G[i, j] = violation

        out["F"] = F
        if G is not None: out["G"] = G
        out["raw_objectives"] = raw_obj
        out["raw_soft_losses"] = raw_soft

    def _compute_metric(self, z, w, config):
        target_name = config.get("target_name") or config.get("attribute")
        obj_type = (config.get("type") or config.get("attribute", "")).lower()
        applied_to = (config.get("applied_to") or "x").lower()

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
            return float(np.sum(np.abs(var_vector - self.w_pre)))

        elif obj_type in ["sum_all", "sum", "somme", "total"]:
            return float(np.sum(var_vector))

        elif obj_type == "linear" or obj_type in self.col_map_case:
            real_col = self.col_map_case.get(obj_type, target_name)
            if real_col in self.df_data.columns:
                targets = config.get("targets", [])
                if targets:
                    targets_norm = [str(t).lower() for t in targets]
                    mask = self.df_data[real_col].astype(str).str.lower().isin(targets_norm).astype(float).values
                    return float(np.dot(var_vector, mask))
                else:
                    if pd.api.types.is_numeric_dtype(self.df_data[real_col]):
                        res = self.df_data[real_col].fillna(0.0).values
                        return float(np.dot(var_vector, res))
        return 0.0

    def _compute_violation(self, val, cstr):
        b_type = (cstr.get("bound_type") or "").lower().strip()
        if b_type == "eq": return abs(val - float(cstr.get("value", 0.0)))
        if b_type == "max": return max(0.0, val - float(cstr.get("value", 0.0)))
        if b_type == "min": return max(0.0, float(cstr.get("value", 0.0)) - val)
        if b_type == "range": return max(0.0, float(cstr.get("min_value", -np.inf)) - val) + max(0.0, val - float(cstr.get("max_value", np.inf)))
        return 0.0


class PortfolioMOOProblemOnly(Problem):
    def __init__(self, n_assets: int, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame):
        self.n_assets = n_assets
        self.cfg = optimization_config or {}
        self.matrix_inputs = matrix_inputs or {}
        self.df_data = df_data

        self.col_map_case = {str(c).lower(): c for c in df_data.columns}
        
        self.has_cashflow = self.cfg.get("cashflow") is not None
        self.w0 = np.array(self.matrix_inputs.get("w0", np.zeros(self.n_assets)))
        self.w_pre = self.w0.copy()
        self.nav0 = 1.0
        self.nav1 = 1.0
        self.cf_amount = 0.0
        
        if self.has_cashflow:
            cf = self.cfg["cashflow"]
            self.nav0 = float(cf["nav0"])
            self.cf_amount = float(cf["amount"])
            self.nav1 = self.nav0 + self.cf_amount
            base_ccy = str(cf.get("base_currency", "USD")).upper().strip()
            
            try:
                from src.cashflow.forward_model import get_cash_indices_by_ccy
                cash_idx_by_ccy = get_cash_indices_by_ccy(self.df_data)
                base_cash_idx = cash_idx_by_ccy.get(base_ccy, -1)
            except Exception:
                base_cash_idx = -1
                    
            for i in range(self.n_assets):
                if i == base_cash_idx:
                    self.w_pre[i] = (self.w0[i] * self.nav0 + self.cf_amount) / self.nav1
                else:
                    self.w_pre[i] = (self.w0[i] * self.nav0) / self.nav1

        obj_list = self.cfg.get("objectives", [])
        if not obj_list:
            single = self.cfg.get("objective")
            if single: obj_list = [single]

        self.objectives_config = []
        for k, o in enumerate(obj_list):
            o = dict(o)
            o["name"] = o.get("name", f"obj_{k}")
            o["direction"] = o.get("direction", "min")
            self.objectives_config.append(o)

        self.n_obj = len(self.objectives_config)
        self.objective_names = [o["name"] for o in self.objectives_config]

        all_constraints = self.cfg.get("constraints", []) or []
        self.soft_constraints = [c for c in all_constraints if not c.get("is_strict", True)]
        self.n_s = len(self.soft_constraints)
        self.soft_names = [c.get("name", f"soft_{i}") for i, c in enumerate(self.soft_constraints)]

        self.hard_nl_constraints = []
        linear_attributes = ["sum_all", "sum", "somme", "total", "element", "min_buy_in", "turnover"]

        for c in all_constraints:
            if c.get("is_strict", True):
                applied_to = (c.get("applied_to") or "x").lower()
                attr = str(c.get("attribute", "")).lower()
                fam = str(c.get("constraint_family", "")).lower()

                if applied_to in ["b", "z"]: continue
                if applied_to in ["x", "w"] and fam != "cardinality" and fam != "min_buy_in":
                    is_linear_col = attr in self.col_map_case
                    is_linear_attr = attr in linear_attributes
                    if not (is_linear_col or is_linear_attr):
                        self.hard_nl_constraints.append(c)

        self.n_ieq_constr = len(self.hard_nl_constraints)
        self._cov_cache, self._vec_cache = {}, {}

        super().__init__(n_var=self.n_assets, n_obj=self.n_obj + self.n_s, n_ieq_constr=self.n_ieq_constr, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F = np.zeros((n_pop, self.n_obj))
        G = np.zeros((n_pop, self.n_ieq_constr)) if self.n_ieq_constr > 0 else None
        raw_obj = np.zeros((n_pop, self.n_obj))
        raw_soft = np.zeros((n_pop, self.n_s))

        for i in range(n_pop):
            w = X[i, :]
            for j, obj_cfg in enumerate(self.objectives_config):
                val = self._compute_metric(w, obj_cfg)
                raw_obj[i, j] = val
                F[i, j] = -val if obj_cfg.get("direction") == "max" else val

            for j, cstr in enumerate(self.soft_constraints):
                val = self._compute_metric(w, cstr)
                loss = self._compute_violation(val, cstr)
                raw_soft[i, j] = loss
                F[i, self.n_obj + j] = loss

            if G is not None:
                for j, cstr in enumerate(self.hard_nl_constraints):
                    val = self._compute_metric(w, cstr)
                    G[i, j] = self._compute_violation(val, cstr)

        out["F"] = F
        if G is not None: out["G"] = G
        out["raw_objectives"] = raw_obj
        out["raw_soft_losses"] = raw_soft

    def _get_cov(self, target_name: str):
        if target_name not in self._cov_cache:
            cov = self.matrix_inputs.get(target_name, None)
            if cov is None: cov = np.eye(self.n_assets)
            self._cov_cache[target_name] = np.asarray(cov, dtype=float)
        return self._cov_cache[target_name]

    def _get_vec(self, key: str, default):
        if key not in self._vec_cache:
            v = self.matrix_inputs.get(key, default)
            self._vec_cache[key] = np.asarray(v, dtype=float)
        return self._vec_cache[key]

    def _compute_metric(self, w, config):
        target_name = config.get("target_name") or config.get("attribute")
        obj_type = (config.get("type") or config.get("attribute", "")).lower()
        applied_to = (config.get("applied_to") or "x").lower()

        if applied_to in ["b", "z"]: return 0.0

        if obj_type in ["quadratic", "variance", "volatility"]:
            return float(w.T @ self._get_cov(target_name) @ w)
        elif obj_type == "tracking_error":
            diff = w - self._get_vec("Benchmark", np.zeros(self.n_assets))
            return float(diff.T @ self._get_cov(target_name) @ diff)
        elif obj_type == "turnover":
            return float(np.sum(np.abs(w - self.w_pre)))
        elif obj_type in ["sum_all", "sum", "somme", "total"]:
            return float(np.sum(w))
        elif obj_type == "linear" or obj_type in self.col_map_case:
            real_col = self.col_map_case.get(obj_type, target_name)
            if real_col in self.df_data.columns:
                targets = config.get("targets", [])
                if targets:
                    mask = self.df_data[real_col].astype(str).str.lower().isin([str(t).lower() for t in targets]).astype(float).values
                    return float(np.dot(w, mask))
                else:
                    if pd.api.types.is_numeric_dtype(self.df_data[real_col]):
                        return float(np.dot(w, self.df_data[real_col].fillna(0.0).values))
        return 0.0

    def _compute_violation(self, val, cstr):
        b_type = (cstr.get("bound_type") or "").lower().strip()
        if b_type == "eq": return abs(val - float(cstr.get("value", 0.0)))
        if b_type == "max": return max(0.0, val - float(cstr.get("value", 0.0)))
        if b_type == "min": return max(0.0, float(cstr.get("value", 0.0)) - val)
        if b_type == "range": return max(0.0, float(cstr.get("min_value", -np.inf)) - val) + max(0.0, val - float(cstr.get("max_value", np.inf)))
        return 0.0