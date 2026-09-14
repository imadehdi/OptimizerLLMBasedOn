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
    """
    Builder CasADi générique.

    Extension CASHFLOW:
    - possibilité d'enregistrer une "expression d'évaluation" pour un nom de variable
      (ex: alias 'x' => w_final_expr), afin d'appliquer objectifs/contraintes sur w_final,
      tout en gardant la variable de décision distincte (ex: 't' trades).
    """

    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        self.vars = {}
        self.binary_vars_names = []
        self.base_objective = 0
        self.objective = 0

        self.g_exprs, self.lbg, self.ubg = [], [], []
        self.lbx, self.ubx = [], []
        self.var_indices = {}
        self.current_var_idx = 0

        # NEW: mapping alias -> CasADi expression (MX)
        # Example: self.eval_exprs["x"] = w_final_expr
        self.eval_exprs = {}

    def set_evaluation_expression(self, alias: str, expr):
        """Associe un alias (ex: 'x') à une expression CasADi (ex: w_final_expr)."""
        if alias is None:
            return
        self.eval_exprs[str(alias)] = expr

    def _resolve_symbol_or_expr(self, name: str):
        """Retourne soit une variable décisionnelle, soit une expression d'évaluation."""
        if name in self.vars:
            return self.vars[name]
        if name in self.eval_exprs:
            return self.eval_exprs[name]
        return None

    def _resolve_vector(self, preferred_name: str = "x"):
        """Fallback: renvoie un vecteur 'raisonnable' pour objectifs/contraintes."""
        v = self._resolve_symbol_or_expr(preferred_name)
        if v is not None:
            return v
        v = self._resolve_symbol_or_expr("x")
        if v is not None:
            return v
        # sinon première variable disponible
        for _, vv in self.vars.items():
            return vv
        return None

    def _resolve_size(self, size_spec):
        """
        Supporte:
        - "n_rows"
        - "n_non_cash" (pour cashflow trades)
        - int
        """
        if isinstance(size_spec, int):
            return int(size_spec)

        if isinstance(size_spec, str):
            s = size_spec.strip().lower()
            if s == "n_rows":
                return int(self.n_rows)
            if s in ["n_non_cash", "n_assets_ex_cash", "n_ex_cash"]:
                # Convention: cash est un instrument, trades dimension = n_rows-1
                return int(self.n_rows - 1)

        # fallback
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
                # RELAXATION CONTINUE: On force les bornes à [0.0, 1.0] pour l'optimisation par gradient
                self.lbx.extend([0.0] * size)
                self.ubx.extend([1.0] * size)
                self.binary_vars_names.append(name)
            else:
                # IMPORTANT:
                # - Pour le mode classique weights: 'x' est borné [0,1]
                # - Pour cashflow trades (ex: 't'): bornes par défaut infinies (les contraintes physiques gèrent le reste)
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
        if current_var is None:
            raise ValueError(f"Variable/Expression '{variable_name}' introuvable pour l'objectif.")

        obj_expr = 0
        if obj_type == "quadratic":
            if target_name is None:
                raise ValueError("Objectif quadratique: target_name manquant.")
            matrix_dm = ca.DM(matrix_inputs[target_name])
            obj_expr = ca.mtimes(ca.mtimes(current_var.T, matrix_dm), current_var)

        elif obj_type == "linear":
            if target_name is None:
                raise ValueError("Objectif linéaire: target_name manquant.")
            objective_vector = df_data[target_name].values.reshape(1, -1)
            obj_expr = ca.mtimes(objective_vector, current_var)

        else:
            # fallback: somme
            obj_expr = ca.sum1(current_var)

        if direction == "max":
            obj_expr = -obj_expr

        self.base_objective = obj_expr
        self.objective = self.base_objective

    def add_alm_penalty(self, lambda_mult: float, mu_val: float = 0.0):
        """Ajoute pénalité de binarisation z(1-z) pour forcer les décisions."""
        penalty_expr = 0
        for b_name in self.binary_vars_names:
            b_var = self.vars[b_name]
            penalty_expr += ca.sum1(b_var * (1.0 - b_var))

        self.objective = self.base_objective + lambda_mult * penalty_expr + mu_val * penalty_expr

    def remove_alm_penalty(self):
        """Retire la pénalité."""
        self.objective = self.base_objective

    def add_constraint(self, expr, lb, ub):
        """
        Ajoute une contrainte lb <= expr <= ub.

        Supporte:
        - expr scalaire ou vecteur (CasADi MX/DM)
        - lb/ub scalaires ou vecteurs (list/np/dm) de taille compatible

        Assure que len(lbg)==len(ubg)==dim(g) après vertcat.
        """
        # 1) Normaliser expr en vecteur colonne
        expr_mx = expr
        # CasADi: un scalaire a size1=size2=1, un vecteur a size2=1 (souvent)
        n = int(expr_mx.size1() * expr_mx.size2())

        if n == 0:
            return

        # Forcer en (n,1) si besoin
        if not (int(expr_mx.size2()) == 1 and int(expr_mx.size1()) == n):
            expr_mx = ca.reshape(expr_mx, n, 1)

        # 2) Helpers pour normaliser lb/ub
        def _to_list(x, n_expected, is_lb: bool):
            # Cas scalaire python
            if isinstance(x, (int, float, np.floating)):
                return [float(x)] * n_expected

            # Cas list/tuple
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.asarray(x, dtype=float).reshape(-1)
                if arr.size == 1:
                    return [float(arr[0])] * n_expected
                if arr.size != n_expected:
                    raise ValueError(
                        f"Constraint bound size mismatch: expected {n_expected} but got {arr.size} "
                        f"({'lb' if is_lb else 'ub'})."
                    )
                return [float(v) for v in arr.tolist()]

            # Cas CasADi DM/MX
            if isinstance(x, (ca.DM, ca.MX)):
                # DM: on peut convertir; MX: on ne peut pas l'évaluer -> on refuse
                if isinstance(x, ca.MX):
                    raise ValueError("lb/ub must be numeric (not CasADi MX).")
                arr = np.array(x).reshape(-1).astype(float)
                if arr.size == 1:
                    return [float(arr[0])] * n_expected
                if arr.size != n_expected:
                    raise ValueError(
                        f"Constraint bound size mismatch: expected {n_expected} but got {arr.size} "
                        f"({'lb' if is_lb else 'ub'})."
                    )
                return [float(v) for v in arr.tolist()]

            # Fallback: try float
            try:
                return [float(x)] * n_expected
            except Exception as e:
                raise ValueError(f"Unsupported bound type for {'lb' if is_lb else 'ub'}: {type(x)}") from e

        lb_list = _to_list(lb, n, is_lb=True)
        ub_list = _to_list(ub, n, is_lb=False)

        # 3) Stockage
        self.g_exprs.append(expr_mx)
        self.lbg.extend(lb_list)
        self.ubg.extend(ub_list)

    def apply_smart_constraints(self, config_constraints: list, df_data: pd.DataFrame, matrix_inputs: dict = None):
        """
        Applique des contraintes strictes au NLP CasADi.

        Supporte:
        - sum_all/sum/total (budget)
        - element (bornes individuelles)
        - contraintes linéaires via colonnes df (Sector, Ticker, etc.) avec/ou sans targets
        - min_buy_in (liant x et b, inchangé)
        - NEW (Option B): variance/quadratic constraint sur weights
          attribute in {"variance", "quadratic", "volatility"}
          target_name doit pointer vers une matrice NxN dans matrix_inputs (ex: "Variance")
          expr = w^T Sigma w
        """
        if config_constraints is None:
            return

        col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}

        for cstr in config_constraints:
            # CasADi ignore les contraintes douces
            if not cstr.get("is_strict", True):
                continue

            lb, ub = parse_casadi_bounds(cstr)

            attribute_lower = (cstr.get("attribute") or "").lower().strip()
            applied_to = (cstr.get("applied_to") or "x").strip()

            current_var = self._resolve_vector(applied_to)
            if current_var is None:
                continue

            family = (cstr.get("constraint_family") or "standard").lower().strip()

            # #################################################========
            # 0) NEW: Variance / Quadratic constraint (Option B)
            # #################################################========
            # Config example:
            # {"applied_to":"x", "attribute":"variance", "target_name":"Variance", "bound_type":"max", "value": 2.5e-6}
            if attribute_lower in ("variance", "quadratic", "volatility"):
                if matrix_inputs is None:
                    raise ValueError("Variance/quadratic constraints require matrix_inputs (covariance matrix).")

                target_name = cstr.get("target_name") or cstr.get("matrix_name") or "Variance"
                if target_name not in matrix_inputs:
                    raise ValueError(f"Missing covariance matrix in matrix_inputs['{target_name}'].")

                Sigma = ca.DM(matrix_inputs[target_name])

                # Basic shape validation
                n = int(current_var.size1())
                if int(Sigma.size1()) != n or int(Sigma.size2()) != n:
                    raise ValueError(
                        f"Covariance shape mismatch for '{target_name}': "
                        f"got ({int(Sigma.size1())}x{int(Sigma.size2())}), expected {n}x{n}."
                    )

                # Variance expression: w^T Sigma w (scalar)
                var_expr = ca.mtimes(ca.mtimes(current_var.T, Sigma), current_var)
                # Ensure scalar (1x1)
                var_expr = ca.reshape(var_expr, 1, 1)

                self.add_constraint(var_expr, lb, ub)
                continue

            # #################################################========
            # 1) Budget total (sum_all)
            # #################################################========
            if attribute_lower in ["sum_all", "sum", "somme", "total"]:
                self.add_constraint(ca.sum1(current_var), lb, ub)
                continue

            # #################################################========
            # 2) Min Buy-in (croisée x et b) - inchangé
            # #################################################========
            if family == "min_buy_in":
                b_var = self.vars.get("b")
                if b_var is None:
                    continue
                min_val = float(cstr.get("min_value", 0.0))
                for i in range(self.n_rows):
                    self.add_constraint(current_var[i] - b_var[i], -ca.inf, 0.0)
                    self.add_constraint(current_var[i] - min_val * b_var[i], 0.0, ca.inf)
                continue

            # #################################################========
            # 3) Bornes individuelles (element)
            # #################################################========
            if attribute_lower == "element":
                # contrainte vectorielle: lb <= w_i <= ub
                self.g_exprs.append(current_var)
                self.lbg.extend([lb] * int(current_var.size1()))
                self.ubg.extend([ub] * int(current_var.size1()))
                continue

            # #################################################========
            # 4) Limites linéaires via colonnes DF
            # #################################################========
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
                continue

            # Sinon: contrainte non gérée ici (silencieux)
            # Tu peux lever une erreur si tu veux une transparence totale.

    def build_nlp(self) -> dict:
        g = ca.vertcat(*self.g_exprs) if self.g_exprs else ca.MX([])
        x = ca.vertcat(*list(self.vars.values())) if self.vars else ca.MX([])
        return {"x": x, "f": self.objective, "g": g}