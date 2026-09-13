import numpy as np
import scipy.optimize as spo
import pandas as pd
from pymoo.core.repair import Repair
from src.solver_v3 import build_ilp_constraints

def parse_bounds(cstr: dict):
    b_type = (cstr.get("bound_type") or "").lower().strip()
    v = float(cstr.get("value", 0.0))
    if b_type == "eq": 
        return v, v
    if b_type == "max": 
        return -np.inf, v
    if b_type == "min": 
        return v, np.inf
    if b_type == "range": 
        return float(cstr.get("min_value", -np.inf)), float(cstr.get("max_value", np.inf))
    return -np.inf, np.inf

class PortfolioRepairZandW(Repair):
    def __init__(self, n_assets: int, constraints_config: list, df_data: pd.DataFrame, eps_select: float = 1e-4):
        super().__init__()
        self.n_assets = n_assets
        self.df_data = df_data
        self.eps_select = float(eps_select)
        
        # ==========================================
        # INITIALISATION BAC A (MILP - SciPy)
        # ==========================================
        self.milp_constraints = build_ilp_constraints(constraints_config, self.n_assets, df_data)
        
        # ==========================================
        # INITIALISATION BAC B (Micro-Solveur SLSQP)
        # ==========================================
        self.col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}
        
        self.min_buy_in = 0.0
        self.element_lbs = np.zeros(self.n_assets)
        self.element_ubs = np.ones(self.n_assets)
        
        A_cont, lb_cont, ub_cont = [], [], []
        budget_added = False
        
        for c in constraints_config:
            if not c.get("is_strict", True):
                continue
                
            applied_to = (c.get("applied_to") or "x").lower()
            if applied_to not in ["x", "w"]:
                continue
                
            attr = str(c.get("attribute", "")).lower()
            fam = str(c.get("constraint_family", "")).lower()
            
            # Extraction du Min Buy-in
            if fam == "min_buy_in":
                self.min_buy_in = float(c.get("min_value", self.eps_select))
                continue
                
            # Extraction du Budget Global
            if attr in ["sum_all", "sum", "somme", "total"]:
                lb, ub = parse_bounds(c)
                A_cont.append([1.0] * self.n_assets)
                lb_cont.append(lb)
                ub_cont.append(ub)
                budget_added = True
                continue
                
            # Extraction des Plafonds Individuels (w_max)
            if attr == "element":
                lb, ub = parse_bounds(c)
                self.element_lbs = np.maximum(self.element_lbs, lb)
                self.element_ubs = np.minimum(self.element_ubs, ub)
                continue
                
            # Extraction des Limites Linéaires Sectorielles/ESG (Bac B)
            if attr in self.col_map_case:
                real_col = self.col_map_case[attr]
                targets = c.get("targets", [])
                lb, ub = parse_bounds(c)
                
                if targets:
                    targets_norm = [str(t).lower() for t in targets]
                    mask = self.df_data[real_col].astype(str).str.lower().isin(targets_norm).astype(float).values
                    A_cont.append(mask.tolist())
                elif pd.api.types.is_numeric_dtype(self.df_data[real_col]):
                    expo = self.df_data[real_col].fillna(0.0).values
                    A_cont.append(expo.tolist())
                else:
                    continue
                lb_cont.append(lb)
                ub_cont.append(ub)
        
        # Sécurité : Forcer le budget à 100% s'il a été oublié par l'IA
        if not budget_added:
            A_cont.append([1.0] * self.n_assets)
            lb_cont.append(1.0)
            ub_cont.append(1.0)
            
        self.slsqp_constraints = []
        A_eq, b_eq = [], []
        # CORRECTION APPLIQUÉE ICI : 3 listes pour 3 variables
        A_ineq, lb_ineq, ub_ineq = [], [], []
        
        # Tri chirurgical : Séparer les égalités des inégalités pour SLSQP
        for a, l, u in zip(A_cont, lb_cont, ub_cont):
            if np.isclose(l, u):
                A_eq.append(a)
                b_eq.append(l)
            else:
                A_ineq.append(a)
                lb_ineq.append(l)
                ub_ineq.append(u)
                
        # Format Dictionnaire pour garantir la compatibilité absolue avec SLSQP
        if A_eq:
            A_eq_arr = np.array(A_eq, dtype=float)
            b_eq_arr = np.array(b_eq, dtype=float)
            self.slsqp_constraints.append({
                'type': 'eq',
                'fun': lambda w, A=A_eq_arr, b=b_eq_arr: np.dot(A, w) - b
            })
            
        if A_ineq:
            A_ineq_arr = np.array(A_ineq, dtype=float)
            lb_ineq_arr = np.array(lb_ineq, dtype=float)
            ub_ineq_arr = np.array(ub_ineq, dtype=float)
            
            def ineq_fun(w, A=A_ineq_arr, lb=lb_ineq_arr, ub=ub_ineq_arr):
                vals = np.dot(A, w)
                res = []
                for i in range(len(vals)):
                    if lb[i] > -np.inf:
                        res.append(vals[i] - lb[i])
                    if ub[i] < np.inf:
                        res.append(ub[i] - vals[i])
                return np.array(res)
                
            self.slsqp_constraints.append({
                'type': 'ineq',
                'fun': ineq_fun
            })

    def _do(self, problem, X, **kwargs):
        Xr = np.zeros_like(X, dtype=float)
        N = self.n_assets

        for k in range(X.shape[0]):
            y = X[k, :]
            y_sel = np.clip(y[:N], 0.0, 1.0)
            y_w = np.clip(y[N:], 0.0, 1.0)

            # ==========================================
            # ETAPE 1 : BAC A (Projection MILP)
            # ==========================================
            c_milp = -y_sel
            integrality = np.ones(N)
            bounds_milp = spo.Bounds(0, 1)

            if self.milp_constraints is not None:
                res_milp = spo.milp(c=c_milp, integrality=integrality, bounds=bounds_milp, constraints=[self.milp_constraints])
            else:
                res_milp = spo.milp(c=c_milp, integrality=integrality, bounds=bounds_milp)

            # Fallback si le problème logique est irréalisable
            if res_milp.success:
                z = np.round(res_milp.x)
            else:
                z = np.zeros(N)
                z[np.argmax(y_sel)] = 1.0

            # Sécurité pour éviter un masque vide
            idx = np.where(z > 0.5)[0]
            if idx.size == 0:
                idx = np.array([np.argmax(y_sel)])
                z[idx[0]] = 1.0

            # ==========================================
            # ETAPE 2 : BAC B (Micro-Solveur SLSQP)
            # ==========================================
            lb_w = np.zeros(N)
            ub_w = np.zeros(N)
            
            # On applique les bornes UNIQUEMENT sur les actifs sélectionnés par le Bac A.
            lb_w[idx] = np.maximum(self.element_lbs[idx], self.min_buy_in)
            ub_w[idx] = self.element_ubs[idx]

            # Format robuste pour SLSQP
            bounds_list = [(float(l), float(u)) for l, u in zip(lb_w, ub_w)]

            # Fonction Objectif : Projeter y_w au plus proche de w
            def obj_fun(w_var):
                return 0.5 * np.sum((w_var - y_w)**2)
                
            def jac_fun(w_var):
                return w_var - y_w
            
            # Démarrage intelligent (1/K sur les actifs sélectionnés)
            w0 = np.zeros(N)
            w0[idx] = 1.0 / len(idx)

            try:
                res_slsqp = spo.minimize(
                    obj_fun, 
                    x0=w0, 
                    jac=jac_fun,
                    method='SLSQP', 
                    bounds=bounds_list, 
                    constraints=self.slsqp_constraints,
                    options={'ftol': 1e-6, 'disp': False}
                )
                w = res_slsqp.x if res_slsqp.success else w0
            except Exception:
                w = w0

            Xr[k, :N] = z
            Xr[k, N:] = w

        return Xr