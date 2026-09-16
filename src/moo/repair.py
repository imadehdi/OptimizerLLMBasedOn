# src/moo/repair.py
import numpy as np
import pandas as pd
import scipy.optimize as spo
import scipy.sparse as sp
from pymoo.core.repair import Repair
from src.solver_v3 import build_ilp_constraints

def parse_bounds(cstr: dict):
    b_type = (cstr.get("bound_type") or "").lower().strip()
    v = float(cstr.get("value", 0.0))
    if b_type == "eq": return v, v
    if b_type == "max": return -np.inf, v
    if b_type == "min": return v, np.inf
    if b_type == "range": return float(cstr.get("min_value", -np.inf)), float(cstr.get("max_value", np.inf))
    return -np.inf, np.inf

def project_to_boxed_simplex(y, s=1.0, lb=None, ub=None, tol=1e-12, max_iter=200):
    y, lb, ub = np.asarray(y, dtype=float), np.asarray(lb if lb is not None else np.zeros(y.size), dtype=float), np.asarray(ub if ub is not None else np.ones(y.size), dtype=float)
    if s < lb.sum() - 1e-9 or s > ub.sum() + 1e-9:
        w = np.clip(y, lb, ub)
        return w / w.sum() * min(max(s, lb.sum()), ub.sum()) if w.sum() > 0 else np.clip(w, lb, ub)
    w, free = np.clip(y, lb, ub).copy(), np.ones(y.size, dtype=bool)
    for _ in range(max_iter):
        r = s - w.sum()
        if abs(r) <= tol or not np.any(free): break
        w[free] += r / free.sum()
        low, up = free & (w < lb), free & (w > ub)
        if not (low.any() or up.any()): continue
        w[low], w[up] = lb[low], ub[up]
        free[low | up] = False
    return np.clip(w, lb, ub)

def _build_casadi_cashflow_solver(obj_instance, is_mip: bool):
    import casadi as ca
    from src.cashflow.forward_model import build_cashflow_expressions_multi_cash, get_cash_indices_by_ccy
    cf = obj_instance.cfg["cashflow"]
    nav0, amount, base_ccy = float(cf["nav0"]), float(cf["amount"]), str(cf.get("base_currency", "USD")).upper().strip()
    tc_enabled, tc_rate = bool(cf.get("tc_enabled", False)), float(cf.get("tc_rate", 0.0))
    w0 = np.asarray(obj_instance.matrix_inputs.get("w0", np.zeros(obj_instance.n_assets)), dtype=float).reshape(-1)
    cash_idx_by_ccy = get_cash_indices_by_ccy(obj_instance.df_data)
    non_cash_idx = obj_instance.df_data.index[obj_instance.df_data["InstrumentType"].astype(str).str.upper().ne("CASH")].astype(int).tolist()
    obj_instance.non_cash_idx = non_cash_idx

    t_buy, t_sell = ca.MX.sym("t_buy", len(non_cash_idx)), ca.MX.sym("t_sell", len(non_cash_idx))
    t_fx_out_by_ccy, t_fx_in_by_ccy, fx_vars = {}, {}, []
    for ccy in cash_idx_by_ccy:
        if ccy != base_ccy:
            out_v, in_v = ca.MX.sym(f"fx_out_{ccy}", 1), ca.MX.sym(f"fx_in_{ccy}", 1)
            t_fx_out_by_ccy[ccy], t_fx_in_by_ccy[ccy] = out_v, in_v
            fx_vars.extend([out_v, in_v])

    x_vars = ca.vertcat(t_buy, t_sell, *fx_vars)
    p_y = ca.MX.sym("p_y", obj_instance.n_assets)
    p_vars = ca.vertcat(p_y, ca.MX.sym("p_z", len(non_cash_idx))) if is_mip else p_y

    fm = build_cashflow_expressions_multi_cash(t_buy=t_buy, t_sell=t_sell, t_fx_out_by_ccy=t_fx_out_by_ccy, t_fx_in_by_ccy=t_fx_in_by_ccy, w0=w0, nav0=nav0, cashflow_amount=amount, df_data=obj_instance.df_data, base_currency=base_ccy, tc_enabled=tc_enabled, tc_rate=tc_rate)
    
    g, lbg, ubg = [fm["v1_bilan"][non_cash_idx]], [0.0] * len(non_cash_idx), [ca.inf] * len(non_cash_idx)
    for expr in fm["cash_post_by_ccy" if tc_enabled else "cash_pre_by_ccy"].values():
        g.append(expr); lbg.append(0.0); ubg.append(ca.inf)

    if is_mip:
        M_val = float(nav0 + amount)
        g.extend([(t_buy / M_val) - p_vars[obj_instance.n_assets:], (t_sell / M_val) - p_vars[obj_instance.n_assets:]])
        lbg.extend([-ca.inf] * (2 * len(non_cash_idx))); ubg.extend([0.0] * (2 * len(non_cash_idx)))

    if hasattr(obj_instance, 'A_cont') and len(obj_instance.A_cont) > 0:
        g.append(ca.mtimes(ca.DM(obj_instance.A_cont.tolist()), fm["w1_expo"]))
        lbg.extend(obj_instance.lb_cont.tolist()); ubg.extend(obj_instance.ub_cont.tolist())

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4, "ipopt.sb": "yes", "ipopt.hessian_approximation": "limited-memory"}
    obj_instance.casadi_solver = ca.nlpsol("repair_cf", "ipopt", {"x": x_vars, "p": p_vars, "f": ca.sum1((fm["w1_expo"] - p_y)**2), "g": ca.vertcat(*g)}, opts)
    obj_instance.casadi_lbx, obj_instance.casadi_ubx, obj_instance.casadi_lbg, obj_instance.casadi_ubg = np.zeros(x_vars.size1()).tolist(), [ca.inf] * x_vars.size1(), lbg, ubg
    obj_instance.w1_expo_fn = ca.Function("w1_expo_fn", [x_vars], [fm["w1_expo"]])

class PortfolioRepairZandW(Repair):
    def __init__(self, n_assets: int, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame, eps_select: float = 1e-4):
        super().__init__()
        self.n_assets, self.cfg, self.matrix_inputs, self.df_data, self.eps_select = n_assets, optimization_config or {}, matrix_inputs or {}, df_data, float(eps_select)
        self.has_cashflow = self.cfg.get("cashflow") is not None
        
        self.w_pre = np.array(self.matrix_inputs.get("w0", np.zeros(self.n_assets))) # PRE-CALCUL DU FALLBACK
        if self.has_cashflow:
            cf = self.cfg["cashflow"]
            nav1 = float(cf["nav0"]) + float(cf["amount"])
            self.w_pre = (self.w_pre * float(cf["nav0"])) / nav1 # Poids dilués parfaits
            
        constraints_config = self.cfg.get("constraints", []) or []
        self.milp_constraints = build_ilp_constraints(constraints_config, self.n_assets, df_data)
        self.col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}
        self.min_buy_in, self.element_lbs, self.element_ubs = 0.0, np.zeros(self.n_assets), np.ones(self.n_assets)
        A_cont, lb_cont, ub_cont, budget_added = [], [], [], False

        for c in constraints_config:
            if not c.get("is_strict", True) or (c.get("applied_to") or "x").lower() not in ["x", "w"]: continue
            attr, fam = str(c.get("attribute", "")).lower(), str(c.get("constraint_family", "")).lower()
            if fam == "min_buy_in": self.min_buy_in = float(c.get("min_value", self.eps_select)); continue
            lb, ub = parse_bounds(c)
            if attr in ["sum_all", "sum", "somme", "total"]: A_cont.append([1.0] * self.n_assets); lb_cont.append(lb); ub_cont.append(ub); budget_added = True; continue
            if attr == "element": self.element_lbs, self.element_ubs = np.maximum(self.element_lbs, lb), np.minimum(self.element_ubs, ub); continue
            if attr in self.col_map_case:
                real_col = self.col_map_case[attr]
                if targets := c.get("targets", []): A_cont.append(self.df_data[real_col].astype(str).str.lower().isin([str(t).lower() for t in targets]).astype(float).values.tolist())
                elif pd.api.types.is_numeric_dtype(self.df_data[real_col]): A_cont.append(self.df_data[real_col].fillna(0.0).values.tolist())
                else: continue
                lb_cont.append(lb); ub_cont.append(ub)

        if not budget_added: A_cont.append([1.0] * self.n_assets); lb_cont.append(1.0); ub_cont.append(1.0)
        self.A_cont, self.lb_cont, self.ub_cont = np.asarray(A_cont, dtype=float), np.asarray(lb_cont, dtype=float), np.asarray(ub_cont, dtype=float)

        if self.has_cashflow: _build_casadi_cashflow_solver(self, is_mip=True)
        else:
            self.slsqp_constraints = []
            A_eq, b_eq, A_ineq, lb_ineq, ub_ineq = [], [], [], [], []
            for a, l, u in zip(A_cont, lb_cont, ub_cont):
                if np.isclose(l, u): A_eq.append(a); b_eq.append(l)
                else: A_ineq.append(a); lb_ineq.append(l); ub_ineq.append(u)
            if A_eq: self.slsqp_constraints.append({'type': 'eq', 'fun': lambda w, A=np.array(A_eq, dtype=float), b=np.array(b_eq, dtype=float): np.dot(A, w) - b})
            if A_ineq:
                A_i, lb_i, ub_i = np.array(A_ineq, dtype=float), np.array(lb_ineq, dtype=float), np.array(ub_ineq, dtype=float)
                def ineq_fun(w, A=A_i, lb=lb_i, ub=ub_i):
                    vals = np.dot(A, w)
                    return np.array([vals[i] - lb[i] if lb[i] > -np.inf else ub[i] - vals[i] for i in range(len(vals)) if lb[i] > -np.inf or ub[i] < np.inf])
                self.slsqp_constraints.append({'type': 'ineq', 'fun': ineq_fun})

    def _do(self, problem, X, **kwargs):
        Xr = np.zeros_like(X, dtype=float)
        x0_cf = np.zeros(len(self.casadi_lbx)) if self.has_cashflow else None

        for k in range(X.shape[0]):
            y_sel, y_w = np.clip(X[k, :self.n_assets], 0.0, 1.0), np.clip(X[k, self.n_assets:], 0.0, 1.0)
            res_milp = spo.milp(c=-y_sel, integrality=np.ones(self.n_assets), bounds=spo.Bounds(0, 1), constraints=self.milp_constraints) if self.milp_constraints else spo.milp(c=-y_sel, integrality=np.ones(self.n_assets), bounds=spo.Bounds(0, 1))
            
            z = np.round(res_milp.x) if res_milp.success else np.zeros(self.n_assets)
            if not res_milp.success: z[np.argmax(y_sel)] = 1.0
            idx = np.where(z > 0.5)[0]
            if idx.size == 0: idx, z[np.argmax(y_sel)] = np.array([np.argmax(y_sel)]), 1.0

            if self.has_cashflow:
                sol = self.casadi_solver(x0=x0_cf, p=np.concatenate([y_w, z[self.non_cash_idx]]), lbx=self.casadi_lbx, ubx=self.casadi_ubx, lbg=self.casadi_lbg, ubg=self.casadi_ubg)
                # FALLBACK AU POIDS INITIAL SI L'OPTIMISATION ECHOUE
                Xr[k, :self.n_assets], Xr[k, self.n_assets:] = z, np.array(self.w1_expo_fn(sol["x"])).reshape(-1) if self.casadi_solver.stats()["success"] else self.w_pre
            else:
                lb_w, ub_w = np.zeros(self.n_assets), np.zeros(self.n_assets)
                lb_w[idx], ub_w[idx] = np.maximum(self.element_lbs[idx], self.min_buy_in), self.element_ubs[idx]
                w0 = np.zeros(self.n_assets); w0[idx] = 1.0 / len(idx)
                try: w = spo.minimize(lambda w: 0.5 * np.sum((w - y_w)**2), x0=w0, jac=lambda w: w - y_w, method='SLSQP', bounds=[(float(l), float(u)) for l, u in zip(lb_w, ub_w)], constraints=self.slsqp_constraints, options={'ftol': 1e-6, 'disp': False}).x
                except Exception: w = w0
                Xr[k, :self.n_assets], Xr[k, self.n_assets:] = z, w
        return Xr

class PortfolioRepairWOnly(Repair):
    def __init__(self, n_assets: int, optimization_config: dict, matrix_inputs: dict, df_data: pd.DataFrame, eps_select: float = 1e-4, feas_tol: float = 1e-8):
        super().__init__()
        self.n_assets, self.cfg, self.matrix_inputs, self.df_data, self.feas_tol = int(n_assets), optimization_config or {}, matrix_inputs or {}, df_data, float(feas_tol)
        self.has_cashflow = self.cfg.get("cashflow") is not None
        
        self.w_pre = np.array(self.matrix_inputs.get("w0", np.zeros(self.n_assets)))
        if self.has_cashflow:
            cf = self.cfg["cashflow"]
            self.w_pre = (self.w_pre * float(cf["nav0"])) / (float(cf["nav0"]) + float(cf["amount"]))

        self.col_map_case, self.element_lbs, self.element_ubs = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}, np.zeros(self.n_assets), np.ones(self.n_assets)
        A_cont, lb_cont, ub_cont, budget_added, self.budget_eq_value = [], [], [], False, None

        for c in self.cfg.get("constraints", []) or []:
            if not c.get("is_strict", True) or (c.get("applied_to") or "x").lower() not in ["x", "w"] or str(c.get("constraint_family", "")).lower() in ["cardinality", "min_buy_in"]: continue
            attr = str(c.get("attribute", "")).lower()
            lb, ub = parse_bounds(c)
            if attr in ["sum_all", "sum", "somme", "total"]: A_cont.append([1.0] * self.n_assets); lb_cont.append(lb); ub_cont.append(ub); budget_added = True; self.budget_eq_value = float(lb) if np.isclose(lb, ub) else None; continue
            if attr == "element": self.element_lbs, self.element_ubs = np.maximum(self.element_lbs, lb), np.minimum(self.element_ubs, ub); continue
            if attr in self.col_map_case:
                real_col = self.col_map_case[attr]
                if targets := c.get("targets", []): A_cont.append(self.df_data[real_col].astype(str).str.lower().isin([str(t).lower() for t in targets]).astype(float).values.tolist())
                elif pd.api.types.is_numeric_dtype(self.df_data[real_col]): A_cont.append(self.df_data[real_col].fillna(0.0).values.tolist())
                else: continue
                lb_cont.append(lb); ub_cont.append(ub)

        if not budget_added: A_cont.append([1.0] * self.n_assets); lb_cont.append(1.0); ub_cont.append(1.0); self.budget_eq_value = 1.0
        self.fast_budget_only = (self.budget_eq_value is not None and len(A_cont) == 1 and np.allclose(np.asarray(A_cont[0], dtype=float), np.ones(self.n_assets)))
        self.A_cont, self.lb_cont, self.ub_cont = np.asarray(A_cont, dtype=float) if len(A_cont) else np.zeros((0, self.n_assets), dtype=float), np.asarray(lb_cont, dtype=float) if len(lb_cont) else np.zeros((0,), dtype=float), np.asarray(ub_cont, dtype=float) if len(ub_cont) else np.zeros((0,), dtype=float)

        if self.has_cashflow: _build_casadi_cashflow_solver(self, is_mip=False)
        else:
            self.osqp_enabled, self.osqp = False, None
            if not self.fast_budget_only:
                try:
                    import osqp
                    self.osqp = osqp.OSQP()
                    self.osqp.setup(P=sp.eye(self.n_assets, format="csc"), q=np.zeros(self.n_assets, dtype=float), A=sp.vstack([sp.eye(self.n_assets, format="csc"), sp.csc_matrix(self.A_cont) if self.A_cont.size else sp.csc_matrix((0, self.n_assets))], format="csc"), l=np.concatenate([np.maximum(0.0, self.element_lbs).astype(float), self.lb_cont]).astype(float), u=np.concatenate([np.minimum(1.0, self.element_ubs).astype(float), self.ub_cont]).astype(float), verbose=False, polish=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=4000)
                    self.osqp_enabled = True
                except ImportError: pass

    def _is_feasible(self, w: np.ndarray) -> bool:
        if np.any(w < (np.maximum(0.0, self.element_lbs) - self.feas_tol)) or np.any(w > (np.minimum(1.0, self.element_ubs) + self.feas_tol)): return False
        if self.A_cont.shape[0] > 0:
            vals = self.A_cont @ w
            if np.any(vals < (self.lb_cont - self.feas_tol)) or np.any(vals > (self.ub_cont + self.feas_tol)): return False
        return True

    def _do(self, problem, X, **kwargs):
        Xr = np.zeros_like(np.asarray(X, dtype=float), dtype=float)
        lb_w, ub_w = np.maximum(0.0, self.element_lbs).astype(float), np.minimum(1.0, self.element_ubs).astype(float)
        x0_cf = np.zeros(len(self.casadi_lbx)) if self.has_cashflow else None

        for k in range(X.shape[0]):
            y = np.clip(X[k, :], 0.0, 1.0)
            if self.has_cashflow:
                sol = self.casadi_solver(x0=x0_cf, p=y, lbx=self.casadi_lbx, ubx=self.casadi_ubx, lbg=self.casadi_lbg, ubg=self.casadi_ubg)
                # FALLBACK AU POIDS INITIAL
                Xr[k, :] = np.array(self.w1_expo_fn(sol["x"])).reshape(-1) if self.casadi_solver.stats()["success"] else self.w_pre
                continue

            if self.fast_budget_only: Xr[k, :] = project_to_boxed_simplex(y, s=float(self.budget_eq_value), lb=lb_w, ub=ub_w); continue
            w_clip = np.clip(y, lb_w, ub_w)
            w0 = project_to_boxed_simplex(w_clip, s=float(self.budget_eq_value), lb=lb_w, ub=ub_w) if self.budget_eq_value is not None else w_clip

            if self._is_feasible(w0): Xr[k, :] = w0; continue
            if self.osqp_enabled:
                try:
                    self.osqp.update(q=-y); self.osqp.warm_start(x=w0)
                    if (res := self.osqp.solve()).info.status_val in (1, 2): Xr[k, :] = np.clip(res.x, lb_w, ub_w); continue
                except Exception: pass
            Xr[k, :] = w0

        return Xr