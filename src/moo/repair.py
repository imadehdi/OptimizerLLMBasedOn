import numpy as np
import pandas as pd
import scipy.optimize as spo
import scipy.sparse as sp

from pymoo.core.repair import Repair


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


def project_to_boxed_simplex(y, s=1.0, lb=None, ub=None, tol=1e-12, max_iter=200):
    """
    Projection rapide sur { w : sum(w)=s, lb <= w <= ub } via une approche "water-filling" active-set.
    Suffisant pour le cas budget + bornes (ton cas rapide).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.ones(n)

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # Feasibility check grossier
    s_min = lb.sum()
    s_max = ub.sum()
    if s < s_min - 1e-9 or s > s_max + 1e-9:
        # Fallback: clip et normalise au mieux (évite crash)
        w = np.clip(y, lb, ub)
        sw = w.sum()
        if sw > 0:
            return w / sw * min(max(s, s_min), s_max)
        return np.clip(w, lb, ub)

    w = np.clip(y, lb, ub).copy()
    free = np.ones(n, dtype=bool)

    for _ in range(max_iter):
        r = s - w.sum()
        if abs(r) <= tol:
            break
        if not np.any(free):
            break

        # Distribuer le résiduel sur les libres
        w[free] = w[free] + r / free.sum()

        # Fixer ceux qui violent les bornes
        low = free & (w < lb)
        up = free & (w > ub)
        if not (low.any() or up.any()):
            continue

        w[low] = lb[low]
        w[up] = ub[up]
        free[low | up] = False

    return np.clip(w, lb, ub)


class PortfolioRepairZandW(Repair):
    def __init__(self, n_assets: int, constraints_config: list, df_data: pd.DataFrame, eps_select: float = 1e-4):
        super().__init__()
        self.n_assets = n_assets
        self.df_data = df_data
        self.eps_select = float(eps_select)

        # =========================================================
        # INITIALISATION BAC A (MILP - SciPy)
        # =========================================================
        self.milp_constraints = build_ilp_constraints(constraints_config, self.n_assets, df_data)

        # =========================================================
        # INITIALISATION BAC B (Micro-Solveur SLSQP)
        # =========================================================
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

            # =========================================================
            # ETAPE 1 : BAC A (Projection MILP)
            # =========================================================
            c_milp = -y_sel
            integrality = np.ones(N)
            bounds_milp = spo.Bounds(0, 1)

            if self.milp_constraints is not None:
                res_milp = spo.milp(c=c_milp, integrality=integrality, bounds=bounds_milp, constraints=self.milp_constraints)
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

            # =========================================================
            # ETAPE 2 : BAC B (Micro-Solveur SLSQP)
            # =========================================================
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


##################################################
##################################################


class PortfolioRepairWOnly(Repair):
    """
    Repair continu pur:
    - Fast path analytique si budget(eq)+bornes seulement
    - Sinon: test faisabilité -> si OK, no-op
    - Sinon: projection QP (convexe) via OSQP, warm-start depuis projection budget+box
    """
    def __init__(self, n_assets: int, constraints_config: list, df_data: pd.DataFrame, eps_select: float = 1e-4, feas_tol: float = 1e-8):
        super().__init__()
        self.n_assets = int(n_assets)
        self.df_data = df_data
        self.eps_select = float(eps_select)
        self.feas_tol = float(feas_tol)

        self.col_map_case = {str(c).lower(): c for c in df_data.columns} if df_data is not None else {}

        # Bornes élémentaires
        self.element_lbs = np.zeros(self.n_assets)
        self.element_ubs = np.ones(self.n_assets)

        # Contraintes linéaires sur w : A_cont w in [lb_cont, ub_cont]
        A_cont, lb_cont, ub_cont = [], [], []
        budget_added = False
        self.budget_eq_value = None

        for c in constraints_config:
            if not c.get("is_strict", True):
                continue

            applied_to = (c.get("applied_to") or "x").lower()
            if applied_to not in ["x", "w"]:
                continue

            attr = str(c.get("attribute", "")).lower()
            fam = str(c.get("constraint_family", "")).lower()

            # En w-only, si on voit cardinalité/min_buy_in, on suppose que solve_moo
            # doit router vers le mode binaire. On ignore ici.
            if fam in ["cardinality", "min_buy_in"]:
                continue

            # Budget
            if attr in ["sum_all", "sum", "somme", "total"]:
                lb, ub = parse_bounds(c)
                A_cont.append([1.0] * self.n_assets)
                lb_cont.append(lb)
                ub_cont.append(ub)
                budget_added = True
                if np.isclose(lb, ub):
                    self.budget_eq_value = float(lb)
                continue

            # Bornes individuelles
            if attr == "element":
                lb, ub = parse_bounds(c)
                self.element_lbs = np.maximum(self.element_lbs, lb)
                self.element_ubs = np.minimum(self.element_ubs, ub)
                continue

            # Expositions linéaires: colonnes DF
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

        # Sécurité: budget 100% si manquant
        if not budget_added:
            A_cont.append([1.0] * self.n_assets)
            lb_cont.append(1.0)
            ub_cont.append(1.0)
            self.budget_eq_value = 1.0

        # Fast path: uniquement budget eq + bornes
        self.fast_budget_only = (
            self.budget_eq_value is not None
            and len(A_cont) == 1
            and np.allclose(np.asarray(A_cont[0], dtype=float), np.ones(self.n_assets))
        )

        # Stockage dense->numpy pour checks/OSQP
        self.A_cont = np.asarray(A_cont, dtype=float) if len(A_cont) else np.zeros((0, self.n_assets), dtype=float)
        self.lb_cont = np.asarray(lb_cont, dtype=float) if len(lb_cont) else np.zeros((0,), dtype=float)
        self.ub_cont = np.asarray(ub_cont, dtype=float) if len(ub_cont) else np.zeros((0,), dtype=float)

        # QP/OSQP initialisation si fast path n'est pas applicable
        self.osqp_enabled = False
        self.osqp = None
        self.A_osqp = None
        self.l_osqp = None
        self.u_osqp = None

        if not self.fast_budget_only:
            self._init_osqp()

    def _init_osqp(self):
        try:
            import osqp
        except ImportError:
            # OSQP absent: on laissera un fallback plus bas (w0)
            self.osqp_enabled = False
            return

        n = self.n_assets

        # Objectif: 1/2 ||w - y||^2 = 1/2 w^T I w - y^T w + const
        P = sp.eye(n, format="csc")

        # Contraintes OSQP: l <= A_w <= u
        # On encode:
        # - bornes: lb_w <= w <= ub_w via I
        # - contraintes A_cont: lb_cont <= A_cont w <= ub_cont
        I = sp.eye(n, format="csc")
        A_lin = sp.csc_matrix(self.A_cont) if self.A_cont.size else sp.csc_matrix((0, n))
        A = sp.vstack([I, A_lin], format="csc")

        lb_w = np.maximum(0.0, self.element_lbs).astype(float)
        ub_w = np.minimum(1.0, self.element_ubs).astype(float)

        l = np.concatenate([lb_w, self.lb_cont]).astype(float)
        u = np.concatenate([ub_w, self.ub_cont]).astype(float)

        # q sera mis à jour à chaque solve: q = -y
        q0 = np.zeros(n, dtype=float)

        solver = osqp.OSQP()
        solver.setup(
            P=P,
            q=q0,
            A=A,
            l=l,
            u=u,
            verbose=False,
            polish=False,       # Tu peux mettre True en "polishing final" si besoin
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=4000
        )

        self.osqp_enabled = True
        self.osqp = solver
        self.A_osqp = A
        self.l_osqp = l
        self.u_osqp = u

    def _is_feasible(self, w: np.ndarray) -> bool:
        # Bounds
        if np.any(w < (np.maximum(0.0, self.element_lbs) - self.feas_tol)):
            return False
        if np.any(w > (np.minimum(1.0, self.element_ubs) + self.feas_tol)):
            return False

        # Linear constraints
        if self.A_cont.shape[0] > 0:
            vals = self.A_cont @ w
            if np.any(vals < (self.lb_cont - self.feas_tol)):
                return False
            if np.any(vals > (self.ub_cont + self.feas_tol)):
                return False

        return True

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=float)
        Xr = np.zeros_like(X, dtype=float)

        n = self.n_assets
        lb_w = np.maximum(0.0, self.element_lbs).astype(float)
        ub_w = np.minimum(1.0, self.element_ubs).astype(float)

        for k in range(X.shape[0]):
            y = np.clip(X[k, :], 0.0, 1.0)

            # 1) Fast path: budget eq + bornes seulement
            if self.fast_budget_only:
                w = project_to_boxed_simplex(y, s=float(self.budget_eq_value), lb=lb_w, ub=ub_w)
                Xr[k, :] = w
                continue

            # 2) Candidate rapide pour test faisabilité / warm start:
            # - clip bornes
            w_clip = np.clip(y, lb_w, ub_w)

            # Si contrainte budget eq existe, on warm-start avec projection simplex boxé
            # (même s'il y a d'autres contraintes, ça donne un bon point initial)
            w0 = w_clip
            if self.budget_eq_value is not None:
                w0 = project_to_boxed_simplex(w_clip, s=float(self.budget_eq_value), lb=lb_w, ub=ub_w)

            # 3) Test faisabilité: si déjà faisable, no-op
            if self._is_feasible(w0):
                Xr[k, :] = w0
                continue

            # 4) QP via OSQP: projection vers polytope linéaire
            if self.osqp_enabled:
                # min 1/2 w^T I w + q^T w
                # q = -y.astype(float)
                try:
                    self.osqp.update(q=-y)
                    # warm start
                    self.osqp.warm_start(x=w0)
                    res = self.osqp.solve()

                    if res.info.status_val in (1, 2):  # solved / solved inaccurate
                        w = res.x
                        # petite sécurité (numérique)
                        w = np.clip(w, lb_w, ub_w)
                        Xr[k, :] = w
                        continue
                except Exception:
                    pass

            # 5) Fallback: si OSQP indisponible ou échec, on renvoie le warm start
            Xr[k, :] = w0

        return Xr