# src/constraints/preprocess_v3.py
import copy
import numpy as np
import pandas as pd

def _normalise_weights(w: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    s = float(np.sum(w))
    if abs(s) < eps:
        return w
    return w / s

def _slice_square(mat, idx):
    m = np.asarray(mat, dtype=float)
    idx = np.asarray(idx, dtype=int)
    return m[np.ix_(idx, idx)]

def preprocess_config_and_df_v3(optimization_config: dict, df_data: pd.DataFrame, matrix_inputs: dict | None = None):
    cfg = copy.deepcopy(optimization_config)
    constraints_in = list(cfg.get("constraints", [])) or []
    df_local = df_data.copy()

    def _norm_ticker(x) -> str:
        return str(x).strip().lower()

    def _is_strict(c) -> bool:
        return bool(c.get("is_strict", True))

    # --- Pass 1: rewrite exclusion/exclusive -> cardinality group form ---
    constraints_pass1 = []
    for c in constraints_in:
        if not _is_strict(c):
            constraints_pass1.append(c)
            continue
        fam = c.get("constraint_family")
        if fam in {"exclusion", "exclusive"}:
            applied_to = c.get("applied_to", "b")
            group_list = c.get("set") or c.get("group") or c.get("assets")
            if group_list is None:
                raise ValueError(f"'{fam}' requires a 'set' (list of tickers).")
            group_norm = [_norm_ticker(t) for t in group_list if str(t).strip() != ""]
            group_norm = list(dict.fromkeys(group_norm))
            if len(group_norm) < 2:
                raise ValueError(f"'{fam}' requires at least 2 tickers in 'set'.")
            c2 = {
                "constraint_family": "cardinality",
                "applied_to": applied_to,
                "group": group_norm,
                "is_strict": True,
                "_rewritten_from": fam,
            }
            if fam == "exclusion":
                c2["bound_type"] = "max"
                c2["value"] = 1.0
            else:
                c2["bound_type"] = "eq"
                c2["value"] = 1.0
            constraints_pass1.append(c2)
        else:
            constraints_pass1.append(c)

    # --- Pass 2: cardinality group[tickers] -> attribute+targets + df_local col ---
    constraints = constraints_pass1
    rewritten_constraints = []
    group_counter = 0
    for c in constraints:
        if not _is_strict(c):
            rewritten_constraints.append(c)
            continue
        if c.get("constraint_family") == "cardinality" and c.get("group") is not None:
            if "Ticker" not in df_local.columns:
                raise ValueError("Cardinality with 'group' requires df_data to have a 'Ticker' column.")
            applied_to = c.get("applied_to", "b")
            group_list = c.get("group") or []
            group_norm = [_norm_ticker(t) for t in group_list if str(t).strip() != ""]
            group_norm = list(dict.fromkeys(group_norm))
            if len(group_norm) == 0:
                raise ValueError("Cardinality constraint has empty 'group' list.")
            ticker_series_norm = df_local["Ticker"].astype(str).map(_norm_ticker)
            universe = set(ticker_series_norm.tolist())
            missing = [t for t in group_norm if t not in universe]
            if missing:
                raise ValueError(f"Cardinality 'group' contains unknown tickers: {missing[:10]}")
            col_name = f"__grp_{group_counter}"
            group_counter += 1
            group_set = set(group_norm)
            df_local[col_name] = np.where(ticker_series_norm.isin(group_set), "IN", "OUT")
            c2 = copy.deepcopy(c)
            c2.pop("group", None)
            c2["applied_to"] = applied_to
            c2["attribute"] = col_name
            c2["targets"] = ["IN"]
            rewritten_constraints.append(c2)
        else:
            rewritten_constraints.append(c)
    constraints = rewritten_constraints

    # --- Pass 3: inject epsilon min_ticket ---
    def _is_cardinality_eq_on_b(c):
        if not _is_strict(c): return False
        if (c.get("constraint_family") or "") != "cardinality": return False
        if (c.get("applied_to") or "") != "b": return False
        if (c.get("bound_type") or "").lower() != "eq": return False
        attr = (c.get("attribute") or "").lower()
        return attr in ["sum_all", "sum", "somme", "total"]

    def _is_min_ticket(c):
        return _is_strict(c) and (str(c.get("constraint_family") or "").lower().strip() == "min_ticket")

    card_eq_constraints = [c for c in constraints if _is_cardinality_eq_on_b(c)]
    has_min_ticket = any(_is_min_ticket(c) for c in constraints)
    if card_eq_constraints and not has_min_ticket:
        K = card_eq_constraints[0].get("value", None)
        K_int = int(round(float(K)))
        if K_int <= 0:
            raise ValueError(f"Cardinality eq constraint has invalid value: {K}")
        constraints.append({
            "applied_to": "b",
            "constraint_family": "min_ticket",
            "min_value": 1.0,
            "is_strict": True,
            "_injected": True,
        })

    # --- Pass 4 (NEW): exclude foreign cash from NAV ---
    # OPTIMISATION PERF : SHALLOW COPY
    if matrix_inputs is None:
        mi_local = None
    else:
        mi_local = dict(matrix_inputs)
        # Recréation profonde uniquement de la matrice modifiable
        if "Variance" in mi_local:
            mi_local["Variance"] = np.array(mi_local["Variance"]).tolist()

    cashflow_cfg = cfg.get("cashflow") or {}
    base_ccy = str(cashflow_cfg.get("base_currency", "")).upper().strip()
    use_foreign_cash = bool(cashflow_cfg.get("use_foreign_cash", True))
    exclude_foreign_cash = bool(cashflow_cfg.get("exclude_foreign_cash_from_nav", False))
    
    if exclude_foreign_cash:
        if not base_ccy:
            raise ValueError("exclude_foreign_cash_from_nav requires cashflow.base_currency to be set.")
        itype = df_local["InstrumentType"].astype(str).str.upper().to_numpy()
        ccy = df_local["Currency"].astype(str).str.upper().to_numpy()
        is_cash = (itype == "CASH")
        is_non_base = (ccy != base_ccy)
        is_foreign_cash = is_cash & is_non_base
        is_foreign_non_cash = (~is_cash) & is_non_base
        keep_mask = ~(is_foreign_cash | is_foreign_non_cash)
        kept_idx = np.where(keep_mask)[0].astype(int).tolist()
        if len(kept_idx) == 0:
            raise ValueError("All rows removed by exclude_foreign_cash_from_nav. Check data/base_currency.")
        df_local = df_local.iloc[kept_idx].reset_index(drop=True)

        if mi_local is not None:
            n0 = int(len(keep_mask))
            for key in ("w0", "Benchmark"):
                if key in mi_local:
                    v = np.asarray(mi_local[key], dtype=float).reshape(-1)
                    v2 = v[kept_idx]
                    v2 = _normalise_weights(v2)
                    mi_local[key] = v2.tolist()
            if "Variance" in mi_local:
                V = np.asarray(mi_local["Variance"], dtype=float)
                V2 = _slice_square(V, kept_idx)
                mi_local["Variance"] = V2.tolist()

        cfg.setdefault("_universe_filter", {})
        cfg["_universe_filter"]["excluded_foreign_cash"] = True
        cfg["_universe_filter"]["kept_original_indices"] = kept_idx

    # --- Pass 5: Liquidity Level by Currency (%) ---
    min_liq = cashflow_cfg.get("min_liquidity_by_ccy") or {}
    if isinstance(min_liq, dict) and len(min_liq) > 0:
        for ccy_k, lvl in min_liq.items():
            ccy_u = str(ccy_k).upper().strip()
            try:
                lvl_f = float(lvl)
            except Exception:
                continue
            if lvl_f <= 0: continue
            if (not use_foreign_cash) and (ccy_u != base_ccy): continue
            ticker = f"CASH_{ccy_u}"
            if not (df_local["Ticker"].astype(str).str.upper() == ticker).any(): continue
            constraints.append({
                "is_strict": True,
                "constraint_family": "standard",
                "applied_to": "x",
                "attribute": "Ticker",
                "targets": [ticker],
                "bound_type": "min",
                "value": lvl_f,
                "_injected_from": "min_liquidity_by_ccy",
                "_liquidity_ccy": ccy_u,
            })
        cfg.setdefault("liquidity", {})
        cfg["liquidity"]["min_by_ccy"] = {str(k).upper(): float(v) for k, v in min_liq.items() if float(v) > 0}

    cfg["constraints"] = constraints
    if mi_local is None:
        return cfg, df_local
    return cfg, df_local, mi_local