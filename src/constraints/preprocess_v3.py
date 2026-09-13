import copy
import numpy as np
import pandas as pd

def preprocess_config_and_df_v3(optimization_config: dict, df_data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Pre-treatment v3:
    A) exclusion/exclusive -> cardinality group
    B) cardinality group=[tickers] -> df column __grp_k IN/OUT + attribute/targets
    C) inject epsilon min_buy_in if strict sum(b)=k and no min_buy_in
    Returns: (cfg_preprocessed, df_local)
    """
    cfg = copy.deepcopy(optimization_config)
    constraints_in = list(cfg.get("constraints", []) or [])
    df_local = df_data.copy()

    def _norm_ticker(x) -> str:
        return str(x).strip().lower()

    def _is_strict(c) -> bool:
        return bool(c.get("is_strict", True))

    # Pass 1: rewrite exclusion/exclusive -> cardinality group form
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
                raise ValueError(f"{fam} requires a 'set' (list of tickers).")

            group_norm = [_norm_ticker(t) for t in group_list if str(t).strip() != ""]
            group_norm = list(dict.fromkeys(group_norm))
            if len(group_norm) < 2:
                raise ValueError(f"{fam} requires at least 2 tickers in 'set'.")

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

    # Pass 2: cardinality group=[tickers] -> attribute+targets + df_local col
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
                raise ValueError(f"Cardinality 'group' contains unknown tickers: {missing}")

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

    # Pass 3: inject epsilon min_buy_in if strict sum(b)=k and no min_buy_in
    def _is_cardinality_eq_on_b(c):
        if not _is_strict(c):
            return False
        if c.get("constraint_family", "") != "cardinality":
            return False
        if c.get("applied_to", "") != "b":
            return False
        if c.get("bound_type", "").lower() != "eq":
            return False
        attr = c.get("attribute") or c.get("sum_all")
        return attr in ["sum_all", "sum", "somme", "total"]

    def _is_min_buy_in(c):
        return _is_strict(c) and c.get("constraint_family", "") == "min_buy_in"

    card_eq_constraints = [c for c in constraints if _is_cardinality_eq_on_b(c)]
    has_min_buy_in = any(_is_min_buy_in(c) for c in constraints)

    selection_epsilon = 1e-6
    if card_eq_constraints and not has_min_buy_in:
        K = card_eq_constraints[0].get("value", None)
        K_int = int(round(float(K)))
        if K_int <= 0:
            raise ValueError(f"Cardinality eq constraint has invalid value: {K}")

        if K_int * selection_epsilon > 1.0 + 1e-12:
            raise ValueError(f"Infeasible: K*epsilon > 1.0")

        constraints.append({
            "applied_to": "x",
            "attribute": "min_buy_in",
            "constraint_family": "min_buy_in",
            "bound_type": "min",
            "min_value": float(selection_epsilon),
            "is_strict": True,
            "injected": True,
        })

    cfg["constraints"] = constraints
    return cfg, df_local