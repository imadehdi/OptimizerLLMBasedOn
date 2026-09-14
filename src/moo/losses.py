import numpy as np
import pandas as pd
from src.moo.losses import bound_loss, normalise_loss


def safe_scale(cstr: dict) -> float:
    if cstr.get("scale") is not None:
        try:
            v = float(cstr["scale"])
            return max(abs(v), 1e-12)
        except Exception:
            return 1.0
    return 1.0


def compile_measure_spec_w_only(cstr: dict, df_data: pd.DataFrame, col_map_case: str) -> dict:
    applied_to = (cstr.get("applied_to") or "x").strip().lower()
    if applied_to not in ("x", "w"):
        return None

    attr = (cstr.get("attribute") or "").strip().lower()
    if attr in ("sum_all", "sum", "somme", "total"):
        return {"kind": "sum_all"}

    if attr not in col_map_case:
        return None

    real_col = col_map_case[attr]
    col = df_data[real_col]

    targets = cstr.get("targets", None)
    if targets is not None and len(targets) > 0:
        targets_norm = [str(t).lower() for t in targets]
        mask = col.astype(str).str.lower().isin(targets_norm).values
        return {"kind": "mask_sum", "mask": mask}

    if pd.api.types.is_numeric_dtype(col):
        expo = col.values.astype(float)
        return {"kind": "dot", "expo": expo}

    return None


def measure_vectorised_w_only(spec: dict, W: np.ndarray) -> np.ndarray:
    kind = spec["kind"]
    if kind == "sum_all":
        return np.sum(W, axis=-1)
    if kind == "mask_sum":
        mask = spec["mask"]
        return np.sum(W[:, mask], axis=-1)
    if kind == "dot":
        expo = spec["expo"]
        return W @ expo
    return np.zeros(W.shape[0], dtype=float)


def bound_loss_vectorised(values: np.ndarray, cstr: dict) -> np.ndarray:
    b_type = (cstr.get("bound_type") or "").strip().lower()

    if b_type == "eq":
        target = float(cstr.get("value", 0.0))
        return np.abs(values - target)
    if b_type == "max":
        ub = float(cstr.get("value", 0.0))
        return np.maximum(0.0, values - ub)
    if b_type == "min":
        lb = float(cstr.get("value", 0.0))
        return np.maximum(0.0, lb - values)
    if b_type == "range":
        lb = float(cstr.get("min_value", -np.inf))
        ub = float(cstr.get("max_value", np.inf))
        return np.maximum(0.0, lb - values) + np.maximum(0.0, values - ub)
    return np.zeros_like(values)


def bound_loss(value: float, cstr: dict) -> float:
    b_type = (cstr.get("bound_type") or "").strip().lower()
    if b_type == "eq":
        target = float(cstr.get("value", 0.0))
        return np.abs(value - target)
    if b_type == "max":
        ub = float(cstr.get("value", 0.0))
        return np.maximum(0.0, value - ub)
    if b_type == "min":
        lb = float(cstr.get("value", 0.0))
        return np.maximum(0.0, lb - value)
    if b_type == "range":
        lb = float(cstr.get("min_value", -np.inf))
        ub = float(cstr.get("max_value", np.inf))
        return np.maximum(0.0, lb - value) + np.maximum(0.0, value - ub)
    return 0.0


def infer_scale(cstr: dict) -> float:
    if cstr.get("scale") is not None:
        try:
            s = float(cstr["scale"])
            return max(abs(s), 1e-12)
        except Exception:
            pass

    fam = (cstr.get("constraint_family") or "").lower().strip()
    b_type = (cstr.get("bound_type") or "").lower().strip()

    if fam == "cardinality":
        if b_type in ("eq", "min", "max"):
            try:
                return max(abs(float(cstr.get("value", 1.0))), 1.0)
            except Exception:
                return 1.0
        if b_type == "range":
            try:
                lb = float(cstr.get("min_value", 0.0))
                ub = float(cstr.get("max_value", 1.0))
                return max(abs(ub - lb), 1.0)
            except Exception:
                return 1.0

    if b_type in ("min", "max", "eq"):
        try:
            v = float(cstr.get("value", 1.0))
            return max(abs(v), 1.0) if abs(v) > 1e-12 else 1.0
        except Exception:
            return 1.0

    if b_type == "range":
        try:
            lb = float(cstr.get("min_value", 0.0))
            ub = float(cstr.get("max_value", 1.0))
            width = ub - lb
            return max(abs(width), 1e-12) if abs(width) > 1e-12 else 1.0
        except Exception:
            return 1.0

    return 1.0


def normalise_loss(raw_loss: float, cstr: dict) -> float:
    scale = infer_scale(cstr)
    return float(raw_loss) / float(scale)


def linear_raw_loss(z: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    Az = A @ z
    loss_low = np.maximum(0.0, lb - Az)
    loss_up = np.maximum(0.0, Az - ub)
    return float(np.sum(loss_low + loss_up))