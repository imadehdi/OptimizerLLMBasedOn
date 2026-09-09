from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class ProblemType(str, Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    UNKNOWN = "unknown"

class ObjectiveConfig(BaseModel):
    type: Literal[
        "linear", 
        "quadratic", 
        "ratio", 
        "minimax", 
        "mad", 
        "turnover", 
        "risk_budgeting", 
        "cvar", 
        "pareto_frontier",
        "tracking_error"   
    ] = Field(..., description="Type of objective function")
    target_name: Optional[str] = None
    secondary_target_name: Optional[str] = None
    variable_name: str = "x"
    direction: Literal["min", "max"] = "min"
    
    target_1: Optional[Dict[str, Any]] = Field(default=None, description="Premier objectif (ex: Maximiser le rendement)")
    target_2: Optional[Dict[str, Any]] = Field(default=None, description="Deuxième objectif (ex: Minimiser le risque)")
    points: int = Field(default=20, description="Nombre de portefeuilles à générer sur la frontière")
    highlight_metric: Optional[str] = Field(default="", description="The specific metric to highlight on a Pareto frontier.")

class DecisionVariable(BaseModel):
    name: str = Field(..., description="Nom symbolique de la variable (ex: 'x' ou 'b')")
    size: Literal["n_rows", "scalar"] = Field(..., description="Taille de la variable")
    type: Literal["continuous", "binary", "integer"] = Field(default="continuous", description="Domaine mathématique de la variable")

class GenericConstraint(BaseModel):
    applied_to: str = Field(..., description="Nom de la variable de décision impactée (ex: 'x' ou 'b')")
    attribute: str = Field(..., description="Nom exact de la colonne du tableau, ou 'sum_all', ou 'element'")
    constraint_family: Literal["standard", "cardinality", "logic", "min_buy_in"] = Field(default="standard", description="Famille algorithmique de la contrainte")
    targets: List[str] = Field(default_factory=list, description="Valeurs textuelles ciblées si colonne catégorielle.")
    bound_type: Literal["eq", "max", "min", "range"]
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class OptimizationConfig(BaseModel):
    decision_variables: List[DecisionVariable]
    objective: ObjectiveConfig
    constraints: List[GenericConstraint]

class FormulatedOptimizationProblem(BaseModel):
    problem_type: ProblemType
    title: str
    user_intent_summary: str
    optimization_config: OptimizationConfig