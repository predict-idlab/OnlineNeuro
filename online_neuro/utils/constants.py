# online_neuro/utils/constants.py
from enum import Enum, auto

"""
Constants and Enumerations for the Online Neuro optimization framework.

This module defines key enumerations used throughout the system to classify
optimization problems and select corresponding Bayesian Optimization (BO) components.
These enumerations help standardize inputs and ensure type safety
when defining problem constraints and BO strategies.
"""


class ProblemType(Enum):
    MULTIOBJECTIVE = auto()
    CLASSIFICATION = auto()
    REGRESSION = auto()


class AcquisitionType(Enum):
    NEGATIVE_PREDICTIVE_MEAN = auto()
    PREDICTIVE_VARIANCE = auto()
