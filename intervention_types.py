from graphical_models import SoftInterventionalDistribution
from typing import Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum


class IvType(Enum):
    """ Type of intervention/causal mechanism change in different contexts.
    """
    PARAM_CHANGE = 1
    # noise interventions
    SCALE = 2
    SHIFT = 3
    # constant
    CONST = 4
    GAUSS = 5
    # hidden variables
    CONFOUNDING = 6
    HIDDEN_PARENT = 7

    def __str__(self):
        if self.value == 1:
            return "IvChange"
        if self.value == 2:
            return "IvScaling"
        if self.value == 3:
            return "IvShift"
        if self.value == 4:
            return "IvPerfect"
        else:
            return "IvOther"
