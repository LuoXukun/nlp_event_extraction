import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from event_extraction_codes.models.baseline_model import BaselineModel
from event_extraction_codes.models.hierarchical_model import HierarchicalModel

ModelDict = {
    "baseline": BaselineModel,
    "baseline-lstm": BaselineModel,
    "hierarchical": HierarchicalModel,
    "hierarchical-bias": HierarchicalModel
}