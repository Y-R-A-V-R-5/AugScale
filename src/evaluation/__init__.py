"""
src/evaluation/__init__.py
===========================
Public interface for Phase-2 evaluation pipeline.

This module exposes all core evaluation components required to:

1. Run robustness evaluation (ModelEvaluator)
2. Define evaluation scenarios (EvalCondition, build_eval_conditions)
3. Collect and structure metrics (MetricsCollector)
4. Compute final model ranking (compute_robustness_ranking)

Design Purpose
--------------
Centralized imports improve usability and keep experiment scripts clean:

    from src.evaluation import ModelEvaluator

instead of deep module imports like:

    from src.evaluation.evaluator import ModelEvaluator

This also enforces a stable public API boundary for the evaluation system.
"""

# ---------------------------------------------------------------------
# Evaluation Conditions
# ---------------------------------------------------------------------

from .conditions import (
    EvalCondition,
    build_eval_conditions,
)

# ---------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------

from .evaluator import ModelEvaluator

# ---------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------

from .metrics import MetricsCollector

# ---------------------------------------------------------------------
# Ranking / Scoring
# ---------------------------------------------------------------------

from .ranking import compute_robustness_ranking