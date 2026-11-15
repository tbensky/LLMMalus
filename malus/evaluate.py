"""Check the law the model hands in against fresh measurements.

finish() asks for the law as a Python expression in A, B and C. Here it is
parsed with a whitelist (numbers, the knob names, arithmetic and functions
from the math module; nothing else), then compared with new readings taken
at random knob settings the model never chose. The readings come from the
apparatus, so the same check works on the real bench.
"""

from __future__ import annotations

import ast
import math
import random
import re
import statistics
from types import SimpleNamespace
from typing import Any, Callable

from .apparatus import Apparatus

_MATH_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_EXTRA_FUNCS = {"abs": abs, "min": min, "max": max}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.UAdd, ast.USub,
)
_LHS_RE = re.compile(r"^\s*[A-Za-z_]\w*\s*(\([^()]*\))?\s*=(?!=)")


class FormulaError(ValueError):
    pass


def normalize_formula(text: str) -> str:
    """Accept the usual ways a law gets written down.

    'V = ...' / 'V(A,B,C) = ...' lose the left-hand side, '^' becomes '**'
    (nobody means bitwise xor in a physics law), and a trailing period or a
    surrounding code fence is dropped.
    """
    expr = str(text or "")
    previous = None
    while expr != previous:          # peel fences, periods and spaces in any order
        previous = expr
        expr = expr.strip().strip("`").rstrip(".")
    if expr.lower().startswith("python"):
        expr = expr[6:].strip()
    expr = _LHS_RE.sub("", expr, count=1).strip()
    return expr.replace("^", "**")


class _FloatConstants(ast.NodeTransformer):
    """Make integer literals floats, so 10**10**10 overflows instead of hanging."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError(f"Only numbers are allowed as constants, not {node.value!r}.")
        return ast.copy_location(ast.Constant(float(node.value)), node)


def compile_formula(text: str, variables: tuple[str, ...] = ("A", "B", "C")) -> Callable[..., float]:
    expr = normalize_formula(text)
    if not expr:
        raise FormulaError("No formula was given.")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Not a valid expression: {exc.msg}") from None
    allowed_names = set(variables) | set(_MATH_NAMES) | set(_EXTRA_FUNCS) | {"math"}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(f"{type(node).__name__} is not allowed in a formula.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise FormulaError(f"Unknown name {node.id!r}; use A, B, C and math functions.")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"
                    and node.attr in _MATH_NAMES):
                raise FormulaError("Only math.<function> attribute access is allowed.")
        if isinstance(node, ast.Call) and node.keywords:
            raise FormulaError("Keyword arguments are not allowed in a formula.")
    tree = ast.fix_missing_locations(_FloatConstants().visit(tree))
    code = compile(tree, "<formula>", "eval")
    namespace: dict[str, Any] = {"__builtins__": {}, **_MATH_NAMES, **_EXTRA_FUNCS,
                                 "math": SimpleNamespace(**_MATH_NAMES)}

    def evaluate(*values: float) -> float:
        scope = dict(namespace)
        scope.update(zip(variables, (float(v) for v in values)))
        result = eval(code, scope)  # noqa: S307 - whitelisted AST, no builtins
        return float(result)

    return evaluate


def collect_holdout(apparatus: Apparatus, *, points: int = 40, samples: int = 3,
                    seed: int | None = None) -> list[dict[str, float]]:
    """Fresh readings at random knob settings the model never chose."""
    rng = random.Random(seed)
    rows = []
    for _ in range(points):
        settings = {}
        for spec in apparatus.knobs:
            low = spec.minimum if spec.minimum is not None else 0.0
            high = spec.maximum if spec.maximum is not None else 360.0
            settings[spec.name] = round(rng.uniform(low, high), 2)
            apparatus.set_knob(spec.name, settings[spec.name])
        measured = statistics.fmean(apparatus.read_signal() for _ in range(samples))
        rows.append({**settings, "measured": measured})
    return rows


def score_formula(formula: str, apparatus: Apparatus, *, points: int = 40,
                  samples: int = 3, seed: int | None = None,
                  threshold: float = 0.05,
                  holdout: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """Compare the formula with `points` fresh readings at random settings.

    The score is the normalized RMS error: RMS(predicted - measured) divided
    by the range of the measured voltages. `discovered` is nrmse <= threshold;
    the default 5% passes a correct law with fitted constants and fails a
    wrong functional form (cos instead of cos^2, a missing knob...).

    Pass `holdout` (from collect_holdout) to score several formulas against
    the same readings without measuring again.
    """
    knob_names = tuple(k.name for k in apparatus.knobs)
    result: dict[str, Any] = {"formula": formula, "normalized": normalize_formula(formula),
                              "threshold": threshold, "discovered": False}
    try:
        law = compile_formula(formula, knob_names)
    except FormulaError as exc:
        result["error"] = str(exc)
        return result
    if holdout is None:
        holdout = collect_holdout(apparatus, points=points, samples=samples, seed=seed)

    rows = []
    for reading in holdout:
        settings = {k: reading[k] for k in knob_names}
        try:
            predicted = law(*[settings[k] for k in knob_names])
            if not math.isfinite(predicted):
                raise ValueError("not a finite number")
        except (ArithmeticError, ValueError, TypeError) as exc:
            result["error"] = f"The formula failed at {settings}: {exc}"
            result["points"] = rows
            return result
        rows.append({**settings, "measured": reading["measured"], "predicted": predicted})

    measured = [r["measured"] for r in rows]
    residuals = [r["predicted"] - r["measured"] for r in rows]
    rmse = math.sqrt(statistics.fmean(e * e for e in residuals))
    spread = max(measured) - min(measured)
    mean = statistics.fmean(measured)
    ss_tot = sum((m - mean) ** 2 for m in measured)
    ss_res = sum(e * e for e in residuals)
    nrmse = rmse / spread if spread > 0 else math.inf
    result.update({
        "points_checked": len(rows),
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "discovered": nrmse <= threshold,
        "points": rows,
    })
    return result
