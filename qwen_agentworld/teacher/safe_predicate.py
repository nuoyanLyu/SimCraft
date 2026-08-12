"""Restricted-expression validator/evaluator for CheckerSpec.executable_predicate.

The research plan's checker constraint is "reads canonical_state only, never a
natural-language reference answer." Freeform Python `eval` can't enforce that
(and is a code-execution hazard besides), so a predicate must first parse as a
*pure expression* over a single `state` variable, using only a small allowlist
of AST node types — no imports, no calls to anything except a handful of safe
builtins, no lambdas. Anything outside that allowlist is rejected before it
ever reaches `eval`.

Comprehensions (`any(x for x in state[...])` etc.) are allowed since Claude
reaches for them naturally when writing "does some item in a list satisfy a
condition" checks; their loop variables are collected up front so plain
`ast.Name` lookups for them don't get rejected as unknown identifiers.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any

SAFE_CALL_NAMES = frozenset({"len", "abs", "min", "max", "sorted", "round", "any", "all"})
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_CALL_NAMES}

_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,  # x[a:b] indexing — no escape risk, same as plain Subscript
    ast.Index,  # py<3.9 compat no-op on 3.9+, harmless to keep
    ast.Load,
    ast.Store,  # only reachable via comprehension loop targets, e.g. `for tag in ...`
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.comprehension,
)


class UnsafePredicateError(ValueError):
    pass


def _comprehension_bound_names(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            target = node.target
            names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            bound.update(n.id for n in names if isinstance(n, ast.Name))
    return bound


def validate_predicate_ast(
    predicate: str, root_names: frozenset[str] = frozenset({"state"})
) -> ast.Expression:
    """Parse `predicate` and reject it unless it is a pure expression over the
    allowed root variable(s) (`state` for an end-state checker, `states` for a
    step-wise one) using only the allowlisted node types. Returns the parsed
    AST on success so callers (e.g. the no-NL-leak audit) can walk it further.
    """
    try:
        tree = ast.parse(predicate, mode="eval")
    except SyntaxError as exc:
        raise UnsafePredicateError(f"predicate is not a valid Python expression: {exc}") from exc

    bound_names = _comprehension_bound_names(tree)
    allowed_names = set(root_names) | SAFE_CALL_NAMES | bound_names
    if not any(isinstance(n, ast.Name) and n.id in root_names for n in ast.walk(tree)):
        # A predicate that never reads the state at all is a constant (e.g.
        # `1 == 1`, `True`) — a trivially-true checker that would pass every
        # trajectory. Reject it here so the checker audit surfaces it.
        raise UnsafePredicateError(
            f"predicate does not reference any of {sorted(root_names)} — it is a constant"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise UnsafePredicateError(f"unexpected identifier '{node.id}' — only {sorted(allowed_names)} are allowed")
            continue
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                # blocks dunder-chain sandbox escapes, e.g. ().__class__.__bases__
                raise UnsafePredicateError(f"attribute access to '{node.attr}' is not allowed")
            continue
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in SAFE_CALL_NAMES:
                    raise UnsafePredicateError(f"call to disallowed function '{func.id}'")
            elif isinstance(func, ast.Attribute):
                pass  # method call, e.g. state.get(...); the Attribute node itself is checked above
            else:
                raise UnsafePredicateError("only direct or attribute calls are allowed")
            continue
        if isinstance(node, _ALLOWED_EXPR_NODES):
            continue
        raise UnsafePredicateError(f"disallowed expression element: {type(node).__name__}")

    return tree


def evaluate_predicate(predicate: str, state: dict[str, Any]) -> bool:
    validate_predicate_ast(predicate)
    # `state` goes in globals, not locals: a comprehension body runs in its own
    # scope that sees globals but not eval()'s locals dict, so a predicate
    # referencing `state` inside `any(... for ... in ...)` would raise NameError.
    result = eval(  # noqa: S307 — validated to a restricted AST allowlist above
        compile(predicate, "<checker>", mode="eval"),
        {"__builtins__": {}} | _SAFE_BUILTINS | {"state": state},
    )
    return bool(result)


def evaluate_step_wise_predicate(predicate: str, states: list[dict[str, Any]]) -> bool:
    """Evaluate a step-wise checker over the ordered list of canonical states
    (initial state followed by the state after each executed step). Used for
    tasks whose final observable state can equal the initial state, where an
    end-state predicate cannot tell real work-then-revert from a no-op.
    """
    validate_predicate_ast(predicate, root_names=frozenset({"states"}))
    # `states` in globals for the same comprehension-scope reason as above.
    result = eval(  # noqa: S307 — validated to a restricted AST allowlist above
        compile(predicate, "<step-checker>", mode="eval"),
        {"__builtins__": {}} | _SAFE_BUILTINS | {"states": states},
    )
    return bool(result)


def is_trivial_tautology(predicate: str) -> bool:
    """Catch the `X == C or X != C` / `X != C or X == C` shape (and its `and`
    negation of a contradiction) that a lazy teacher emits to make a checker
    pass unconditionally — observed live on a create-then-delete task where the
    end state was unverifiable. Not a general tautology prover (undecidable in
    the limit); it targets the one degenerate form seen in practice, on top of
    the constant-predicate guard in `validate_predicate_ast`.
    """
    try:
        tree = ast.parse(predicate, mode="eval").body
    except SyntaxError:
        return False
    if not (isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.Or) and len(tree.values) == 2):
        return False
    left, right = tree.values
    if not (isinstance(left, ast.Compare) and isinstance(right, ast.Compare)):
        return False
    if len(left.ops) != 1 or len(right.ops) != 1:
        return False
    dumped = (ast.dump(left.left) == ast.dump(right.left)
              and ast.dump(left.comparators[0]) == ast.dump(right.comparators[0]))
    if not dumped:
        return False
    op_pair = {type(left.ops[0]), type(right.ops[0])}
    return op_pair == {ast.Eq, ast.NotEq} or op_pair == {ast.Lt, ast.GtE} or op_pair == {ast.Gt, ast.LtE}
