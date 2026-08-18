import ast
import builtins

import numpy as np
import pandas as pd

from app.data.store import scored_df

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Constant,
    ast.Attribute, ast.Call, ast.keyword, ast.Subscript, ast.Slice, ast.Tuple,
    ast.List, ast.Dict, ast.Set, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Lambda, ast.arguments, ast.arg, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Starred, ast.JoinedStr,
    ast.FormattedValue, ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    ast.expr_context,
)

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
}

SAFE_BUILTINS = {name: getattr(builtins, name) for name in [
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
    "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
]}


def _reject_reason(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return f"{type(node).__name__} statements are not allowed in the sandbox"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"attribute access to {node.attr!r} is not allowed in the sandbox"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"{node.id!r} is not available in the sandbox"
    return None


def run_pandas(code: str) -> dict:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return {"error": f"could not parse code: {exc}"}

    reason = _reject_reason(tree)
    if reason:
        return {"error": reason}

    namespace = {"df": scored_df.copy(), "pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    try:
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            setup = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(setup, "<sandbox>", "exec"), namespace)
            result = eval(compile(ast.Expression(tree.body[-1].value), "<sandbox>", "eval"), namespace)
        else:
            exec(compile(tree, "<sandbox>", "exec"), namespace)
            result = namespace.get("result")
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {"result": repr(result)[:2000], "result_type": type(result).__name__}
