from typing import Callable, Any


def declare(f: str) -> Callable[..., Any]:
    """
    Return local scope function
    based on string definition
    """
    local_vars = {}
    exec(f, {}, local_vars)
    func_name = next(iter(local_vars))
    return local_vars[func_name]
