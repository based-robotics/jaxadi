from casadi import densify as cs_densify
from casadi import Function


def densify(func: Function):
    _i = func.sx_in()
    _o = func(*_i)
    if not isinstance(_o, tuple):
        _o = [_o]
    _dense_o = []
    for i, o in enumerate(_o):
        _dense_o.append(cs_densify(o))
    _func = Function(func.name(), _i, _dense_o)
    return _func
