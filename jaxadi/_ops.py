from casadi import (
    OP_ACOS,
    OP_ACOSH,
    OP_ADD,
    OP_AND,
    OP_ASIN,
    OP_ASINH,
    OP_ASSIGN,
    OP_ATAN,
    OP_ATAN2,
    OP_ATANH,
    OP_CEIL,
    OP_CONST,
    OP_CONSTPOW,
    OP_COPYSIGN,
    OP_COS,
    OP_COSH,
    OP_DIV,
    OP_EQ,
    OP_ERF,
    OP_EXP,
    OP_FABS,
    OP_FLOOR,
    OP_FMAX,
    OP_FMIN,
    OP_FMOD,
    OP_IF_ELSE_ZERO,
    OP_INPUT,
    OP_INV,
    OP_LE,
    OP_LOG,
    OP_LT,
    OP_MUL,
    OP_NE,
    OP_NEG,
    OP_NOT,
    OP_OR,
    OP_OUTPUT,
    OP_POW,
    OP_SIGN,
    OP_SIN,
    OP_SINH,
    OP_SQ,
    OP_SQRT,
    OP_SUB,
    OP_TAN,
    OP_TANH,
    OP_TWICE,
)

OP_JAX_VALUE_DICT = {
    OP_ASSIGN: "work[{0}]",
    OP_ADD: "work[{0}] + work[{1}]",
    OP_SUB: "work[{0}] - work[{1}]",
    OP_MUL: "work[{0}] * work[{1}]",
    OP_DIV: "work[{0}] / work[{1}]",
    OP_NEG: "-work[{0}]",
    OP_EXP: "jnp.exp(work[{0}])",
    OP_LOG: "jnp.log(work[{0}])",
    OP_POW: "jnp.power(work[{0}], work[{1}])",
    OP_CONSTPOW: "jnp.power(work[{0}], work[{1}])",
    OP_SQRT: "jnp.sqrt(work[{0}])",
    OP_SQ: "work[{0}] * work[{0}]",
    OP_TWICE: "2 * work[{0}]",
    OP_SIN: "jnp.sin(work[{0}])",
    OP_COS: "jnp.cos(work[{0}])",
    OP_TAN: "jnp.tan(work[{0}])",
    OP_ASIN: "jnp.arcsin(work[{0}])",
    OP_ACOS: "jnp.arccos(work[{0}])",
    OP_ATAN: "jnp.arctan(work[{0}])",
    OP_LT: "work[{0}] < work[{1}]",
    OP_LE: "work[{0}] <= work[{1}]",
    OP_EQ: "work[{0}] == work[{1}]",
    OP_NE: "work[{0}] != work[{1}]",
    OP_NOT: "jnp.logical_not(work[{0}])",
    OP_AND: "jnp.logical_and(work[{0}], work[{1}])",
    OP_OR: "jnp.logical_or(work[{0}], work[{1}])",
    OP_FLOOR: "jnp.floor(work[{0}])",
    OP_CEIL: "jnp.ceil(work[{0}])",
    OP_FMOD: "jnp.fmod(work[{0}], work[{1}])",
    OP_FABS: "jnp.abs(work[{0}])",
    OP_SIGN: "jnp.sign(work[{0}])",
    OP_COPYSIGN: "jnp.copysign(work[{0}], work[{1}])",
    OP_IF_ELSE_ZERO: "jnp.where(work[{0}] == 0, 0, work[{1}])",
    OP_ERF: "jax.scipy.special.erf(work[{0}])",
    OP_FMIN: "jnp.minimum(work[{0}], work[{1}])",
    OP_FMAX: "jnp.maximum(work[{0}], work[{1}])",
    OP_INV: "1.0 / work[{0}]",
    OP_SINH: "jnp.sinh(work[{0}])",
    OP_COSH: "jnp.cosh(work[{0}])",
    OP_TANH: "jnp.tanh(work[{0}])",
    OP_ASINH: "jnp.arcsinh(work[{0}])",
    OP_ACOSH: "jnp.arccosh(work[{0}])",
    OP_ATANH: "jnp.arctanh(work[{0}])",
    OP_ATAN2: "jnp.arctan2(work[{0}], work[{1}])",
    OP_CONST: "{0:.16f}",
    OP_INPUT: "inputs[{0}][{1}]",
    OP_OUTPUT: "work[{0}][0]",
}
OP_JAX_EXPAND_VALUE_DICT = {
    OP_ASSIGN: "{0}",
    OP_ADD: "{0} + {1}",
    OP_SUB: "{0} - {1}",
    OP_MUL: "{0} * {1}",
    OP_DIV: "{0} / {1}",
    OP_NEG: "-{0}",
    OP_EXP: "jnp.exp({0})",
    OP_LOG: "jnp.log({0})",
    OP_POW: "jnp.power({0}, {1})",
    OP_CONSTPOW: "jnp.power({0}, {1})",
    OP_SQRT: "jnp.sqrt({0})",
    OP_SQ: "{0} * {0}",
    OP_TWICE: "2 * {0}",
    OP_SIN: "jnp.sin({0})",
    OP_COS: "jnp.cos({0})",
    OP_TAN: "jnp.tan({0})",
    OP_ASIN: "jnp.arcsin({0})",
    OP_ACOS: "jnp.arccos({0})",
    OP_ATAN: "jnp.arctan({0})",
    OP_LT: "{0} < {1}",
    OP_LE: "{0} <= {1}",
    OP_EQ: "{0} == {1}",
    OP_NE: "{0} != {1}",
    OP_NOT: "jnp.logical_not({0})",
    OP_AND: "jnp.logical_and({0}, {1})",
    OP_OR: "jnp.logical_or({0}, {1})",
    OP_FLOOR: "jnp.floor({0})",
    OP_CEIL: "jnp.ceil({0})",
    OP_FMOD: "jnp.fmod({0}, {1})",
    OP_FABS: "jnp.abs({0})",
    OP_SIGN: "jnp.sign({0})",
    OP_COPYSIGN: "jnp.copysign({0}, {1})",
    OP_IF_ELSE_ZERO: "jnp.where({0} == 0, 0, {1})",
    OP_ERF: "jax.scipy.special.erf({0})",
    OP_FMIN: "jnp.minimum({0}, {1})",
    OP_FMAX: "jnp.maximum({0}, {1})",
    OP_INV: "1.0 / {0}",
    OP_SINH: "jnp.sinh({0})",
    OP_COSH: "jnp.cosh({0})",
    OP_TANH: "jnp.tanh({0})",
    OP_ASINH: "jnp.arcsinh({0})",
    OP_ACOSH: "jnp.arccosh({0})",
    OP_ATANH: "jnp.arctanh({0})",
    OP_ATAN2: "jnp.arctan2({0}, {1})",
    OP_CONST: "{0:.16f}",
    OP_INPUT: "inputs[{0}][{1}]",
    OP_OUTPUT: "{0}[0]",
}
