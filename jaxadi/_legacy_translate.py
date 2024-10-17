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
    Function,
)

OP_JAX_DICT = {
    OP_ASSIGN: "\n    work = work.at[{0}].set(work[{1}])",
    OP_ADD: "\n    work = work.at[{0}].set(work[{1}] + work[{2}])",
    OP_SUB: "\n    work = work.at[{0}].set(work[{1}] - work[{2}])",
    OP_MUL: "\n    work = work.at[{0}].set(work[{1}] * work[{2}])",
    OP_DIV: "\n    work = work.at[{0}].set(work[{1}] / work[{2}])",
    OP_NEG: "\n    work = work.at[{0}].set(-work[{1}])",
    OP_EXP: "\n    work = work.at[{0}].set(jnp.exp(work[{1}]))",
    OP_LOG: "\n    work = work.at[{0}].set(jnp.log(work[{1}]))",
    OP_POW: "\n    work = work.at[{0}].set(jnp.power(work[{1}], work[{2}]))",
    OP_CONSTPOW: "\n    work = work.at[{0}].set(jnp.power(work[{1}], work[{2}]))",
    OP_SQRT: "\n    work = work.at[{0}].set(jnp.sqrt(work[{1}]))",
    OP_SQ: "\n    work = work.at[{0}].set(work[{1}] * work[{2}])",
    OP_TWICE: "\n    work = work.at[{0}].set(2 * work[{1}])",
    OP_SIN: "\n    work = work.at[{0}].set(jnp.sin(work[{1}]))",
    OP_COS: "\n    work = work.at[{0}].set(jnp.cos(work[{1}]))",
    OP_TAN: "\n    work = work.at[{0}].set(jnp.tan(work[{1}]))",
    OP_ASIN: "\n    work = work.at[{0}].set(jnp.arcsin(work[{1}]))",
    OP_ACOS: "\n    work = work.at[{0}].set(jnp.arccos(work[{1}]))",
    OP_ATAN: "\n    work = work.at[{0}].set(jnp.arctan(work[{1}]))",
    OP_LT: "\n    work = work.at[{0}].set(work[{1}] < work[{2}])",
    OP_LE: "\n    work = work.at[{0}].set(work[{1}] <= work[{2}])",
    OP_EQ: "\n    work = work.at[{0}].set(work[{1}] == work[{2}])",
    OP_NE: "\n    work = work.at[{0}].set(work[{1}] != work[{2}])",
    OP_NOT: "\n    work = work.at[{0}].set(jnp.logical_not(work[{1}]))",
    OP_AND: "\n    work = work.at[{0}].set(jnp.logical_and(work[{1}], work[{2}]))",
    OP_OR: "\n    work = work.at[{0}].set(jnp.logical_or(work[{1}], work[{2}]))",
    OP_FLOOR: "\n    work = work.at[{0}].set(jnp.floor(work[{1}]))",
    OP_CEIL: "\n    work = work.at[{0}].set(jnp.ceil(work[{1}]))",
    OP_FMOD: "\n    work = work.at[{0}].set(jnp.fmod(work[{1}], work[{2}]))",
    OP_FABS: "\n    work = work.at[{0}].set(jnp.abs(work[{1}]))",
    OP_SIGN: "\n    work = work.at[{0}].set(jnp.sign(work[{1}]))",
    OP_COPYSIGN: "\n    work = work.at[{0}].set(jnp.copysign(work[{1}], work[{2}]))",
    OP_IF_ELSE_ZERO: "\n    work = work.at[{0}].set(jnp.where(work[{1}] == 0, 0, work[{2}]))",
    OP_ERF: "\n    work = work.at[{0}].set(jax.scipy.special.erf(work[{1}]))",
    OP_FMIN: "\n    work = work.at[{0}].set(jnp.minimum(work[{1}], work[{2}]))",
    OP_FMAX: "\n    work = work.at[{0}].set(jnp.maximum(work[{1}], work[{2}]))",
    OP_INV: "\n    work = work.at[{0}].set(1.0 / work[{1}])",
    OP_SINH: "\n    work = work.at[{0}].set(jnp.sinh(work[{1}]))",
    OP_COSH: "\n    work = work.at[{0}].set(jnp.cosh(work[{1}]))",
    OP_TANH: "\n    work = work.at[{0}].set(jnp.tanh(work[{1}]))",
    OP_ASINH: "\n    work = work.at[{0}].set(jnp.arcsinh(work[{1}]))",
    OP_ACOSH: "\n    work = work.at[{0}].set(jnp.arccosh(work[{1}]))",
    OP_ATANH: "\n    work = work.at[{0}].set(jnp.arctanh(work[{1}]))",
    OP_ATAN2: "\n    work = work.at[{0}].set(jnp.arctan2(work[{1}], work[{2}]))",
    OP_CONST: "\n    work = work.at[{0}].set({1:.16f})",
    OP_INPUT: "\n    work = work.at[{0}].set(inputs[{1}][{2}, {3}])",
    OP_OUTPUT: "\n    outputs[{0}] = outputs[{0}].at[{1}, {2}].set(work[{3}][0])",
}


def translate(func: Function, add_jit=False, add_import=False) -> str:
    # Get information about Casadi function
    n_instr = func.n_instructions()
    n_out = func.n_out()  # number of outputs in the function
    n_in = func.n_in()  # number of outputs in the function

    # Get the shapes of input and output
    out_shapes = [func.size_out(i) for i in range(n_out)]
    in_shapes = [func.size_in(i) for i in range(n_in)]

    # Number of work variables
    n_w = func.sz_w()

    input_idx = [func.instruction_input(i) for i in range(n_instr)]
    output_idx = [func.instruction_output(i) for i in range(n_instr)]
    operations = [func.instruction_id(i) for i in range(n_instr)]
    const_instr = [func.instruction_constant(i) for i in range(n_instr)]

    # Generate string with complete code
    codegen = ""
    if add_import:
        codegen += "import jax\nimport jax.numpy as jnp\n\n"
    codegen += "@jax.jit\n" if add_jit else ""
    codegen += f"def evaluate_{func.name()}(*args):\n"
    codegen += "    inputs = args\n"  # Combine all inputs into a single list
    # Output variables
    codegen += f"    outputs = [jnp.zeros(out) for out in {out_shapes}]\n"
    codegen += f"    work = jnp.zeros(({n_w}, 1))\n"  # Work variables

    for k in range(n_instr):
        op = operations[k]
        o_idx = output_idx[k]
        i_idx = input_idx[k]
        if op == OP_CONST:
            codegen += OP_JAX_DICT[op].format(o_idx[0], const_instr[k])
        elif op == OP_INPUT:
            this_shape = in_shapes[i_idx[0]]
            rows, cols = this_shape  # Get the shape of the output
            row_number = i_idx[1] % rows  # Compute row index for JAX
            column_number = i_idx[1] // rows  # Compute column index for JAX
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], row_number, column_number)
        elif op == OP_OUTPUT:
            # Fix for OP_OUTPUT
            rows, cols = out_shapes[o_idx[0]]  # Get the shape of the output matrix
            # Adjust the calculation to switch from CasADi's column-major to JAX's row-major
            row_number = o_idx[1] % rows  # Compute row index for JAX
            column_number = o_idx[1] // rows  # Compute column index for JAX
            codegen += OP_JAX_DICT[op].format(o_idx[0], row_number, column_number, i_idx[0])
        elif op == OP_SQ:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[0])
        elif OP_JAX_DICT[op].count("{") == 3:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[1])
        elif OP_JAX_DICT[op].count("{") == 2:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0])
        else:
            raise Exception("Unknown CasADi operation: " + str(op))

    # Footer
    codegen += "\n    return outputs\n"

    return codegen
