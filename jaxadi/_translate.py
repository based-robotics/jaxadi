import textwrap

from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function

from ._ops import OP_JAX_DICT


def translate(func: Function) -> list[str]:
    # Get information about Casadi function
    n_instr = func.n_instructions()
    n_out = func.n_out()  # number of outputs in the function

    # get the shapes of input and output
    out_shapes = [func.size_out(i) for i in range(n_out)]

    # number of work variables
    n_w = func.sz_w()

    input_idx = [func.instruction_input(i) for i in range(n_instr)]
    output_idx = [func.instruction_output(i) for i in range(n_instr)]
    operations = [func.instruction_id(i) for i in range(n_instr)]
    const_instr = [func.instruction_constant(i) for i in range(n_instr)]

    # generate string with complete code
    codegen = textwrap.dedent(
        """
    import jax
    import jax.numpy as jnp


    """
    )
    codegen += "@jax.jit\n"
    codegen += f"def evaluate_{func.name()}(*args):\n"
    codegen += "    inputs = args\n"  # combine all inputs into a single list
    codegen += f"    outputs = [jnp.zeros(out) for out in {out_shapes}]\n"  # output variables
    codegen += f"    work = jnp.zeros(({n_w}, 1))\n"  # work variables

    for k in range(n_instr):
        op = operations[k]
        o_idx = output_idx[k]
        i_idx = input_idx[k]
        if op == OP_CONST:
            codegen += OP_JAX_DICT[op].format(o_idx[0], const_instr[k])
        elif op == OP_INPUT:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[1])
        elif op == OP_OUTPUT:
            codegen += OP_JAX_DICT[op].format(o_idx[0], o_idx[1], i_idx[0])
        elif op == OP_SQ:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[0])
        elif OP_JAX_DICT[op].count("{") == 3:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[1])
        elif OP_JAX_DICT[op].count("{") == 2:
            codegen += OP_JAX_DICT[op].format(o_idx[0], i_idx[0])
        else:
            raise Exception("Unknown CasADi operation: " + str(op))

    # footer
    codegen += "\n    return outputs\n"

    return codegen.split("\n")
