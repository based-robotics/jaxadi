from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function
from ._stages import stage_generator, squeze_stages


def translate(func: Function, add_jit=False, add_import=False) -> str:
    stages = stage_generator(func)
    stages = squeze_stages(stages)
    # Get information about Casadi function
    n_out = func.n_out()  # number of outputs in the function

    # Get the shapes of input and output
    out_shapes = [func.size_out(i) for i in range(n_out)]
    n_w = func.sz_w()

    # Generate string with complete code
    codegen = ""
    if add_import:
        codegen += "import jax\nimport jax.numpy as jnp\n\n"
    codegen += "@jax.jit\n" if add_jit else ""
    codegen += f"def evaluate_{func.name()}(*args):\n"
    # Combine all inputs into a single list
    codegen += "    inputs = jnp.expand_dims(jnp.array(args), axis=-1)\n"
    # Output variables
    codegen += f"    outputs = [jnp.zeros(out) for out in {out_shapes}]\n"
    codegen += f"    work = jnp.zeros(({n_w}, 1))\n"  # Work variables

    for stage in stages:
        codegen += stage.codegen()

    # Footer
    codegen += "\n    return outputs\n"

    return codegen


def legacy_translate(func: Function, add_jit=False, add_import=False) -> str:
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
            # Get the shape of the output matrix
            rows, cols = out_shapes[o_idx[0]]
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
