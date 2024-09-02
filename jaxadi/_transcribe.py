import textwrap

from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function

from ._ops import OP_JAX_DICT


def transcribe(func: Function) -> list[str]:
    codegen_strings = {}

    # Get information about Casadi function
    n_instr = func.n_instructions()
    n_in = func.n_in()
    n_out = func.n_out()
    nnz_in = [func.nnz_in(i) for i in range(n_in)]
    nnz_out = [func.nnz_out(i) for i in range(n_out)]
    n_w = func.sz_w()

    INSTR_LIMIT = n_instr
    input_idx = [func.instruction_input(i) for i in range(INSTR_LIMIT)]
    output_idx = [func.instruction_output(i) for i in range(INSTR_LIMIT)]
    operations = [func.instruction_id(i) for i in range(INSTR_LIMIT)]
    const_instr = [func.instruction_constant(i) for i in range(INSTR_LIMIT)]

    # * Codegen for const declarations and indices
    codegen_strings["header"] = "# ! AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    codegen_strings["includes"] = textwrap.dedent(
        """
    import jax

    """
    )
    codegen_strings["nnz_in"] = f"nnz_in = [{','.join(map(str, nnz_in))}]\n"
    codegen_strings["nnz_out"] = f"nnz_out = [{','.join(map(str, nnz_out))}]\n"
    codegen_strings["n_w"] = f"n_w = {n_w}\n\n"

    # Codegen for jax function
    str_operations = "@jax.jit\n"
    str_operations += f"def evaluate_{func.name()}(outputs, inputs, work):"

    for k in range(INSTR_LIMIT):
        op = operations[k]
        o_idx = output_idx[k]
        i_idx = input_idx[k]
        if op == OP_CONST:
            str_operations += OP_JAX_DICT[op].format(o_idx[0], const_instr[k])
        elif op == OP_INPUT:
            str_operations += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[1])
        elif op == OP_OUTPUT:
            str_operations += OP_JAX_DICT[op].format(o_idx[0], o_idx[1], i_idx[0])
        elif op == OP_SQ:
            str_operations += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[0])
        elif OP_JAX_DICT[op].count("{}") == 3:
            str_operations += OP_JAX_DICT[op].format(o_idx[0], i_idx[0], i_idx[1])
        elif OP_JAX_DICT[op].count("{}") == 2:
            str_operations += OP_JAX_DICT[op].format(o_idx[0], i_idx[0])
        else:
            raise Exception("Unknown CasADi operation: " + str(op))

    codegen_strings["pytorch_operations"] = str_operations

    # * Write codegen to file
    return list(codegen_strings.values())
