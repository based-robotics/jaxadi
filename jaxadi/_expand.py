"""
This module implements legacy translation
function, which automatically expands all
expressions into a single line of code.
While this approach proved to be very
efficient during execution, most often
resulting string cannot be processed by
python due to it's size.

Consider avoiding this translation method
unless you know your number of operations
or expansions is relatively low.
"""

import re
from typing import Any

from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function

from ._ops import OP_JAX_EXPAND_VALUE_DICT as OP_JAX_VALUE_DICT


class Stage:
    """
    A class storing information of the operations
    which are performed at a single stage. Initial
    idea revolves around the fact that multiple
    operations might be fused together and performed
    simultaneously.
    """

    def __init__(self):
        self.output_idx: list[int] = []
        self.work_idx: list[int] = []
        self.ops: list[Operation] = []

    def codegen(self) -> str:
        """
        Generate the code string for the stage.
        :return: Codestring
        """
        if self.ops[0].op == OP_OUTPUT:
            return self.ops[0].codegen()

        values = "["
        outputs = "jnp.array(["
        for op in self.ops:
            if values[-1] != "[":
                values += ", "
            if outputs[-1] != "[":
                outputs += ", "
            values += f"{op.value}"
            outputs += str(op.output_idx)
        values += "]"
        outputs += "])"

        return f"\n    work = work.at[{outputs}].set({values})"


class Operation:
    """
    A class, representing a single computation
    operation in computation graph, storing
    necessary information about RHS and indices.
    """

    def __init__(self):
        self.op: int = None
        self.value: str = ""
        self.work_idx: list[int] = []
        self.output_idx: Any = None

    def codegen(self):
        """
        Generate the code string for the operation.
        :return: Codestring
        """
        return f"\n    work = work.at[{self.output_idx}].set({self.value})"


class OutputOperation(Operation):
    """
    Special kind of Operation, which represents
    output stage and stores indices information.
    """

    def __init__(self):
        self.exact_idx1: Any = None
        self.exact_idx2: Any = None

        super().__init__()

    def codegen(self):
        """
        Generate the code string for the operation.
        :return: Codestring
        """
        return f"\n    outputs[{self.output_idx}] = outputs[{self.output_idx}].at[{self.exact_idx1}, {self.exact_idx2}].set({self.value})"


def stage_generator(func: Function) -> list[Stage]:
    """
    Generate stages of the computation.

    :param func: Reference casadi function
    :return: List of stages
    """
    n_instr = func.n_instructions()
    n_out = func.n_out()
    n_in = func.n_in()
    n_w = func.sz_w()

    workers = [""] * n_w

    # Get the shapes of input and output
    out_shapes = [func.size_out(i) for i in range(n_out)]
    in_shapes = [func.size_in(i) for i in range(n_in)]

    input_idx = [func.instruction_input(i) for i in range(n_instr)]
    output_idx = [func.instruction_output(i) for i in range(n_instr)]
    operations = [func.instruction_id(i) for i in range(n_instr)]
    const_instr = [func.instruction_constant(i) for i in range(n_instr)]

    stages = []
    for k in range(n_instr):
        op = operations[k]
        o_idx = output_idx[k]
        i_idx = input_idx[k]
        operation = Operation()
        operation.op = op
        if op == OP_CONST:
            workers[o_idx[0]] = "jnp.array([" + OP_JAX_VALUE_DICT[op].format(const_instr[k]) + "])"

        elif op == OP_INPUT:
            workers[o_idx[0]] = OP_JAX_VALUE_DICT[op].format(i_idx[0], i_idx[1])
        elif op == OP_OUTPUT:
            operation = OutputOperation()
            operation.op = op
            rows, cols = out_shapes[o_idx[0]]
            row_number = o_idx[1] % rows
            column_number = o_idx[1] // rows
            operation.exact_idx1 = row_number
            operation.exact_idx2 = column_number
            operation.output_idx = o_idx[0]
            operation.work_idx.append(i_idx[0])
            operation.value = OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]])
            stage = Stage()
            stage.output_idx.append(operation.output_idx)
            stage.work_idx.extend(operation.work_idx)
            stage.ops.append(operation)
            stages.append(stage)
        elif op == OP_SQ:
            workers[o_idx[0]] = "(" + OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]]) + ")"
        elif OP_JAX_VALUE_DICT[op].count("}") == 2:
            workers[o_idx[0]] = "(" + OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]], workers[i_idx[1]]) + ")"
        elif OP_JAX_VALUE_DICT[op].count("}") == 1:
            workers[o_idx[0]] = OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]])
        else:
            raise Exception("Unknown CasADi operation: " + str(op))

    return stages


def combine_outputs(stages: list[Stage]) -> str:
    """
    Combine all adjacent output operations
    into a single output operation over
    a several input/output indices

    :param stages: List of stages
    :return: A codegen string with merged outputs.
    """
    output_groups: dict[int, list[Operation]] = {}

    for stage in stages:
        for op in stage.ops:
            if op.op == OP_OUTPUT:
                if op.output_idx not in output_groups:
                    output_groups[op.output_idx] = []
                output_groups[op.output_idx].append(op)

    commands = []
    for output_idx, ops in output_groups.items():
        row_indices = []
        column_indices = []
        values = []
        for op in ops:
            row_indices.append(f"{op.exact_idx1}")
            column_indices.append(f"{op.exact_idx2}")
            values.append(op.value)

        rows = "[" + ", ".join(row_indices) + "]"
        columns = "[" + ", ".join(column_indices) + "]"
        values_str = ", ".join(values)
        command = f"    outputs[{output_idx}] = outputs[{output_idx}].at[({rows}, {columns})].set([{values_str}])"
        commands.append(command)
    combined_command = "\n".join(commands)
    return combined_command


def recursive_subs(stages: list[Stage], idx: int) -> str:
    """
    Implementation of recursive substitution.
    Given index of the current stage, iterate
    over all previous stages expanding the value
    of the RHS.

    :param stages: List of stages
    :param idx: Current stage index
    :return: A string with RHS
    """
    current_value = stages[idx].ops[0].value
    work_pattern = r"work\[(\d+)\]"
    matches = re.findall(work_pattern, current_value)
    if not matches:
        return f"({current_value})"

    result = current_value
    for match in set(matches):
        number = int(match)
        for i in range(idx - 1, -1, -1):
            if stages[i].ops[0].output_idx == number and stages[i].ops[0].op != OP_OUTPUT:
                stages[i].ops[0].value = recursive_subs(stages, i)
                result = result.replace(f"work[{number}]", stages[i].ops[0].value)
                break

    return f"({result})"


def squeeze(stages: list[Stage]) -> str:
    """
    Recursively evaluate the RHS of the
    assignment to substitute it with the
    result expressed with respect to
    scalar inputs and constants.
    Finally, all outputs are fused together.

    :param stages: List of stages
    :return: A single line of code, which represents fused stages
    """
    new_stages = []
    working_stages = []
    for i, stage in enumerate(stages):
        if len(stage.ops) != 0 and stage.ops[0].op == OP_OUTPUT:
            working_stages.append((i, stage))
    for i in range(len(working_stages)):
        i, stage = working_stages[i]
        stage.value = recursive_subs(stages, i)
        new_stages.append(stage)

    cmd = combine_outputs(new_stages)
    return cmd


def translate(func: Function, add_jit: bool = False, add_import: bool = False) -> str:
    """
    Generate the string with jax
    equivalent of the given casadi
    function with some optionals.

    :param func: Reference casadi function
    :param add_jit: Add `@jit` to source str
    :param add_import: Add `import jax.numpy as jnp` to source str
    :return: Jax equivalent code string
    """
    stages = stage_generator(func)
    stages = squeeze(stages)
    n_out = func.n_out()

    out_shapes = [func.size_out(i) for i in range(n_out)]

    codegen = ""
    if add_import:
        codegen += "import jax\nimport jax.numpy as jnp\n\n"
    codegen += "@jax.jit\n" if add_jit else ""
    codegen += f"def evaluate_{func.name()}(*args):\n"
    # combine all inputs into a single list
    codegen += "    inputs = [jnp.expand_dims(jnp.ravel(jnp.array(arg).T), axis=-1) for arg in args]\n"
    # output variables
    codegen += f"    outputs = [jnp.zeros(out) for out in {out_shapes}]\n"

    # for stage in stages:
    #     codegen += stage.codegen()
    codegen += stages

    # footer
    codegen += "\n    return outputs\n"

    return codegen
