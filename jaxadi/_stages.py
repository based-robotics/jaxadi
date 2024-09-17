from typing import List, Any, Dict
from ._ops import OP_JAX_VALUE_DICT
from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function
import re
from multiprocessing import Pool, cpu_count


class Stage:
    def __init__(self):
        self.output_idx: List[int] = []
        self.work_idx: List[int] = []
        self.ops: List[Operation] = []

    def codegen(self):
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
    def __init__(self):
        self.op: int = None
        self.value: str = ""
        self.work_idx: List[int] = []
        self.output_idx: Any = None

    def codegen(self):
        return f"\n    work = work.at[{self.output_idx}].set({self.value})"


class OutputOperation(Operation):
    def __init__(self):
        self.exact_idx1: Any = None
        self.exact_idx2: Any = None

        super().__init__()

    def codegen(self):
        return f"\n    outputs[{self.output_idx}] = outputs[{self.output_idx}].at[{self.exact_idx1}, {self.exact_idx2}].set({self.value})"


def stage_generator(func: Function) -> str:
    # Get information about Casadi function
    n_instr = func.n_instructions()
    n_out = func.n_out()  # number of outputs in the function
    n_in = func.n_in()  # number of outputs in the function
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
            this_shape = in_shapes[i_idx[0]]
            rows, cols = this_shape  # Get the shape of the output
            row_number = i_idx[1] % rows  # Compute row index for JAX
            column_number = i_idx[1] // rows  # Compute column index for JAX
            workers[o_idx[0]] = OP_JAX_VALUE_DICT[op].format(i_idx[0], row_number, column_number)
        elif op == OP_OUTPUT:
            operation = OutputOperation()
            operation.op = op
            rows, cols = out_shapes[o_idx[0]]
            row_number = o_idx[1] % rows  # Compute row index for JAX
            column_number = o_idx[1] // rows  # Compute column index for JAX
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


def combine_outputs(stages: List[Stage]) -> str:
    output_groups: Dict[int, List[Operation]] = {}

    # Collect all OP_OUTPUT operations and group them by output_idx
    for stage in stages:
        for op in stage.ops:
            if op.op == OP_OUTPUT:
                if op.output_idx not in output_groups:
                    output_groups[op.output_idx] = []
                output_groups[op.output_idx].append(op)

    # Create combined JAX-style commands for each output_idx
    commands = []
    for output_idx, ops in output_groups.items():
        # Prepare the list of index pairs and respective values
        row_indices = []
        column_indices = []
        values = []
        for op in ops:
            row_indices.append(f"{op.exact_idx1}")
            column_indices.append(f"{op.exact_idx2}")
            values.append(op.value)

        # Combine the pairs and values into the final command for each output_idx
        rows = "[" + ", ".join(row_indices) + "]"
        columns = "[" + ", ".join(column_indices) + "]"
        values_str = ", ".join(values)
        command = f"    o[{output_idx}] = o[{output_idx}].at[({rows}, {columns})].set([{values_str}])"
        commands.append(command)

    # Combine all the commands into a single string
    combined_command = "\n".join(commands)

    return combined_command


def recursive_subs(stages: List[Stage], idx: int) -> str:
    # Get the current stage's value
    current_value = stages[idx].ops[0].value
    work_pattern = r"work\[(\d+)\]"

    # Find all occurrences of work[<number>]
    matches = re.findall(work_pattern, current_value)

    # If there are no work mentions, return the value as is, wrapped in parentheses
    if not matches:
        return f"({current_value})"

    # For each match, find and replace all work mentions
    result = current_value
    # Use set to avoid multiple replacements of the same work[number]
    for match in set(matches):
        number = int(match)

        # Find the stage that has output_idx equal to the number
        for i in range(idx - 1, -1, -1):
            if stages[i].ops[0].output_idx == number and stages[i].ops[0].op != OP_OUTPUT:
                # Recursively replace the found work[<number>] with expanded value
                stages[i].ops[0].value = recursive_subs(stages, i)
                result = result.replace(f"work[{number}]", stages[i].ops[0].value)
                break

    return f"({result})"


def process_stage(args):
    stages, i = args
    if len(stages[i].ops) != 0:
        stages[i].ops[0].value = recursive_subs(stages, i)
        return stages[i]
    return None


def squeeze(stages: List[Stage], num_threads=1) -> List[Stage]:
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
