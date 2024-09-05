from typing import List, Any
from ._ops import OP_JAX_DICT, OP_JAX_VALUE_DICT
from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function


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
            operation.output_idx = o_idx[0]
            operation.value = "jnp.array([" + OP_JAX_VALUE_DICT[op].format(const_instr[k]) + "])"
            # codegen += OP_JAX_DICT[op].format(o_idx[0], const_instr[k])
        elif op == OP_INPUT:
            this_shape = in_shapes[i_idx[0]]
            rows, cols = this_shape  # Get the shape of the output
            row_number = i_idx[1] % rows  # Compute row index for JAX
            column_number = i_idx[1] // rows  # Compute column index for JAX
            operation.output_idx = o_idx[0]
            operation.value = OP_JAX_VALUE_DICT[op].format(i_idx[0], row_number, column_number)
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
            operation.value = OP_JAX_VALUE_DICT[op].format(i_idx[0])
        elif op == OP_SQ:
            operation.output_idx = o_idx[0]
            operation.work_idx.append(i_idx[0])
            operation.value = OP_JAX_VALUE_DICT[op].format(i_idx[0])
        elif OP_JAX_DICT[op].count("}") == 3:
            operation.output_idx = o_idx[0]
            operation.work_idx.extend([i_idx[0], i_idx[1]])
            operation.value = OP_JAX_VALUE_DICT[op].format(i_idx[0], i_idx[1])
        elif OP_JAX_DICT[op].count("}") == 2:
            operation.output_idx = o_idx[0]
            operation.work_idx.append(i_idx[0])
            operation.value = OP_JAX_VALUE_DICT[op].format(i_idx[0])
        else:
            raise Exception("Unknown CasADi operation: " + str(op))

        stage = Stage()
        stage.output_idx.append(operation.output_idx)
        stage.work_idx.extend(operation.work_idx)
        stage.ops.append(operation)
        stages.append(stage)

    return stages


def update_stage(stage: Stage):
    stage.output_idx = []
    stage.work_idx = []

    for op in stage.ops:
        stage.output_idx.append(op.output_idx)
        stage.work_idx.extend(op.work_idx)


def squeze_stages(stages: List[Stage]) -> List[Stage]:
    for i, stage in enumerate(stages):
        stage_ops = []
        for j, op in enumerate(stage.ops):
            current_stage = stage
            for k, new_stage in enumerate(reversed(stages[0:i])):
                if op.op == OP_OUTPUT:
                    break
                if len(new_stage.ops) == 0:
                    continue
                if new_stage.ops[0].op == OP_OUTPUT:
                    if op.output_idx in new_stage.output_idx:
                        break
                    elif op.output_idx in new_stage.work_idx:
                        break
                    else:
                        continue
                if op.output_idx in new_stage.output_idx:
                    break
                if set(op.work_idx).intersection(set(new_stage.output_idx)):
                    break
                if op.output_idx in new_stage.work_idx:
                    current_stage = new_stage
                    break
                current_stage = new_stage
            if current_stage == stage:
                stage_ops.append(op)
            else:
                current_stage.ops.append(op)

            update_stage(current_stage)
        stage.ops = stage_ops
        update_stage(stage)

    new_stages = []
    for stage in stages:
        if len(stage.ops) != 0:
            new_stages.append(stage)
    return new_stages
