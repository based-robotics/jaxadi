"""
This module is supposed to implement graph
creation, traversion, code-generation and
compression/fusion if necessary/possible
"""

from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function

from ._ops import OP_JAX_VALUE_DICT


def sort_by_height(graph: list[list[int]], antigraph: list[list[int]], heights: list[int]):
    """
    Sort graph nodes by their heights.

    :param graph: Adjacency vector
    :param antigraph: Adjacency vector of antigraph
    :param heights: Heights of nodes
    :return: List of nodes sorted by height
    """
    nodes = [[] for i in range(max(heights) + 1)]
    for i, h in enumerate(heights):
        nodes[h].append(i)

    return nodes


def codegen(
    graph: list[list[int]],
    antigraph: list[list[int]],
    heights: list[int],
    output_map: dict[int, tuple[int, int, int]],
    values: list[str],
) -> str:
    """
    Main codegeneration function.
    Given the order of the nodes
    with respect to their height
    in the computation graph,
    merge work nodes in the same.

    :param graph: Adjacency vector
    :param antigraph: Adjacency vector of antigraph
    :param heights: Heights of nodes
    :param output_map: Output info -> node: o_idx, row, col
    :param values: Values of the nodes
    :return: Merged layers
    """
    sorted_nodes = sort_by_height(graph, antigraph, heights)
    code = ""
    outputs = {}
    for layer in sorted_nodes:
        indices = []
        assignment = "["
        for node in layer:
            if len(graph[node]) == 0 and node not in output_map:
                continue
            if node in output_map:
                oo = output_map[node]
                if outputs.get(oo[0], None) is None:
                    outputs[oo[0]] = {"rows": [], "cols": [], "values": []}
                outputs[oo[0]]["rows"].append(oo[1])
                outputs[oo[0]]["cols"].append(oo[2])
                outputs[oo[0]]["values"].append(values[node])
            else:
                if len(assignment) > 1:
                    assignment += ", "
                assignment += values[node]
                indices += [node]
        if len(indices) == 0:
            continue
        assignment += "]"
        code += f"    work = work.at[jnp.array({indices})].set({assignment})\n"

    for k, v in outputs.items():
        code += f"    outputs[{k}] = outputs[{k}].at[({v['rows']}, {v['cols']})].set([{', '.join(v['values'])}])\n"

    return code


def compute_heights(func: Function, graph: list[list[int]], antigraph: list[list[int]]) -> list[int]:
    """
    Heights computation function
    based on the simple BFS.

    :param func: Reference casadi function
    :param graph: Adjacency vector
    :param antigraph: Adjacency vector of antigraph
    :return: Vertices heights
    """
    heights = [0 for _ in range(len(graph))]
    current_layer = set()
    next_layer = set()

    for i in range(func.n_instructions()):
        op = func.instruction_id(i)
        if op == OP_INPUT:
            current_layer.add(i)

    while current_layer:
        instr = current_layer.pop()
        for parent in antigraph[instr]:
            heights[instr] = max(heights[instr], heights[parent] + 1)
        for child in graph[instr]:
            next_layer.add(child)

        if not current_layer:
            current_layer, next_layer = next_layer, current_layer

    return heights


def create_graph(func: Function):
    """
    Create the computation graph
    of the given casadi function.

    :param func: Reference casadi function
    :return: Graph and its properties
    """
    N = func.n_instructions()
    graph = [[] for _ in range(N)]
    values = ["" for _ in range(N)]
    antigraph = [[] for _ in range(N)]
    output_map = {}
    workers = [0 for _ in range(func.sz_w())]

    for i in range(N):
        op = func.instruction_id(i)
        o_idx = func.instruction_output(i)
        i_idx = func.instruction_input(i)

        if op == OP_CONST:
            values[i] = "jnp.array([" + OP_JAX_VALUE_DICT[op].format(func.instruction_constant(i)) + "])"
            workers[o_idx[0]] = i
        elif op == OP_INPUT:
            values[i] = OP_JAX_VALUE_DICT[op].format(i_idx[0], i_idx[1])
            workers[o_idx[0]] = i
        elif op == OP_OUTPUT:
            rows, cols = func.size_out(o_idx[0])
            row_number = o_idx[1] % rows
            column_number = o_idx[1] // rows
            output_map[i] = (o_idx[0], row_number, column_number)
            values[i] = OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]])

            # Update the graph: add this output node as a child of its input (work node)
            parent = workers[i_idx[0]]
            graph[parent].append(i)
            antigraph[i].append(parent)
        elif op == OP_SQ:
            values[i] = OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]])
            graph[workers[i_idx[0]]].append(i)
            antigraph[i].append(workers[i_idx[0]])
            workers[o_idx[0]] = i
        elif OP_JAX_VALUE_DICT[op].count("}") == 2:
            w_id0 = workers[i_idx[0]]
            w_id1 = workers[i_idx[1]]
            graph[w_id0].append(i)
            graph[w_id1].append(i)
            antigraph[i].extend([w_id0, w_id1])
            values[i] = OP_JAX_VALUE_DICT[op].format(w_id0, w_id1)
            workers[o_idx[0]] = i
        elif OP_JAX_VALUE_DICT[op].count("}") == 1:
            graph[workers[i_idx[0]]].append(i)
            antigraph[i].append(workers[i_idx[0]])
            values[i] = OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]])
            workers[o_idx[0]] = i
        else:
            raise Exception("Unknown CasADi operation: " + str(op))

    return graph, antigraph, output_map, values


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
    graph, antigraph, output_map, values = create_graph(func)
    heights = compute_heights(func, graph, antigraph)

    code = ""
    if add_import:
        code += "import jax\nimport jax.numpy as jnp\n\n"
    if add_jit:
        code += "@jax.jit\n"
    code += f"def evaluate_{func.name()}(*args):\n"
    code += "    inputs = [jnp.expand_dims(jnp.ravel(jnp.array(arg).T), axis=-1) for arg in args]\n"
    code += f"    outputs = [jnp.zeros(out) for out in {[func.size_out(i) for i in range(func.n_out())]}]\n"
    code += f"    work = jnp.zeros(({func.n_instructions()}, 1))\n"
    code += codegen(graph, antigraph, heights, output_map, values)
    code += "    return outputs"

    return code
