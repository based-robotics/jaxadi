"""
This module is supposed to implement graph
creation, traversion, code-generation and
compression/fusion if necessary/possible
"""

import random
import re
from collections import deque

from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, Function

from ._ops import OP_JAX_VALUE_DICT


def test_and_compress(s):
    # Step 1: Check if the string has the desired form using a regex
    pattern = re.compile(r"\[\s*work\[(\d+)\]\s*\*\s*work\[(\d+)\](?:\s*,\s*work\[(\d+)\]\s*\*\s*work\[(\d+)\])*\s*\]")

    if not pattern.fullmatch(s.strip()):
        return s

    # Step 2.1: Extract the indices from the matches
    matches = re.findall(r"work\[(\d+)\]\s*\*\s*work\[(\d+)\]", s)
    left_indices = [int(m[0]) for m in matches]
    right_indices = [int(m[1]) for m in matches]

    # Construct the compressed string
    compressed_string = f"jnp.multiply(work[jnp.array({left_indices})], work[jnp.array({right_indices})])"
    return compressed_string


def sort_by_height(graph, antigraph, heights):
    nodes = [[] for i in range(max(heights) + 1)]
    for i, h in enumerate(heights):
        nodes[h].append(i)

    return nodes


def codegen(graph, antigraph, heights, output_map, values):
    sorted_nodes = sort_by_height(graph, antigraph, heights)
    code = ""
    outputs = {}
    for layer in sorted_nodes:
        indices = []
        assignment = "["
        for node in layer:
            if len(graph[node]) == 0 and not node in output_map:
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
        # assignment = test_and_compress(assignment)
        code += f"    work = work.at[jnp.array({indices})].set({assignment})\n"

    for k, v in outputs.items():
        code += f"    outputs[{k}] = outputs[{k}].at[({v['rows']}, {v['cols']})].set([{', '.join(v['values'])}])\n"

    return code


def compute_heights(func, graph, antigraph):
    heights = [0 for _ in range(len(graph))]
    current_layer = set()
    next_layer = set()
    # queue = deque()

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
    N = func.n_instructions()
    graph = [[] for _ in range(N)]
    values = [None for _ in range(N)]
    antigraph = [[] for _ in range(N)]
    output_map = {}
    workers = [None for _ in range(func.sz_w())]

    for i in range(N):
        op = func.instruction_id(i)
        o_idx = func.instruction_output(i)
        i_idx = func.instruction_input(i)

        if op == OP_CONST:
            values[i] = "jnp.array([" + OP_JAX_VALUE_DICT[op].format(func.instruction_constant(i)) + "])"
            workers[o_idx[0]] = i
        elif op == OP_INPUT:
            this_shape = func.size_in(i_idx[0])
            rows, cols = this_shape  # Get the shape of the output
            row_number = i_idx[1] % rows  # Compute row index for JAX
            column_number = i_idx[1] // rows  # Compute column index for JAX

            values[i] = OP_JAX_VALUE_DICT[op].format(i_idx[0], row_number, column_number)
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
            # rows, cols = func.size_out(o_idx[0])
            # row_number = o_idx[1] % rows  # Compute row index for JAX
            # column_number = o_idx[1] // rows  # Compute column index for JAX
            # output_map[i] = (o_idx[0], row_number, column_number)
            # values[i] = OP_JAX_VALUE_DICT[op].format(workers[i_idx[0]])
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


def expand_graph(func, graph, antigraph, output_map, values):
    heights = compute_heights(func, graph, antigraph)
    sorted_nodes = sort_by_height(graph, antigraph, heights)

    # Calculate the average number of vertices per layer
    total_vertices = sum(len(layer) for layer in sorted_nodes)
    avg_vertices = total_vertices / len(sorted_nodes)

    new_graph = [[] for _ in range(len(graph))]
    new_antigraph = [[] for _ in range(len(antigraph))]

    # Iterate over layers
    for layer in sorted_nodes:
        expand_layer = len(layer) < avg_vertices and not any(node in output_map for node in layer)

        if expand_layer:
            # Expand nodes and update their values
            for node in layer:
                value_expr = values[node]
                expanded_expr = re.sub(r"work\[(\d+)\]", lambda m: f"({values[int(m.group(1))]})", value_expr)
                values[node] = expanded_expr

            # Recalculate dependencies for expanded nodes
            for node in layer:
                new_parents = set()
                for parent in antigraph[node]:
                    new_parents.update(new_antigraph[parent])  # Use updated parents

                # Update new_antigraph and new_graph accordingly
                new_antigraph[node] = list(new_parents)

                for new_parent in new_parents:
                    new_graph[new_parent].append(node)

                # Retain the original child relationships
                for child in graph[node]:
                    new_graph[node].append(child)
                    new_antigraph[child].append(node)
        else:
            # Maintain existing connections for nodes without expansion
            for node in layer:
                for parent in antigraph[node]:
                    new_graph[parent].append(node)
                    new_antigraph[node].append(parent)
                for child in graph[node]:
                    new_graph[node].append(child)
                    new_antigraph[child].append(node)

    return new_graph, new_antigraph, output_map, values


def translate(func: Function, add_jit=False, add_import=False):
    graph, antigraph, output_map, values = create_graph(func)
    # graph, antigraph, output_map, values = expand_graph(func, graph, antigraph, output_map, values)
    heights = compute_heights(func, graph, antigraph)

    code = ""
    if add_import:
        code += "import jax\nimport jax.numpy as jnp\n\n"
    if add_jit:
        code += "@jax.jit\n"
    code += f"def evaluate_{func.name()}(*args):\n"
    code += "    inputs = [jnp.expand_dims(jnp.array(arg), axis=-1) for arg in args]\n"
    code += f"    outputs = [jnp.zeros(out) for out in {[func.size_out(i) for i in range(func.n_out())]}]\n"
    code += f"    work = jnp.zeros(({func.n_instructions()}, 1))\n"
    code += codegen(graph, antigraph, heights, output_map, values)
    code += "    return outputs"

    return code
