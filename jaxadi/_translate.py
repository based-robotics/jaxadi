from casadi import Function
from ._stages import stage_generator, squeeze


def translate(func: Function, add_jit=False, add_import=False) -> str:
    stages = stage_generator(func)
    stages = squeeze(stages)
    # get information about casadi function
    n_out = func.n_out()  # number of outputs in the function

    # get the shapes of input and output
    out_shapes = [func.size_out(i) for i in range(n_out)]

    # generate string with complete code
    codegen = ""
    if add_import:
        codegen += "import jax\nimport jax.numpy as jnp\n\n"
    codegen += "@jax.jit\n" if add_jit else ""
    codegen += f"def evaluate_{func.name()}(*args):\n"
    # combine all inputs into a single list
    codegen += "    inputs = jnp.expand_dims(jnp.array(args), axis=-1)\n"
    # output variables
    codegen += f"    outputs = [jnp.zeros(out) for out in {out_shapes}]\n"

    # for stage in stages:
    #     codegen += stage.codegen()
    codegen += stages

    # footer
    codegen += "\n    return outputs\n"

    return codegen
