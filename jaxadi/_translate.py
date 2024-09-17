from casadi import Function
from ._stages import stage_generator, squeeze


def translate(func: Function, add_jit=False, add_import=False, num_threads=1) -> str:
    n_out = func.n_out()  # number of outputs in the function

    # get the shapes of input and output
    out_shapes = [func.size_out(i) for i in range(n_out)]
    stages = stage_generator(func)
    stages = squeeze(stages, num_threads=num_threads)
    # get information about casadi function

    # generate string with complete code
    codegen = ""
    if add_import:
        codegen += "import jax\nimport jax.numpy as jnp\n\n"
    codegen += "@jax.jit\n" if add_jit else ""
    codegen += f"def evaluate_{func.name()}(*args):\n"
    # combine all inputs into a single list
    codegen += "    inputs = [jnp.expand_dims(jnp.array(arg), axis=-1) for arg in args]\n"
    # output variables
    codegen += f"    o = [jnp.zeros(out) for out in {out_shapes}]\n"

    # for stage in stages:
    #     codegen += stage.codegen()
    codegen += stages

    # footer
    codegen += "\n    return o\n"

    return codegen
