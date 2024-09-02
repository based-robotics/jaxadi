# from codegen import evaluate_f

# print(evaluate_f([2.0]))

import jax
import jax.numpy as jnp

nnz_in = [1]
nnz_out = [1]
n_w = 1


@jax.jit
def evaluate_f(inputs):
    # Initialize the buffer array
    work = jnp.zeros(1)

    # Apply operations in sequence
    work = (
        work.at[0]
        .set(inputs[0])  # Set initial value
        .at[0]
        .set(work[0] * work[0])  # Square the value
        # Additional arbitrary operations can be chained here
    )
    return work


result = evaluate_f([2.0])
print(result)  # Should print [4.0]
