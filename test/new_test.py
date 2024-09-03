import jax.numpy as jnp
import numpy as np


def evaluate_sum1(*args):
    inputs = list(args)
    for i in range(len(inputs)):
        inputs[i] = inputs[i].flatten()
    outputs = [jnp.zeros(out) for out in [(1, 2)]]
    print(outputs[0].shape)
    work = jnp.zeros(2)

    work = work.at[0].set(inputs[0][0])
    work = work.at[1].set(inputs[0][1])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[0, 0].set(work[0])
    work = work.at[0].set(inputs[0][2])
    work = work.at[1].set(inputs[0][3])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[0, 1].set(work[0])
    return outputs

arg = np.random.randn(2, 2)
print(evaluate_sum1(arg))
