# ! AUTOMATICALLY GENERATED CODE FOR CUSADI

import jax
import jax.numpy as jnp


@jax.jit
def evaluate_cf(*args):
    inputs = args
    outputs = [jnp.zeros(out) for out in [(3, 1)]]
    work = jnp.zeros((3, 1))

    work = work.at[0].set(inputs[0][0])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[1].set(inputs[1][0])
    work = work.at[1].set(work[1] * work[1])
    work = work.at[0].set(work[0] + work[1])
    work = work.at[2].set(1.0000000000000000)
    work = work.at[0].set(work[0] + work[2])
    outputs[0] = outputs[0].at[0].set(work[0])
    work = work.at[0].set(inputs[0][1])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[0].set(work[0] + work[1])
    work = work.at[0].set(work[0] + work[2])
    outputs[0] = outputs[0].at[1].set(work[0])
    work = work.at[0].set(inputs[0][2])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[0].set(work[0] + work[1])
    work = work.at[0].set(work[0] + work[2])
    outputs[0] = outputs[0].at[2].set(work[0])
    return outputs
