# ! AUTOMATICALLY GENERATED CODE FOR CUSADI

import jax

nnz_in = [1]
nnz_out = [1]
n_w = 1

@jax.jit
def evaluate_f(outputs, inputs, work):
        work = work.at[0].set(inputs[0][0])
        work = work.at[0].set(work[0] * work[0])
        outputs = outputs.at[0][0].set(work[0])