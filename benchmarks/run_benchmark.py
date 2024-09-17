import os
import time

import jax
import numpy as np
from casadi import Function

from jaxadi import convert

# setup seed
key = jax.random.PRNGKey(0)
cur_dir = os.path.dirname(os.path.abspath(__file__))


# setup all the paths to run benchmarks
class Paths:
    FUNCTIONS_DIR = os.path.join(cur_dir, "data")
    OUTPUT_DIR = cur_dir
    RUN_CUSADI = False


class ColabPaths:
    FUNCTIONS_DIR = os.path.join(cur_dir, "src/benchmark_functions")
    OUTPUT_DIR = cur_dir
    RUN_CUSADI = True


# Choose the paths provider
PathsProvider = Paths if not os.environ.get("COLAB_GPU", False) else ColabPaths


# Run cusadi benchmark only if the environment variable is set.
# Installation procedure for cusadi is not included
# in the repository and might be challenging to set up.
# Note, that the directories are valid only for the colab setup. One might need to adjust them.
CUSADI_BENCHMARK_DIR = os.environ.get("CUSADI_BENCHMARK_DIR", "src/benchmark_functions")
CUSADI_DATA_DIR = os.environ.get("CUSADI_DATA_DIR", "data")

if PathsProvider.RUN_CUSADI:
    import torch
    from src import CusadiFunction, generateCMakeLists, generateCUDACodeDouble

    torch.manual_seed(0)

N_ENVS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # , 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_EVALS = 20

# Load functions for CUDA benchmarking
fn_files = ["fn_1e1.casadi", "fn_1e2.casadi", "fn_1e3.casadi"]  # , "fn_1e4.casadi", "fn_1e5.casadi"]
benchmark_fns = [Function.load(os.path.join(PathsProvider.FUNCTIONS_DIR, fn)) for fn in fn_files]


def run_cuda_benchmarks():
    def sample_cusadi_input(cs_fun: Function, envs) -> list[torch.Tensor]:
        in_shapes = [cs_fun.size_in(i) for i in range(cs_fun.n_in())]

        return [torch.rand(envs, *shape, device="cuda", dtype=torch.double) for shape in in_shapes]

    def run_cusadi_benchmark(fn, inputs):
        start_time = time.perf_counter()
        fn.evaluate(inputs)
        return time.perf_counter() - start_time

    # Generate CUDA code
    for fn in benchmark_fns:
        generateCUDACodeDouble(fn)

    generateCMakeLists(benchmark_fns)
    os.system("mkdir -p build && cd build && cmake .. && make -j")

    results = {fn.name(): np.zeros((len(N_ENVS_SWEEP), N_EVALS)) for fn in benchmark_fns}
    results["N_ENVS"] = N_ENVS_SWEEP
    results["N_EVALS"] = N_EVALS

    for fn in benchmark_fns:
        fn_name = fn.name()
        for i, n_envs in enumerate(N_ENVS_SWEEP):
            print(f"Running CUDA benchmark for {n_envs} environments with function {fn_name}...")

            inputs = sample_cusadi_input(fn, n_envs)

            fn_cusadi = CusadiFunction(fn, n_envs)
            for j in range(N_EVALS):
                results[fn_name][i, j] = run_cusadi_benchmark(fn_cusadi, inputs)

    return results


def run_jaxadi_benchmarks():
    def sample_jaxadi_input(cs_fun: Function, envs) -> list[jax.Array]:
        in_shapes = [cs_fun.size_in(i) for i in range(cs_fun.n_in())]

        return [jax.random.uniform(key, (envs, *shape)) for shape in in_shapes]

    def run_jaxadi_benchmark(fn, inputs):
        start_time = time.perf_counter()
        fn(*inputs)
        return time.perf_counter() - start_time

    results = {fn.name(): np.zeros((len(N_ENVS_SWEEP), N_EVALS)) for fn in benchmark_fns}
    results["N_ENVS"] = N_ENVS_SWEEP
    results["N_EVALS"] = N_EVALS

    for fn in benchmark_fns:
        fn_name = fn.name()

        # apply jaxadi
        jax_fn = convert(fn, compile=True)
        vmapped_fn = jax.vmap(jax_fn)

        for i, n_envs in enumerate(N_ENVS_SWEEP):
            print(f"Running Jaxadi benchmark for {n_envs} environments with function {fn_name}...")
            inputs = sample_jaxadi_input(fn, n_envs)

            # warmup
            vmapped_fn(*inputs)

            for j in range(N_EVALS):
                results[fn_name][i, j] = run_jaxadi_benchmark(vmapped_fn, inputs)

            # remove the compiled function from the memory and inputs
            del inputs

    return results


def main():
    if PathsProvider.RUN_CUSADI:
        cuda_results = run_cuda_benchmarks()
        np.savez(f"{cur_dir}/cuda_benchmark_results.npz", **cuda_results)

    jaxadi_results = run_jaxadi_benchmarks()
    np.savez(f"{cur_dir}/jaxadi_benchmark_results.npz", **jaxadi_results)

    print("Benchmark results saved.")


if __name__ == "__main__":
    main()
