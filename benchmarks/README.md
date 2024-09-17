# Benchmarking

In order to evaluate the performance of the `jaxadi` library vs `cusadi` we have tried to reproduce the benchmarks from the `cusadi` library first and faced some issues with the proper `cuda` installation.

![meme](https://preview.redd.it/explain-please-v0-ma2mz5wxftod1.jpeg?auto=webp&s=2b90dfa3b12e064f54333e1080b3dabbad914f48)

Adding the complexity of setup of benchmarks of [cusadi](https://github.com/se-hwan/cusadi) we have copied and modified the benchmarks to be able to run them in the [`colab` environment](https://colab.research.google.com/github/based-robotics/jaxadi/blob/feature%2Fbenchmarking/benchmarks/jaxadi_vs_cusadi.ipynb) side by side (CUDA vs Jax).

Due limitations we cover only the functions with less than 1e3 operations. All of them are located in the [data](data) directory.

One may run the benchmarks in the colab environment and get the [cusadi results](cuda_benchmark_results.npz) and [jaxadi results](jax_benchmark_results.npz) for comparison.
