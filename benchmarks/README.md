# Benchmarking

In order to evaluate the performance of the `jaxadi` library vs `cusadi` we have tried to reproduce the benchmarks from the `cusadi` library first and faced some issues with the proper `cuda` installation.

![meme](https://preview.redd.it/explain-please-v0-ma2mz5wxftod1.jpeg?auto=webp&s=2b90dfa3b12e064f54333e1080b3dabbad914f48)

Adding the complexity of setup of benchmarks of [cusadi](https://github.com/se-hwan/cusadi) we have copied and modified the benchmarks to be able to run them in the [`colab` environment](https://colab.research.google.com/github/based-robotics/jaxadi/blob/feature%2Fbenchmarking/benchmarks/jaxadi_vs_cusadi.ipynb) side by side (CUDA vs Jax).

Due limitations we cover only the functions with less than 1e3 operations. All of them are located in the [data](data) directory.

One may run the benchmarks in the colab environment and get the [cusadi results](cuda_benchmark_results.npz) and [jaxadi results](jax_benchmark_results.npz) for comparison.

#### Overview

This analysis compares the performance of two computational tools, **Jaxadi** and **Cusadi**, using a range of batch sizes for different operations. The tools are evaluated based on average computation times and their relative speedup across various workloads.

![1e1](https://github.com/based-robotics/jaxadi/blob/master/docs/static/images/compare_1e1_bar.png?raw=true)

![1e2](https://github.com/based-robotics/jaxadi/blob/master/docs/static/images/compare_1e2_bar.png?raw=true)

![Speedup](https://github.com/based-robotics/jaxadi/blob/master/docs/static/images/speedup_ratio.png?raw=true)

#### Key Findings

- **Jaxadi's Consistent Advantage:** Across all tested scenarios, Jaxadi consistently shows better performance than Cusadi. This is evident from the lower average computation times for Jaxadi, particularly as the workload increases.

- **Scalability Differences:** As batch sizes grow, the gap between the two tools becomes more pronounced. Cusadi's computation time increases rapidly with larger batch sizes, especially beyond \(2^{12}\). In contrast, Jaxadi maintains a more stable performance with a much slower increase in computation time, indicating better batch scalability.

#### Conclusion

Overall, Jaxadi offers a clear performance advantage over Cusadi, particularly for small functions which can be translated propely.
