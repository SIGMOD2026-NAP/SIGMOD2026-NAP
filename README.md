# The Source Code for HT-NSW (Submitted to SIGMOD 2026 Round 4)
-----------------------------------------------------------------------------------------------------------------
## Introduction
This is a source code for the algorithm described in the paper **Enhancing Graph-based Approximate Maximum Inner Product Search via Norm-Adaptive Partitioning**. We call it as **ht** project.
**ht** project is written by **C++17** and can be complied by **g++9.5.0** or higher in **Linux**.

### Usage
```bash
./run_rt.sh
```

### Dataset Format

In our project, the format of the input file (such as `audio.data_new`, which is in `float` data type) is a binary file, which is organized as the following format:

>{Bytes of the data type (int)} {The size of the vectors (int)} {The dimension of the vectors (int)} {All of the binary vector, arranged in turn (float)}


For your application, you should also transform your dataset into this binary format, then rename it as `[datasetName].data_new` and put it in the directory `./datasets`.

A sample dataset `audio.data_new` has been put in the directory `./datasets`.
