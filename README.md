# ATPQC-CUDA

Implementations and automatic parameter tuning framework for the 
[CRYSTALS-Kyber key encapsulation algorithm](https://pq-crystals.org/kyber/ "Kyber") on CUDA. 

We also plan to integrate other post-quantum key 
encapsulation algorithms on GPUs into our framework in the future.


## Requirements

- NVIDIA GPU (Compute Capability (CC) 3.x or later)
- CUDA Toolkit 11.2 or later
- Python3 with NumPy (Tested on Python 3.8 with NumPy 1.19.5)
- Tested OS: Ubuntu 20.04

## Quick Start

At first, edit Makefile to set host compiler, path to CUDA, and CC.

```makefile
Makefile
1 ## Please edit according to your environment
2 CXX := (Your host compiler supporting CUDA, ex. g++-9)
3 CUDAPATH := (Path-to-CUDA, ex. /usr/local/cuda-11.2)
4 CUDA_GENCODE_FLAG := -arch=compute_(CC of your GPU) -code=sm_(CC of your GPU) (ex. -arch=compute_75 -code=sm_75)
...
```

Then, execute `bench_kyber.sh` in cloned directory for measurement.

```sh
$ ./bench_kyber.sh
```

Finally, execute `param_search.py` using Python3 with NumPy.

```sh
$ python param_search.py
```

## Installation of dependencies

### CUDA Toolkit

Please refer to [CUDA Toolkit web page](https://developer.nvidia.com/cuda-toolkit "CUDA Toolkit | NVIDIA Developer").

### Python

Normally, you should be able to install python and numpy using the package manager on your distribution and pip.

```sh
ex. Ubuntu
$ apt install python3
$ pip3 install numpy
```

[pyenv](https://github.com/pyenv/pyenv "pyenv/pyenv: Simple Python version management")
or [poetry](https://python-poetry.org/ "Poetry - Python dependency management and packaging made easy.") may help.

## Directory Layout

- Kyber on GPU
  - src/lib/kyber/primitive/: Host functions of Kyber primitives
  - src/lib/fips202_ws/: SHA-3 function family using warp shuffle instruction
  - src/lib/kyber/arithmetic/: Kernels for Matrices/modules/polynomials arithmetic
  - src/lib/kyber/endecode_mt/: Kernels for encoding/decoding polynomials
  - src/lib/kyber/genpoly_warp/: Kernels for generating polynomials
  - src/lib/kyber/ntt_ctgs_64t/: Kernels for number theoretic transforms (NTT) using 64 threads
  - src/lib/kyber/ntt_ctgs_128t/: Kernels for NTT using 128 threads
  - src/lib/kyber/symmetric_ws/: Kernels for symmetric functions
  - src/lib/verify_cmov_ws/: Kernels for verification and copying ciphertext
  - src/lib/kyber/params.cuh: Parameters of Kyber
  - src/lib/kyber/reduce.cuh: Modular reduction
  - src/lib/kyber/variants.cuh: Classes for three variants of Kyber
  - src/lib/kyber/zetas_table.cu(h): Table of constants used by NTT
  - src/lib/rng/: Various RNGs
- Tuning framework
  - src/main/kyber/bench/main.cu: Measurement program
  - param_search.py: Measurement program
  - bench_kyber.sh: Executing measurement program for each parameter sets
  - param_search.py: Searching optimal parameters using evaluation functions defined by user

## Publications

More details can be found in our work accepted by ISCAS 2021 (full
paper also available at [here](https://eprint.iacr.org/2021/198 "Cryptology ePrint Archive: Report 2021/198 - Automatic Parallelism Tuning for Module Learning with Errors Based Post-Quantum Key Exchanges on GPUs")).

```bibtex
Automatic Parallelism Tuning for Module Learning with Errors Based Post-Quantum Key Exchanges on GPUs
Tatsuki Ono, Song Bian, and Takashi Sato
ISCAS 2021

@inproceedings{atpqccuda2021ono,
        title        = {Automatic Parallelism Tuning for Module Learning with Errors Based Post-Quantum Key Exchanges on {GPUs}},
        author       = {Ono, Tatsuki and Bian, Song and Sato, Takashi},
        booktitle    = {Proceedings of IEEE International Symposium on Circuits and Systems},
        month        = may,
        year         = 2021,
        address      = {Daegu, Korea},
}
```

## License

This software is released under the MIT License, see LICENSE.
