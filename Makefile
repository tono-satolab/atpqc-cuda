## Please edit according to your environment
CXX := g++-9
CUDAPATH := /usr/local/cuda-11.2
CUDA_GENCODE_FLAG := -arch=compute_75 -code=sm_75

## You don't need to edit the following

NVCC := $(CUDAPATH)/bin/nvcc
CUDAINCLUDEDIR := $(CUDAPATH)/include
CUDALIBDIR := $(CUDAPATH)/lib64

NVCCFLAGS := -std=c++17 --compiler-bindir=$(CXX) $(CUDA_GENCODE_FLAG) -rdc=true \
-Werror=all-warnings -lcuda -lnvToolsExt $(NVCC_DEFINES) \
-O3
# -g -O0 -Xcompiler -rdynamic -lineinfo

SRCDIR := ./src
BLDDIR := ./target

FIPS202_WS_DIR := $(SRCDIR)/lib/fips202_ws
FIPS202_WS_SOURCE = $(wildcard $(FIPS202_WS_DIR)/*.cu)
VERIFY_CMOV_WS_DIR := $(SRCDIR)/lib/verify_cmov_ws
VERIFY_CMOV_WS_SOURCE = $(wildcard $(VERIFY_CMOV_WS_DIR)/*.cu)
KYBER_DIR := $(SRCDIR)/lib/kyber
KYBER_SOURCE = $(wildcard $(KYBER_DIR)/*.cu) $(wildcard $(KYBER_DIR)/*/*.cu)

SOURCE = $(FIPS202_WS_SOURCE) $(VERIFY_CMOV_WS_SOURCE) $(KYBER_SOURCE)

.PHONY: all clean test bench

test: test_kyber512 test_kyber768 test_kyber1024
bench: bench_kyber512 bench_kyber768 bench_kyber1024

test_kyber512: $(BLDDIR)/test_kyber512.out
test_kyber768: $(BLDDIR)/test_kyber768.out
test_kyber1024: $(BLDDIR)/test_kyber1024.out

bench_kyber512: $(BLDDIR)/bench_kyber512.out
bench_kyber768: $(BLDDIR)/bench_kyber768.out
bench_kyber1024: $(BLDDIR)/bench_kyber1024.out

$(BLDDIR)/test_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/test/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 -DRNG_KIND=zero -DCUDA_DEBUG $^ -o $@
$(BLDDIR)/test_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/test/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 -DRNG_KIND=zero -DCUDA_DEBUG $^ -o $@
$(BLDDIR)/test_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/test/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 -DRNG_KIND=zero -DCUDA_DEBUG $^ -o $@

$(BLDDIR)/bench_kyber512.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber512 $^ -o $@
$(BLDDIR)/bench_kyber768.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber768 $^ -o $@
$(BLDDIR)/bench_kyber1024.out: $(SOURCE) $(SRCDIR)/main/kyber/bench/main.cu
	$(NVCC) $(NVCCFLAGS) -DKYBER_VARIANT=kyber1024 $^ -o $@

clean:
	$(RM) -fv $(BLDDIR)/*.out
