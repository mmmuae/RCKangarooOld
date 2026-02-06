CC := g++

# Respect CUDA_HOME/CUDA_PATH from the environment but default to /usr/local/cuda
DEFAULT_CUDA_PATH := $(if $(CUDA_HOME),$(CUDA_HOME),/usr/local/cuda)
CUDA_PATH ?= $(DEFAULT_CUDA_PATH)

# nvcc can also be overridden from the environment.  When not explicitly set we
# fall back to $(CUDA_PATH)/bin/nvcc and finally to whatever nvcc is on PATH.
NVCC ?= $(CUDA_PATH)/bin/nvcc
ifeq (,$(wildcard $(NVCC)))
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_PATH := $(dir $(dir $(NVCC)))
endif
ifeq (,$(NVCC))
$(error "nvcc not found. Set CUDA_PATH or NVCC to a valid CUDA installation")
endif

CCFLAGS := -O3 -I$(CUDA_PATH)/include

# CUDA 13.0+ only supports newer architectures (Ampere and later)
# RTX 5090: sm_120 (Blackwell), RTX 4090: sm_89 (Ada), H100: sm_90 (Hopper), RTX 3090: sm_86 (Ampere)
NVCCFLAGS := -O3 \
    -gencode=arch=compute_120,code=sm_120 \
	-gencode=arch=compute_90,code=sm_90 \
    -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_86,code=sm_86

LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS)




