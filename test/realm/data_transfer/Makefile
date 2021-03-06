# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif
# vpath stuff below doesn't like a trailing slash in LG_RT_DIR
override LG_RT_DIR := $(patsubst %/,%,$(LG_RT_DIR))

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE		:= 
# List all the application source files here
GEN_SRC		:= 				# .cc files
GEN_GPU_SRC	=	copy_kernels.cu			# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= -DCHECK_COPY 
CC_FLAGS	?=
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:=

CXX ?= g++

I_TERRA = -Xcompiler="-I../../../language/terra/release/include/terra -pthread" 
L_TERRA = -L../../../language/terra/release/lib -lterra -ldl

TERRA_STUFF = -Xcompiler="-L/projects/opt/centos7/cuda/9.1/lib64 -lcuda -lcudart"

# we're going to include runtime.mk to get variable settings, but then
#  do our own build steps
NO_BUILD_RULES=1
include $(LG_RT_DIR)/runtime.mk

# since we're just doing Realm and not Legion, we need to strip out a few
#  things that might have come in from CC_FLAGS that require Legion goo
override CC_FLAGS := $(filter-out -DBOUNDS_CHECKS, \
                     $(filter-out -DPRIVILEGE_CHECKS, \
                     $(filter-out -DLEGION_SPY, \
                       $(CC_FLAGS))))
TESTS := kernel_transpose

ifeq ($(strip $(USE_GASNET)),1)
  ifdef NODECOUNT
    ifeq ($(CONDUIT),ibv)
      LAUNCHER = mpirun -H localhost -n $(NODECOUNT) $(1)
    else
      $(error unsupported conduit $(CONDUIT))
    endif
  else
    LAUNCHER = $(1)
    TESTS += $(TESTS_SINGLENODE)
  endif
else
  LAUNCHER = $(1)
  TESTS += $(TESTS_SINGLENODE)
endif

# can set arguments to be passed to a test when running
TESTARGS_ctxswitch := -ll:io 1 -t 20 -i 10000
TESTARGS_proc_group := -ll:cpu 4

REALM_OBJS := $(patsubst %.cc,%.o,$(notdir $(REALM_SRC))) \
              $(patsubst %.S,%.o,$(notdir $(ASM_SRC)))
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
RUNTIME_VPATH := $(subst $(SPACE),:,$(sort $(dir $(REALM_SRC))))
vpath %.cc .:$(RUNTIME_VPATH)
vpath %.S .:$(RUNTIME_VPATH)
#VPATH = .:$(RUNTIME_VPATH)

REALM_LIB := librealm.a

# run_all : $(TESTS:%=run_%)
run_all : $(REALM_LIB) $(REALM_OBJS)

# This should definitely be improved
multi_kernel_transpose : multi_kernel_transpose.cc
	$(NVCC) -ccbin g++ -I../../Common -o $@_gpu64.ptx -ptx $@_gpu.cu
	$(NVCC) -ccbin g++ -I./Common $(INC_FLAGS) -I$(CUDA)/include -o $@.o -c $@.cc
	$(NVCC) -ccbin g++ -o $@ $@.o  -L$(CUDA)/lib64/stubs -lcuda -L. -lrealm $(INC_FLAGS) $(NVCC_FLAGS)
	
# This should definitely be improved
generated : kernel_transpose.cc
	$(NVCC) -ccbin g++ -I./Common $(INC_FLAGS) -I$(CUDA)/include -o $@.o -c $@.cc
	$(NVCC) -ccbin g++ -o $@ $@.o  -L$(CUDA)/lib64/stubs -lcuda -L. -lrealm $(INC_FLAGS) $(NVCC_FLAGS)
	
# This should definitely be improved
kernel_transpose : kernel_transpose.cc
	$(NVCC) -ccbin g++ -I../../Common  -o $@_gpu64.ptx -ptx $@_gpu.cu
	$(NVCC) -ccbin g++ -I./Common $(INC_FLAGS) -I$(CUDA)/include -o $@.o -c $@.cc
	$(NVCC) -ccbin g++   -o $@ $@.o  -L$(CUDA)/lib64/stubs -lcuda -L. -lrealm $(INC_FLAGS) $(NVCC_FLAGS)
	
dsl_ptx_kernel_transpose : dsl_ptx_kernel_transpose.cc
	$(NVCC) -ccbin g++ $(I_TERRA) -I./Common  $(INC_FLAGS) -I$(CUDA)/include  -o $@.o -c $@.cc
	$(NVCC) -ccbin g++  $(L_TERRA) -o $@ $@.o  -L$(CUDA)/lib64/stubs -lcuda -L. -lrealm $(INC_FLAGS) $(NVCC_FLAGS)
	
dsl_kernel_transpose : dsl_kernel_transpose.cc
	$(NVCC) -ccbin g++ -I./Common $(INC_FLAGS) -I$(CUDA)/include  -o $@.o -c $@.cc
	$(NVCC) -ccbin g++  -o $@ $@.o  -L$(CUDA)/lib64/stubs -lcuda -L. -lrealm $(INC_FLAGS) $(NVCC_FLAGS)
	
#-DUSE_DISK -DUSE_LIBDL -DUSE_CUDA -DUSE_ZLIB -DDEBUG_REALM -DDEBUG_LEGION -DCOMPILE_TIME_MIN_LEVEL=LEVEL_DEBUG  -lrt -lpthread

	

run_% : %
	@# this echos exactly once, even if -s was specified
	@echo $(call LAUNCHER, ./$*) $(TESTARGS_$*)
	@$(call LAUNCHER, ./$*) $(TESTARGS_$*)

build : $(TESTS)

clean :
	rm -f $(REALM_LIB) $(REALM_OBJS) $(TESTS)

#$(TESTS) : % : %.cc librealm.a
#	$(CXX) -o $@ $< $(EXTRAOBJS_$*) -L. -lrealm $(INC_FLAGS) $(CC_FLAGS) $(LEGION_LD_FLAGS)


$(REALM_LIB) : $(REALM_OBJS)
	rm -f $(REALM_LIB)
	ar rc $(REALM_LIB) $(REALM_OBJS)

ifeq ($(strip $(USE_CUDA)),1)
EXTRAOBJS_memspeed := memspeed_gpu.o
memspeed : memspeed_gpu.o

%.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)
endif

%.o : %.cc
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

%.o : %.S
	$(CXX) -fPIC -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
