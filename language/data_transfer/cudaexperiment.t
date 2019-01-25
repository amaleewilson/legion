if not terralib.cudacompile then
	print("CUDA not enabled, not performing test...")
	return
end
if os.getenv("CI") then
	print("Running in CI environment without a GPU, not performing test...")
	return
end

local tid = cudalib.nvvm_read_ptx_sreg_tid_x--terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

foo = terra(result : &int, to_add : int)
    var t = tid()
    result[t] = t + to_add
end

terralib.includepath = terralib.includepath..";/usr/local/cuda/include"


local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

sync = terralib.externfunction("cudaThreadSynchronize", {} -> int)

local R = terralib.cudacompile({ bar = foo },true)

terra doit(N : int, n : int)
    var data = arrayof(int,1,2,3,4)
    var location : &int
    C.cudaMalloc([&&opaque](&location),sizeof(int)*4)
    C.cudaMemcpy(location,&data,sizeof(int)*4,1)
    
	var launch = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
	R.bar(&launch,location, n)
	var data2 : int[4]
  sync()
	C.cudaMemcpy(&data2,location,sizeof(int)*4,2)
  C.printf("!!%d!!\n",data2[2])
    return data2[2]
end

doit(15, 30)


