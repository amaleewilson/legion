if not terralib.cudacompile then
	print("CUDA not enabled, not performing test...")
	return
end

print("test 1")

local tid = cudalib.nvvm_read_ptx_sreg_tid_x
local ntid = cudalib.nvvm_read_ptx_sreg_ntid_x

theone = global(0)

theconst = cudalib.constantmemory(int,1)

print("test 2")

terra foo(result : &float)
    result[tid()] = tid() + theone + theconst[0]
end

local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

local R,L = terralib.cudacompile({ foo = foo, aone = theone, theconst = theconst },nil,nil,false)
print("test 3")

terra doit(N : int)
  C.printf("test :) \n")
	var data : &float
	C.cudaMalloc([&&opaque](&data),sizeof(float)*N)
	var one = 1
	var two = 2
	C.cudaMemcpy(R.aone,&one,sizeof(int),1)
	C.cudaMemcpy(R.theconst,&two,sizeof(int),1)
	var launch = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
	R.foo(&launch,data)
	var results : &float = [&float](C.malloc(sizeof(float)*N))
	C.cudaMemcpy(results,data,sizeof(float)*N,2)
	var result = 0.f
	for i = 0,N do
		result = result + results[i]
	end
	return result
end


terra run_main() : int
    if L(nil,nil,nil,0) ~= 0 then
        C.printf("WHAT\n")
    end
    var N = 16
    var expected = (N - 1)*N/2 + 3*N
    return doit(N)
end

print("test 4")
local ffi = require 'ffi'
local path = "/lib64"
path = terralib.cudahome..path
local args = {"-L"..path, "-Wl,-rpath,"..path, "-lcuda", "-lcudart"}
terralib.saveobj("cudaol.so",{  run_main = run_main },args)

