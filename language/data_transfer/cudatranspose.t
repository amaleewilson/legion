if not terralib.cudacompile then
	print("CUDA not enabled, not performing test...")
	return
end
if os.getenv("CI") then
	print("Running in CI environment without a GPU, not performing test...")
	return
end

local tidx = cudalib.nvvm_read_ptx_sreg_tid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local tidy = cudalib.nvvm_read_ptx_sreg_tid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bidx = cudalib.nvvm_read_ptx_sreg_ctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bidy = cudalib.nvvm_read_ptx_sreg_ctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bdimx = cudalib.nvvm_read_ptx_sreg_ntid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bdimy = cudalib.nvvm_read_ptx_sreg_ntid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local gdimx = cudalib.nvvm_read_ptx_sreg_nctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local gdimy = cudalib.nvvm_read_ptx_sreg_nctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

foo = terra(d_A : &float, d_B : &float, elem_count : int, fid_count : int, c_sz : int)
    var real_tid = ((bidx() + bidy()*gdimx()) * (bdimx()*bdimy()) + (tidy()*bdimx()) + tidx());
        
  var dst_base0 = 0;
  var dst_base1 = elem_count;
  var dst_base2 = elem_count*2;
  var dst_base3 = elem_count*3;
  var lt = c_sz / fid_count; 
  var loop_term = elem_count/lt;
  var inc = gdimx()*gdimy()*bdimx()*bdimy();
  var src_base = 0;
  
  for t_id = real_tid,loop_term, inc  do
    src_base = t_id*c_sz;
    -- #pragma unroll 
  
    for i = 0,lt  do
      d_B[dst_base0 + t_id*lt + i] = d_A[src_base + 0 + i*fid_count];
      d_B[dst_base1 + t_id*lt + i] = d_A[src_base + 1 + i*fid_count];
      d_B[dst_base2 + t_id*lt + i] = d_A[src_base + 2 + i*fid_count];
      d_B[dst_base3 + t_id*lt + i] = d_A[src_base + 3 + i*fid_count];
    end
  end


    -- result[real_tid] = real_tid
end

terralib.includepath = terralib.includepath..";/usr/local/cuda/include"


local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

sync = terralib.externfunction("cudaThreadSynchronize", {} -> int)

local R = terralib.cudacompile({ bar = foo },true)

terra doit(N : int, n : float)
    var data : float[32]
    var locationA : &float
    var locationB : &float
    
    for i = 0,8 do
        data[i*4] = 0
        data[i*4 + 1] = 1
        data[i*4 + 2] = 2
        data[i*4 + 3] = 3
    end
    
    for i = 0,32 do
        C.printf("%f\n",data[i])
    end

    C.cudaMalloc([&&opaque](&locationA),sizeof(float)*32)
    C.cudaMemcpy(locationA,&data,sizeof(float)*32,1)
    
    C.cudaMalloc([&&opaque](&locationB),sizeof(float)*32)
    -- C.cudaMemcpy(locationB,&data,sizeof(float)*32,1)
    
	var launch = terralib.CUDAParams { 1,1,1, 32,1,1, 0, nil }
	R.bar(&launch,locationA, locationB, 8, 4, 4)
	var data2 : float[32]
  sync()
	C.cudaMemcpy(&data2,locationB,sizeof(float)*32,2)
   
    C.printf("\n") 
    for i = 0,32 do
        C.printf("%f\n",data2[i])
    end

    return data2[2]
end

doit(4, 30.1)


