if not terralib.cudacompile then
	print("CUDA not enabled, not performing test...")
	return
end
if os.getenv("CI") then
	print("Running in CI environment without a GPU, not performing test...")
	return
end

local tidx = cudalib.nvvm_read_ptx_sreg_tid_x--terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

--C = terralib.includec("stdio.h")
local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)


local parser = require("data_transfer/parser")
local codegen = require("data_transfer/codegen")


local tidy = cudalib.nvvm_read_ptx_sreg_tid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bidx = cudalib.nvvm_read_ptx_sreg_ctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bidy = cudalib.nvvm_read_ptx_sreg_ctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bdimx = cudalib.nvvm_read_ptx_sreg_ntid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bdimy = cudalib.nvvm_read_ptx_sreg_ntid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local gdimx = cudalib.nvvm_read_ptx_sreg_nctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local gdimy = cudalib.nvvm_read_ptx_sreg_nctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)


function nu_generate_kernel(elem_cnt, fid_cnt, c_size)
  print("test")
  
  --local test_lst_a = terralib.newlist()
  --test_lst_a:insert(A)  

  local lst = terralib.newlist()
  local lst2 = terralib.newlist()
  local d_A = symbol(&float, "d_A")
  local d_B = symbol(&float, "d_B")
  local elem_count = symbol(int, "elem_count")
  local fid_count = symbol(int, "fid_count")
  local c_sz = symbol(int, "c_sz")
  local real_tid = symbol(int, "real_tid")


  local setup = terralib.newlist()
  setup:insert(quote
    var [fid_count] = [fid_cnt]
    var [elem_count] = [elem_cnt]
    var [c_sz] = [c_size]
    var [real_tid] = ((bidx() + bidy()*gdimx()) * (bdimx()*bdimy()) + (tidy()*bdimx()) + tidx());
  end)

  local dst_base = symbol(int[4], "dst_base")
  setup:insert(quote var [dst_base] end)
  for i = 0,fid_cnt-1 do
    setup:insert(quote [dst_base][i] = [i]*elem_count end)
  end

  local test_thing = symbol(int, "test_thing")
  lst2:insert(quote var [test_thing] = [real_tid] end)

  local loop_term = symbol(int, "loop_term") 
  local inc = symbol(int, "inc") 
  local lt = symbol(int, "lt") 
  local src_base = symbol(int, "src_base") 
 
  lst:insert(quote

    var [lt] = [c_sz] / [fid_count]; 
    var [loop_term] = [elem_count]/lt;
    var [inc] = gdimx()*gdimy()*bdimx()*bdimy();
    var [src_base] = 0;
    [lst2]
    end)

  local transpose_ops = terralib.newlist()
  local t_id = symbol(int, "t_id")
  local i = symbol(int, "i")

print (fid_cnt)

for n = 0,((c_size/fid_cnt)-1) do
  for p = 0,fid_cnt-1 do
    transpose_ops:insert(quote [d_B][dst_base[p] + [t_id]*[lt] + [n]] = [d_A][src_base + p + n*[fid_count]] end)
  end
end

  lst:insert(quote 

    for [t_id] = [real_tid],[loop_term], [inc]  do
      [src_base] = t_id*[c_sz];
      [transpose_ops]
    end
  end)
  return terra([d_A], [d_B], [elem_count], [fid_count], [c_sz])
      [setup]; 
      [lst];
    end
end

terralib.includepath = terralib.includepath..";/usr/local/cuda/include"


local transfer_language = {
  name = "transfer_language",
  entrypoints = {"layout_transform_copy"},
  keywords = {"layout_transform_copy", "aos", "soa", "src", "dst", "done"},
}

sync = terralib.externfunction("cudaThreadSynchronize", {} -> int)

-- TODO: revist wether you should do statement or expr or both
function transfer_language:statement(lex)
  print("statment")
  --local node = parser:parse(lex)
  
  local p = lex
  if not p:matches("layout_transform_copy") then
    p:error("no match unexpected token in top-level statement")
  end
  p:expect("layout_transform_copy")
    local a = nil
    local b = nil
    if not p:matches("done") then
      repeat
        if p:matches("src") then
          p:next()
          if p:matches(p.name) then
            local name = p:next().value
            p:ref(name)
            a = name
          end
        end
        if p:matches("dst") then
          p:next()
          if p:matches(p.name) then
            local name = p:next().value
            p:ref(name)
            b = name
          end
        end
      until not p:nextif(",")
    end 
     p:expect("done")
    print("parser.top")
  return function(environment_function)
      local env = environment_function()
      local kernel = nu_generate_kernel(8, 4, 4)
      local k = kernel
      kernel:printpretty()
      print("returning kernel in stmt")
    
      local R = terralib.cudacompile({ bar = kernel })
      
      local terra dothing()
    var data : float[32]
    var locationA : &float
    var locationB : &float
    
    for i = 0,8 do
        data[i*4] = i*4
        data[i*4 + 1] = i*4 + 1
        data[i*4 + 2] = i*4 + 2
        data[i*4 + 3] = i*4 + 3
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

        C.printf("and were done\n")
      end
      dothing()
      return kernel
  end

end

function transfer_language:expression(lex)
  print("statment")
  --local node = parser:parse(lex)
  
  local p = lex
  if not p:matches("layout_transform_copy") then
    p:error("no match unexpected token in top-level statement")
  end
  p:expect("layout_transform_copy")
    local a = nil
    local b = nil
    if not p:matches("done") then
      repeat
        if p:matches("src") then
          p:next()
          if p:matches(p.name) then
            local name = p:next().value
            p:ref(name)
            a = name
          end
        end
        if p:matches("dst") then
          p:next()
          if p:matches(p.name) then
            local name = p:next().value
            p:ref(name)
            b = name
          end
        end
      until not p:nextif(",")
    end 
     p:expect("done")
    print("parser.top")
  return function(environment_function)
      local env = environment_function()
      local kernel = nu_generate_kernel(8, 4, 4)
      local k = kernel
      kernel:printpretty()
      print("returning kernel in expr")
    
      return kernel
  end

end


return transfer_language

