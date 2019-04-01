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


local parser = require("parser")
local codegen = require("codegen")


local tidy = cudalib.nvvm_read_ptx_sreg_tid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bidx = cudalib.nvvm_read_ptx_sreg_ctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bidy = cudalib.nvvm_read_ptx_sreg_ctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bdimx = cudalib.nvvm_read_ptx_sreg_ntid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bdimy = cudalib.nvvm_read_ptx_sreg_ntid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local gdimx = cudalib.nvvm_read_ptx_sreg_nctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local gdimy = cudalib.nvvm_read_ptx_sreg_nctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)



function nu_generate_kernel(src_layout, dst_layout, elem_cnt, fid_cnt, c_size)
 

  if src_layout == "aos" and dst_layout == "soa"
  then
   
    local lt = c_size 
    local loop_term = elem_cnt / lt
 
    local d_A = symbol(&float, "d_A")
    local d_B = symbol(&float, "d_B")

    local c_sz = symbol(int, "c_sz")
    local dst_base = symbol(int[fid_cnt], "dst_base")
    local src_base = symbol(int, "src_base") 


    local setup = terralib.newlist()
    setup:insert(quote
      var [c_sz] = c_size*fid_cnt
      var [dst_base] 
      var [src_base] = 0;
    end)

    for i = 0,fid_cnt-1 do
      setup:insert(quote [dst_base][i] = i*elem_cnt end)
    end
 
    local t_id = symbol(int, "t_id")
    local loop_innards = terralib.newlist()
    loop_innards:insert(quote [src_base] = t_id*[c_sz] end)

    for n = 0,(c_size-1) do
      for p = 0,fid_cnt-1 do
        loop_innards:insert(quote [d_B][dst_base[p] + [t_id]*lt + [n]] = [d_A][src_base + p + n*fid_cnt] end)
      end
    end
    
    return terra([d_A], [d_B])
        var real_tid = ((bidx() + bidy()*gdimx()) * (bdimx()*bdimy()) + (tidy()*bdimx()) + tidx());
        var inc = gdimx()*gdimy()*bdimx()*bdimy();
        [setup]; 
        for [t_id] = real_tid,loop_term, inc do 
          [loop_innards]
        end
      end
  end
end

terralib.includepath = terralib.includepath..";/usr/local/cuda/include"


local transfer_language = {
  name = "transfer_language",
  entrypoints = {"layout_transform_copy"},
  keywords = {"layout_transform_copy", "aos", "soa", "size", "fid_count", "copy_size_per_thread", "src", "dst", "done"},
}

sync = terralib.externfunction("cudaThreadSynchronize", {} -> int)


function transfer_language:expression(lex)
    
  local info = parser:parse(lex)
  return function(environment_function)
      -- local env = environment_function()
      local kernel = codegen.gen_kernel(info)
--      kernel:printpretty()
      return kernel
  end

end


return transfer_language

