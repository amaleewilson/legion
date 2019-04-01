
local c = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]

local codegen = {}


local tidx = cudalib.nvvm_read_ptx_sreg_tid_x--terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local tidy = cudalib.nvvm_read_ptx_sreg_tid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bidx = cudalib.nvvm_read_ptx_sreg_ctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bidy = cudalib.nvvm_read_ptx_sreg_ctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bdimx = cudalib.nvvm_read_ptx_sreg_ntid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bdimy = cudalib.nvvm_read_ptx_sreg_ntid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local gdimx = cudalib.nvvm_read_ptx_sreg_nctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local gdimy = cudalib.nvvm_read_ptx_sreg_nctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)




function codegen.gen_kernel(info)
    
  local lt = info.copy_count 
  local loop_term = info.size / lt
 
  local d_A = symbol(&float, "d_A")
  local d_B = symbol(&float, "d_B")

  local c_sz = symbol(int, "c_sz")
  local dst_base = nil
  local src_base = nil
    

  local setup = terralib.newlist()
  setup:insert(quote
    var [c_sz] = info.copy_count*info.fid_count
  end)
  
  local t_id = symbol(int, "t_id")
  local loop_innards = terralib.newlist()

  if info.src_layout == "aos" then
    src_base = symbol(int, "src_base") 
    setup:insert(quote var [src_base] end)   
    loop_innards:insert(quote [src_base] = t_id*[c_sz] end)
  elseif info.src_layout == "soa" then
    src_base = symbol(int[info.fid_count], "src_base")
    setup:insert(quote var [src_base] end)
    for i = 0,info.fid_count-1 do
      setup:insert(quote [src_base][i] = i*info.size end)
    end
  end

  if info.dst_layout == "soa" then
    dst_base = symbol(int[info.fid_count], "dst_base")
    setup:insert(quote var [dst_base] end)
    for i = 0,info.fid_count-1 do
      setup:insert(quote [dst_base][i] = i*info.size end)
    end
  elseif info.dst_layout == "aos" then
    dst_base = symbol(int, "dst_base") 
    setup:insert(quote var [dst_base] end)   
    loop_innards:insert(quote [dst_base] = t_id*[c_sz] end)
  end


  if info.src_layout == "aos" and info.dst_layout == "soa" then
    for n = 0,(info.copy_count-1) do
      for p = 0,info.fid_count-1 do
        loop_innards:insert(quote [d_B][dst_base[p] + [t_id]*lt + [n]] = [d_A][src_base + p + n*info.fid_count] end)
      end
    end
  elseif info.src_layout == "soa" and info.dst_layout == "aos" then 
    for n = 0,(info.copy_count-1) do
      for p = 0,info.fid_count-1 do
        loop_innards:insert(quote [d_B][dst_base + p + n*info.fid_count] = [d_A][src_base[p] + [t_id]*lt + [n]] end)
      end
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

return codegen
