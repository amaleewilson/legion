local parsing = require("parsing")
local ast = require("data_transfer/ast")

local parser = {}

local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]


local tidx = cudalib.nvvm_read_ptx_sreg_tid_x--terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local tidy = cudalib.nvvm_read_ptx_sreg_tid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bidx = cudalib.nvvm_read_ptx_sreg_ctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bidy = cudalib.nvvm_read_ptx_sreg_ctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local bdimx = cudalib.nvvm_read_ptx_sreg_ntid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local bdimy = cudalib.nvvm_read_ptx_sreg_ntid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

local gdimx = cudalib.nvvm_read_ptx_sreg_nctaid_x --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)
local gdimy = cudalib.nvvm_read_ptx_sreg_nctaid_y --terralib.intrinsic("llvm.nvvm.read.ptx.sreg.tid.x",{} -> int)

-- TODO: an AST is a better long term solution
local info_struct = {}


function parser.src(p)
  p:expect("src")
    if p:matches("aos") then
      p:next()
      -- nicer way to parse this?
      info_struct["src_layout"] = "aos"
    elseif p:matches("soa") then
      p:next()
      info_struct["src_layout"] = "soa" 
    end
end

function parser.dst(p)
  p:expect("dst")
    if p:matches("aos") then
      p:next()
      -- nicer way to parse this?
      info_struct["dst_layout"] = "aos"
    elseif p:matches("soa") then
      p:next()
      info_struct["dst_layout"] = "soa" 
    end
end

function parser.top(p)
  if not p:matches("layout_transform_copy") then
    p:error("unexpected token in top-level statement")
  end

  p:expect("layout_transform_copy")
  if not p:matches("done") then
    repeat
      if p:matches("src") then
        p:src() 
      end
      if p:matches("dst") then
        p:dst()
      end
      if p:matches("size") then
        p:next()
        info_struct["size"] = p:expect(p.number).value
      end
      if p:matches("copy_size_per_thread") then
        p:next()
        info_struct["copy_count"] = p:expect(p.number).value
      end
      if p:matches("fid_count") then
        p:next()
        info_struct["fid_count"] = p:expect(p.number).value
      end
    until not p:nextif(",")
  end 
  p:expect("done")

  return info_struct

end


function parser:parse(lex)
  return parsing.Parse(self, lex, "top")
end

return parser

