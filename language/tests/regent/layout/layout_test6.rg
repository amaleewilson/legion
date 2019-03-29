-- Copyright 2019 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Compile and link layout_test_colocation.cc
local clayout_test
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local layout_test_cc = root_dir .. "layout_test_colocation.cc"
  local layout_test_so
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    layout_test_so = out_dir .. "liblayout_test.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    layout_test_so = root_dir .. "liblayout_test.so"
  else
    layout_test_so = os.tmpname() .. ".so" -- root_dir .. "layout_test.so"
  end
  local cxx = os.getenv('CXX') or 'c++'
  local max_dim = os.getenv('MAX_DIM') or '3'

  local cxx_flags = os.getenv('CC_FLAGS') or ''
  --cxx_flags = cxx_flags .. " -O2 -Wall -Werror -DLEGION_MAX_DIM=" .. max_dim .. " -DREALM_MAX_DIM=" .. max_dim
  cxx_flags = cxx_flags .. " -g -O0 -DLEGION_MAX_DIM=" .. max_dim .. " -DREALM_MAX_DIM=" .. max_dim
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 layout_test_cc .. " -o " .. layout_test_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. layout_test_cc)
    assert(false)
  end
  terralib.linklibrary(layout_test_so)
  clayout_test = terralib.includec("layout_test.h", {"-I", root_dir, "-I", runtime_dir})
end

import "regent"

fspace fs
{
  input : double;
  output1 : double;
  output2 : double;
}

task init(r : region(ispace(int2d), fs))
where
  reads writes(r)
do
  for e in r do
    e.input = [double](e.x + e.y)
    e.output1 = 0.0
    e.output2 = 0.0
  end
end

task colocate(r : region(ispace(int2d), fs),
              g : region(ispace(int2d), fs))
where
  reads(r.input, g.input),
  writes(r.output1)
do
  for e in r do
    r[e].output1 = r[e].input + 0.25 * (
                   r[e + {  0,  1 }].input +
                   r[e + {  0, -1 }].input +
                   r[e + { -1,  0 }].input +
                   r[e + {  1,  0 }].input)
  end
end

colocate:get_primary_variant():add_execution_constraint(
  regentlib.layout.colocation_constraint(
    terralib.newlist({
      regentlib.layout.field_constraint("r",
        terralib.newlist({regentlib.layout.field_path("input")})),
      regentlib.layout.field_constraint("g",
        terralib.newlist({regentlib.layout.field_path("input")}))
    })))

task donut(r : region(ispace(int2d), fs),
           g : region(ispace(int2d), fs))
where
  reads(r.input, g.input),
  writes(r.output2)
do
  for e in r do
    r[e].output2 = r[e].input + 0.25 * (
                   g[e + {  0,  1 }].input +
                   g[e + {  0, -1 }].input +
                   g[e + { -1,  0 }].input +
                   g[e + {  1,  0 }].input)
  end
end

task check(r : region(ispace(int2d), fs))
where
  reads(r.{output1, output2})
do
  for e in r do
    regentlib.assert(e.output1 == e.output2, "test failed")
  end
end

task init_halo(r_interior : region(ispace(int2d), fs),
               p_r_interior : partition(disjoint, r_interior, ispace(int2d)),
               sel_halo : region(ispace(int2d), rect2d))
where
  reads writes(sel_halo)
do
  for p in p_r_interior.colors do
    var bounds = p_r_interior[p].bounds
    var halo_bounds = rect2d { lo = bounds.lo - {1, 1},
                               hi = bounds.hi + {1, 1} }
    sel_halo[p] = halo_bounds
  end
end

__demand(__inner)
task toplevel()
  var size = 100
  var r = region(ispace(int2d, {size, size}), fs)
  var interior_cs = ispace(int1d, 1)
  var t : transform(2, 1); t[{0, 0}] = 0; t[{1, 0}] = 0;
  var e = rect2d { int2d { 1, 1 }, int2d { size - 2, size - 2 } }
  var p_interior = restrict(r, t, e, interior_cs)
  var r_interior = p_interior[0]

  var cs = ispace(int2d, {2, 2})
  var p_r_interior = partition(equal, r_interior, cs)
  var sel_halo  = region(ispace(int2d, {2, 2}), rect2d)
  init_halo(r_interior, p_r_interior, sel_halo)

  var p_sel_halo  = partition(equal, sel_halo , cs)

  var p_halo = image(r, p_sel_halo, sel_halo)
  var p_donut = p_halo - p_r_interior

  var p_init = partition(equal, r, p_r_interior.colors)

  for p in p_r_interior.colors do
    init(p_init[p])
  end

  for p in p_r_interior.colors do
    colocate(p_r_interior[p], p_donut[p])
  end
  for p in p_r_interior.colors do
    donut(p_r_interior[p], p_halo[p])
  end
  for p in p_r_interior.colors do
    check(p_r_interior[p])
  end
end

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local link_flags = terralib.newlist({"-L" .. root_dir, "-llayout_test"})
  regentlib.saveobj(toplevel, "layout_test6", "executable", clayout_test.register_mappers, link_flags)
else
  regentlib.start(toplevel, clayout_test.register_mappers)
end
