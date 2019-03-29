-- Copyright 2019 Stanford University, NVIDIA Corporation
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

-- Bishop Language Entry Point

local parser = require("bishop/parser")
local type_check = require("bishop/type_check")
local optimize_match = require("bishop/optimize_match")
local codegen = require("bishop/codegen")
local std = require("bishop/std")

-- Add Language Builtins to Global Environment

local function add_builtin(k, v)
  assert(rawget(_G, k) == nil, "Builtin " .. tostring(k) .. " already defined")
  rawset(_G, k, v)
end

add_builtin("bishoplib", std)

local language = {
  name = "mapper",
  entrypoints = {
    "mapper",
  },
  keywords = {
    -- elements
    "task",
    "region",
    "for",
    "while",
    "do",

    -- parameter
    "parameter",

    -- trigger
    "trigger",
    "when",

    "mapper",
    "end",
  },
}

function language:statement(lex)
  print("bishop.t function language:statement(lex)")
  local node = parser:parse(lex)
  print("bishop.t node parsed")
  local function ctor(environment_function)
    print("bishop.t ctor")
    node = type_check.mapper(node)
    print("bishop.t node type checked")
    return function()
      node = optimize_match.mapper(node)
      print("bishop.t node optimize_matched")
      return codegen.mapper(node)
    end
  end
  return ctor, {"__bishop_jit_mappers__"}
end

return language
