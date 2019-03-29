if not terralib.cudacompile then
	print("CUDA not enabled, not performing test...")
	return
end
if os.getenv("CI") then
	print("Running in CI environment without a GPU, not performing test...")
	return
end
import "transfer"

local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]



local elem_count = 55
local fid_count = 2
local c_sz = 2
--print(layout_transform_copy src a, dst b done)
--print ("after print")

-- local terra fun_test() 
--   C.printf("hello \n")
-- end
-- 
-- local test_fun_output = fun_test
-- test_fun_output()

local test_kernel_output = layout_transform_copy src a, dst b done 

-- layout_transform_copy src a, dst b done
-- desired: layout_transform_copy aos soa elem_count fid_count c_sz
-- desired return: transpose function that can be used with any 
--                 src/dst with these layouts and sizes 

print("after transform")
   

local R = terralib.cudacompile({ bar = test_kernel_output })

 
local terra dothing()
  var data : float[32]
    var locationA : &float
    var locationB : &float
    
    for i = 0,8 do
        data[i*4] = i*4 + 4
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

end

dothing()

--test_kernel_output()

-- print(a)
--layout_transform_copy done


