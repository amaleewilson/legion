if not terralib.cudacompile then
	print("CUDA not enabled, not performing test...")
	return
end
if os.getenv("CI") then
	print("Running in CI environment without a GPU, not performing test...")
	return
end
import "transfer_lang"

local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]



local elem_count = 55
local fid_count = 2
local c_sz = 2

local test_kernel_output = layout_transform_copy src aos, dst soa, size 8, copy_size_per_thread 2, fid_count 5 done 
-- local test_kernel_output = layout_transform_copy src soa, dst aos, size 8, copy_size_per_thread 2, fid_count 5 done 

-- layout_transform_copy src a, dst b done
-- desired: layout_transform_copy aos soa elem_count fid_count c_sz
-- desired return: transpose function that can be used with any 
--                 src/dst with these layouts and sizes 

print("after gen")
   
print(test_kernel_output)

local R = terralib.cudacompile({ bar = test_kernel_output })

print(R.bar)
 
local terra dothing()

  var the_kernel : {&float,&float} -> {}
 

  var data : float[40]
    var locationA : &float
    var locationB : &float
    
    for i = 0,8 do
        data[i*5] = i*5 + 5
        data[i*5 + 1] = i*5 + 1
        data[i*5 + 2] = i*5 + 2
        data[i*5 + 3] = i*5 + 3
        data[i*5 + 4] = i*5 + 4
    end
    
    for i = 0,40 do
        C.printf("%f\n",data[i])
    end

    C.cudaMalloc([&&opaque](&locationA),sizeof(float)*40)
    C.cudaMemcpy(locationA,&data,sizeof(float)*40,1)
    
    C.cudaMalloc([&&opaque](&locationB),sizeof(float)*40)
    
	var launch = terralib.CUDAParams { 1,1,1, 32,1,1, 0, nil }
	R.bar(&launch,locationA, locationB)
	var data2 : float[40]
  sync()
	C.cudaMemcpy(&data2,locationB,sizeof(float)*40,2)
   
    C.printf("\n") 
    for i = 0,40 do
        C.printf("%f\n",data2[i])
    end

end

dothing()

--test_kernel_output()

-- print(a)
--layout_transform_copy done


