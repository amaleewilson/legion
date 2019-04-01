#include <stdio.h>
#include <string>
#include "terra.h"

int main(int argc, char ** argv) {
    lua_State * L = luaL_newstate(); //create a plain lua state
    luaL_openlibs(L);                //initialize its libraries
    //initialize the terra state in lua
    terra_init(L);

    std::string s = "print (\"hello\")  \
if not terralib.cudacompile then \n \
  print(\"CUDA not enabled, not performing test...\")\n\
	return \n\
end \n\
if os.getenv(\"CI\") then \n\
  print(\"Running in CI environment without a GPU, not performing test...\")\n\
 	return \n\
end \n\
import \"data_transfer/transfer_lang\" \n\
local C = terralib.includecstring [[\n \
#include \"cuda_runtime.h\" \n \
#include <stdlib.h> \n \
#include <stdio.h> \n \
]]\n \
local test_kernel_output = layout_transform_copy src aos, dst soa, size 8, copy_size_per_thread 2, fid_count 5 done \n\
-- print(test_kernel_output) \n \
print(999) \n\
local R = terralib.cudacompile({ bar = test_kernel_output })\n\
print(R.bar) \n\
local terra dothing()\n\
  var data : float[40]\n\
    var locationA : &float\n\
    var locationB : &float\n\
    for i = 0,8 do\n\
        data[i*5] = i*5 + 5\n\
        data[i*5 + 1] = i*5 + 1\n\
        data[i*5 + 2] = i*5 + 2\n\
        data[i*5 + 3] = i*5 + 3\n\
        data[i*5 + 4] = i*5 + 4\n\
    end\n\
    for i = 0,40 do\n\
        C.printf(\"%f\\n\",data[i])\n\
    end\n\
    C.cudaMalloc([&&opaque](&locationA),sizeof(float)*40)\n\
    C.cudaMemcpy(locationA,&data,sizeof(float)*40,1)\n\
    C.cudaMalloc([&&opaque](&locationB),sizeof(float)*40)\n\
	var launch = terralib.CUDAParams { 1,1,1, 32,1,1, 0, nil }\n\
	R.bar(&launch,locationA, locationB)\n\
	var data2 : float[40]\n\
  sync()\n\
	C.cudaMemcpy(&data2,locationB,sizeof(float)*40,2)\n\
    C.printf(\"\\n\") \n\
    for i = 0,40 do\n\
        C.printf(\"%f\\n\",data2[i])\n\
    end\n\
end\n\
dothing()\n\
      ";
    const char *st = s.c_str();

    if (terra_dostring(L, st)){
        printf("error\n"); 
    }
    return 0;
}
