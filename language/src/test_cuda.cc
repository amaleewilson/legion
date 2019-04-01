#include "cuda_runtime.h"
#include <stdio.h>
#include <string>
#include "terra.h"
#include <dlfcn.h>
#include <stdlib.h>

int main(int argc, char ** argv) { 
    void *handle;
    void (*cosine)(double*, double*);
    char *error;

    lua_State * L = luaL_newstate(); //create a plain lua state
    luaL_openlibs(L);                //initialize its libraries
    //initialize the terra state in lua
    terra_init(L);

    std::string s = "local tid = cudalib.nvvm_read_ptx_sreg_tid_x\n\
local ntid = cudalib.nvvm_read_ptx_sreg_ntid_x\n\
\n\
-- theone = global(0)\n\
-- \n\
-- theconst = cudalib.constantmemory(int,1)\n\
-- \n\
-- terra foo(result : &float)\n\
--     result[tid()] = tid() + theone + theconst[0]\n\
-- end\n\
-- \n\
-- local C = terralib.includecstring [[\n\
-- #include \"cuda_runtime.h\"\n\
-- #include <stdlib.h>\n\
-- #include <stdio.h>\n\
-- ]]\n\
-- \n\
-- local R,L = terralib.cudacompile({ foo = foo, aone = theone, theconst = theconst },nil,nil,false)\n\
-- \n\
-- terra doit(N : int) : int\n\
--   C.printf(\"test :) \n\")\n\
-- 	var data : &float\n\
-- 	C.cudaMalloc([&&opaque](&data),sizeof(float)*N)\n\
-- 	var one = 1\n\
-- 	var two = 2\n\
-- 	C.cudaMemcpy(R.aone,&one,sizeof(int),1)\n\
-- 	C.cudaMemcpy(R.theconst,&two,sizeof(int),1)\n\
-- 	var launch = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }\n\
-- 	R.foo(&launch,data)\n\
-- 	var results : &float = [&float](C.malloc(sizeof(float)*N))\n\
-- 	C.cudaMemcpy(results,data,sizeof(float)*N,2)\n\
-- 	var result = 0.f\n\
-- 	for i = 0,N do\n\
-- 		result = result + results[i]\n\
-- 	end\n\
-- 	return result\n\
-- end\n\
-- \n\
-- \n\
-- terra main()\n\
--     if L(nil,nil,nil,0) ~= 0 then\n\
--         C.printf(\"WHAT\\n\")\n\
--     end\n\
--     var N = 16\n\
--     var expected = (N - 1)*N/2 + 3*N\n\
--     var ret = terralib.select(doit(N) == expected,0,1)\n\
--     C.printf(\"return %d\" ret);\n\
-- end\n\
-- \n\
-- \n\
-- local ffi = require 'ffi'\n\
-- \n\
-- local path = \"/lib64\"\n\
-- path = terralib.cudahome..path\n\
-- \n\
-- local args = {\"-L\"..path, \"-Wl,-rpath,\"..path, \"-lcuda\", \"-lcudart\"}\n\
-- \n\
-- local name = \"./cuda_test.so\"\n\
-- terralib.saveobj(\"cuda_test.so\",{ doit = doit, main = main },args)\n";


    const char *st = s.c_str();

    if (terra_dostring(L, st)){
        printf("error\n"); 
    }

    printf("\nTEST\n");

    double *in = (double*)malloc(sizeof(double)*64);
    double *out = (double*)malloc(sizeof(double)*64); 

    for (int i = 0; i < 64; ++i){
      in[i] = i;
      out[i] = 0;
    }

    double *d_in, *d_out;

    cudaMalloc((void **)&d_in, sizeof(double)*64);
    cudaMalloc((void **)&d_out, sizeof(double)*64);

    cudaMemcpy(d_in,in, sizeof(double)*64, cudaMemcpyHostToDevice);



   handle = dlopen("/home/amaleewilson/forked_legion/legion/language/src/cuda_test.so", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

   dlerror();    /* Clear any existing error */

   /* Writing: cosine = (double (*)(double)) dlsym(handle, "cos");
       would seem more natural, but the C99 standard leaves
       casting from "void *" to a function pointer undefined.
       The assignment used below is the POSIX.1-2003 (Technical
       Corrigendum 1) workaround; see the Rationale for the
       POSIX specification of dlsym(). */

//   *(void **) (&cosine) = dlsym(handle, "terra_hello");
    void (*callmain)();
   //*(void **) (&cosine) = dlsym(handle, "run_kernel");

   if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

   //printf("%f\n", (*cosine)(d_in, d_out));
   (*cosine)(d_in, d_out);
    dlclose(handle);
    exit(EXIT_SUCCESS);

}

