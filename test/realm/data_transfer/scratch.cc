#include "cuda_runtime.h"
#include <stdio.h>
#include <string>
#include "terra.h"
#include <cuda.h>
#include <dlfcn.h>
#include <stdlib.h>


void data_transform(CUdeviceptr d_A, CUdeviceptr d_B, std::string src_layout, std::string dst_layout, int size, int copy_per_thread, int fid_count){



    void *handle;
    void (*cosine)(float*, float*);
    char *error;

    lua_State * Lu = luaL_newstate(); //create a plain lua state
    luaL_openlibs(Lu);                //initialize its libraries
    //initialize the terra state in lua
    terra_init(Lu);

    std::string fid_cnt = std::to_string(fid_count);
    std::string sz = std::to_string(size);
    std::string cpt = std::to_string(copy_per_thread);


std::string s = "\n\
  local C = terralib.includecstring [[\n\
  #include \"cuda_runtime.h\"\n\
  #include <stdlib.h>\n\
  #include <stdio.h>\n\
  ]]\n\
  \n\
  import \"data_transfer/transfer_lang\"\n\
  local kf = layout_transform_copy src " + src_layout + ", dst " + dst_layout + ", size " + sz+ ", copy_size_per_thread " + cpt + ", fid_count " + fid_cnt + " done\n\
  local R,L = terralib.cudacompile({ kf = kf },nil,nil,false)\n\
  \n\
  terra run_main(A : &float, B : &float)\n\
    -- I do not understand what this L is doing, but it seems necessary.\n\
    if L(nil,nil,nil,0) ~= 0 then\n\
        C.printf(\"WHAT\\n\")\n\
    end\n\
    var N = 16\n\
   	var launch = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }\n\
	  R.kf(&launch, A, B)\n\
  end\n\
  \n\
  local path = \"/lib64\"\n\
  path = terralib.cudahome..path\n\
  local args = {\"-L\"..path, \"-Wl,-rpath,\"..path, \"-lcuda\", \"-lcudart\"}\n\
  terralib.saveobj(\"kernel_gen.so\",{  run_main = run_main },args)";

    const char *st = s.c_str();

    if (terra_dostring(Lu, st)){
        printf("error\n"); 
    }

    printf("\nTEST\n");

//     double *in = (double*)malloc(sizeof(double)*64);
//     double *out = (double*)malloc(sizeof(double)*64); 
// 
//     for (int i = 0; i < 64; ++i){
//       in[i] = i;
//       out[i] = 0;
//     }
// 
//     double *d_in, *d_out;
// 
//     cudaMalloc((void **)&d_in, sizeof(double)*64);
//     cudaMalloc((void **)&d_out, sizeof(double)*64);
// 
//     cudaMemcpy(d_in,in, sizeof(double)*64, cudaMemcpyHostToDevice);



   //handle = dlopen("/home/amaleewilson/forked_legion/legion/language/src/cuda_hello.so", RTLD_LAZY);
   handle = dlopen("/home/amaleewilson/forked_legion/legion/language/src/kernel_gen.so", RTLD_LAZY);

    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

   dlerror();    /* Clear any existing error */

//   *(void **) (&cosine) = dlsym(handle, "terra_hello");
    void (*run_main)(float*, float*);
   *(void **) (&run_main) = dlsym(handle, "run_main");

   if ((error = dlerror()) != NULL)  {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }

   
   //printf("%f\n", (*cosine)(d_in, d_out));
   //(*cosine)(d_in, d_out);
  //(*cosine)(in, out);
    //(*run_main)(d_in, d_out);
    
    (*run_main)((float *)d_A, (float *)d_B);

//  cudaMemcpy(out, d_out, sizeof(double)*64, cudaMemcpyDeviceToHost);

//   
//    for (int i = 0; i < 64; ++i){
//     printf("out %f\n", out[i]);
// 
//   }



    dlclose(handle);
    exit(EXIT_SUCCESS);

}

