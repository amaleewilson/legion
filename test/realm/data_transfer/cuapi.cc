#include "realm.h"

#include "realm/realm_config.h"
#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"
#include "realm/threads.h"
#include "realm/transfer/transfer.h"

// includes, system
#include <builtin_types.h>
#include <cuda.h>
#include <drvapi_error_string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

// includes, project
#include <helper_cuda_drvapi.h>
#include <helper_image.h>
#include <helper_string.h>
#include <helper_timer.h>

#include <cstring>
#include <iostream>
#include <string>


#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>
#include <unistd.h>



#include "terra.h"


namespace test_api {

size_t block_count = 1;
size_t thread_count = 1;

static CUresult initCUDA(int argc, char **argv, CUfunction *SoAtoAos);

//#define PTX_FILE "kernel_transpose_gpu64.ptx"
#define PTX_FILE "test_ptx_output.ptx"

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
size_t totalGlobalMem;


void run_test(CUdeviceptr d_A, CUdeviceptr d_B, std::string src, std::string dst, int num_elems, int fid_count, int c_sz, int argc, char **argv){

  CUfunction copy_func = NULL;
  
  int block_size = 32;

    lua_State * Lu = luaL_newstate(); //create a plain lua state
    luaL_openlibs(Lu);                //initialize its libraries
    //initialize the terra state in lua
    terra_init(Lu);


    std::string s = "\n\
   local C = terralib.includecstring [[\n\
   #include \"cuda_runtime.h\"\n\
   #include <stdlib.h>\n\
   #include <stdio.h>\n\
   ]]\n\
   \n\
   import \"transfer_lang\"\n\
   local kf_test = layout_transform_copy src " + src +", dst " + dst + ", size " + std::to_string(num_elems) +", copy_size_per_thread " + std::to_string(c_sz / fid_count) + ", fid_count " + std::to_string(fid_count) + " done\n\
   local R,L = terralib.cudacompile({ kf_test = kf_test },true,nil,false)\n";  
   

    const char *st = s.c_str();

    if (terra_dostring(Lu, st)){
        printf("error\n"); 
    }
  
  CUresult error_id = initCUDA(argc, argv, &copy_func);
  if (error_id != CUDA_SUCCESS) {
    printf("initCUDA() returned %d\n-> %s\n", error_id,
           getCudaDrvErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  unsigned int size_A = num_elems * fid_count;
  unsigned int mem_size_A = sizeof(float) * size_A;
  
  size_t elem_size = sizeof(float);
 
  std::vector<void*> vector_args;
  vector_args.push_back(&d_A);
  vector_args.push_back(&d_B);
 
  size_t grid_size = size_A/block_size;
  
  size_t shared_size = block_size * (c_sz/fid_count) * sizeof(float); 
  grid_size = block_count;
 
  dim3 block(block_size, 1, 1);
  dim3 grid(grid_size, 1, 1); 
  
  size_t threads_per_block = block.x * block.y * block.z;
  size_t blocks_per_grid = grid.x * grid.y * grid.z;

  //std::cout << "blocks per grid : " <<  blocks_per_grid << "\n";//(size_A)/block_size/ne_per_t << "\n";
  //std::cout << "threads per block : " <<  threads_per_block << "\n";//(size_A)/block_size/ne_per_t << "\n";

    
  checkCudaErrors(cuLaunchKernel( // TODO: double check the culaunch kernel api 
      copy_func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
      shared_size, NULL, &vector_args[0]/*args*/ , NULL));


  checkCudaErrors(cuCtxSynchronize());
 
  checkCudaErrors(cuCtxDestroy(cuContext));

}



bool inline findModulePath(const char *module_file, std::string &module_path,
                           char **argv, std::string &ptx_source) {
  char *actual_path = sdkFindFilePath(module_file, "/home/amaleewilson/forked_legion/legion/language/src/test_ptx_output");
  std::cout << "actual_path" << "\n"; 

  if (actual_path) {
    module_path = actual_path;
  } else {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  }

  if (module_path.empty()) {
    printf("> findModulePath file not found: <%s> \n", module_file);
    return false;
  } else {
    //printf("> findModulePath <%s>\n", module_path.c_str());

    if (module_path.rfind(".ptx") != std::string::npos) {
      FILE *fp = fopen(module_path.c_str(), "rb");
      fseek(fp, 0, SEEK_END);
      int file_size = ftell(fp);
      char *buf = new char[file_size + 1];
      fseek(fp, 0, SEEK_SET);
      fread(buf, sizeof(char), file_size, fp);
      fclose(fp);
      buf[file_size] = '\0';
      ptx_source = buf;
      delete[] buf;
    }

    return true;
  }
}

static CUresult initCUDA(int argc, char **argv, CUfunction *SoAtoAos) {
  CUfunction cuFunction = 0;
  CUresult status;
  int major = 0, minor = 0;
  char deviceName[100];
  std::string module_path, ptx_source;

  cuDevice = findCudaDeviceDRV(argc, (const char **)argv);
  
  status = cuCtxCreate(&cuContext, 1, cuDevice);

  if (CUDA_SUCCESS != status) {
    goto Error;
  }

  // first search for the module path before we load the results
  if (!findModulePath(PTX_FILE, module_path, argv, ptx_source)) {
      printf(
          "> findModulePath could not find <matrixMul_kernel> ptx or cubin\n");
      status = CUDA_ERROR_NOT_FOUND;
      goto Error;
  } else {
    //printf("> initCUDA loading module: <%s>\n", module_path.c_str());
  }

  std::cout << "module path " << module_path << "\n";

  if (module_path.rfind("ptx") != std::string::npos) {
    // in this branch we use compilation with parameters
    const unsigned int jitNumOptions = 3;
    CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
    void **jitOptVals = new void *[jitNumOptions];

    // set up size of compilation log buffer
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = reinterpret_cast<void *>(jitLogBufferSize);

    // set up pointer to the compilation log buffer
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    char *jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;

    // set up pointer to set the Maximum # of registers for a particular kernel
    jitOptions[2] = CU_JIT_MAX_REGISTERS;
    int jitRegCount = 32;
    jitOptVals[2] = reinterpret_cast<void *>(jitRegCount);

    status =
        cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions,
                           jitOptions, reinterpret_cast<void **>(jitOptVals));

    //printf("> PTX JIT log:\n%s\n", jitLogBuffer);
  } else {
    status = cuModuleLoad(&cuModule, module_path.c_str());
  }

  if (CUDA_SUCCESS != status) {
    goto Error;
  }



  status = cuModuleGetFunction(&cuFunction, cuModule, "kf_test"); 
  if (CUDA_SUCCESS != status) {
    goto Error;
  }

 *SoAtoAos = cuFunction;


  return CUDA_SUCCESS;
Error:
  cuCtxDestroy(cuContext);
  return status;
}


}

