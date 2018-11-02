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


using namespace Realm;

int num_elems = 0;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void new_runSoAtoAoSTest(int argc, char **argv, Memory src_mem);


static CUresult initCUDA(int argc, char **argv, CUfunction *SoAtoAos);

// define input ptx file for different platforms
#if defined(_WIN64) || defined(__LP64__)
#define PTX_FILE "kernel_transpose_gpu64.ptx"
#define CUBIN_FILE "kernel_transpose_gpu64.cubin"
#else
#define PTX_FILE "kernel_transpose_gpu32.ptx"
#define CUBIN_FILE "kernel_transpose_gpu32.cubin"
#endif

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
size_t totalGlobalMem;


Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  MEMSPEED_TASK,
  COPYPROF_TASK,
};

enum FieldIDs {
 FID_Q,
 FID_R,
 FID_S,
 FID_T,
 FID_U,
 FID_V,
};


std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
 // log_app.print() << "Copy/transpose test";

  // just sysmem for now
  Machine machine = Machine::get_machine();
  Memory m = Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).first();
  assert(m.exists());

  typedef int FT;
  //do_single_dim<1, FT>(m, m, log2_buffer_size, p);
  // do_single_dim<2, FT>(m, m, log2_buffer_size, p);
  // do_single_dim<3, FT>(m, m, log2_buffer_size, p);


  char *arg_v = NULL;
  new_runSoAtoAoSTest(0, &arg_v, m);
}

void new_runSoAtoAoSTest(int argc, char **argv, Memory src_mem){

  Rect<1> bounds(0, num_elems-1);
  IndexSpace<1> is(bounds);

  Rect<1> bounds_pad(0, num_elems-1);
  IndexSpace<1> is_pad(bounds_pad);

  std::vector<RegionInstance> src_insts, dst_insts;

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_Q] = sizeof(float);
  field_sizes[FID_R] = sizeof(float);
  field_sizes[FID_S] = sizeof(float);
  field_sizes[FID_T] = sizeof(float);
  InstanceLayoutConstraints aos_ilc(field_sizes, 1); //AOS 
  InstanceLayoutConstraints soa_ilc(field_sizes, 0); //SOA 

  /*
  int dim_order[4];
  dim_order[0] = FID_Q;
  dim_order[1] = FID_R;
  dim_order[2] = FID_S;
  dim_order[3] = FID_T;
*/

  int dim_order[2];
  dim_order[0] = FID_Q;
  dim_order[1] = FID_R;


    // src mem
    {
      InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout(is_pad, soa_ilc, dim_order);
      RegionInstance s_inst;

      Event e = RegionInstance::create_instance(s_inst, src_mem, ilg, ProfilingRequestSet());
      std::vector<CopySrcDstField> tgt(1);
      tgt[0].inst = s_inst;
      tgt[0].field_id = FID_Q;
      tgt[0].size = field_sizes[FID_Q];

      int fill_value = 89;
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();

      fill_value = 29;
      tgt[0].field_id = FID_R;
      tgt[0].size = field_sizes[FID_R];
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();
/*
      fill_value = 59;
      tgt[0].field_id = FID_S;
      tgt[0].size = field_sizes[FID_S];
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();

      fill_value = 39;
      tgt[0].field_id = FID_T;
      tgt[0].size = field_sizes[FID_T];
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();
*/
      src_insts.push_back(s_inst);

      /*
    int tst[20];
    int* tst_ptr = tst;

    s_inst.read_untyped(0, tst_ptr, 80);

    log_app.print() << "s_inst";

    for (int i = 0; i < 20; ++i){
      log_app.print() << tst[i];
    }
    */


    } // end src_mem thing

  CUfunction copy_func = NULL;
  
  //Adjust this variable.
  int block_size = 32;

  CUresult error_id = initCUDA(argc, argv, &copy_func);
  if (error_id != CUDA_SUCCESS) {
    printf("initCUDA() returned %d\n-> %s\n", error_id,
           getCudaDrvErrorString(error_id));
    exit(EXIT_FAILURE);
  }


    // 2 fid's 
  unsigned int size_A = num_elems * 2;
  unsigned int mem_size_A = sizeof(float) * size_A;
  
  
    int tst[size_A];
    int* tst_ptr = tst;
    
    //MemoryImpl *src_memory = get_runtime()->get_memory_impl(src_insts[0]); 
    //float *src_base =  reinterpret_cast<float *>(src_memory->get_direct_ptr(0, mem_size_A)); 
    //std::cout << "test " << src_base[0] << std::endl;


    //void *src_base = LocalCPUMemory(src_insts[0].get_location()).get_direct_ptr(0, mem_size_A);
// necessary to read this again???
    src_insts[0].read_untyped(0, tst_ptr, mem_size_A);
    //src_insts[0].read_untyped(0, h_A, mem_size_A);

    /*
    log_app.print() << " og s_inst";

    for (int i = 0; i < 20; ++i){
      log_app.print() << tst[i];
    }
    */


  float *h_A2;
  checkCudaErrors(cuMemHostAlloc((void**)&h_A2, mem_size_A, 0));



    for (size_t i = 0; i < size_A; ++i){
        h_A2[i] = tst_ptr[i];
        //std::cout << "test " << src_base[i] << std::endl;
        //h_A[i] = *((float*)(src_base[i]));
    }

  // I think offset should be 0 here but not always
  // read_untyped does not work with the cumemhostalloc'd h_A
  //src_insts[0].read_untyped(0, h_A, mem_size_A);

    float *h_B2;
  h_B2 = &(h_A2[num_elems]);

  //std::cout << "h_A[0] " << h_A[0] << "\n";

  CUdeviceptr d_C2;
  checkCudaErrors(cuMemAlloc(&d_C2, mem_size_A));
  
  // create and start timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  // start the timer
  sdkStartTimer(&timer);
  

////// kernel timing
  dim3 block(block_size, 1, 1);
  dim3 grid((num_elems*2)/block_size, 1, 1);

      size_t num_elems2 = (size_t)(size_A / 2);
      size_t elem_size = sizeof(float);
      void *args[5] = {&h_A2, &h_B2, &d_C2, &elem_size, &num_elems2};
      // new CUDA 4.0 Driver API Kernel launch call
      checkCudaErrors(cuLaunchKernel( // TODO: double check the culaunch kernel api 
          copy_func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
          2 * block_size * block_size * sizeof(float), NULL, args, NULL));

  checkCudaErrors(cuCtxSynchronize());

  // stop and destroy timer
  sdkStopTimer(&timer);
  
    float kernel_time = sdkGetTimerValue(&timer);

  //printf("Copy time for kernel: %f (ms)\n", sdkGetTimerValue(&timer));
  //printf("Copy time for cumemcpyhtod: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);
  ///// end kernel timing 
  


  checkCudaErrors(cuMemFreeHost(h_A2));
  checkCudaErrors(cuMemFree(d_C2));
  checkCudaErrors(cuCtxDestroy(cuContext));

  std::cout << "kernel_transpose," << kernel_time  << ",2," << num_elems << "," << mem_size_A << "," << mem_size_A/kernel_time/1000000 << std::endl;
}



bool inline findModulePath(const char *module_file, std::string &module_path,
                           char **argv, std::string &ptx_source) {
  char *actual_path = sdkFindFilePath(module_file, argv[0]);

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

  // get compute capabilities and the devicename
  checkCudaErrors(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  checkCudaErrors(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
  //printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
  //printf("  Total amount of global memory:     %llu bytes\n",
  //       (long long unsigned int)totalGlobalMem);
  //printf("  64-bit Memory Address:             %s\n",
  //       (totalGlobalMem > (uint64_t)4 * 1024 * 1024 * 1024L) ? "YES" : "NO");

  status = cuCtxCreate(&cuContext, 0, cuDevice);

  if (CUDA_SUCCESS != status) {
    goto Error;
  }

  // first search for the module path before we load the results
  if (!findModulePath(PTX_FILE, module_path, argv, ptx_source)) {
    if (!findModulePath(CUBIN_FILE, module_path, argv, ptx_source)) {
      printf(
          "> findModulePath could not find <matrixMul_kernel> ptx or cubin\n");
      status = CUDA_ERROR_NOT_FOUND;
      goto Error;
    }
  } else {
    //printf("> initCUDA loading module: <%s>\n", module_path.c_str());
  }

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

    status = cuModuleGetFunction(&cuFunction, cuModule, "copykernelAoS32_32bit");

  if (CUDA_SUCCESS != status) {
    goto Error;
  }

 *SoAtoAos = cuFunction;


  return CUDA_SUCCESS;
Error:
  cuCtxDestroy(cuContext);
  return status;
}



int main(int argc, char **argv)
{
num_elems = std::stoi(argv[1]);

  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}

