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


#include <cufft.h>
#include "terra.h"


using namespace Realm;

enum T_method {
  BP_SOA_TO_SOA,
  BP_AOS_TO_AOS,
  BP_SOA_TO_AOS_SINGLE,
  BP_AOS_TO_SOA_SINGLE,
  BP_SOA_TO_AOS,
  BP_AOS_TO_SOA,
  CPU_AOS_TO_SOA,
  CPU_SOA_TO_AOS,
  MEMCPY_ONLY
};

T_method method = BP_SOA_TO_AOS;

int num_elems = 0;
size_t block_count = 1;
size_t thread_count = 1;
int c_sz = 1;

void new_runSoAtoAoSTest(int argc, char **argv, Memory src_mem);

static CUresult initCUDA(int argc, char **argv, CUfunction *SoAtoAos);

// define input ptx file for different platforms
#if defined(_WIN64) || defined(__LP64__)
//#define PTX_FILE "kernel_transpose_gpu64.ptx"
#define PTX_FILE "test_ptx_output.ptx"
#define CUBIN_FILE "kernel_transpose_gpu64.cubin"
#else
#define PTX_FILE "kernel_transpose_gpu32.ptx"
#define CUBIN_FILE "kernel_transpose_gpu32.cubin"
#endif

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
		    const void *userdata, size_t userlen, Processor p){

  Machine machine = Machine::get_machine();
  Memory m = Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).first();
  assert(m.exists());

  typedef int FT;

  char *arg_v = NULL;
  new_runSoAtoAoSTest(0, &arg_v, m);
}

void cpu_tranpose_soa_to_aos(float *h_A, unsigned int mem_size_A, unsigned int size_A, size_t fid_count){

  float *h_C;
  checkCudaErrors(cuMemHostAlloc((void**)&h_C, mem_size_A, 0));
  
  std::string trans_method = "cpu";

  int elem_count = size_A/fid_count;

  // create and start timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  // start the timer
  sdkStartTimer(&timer);
  
    trans_method += "_aos_to_soa";
    for (size_t i = 0; i < size_A; ++i){
      h_C[i] = h_A[i/fid_count + (i%fid_count)*elem_count];
    } 

  // stop and destroy timer
  sdkStopTimer(&timer);
  
  float memcpy_time = sdkGetTimerValue(&timer);
  sdkDeleteTimer(&timer);


#ifdef CHECK_COPY
  // TODO
#endif
    

  // should the mem ops be 2??
  int mem_ops = 1;
  int total_bytes = mem_size_A * mem_ops;
  std::cout << trans_method << "," << memcpy_time  << "," << fid_count << "," << num_elems << "," << total_bytes << "," << total_bytes/memcpy_time/1000000 << std::endl;

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


  int dim_order[1];
  dim_order[0] = 0;


    // src mem
    {
      InstanceLayoutGeneric *ilg; 
      if (method == BP_AOS_TO_SOA || method == BP_AOS_TO_AOS){
        ilg = InstanceLayoutGeneric::choose_instance_layout(is_pad, aos_ilc, dim_order);
      }
      else{
        ilg = InstanceLayoutGeneric::choose_instance_layout(is_pad, soa_ilc, dim_order);
      }
      RegionInstance s_inst;

      Event e = RegionInstance::create_instance(s_inst, src_mem, ilg, ProfilingRequestSet());
      std::vector<CopySrcDstField> tgt(1);
      tgt[0].inst = s_inst;
      tgt[0].field_id = FID_Q;
      tgt[0].size = field_sizes[FID_Q];

      float fill_value = 89.0;
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();

      fill_value = 29.0;
      tgt[0].field_id = FID_R;
      tgt[0].size = field_sizes[FID_R];
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();

      fill_value = 59.0;
      tgt[0].field_id = FID_S;
      tgt[0].size = field_sizes[FID_S];
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();
      
      fill_value = 39.0;
      tgt[0].field_id = FID_T;
      tgt[0].size = field_sizes[FID_T];
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();
      src_insts.push_back(s_inst);

      /*
    int tst[20];
    sharedtst_ptr = tst;

    s_inst.read_untyped(0, tst_ptr, 80);

    log_app.print() << "s_inst";

    for (int i = 0; i < 20; ++i){
      log_app.print() << tst[i];
    }
    */


    } // end src_mem thing




  CUfunction copy_func = NULL;
  
  int block_size = 32;
  //int block_size = 64; // This cut bw in half 




  size_t fid_count = field_sizes.size(); 


  std::string src;
  std::string dst;

  switch(method){
    case BP_SOA_TO_AOS :
      src = "soa";
      dst = "aos"; 
      break;
    case BP_AOS_TO_SOA :
      src = "aos";
      dst = "soa"; 
      break;
    default: 
      src = "soa";
      dst = "aos"; 
      break;
  }



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
  


  std::cout << "terra called before initCUDA\n";


  CUresult error_id = initCUDA(argc, argv, &copy_func);
  if (error_id != CUDA_SUCCESS) {
    printf("initCUDA() returned %d\n-> %s\n", error_id,
           getCudaDrvErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  unsigned int size_A = num_elems * fid_count;
  unsigned int mem_size_A = sizeof(float) * size_A;
  
  float* tst_ptr = new float[size_A];
    
  //MemoryImpl *src_memory = get_runtime()->get_memory_impl(src_insts[0]); 
  //float *src_base =  reinterpret_cast<float *>(src_memory->get_direct_ptr(0, mem_size_A)); 
  //std::cout << "test " << src_base[0] << std::endl;


  //void *src_base = LocalCPUMemory(src_insts[0].get_location()).get_direct_ptr(0, mem_size_A);
  // necessary to read this again???
    
    src_insts[0].read_untyped(0, tst_ptr, mem_size_A);
    
  //src_insts[0].read_untyped(0, h_A, mem_size_A);
  /*
    log_app.print() << V" og s_inst";

    for (int i = 0; i < 20; ++i){
      log_app.print() <<  tst_ptr[i];
    }
  */

  float *h_A;
  checkCudaErrors(cuMemHostAlloc((void**)&h_A, mem_size_A, 0));



  for (size_t i = 0; i < size_A; ++i){
    h_A[i] = tst_ptr[i];
    //std::cout << "test " << src_base[i] << std::endl;
    //h_A[i] = *((float*)(src_base[i]));
  }

/*
  //DEBUGGING THING
  for (size_t i = 0; i < 12; ++i){
    h_A[i] = i * 2 + 1;
  }
  for (size_t i = 13; i < 24; ++i){
    h_A[i] = i * 3 + 1;
  }
  for (size_t i = 0; i < 36; ++i){
    h_A[i] = i * 5 + 1;
  }
  for (size_t i = 0; i < 48; ++i){
    h_A[i] = i * 7 + 1;
  }
*/
  
  // Seems that read_untyped does not work with the cumemhostalloc'd h_A
  // src_insts[0].read_untyped(0, h_A, mem_size_A);


  //std::cout << "h_A[0] " << h_A[0] << "\n";

  CUdeviceptr d_A;
  checkCudaErrors(cuMemAlloc(&d_A, mem_size_A));
      
  CUdeviceptr d_B;
  checkCudaErrors(cuMemAlloc(&d_B, mem_size_A));
  
  checkCudaErrors(cuMemcpyHtoD(d_A, h_A, mem_size_A));
  checkCudaErrors(cuCtxSynchronize());    
  
 
    std::string trans_method = "kernel";
    // create and start timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);

  size_t ne_per_t = c_sz;
  size_t num_elems2 = (size_t)(size_A / fid_count);
  size_t elem_size = sizeof(float);
 
  void *args[6] = {&d_A, &d_B};
  std::vector<void*> vector_args;
  vector_args.push_back(&d_A);
  vector_args.push_back(&d_B);
 
  size_t grid_size = size_A/block_size;

  switch(method){
    case BP_SOA_TO_AOS :
      //TODO
      vector_args = {};
      vector_args.push_back(&d_A);
      vector_args.push_back(&d_B);
      trans_method += "_bp_soa_to_aos";
      break;
    case BP_AOS_TO_SOA :
      //TODO
      vector_args = {};
      vector_args.push_back(&d_A);
      vector_args.push_back(&d_B);
      trans_method += "_bp_aos_to_soa";
      break;
    default: 
      vector_args = {};
      vector_args.push_back(&d_A);
      vector_args.push_back(&d_B);
      trans_method += "_bp_soa_to_aos";
      break;
  }
  
  size_t shared_size = block_size * ne_per_t * sizeof(float); 
  grid_size = block_count;
 
  dim3 block(block_size, 1, 1);
  dim3 grid(grid_size, 1, 1); 
  
  size_t threads_per_block = block.x * block.y * block.z;
  size_t blocks_per_grid = grid.x * grid.y * grid.z;

  //std::cout << "blocks per grid : " <<  blocks_per_grid << "\n";//(size_A)/block_size/ne_per_t << "\n";
  //std::cout << "threads per block : " <<  threads_per_block << "\n";//(size_A)/block_size/ne_per_t << "\n";

  // start the timer
  sdkStartTimer(&timer);
    
  checkCudaErrors(cuLaunchKernel( // TODO: double check the culaunch kernel api 
      copy_func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
      shared_size, NULL, &vector_args[0]/*args*/ , NULL));


  checkCudaErrors(cuCtxSynchronize());

  // stop and destroy timer
  sdkStopTimer(&timer);
  
  float kernel_time = sdkGetTimerValue(&timer);

  sdkDeleteTimer(&timer);
 
#ifdef CHECK_COPY
// TODO: update this for new version

  float *h_Ccheck = reinterpret_cast<float *>(malloc(mem_size_A));
  checkCudaErrors(cuMemcpyDtoH(reinterpret_cast<void *>(h_Ccheck), d_B, mem_size_A)); 
  checkCudaErrors(cuCtxSynchronize());

  bool correct = true;
 
 if (method == BP_SOA_TO_AOS || method == BP_SOA_TO_AOS_SINGLE){ 
    for (int i = 0; i < size_A; i++) {
    //std::cout << "transpose h_Ccheck[" << i << "] : " << h_Ccheck[i] <<  " h_A : " << h_A[i] << std::endl;
  
      if (fabs(h_Ccheck[i] - h_A[i/fid_count + (i%fid_count)*num_elems]) > 1e-5) {
 //       std::cout << "transpose h_Ccheck[" << i << "] : " << h_Ccheck[i] << std::endl;
        correct = false;
      }
    }
 }
 else if (method == BP_AOS_TO_SOA || method == BP_AOS_TO_SOA_SINGLE){
  
    for (int i = 0; i < size_A; i++) {
   // std::cout << "transpose h_Ccheck[" << i << "] : " << h_Ccheck[i] << " h_A : " << h_A[i] << std::endl;
  
      if (fabs(h_Ccheck[i/fid_count + (i%fid_count)*num_elems] - h_A[i]) > 1e-5) {
//        std::cout << "transpose h_Ccheck[" << i << "] : " << h_Ccheck[i/fid_count + (i%fid_count)*num_elems] << " h_A : " << h_A[i] << std::endl;
        correct = false;
      }
    }

 }
 else{
    for (int i = 0; i < size_A; i++) {
    //std::cout << "transpose h_Ccheck[" << i << "] : " << h_Ccheck[i] << " h_A : " << h_A[i] << std::endl;
  
      if (fabs(h_Ccheck[i] - h_A[i]) > 1e-5) {
        //std::cout << "transpose h_Ccheck[" << i << "] : " << h_Ccheck[i/fid_count + (i%fid_count)*num_elems] << " h_A : " << h_A[i] << std::endl;
        correct = false;
      }
    }

 }

  if (!correct){
      std::cout << "failed test" << std::endl;
  }
  else{
      std::cout << "PASSED test" << std::endl;
  }

  free(h_Ccheck);

#endif


  checkCudaErrors(cuMemFreeHost(h_A));
  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuCtxDestroy(cuContext));

  int mem_ops = 1;
  int total_bytes = mem_size_A * mem_ops;
  std::cout << trans_method << "," << kernel_time  << "," << fid_count << "," << num_elems << "," << total_bytes << "," << total_bytes/kernel_time/1000000 << "," << block_count << "," << block_size << "," << c_sz << std::endl;
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
    printf("> initCUDA loading module: <%s>\n", module_path.c_str());
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



int main(int argc, char **argv)
{

  Runtime rt;

  rt.init(&argc, &argv);


  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-block_count")) {
      block_count = std::stoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-thread_count")) {
      thread_count = std::stoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-ne")) {
      num_elems = std::stoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-copy_count")) {
      c_sz = std::stoi(argv[++i]);
      continue;
    }
    if(!strcmp(argv[i], "-method")) {
      const char *meth = argv[++i];
      if (!strcmp(meth, "bp_soa_to_aos"))
        method = BP_SOA_TO_AOS;
      else if (!strcmp(meth, "bp_aos_to_soa"))
        method = BP_AOS_TO_SOA;
      else if (!strcmp(meth, "cpu_soa_to_aos")) 
        method = CPU_SOA_TO_AOS;
      else if (!strcmp(meth, "cpu_aos_to_soa")) 
        method = CPU_AOS_TO_SOA;
      else if (!strcmp(meth, "bp_soa_to_soa"))
        method = BP_SOA_TO_SOA;
      else if (!strcmp(meth, "bp_aos_to_aos"))
        method = BP_AOS_TO_AOS;
      else if (!strcmp(meth, "bp_soa_to_aos_single"))
        method = BP_SOA_TO_AOS_SINGLE;
      else if (!strcmp(meth, "bp_aos_to_soa_single"))
        method = BP_AOS_TO_SOA_SINGLE;
      else
        method = MEMCPY_ONLY;
      
      continue;
    }
  }
  //std::cout << "method " << method  << "\n";

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

