#include "cuda_runtime.h"
#include <stdio.h>
#include <string>
#include "terra.h"
#include <dlfcn.h>
#include <stdlib.h>

int main(int argc, char ** argv) { 



      //   FILE *fp;  
      //   fp = fopen("/home/amaleewilson/forked_legion/legion/language/src/test_ptx_output2.ptx", "w");//opening file  
      //   fprintf(fp, "testo\n");//writing data into file  
      //   fclose(fp);//closing file  


    void *handle;
    void (*cosine)(float*, float*);
    char *error;

    lua_State * Lu = luaL_newstate(); //create a plain lua state
    luaL_openlibs(Lu);                //initialize its libraries
    //initialize the terra state in lua
    terra_init(Lu);

    std::string fid_count = "4";


std::string s = "\n\
  local C = terralib.includecstring [[\n\
  #include \"cuda_runtime.h\"\n\
  #include <stdlib.h>\n\
  #include <stdio.h>\n\
  ]]\n\
  \n\
  import \"transfer_lang\"\n\
  local kf = layout_transform_copy src aos, dst soa, size 16, copy_size_per_thread 2, fid_count " + fid_count + " done\n\
  local R,L = terralib.cudacompile({ kf = kf },true,nil,false)\n\
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
  terralib.saveobj(\"cudaoldc.so\",{  run_main = run_main },args)";

    const char *st = s.c_str();

    if (terra_dostring(Lu, st)){
        printf("error\n"); 
    }

    printf("\nTEST\n");

    float *in = (float*)malloc(sizeof(float)*64);
    float *out = (float*)malloc(sizeof(float)*64); 

    for (int i = 0; i < 64; ++i){
      in[i] = i;
      out[i] = 0;
    }
 
  printf("input data\n"); 
   for (int i = 0; i < 64; ++i){
    printf("in %f\n", in[i]);

  }

    float *d_in, *d_out;

    cudaMalloc((void **)&d_in, sizeof(float)*64);
    cudaMalloc((void **)&d_out, sizeof(float)*64);

    cudaMemcpy(d_in,in, sizeof(float)*64, cudaMemcpyHostToDevice);



   //handle = dlopen("/home/amaleewilson/forked_legion/legion/language/src/cuda_hello.so", RTLD_LAZY);
   handle = dlopen("/home/amaleewilson/forked_legion/legion/language/src/cudaoldc.so", RTLD_LAZY);

    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

   dlerror();    /* Clear any existing error */

   /* Writing: cosine = (float (*)(float)) dlsym(handle, "cos");
       would seem more natural, but the C99 standard leaves
       casting from "void *" to a function pointer undefined.
       The assignment used below is the POSIX.1-2003 (Technical
       Corrigendum 1) workaround; see the Rationale for the
       POSIX specification of dlsym(). */

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
    (*run_main)(d_in, d_out);

  cudaMemcpy(out, d_out, sizeof(float)*64, cudaMemcpyDeviceToHost);

  printf("\n\noutput data\n"); 
  
   for (int i = 0; i < 64; ++i){
    printf("out %f\n", out[i]);

  }



    dlclose(handle);
    exit(EXIT_SUCCESS);

}
