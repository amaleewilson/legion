#include "cuda_runtime.h"
#include <stdio.h>
#include <string>
#include "terra.h"
#include <dlfcn.h>
#include <stdlib.h>

int main(int argc, char ** argv) { 
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
  C.fprintf(\"test]n\");";

    const char *st = s.c_str();

    if (terra_dostring(Lu, st)){
        printf("error\n"); 
    }

    printf("\nTEST\n");


}
