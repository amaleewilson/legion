#include <stdio.h>
#include <string>
#include "terra.h"

int main(int argc, char ** argv) {
    lua_State * L = luaL_newstate(); //create a plain lua state
    luaL_openlibs(L);                //initialize its libraries
    //initialize the terra state in lua
    terra_init(L);

    std::string s = "print (\"hello\")";
    const char *st = s.c_str();

    if (terra_dostring(L, st)){
        printf("error\n"); 
    }
    return 0;
}
