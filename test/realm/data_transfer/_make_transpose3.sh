

/projects/opt/centos7/cuda/9.1/bin/nvcc -ccbin g++ -I../../Common  -m64    -gencode arch=compute_30,code=compute_30 -o transpose3_gpu64.ptx -ptx transpose3_gpu.cu
 
/projects/opt/centos7/cuda/9.1/bin/nvcc -ccbin g++ -I./Common  -I/home/amaleewilson/legion/runtime -I/home/amaleewilson/legion/runtime/mappers -I/projects/opt/centos7/cuda/9.1/include -I/home/amaleewilson/legion/runtime/realm/transfer  -m64    -gencode arch=compute_30,code=compute_30 -o transpose3.o -c transpose3.cc 

/projects/opt/centos7/cuda/9.1/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_30,code=compute_30 -o transpose3 transpose3.o  -L/projects/opt/centos7/cuda/9.1/lib64/stubs -lcuda -L. -lrealm  -I/home/amaleewilson/legion/runtime -I/home/amaleewilson/legion/runtime/mappers -I/home/amaleewilson/legion/runtime/realm/transfer -DUSE_DISK -DUSE_LIBDL -DUSE_CUDA -DUSE_ZLIB -DDEBUG_REALM -DDEBUG_LEGION -DCOMPILE_TIME_MIN_LEVEL=LEVEL_DEBUG  -lrt -lpthread



 
 

