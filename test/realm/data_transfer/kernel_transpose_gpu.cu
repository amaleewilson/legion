
#ifndef _COPY_KERNEL_H_
#define _COPY_KERNEL_H_

#include <stdio.h>


// Keeping this bank conflict ctuff for reference later
/*
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) \
  cutilBankChecker((reinterpret_cast<float *>(&As[0][0])), (block_size * i + j))
#define BS(i, j) \
  cutilBankChecker((reinterpret_cast<float *>(&Bs[0][0])), (block_size * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
*/

template <int block_size, typename size_type>
__device__ void copykernelAoSshared(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, size_type elem_count) {
    // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    // want ptr to start of each field id. 

    __shared__ float tmp_d_dst[block_size];
  //  tmp_d_dst[0] = 22.3;

    size_type dst_idx = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 
    size_t t_idx = dst_idx % block_size;
    //size_type dst_idx = bx*bdx + tx; 


    //getting same perf for one elem per thread and for 2 elem per thread,
   //but 8 elem per thread slows it down a lot. 
    
    if (dst_idx % 2 == 0){
        //d_dst[dst_idx] = h_src_A[dst_idx/2];
        // May be worth seeing if accessing host memory differently will improve performance
        tmp_d_dst[t_idx] = h_src_A[dst_idx/2];
    }
    else{
        //d_dst[dst_idx] = h_src_B[dst_idx/2];
        tmp_d_dst[t_idx] = h_src_B[dst_idx/2];
    }

    __syncthreads();

    d_dst[dst_idx] = tmp_d_dst[t_idx];
}

template <int block_size, typename size_type>
__device__ void copykernelAoS(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, size_type elem_count) {
    // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    // want ptr to start of each field id. 

    //__shared__ float tmp_d_dst[ELEM_SIZE];

  //  tmp_d_dst[0] = 22.3;

    size_type dst_idx = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 
    //size_type dst_idx = bx*bdx + tx; 


    //getting same perf for one elem per thread and for 2 elem per thread,
   //but 8 elem per thread slows it down a lot. 
    
    if (dst_idx % 2 == 0){
        d_dst[dst_idx] = h_src_A[dst_idx/2];
        // May be worth seeing if accessing host memory differently will improve performance
    }
    else{
        d_dst[dst_idx] = h_src_B[dst_idx/2];
    }
}

template <int block_size, typename size_type>
__device__ void copykernelAoS2(float *h_src_A, float *d_dst, size_type elem_count) {
    // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    // want ptr to start of each field id. 

    //__shared__ float tmp_d_dst[ELEM_SIZE];

  //  tmp_d_dst[0] = 22.3;

    size_type dst_idx = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 
    //size_type dst_idx = bx*bdx + tx; 

    if (dst_idx < elem_count){
        d_dst[2*dst_idx] = h_src_A[dst_idx];
    }
    else{
        d_dst[2*(dst_idx-elem_count) + 1] = h_src_A[dst_idx];
    }

    //getting same perf for one elem per thread and for 2 elem per thread,
   //but 8 elem per thread slows it down a lot. 
}

template <int block_size, typename size_type>
__device__ void copykernelAoSbasic(float *h_src_A, float *d_dst) {
    // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    // want ptr to start of each field id. 

    //__shared__ float tmp_d_dst[ELEM_SIZE];

  //  tmp_d_dst[0] = 22.3;

    size_type dst_idx = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 
    //size_type dst_idx = bx*bdx + tx; 

    d_dst[dst_idx] = h_src_A[dst_idx];

}
// This seems to be broken for large inputs. 
template <int block_size, typename size_type>
__device__ void copykernelAoSmulti(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, size_type elem_count) {
    // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    // want ptr to start of each field id. 

    //__shared__ float tmp_d_dst[ELEM_SIZE];

  //  tmp_d_dst[0] = 22.3;
    size_type tid = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 
    
    //size_type dst_idx = tid*8;
    size_type dst_idx = tid*2;
   
    //size_type dst_idx = bx*bdx + tx; 
    d_dst[dst_idx] = h_src_A[dst_idx/2];
    d_dst[dst_idx + 1] = h_src_B[dst_idx/2];

}

// C wrappers around our template kernel
extern "C" __global__ void copykernelAoSbasic32_32bit(float *h_src_A, float *d_dst) {
  copykernelAoSbasic<32, int>(h_src_A, d_dst);
}
extern "C" __global__ void copykernelAoS232_32bit(float *h_src_A, float *d_dst,
                                                int e_count) {
  copykernelAoS2<32, int>(h_src_A, d_dst, e_count);
}
extern "C" __global__ void copykernelAoSmulti32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count) {
  copykernelAoSmulti<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count);
}
extern "C" __global__ void copykernelAoSshared32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count) {
  copykernelAoSshared<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count);
}
extern "C" __global__ void copykernelAoS32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count) {
  copykernelAoS<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count);
}
#endif  // #ifndef _COPY_KERNEL_H_
