
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

// C wrappers around our template kernel
extern "C" __global__ void copykernelAoS32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count) {
  copykernelAoS<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count);
}
#endif  // #ifndef _COPY_KERNEL_H_
