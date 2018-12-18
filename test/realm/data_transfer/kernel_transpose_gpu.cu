
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




////////// Base pointer versions


/*
 */
template <int block_size, typename size_type>
__device__ void bp_soa_to_aos(float *h_0, float *h_1, float *h_2, float *h_3, float *d_dst, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 

  // TODO: remove this later
  //c_sz = 8;

  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base = 0;
  size_type loop_term = (elem_count*fid_count)/c_sz;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  size_type lt = c_sz / fid_count; 
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
  
    dst_base = t_id*c_sz;
    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
        
      d_dst[dst_base + 0 + i*fid_count] = h_0[t_id + i];
      d_dst[dst_base + 1 + i*fid_count] = h_1[t_id + i];
      d_dst[dst_base + 2 + i*fid_count] = h_2[t_id + i];
      d_dst[dst_base + 3 + i*fid_count] = h_3[t_id + i];
    }
  }
}


/*
 */
// TODO: this method is direct copy of above and a work in progress. 
template <int block_size, typename size_type>
__device__ void bp_copy_var(float *h_0, float *h_1, float *h_2, float *h_3, float *d_dst, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 

  // TODO: remove this later
  //c_sz = 8;

  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base = 0;
  size_type loop_term = (elem_count*fid_count)/c_sz;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  size_type lt = c_sz / fid_count; 
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
  
    dst_base = t_id*c_sz;
    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
      d_dst[dst_base + 0 + i*fid_count] = h_0[t_id + i];
      d_dst[dst_base + 1 + i*fid_count] = h_1[t_id + i];
      d_dst[dst_base + 2 + i*fid_count] = h_2[t_id + i];
      d_dst[dst_base + 3 + i*fid_count] = h_3[t_id + i];
    }
  }
}



// C wrappers around our template kernel
extern "C" __global__ void bp_soa_to_aos(float *h_0, float *h_1, float *h_2, float *h_3, float *d_dst,
                                                int e_size, int e_count, int fid_count, int c_sz) {

  bp_soa_to_aos<32, int>(h_0, h_1, h_2, h_3, d_dst, e_size, e_count, fid_count, c_sz);
}

#endif  // #ifndef _COPY_KERNEL_H_
