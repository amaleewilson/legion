
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
__device__ void bp_aos_to_aos_test(float *d_A, float *d_B, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 

  // TODO: remove this later
  //c_sz = 8;

  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base = 0;
  size_type lt = c_sz / fid_count; 
  size_type loop_term = elem_count/lt;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  size_type src_base = 0;
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
 
    src_base = t_id*c_sz;
    dst_base = t_id*c_sz;
    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
      d_B[dst_base + 0 + i*fid_count] = d_A[src_base + 0 + i*fid_count];
      d_B[dst_base + 1 + i*fid_count] = d_A[src_base + 1 + i*fid_count];
      d_B[dst_base + 2 + i*fid_count] = d_A[src_base + 2 + i*fid_count];
      d_B[dst_base + 3 + i*fid_count] = d_A[src_base + 3 + i*fid_count];
    }
  }
}
template <int block_size, typename size_type>
__device__ void bp_soa_to_soa_test(float *d_A, float *d_B, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {

  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
 
  size_type dst_base0 = 0;
  size_type dst_base1 = elem_count;
  size_type dst_base2 = elem_count*2;
  size_type dst_base3 = elem_count*3;
  
  size_type src_base0 = 0;
  size_type src_base1 = elem_count;
  size_type src_base2 = elem_count*2;
  size_type src_base3 = elem_count*3;

  size_type loop_term = (elem_count*fid_count)/c_sz;

  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;

  size_type lt = c_sz / fid_count; 

  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
  

    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
      d_B[dst_base0 + t_id*lt + i] = d_A[src_base0 + t_id*lt + i];
      d_B[dst_base1 + t_id*lt + i] = d_A[src_base1 + t_id*lt + i];
      d_B[dst_base2 + t_id*lt + i] = d_A[src_base2 + t_id*lt + i];
      d_B[dst_base3 + t_id*lt + i] = d_A[src_base3 + t_id*lt + i];
    }
  }
}

/*
h_0, h_1, h_2, h_3: base pointers for the host data.
d_dst: pointer to device memory. 
elem_size: unused artefact, will be removed. 
elem_count: the number of elements per field.
fid_count: the number of fields.
c_sz: the total number of elements (e.g. 8 means 2 elements per 4 fields) to be copied per thread. 
 */
template <int block_size, typename size_type>
__device__ void bp_soa_to_aos(float *d_A, float *d_B, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 

  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
 
  // Destination is array of struct layout.
  // Memory is assumed to be contiguous on device.  
  size_type dst_base = 0;
  size_type src_base0 = 0;
  size_type src_base1 = elem_count;
  size_type src_base2 = elem_count*2;
  size_type src_base3 = elem_count*3;

  // loop_term: terminating condition for the outer loop,
  // equal to the total number of elements divided by the number of elements copied per thread. 
  size_type loop_term = (elem_count*fid_count)/c_sz;

  // inc: increments outer loop by the total number of threads that are available.
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;

  // lt: inner loop terminating condition. 
  size_type lt = c_sz / fid_count; 

  // Iterate over all available threads, t_id is the logical thread id for indexing into the arrays. 
//#pragma unroll
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
  
    // Each thread copies a total of c_sz elements, and d_dst is assumed contiguous and 
    // will be AoS layout, so the logical t_id times c_sz results in the correct base 
    // for the iteration of the inner loop. 
    dst_base = t_id*c_sz;

    // Since lt is the number of full fid_count-sized elements being copied, 
    // t_id*lt gives the correct base index for each field in the host subarrays.

    // This kernel is hard coded to handle an instance with 4 fields.
    // Some of the arithmetic could probably be removed from this loop, 
    // but that's not likely to be the limiting factor for performance. 
    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
      d_B[dst_base + 0 + i*fid_count] = d_A[src_base0 + t_id*lt + i];
      d_B[dst_base + 1 + i*fid_count] = d_A[src_base1 + t_id*lt + i];
      d_B[dst_base + 2 + i*fid_count] = d_A[src_base2 + t_id*lt + i];
      d_B[dst_base + 3 + i*fid_count] = d_A[src_base3 + t_id*lt + i];
    }
  }
}

/*
   Same as above, except for going from array of struct on cpu to struct of array on GPU. 
 */
template <int block_size, typename size_type>
__device__ void bp_aos_to_soa(float *d_A, float *d_B, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 

  // TODO: remove this later
  //c_sz = 8;

  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base0 = 0;
  size_type dst_base1 = elem_count;
  size_type dst_base2 = elem_count*2;
  size_type dst_base3 = elem_count*3;
  size_type lt = c_sz / fid_count; 
  size_type loop_term = elem_count/lt;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  size_type src_base = 0;
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
 
    src_base = t_id*c_sz;
    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
      d_B[dst_base0 + t_id*lt + i] = d_A[src_base + 0 + i*fid_count];
      d_B[dst_base1 + t_id*lt + i] = d_A[src_base + 1 + i*fid_count];
      d_B[dst_base2 + t_id*lt + i] = d_A[src_base + 2 + i*fid_count];
      d_B[dst_base3 + t_id*lt + i] = d_A[src_base + 3 + i*fid_count];
    }
  }
}





template <int block_size, typename size_type>
__device__ void bp_aos_to_soa_single(float *d_A, float *d_B, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 

  // TODO: remove this later
  //c_sz = 8;

  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base0 = 0;
  size_type dst_base1 = elem_count;
  size_type dst_base2 = elem_count*2;
  size_type dst_base3 = elem_count*3;
  size_type lt = c_sz / fid_count; 
  size_type loop_term = elem_count/lt;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  size_type src_base = 0;
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
 
    src_base = t_id*c_sz;
    #pragma unroll 
    for (size_type i = 0; i < lt; ++i){
      d_B[dst_base0 + t_id*lt + i] = d_A[src_base + 0 + i*fid_count];
      d_B[dst_base1 + t_id*lt + i] = d_A[src_base + 1 + i*fid_count];
      d_B[dst_base2 + t_id*lt + i] = d_A[src_base + 2 + i*fid_count];
      d_B[dst_base3 + t_id*lt + i] = d_A[src_base + 3 + i*fid_count];
    }
  }
}

template <int block_size, typename size_type>
__device__ void bp_soa_to_aos_single(float *d_A, float *d_B, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
 
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;

  size_type of = (real_tid % fid_count); 
  size_type offset = of*elem_count;  
  
  size_type loop_term = elem_count*fid_count;  

  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
    d_B[t_id] = d_A[offset + t_id/4];
  }
      //d_B[dst_base + 0 + i*fid_count] = d_A[src_base0 + t_id*lt + i];

}




// C wrappers around our template kernel
extern "C" __global__ void bp_soa_to_aos_single(float *d_A, float *d_B, 
                                                int e_size, int e_count, int fid_count, int c_sz) {
  bp_soa_to_aos_single<32, int>(d_A, d_B, e_size, e_count, fid_count, c_sz);
}

extern "C" __global__ void bp_aos_to_soa_single(float *d_A, float *d_B,
                                                int e_size, int e_count, int fid_count, int c_sz) {
  bp_aos_to_soa_single<32, int>(d_A, d_B, e_size, e_count, fid_count, c_sz);
}

extern "C" __global__ void bp_soa_to_aos(float *d_A, float *d_B, 
                                                int e_size, int e_count, int fid_count, int c_sz) {
  bp_soa_to_aos<32, int>(d_A, d_B, e_size, e_count, fid_count, c_sz);
}

extern "C" __global__ void bp_aos_to_soa(float *d_A, float *d_B,
                                                int e_size, int e_count, int fid_count, int c_sz) {
  bp_aos_to_soa<32, int>(d_A, d_B, e_size, e_count, fid_count, c_sz);
}

extern "C" __global__ void bp_aos_to_aos_test(float *d_A, float *d_B,
                                                int e_size, int e_count, int fid_count, int c_sz) {
  bp_aos_to_aos_test<32, int>(d_A, d_B, e_size, e_count, fid_count, c_sz);
}
extern "C" __global__ void bp_soa_to_soa_test(float *d_A, float *d_B,
                                                int e_size, int e_count, int fid_count, int c_sz) {
  bp_soa_to_soa_test<32, int>(d_A, d_B, e_size, e_count, fid_count, c_sz);
}

#endif  // #ifndef _COPY_KERNEL_H_
