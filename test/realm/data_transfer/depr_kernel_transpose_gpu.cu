
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
__device__ void bp_copy(float *h_0, float *h_1, float *h_2, float *h_3, float *d_dst, size_type elem_size, 
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





// end base pointer versions 





////////// Batched versions 

/*
   Copies c_sz (should be 4, 8, 16, 32) elements without writing through shared memory.
   This version aims to align src mem reads.  
   TODO: Why does c_sz=12 not work? I think it is related to thread count being 32.
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_trans_multi_batch(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
  
  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base = 0;
  size_type src_base = 0;
  size_type loop_term = (elem_count*fid_count)/c_sz;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
  
    dst_base = t_id*c_sz;
    src_base = t_id*c_sz;

    #pragma unroll 
    for (size_type i = 0; i < c_sz; ++i){
      d_dst[((dst_base + i) - ((dst_base+i)/elem_count*elem_count))*fid_count + (dst_base + i)/elem_count] = h_src_A[src_base + i];
    }
  }
}

/*
   Copies c_sz (should be 4, 8, 16, 32) elements without writing through shared memory.
   This version aims to align dst mem writes.  
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_trans2_multi_batch(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, int fid_count, int c_sz) {
  
  // Actual thread id 
  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base = 0;
  size_type src_base = 0;
  size_type loop_term = (elem_count*fid_count)/c_sz;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
 
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
  
    src_base = t_id*c_sz;
    dst_base = t_id*c_sz;

    #pragma unroll 
    for (size_type i = 0; i < c_sz; ++i){
      //d_dst[dst_base + i] = h_src_A[elem_count*(i%fid_count) + (c_sz/fid_count)*(t_id*c_sz) + i/fid_count];
      d_dst[dst_base + i] = h_src_A[((src_base+i)/fid_count) + ((src_base + i)%fid_count)*elem_count];
    }
  }
}

/*
   Launches fewer threads than original trans1.
   This version aims to align src mem reads. 
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_trans1_batch(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, size_type fid_count) {

  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  size_type loop_term = elem_count*fid_count;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y; 
  
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
      d_dst[(t_id - ((t_id/elem_count)*elem_count))*fid_count + t_id/elem_count] = h_src_A[t_id];
  }
}

/*
   Launches fewer threads than original trans2.
   This version aims to align dst mem writes. 
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_trans2_batch(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, size_type fid_count) {

  size_type real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  size_type loop_term = elem_count*fid_count;
  size_type inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y; 
  
  for (size_type t_id = real_tid; t_id < loop_term; t_id += inc){
      d_dst[t_id] = h_src_A[(t_id/fid_count) + (t_id%fid_count)*elem_count];
  }
}

// end batched versions



////// non-batched shared memory versions

/*
   Copies c_sz (4 or 8) elements through shared memory, writes to d_dst should be coalesced. 
 */
template <int block_size, typename size_type, int c_sz>
__device__ void copykernelAoSsharedmulti(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, int fid_count) {

  size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);

  __shared__ float tmp_d_dst[block_size*c_sz];

  size_type tmp_base = t_id%block_size*c_sz; 
  size_type dst_base = t_id*c_sz;
   
#pragma unroll
  for (size_type i = 0; i < c_sz; ++i){
    tmp_d_dst[tmp_base + i] = h_src_A[elem_count*(i%fid_count) + (c_sz/fid_count)*t_id + i/fid_count];
    d_dst[dst_base + i] = tmp_d_dst[tmp_base + i];  
  }
  
  // This approach gave very bad performance wrt bandwidth. 
/*
    size_type tmp_base = t_id%block_size*c_sz; 
    size_type src_base = t_id*c_sz;
  
    for (size_type i = 0; i < c_sz; ++i){
      tmp_d_dst[tmp_base + i] = h_src_A[src_base + i];
    }
    for (size_type i = 0; i < c_sz; ++i){
      
      size_type s_idx = src_base + i;
      size_type offset = s_idx/elem_count;
      size_type d_idx = (s_idx - ((offset)*elem_count))*fid_count + offset;
      d_dst[d_idx] = tmp_d_dst[tmp_base + i];  
    }
*/

}

/*
   Copies one element per thread through shared memory, writes to d_dst should be coalesced. 
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_shared(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, size_type fid_count) {

  size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);

  __shared__ float tmp_d_dst[block_size];

  size_t t_idx = t_id % block_size;

  tmp_d_dst[t_idx] = h_src_A[(t_id/fid_count) + (t_id%fid_count)*elem_count];

  d_dst[t_id] = tmp_d_dst[t_idx];
}






// end non-batched shared memory versions



/*
   Copies c_sz (4 or 8) elements without writing through shared memory, writes to d_dst should be coalesced. 
 */
template <int block_size, typename size_type, int c_sz>
__device__ void copykernelAoS_trans_multi(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, int fid_count) {
    
  size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  
  size_type dst_base = t_id*c_sz;
   
#pragma unroll 
  for (size_type i = 0; i < c_sz; ++i){
    d_dst[dst_base + i] = h_src_A[elem_count*(i%fid_count) + (c_sz/fid_count)*t_id + i/fid_count];
  }
}

/*
   Copies one element per thread, writes to d_dst may or may not be coalesced. 
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_trans1(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, 
        size_type elem_count, size_type fid_count) {

  size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);

  d_dst[(t_id - (t_id/elem_count*elem_count))*fid_count + t_id/elem_count] = h_src_A[t_id];
    
}

/*
   Copies one element per thread, writes to d_dst should be coalesced. 
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_trans2(float *h_src_A, float *d_dst, size_type elem_count, size_type fid_count) {
    
  size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
    
  d_dst[t_id] = h_src_A[(t_id/fid_count) + (t_id%fid_count)*elem_count];
}

/*
   Simply copy each element over without changing the data layout.
 */
template <int block_size, typename size_type>
__device__ void copykernelAoS_no_trans(float *h_src_A, float *d_dst) {
  
    size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);

    size_type dst_idx = t_id; 
    //size_type dst_idx = bx*bdx + tx; 

    d_dst[dst_idx] = h_src_A[dst_idx];

}

/*
   To compare with one thread per element.
 */
template <int block_size, typename size_type, int c_sz>
__device__ void copykernelAoS_no_trans_multi(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, size_type elem_count) {

    size_type t_id = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
    
    for (size_type i = 0; i < c_sz; ++i){
      d_dst[t_id + i] = h_src_A[t_id + i];
    }
   
}



// C wrappers around our template kernel
extern "C" __global__ void bp_copy_32(float *h_0, float *h_1, float *h_2, float *h_3, float *d_dst,
                                                int e_size, int e_count, int fid_count, int c_sz) {

  bp_copy<32, int>(h_0, h_1, h_2, h_3, d_dst, e_size, e_count, fid_count, c_sz);
}


extern "C" __global__ void copykernelAoS_no_trans32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_no_trans<32, int>(h_src_A, d_dst);
}
extern "C" __global__ void copykernelAoS_trans232_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_trans2<32, int>(h_src_A, d_dst, e_count, fid_count);
}
extern "C" __global__ void copykernelAoS_no_trans_multi32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_no_trans_multi<32, int, 8>(h_src_A, h_src_B, d_dst, e_size, e_count);
}
extern "C" __global__ void copykernelAoS_trans2_multi_batch32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count, int c_sz) {
  copykernelAoS_trans2_multi_batch<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count, c_sz);
}
extern "C" __global__ void copykernelAoS_trans_multi_batch32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count, int c_sz) {
  copykernelAoS_trans_multi_batch<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count, c_sz);
}
extern "C" __global__ void copykernelAoS_trans_multi32_32bit_8(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_trans_multi<32, int, 8/*copy_size*/>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoS_trans_multi32_32bit_4(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_trans_multi<32, int, 4/*copy_size*/>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoSsharedmulti32_32bit_8(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoSsharedmulti<32, int, 8/*copy_size*/>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoSsharedmulti32_32bit_4(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoSsharedmulti<32, int, 4/*copy_size*/>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoS_shared32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_shared<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoS_trans2_batch32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_trans2_batch<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoS_trans1_batch32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_trans1_batch<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
extern "C" __global__ void copykernelAoS_trans132_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int e_size, int e_count, int fid_count) {
  copykernelAoS_trans1<32, int>(h_src_A, h_src_B, d_dst, e_size, e_count, fid_count);
}
#endif  // #ifndef _COPY_KERNEL_H_
