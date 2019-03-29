
#ifndef _COPY_KERNEL_H_
#define _COPY_KERNEL_H_

#include <stdio.h>

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

#define ELEM_SIZE 2048

template <int block_size, typename size_type>
__device__ void copykernelAoS_shared(float *h_src_A, float *h_src_B, float *d_dst, size_type elem_size, size_type elem_count) {
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
    //size_type dst_idx = bx*bdx + tx; 


    size_type dst_idx = tid*2;
    // this can be made faster, you should not num_elems threads
    
    d_dst[dst_idx] = h_src_A[dst_idx/2];
    d_dst[dst_idx + 1] = h_src_B[dst_idx/2];

    /*
   //d_dst[dst_idx] = h_src_A[dst_idx]; 
    if (dst_idx % 2 == 0){
        d_dst[dst_idx] = h_src_A[dst_idx/2];
        //tmp_d_dst[dst_idx] = h_src_A[dst_idx/2];
        // May be worth seeing if accessing host memory differently will improve performance
        //d_dst[dst_idx/2] = h_src[dst_idx];
    }
    else{
        // May be worth seeing if accessing host memory differently will improve performance
        
        // these have the same perf
        d_dst[dst_idx] = h_src_B[dst_idx/2];
        //d_dst[dst_idx] = h_src_A[elem_count + dst_idx/2];
        
        //tmp_d_dst[dst_idx] = h_src_B[dst_idx/2];
        //d_dst[dst_idx] = h_src_B[dst_idx/2 + elem_count/2];
        //d_dst[dst_idx/2 + elem_count/2] = h_src[dst_idx];
    }
    */
//    d_dst[0] = tmp_d_dst[0];

    
    //d_dst[(bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx] = h_src[(bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx];

}


// This kernel is copying everything over and rearranging the data from AoS to SoA layout. 
// TODO: calculate bandwidth!!
template <int block_size, typename size_type>
__device__ void copykernelAoS(float *h_src, float *d_dst, size_type elem_size, size_type elem_count) {
  // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    // tid used to idx into dst
    // 0, 1, 2, ... elem_count - 1
    // one thread per element (consider one thread per chunk of elems in the aos struct)
    
    /*
    size_type dst_idx = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 

    if (dst_idx % 2 == 0){
        d_dst[dst_idx] = h_src[dst_idx/2];
    }
    else{
        d_dst[dst_idx] = h_src[dst_idx/2 + elem_count/2];
    }

    */

    size_type dst_idx = (bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx; 

    if (dst_idx % 2 == 0){
        d_dst[dst_idx] = h_src[dst_idx/2];
        // May be worth seeing if accessing host memory differently will improve performance
        //d_dst[dst_idx/2] = h_src[dst_idx];
    }
    else{
        // May be worth seeing if accessing host memory differently will improve performance
        d_dst[dst_idx] = h_src[dst_idx/2 + elem_count/2];
        //d_dst[dst_idx/2 + elem_count/2] = h_src[dst_idx];
    }


    //d_dst[(bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx] = h_src[(bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx];

}

//This kernel is copying everything over.
template <int block_size, typename size_type>
__device__ void copykernel(float *h_src, float *d_dst, size_type elem_size, size_type elem_count) {
  // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

    size_type bdx = blockDim.x;
    size_type bdy = blockDim.y;

    //for (size_type i = block_size * bx + tx; i < elem_count; i += gridDim.x * block_size){
    //    d_dst[i] = h_src[i];
        //cudaMemcpy(&(d_dst[i]), &(h_src), elem_size);
    //}
    
    //memcpy(&(d_dst[block_size*bx + tx]), &(h_src[block_size*bx + tx]), gridDim.x *block_size);
    d_dst[(bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx] = h_src[(bx + by*gridDim.x) * (bdx*bdy) + (ty*bdx) + tx];


    //todo: calculate the bandwidth. 
}

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
template <int block_size, typename size_type>
__device__ void matrixMul(float *C, float *A, float *B, size_type wA,
                          size_type wB) {
  // Block index
  size_type bx = blockIdx.x;
  size_type by = blockIdx.y;

  // Thread index
  size_type tx = threadIdx.x;
  size_type ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  size_type aBegin = wA * block_size * by;

  // Index of the last sub-matrix of A processed by the block
  size_type aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  size_type aStep = block_size;

  // Index of the first sub-matrix of B processed by the block
  size_type bBegin = block_size * bx;

  // Step size used to iterate through the sub-matrices of B
  size_type bStep = block_size * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (size_type a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[block_size][block_size];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[block_size][block_size];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    AS(ty, tx) = A[a + wA * ty + tx];
    BS(ty, tx) = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (size_type k = 0; k < block_size; ++k) Csub += AS(ty, k) * BS(k, tx);

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  size_type c = wB * block_size * by + block_size * bx;
  C[c + wB * ty + tx] = Csub;
}

// C wrappers around our template kernel
//extern "C" __global__ void matrixMul_bs16_32bit(float *C, float *A, float *B,
//                                                int wA, int wB) {
//  matrixMul<16, int>(C, A, B, wA, wB);
//}
//extern "C" __global__ void matrixMul_bs16_64bit(float *C, float *A, float *B,
//                                                size_t wA, size_t wB) {
//  matrixMul<16, size_t>(C, A, B, wA, wB);
//}
extern "C" __global__ void copykernel32_32bit(float *h_src, float *d_dst,
                                                int wA, int wB) {
  copykernel<32, int>(h_src, d_dst, wA, wB);
}
extern "C" __global__ void copykernelAoS_shared32_32bit(float *h_src_A, float *h_src_B, float *d_dst,
                                                int wA, int wB) {
  copykernelAoS_shared<32, int>(h_src_A, h_src_B, d_dst, wA, wB);
}
extern "C" __global__ void copykernelAoS32_32bit(float *h_src, float *d_dst,
                                                int wA, int wB) {
  copykernelAoS<32, int>(h_src, d_dst, wA, wB);
}
extern "C" __global__ void matrixMul_bs32_32bit(float *C, float *A, float *B,
                                                int wA, int wB) {
  matrixMul<32, int>(C, A, B, wA, wB);
}
//extern "C" __global__ void matrixMul_bs32_64bit(float *C, float *A, float *B,
//                                                size_t wA, size_t wB) {
//  matrixMul<32, size_t>(C, A, B, wA, wB);
//}

#endif  // #ifndef _COPY_KERNEL_H_
