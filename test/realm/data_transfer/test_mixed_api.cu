#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <iostream>
#include <math.h>

#define NX 3200000 
#define NY 4
//const int NX = 4;
//const int NY = 4;

const int DEFAULT_FFT_TRIALS = 1000;
const int DEFAULT_META_TRIALS = 10;

/* Transpose methods needed some pretty significant changes.
 * Need to figure the best way to account for this in codegen.
 * e.g. specifying types, size, etc...
 * Are real and imag next to each other in mem? 
 */ 

__global__ void bp_soa_to_aos(cufftComplex *d_A, cufftComplex *d_B,
        int elem_count, int fid_count, int c_sz) {
  int real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);
  int dst_base = 0;
  int src_base0 = 0;
  int src_base1 = elem_count;
  int src_base2 = elem_count*2;
  int src_base3 = elem_count*3;
  int loop_term = (elem_count*fid_count)/c_sz;
  int inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  int lt = c_sz / fid_count; 
  for (int t_id = real_tid; t_id < loop_term; t_id += inc){
  
    dst_base = t_id*c_sz;

    #pragma unroll 
    for (int i = 0; i < lt; ++i){
      d_B[dst_base + 0 + i*fid_count].x = d_A[src_base0 + t_id*lt + i].x;
      d_B[dst_base + 0 + i*fid_count].y = d_A[src_base0 + t_id*lt + i].y;
      d_B[dst_base + 1 + i*fid_count].x = d_A[src_base1 + t_id*lt + i].x;
      d_B[dst_base + 1 + i*fid_count].y = d_A[src_base1 + t_id*lt + i].y;
      d_B[dst_base + 2 + i*fid_count].x = d_A[src_base2 + t_id*lt + i].x;
      d_B[dst_base + 2 + i*fid_count].y = d_A[src_base2 + t_id*lt + i].y;
      d_B[dst_base + 3 + i*fid_count].x = d_A[src_base3 + t_id*lt + i].x;
      d_B[dst_base + 3 + i*fid_count].y = d_A[src_base3 + t_id*lt + i].y;
    }
  }
}

__global__ void bp_aos_to_soa(cufftComplex *d_A, cufftComplex *d_B, 
        int elem_count, int fid_count, int c_sz) {
  int real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);

  int dst_base0 = 0;
  int dst_base1 = elem_count;
  int dst_base2 = elem_count*2;
  int dst_base3 = elem_count*3;
  int lt = c_sz / fid_count; 
  int loop_term = elem_count/lt;
  int inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  int src_base = 0;
  for (int t_id = real_tid; t_id < loop_term; t_id += inc){
 
    src_base = t_id*c_sz;
    #pragma unroll 
    for (int i = 0; i < lt; ++i){
      d_B[dst_base0 + t_id*lt + i].x = d_A[src_base + 0 + i*fid_count].x;
      d_B[dst_base0 + t_id*lt + i].y = d_A[src_base + 0 + i*fid_count].y;
      d_B[dst_base1 + t_id*lt + i].x = d_A[src_base + 1 + i*fid_count].x;
      d_B[dst_base1 + t_id*lt + i].y = d_A[src_base + 1 + i*fid_count].y;
      d_B[dst_base2 + t_id*lt + i].x = d_A[src_base + 2 + i*fid_count].x;
      d_B[dst_base2 + t_id*lt + i].y = d_A[src_base + 2 + i*fid_count].y;
      d_B[dst_base3 + t_id*lt + i].x = d_A[src_base + 3 + i*fid_count].x;
      d_B[dst_base3 + t_id*lt + i].y = d_A[src_base + 3 + i*fid_count].y;
    }
  }
}

int main(int argc, char **argv) {
    int fft_trials = DEFAULT_FFT_TRIALS;
    int meta_trials = DEFAULT_META_TRIALS;

    printf("[INFO] META trials: %d\n", meta_trials);
    printf("[INFO] FFT trials: %d\n", fft_trials);

    int nx = NX;
    int ny = NY;
    printf("[INFO] NX Length: %d\n", nx);
    printf("[INFO] NY Length: %d\n", ny);

    cufftComplex *h_M, *h_M2, *h_M3;
    cudaMallocHost((void **) &h_M, sizeof(cufftComplex) * NX * NY);
    cudaMallocHost((void **) &h_M2, sizeof(cufftComplex) * NX * NY);
    cudaMallocHost((void **) &h_M3, sizeof(cufftComplex) * NX * NY);

    cufftComplex *d_M, *d_M1, *d_M2, *d_M3, *d_M4, *d_M5;
    cudaMalloc((void **) &d_M, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_M1, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_M2, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_M3, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_M4, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_M5, sizeof(cufftComplex) * NX * NY);




    /*
     * random signal, ints to be easy to read original
     */
    srand(0); // initialize random seed
    for (int i = 0; i < NX*NY; i++) {
        h_M[i].x = (int)((float)rand()) % 10;
        h_M[i].y = 0.0;
    }

    cudaMemcpy(d_M, h_M, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M1, h_M, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);

    cufftHandle fft_plan2D;
    cufftPlan2d(&fft_plan2D, nx, ny, CUFFT_C2C);

    cufftHandle fft_plan1D1D;
    int *n = new int[1];
    n[0] = nx;
    int inembed_pre = nx;
    int *inembed = &inembed_pre;
    int istride = ny;
    int idist = 1;
    cufftPlanMany(&fft_plan1D1D, 1, n, inembed, istride, idist, inembed, istride, idist, CUFFT_C2C, ny);
    
    cufftHandle fft_plan1D1DR;
    int *nR = new int[1];
    nR[0] = ny;
    int inembed_preR = ny;
    int *inembedR = &inembed_preR;
    int istrideR = 1;
    int idistR = ny;
    cufftPlanMany(&fft_plan1D1DR, 1, nR, inembedR, istrideR, idistR, inembedR, istrideR, idistR, CUFFT_C2C, nx);
   

    cufftHandle fft_plan1D1DR2;
    int *nR2 = new int[1];
    nR2[0] = nx;
    int inembed_preR2 = nx;
    int *inembedR2 = &inembed_preR2;
    int istrideR2 = 1;
    int idistR2 = nx;
    cufftPlanMany(&fft_plan1D1DR2, 1, nR2, inembedR2, istrideR2, idistR2, inembedR2, istrideR2, idistR2, CUFFT_C2C, ny);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum_of_elapsed_times2D = 0.0;
    float sum_of_elapsed_times1D1D = 0.0;

    
    printf("[INFO] Run benchmark 1D1D\n");
    for (int i = 0; i < meta_trials; i++) {
        cudaEventRecord(start, 0);

        for (int j = 0; j < fft_trials; j++) {
            

// 1D on the columns
//          cufftExecC2C(fft_plan1D1D, d_M, d_M5, CUFFT_FORWARD);
          cufftExecC2C(fft_plan1D1D, d_M, d_M5, CUFFT_FORWARD);

//transpose and 1D on the "colms" which are now rows, transpose back
          bp_aos_to_soa<<<500,32>>>((cufftComplex*)d_M1, (cufftComplex*)d_M2, NX, NY, NY);
          cufftExecC2C(fft_plan1D1DR2, d_M2, d_M3, CUFFT_FORWARD);
          bp_soa_to_aos<<<500,32>>>((cufftComplex*)d_M3, (cufftComplex*)d_M4, NX, NY, NY);

        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        float elapsed_time_sec = elapsed_time_ms / 1000.0;
        sum_of_elapsed_times1D1D += elapsed_time_sec;
        printf("%f sec\n", elapsed_time_sec);
    }


    //h_M3 is the 1D only result 
    cudaMemcpy(h_M3, d_M5, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);
    //h_M2 is the T 1D T version 
    cudaMemcpy(h_M2, d_M4, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);


    bool correct = true;
    for (int i = 0; i < NX; ++i){
      for (int j = 0; j < NY; ++j){
//        std::cout << h_M3[i*NY +j].x << " ";
//        std::cout << h_M2[i*NY +j].x << " ";
        if (fabs(h_M3[i*NY +j].x - h_M2[i*NY +j].x) > 1e-1 || fabs(h_M3[i*NY +j].y - h_M2[i*NY +j].y) > 1e-1){
          correct = false;
        }
      }
//      std::cout << std::endl;
    }

    std::cout << "correct result? " << (correct ? "yes" : "no") << std::endl;

    printf("[INFO] Finished!\n");
    printf("[INFO] Average 2D: %lf sec\n", sum_of_elapsed_times2D / meta_trials);
    printf("[INFO] Average 1D1D: %lf sec\n", sum_of_elapsed_times1D1D / meta_trials);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



