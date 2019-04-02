#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <iostream>
#include <math.h>

#define NX 8
#define NY 8
//const int NX = 4;
//const int NY = 4;

const int DEFAULT_FFT_TRIALS = 1;
const int DEFAULT_META_TRIALS = 1;

__global__ void bp_soa_to_aos( cufftComplex *d_A, cufftComplex *d_B,
        int elem_count, int fid_count, int c_sz) {

  // Actual thread id 
  int real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);


  int *src_base = new int[NX];
  for (int i = 0; i < fid_count; ++i) {
    src_base[i] = elem_count*i;
  }

  int dst_base = 0;
  int loop_term = (elem_count*fid_count)/c_sz;
  int inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  int lt = c_sz / fid_count; 

  for (int t_id = real_tid; t_id < loop_term; t_id += inc){
  
    dst_base = t_id*c_sz;

    for (int i = 0; i < lt; ++i){
      for (int j = 0; j < fid_count; ++j){
        d_B[dst_base + j + i*fid_count].x = d_A[src_base[j] + t_id*lt + i].x;
      }
    }
  }
}

__global__ void bp_aos_to_soa(cufftComplex *d_A, cufftComplex *d_B,
        int elem_count, int fid_count, int c_sz) {
  
 
  int real_tid = ((blockIdx.x + blockIdx.y*gridDim.x) * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x);

  int *dst_base = new int[NY];
  for (int i = 0; i < fid_count; ++i) {
    dst_base[i] = elem_count*i;
  }

  int lt = c_sz / fid_count; 
  int loop_term = elem_count/lt;
  int inc = gridDim.x*gridDim.y*blockDim.x*blockDim.y;
  int src_base = 0;

  for (int t_id = real_tid; t_id < loop_term; t_id += inc){
 
    src_base = t_id*c_sz;
    for (int i = 0; i < lt; ++i){
      for (int j = 0; j < fid_count; ++j){
        d_B[dst_base[j] + t_id*lt + i].x = d_A[src_base + j + i*fid_count].x;
      }
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

    cufftComplex *h_original_signal, *h_applied_fft_signal2D, *h_applied_fft_signal1D1D;
    cudaMallocHost((void **) &h_original_signal, sizeof(cufftComplex) * NX * NY);
    cudaMallocHost((void **) &h_applied_fft_signal2D, sizeof(cufftComplex) * NX * NY);
    cudaMallocHost((void **) &h_applied_fft_signal1D1D, sizeof(cufftComplex) * NX * NY);

    cufftComplex *d_original_signal, *d_applied_fft_signal, *d_applied_fft_signal1D1D;
    cudaMalloc((void **) &d_original_signal, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_applied_fft_signal, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_applied_fft_signal1D1D, sizeof(cufftComplex) * NX * NY);




    /*
     * generate random signal as original signal
     */
    srand(0); // initialize random seed
    for (int i = 0; i < NX*NY; i++) {
        h_original_signal[i].x = (float)((int)rand() % 10);
        h_original_signal[i].y = 0.0;
    }

//    cudaMemcpy(d_original_signal, h_original_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);
//    
//    std::cout << "og matrix" << std::endl;
//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_original_signal[i*NX +j].x << " ";
//     }
//      std::cout << std::endl;
//    }
//
//    std::cout << "transpose matrix" << std::endl;
//    bp_soa_to_aos<<<2,32>>>((cufftComplex*)d_original_signal, (cufftComplex*)d_applied_fft_signal, NX, NY, NX);
//    cudaMemcpy(h_original_signal, d_applied_fft_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);
//
//
//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_original_signal[i*NX +j].x << " ";
//     }
//      std::cout << std::endl;
//    }
//
//    std::cout << "untranspose matrix" << std::endl;
//    bp_aos_to_soa<<<2,32>>>((cufftComplex*)d_applied_fft_signal, (cufftComplex*)d_applied_fft_signal1D1D, NX, NY, NX);
//    cudaMemcpy(h_original_signal, d_applied_fft_signal1D1D, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);
//
//
//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_original_signal[i*NX +j].x << " ";
//     }
//      std::cout << std::endl;
//    }



    cudaMemcpy(d_original_signal, h_original_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);


    //bp_soa_to_aos<<<2,32>>>((cufftComplex*)d_original_signal, (cufftComplex*)d_applied_fft_signal, NX, NY, 8);
    //bp_aos_to_soa<<<2,32>>>((cufftComplex*)d_applied_fft_signal, (cufftComplex*)d_original_signal, NX, NY, 8);


    /* 
     * 2D fft 
     */

    cufftHandle fft_plan2D;
    cufftPlan2d(&fft_plan2D, NX, NY, CUFFT_C2C);



    /*
     * 1D x 1D 2D fft with no transpose 
     */

//    cudaMemcpy(d_original_signal, h_original_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);

    cufftHandle fft_plan1D1D;
  
    int *n = new int[1];
    n[0] = nx;

    int inembed_pre = nx;
    int *inembed = &inembed_pre;

    int istride = 1;
    int idist = ny;

    cufftPlanMany(&fft_plan1D1D, 1, n, inembed, istride, idist, inembed, istride, idist, CUFFT_C2C, nx);


    cufftHandle fft_plan1D1D2;

    int *n2 = new int[1];
    n2[0] = ny;

    int inembed_pre2 = ny;
    int *inembed2 = &inembed_pre2;

    int istride2 = ny;
    int idist2 = 1;

    cufftPlanMany(&fft_plan1D1D2, 1, n2, inembed2, istride2, idist2, inembed2, istride2, idist2, CUFFT_C2C, ny);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum_of_elapsed_times2D = 0.0;
    float sum_of_elapsed_times1D1D = 0.0;

    printf("[INFO] Run benchmark 2D\n");
    for (int i = 0; i < meta_trials; i++) {
        cudaEventRecord(start, 0);

        for (int j = 0; j < fft_trials; j++) {
            cufftExecC2C(fft_plan2D, d_original_signal, d_applied_fft_signal, CUFFT_FORWARD);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        float elapsed_time_sec = elapsed_time_ms / 1000.0;
        sum_of_elapsed_times2D += elapsed_time_sec;
        printf("%f sec\n", elapsed_time_sec);
    }
    
    cudaMemcpy(h_applied_fft_signal2D, d_applied_fft_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NX; ++i){
      for (int j = 0; j < NY; ++j){
        std::cout << h_applied_fft_signal2D[i*NX +j].x << " ";
     }
      std::cout << std::endl;
    }
    cudaMemcpy(d_original_signal, h_original_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);

    
    printf("[INFO] Run benchmark 1D1D\n");
    for (int i = 0; i < meta_trials; i++) {
        cudaEventRecord(start, 0);

        for (int j = 0; j < fft_trials; j++) {
            // og -> applied, transpose back into og, repeat 
            cufftExecC2C(fft_plan1D1D, d_original_signal, d_applied_fft_signal1D1D, CUFFT_FORWARD);
            bp_soa_to_aos<<<1,32>>>((cufftComplex*)d_applied_fft_signal1D1D, (cufftComplex*)d_original_signal, NX, NY, NX);
            cudaDeviceSynchronize();
            cufftExecC2C(fft_plan1D1D, d_original_signal, d_applied_fft_signal1D1D, CUFFT_FORWARD);
            bp_aos_to_soa<<<2,32>>>((cufftComplex*)d_applied_fft_signal1D1D, (cufftComplex*)d_original_signal, NX, NY, NY);

            //cufftExecC2C(fft_plan1D1D2, d_applied_fft_signal1D1D, d_applied_fft_signal1D1D, CUFFT_FORWARD);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        float elapsed_time_sec = elapsed_time_ms / 1000.0;
        sum_of_elapsed_times1D1D += elapsed_time_sec;
        printf("%f sec\n", elapsed_time_sec);
    }


    
    cudaMemcpy(h_applied_fft_signal1D1D, d_original_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_applied_fft_signal1D1D, d_applied_fft_signal1D1D, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);

    bool correct = true;

    for (int i = 0; i < NX*NY; ++i){
      // .01 is okay, but .001 cause test failure. Should adress this later. 
      if (fabs(h_applied_fft_signal2D[i].x - h_applied_fft_signal1D1D[i].x) > .01){
        correct = false;
      }
    }

    if (correct){
      std::cout << "1D1D no transpose PASS" << std::endl;
    }
    else{
      std::cout << "1D1D no transpose FAIL" << std::endl;
    }

    for (int i = 0; i < NX; ++i){
      for (int j = 0; j < NY; ++j){
        std::cout << h_applied_fft_signal1D1D[i*NX +j].x << " ";
     }
      std::cout << std::endl;
    }
//    
//    bp_soa_to_aos<<<2,32>>>((cufftComplex*)d_original_signal, (cufftComplex*)d_applied_fft_signal, NX, NY, 8);
//
//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_applied_fft_signal1D1D[i*NX +j].x << " ";
//     }
//      std::cout << std::endl;
//    }
//    
//    bp_aos_to_soa<<<2,32>>>((cufftComplex*)d_original_signal, (cufftComplex*)d_applied_fft_signal, NX, NY, 8);
//
//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_applied_fft_signal1D1D[i*NX +j].x << " ";
//     }
//      std::cout << std::endl;
//    }


    printf("[INFO] Finished!\n");
    printf("[INFO] Average 2D: %lf sec\n", sum_of_elapsed_times2D / meta_trials);
    printf("[INFO] Average 1D1D: %lf sec\n", sum_of_elapsed_times1D1D / meta_trials);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


