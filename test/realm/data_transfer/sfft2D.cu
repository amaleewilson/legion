#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <iostream>

const int NX = 2048;
const int NY = 2048;

const int DEFAULT_FFT_TRIALS = 10000;
const int DEFAULT_META_TRIALS = 10;

const int BATCH_SIZE = 1;

int main(int argc, char **argv) {
    int fft_trials = DEFAULT_FFT_TRIALS;
    int meta_trials = DEFAULT_META_TRIALS;

    printf("[INFO] META trials: %d\n", meta_trials);
    printf("[INFO] FFT trials: %d\n", fft_trials);

    long nx = NX;
    long ny = NX;
    printf("[INFO] NX Length: %ld\n", nx);
    printf("[INFO] NY Length: %ld\n", ny);

    cufftComplex *h_original_signal, *h_applied_fft_signal;
    cudaMallocHost((void **) &h_original_signal, sizeof(cufftComplex) * NX * NY);
    cudaMallocHost((void **) &h_applied_fft_signal, sizeof(cufftComplex) * NX * NY);

    cufftComplex *d_original_signal, *d_applied_fft_signal;
    cudaMalloc((void **) &d_original_signal, sizeof(cufftComplex) * NX * NY);
    cudaMalloc((void **) &d_applied_fft_signal, sizeof(cufftComplex) * NX * NY);

    /*
     * generate random signal as original signal
     */
    srand(0); // initialize random seed
    for (int i = 0; i < NX*NY; i++) {
        h_original_signal[i].x = (float)((int)rand() % 10);
        h_original_signal[i].y = 0.0;
    }

//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_original_signal[i*NX + j].x << " ";
//     }
//      std::cout << std::endl;
//    }

    cudaMemcpy(d_original_signal, h_original_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);

    cufftHandle fft_plan;
    //cufftPlan1d(&fft_plan, NX, CUFFT_C2C, BATCH_SIZE);
    cufftPlan2d(&fft_plan, NX, NY, CUFFT_C2C);
    
//    int *n = new int[2];
//    n[0] = nx;
//    n[1] = ny;
//
//    int *inembed = new int[2];
//    inembed[0] = nx;
//    inembed[1] = ny;
//
//    int istride = 1;
//    int idist = nx*ny;
//
//    cufftPlanMany(&fft_plan, 2, n, inembed, istride, idist, inembed, istride, idist, CUFFT_C2C, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sum_of_elapsed_times = 0.0;

    printf("[INFO] Run benchmark...\n");
    for (int i = 0; i < meta_trials; i++) {
        cudaEventRecord(start, 0);

        for (int j = 0; j < fft_trials; j++) {
            cufftExecC2C(fft_plan, d_original_signal, d_applied_fft_signal, CUFFT_FORWARD);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        float elapsed_time_sec = elapsed_time_ms / 1000.0;
        sum_of_elapsed_times += elapsed_time_sec;
        printf("%f sec\n", elapsed_time_sec);
    }


    cudaMemcpy(h_applied_fft_signal, d_applied_fft_signal, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);

//    printf("[INFO] computing sum...\n");
//    
//    for (int i = 0; i < NX; ++i){
//      for (int j = 0; j < NY; ++j){
//        std::cout << h_applied_fft_signal[i*NX + j].x << " ";
//     }
//      std::cout << std::endl;
//    }
//    
//    float red = 0;
//    for (int i = 0; i < NX*NY; i++) {
//        red += h_applied_fft_signal[i].x;
//        red -= h_applied_fft_signal[i].y;
//    }
//
//    printf("SUM : %f\n", red);


    printf("[INFO] Finished!\n");
    printf("[INFO] Average: %lf sec\n", sum_of_elapsed_times / meta_trials);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

