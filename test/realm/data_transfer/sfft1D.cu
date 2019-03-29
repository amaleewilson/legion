#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>

const int DEFAULT_SIGNAL_LENGTH = 4096;
const int DEFAULT_FFT_TRIALS = 10000;
const int DEFAULT_META_TRIALS = 10;

const int BATCH_SIZE = 1;

int main(int argc, char **argv) {
    int fft_trials = DEFAULT_FFT_TRIALS;
    int meta_trials = DEFAULT_META_TRIALS;

    printf("[INFO] META trials: %d\n", meta_trials);
    printf("[INFO] FFT trials: %d\n", fft_trials);

    long signal_length = DEFAULT_SIGNAL_LENGTH;
    printf("[INFO] Signal Length: %ld\n", signal_length);

    cufftComplex *h_original_signal;
    cudaMallocHost((void **) &h_original_signal, sizeof(cufftComplex) * signal_length);

    cufftComplex *d_original_signal, *d_applied_fft_signal;
    cudaMalloc((void **) &d_original_signal, sizeof(cufftComplex) * signal_length);
    cudaMalloc((void **) &d_applied_fft_signal, sizeof(cufftComplex) * signal_length);

    /*
     * generate random signal as original signal
     */
    srand(0); // initialize random seed
    for (int i = 0; i < signal_length; i++) {
        h_original_signal[i].x = (float)rand() / RAND_MAX;
        h_original_signal[i].y = 0.0;
    }
    cudaMemcpy(d_original_signal, h_original_signal, sizeof(cufftComplex) * signal_length, cudaMemcpyHostToDevice);

    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, signal_length, CUFFT_C2C, BATCH_SIZE);

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
    printf("[INFO] Finished!\n");
    printf("[INFO] Average: %lf sec\n", sum_of_elapsed_times / meta_trials);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
