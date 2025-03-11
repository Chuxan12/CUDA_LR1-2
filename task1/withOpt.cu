#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <iostream>
#include <cmath>

using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define TILE_SIZE 16

// Ядро с использованием Shared Memory
__global__ void matrixMulOptKernel(float* C, const float* A, const float* B, int M, int N, int P) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        int A_col = k * TILE_SIZE + tx;
        int B_row = k * TILE_SIZE + ty;

        As[ty][tx] = (row < M && A_col < N) ? A[row * N + A_col] : 0.0f;
        Bs[ty][tx] = (B_row < N && col < P) ? B[B_row * P + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < P) {
        C[row * P + col] = sum;
    }
}

// Хост-функция для оптимизированного умножения
void matrixMul(float* h_C, const float* h_A, const float* h_B, int M, int N, int P) {
    float* d_A, * d_B, * d_C;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((P + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matrixMulOptKernel << <gridSize, blockSize >> > (d_C, d_A, d_B, M, N, P);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    cout << "Time: " << milliseconds << " ms" << endl;

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

int main() {
    int M = 16384, N = 16384, P = 16384; // Размеры матриц
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * P * sizeof(float);
    size_t sizeC = M * P * sizeof(float);

    float* h_A = new float[M * N];
    float* h_B = new float[N * P];
    float* h_C = new float[M * P];

    // Инициализация случайными значениями
    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * P; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Запуск оптимизированной версии
    matrixMul(h_C, h_A, h_B, M, N, P);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}