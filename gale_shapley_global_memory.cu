#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

#define N 5

// data structures in constant memory
__constant__ int preference_list_men[5][5] = {
    {4, 1, 2, 0, 3},  // A
    {3, 2, 1, 0, 4},  // B
    {1, 3, 0, 4, 2},  // C
    {3, 1, 4, 2, 0},  // D
    {4, 0, 1, 2, 3},  // E
};

__constant__ int ranking_matrix_women[5][5] = {
    {4, 1, 3, 0, 2},  // L
    {1, 0, 3, 2, 4},  // M
    {0, 4, 1, 3, 2},  // N
    {1, 3, 2, 0, 4},  // O
    {2, 0, 3, 4, 1},  // P
};

// kernel function
__global__ void gale_shapley_algorithm_in_parallelism(int* men_state, int* men_next_proposal, int* women_proposed)
{
    // Function body
    if (threadIdx.x == 0) {
        printf("Hello from master thread %d\n", threadIdx.x);
    } else {
        printf("Hello from slave thread %d\n", threadIdx.x - 1);
    }
    int num_free_men = N;
    // Master processor
    

    __syncthreads();
    // Slave processor
}

int main() {
    // Allocate data structures in global memory
    // data structure for master processor
    int* men_state;
    int* men_next_proposal;
    cudaMalloc((void **)&men_state, N * sizeof(int));
    cudaMalloc((void **)&men_next_proposal, N * sizeof(int));

    // data structure for slave processor
    int* women_prposed;
    cudaMalloc((void**)&women_prposed, N * N * sizeof(int));

    // launch kernel function
    gale_shapley_algorithm_in_parallelism<<<1, 1 + N>>>(men_state, men_next_proposal, women_prposed);

    // free data structures in global memory
    cudaFree(men_state);
    cudaFree(men_next_proposal);

    cudaFree(women_prposed);

    return 0;
}