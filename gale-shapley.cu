#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>

#define N 5

// data structures in constant memory
__constant__ int preference_list_men[5][5] = {
    {3, 1, 2, 0, 4},  // A
    {4, 2, 1, 0, 3},  // B
    {1, 4, 0, 3, 2},  // C
    {4, 1, 3, 2, 0},  // D
    {3, 0, 1, 2, 4},  // E
};

__constant__ int ranking_matrix_women[5][5] = {
    {4, 1, 3, 0, 2},  // L
    {1, 0, 3, 2, 4},  // M
    {0, 4, 1, 3, 2},  // N
    {1, 3, 2, 0, 4},  // O
    {2, 0, 3, 4, 1},  // P
};

// kernel function
__global__ void gale_shapley_algorithm_in_parallelism()
{
    __shared__ int men_state[N];
    __shared__ int men_next_proposal[N];
    __shared__ int women_proposed[N*N];

    // Function body
    if (threadIdx.x == 0) {
        printf("Hello from master thread %d\n", threadIdx.x);
    } else {
        printf("Hello from slave thread %d\n", threadIdx.x - 1);
    }
    
    int iterations = 0;
    while (true && iterations < 10) {
        // Master processor
        __syncthreads();
        if (threadIdx.x == 0) {
            // check whether all men have been engaged
            // if one man is free, let him propose next woman
            bool done = true;
            for (int i = 0; i < N; i++) {
                if (men_state[i] == 0) {
                    // if one man is free, master processor cannot stop
                    done = false;
                    // propose next woman and update next proposal
                    int man_idx = i;
                    int woman_rank = men_next_proposal[i]++;
                    int woman_idx = preference_list_men[i][woman_rank];
                    // update women_proposed                
                    women_proposed[woman_idx * N + man_idx] = 1;
                    printf("man %d proposed to women %d\n", man_idx, woman_idx);
                }
            }
            if (done == true) {
                return;
            }
        }

        // Slave processor
        __syncthreads();
        if (threadIdx.x > 0) {
            // First, wait for packets from master thread            
            // check which men have proposed to this woman and engage with the best one
            int woman_idx = threadIdx.x - 1;
            int best_man_rank = N;
            int best_man_idx = N;
            for (int i = 0; i < N; i++) {
                if (women_proposed[woman_idx * N + i] == 1) {
                    printf("woman %d has been proposed by man %d\n", woman_idx, i);
                    int cur_man_rank = ranking_matrix_women[woman_idx][i];
                    if (cur_man_rank < best_man_rank) {
                        best_man_rank = cur_man_rank;
                        best_man_idx = i;
                    }
                }
            }

            // set current wman engaged with the best man
            if (best_man_idx != N) {
                printf("woman %d has been engaged with man %d\n", woman_idx, best_man_idx);
                men_state[best_man_idx] = 1;
            }

            // set other proposing men as free and reset proposal from that man
            for (int i = 0; i < N; i++) {
                if (women_proposed[woman_idx * N + i] == 1 && i != best_man_idx) {
                    printf("woman %d has rejeted proposal from man %d\n", woman_idx, i);
                    men_state[i] = 0;
                    women_proposed[woman_idx * N + i] = 0;
                }
            }
        }    
        iterations++;
    }

}

int main() {
    // launch kernel function
    gale_shapley_algorithm_in_parallelism<<<1, 1 + N>>>();

    cudaStreamSynchronize(0);
    return 0;
}