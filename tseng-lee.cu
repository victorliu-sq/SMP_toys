#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>

#define N 8

// data structures in constant memory
__constant__ int preference_list_men[N][N] = 
{
    {0, 3, 4, 5, 1, 2, 6, 7}, // M0
    {0, 1, 3, 7, 6, 5, 2, 4}, // M1
    {2, 1, 0, 3, 7, 6, 4, 5}, // M2
    {3, 5, 6, 7, 0, 2, 1, 4}, // M3
    {1, 2, 3, 5, 6, 0, 7, 4}, // M4
    {1, 0, 7, 5, 4, 6, 2, 3}, // M5
    {4, 2, 0, 3, 7, 5, 1, 6}, // M6
    {4, 6, 5, 2, 0, 1, 3, 7}  // M7
};

__constant__ int ranking_matrix_women[N][N] = 
{
    {6, 7, 4, 3, 5, 2, 1, 0}, // W0
    {0, 5, 3, 2, 1, 4, 6, 7}, // W1
    {0, 2, 1, 3, 7, 6, 4, 5}, // W2
    {5, 3, 6, 0, 7, 4, 1, 2}, // W3
    {1, 2, 0, 7, 3, 4, 6, 5}, // W4
    {0, 1, 2, 3, 4, 5, 6, 7}, // W5
    {1, 0, 2, 4, 3, 5, 6, 7}, // W6
    {3, 2, 7, 1, 4, 5, 0, 6}  // W7
};

// kernel function
__global__ void tseng_lee_algorithm_in_parallelism()
{
    // __shared__ int men_state[N];
    __shared__ int men_proposal[N];
    // __shared__ int women_proposed[N*N];
    unsigned int tid = threadIdx.x;
    
    // Let the man propose to his best choice
    // int man_idx = tid;
    // int rank = men_proposal[man_idx];
    // int woman_idx = preference_list_men[man_idx][rank];
    // men_next_proposal[man_idx]++;

    // printf("man %d propose to woman %d\n", man_idx, woman_idx);
    // women_proposed[woman_idx * N + rank] = 1;

    // Merge up and resolve conflicts
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            int num_men = N / stride;
            printf("Hello from threadIdx %d, current stride is %d, num of men to merge is %d\n", tid, stride, num_men);
            int women_proposed[N*N];
            // size of men_state is known at runtime, we need to use malloc()
            int men_state[N];
            bool hasConflict = false;
            do {
                for (int i = 0; i < num_men; i++) {
                    // initialize all men as engaged first
                    int man_idx = tid + i * stride;
                    men_state[man_idx] = 1;
                    // let men propose to his woman
                    int rank = men_proposal[man_idx];
                    int woman_idx = preference_list_men[man_idx][rank];
                    printf("in thread %d, man %d propose to woman %d\n", tid, man_idx, woman_idx);
                    women_proposed[woman_idx * N + man_idx] = 1;
                }

                hasConflict = false;
                for (int w = 0; w < N; w++) {
                    int best_man_rank = N;
                    int best_man_idx = N;
                    int num_men_proposing = 0;
                    for (int m = 0; m < N; m++) {
                        if (women_proposed[w * N + m] == 1) {
                            printf("in thread %d, woman %d has been proposed by man %d\n", tid, w, m);
                            int cur_man_rank = ranking_matrix_women[w][m];
                            if (cur_man_rank < best_man_rank) {
                                best_man_rank = cur_man_rank;
                                best_man_idx = m;
                            }
                            num_men_proposing++;
                        }
                    }
                    if (num_men_proposing > 1) {
                        hasConflict = true;
                    }

                    // set current wman engaged with the best man
                    if (best_man_idx != N) {
                        printf("woman %d has been engaged with man %d\n", w, best_man_idx);
                        men_state[best_man_idx] = 1;
                    }

                    // set other proposing men as free and reset proposal from that man
                    for (int m = 0; m < N; m++) {
                        if (women_proposed[w * N + m] == 1 && m != best_man_idx) {
                            printf("woman %d has rejeted proposal from man %d\n", w, m);
                            men_state[m] = 0;
                            men_proposal[m]++;
                            women_proposed[w * N + m] = 0;
                        }
                    }
                }
                if (hasConflict) {
                    printf("current thread %d still has conflict\n", tid);
                } else {
                    printf("current thread %d has no conflict now\n", tid);
                }
            } while(hasConflict);
        }
    }
}

int main() {
    // Get women's ranking matrix 
    // int prefernce[8][8] = {
    //     {8, 7, 6, 4, 3, 5, 1, 2}, // M0
    //     {1, 5, 4, 3, 6, 2, 7, 8}, // M1
    //     {1, 3, 2, 4, 7, 8, 6, 5}, // M2
    //     {4, 7, 8, 2, 6, 1, 3, 5}, // M3
    //     {3, 1, 2, 5, 6, 8, 7, 4}, // M4
    //     {1, 2, 3, 4, 5, 6, 7, 8}, // M5
    //     {2, 1, 3, 5, 4, 6, 7, 8}, // M6
    //     {7, 4, 2, 1, 5, 6, 8, 3}  // M7
    // };

    // int rank[8][8];

    // for (int i = 0; i < 8; ++i) {
    //     for (int j = 0; j < 8; ++j) {
    //         // Subtract 1 from the preference as we're using 0-indexing
    //         int man = prefernce[i][j] - 1;
    //         // Set the rank of man in the preference list of the woman
    //         rank[i][man] = j;
    //     }
    // }

    // // Print the rank array
    // for (int i = 0; i < 8; ++i) {
    //     printf("W%d: ", i + 1);
    //     for (int j = 0; j < 8; ++j) {
    //         printf("%d ", rank[i][j]);
    //     }
    //     printf("\n");
    // }
    tseng_lee_algorithm_in_parallelism<<<1, N>>>();
    cudaStreamSynchronize(0);
    return 0;
}