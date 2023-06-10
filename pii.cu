#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include <time.h>

#define N 8

__constant__ int rank_matrix[N][N][2] = 
{
    {{0, 8}, {3, 1}, {4, 1}, {5, 4}, {1, 3}, {2, 1}, {6, 2}, {7, 7}}, // M0
    {{0, 7}, {1, 5}, {3, 3}, {7, 7}, {6, 1}, {5, 2}, {2, 1}, {4, 4}}, // M1
    {{2, 6}, {1, 4}, {0, 2}, {3, 8}, {7, 2}, {6, 3}, {4, 3}, {5, 2}}, // M2
    {{3, 4}, {5, 3}, {6, 4}, {7, 2}, {0, 5}, {2, 4}, {1, 5}, {4, 1}}, // M3
    {{1, 3}, {2, 6}, {3, 7}, {5, 6}, {6, 8}, {0, 5}, {7, 4}, {4, 5}}, // M4
    {{1, 5}, {0, 2}, {7, 8}, {5, 1}, {4, 8}, {6, 6}, {2, 6}, {3, 8}}, // M5
    {{4, 1}, {2, 7}, {0, 6}, {3, 3}, {7, 7}, {5, 7}, {1, 7}, {6, 8}}, // M6
    {{4, 2}, {6, 8}, {5, 5}, {2, 5}, {0, 4}, {1, 8}, {3, 8}, {7, 3}}  // M7
};


struct Node {
    int y;
    int x;
    // x of next node
    int nx;
    Node* next;
    curandState* state;
    // broadcast
    bool reached_row;
    bool reached_col;
    int rank_row[2];
    int rank_col[2];
    // stable
    bool unstable;
};

__device__ Node* createNode(unsigned long long seeds, int ty, int tx) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->state = (curandState*)malloc(sizeof(curandState));
    if (!newNode) {
        printf("Memory error\n");
        return NULL;
    }
    newNode->y = ty;
    newNode->x = tx;
    newNode->next = NULL;
    newNode->reached_row = false;
    newNode->reached_col = false;
    curand_init(seeds, threadIdx.x + threadIdx.y * N, 0, newNode->state);
    return newNode;
}

__global__ void parallel_iterative_improvement_algorithm_in_parallelism(unsigned long long seeds)
{
    // declare a 2d array of nodes
    __shared__ Node* nodes[N][N];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    nodes[ty][tx] = createNode(seeds, ty, tx);
    printf("Create node by threadIdx ty: %d, tx: %d, node's y: %d, x: %d\n", ty, tx, nodes[ty][tx]->y, nodes[ty][tx]->x);
    __syncthreads();

    // point each node(y, x) to node(y + 1, x)
    if (ty < N - 1) {
        nodes[ty][tx]->next = nodes[ty+1][tx];
        nodes[ty][tx]->nx = nodes[ty][tx]->next->nx;
        printf("redirect next pointer of node (y: %d, x: %d) to node (y: %d, x: %d)\n", nodes[ty][tx]->y, nodes[ty][tx]->x, nodes[ty][tx]->next->y, nodes[ty][tx]->next->x);
    } else {
        nodes[ty][tx]->next = NULL;
        nodes[ty][tx]->nx = tx;
    }

    __syncthreads();
    // Constructing an Initial Matching
    if (tx == ty && tx != N) {
        int random_number = tx + curand_uniform(nodes[ty][tx]->state) * (N - tx); 
        printf("Random number: %d in thread tx, ty: %d\n", random_number, tx);

        // swap next pointer with tx, t_randomNumber / t_rx
        int rx = random_number;
        if (rx != tx) {
            Node* temp_node = nodes[ty][tx]->next;
            nodes[ty][tx]->next = nodes[ty][rx]->next;
            nodes[ty][rx]->next = temp_node;
            printf("node (ty: %d tx: %d)'s next node becomes (ty: %d tx: %d) whereas node (ty: %d rx: %d)'s next node becomes (ty: %d tx: %d)\n", ty, tx, nodes[ty][tx]->next->y, nodes[ty][tx]->next->x, ty, rx, nodes[ty][rx]->next->y, nodes[ty][rx]->next->x);
        }
    }

    // Pointer Jump
    int total = int(ceilf(log2f(N)));
    for (int i = 1; i <= total; i++) {
        __syncthreads();
        int new_nx = nodes[ty][tx]->nx;
        Node* new_next = nodes[ty][tx]->next;
        if (nodes[ty][tx]->next != NULL) {
            new_nx = nodes[ty][tx]->next->nx;
            new_next = nodes[ty][tx]->next->next;
        }
        __syncthreads();
        nodes[ty][tx]->nx = new_nx;
        nodes[ty][tx]->next = new_next;
    }

    __syncthreads();
    if (ty == 0) {
        printf("thread %d's nx is %d\n", tx, nodes[ty][tx]->nx);
        int my = tx;
        int mx = nodes[ty][tx]->nx;
        nodes[my][mx]->reached_col = true;
        nodes[my][mx]->reached_row = true;
        // printf("nodes[%d][%d] now is a matching, %d, %d\n", my, mx, nodes[my][mx]->reached_col, nodes[my][mx]->reached_row);
    }

    // initialize all nodes
    __syncthreads();
    if (nodes[ty][tx]->reached_col && nodes[ty][tx]->reached_row) {
        printf("nodes[%d][%d] is a matching\n", ty, tx);
    }
    // printf("nodes %d %d: %d %d\n", ty, tx, nodes[ty][tx]->reached_col, nodes[ty][tx]->reached_row);
    for (int i = 1; i <= total; i++) {
        __syncthreads();
        int stride = 1 << (i - 1);
        if (nodes[ty][tx]->reached_col) {
            // printf("Iteration: %d, current node: %d\n", i, tid);
            if (tx + stride < N) {
                nodes[ty][tx + stride]->reached_col = true;
                nodes[ty][tx + stride]->rank_col[0] = rank_matrix[ty][tx][0];
                nodes[ty][tx + stride]->rank_col[1] = rank_matrix[ty][tx][1];
            }
            if (tx - stride >= 0) {
                nodes[ty][tx - stride]->reached_col = true;
                nodes[ty][tx - stride]->rank_col[0] = rank_matrix[ty][tx][0];
                nodes[ty][tx - stride]->rank_col[1] = rank_matrix[ty][tx][1];
            }
        }

        if (nodes[ty][tx]->reached_row) {
            // printf("Iteration: %d, current node: %d\n", i, tid);
            if (ty + stride < N) {
                nodes[ty + stride][tx]->reached_row = true;
                nodes[ty + stride][tx]->rank_row[0] = rank_matrix[ty][tx][0];
                nodes[ty + stride][tx]->rank_row[1] = rank_matrix[ty][tx][1];
            }
            if (ty - stride >= 0) {
                nodes[ty - stride][tx]->reached_row = true;
                nodes[ty - stride][tx]->rank_row[0] = rank_matrix[ty][tx][0];
                nodes[ty - stride][tx]->rank_row[1] = rank_matrix[ty][tx][1];
            }
        }
    }

    __syncthreads();
    // Check if all nodes have been broadcasted
    // if (tx == 0 && ty == 0) {
    //     for (int i = 0; i < N; i++) {
    //         for (int j = 0; j < N; j++) {
    //             if (nodes[i][j]->reached_col && nodes[i][j]->reached_row) {
    //                 printf("nodes[%d][%d] is reached w/ left: %d, right: %d\n", i, j);
    //             }
    //         }
    //     }
    // }

    // Stability Checking
    if ((rank_matrix[ty][tx][0] < nodes[ty][tx]->rank_col[0] && rank_matrix[ty][tx][1] < nodes[ty][tx]->rank_col[1]) || (rank_matrix[ty][tx][0] < nodes[ty][tx]->rank_row[0] && rank_matrix[ty][tx][1] < nodes[ty][tx]->rank_row[1]))  {
        printf("nodes[%d][%d] is unstable\n", ty, tx);
        nodes[ty][tx]->unstable = false;
    } else {
        nodes[ty][tx]->unstable = true;
    }

    free(nodes[ty][tx]);
}

int main() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    unsigned long long currentTimeInNanoseconds = ts.tv_sec * 1e9 + ts.tv_nsec;
    unsigned long long seeds = currentTimeInNanoseconds;
    printf("Seed is %lld\n", currentTimeInNanoseconds);

    dim3 nodes_dim(N, N);
    parallel_iterative_improvement_algorithm_in_parallelism<<<1, nodes_dim>>>(seeds);
    cudaStreamSynchronize(0);
    return 0;
}