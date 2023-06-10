#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include <time.h>

#define N 11

struct Node {
    int y;
    int x;
    // x of next node
    int nx;
    Node* next;
    curandState* state;
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
        if (nodes[ty][tx]->next != NULL) {
            int new_nx = nodes[ty][tx]->next->nx;
            Node* new_next = nodes[ty][tx]->next->next;
            __syncthreads();
            nodes[ty][tx]->nx = new_nx;
            nodes[ty][tx]->next = new_next;
        }
    }
    if (ty == 0) {
        printf("thread %d's nx is %d\n", tx, nodes[ty][tx]->nx);
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