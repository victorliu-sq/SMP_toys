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
        printf("redirect next pointer of node (y: %d, x: %d) to node (y: %d, x: %d)\n", nodes[ty][tx]->y, nodes[ty][tx]->x, nodes[ty][tx]->next->y, nodes[ty][tx]->next->x);
    }

    __syncthreads();
    // initial match
    if (tx == ty) {
        int randomNumber = tx + curand_uniform(nodes[ty][tx]->state) * (N - tx); 
        printf("Random number: %d in thread tx, ty: %d\n", randomNumber, tx);
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