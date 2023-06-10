#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>

#define N 100
#define SOURCE 50

struct Node {
    int k;
    bool reached;
};

__device__ Node* createNode() {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->reached = false;
    return newNode;
}

__global__ void pointer_jump_algorithm_() {
    __shared__ Node* nodes[N];

    int tid = threadIdx.x;
    // printf("Hello from thread %d\n", tid);

    nodes[tid] = createNode();
    if (tid == SOURCE) {
        nodes[tid]->reached = true;
        nodes[tid]->k = N;
        printf("Try to broadcast k %d from node %d to all other nodes\n", nodes[tid]->k, tid);
    }

    // __syncthreads();

    int total = int(ceilf(log2f(N)));
    for(int i = 1; i <= total; i++) {
        __syncthreads();
        int stride = 1 << (i - 1);
        if (nodes[tid]->reached) {
            printf("Iteration: %d, current node: %d\n", i, tid);
            if (tid + stride < N) {
                nodes[tid + stride]->reached = true;
            }
            if (tid - stride >= 0) {
                nodes[tid - stride]->reached = true;
            }
        }
    }

    __syncthreads();
    if (nodes[tid]->reached) {
        printf("reached node: %d\n", tid);
    }
}


int main () {
    pointer_jump_algorithm_<<<1, N>>>();
    cudaStreamSynchronize(0);
    return 0;
}

