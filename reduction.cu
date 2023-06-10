#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#define N 13

struct Node
{
    int min;
    int idx;
    bool ok;
    curandState* state;
};

__device__ Node* createNode() {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->state = (curandState*)malloc(sizeof(curandState));
    curand_init(2, threadIdx.x, 0, newNode->state);
    return newNode;
}


__global__ void reduction_algorithm() {
    __shared__ Node* nodes[N];
    int tid = threadIdx.x;
    nodes[tid] = createNode();
    int rand = int(curand_uniform(nodes[tid]->state) * 100);
    nodes[tid]->idx = tid;
    nodes[tid]->min = rand;
    printf("node[%d]'s value is %d\n", tid, rand);

    // reduction
    int remaining = N;
    int total = int(ceilf(log2f(N)));
    for (int i = 1; i <= total; i++) {
        __syncthreads();
        int stride = 1 << (total - i);
        int reduction = 0;
        if (tid + stride < remaining) {
            printf("Reduction on nodes[%d] during with stride%d, remainging: %d\n", tid, stride, remaining);
            if (nodes[tid]->min > nodes[tid + stride]->min) {
                nodes[tid]-> min = nodes[tid + stride]->min;
                nodes[tid]->idx = nodes[tid + stride]->idx;
            }
            reduction++;
            printf("Get min value %d\n", nodes[tid]-> min);
        }
        remaining -= reduction;
    }

    __syncthreads();
    if (tid == 0) {
        printf("min value is %d\n", nodes[tid]->min);
    }
    return;
}

int main() {
    reduction_algorithm<<<1, N>>>();
    cudaStreamSynchronize(0);
    return 0;
}