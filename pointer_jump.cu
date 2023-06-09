#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda_runtime_api.h>

#define N 10

struct Node {
    int dist;
    Node* next;
};

__device__ Node* createNode() {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->dist = 0;
    newNode->next = NULL;
    return newNode;
}

__global__ void pointer_jump_algorithm_() {
    __shared__ Node* nodes[N];

    int tid = threadIdx.x;

    nodes[tid] = createNode();
    __syncthreads();
    if (tid != N - 1) {
        nodes[tid]->dist = 1;
        nodes[tid]->next = nodes[tid + 1];
    }
    printf("Hello from thread %d\n", tid);
    int total = int(ceilf(log2f(N)));
    printf("Iteration number is %d\n", total);
    for (int i = 1; i <= total; i++) {
        __syncthreads();
        if (nodes[tid]->next != NULL) {
            int new_dist = nodes[tid]->dist + nodes[tid]->next->dist;
            Node* new_next = nodes[tid]->next->next;
            __syncthreads();
            nodes[tid]->dist = new_dist;
            nodes[tid]->next = new_next;
        }    
        printf("In iteration %d, nodes[%d].dist becomes %d\n", i, tid, nodes[tid]->dist);
    }
}


int main () {
    pointer_jump_algorithm_<<<1, N>>>();
    cudaStreamSynchronize(0);
    return 0;
}