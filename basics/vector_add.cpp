#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);
    
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vector_add, dim3(numBlocks), dim3(blockSize), 0, 0, d_a, d_b, d_c, N);
    
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);
    
    // Verify
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            ok = false;
            break;
        }
    }
    printf("Vector add: %s\n", ok ? "PASSED" : "FAILED");
    
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
