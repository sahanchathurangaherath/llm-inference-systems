#include <cstdio>
__global__ void hello() { printf("Hello from GPU thread %d\n", threadIdx.x); }
int main() { hello<<<1, 4>>>(); cudaDeviceSynchronize(); return 0; }
