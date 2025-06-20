extern "C"
__global__ void kernelTest(const double* a1, const double* a2, double* r)
{
    int ind = blockIdx.x;
    r[ind] = a1[ind] + a2[ind];
}

extern "C"
void kernelLauncher(const double* A1, const double* A2, double* R, const uint64_t N)
{

    double *a1, *a2, *r;
    cudaMalloc(&a1, N*sizeof(double));
    cudaMalloc(&a2, N*sizeof(double));

    cudaMemcpy(a1, A1, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(a2, A2, N*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&r, N*sizeof(double));

    kernelTest<<<N, 1>>>(a1, a2, r);

    cudaMemcpy(R, r, N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(a1);
    cudaFree(a2);
    cudaFree(r);
}
