#include <stdlib.h>
#include <iostream>

extern "C"
void kernelLauncher(const double* A1, const double* A2, double* R, const uint64_t N);

const uint64_t N = static_cast<uint64_t> (pow(static_cast<double>(2), static_cast<double>(26)));

double testCUDAinProject()
{
    double *A1, *A2, *R;
    //std::cout << "N = " << N << std::endl;

    A1 = (double*) malloc(N*sizeof(double));
    A2 = (double*) malloc(N*sizeof(double));
    R = (double*) malloc(N*sizeof(double));

    for (int ind = 0; ind < N; ind++)
    {
        A1[ind] = (double) ind;
        A2[ind] = (double) ind;
    }

    kernelLauncher(A1, A2, R, N);

    // for (uint64_t ind = 0; ind < N; ind++)
    //     std::cout << R[ind] << ", ";

    // std::cout << "GPU work: R[1024] = " << R[1024] << " - correct if 2028" << std::endl;
    return R[1024];
}
