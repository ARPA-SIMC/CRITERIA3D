#ifndef SOILFLUXES3DCUDA_ITERATIONKERNEL_H
#define SOILFLUXES3DCUDA_ITERATIONKERNEL_H

#include <vector>
#include <cuda_runtime.h>
#include "dataTypes.h"

namespace soilFluxes3D_CUDA
{
    #define cudaCheck(retValue) {if(retValue != cudaSuccess) {return retValue;}}

    class iterationKernel
    {
        private:
            uint32_t numIteration = 0;
            uint32_t numThreadperBlock = 3; //64

            CUDAoptimizedVector<double> *d_constantTerm = nullptr;
            CUDAoptimizedVector<double> *d_unknownVector = nullptr;
            CUDAoptimizedMatrix<double> *d_iterationMatrix = nullptr;

            CUDAoptimizedVector<double> *h_constantTerm = nullptr;
            CUDAoptimizedVector<double> *h_unknownVector = nullptr;
            CUDAoptimizedMatrix<double> *h_iterationMatrix = nullptr;

            kernelStatus currentStatus;

            cudaError_t uploadData();
            cudaError_t downloadData();

        public:
            iterationKernel() = delete;
            iterationKernel(const uint32_t n) : numIteration(n), currentStatus(kernelCreated) {}
            ~iterationKernel() {}

            cudaError_t launchKernel(void);

            template<class T>
            cudaError_t getKernelOutput(T*&);

            template<class T>
            cudaError_t inizializeKernelData(T**, T*, const uint32_t);

            template<class T>
            cudaError_t inizializeKernelData(std::vector<std::vector<T>>, std::vector<T>);
    };

    extern "C"
        cudaError_t launchKernelFunction(CUDAoptimizedMatrix<double>*, CUDAoptimizedVector<double>*, CUDAoptimizedVector<double>*, const uint32_t, const uint32_t);

}

#include "iterationkernel_inl.h"
#endif // ITERATIONKERNEL_H
