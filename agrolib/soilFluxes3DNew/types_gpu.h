#ifndef SOILFLUXES3D_TYPES_GPU_H
#define SOILFLUXES3D_TYPES_GPU_H

#define NOMINMAX

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cub/device/device_reduce.cuh>

namespace soilFluxes3D::New
{
    struct MatrixGPU
    {
        cusparseSpMatDescr_t cusparseDescriptor;

        int64_t numRows;
        int64_t numCols;
        uint16_t* d_numColsInRow = nullptr;
        int64_t numNonZeroElement;
        int64_t totValuesSize;
        int64_t sliceSize;
        int64_t* d_offsets = nullptr;
        int64_t* d_columnIndeces = nullptr;
        double* d_values = nullptr;
        double* d_diagonalValues = nullptr;
        const cusparseIndexType_t offsetType = CUSPARSE_INDEX_64I;
        const cusparseIndexType_t colIdxType = CUSPARSE_INDEX_64I;
        const cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        const cudaDataType valueType = CUDA_R_64F;
    };

    struct VectorGPU
    {
        cusparseDnVecDescr_t cusparseDescriptor;
        int64_t numElements;
        double *d_values;
        const cudaDataType valueType = CUDA_R_64F;
    };
}
#endif // SOILFLUXES3D_TYPES_GPU_H
