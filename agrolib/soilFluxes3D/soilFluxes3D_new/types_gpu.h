#ifndef SOILFLUXES3D_TYPES_GPU_H
#define SOILFLUXES3D_TYPES_GPU_H

#include <cuda.h>
#include <cusparse_v2.h>

#include "../types.h"

namespace soilFluxes3D::New
{
    struct MatrixGPU
    {
        cusparseConstSpMatDescr_t cusparseDescriptor;

        uint64_t numRows;
        uint64_t numCols;
        uint64_t numNonZeroElement;
        uint64_t totValuesSize;
        uint64_t sliceSize;
        uint64_t* d_offsets = nullptr;
        uint64_t* d_columnIndeces = nullptr;
        double* d_values = nullptr;
        const cusparseIndexType_t offsetType = CUSPARSE_INDEX_64I;
        const cusparseIndexType_t colIdxType = CUSPARSE_INDEX_64I;
        const cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        const cudaDataType valueType = CUDA_R_64F;
    };

    struct VectorGPU
    {
        cusparseDnVecDescr_t cusparseDescriptor;
        uint64_t numElements;
        double *d_values;
        const cudaDataType valueType = CUDA_R_64F;
    };
}
#endif // SOILFLUXES3D_TYPES_GPU_H
