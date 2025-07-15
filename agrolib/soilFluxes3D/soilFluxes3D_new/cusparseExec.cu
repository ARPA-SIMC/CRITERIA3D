#include "cudaFunctions.h"
#include "gpuEntryPoints.h"
#include "gpusolver.h"


using namespace soilFluxes3D::New;

void runCUSPARSEtest(double* &vecSol)
{
    GPUSolver solverCUsparse;

    TmatrixElement **Atest = (TmatrixElement**) calloc(5, sizeof(TmatrixElement*));
    for (int i = 0; i < 5; ++i)
        Atest[i] = (TmatrixElement*) calloc(2, sizeof(TmatrixElement));

    Atest[0][0].index = 1; Atest[0][0].val = 0.5;
    Atest[0][1].index = 4; Atest[0][1].val = 0.3;
    Atest[1][0].index = 0; Atest[1][0].val = 0.2;
    Atest[1][1].index = 3; Atest[1][1].val = 0.4;
    Atest[2][0].index = 2; Atest[2][0].val = 0.1;
    Atest[2][1].index = 4; Atest[2][1].val = 0.6;
    Atest[3][0].index = 1; Atest[3][0].val = 0.1;
    Atest[3][1].index = 3; Atest[3][1].val = 0.2;
    Atest[4][0].index = 0; Atest[4][0].val = 0.4;
    Atest[4][1].index = 4; Atest[4][1].val = 0.1;

    double Btest[] = {1, 2, 3, 4, 5};
    double Xtest[] = {1, 1, 1, 1, 1};

    solverCUsparse.copyMatrixVectorFromOld(Atest, Btest, Xtest, 5);
    solverCUsparse.run();

    solverCUsparse.gatherOutput(vecSol);
}

void runCUSPARSEiteration(TmatrixElement **A, double* b, double* x, double* &vecSol, uint64_t numNodes)
{
    GPUSolver solverCUsparse;

    solverCUsparse.copyMatrixVectorFromOld(A, b, x, numNodes);
    solverCUsparse.run();

    solverCUsparse.gatherOutput(vecSol);
}


void copyMatrixVectorFromOld_k(MatrixGPU &iterationMatrix, VectorGPU &constantTerm, VectorGPU &tempSolution1, VectorGPU &tempSolution2, TmatrixElement **matA, double *vecB, double *vecX, uint64_t numNodes)
{
    iterationMatrix.numRows = numNodes;
    iterationMatrix.numCols = numNodes;
    uint64_t sliceSizeTemp = 64;             //Real: multiple of warp-size (32).
    uint64_t maxNumColsTemp = 11;             //Real: setted in _parameters
    iterationMatrix.sliceSize = sliceSizeTemp;

    uint64_t numSlice = (uint64_t) ceil((double) numNodes / sliceSizeTemp);

    uint64_t nnz = 0;
    uint64_t tnv = numSlice * sliceSizeTemp * maxNumColsTemp;

    uint64_t* h_offsets = (uint64_t*) calloc(numSlice + 1, sizeof(uint64_t));
    uint64_t* h_columnIndeces = (uint64_t*) calloc(tnv, sizeof(uint64_t));
    double* h_values = (double*) calloc(tnv, sizeof(double));

    for (uint64_t indS = 0; indS < numSlice; ++indS)
    {
        h_offsets[indS] = indS * sliceSizeTemp * maxNumColsTemp;
        for (uint64_t indC = 0; indC < maxNumColsTemp; ++indC)
        {
            for (uint64_t indRinS = 0; indRinS < sliceSizeTemp; ++indRinS)
            {
                uint64_t indR = indS * sliceSizeTemp + indRinS;
                uint64_t lin_idx = (indS * sliceSizeTemp * maxNumColsTemp) + (indC * sliceSizeTemp) + indRinS;

                int64_t col_idx = NOLINK;
                double value = 0;
                if(indR < numNodes)
                {
                    col_idx = (int64_t) matA[indR][indC].index;
                    if(col_idx != NOLINK)
                    {
                        if(indR != col_idx) //Omit diag elements
                            value = (double) matA[indR][indC].val;

                        nnz++;
                    }
                }
                h_columnIndeces[lin_idx] = col_idx;
                h_values[lin_idx] = value;
            }
        }
    }
    h_offsets[numSlice] = tnv;

    iterationMatrix.totValuesSize = tnv;
    iterationMatrix.numNonZeroElement = nnz;

    cudaMalloc((void**) &(iterationMatrix.d_offsets), (numSlice + 1) * sizeof(uint64_t));
    cudaMalloc((void**) &(iterationMatrix.d_columnIndeces), tnv * sizeof(uint64_t));
    cudaMalloc((void**) &(iterationMatrix.d_values), tnv * sizeof(double));

    cudaMemcpy(iterationMatrix.d_offsets, h_offsets, (numSlice + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(iterationMatrix.d_columnIndeces, h_columnIndeces, tnv * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(iterationMatrix.d_values, h_values, tnv * sizeof(double), cudaMemcpyHostToDevice);

    cusparseCreateConstSlicedEll(&(iterationMatrix.cusparseDescriptor),
                                 iterationMatrix.numRows,
                                 iterationMatrix.numCols,
                                 iterationMatrix.numNonZeroElement,
                                 iterationMatrix.totValuesSize,
                                 iterationMatrix.sliceSize,
                                 iterationMatrix.d_offsets,
                                 iterationMatrix.d_columnIndeces,
                                 iterationMatrix.d_values,
                                 iterationMatrix.offsetType,
                                 iterationMatrix.colIdxType,
                                 iterationMatrix.idxBase,
                                 iterationMatrix.valueType);

    constantTerm.numElements = numNodes;
    cudaMalloc((void**) &(constantTerm.d_values), numNodes * sizeof(double));
    cudaMemcpy(constantTerm.d_values, vecB, numNodes * sizeof(double), cudaMemcpyHostToDevice);
    cusparseCreateDnVec(&(constantTerm.cusparseDescriptor), constantTerm.numElements, constantTerm.d_values, constantTerm.valueType);

    tempSolution1.numElements = numNodes;
    cudaMalloc((void**) &(tempSolution1.d_values), numNodes * sizeof(double));
    cudaMemcpy(tempSolution1.d_values, vecX, numNodes * sizeof(double), cudaMemcpyHostToDevice);
    cusparseCreateDnVec(&(tempSolution1.cusparseDescriptor), tempSolution1.numElements, tempSolution1.d_values, tempSolution1.valueType);

    tempSolution2.numElements = numNodes;
    cudaMalloc((void**) &(tempSolution2.d_values), numNodes * sizeof(double));
    cudaMemcpy(tempSolution2.d_values, vecB, numNodes * sizeof(double), cudaMemcpyHostToDevice);
    cusparseCreateDnVec(&(tempSolution2.cusparseDescriptor), tempSolution2.numElements, tempSolution2.d_values, tempSolution2.valueType);

    destructHostPointer(h_offsets);
    destructHostPointer(h_columnIndeces);
    destructHostPointer(h_values);
}


void run_k(cusparseHandle_t libHandle, MatrixGPU &iterationMatrix, VectorGPU &constantTerm, VectorGPU &tempSolution1, VectorGPU &tempSolution2)
{
    size_t bufSize;
    const double alpha = -1, beta = 1;
    void *externalBuffer;

    cusparseSpMV_bufferSize(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, tempSolution1.cusparseDescriptor, &beta, tempSolution2.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
    cudaDeviceSynchronize();
    cudaMalloc(&externalBuffer, bufSize);

    //x2 = b
    //x1 = x(0)
    //x2 = alpha * A * x1 + beta * x2
    //x1 = b
    //x1 = alpha * A * x2 + beta * x1

    size_t numIterTemp = 200; //Real: setted in _parameters
    for (size_t i = 0; i < numIterTemp; ++i)
    {
        cusparseSpMV(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, tempSolution1.cusparseDescriptor, &beta, tempSolution2.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, externalBuffer);
        cudaMemcpy(tempSolution1.d_values, constantTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
        cusparseSpMV(libHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, iterationMatrix.cusparseDescriptor, tempSolution2.cusparseDescriptor, &beta, tempSolution1.cusparseDescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, externalBuffer);
        cudaMemcpy(tempSolution2.d_values, constantTerm.d_values, constantTerm.numElements * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaFree(externalBuffer);
}

void gatherOutput_k(VectorGPU &tempSolution1, double* &vecX)
{
    if(vecX == nullptr)
        vecX = (double*) malloc(tempSolution1.numElements * sizeof(double));

    cudaMemcpy(vecX, tempSolution1.d_values, tempSolution1.numElements * sizeof(double), cudaMemcpyDeviceToHost);
}


void destructCUSPARSEobjects(cusparseHandle_t &libHandle, MatrixGPU &iterationMatrix, VectorGPU &constantTerm, VectorGPU &tempSolution1, VectorGPU &tempSolution2)
{
    destructDevicePointer(iterationMatrix.d_columnIndeces);
    destructDevicePointer(iterationMatrix.d_offsets);
    destructDevicePointer(iterationMatrix.d_values);
    cusparseDestroySpMat(iterationMatrix.cusparseDescriptor);

    destructDevicePointer(constantTerm.d_values);
    cusparseDestroyDnVec(constantTerm.cusparseDescriptor);

    destructDevicePointer(tempSolution1.d_values);
    cusparseDestroyDnVec(tempSolution1.cusparseDescriptor);
    destructDevicePointer(tempSolution2.d_values);
    cusparseDestroyDnVec(tempSolution2.cusparseDescriptor);

    cusparseDestroy(libHandle);

}
