#ifndef SOILFLUXES3D_GPUSOLVER_H
#define SOILFLUXES3D_GPUSOLVER_H

#include "solver.h"
#include "types_gpu.h"

namespace soilFluxes3D::New
{
    class GPUSolver : public Solver
    {
        private:
            cusparseHandle_t libHandle;

            MatrixGPU iterationMatrix;              //A
            VectorGPU constantTerm, unknownTerm;    //b e x
            VectorGPU tempSolution;                 //tempX

            double* d_Cvalues = nullptr;
            surfaceData_t* d_surfaceList = nullptr;
            soilData_t* d_soilList = nullptr;

            uint64_t numThreadsPerBlock;
            uint64_t numBlocks;

            void computeMatrix();
            void computeVector();

            void waterMainLoop(double maxTimeStep, double& acceptedTimeStep);
            balanceResult_t waterApproximationLoop(double deltaT);

            bool solveLinearSystem(uint8_t approximationNumber, processType computationType) override;
            SF3Derror_t upCopyData();
            SF3Derror_t upMoveSoilSurfacePtr();
            SF3Derror_t downCopyData();
            SF3Derror_t downMoveSoilSurfacePtr();
            SF3Derror_t createCUsparseDescriptors();

            /*temp*/ nodesData_t* ptr = nullptr;
            /*temp*/ MatrixCPU matrixA;
            /*temp*/ VectorCPU vectorB, vectorX;

            /*temp*/ VectorCPU vectorC;

        public:
            GPUSolver() : Solver(GPU, Jacobi) {}

            __cudaSpec double getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex) override;

            SF3Derror_t inizialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
            SF3Derror_t clean() override;

            SF3Derror_t gatherOutput(double *&vecX);
    };

    inline __cudaSpec double GPUSolver::getMatrixElementValue(uint64_t rowIndex, uint64_t colIndex)
    {
        /*temp*/ uint8_t cpuColIdx;
        /*temp*/ for(cpuColIdx = 0; cpuColIdx < matrixA.numColumns[rowIndex]; ++cpuColIdx)
        /*temp*/     if(matrixA.colIndeces[rowIndex][cpuColIdx] == colIndex)
        /*temp*/         break;

        /*temp*/ //assert(cpuColIdx < matrixA.numColumns[rowIndex]);
        /*temp*/ return matrixA.values[rowIndex][cpuColIdx] * matrixA.values[rowIndex][0];


        uint64_t sliceIndex = (uint64_t)(floor((double) rowIndex / iterationMatrix.sliceSize));
        uint64_t baseIndex = (uint64_t)(iterationMatrix.d_offsets[sliceIndex]);
        uint64_t pOffIndex = rowIndex % iterationMatrix.sliceSize;

        for(uint16_t colSELLIdx = 0; colSELLIdx < iterationMatrix.d_numColsInRow[rowIndex]; ++colSELLIdx)
        {
            uint64_t finalIndex = baseIndex + pOffIndex + (colSELLIdx * iterationMatrix.sliceSize);
            if(iterationMatrix.d_columnIndeces[finalIndex] == colIndex)
                return iterationMatrix.d_values[finalIndex];
        }

        return 0.;
    }

    __global__ void init_SurfaceC_SubSurfaceSe(double *vectorC);
    __global__ void updateBoundaryWaterData_k(double deltaT);
    __global__ void updateSaturationDegree_k();

    __global__ void computeCapacity_k(double *vectorC);
    __global__ void computeLinearSystemElement_k(MatrixGPU matrixA, VectorGPU vectorB, const double* Cvalues, uint8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType);

    __global__ void computeNormalizedError(double *vectorNorm, double *vectorX, const double *previousX);

} // namespace soilFluxes3D::New

#endif // SOILFLUXES3D_GPUSOLVER_H
