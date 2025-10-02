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

            SF3Duint_t numThreadsPerBlock;
            SF3Duint_t numBlocks;

            void waterMainLoop(double maxTimeStep, double& acceptedTimeStep);
            balanceResult_t waterApproximationLoop(double deltaT);
            bool solveLinearSystem(uint8_t approximationNumber, processType computationType) override;

            //TEMP: maybe can be unified with CPU code in single __host__ __device__ function
            balanceResult_t evaluateWaterBalance_m(uint8_t approxNr, double& bestMBRerror, double deltaT);
            void computeCurrentMassBalance_m(double deltaT);
            double computeTotalWaterContent_m();
            double computeWaterSinkSourceFlowsSum_m(double deltaT);
            void acceptStep_m(double deltaT);
            void restoreBestStep_m(double deltaT);

            SF3Derror_t upCopyData();
            SF3Derror_t upMoveSoilSurfacePtr();
            SF3Derror_t downCopyData();
            SF3Derror_t downMoveSoilSurfacePtr();
            SF3Derror_t createCUsparseDescriptors();

        public:
            GPUSolver() : Solver(solverType::GPU, numericalMethod::Jacobi) {}

            __cudaSpec double getMatrixElementValue(SF3Duint_t rowIndex, SF3Duint_t colIndex) const noexcept;

            SF3Derror_t initialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
            SF3Derror_t clean() override;
    };

    inline __cudaSpec double GPUSolver::getMatrixElementValue(SF3Duint_t rowIndex, SF3Duint_t colIndex) const noexcept
    {
        //Valori diagonali
        if(rowIndex == colIndex)
            return iterationMatrix.d_diagonalValues[rowIndex];

        SF3Duint_t sliceIndex = static_cast<SF3Duint_t>(floor((double) rowIndex / iterationMatrix.sliceSize));
        SF3Duint_t baseIndex = static_cast<SF3Duint_t>(iterationMatrix.d_offsets[sliceIndex]);
        SF3Duint_t pOffIndex = rowIndex % iterationMatrix.sliceSize;

        for(uint16_t colSELLIdx = 0; colSELLIdx < iterationMatrix.d_numColsInRow[rowIndex]; ++colSELLIdx)
        {
            SF3Duint_t finalIndex = baseIndex + pOffIndex + (colSELLIdx * iterationMatrix.sliceSize);
            if(iterationMatrix.d_columnIndeces[finalIndex] == colIndex)
                return iterationMatrix.d_values[finalIndex] * iterationMatrix.d_diagonalValues[rowIndex];
        }

        return 0.;
    }

    __global__ void initializeCapacityAndSaturationDegree_k(double *vectorC);
    __global__ void computeCapacity_k(double *vectorC);
    __global__ void updateBoundaryWaterData_k(double deltaT);

    __global__ void computeLinearSystemElement_k(MatrixGPU matrixA, VectorGPU vectorB, const double* Cvalues, uint8_t approxNum, double deltaT, double lateralVerticalRatio, meanType_t meanType);
    __global__ void computeNormalizedError(double *vectorNorm, double *vectorX, const double *previousX);

    __global__ void updateSaturationDegree_k();
    __global__ void updateWaterConductivity_k();
    __global__ void updateWaterFlows_k(double deltaT);
    __global__ void computeWaterContent_k(double* outVector);


    template<typename deviceError_t>
    inline SF3Derror_t solverDeviceCheckError(deviceError_t retError, solverStatus& status, const SF3Derror_t contextErrorType)
    {
        if(retError == static_cast<deviceError_t>(0))
            return SF3Derror_t::SF3Dok;

        status = solverStatus::Error;
        return contextErrorType;
    }

} // namespace soilFluxes3D::New

#endif // SOILFLUXES3D_GPUSOLVER_H
