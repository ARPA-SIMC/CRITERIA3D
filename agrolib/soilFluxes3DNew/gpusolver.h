#ifndef SOILFLUXES3D_GPUSOLVER_H
#define SOILFLUXES3D_GPUSOLVER_H

#include "solver_new.h"
#include "types_gpu.h"

namespace soilFluxes3D::New
{
    class GPUSolver : public Solver
    {
        private:
            cusparseHandle_t libHandle;

            MatrixGPU iterationMatrix;              //A
            VectorGPU constantTerm;                 //b
            VectorGPU tempSolution1, tempSolution2; //X e oldX

            void computeMatrix();
            void computeVector();

            bool solveLinearSystem(uint8_t approximationNumber, processType computationType) override;

        public:
            GPUSolver(numericalMethod method = Jacobi);
            ~GPUSolver();

            SF3Derror_t inizialize() override;
            SF3Derror_t run(double maxTimeStep, double &acceptedTimeStep, processType process) override;
            SF3Derror_t gatherOutput(double *&vecX);

            //Temp function
            void copyMatrixVectorFromOld(TmatrixElement **matA, double *vecB, double *vecX, uint64_t numNodes);
    };

    #define destructDevicePointer(ptr) {if(ptr != nullptr) {cudaFree(ptr); ptr = nullptr;}}
    #define destructHostPointer(ptr) {if(ptr != nullptr) {free(ptr); ptr = nullptr;}}
} // namespace soilFluxes3D::New

#endif // SOILFLUXES3D_GPUSOLVER_H
