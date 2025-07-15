#ifndef SOILFLUXES3D_GPUSOLVER_H
#define SOILFLUXES3D_GPUSOLVER_H

#include "solver_new.h"
#include "types_gpu.h"
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

        public:
            GPUSolver(numericalMethod method = Jacobi);
            ~GPUSolver();

            void inizialize() override;
            void run() override;
            void gatherOutput(double *&vecX) override;

            //Temp function
            void copyMatrixVectorFromOld(TmatrixElement **matA, double *vecB, double *vecX, uint64_t numNodes);
    };

#define destructDevicePointer(ptr) {if(ptr != nullptr) {cudaFree(ptr); ptr = nullptr;}}
#define destructHostPointer(ptr) {if(ptr != nullptr) {free(ptr); ptr = nullptr;}}
} // namespace soilFluxes3D_New

#endif // SOILFLUXES3D_GPUSOLVER_H
