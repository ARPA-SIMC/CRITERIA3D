#include <iostream>
#include <string>
#include "iterationkernel.h"

#include "entryPointTest.h"

using namespace soilFluxes3D_CUDA;

void CUDAsoilEntryPoint(double** matrixA, double* vectorB, double*& vectorX, int sisSize)
{
    iterationKernel testKernel(20);

    testKernel.inizializeKernelData(matrixA, vectorB, sisSize);
    testKernel.launchKernel();
    testKernel.getKernelOutput(vectorX);
}

std::string CUDAsoilEntryPointProject()
{
    double **matrixTest = (double**) malloc(3*sizeof(double*));
    matrixTest[0] = (double*) calloc(3, sizeof(double));
    matrixTest[1] = (double*) calloc(3, sizeof(double));
    matrixTest[2] = (double*) calloc(3, sizeof(double));
    matrixTest[0][0] = 4;
    matrixTest[1][1] = 2;
    matrixTest[2][1] = 1;
    matrixTest[2][2] = 3;

    double *vectorTest;
    vectorTest = (double*) calloc(3, sizeof(double));
    vectorTest[0] = 1;
    vectorTest[1] = 2;
    vectorTest[2] = 3;

    double *vectorSol;
    vectorSol = (double*) calloc(3, sizeof(double));
    vectorSol[0] = 0;
    vectorSol[1] = 0;
    vectorSol[2] = 0;

    iterationKernel testKernel(500);

    testKernel.inizializeKernelData(matrixTest, vectorTest, 3);
    destrucHostPointer(matrixTest);
    destrucHostPointer(vectorTest);
    testKernel.launchKernel();
    testKernel.getKernelOutput(vectorSol);

    std::string outputString = "";

    for (int idx = 0; idx < 3; ++idx)
        outputString += std::to_string(vectorSol[idx]);

    destrucHostPointer(vectorSol);
    return outputString;
}
