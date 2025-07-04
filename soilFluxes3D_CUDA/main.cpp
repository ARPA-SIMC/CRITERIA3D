#include <stdlib.h>
#include "dataTypes.h"

extern void CUDAsoilEntryPoint(double** matrixA, double* vectorB, double*& vectorX, int sisSize);

int main()
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

    CUDAsoilEntryPoint(matrixTest, vectorTest, vectorSol, 3);
    for (int idx = 0; idx < 3; ++idx)
        dbg(vectorSol[idx]);
}
