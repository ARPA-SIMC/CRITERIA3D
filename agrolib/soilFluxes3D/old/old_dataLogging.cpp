#include <QString>
#include "old_soilFluxes3D.h"
#include "old_types.h"

//void runCUSPARSEtest(double* &vecSol);

namespace soilFluxes3D
{
    __EXTERN QString getMatrixLog()
    {
        QString matrixString = "";

        //matrixString.append("[");
        for (long i = 0; i < myStructure.nrNodes; i++)
        {
            if (i > 0)
                matrixString.append(";");

            for (int j = 0; j < myStructure.maxNrColumns; j++)
            {
                if (A[i][j].index == NOLINK)
                    break;

                if (j > 0)
                    matrixString.append(",");

                matrixString.append(QString::number(A[i][j].index));
                matrixString.append(":");
                matrixString.append(QString::number(A[i][j].val));
            }
        }
        //matrixString.append("]");

        return matrixString;
    }

    __EXTERN QString getMatrixLog_formatted()
    {
        QString valueString = "";
        QString indexString_r = "";
        QString indexString_c = "";

        for (long i = 0; i < myStructure.nrNodes; i++)
            for (int j = 0; j < myStructure.maxNrColumns; j++)
            {
                if (A[i][j].index == NOLINK)
                    break;

                if (j > 0)
                {
                    indexString_r.append(",");
                    indexString_c.append(",");
                    valueString.append(",");
                }

                indexString_r.append(QString::number(i));
                indexString_c.append(QString::number(A[i][j].index));
                valueString.append(QString::number(A[i][j].val));
            }

        return indexString_r + "\n" + indexString_c + "\n" + valueString;
    }

    __EXTERN QString getVectorLog()
    {
        QString vectorString = "";

        //vectorString.append("[");
        for (long i = 0; i < myStructure.nrNodes; i++)
        {
            if (i > 0)
                vectorString.append(",");

            vectorString.append(QString::number(nodeList[i].H));
        }
        //vectorString.append("]");

        return vectorString;
    }

    __EXTERN QString getLinSystLog()
    {
        QString stringData = "TypeSolver = ";
        switch (logLinSyst.solver)
        {
            case Jacobi_thread:
                stringData.append("Jacobi_thread");
                break;
            case GaussSeidel_thread:
                stringData.append("GaussSeidel_thread");
                break;
            case Jacobi_openMP:
                stringData.append("Jacobi_openMP");
                break;
            case GaussSeidel_openMP:
                stringData.append("GaussSeidel_openMP");
                break;
            case Jacobi_cusparse:
                stringData.append("Jacobi_cusparse");
                break;
            default:
                stringData.append("Unknown method");
        }
        stringData.append(" - N.Approx = " + QString::number(logLinSyst.numberApprox));
        stringData.append("\nNumero iterazioni effettive: ");
        for (auto it = logLinSyst.numberIterations.begin(); it != logLinSyst.numberIterations.end(); ++it)
            stringData.append(QString::number(*it).append(" "));

        stringData.append("\nNumero iterazioni massimo: ");
        for (auto it = logLinSyst.maxNumberIterations.begin(); it != logLinSyst.maxNumberIterations.end(); ++it)
            stringData.append(QString::number(*it).append(" "));

        return stringData;
    }

    __EXTERN QString getCUDArun(int x)
    {
        double* vecSol = nullptr;
        //runCUSPARSEtest(vecSol);

        QString stringData = "TestCUDA NEW - ";
        for (int i = 0; i < 5; ++i)
        {
            stringData.append(QString::number(vecSol[i]));
            stringData.append(" ");
        }
        return stringData.append(" - correct if ").append(QString::number(x));
    }
}

linSystData logLinSyst;
