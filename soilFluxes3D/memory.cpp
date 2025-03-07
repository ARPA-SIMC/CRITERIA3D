/*!
    \name memory.cpp
    \copyright (C) 2011 Fausto Tomei, Gabriele Antolini, Antonio Volta,
                        Alberto Pistocchi, Marco Bittelli

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include <stdio.h>
#include <stdlib.h>
#include "header/types.h"


void cleanArrays()
{
    /*! free matrix A */
    if (A != nullptr)
    {
            for (long i=0; i < myStructure.nrNodes; i++)
            {
                if (A[i] != nullptr)
                    free(A[i]);
            }
            free(A);
            A = nullptr;
    }

    /*! free arrays */
    if (invariantFlux != nullptr) { free(invariantFlux); invariantFlux = nullptr; }
    if (b != nullptr) { free(b); b = nullptr; }
    if (C != nullptr) { free(C); C = nullptr; }
    if (X != nullptr) { free(X); X = nullptr; }
    if (X1 != nullptr) { free(X1); X1 = nullptr; }
}


void cleanNodes()
{
    if (nodeList != nullptr)
    {
        for (long i = 0; i < myStructure.nrNodes; i++)
        {
            if (nodeList[i].boundary != nullptr) free(nodeList[i].boundary);
            free(nodeList[i].lateral);
        }
        free(nodeList);
        nodeList = nullptr;
    }
}


/*!
 * \brief initialize matrix and arrays
 * \return OK/ERROR
 */
int initializeArrays()
{
    /*! clean previous arrays */
    cleanArrays();

    /*! matrix A: rows */
    A = (TmatrixElement **) calloc(myStructure.nrNodes, sizeof(TmatrixElement *));

    /*! matrix A: columns */
    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        A[i] = (TmatrixElement *) calloc(myStructure.maxNrColumns, sizeof(TmatrixElement));
    }

    /*! initialize matrix A */
    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        for (int j = 0; j < myStructure.maxNrColumns; j++)
        {
            A[i][j].index   = NOLINK;
            A[i][j].val     = 0.;
        }
    }

    b = (double *) calloc(myStructure.nrNodes, sizeof(double));
    for (long i = 0; i < myStructure.nrNodes; i++)
    {
        b[i] = 0.;
    }

    X = (double *) calloc(myStructure.nrNodes, sizeof(double));

    if (myParameters.numericalSolutionMethod == JACOBI)
    {
        X1 = (double *) calloc(myStructure.nrNodes, sizeof(double));
    }

    /*! mass diagonal matrix */
    C = (double *) calloc(myStructure.nrNodes, sizeof(double));
    invariantFlux = (double *) calloc(myStructure.nrNodes, sizeof(double));
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        C[n] = 0.;
        invariantFlux[n] = 0.;
    }

    if (A == nullptr)
        return MEMORY_ERROR;
    else
        return CRIT3D_OK;
}
