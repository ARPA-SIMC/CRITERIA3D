#include <stdio.h>
#include <math.h>
#include "wg2D.h"

void weatherGenerator2D::randomSet(double *arrayNormal,int dimArray)
{
    double randomArrayNormal[10000];
    for (int i=0; i<10000; i++)
    {
        randomArrayNormal[i] = normalRandomNumbers[i];
    }
    for (int i=0; i<dimArray; i++)
    {
        int j;
        j = i%10000;
        arrayNormal[i] = randomArrayNormal[j];
    }



}
