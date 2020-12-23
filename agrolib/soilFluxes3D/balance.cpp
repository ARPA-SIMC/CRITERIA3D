/*!
    \name balance.cpp
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
    fausto.tomei@gmail.com
    ftomei@arpae.emr.it
    gantolini@arpae.emr.it
*/


#include <stdio.h>
#include <math.h>

#include "../mathFunctions/commonConstants.h"
#include "header/types.h"
#include "header/balance.h"
#include "header/soilPhysics.h"
#include "header/solver.h"
#include "header/boundary.h"
#include "header/heat.h"
#include "header/water.h"


Tbalance balanceCurrentTimeStep, balancePreviousTimeStep, balanceCurrentPeriod, balanceWholePeriod;
double Courant = 0.0;

static double bestMBRerror;
static bool isHalfTimeStepForced = false;


inline void doubleTimeStep()
{
    myParameters.current_delta_t *= 2.0;
    myParameters.current_delta_t = MINVALUE(myParameters.current_delta_t, myParameters.delta_t_max);
}


void halveTimeStep()
{
    myParameters.current_delta_t /= 2.0;
    myParameters.current_delta_t = MAXVALUE(myParameters.current_delta_t, myParameters.delta_t_min);
}


void InitializeBalanceWater()
{
     bestMBRerror = 100.;

     balanceWholePeriod.storageWater = computeTotalWaterContent();
     balanceCurrentTimeStep.storageWater = balanceWholePeriod.storageWater;
     balancePreviousTimeStep.storageWater = balanceWholePeriod.storageWater;
     balanceCurrentPeriod.storageWater = balanceWholePeriod.storageWater;

     balanceCurrentTimeStep.sinkSourceWater = 0.;
     balancePreviousTimeStep.sinkSourceWater = 0.;
     balanceCurrentTimeStep.waterMBR = 0.;
     balanceCurrentTimeStep.waterMBE = 0.;
     balanceCurrentPeriod.sinkSourceWater = 0.;
     balanceWholePeriod.sinkSourceWater = 0.;
     balanceWholePeriod.waterMBE = 0.;
     balanceWholePeriod.waterMBR = 0.;

    /*! initialize link flow */
    for (long n = 0; n < myStructure.nrNodes; n++)
        {
        myNode[n].up.sumFlow = 0.;
        myNode[n].down.sumFlow = 0.;
        for (short i = 0; i < myStructure.nrLateralLinks; i++)
             myNode[n].lateral[i].sumFlow = 0.;
        }

    /*! initialize boundary flow */
    for (long n = 0; n < myStructure.nrNodes; n++)
        if (myNode[n].boundary != nullptr)
            myNode[n].boundary->sumBoundaryWaterFlow = 0.;
}


/*!
 * \brief computes total water content          [m^3]
 * \return result
 */
double computeTotalWaterContent()
{
   double theta, sum = 0.0;

   for (unsigned long i = 0; i < unsigned(myStructure.nrNodes); i++)
       if  (myNode[i].isSurface)
       {
           sum += (myNode[i].H - double(myNode[i].z)) * myNode[i].volume_area;
       }
       else
       {
           theta = theta_from_Se(i);
           sum += theta * myNode[i].volume_area;
       }
   return(sum);
}


/*!
 * \brief computes sum of water sink/source          [m^3]
 * \param deltaT
 * \return result
 */
double sumWaterFlow(double deltaT)
{
    double sum = 0.0;
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        if (myNode[n].Qw != 0.)
            sum += myNode[n].Qw * deltaT;
    }
    return (sum);
}



void computeMassBalance(double deltaT)
{
     balanceCurrentTimeStep.storageWater = computeTotalWaterContent();

	 double dStorage = balanceCurrentTimeStep.storageWater - balancePreviousTimeStep.storageWater;

     balanceCurrentTimeStep.sinkSourceWater = sumWaterFlow(deltaT);

     balanceCurrentTimeStep.waterMBE = dStorage - balanceCurrentTimeStep.sinkSourceWater;

     /*! reference water: sumWaterFlow or 0.1% of storage */
     double denominator = MAXVALUE(fabs(balanceCurrentTimeStep.sinkSourceWater), balanceCurrentTimeStep.storageWater * 1e-3);

     /*! no water - minimum 1 liter */
     denominator = MAXVALUE(denominator, 0.001);

	 balanceCurrentTimeStep.waterMBR = balanceCurrentTimeStep.waterMBE / denominator;
}


double getMatrixValue(long i, TlinkedNode *link)
{
	if (link != nullptr)
        {
        int j = 1;
        while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK) && (A[i][j].index != (*link).index)) j++;

        /*! Rebuild the A elements (previously normalized) */
		if (A[i][j].index == (*link).index)
			return (A[i][j].val * A[i][0].val);
        }
	return double(INDEX_ERROR);
}


/*!
 * \brief updating of in- and out-flows [m^3]
 * \param index
 * \param link TlinkedNode pointer
 * \param delta_t
 */
void update_flux(long index, TlinkedNode *link, double delta_t)
{
    if (link->index != NOLINK)
        (*link).sumFlow += float(getWaterExchange(index, link, delta_t));
}



void saveBestStep()
{
	for (long n = 0; n < myStructure.nrNodes; n++)
		myNode[n].bestH = myNode[n].H;
}




void restoreBestStep(double deltaT)
{
    for (unsigned long n = 0; n < unsigned(myStructure.nrNodes); n++)
    {
        myNode[n].H = myNode[n].bestH;

        /*! compute new soil moisture (only sub-surface nodes) */
        if (!myNode[n].isSurface)
                myNode[n].Se = computeSe(n);
    }

     computeMassBalance(deltaT);
}


void acceptStep(double deltaT)
{
    /*! update balanceCurrentPeriod and balanceWholePeriod */
    balancePreviousTimeStep.storageWater = balanceCurrentTimeStep.storageWater;
    balancePreviousTimeStep.sinkSourceWater = balanceCurrentTimeStep.sinkSourceWater;
    balanceCurrentPeriod.sinkSourceWater += balanceCurrentTimeStep.sinkSourceWater;

    /*! update sum of flow */
    for (long i = 0; i < myStructure.nrNodes; i++)
        {
		update_flux(i, &(myNode[i].up), deltaT);
        update_flux(i, &(myNode[i].down), deltaT);
        for (short j = 0; j < myStructure.nrLateralLinks; j++)
			update_flux(i, &(myNode[i].lateral[j]), deltaT);

        if (myNode[i].boundary != nullptr)
            myNode[i].boundary->sumBoundaryWaterFlow += myNode[i].boundary->waterFlow * deltaT;
        }

}

bool waterBalance(double deltaT, int approxNr)
{
	computeMassBalance(deltaT);
	double MBRerror = fabs(balanceCurrentTimeStep.waterMBR);

	isHalfTimeStepForced = false;

    /*! error better than previuosly */
	if ((approxNr == 0) || (MBRerror < bestMBRerror))
	{
		saveBestStep();
		bestMBRerror = MBRerror;
	}

    /*! best case */
    if (MBRerror < myParameters.MBRThreshold)
        {
        acceptStep(deltaT);
		if ((approxNr < 2) && (Courant < 0.5) && (MBRerror < (myParameters.MBRThreshold * 0.5)))
            {
            /*! system is stable: double time step */
            doubleTimeStep();
            }
        return (true);
        }

    /*! worst case: error high or last approximation */
    if ((MBRerror > (bestMBRerror * 2.0))
        ||(approxNr == (myParameters.maxApproximationsNumber-1)))
        {
        if (deltaT > myParameters.delta_t_min)
            {
            halveTimeStep();
            isHalfTimeStepForced = true;
            return (false);
            }
        else
            {
            restoreBestStep(deltaT);
            acceptStep(deltaT);
            return (true);
            }
        }

    // default
    return (false);
}



void updateBalanceWaterWholePeriod()
{
    /*! update the flows in the balance (balanceWholePeriod) */
    balanceWholePeriod.sinkSourceWater  += balanceCurrentPeriod.sinkSourceWater;

    double deltaStoragePeriod = balanceCurrentTimeStep.storageWater - balanceCurrentPeriod.storageWater;

    double deltaStorageHistorical = balanceCurrentTimeStep.storageWater - balanceWholePeriod.storageWater;

    /*! compute waterMBE and waterMBR */
    balanceCurrentPeriod.waterMBE = fabs(deltaStoragePeriod - balanceCurrentPeriod.sinkSourceWater);
    if ((balanceWholePeriod.storageWater == 0.) && (balanceWholePeriod.sinkSourceWater == 0.)) balanceWholePeriod.waterMBR = 1.;
    else if (balanceCurrentTimeStep.storageWater > fabs(balanceWholePeriod.sinkSourceWater))
        balanceWholePeriod.waterMBR = balanceCurrentTimeStep.storageWater / (balanceWholePeriod.storageWater + balanceWholePeriod.sinkSourceWater);
    else
        balanceWholePeriod.waterMBR = deltaStorageHistorical / balanceWholePeriod.sinkSourceWater;

    /*! update storageWater in balanceCurrentPeriod */
    balanceCurrentPeriod.storageWater = balanceCurrentTimeStep.storageWater;
}



bool getForcedHalvedTime()
{
    return (isHalfTimeStepForced);
}

void setForcedHalvedTime(bool isForced)
{
    isHalfTimeStepForced = isForced;
}

