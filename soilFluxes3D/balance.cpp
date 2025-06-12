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
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include <stdio.h>
#include <math.h>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"
#include "header/types.h"
#include "header/balance.h"
#include "header/soilPhysics.h"
#include "header/solver.h"
#include "header/boundary.h"
#include "header/heat.h"
#include "header/water.h"


Tbalance balanceCurrentTimeStep, balancePreviousTimeStep, balanceCurrentPeriod, balanceWholePeriod;
double CourantWater = 0.0;

static double _bestMBRerror;
static bool _isHalfTimeStepForced = false;


inline void doubleTimeStep()
{
    myParameters.current_delta_t *= 2.0;
    myParameters.current_delta_t = std::min(myParameters.current_delta_t, myParameters.delta_t_max);
}


void halveTimeStep()
{
    myParameters.current_delta_t /= 2.0;
    myParameters.current_delta_t = std::max(myParameters.current_delta_t, myParameters.delta_t_min);
}


void InitializeBalanceWater()
{
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

    // initialize link flow
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        nodeList[n].up.sumFlow = 0.;
        nodeList[n].down.sumFlow = 0.;
        for (short i = 0; i < myStructure.nrLateralLinks; i++)
        {
             nodeList[n].lateral[i].sumFlow = 0.;
        }
    }

    // initialize boundary flow
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        if (nodeList[n].boundary != nullptr)
            nodeList[n].boundary->sumBoundaryWaterFlow = 0.;
    }
}


/*!
 * \brief computes the total water content
 * \return total water content [m3]
 */
double computeTotalWaterContent()
{
   double sum = 0.0;

   for (unsigned long i = 0; i < unsigned(myStructure.nrNodes); i++)
   {
       if (nodeList[i].isSurface)
       {
           sum += (nodeList[i].H - double(nodeList[i].z)) * nodeList[i].volume_area;
       }
       else
       {
           sum += theta_from_Se(i) * nodeList[i].volume_area;
       }
   }

   return sum;
}


/*!
 * \brief computes sum of water sink/source
 * \param deltaT    [s]
 * \return sum of water sink/source  [m3]
 */
double sumWaterFlow(double deltaT)
{
    double sum = 0.0;
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        if (nodeList[n].Qw != 0.)
            sum += nodeList[n].Qw * deltaT;
    }
    return sum;
}


/*!
 * \brief computes the mass balance error in balanceCurrentTimeStep
 * \param deltaT    [s]
 */
void computeMassBalance(double deltaT)
{
    balanceCurrentTimeStep.storageWater = computeTotalWaterContent();                               // [m3]

    double dStorage = balanceCurrentTimeStep.storageWater - balancePreviousTimeStep.storageWater;   // [m3]

    balanceCurrentTimeStep.sinkSourceWater = sumWaterFlow(deltaT);                                  // [m3]

    balanceCurrentTimeStep.waterMBE = dStorage - balanceCurrentTimeStep.sinkSourceWater;            // [m3]

    // minimum reference water storage [m3] as % of current storage
    double timePercentage = 0.01 * std::max(deltaT, 1.) / HOUR_SECONDS;
    double minRefWaterStorage = balanceCurrentTimeStep.storageWater * timePercentage;
    minRefWaterStorage = std::max(minRefWaterStorage, 0.001);                                       // [m3] minimum 1 liter

    // reference water for computation of mass balance error ratio
    // when the water sink/source is too low, use the reference water storage
    double referenceWater = std::max(fabs(balanceCurrentTimeStep.sinkSourceWater), minRefWaterStorage);     // [m3]

    balanceCurrentTimeStep.waterMBR = balanceCurrentTimeStep.waterMBE / referenceWater;
}


double getMatrixValue(long i, TlinkedNode *link)
{
	if (link != nullptr)
    {
        int j = 1;
        while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK) && (A[i][j].index != (*link).index))
            j++;

        /*! Rebuild the A elements (previously normalized) */
		if (A[i][j].index == (*link).index)
        {
			return (A[i][j].val * A[i][0].val);
        }
    }

	return double(INDEX_ERROR);
}


/*!
 * \brief updates in and out flows [m3]
 * \param index
 * \param link      TlinkedNode pointer
 * \param delta_t   [s]
 */
void update_flux(long index, TlinkedNode *link, double delta_t)
{
    if (link->index != NOLINK)
    {
        (*link).sumFlow += float(getWaterExchange(index, link, delta_t));       // [m3]
    }
}


void saveBestStep()
{
	for (long n = 0; n < myStructure.nrNodes; n++)
    {
        nodeList[n].bestH = nodeList[n].H;
    }
}


void restoreBestStep(double deltaT)
{
    for (unsigned long n = 0; n < unsigned(myStructure.nrNodes); n++)
    {
        nodeList[n].H = nodeList[n].bestH;

        /*! compute new soil moisture (only sub-surface nodes) */
        if (! nodeList[n].isSurface)
        {
            nodeList[n].Se = computeSe(n);
        }
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
        update_flux(i, &(nodeList[i].up), deltaT);
        update_flux(i, &(nodeList[i].down), deltaT);
        for (short j = 0; j < myStructure.nrLateralLinks; j++)
        {
            update_flux(i, &(nodeList[i].lateral[j]), deltaT);
        }

        if (nodeList[i].boundary != nullptr)
            nodeList[i].boundary->sumBoundaryWaterFlow += nodeList[i].boundary->waterFlow * deltaT;
    }
}


bool waterBalance(double deltaT, int approxNr)
{
    setForcedHalvedTime(false);

    computeMassBalance(deltaT);
	double MBRerror = fabs(balanceCurrentTimeStep.waterMBR);

    // good error: step is accepted
    if (MBRerror < myParameters.MBRThreshold)
    {
        acceptStep(deltaT);

        // best case: system is stable, try to increase time step
        if (CourantWater < 0.8 && approxNr <= 3 && MBRerror < (myParameters.MBRThreshold * 0.5))
        {
            if (CourantWater < 0.5)
            {
                doubleTimeStep();
            }
            else
            {
                myParameters.current_delta_t = std::min(myParameters.current_delta_t / CourantWater, myParameters.delta_t_max);
                if (myParameters.current_delta_t > 1.)
                {
                    myParameters.current_delta_t = floor(myParameters.current_delta_t);
                }
            }
        }

        return true;
    }

    // first approximation or error is better than previuos one
    if (approxNr == 0 || MBRerror < _bestMBRerror)
	{
		saveBestStep();
        _bestMBRerror = MBRerror;
	}

    // system is unstable or last approximation
    int lastApproximation = myParameters.maxApproximationsNumber-1;
    if (MBRerror > (_bestMBRerror * 3.0) || approxNr == lastApproximation)
    {
        if (deltaT > myParameters.delta_t_min)
        {
            halveTimeStep();
            setForcedHalvedTime(true);
            return false;
        }
        else
        {
            // worst case: forced to accept the time step, restore best error
            restoreBestStep(deltaT);
            acceptStep(deltaT);
            return true;
        }
    }

    // move to the next approximation
    return false;
}



void updateBalanceWaterWholePeriod()
{
    /*! update the flows in the balance (balanceWholePeriod) */
    balanceWholePeriod.sinkSourceWater += balanceCurrentPeriod.sinkSourceWater;

    /*! compute waterMBE and waterMBR */
    double dStoragePeriod = balanceCurrentTimeStep.storageWater - balanceCurrentPeriod.storageWater;
    balanceCurrentPeriod.waterMBE = dStoragePeriod - balanceCurrentPeriod.sinkSourceWater;

    double dStorageHistorical = balanceCurrentTimeStep.storageWater - balanceWholePeriod.storageWater;
    balanceWholePeriod.waterMBE = dStorageHistorical - balanceWholePeriod.sinkSourceWater;

    /*! reference water volume [m3] minimum 1 liter */
    double refWater = std::max(balanceWholePeriod.sinkSourceWater, 0.001);
    balanceWholePeriod.waterMBR = balanceWholePeriod.waterMBE / refWater;

    /*! update storageWater in balanceCurrentPeriod */
    balanceCurrentPeriod.storageWater = balanceCurrentTimeStep.storageWater;
}


bool getForcedHalvedTime()
{
    return (_isHalfTimeStepForced);
}

void setForcedHalvedTime(bool isForced)
{
    _isHalfTimeStepForced = isForced;
}

