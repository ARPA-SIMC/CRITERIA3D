/*!
    \file crop.cpp

    \abstract
    Crop class functions

    \authors
    Fausto Tomei        ftomei@arpae.it
    Gabriele Antolini   gantolini@arpe.it
    Antonio Volta       avolta@arpae.it

    \copyright
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

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
*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#include "crit3dDate.h"
#include "commonConstants.h"
#include "crop.h"
#include "root.h"


Crit3DCrop::Crit3DCrop()
    : idCrop{""}
{
    type = HERBACEOUS_ANNUAL;

    /*!
     * \brief crop cycle
     */
    isLiving = false;
    isEmerged = false;
    sowingDoy = NODATA;
    currentSowingDoy = NODATA;
    doyStartSenescence = NODATA;
    plantCycle = NODATA;
    LAImin = NODATA;
    LAImax = NODATA;
    LAIgrass = NODATA;
    LAIcurve_a = NODATA;
    LAIcurve_b = NODATA;
    LAIstartSenescence = NODATA;
    thermalThreshold = NODATA;
    upperThermalThreshold = NODATA;
    degreeDaysIncrease = NODATA;
    degreeDaysDecrease = NODATA;
    degreeDaysEmergence = NODATA;

    /*!
     * \brief water need
     */
    kcMax  = NODATA;
    psiLeaf = NODATA;
    stressTolerance = NODATA;
    fRAW = NODATA;

    /*!
     * \brief irrigation
     */
    irrigationShift = NODATA;
    irrigationVolume = NODATA;
    degreeDaysStartIrrigation = NODATA;
    degreeDaysEndIrrigation = NODATA;
    doyStartIrrigation = NODATA;
    doyEndIrrigation = NODATA;
    maxSurfacePuddle = NODATA;
    daysSinceIrrigation = NODATA;

    degreeDays = NODATA;
    LAI = NODATA;
}


bool Crit3DCrop::isWaterSurplusResistant()
{
    return (idCrop == "RICE" || idCrop == "KIWIFRUIT" || type == GRASS || type == FALLOW);
}


int Crit3DCrop::getDaysFromTypicalSowing(int myDoy)
{
    return (myDoy - sowingDoy) % 365;
}


int Crit3DCrop::getDaysFromCurrentSowing(int myDoy)
{
    if (currentSowingDoy != NODATA)
        return (myDoy - currentSowingDoy) % 365;
    else
        return getDaysFromTypicalSowing(myDoy);
}


bool Crit3DCrop::isInsideTypicalCycle(int myDoy)
{
    return ((myDoy >= sowingDoy) && (getDaysFromTypicalSowing(myDoy) < plantCycle));
}


bool Crit3DCrop::isPluriannual()
{
    return (type == HERBACEOUS_PERENNIAL ||
            type == GRASS ||
            type == FALLOW ||
            type == FRUIT_TREE);
}


bool Crit3DCrop::needReset(Crit3DDate myDate, float latitude, float waterTableDepth)
{
    int currentDoy = getDoyFromDate(myDate);

    if (isPluriannual())
    {
        // pluriannual crop: reset at the end of year (january at north / july at south)
        if ((latitude >= 0 && myDate.month == 1 && myDate.day == 1)
            || (latitude < 0 && myDate.month == 7 && myDate.day == 1))
        {
            isLiving = true;
            return true;
        }
    }
    else
    {
        // annual crop
        if (isLiving)
        {
            // living crop: check end of crop cycle
            double cycleDD = degreeDaysEmergence + degreeDaysIncrease + degreeDaysDecrease;

            if ((degreeDays > cycleDD) || (getDaysFromCurrentSowing(currentDoy) > plantCycle))
            {
                isLiving = false;
                return true;
            }
        }
        else
        {
            // naked soil: check sowing
            int sowingDoyPeriod = 30;
            int daysFromSowing = getDaysFromTypicalSowing(currentDoy);

            // is sowing possible? (check period and watertable depth)
            if (daysFromSowing >= 0 && daysFromSowing <= sowingDoyPeriod)
            {
                float waterTableThreshold = 0.2f;

                if (isWaterSurplusResistant()
                        || int(waterTableDepth) == int(NODATA)
                        || waterTableDepth >= waterTableThreshold)
                {
                    isLiving = true;
                    // update sowing doy
                    currentSowingDoy = sowingDoy + daysFromSowing;
                    return true;
                }
            }
        }
    }

    return false;
}


// reset of (already initialized) crop
// TODO: partenza intelligente (usando sowing doy e ciclo)
void Crit3DCrop::resetCrop(int nrLayers)
{
    // roots
    if (! isPluriannual())
    {
        roots.rootDensity[0] = 0.0;
        for (int i = 1; i < nrLayers; i++)
            roots.rootDensity[i] = 0;
    }

    isEmerged = false;

    if (isLiving)
    {
        degreeDays = 0;

        // LAI
        LAI = LAImin;

        if (type == FRUIT_TREE)
            LAI += LAIgrass;
    }
    else
    {
        degreeDays = NODATA;
        LAI = NODATA;
        currentSowingDoy = NODATA;

        // roots
        roots.rootLength = 0.0;
        roots.rootDepth = NODATA;
    }

    LAIstartSenescence = NODATA;
    daysSinceIrrigation = NODATA;
}


speciesType getCropType(std::string cropType)
{
    // lower case
    std::transform(cropType.begin(), cropType.end(), cropType.begin(), ::tolower);

    if (cropType == "herbaceous")
        return HERBACEOUS_ANNUAL;
    else if (cropType == "herbaceous_perennial")
        return HERBACEOUS_PERENNIAL;
    else if (cropType == "horticultural")
        return HORTICULTURAL;
    else if (cropType == "grass")
        return GRASS;
    else if (cropType == "grass_first_year")  //LC perchè si perde la distinzione grass o grass_first_year, nella get quale devo considerare?
        return GRASS;
    else if (cropType == "fallow")
        return FALLOW;
    else if (cropType == "tree" || cropType == "fruit_tree")
        return FRUIT_TREE;
    else
        return HERBACEOUS_ANNUAL;
}

std::string getCropTypeString(speciesType cropType)
{
    switch (cropType)
    {
    case HERBACEOUS_ANNUAL:
        return "herbaceous";
    case HERBACEOUS_PERENNIAL:
        return "herbaceous_perennial";
    case HORTICULTURAL:
        return "horticultural";
    case GRASS:
        return "grass";
    case FALLOW:
        return "fallow";
    case FRUIT_TREE:
        return "fruit_tree";
    }
}

double computeDegreeDays(double myTmin, double myTmax, double myLowerThreshold, double myUpperThreshold)
{
    return MAXVALUE((myTmin + MINVALUE(myTmax, myUpperThreshold)) / 2. - myLowerThreshold, 0);
}

