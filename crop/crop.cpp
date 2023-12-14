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
#include "basicMath.h"
#include "crop.h"
#include "root.h"
#include "development.h"


Crit3DCrop::Crit3DCrop() 
{
    this->clear();
}

void Crit3DCrop::clear()
{
    idCrop = "";
    type = HERBACEOUS_ANNUAL;

    roots.clear();

    // crop cycle
    sowingDoy = NODATA;
    currentSowingDoy = NODATA;
    doyStartSenescence = NODATA;
    plantCycle = NODATA;
    LAImin = NODATA;
    LAImax = NODATA;
    LAIgrass = NODATA;
    LAIcurve_a = NODATA;
    LAIcurve_b = NODATA;
    thermalThreshold = NODATA;
    upperThermalThreshold = NODATA;
    degreeDaysIncrease = NODATA;
    degreeDaysDecrease = NODATA;
    degreeDaysEmergence = NODATA;

    // water need
    kcMax  = NODATA;
    psiLeaf = NODATA;
    stressTolerance = NODATA;
    fRAW = NODATA;

    // irrigation
    irrigationShift = NODATA;
    irrigationVolume = NODATA;
    degreeDaysStartIrrigation = NODATA;
    degreeDaysEndIrrigation = NODATA;
    doyStartIrrigation = NODATA;
    doyEndIrrigation = NODATA;
    maxSurfacePuddle = NODATA;

    // variables
    isLiving = false;
    isEmerged = false;
    LAIstartSenescence = NODATA;
    daysSinceIrrigation = NODATA;
    degreeDays = NODATA;
    LAI = NODATA;
    LAIpreviousDay = NODATA;
    layerTranspiration.clear();
}


void Crit3DCrop::initialize(double latitude, unsigned int nrLayers, double totalSoilDepth, int currentDoy)
{
    // initialize vector
    roots.rootDensity.clear();
    roots.rootDensity.resize(nrLayers);
    layerTranspiration.clear();
    layerTranspiration.resize(nrLayers);

    // initialize root depth
    roots.rootDepth = 0;

    if (totalSoilDepth == 0 || roots.rootDepthMax < totalSoilDepth)
        roots.actualRootDepthMax = roots.rootDepthMax;
    else
        roots.actualRootDepthMax = totalSoilDepth;

    degreeDays = 0;

    if (latitude > 0)
        doyStartSenescence = 305;
    else
        doyStartSenescence = 120;

    LAIstartSenescence = NODATA;
    currentSowingDoy = NODATA;

    daysSinceIrrigation = NODATA;

    // check if the crop is living
    if (isSowingCrop())
    {
        isLiving = isInsideTypicalCycle(currentDoy);

        if (isLiving == true)
            currentSowingDoy = sowingDoy;
    }
    else
    {
        isLiving = true;
    }

    resetCrop(nrLayers);
}


double Crit3DCrop::getDailyDegreeIncrease(double tmin, double tmax, int doy)
{
    if (isEqual(tmin, NODATA) || isEqual(tmax, NODATA))
        return NODATA;

    // check crop cycle
    if (isSowingCrop() && ! isInsideTypicalCycle(doy))
    {
        return 0;
    }

    double tmed = (tmin + MINVALUE(tmax, upperThermalThreshold)) * 0.5;
    return MAXVALUE(tmed - thermalThreshold, 0);
}


double Crit3DCrop::computeSimpleLAI(double myDegreeDays, double latitude, int currentDoy)
{
    double currentLAI = 0;

    if (isSowingCrop())
    {
        if (myDegreeDays < degreeDaysEmergence)
        {
            currentLAI = 0;
        }
        else
        {
            myDegreeDays -= degreeDaysEmergence;
            currentLAI = leafDevelopment::getLAICriteria(this, myDegreeDays);
        }
    }
    else
    {
        if (myDegreeDays > 0)
            currentLAI = leafDevelopment::getLAICriteria(this, myDegreeDays);
        else
            currentLAI = LAImin;

        if (type == TREE)
        {
            bool isLeafFall;
            if (latitude > 0)   // north
            {
                doyStartSenescence = 305;
                isLeafFall = (currentDoy >= doyStartSenescence);
            }
            else                // south
            {
                doyStartSenescence = 120;
                isLeafFall = ((currentDoy >= doyStartSenescence) && (currentDoy < 182));
            }

            if (isLeafFall)
            {
                currentLAI = leafDevelopment::getLAISenescence(LAImin, LAImax*0.75, currentDoy - doyStartSenescence);
            }

            currentLAI += LAIgrass;
        }
    }

    return currentLAI;
}


bool Crit3DCrop::updateLAI(double latitude, unsigned int nrLayers, int currentDoy)
{
    double degreeDaysLai = 0;
    double myLai = 0;

    if (isSowingCrop())
    {
        if (! isEmerged)
        {
            if (degreeDays < degreeDaysEmergence)
                return true;
            else if (currentDoy - sowingDoy >= MIN_EMERGENCE_DAYS)
            {
                isEmerged = true;
                degreeDaysLai = degreeDays - degreeDaysEmergence;
            }
            else
                return true;
        }
        else
        {
            degreeDaysLai = degreeDays - degreeDaysEmergence;
        }

        if (degreeDaysLai > 0)
            myLai = leafDevelopment::getLAICriteria(this, degreeDaysLai);
    }
    else
    {
        if (type == GRASS)
            // grass cut
            if (degreeDays >= degreeDaysIncrease)
                resetCrop(nrLayers);

        if (degreeDays > 0)
            myLai = leafDevelopment::getLAICriteria(this, degreeDays);
        else
            myLai = LAImin;

        if (type == TREE)
        {
            bool isLeafFall;
            if (latitude > 0)   // north
            {
                isLeafFall = (currentDoy >= doyStartSenescence);
            }
            else                // south
            {
                isLeafFall = ((currentDoy >= doyStartSenescence) && (currentDoy < 182));
            }

            if (isLeafFall)
            {
                if (currentDoy == doyStartSenescence || int(LAIstartSenescence) == int(NODATA))
                    LAIstartSenescence = myLai;
                else
                    myLai = leafDevelopment::getLAISenescence(LAImin, LAIstartSenescence, currentDoy - doyStartSenescence);
            }

            myLai += LAIgrass;
        }
    }
    LAIpreviousDay = LAI;
    LAI = myLai;
    return true;
}


int Crit3DCrop::getDaysFromTypicalSowing(int myDoy) const
{
    return (myDoy - sowingDoy) % 365;
}


int Crit3DCrop::getDaysFromCurrentSowing(int myDoy) const
{
    if (currentSowingDoy != NODATA)
        return (myDoy - currentSowingDoy) % 365;
    else
        return getDaysFromTypicalSowing(myDoy);
}


bool Crit3DCrop::isInsideTypicalCycle(int myDoy) const
{
    return (getDaysFromTypicalSowing(myDoy) < plantCycle);
}


bool Crit3DCrop::isSowingCrop() const
{
    return (type == HERBACEOUS_ANNUAL || type == HORTICULTURAL);
}


bool Crit3DCrop::isWaterSurplusResistant() const
{
    return (idCrop == "RICE");
}


bool Crit3DCrop::isRootStatic() const
{
    return (type == HERBACEOUS_PERENNIAL ||
            type == GRASS ||
            type == FALLOW ||
            type == TREE);
}


/*!
 * \brief getSurfaceWaterPonding
 * \return maximum height of water pond [mm]
 */
double Crit3DCrop::getSurfaceWaterPonding() const
{
    // TODO taking into account tillage and crop development
    double clodHeight;          // [mm] height of clod

    if (isSowingCrop())
        clodHeight = 5.0;
    else
        clodHeight = 0.0;

    if (maxSurfacePuddle == NODATA)
        return clodHeight;
    else
        return maxSurfacePuddle + clodHeight;
}


bool Crit3DCrop::needReset(Crit3DDate myDate, double latitude, double waterTableDepth)
{
    int currentDoy = getDoyFromDate(myDate);

    if (isSowingCrop())
    {
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
            // bare soil: check sowing
            int sowingDoyPeriod = 30;
            int daysFromSowing = getDaysFromTypicalSowing(currentDoy);

            // is sowing possible? (check period and watertable depth)
            if (daysFromSowing >= 0 && daysFromSowing <= sowingDoyPeriod)
            {
                double waterTableThreshold = 0.2;

                if (isWaterSurplusResistant()
                        || isEqual(waterTableDepth, NODATA)
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
    else
    {
        // pluriannual crop: reset at the end of year
        // January at north hemisphere, July at south
        if ((latitude >= 0 && myDate.month == 1 && myDate.day == 1)
            || (latitude < 0 && myDate.month == 7 && myDate.day == 1))
        {
            isLiving = true;
            return true;
        }
    }

    return false;
}


// reset of (already initialized) crop
// TODO: smart start (using meteo settings)
void Crit3DCrop::resetCrop(unsigned int nrLayers)
{
    // roots
    if (! isRootStatic())
    {
        for (unsigned int i = 0; i < nrLayers; i++)
            roots.rootDensity[i] = 0;
    }

    isEmerged = false;

    if (isLiving)
    {
        degreeDays = 0;

        // LAI
        LAI = LAImin;
        LAIpreviousDay = LAImin;
        if (type == TREE) LAI += LAIgrass;
    }
    else
    {
        degreeDays = NODATA;
        LAI = NODATA;
        LAIpreviousDay = NODATA;
        currentSowingDoy = NODATA;

        // roots
        roots.currentRootLength = 0.0;
        roots.rootDepth = NODATA;
    }

    LAIstartSenescence = NODATA;
    daysSinceIrrigation = NODATA;
}


bool Crit3DCrop::dailyUpdate(const Crit3DDate &myDate, double latitude, const std::vector<soil::Crit3DLayer> &soilLayers,
                             double tmin, double tmax, double waterTableDepth, std::string &myError)
{
    myError = "";
    if (idCrop == "") return false;

    unsigned int nrLayers = unsigned(soilLayers.size());

    // check start/end crop cycle
    if (needReset(myDate, latitude, waterTableDepth))
    {
        resetCrop(nrLayers);
    }

    if (isLiving)
    {
        int currentDoy = getDoyFromDate(myDate);

        // update degree days
        double dailyDD = getDailyDegreeIncrease(tmin, tmax, currentDoy);
        if (isEqual(dailyDD, NODATA))
        {
            myError = "Error in computing degree dyas for " + myDate.toStdString();
            return false;
        }
        degreeDays += dailyDD;

        // update LAI
        if ( !updateLAI(latitude, nrLayers, currentDoy))
        {
            myError = "Error in updating LAI for crop " + idCrop;
            return false;
        }

        // update roots
        updateRootDepth(degreeDays, waterTableDepth);
        root::computeRootDensity(this, soilLayers);
    }

    return true;
}


bool Crit3DCrop::restore(const Crit3DDate &myDate, double latitude, const std::vector<soil::Crit3DLayer> &soilLayers,
                         double currentWaterTable, std::string &myError)
{
    myError = "";
    if (idCrop == "") return false;

    unsigned int nrLayers = unsigned(soilLayers.size());

    // check start/end crop cycle (update isLiving)
    if (needReset(myDate, latitude, currentWaterTable))
    {
        resetCrop(nrLayers);
    }

    if (isLiving)
    {
        int currentDoy = getDoyFromDate(myDate);

        if ( !updateLAI(latitude, nrLayers, currentDoy))
        {
            myError = "Error in updating LAI for crop " + idCrop;
            return false;
        }

        // update roots
        updateRootDepth(degreeDays, currentWaterTable);
        root::computeRootDensity(this, soilLayers);
    }

    return true;
}


// update current root depth [m]
void Crit3DCrop::updateRootDepth(double currentDD, double waterTableDepth)
{
    if (! isLiving)
    {
        roots.currentRootLength = 0.0;
        roots.rootDepth = NODATA;
    }
    else
    {
        roots.currentRootLength = computeRootLength(currentDD, waterTableDepth);
        roots.rootDepth = roots.rootDepthMin + roots.currentRootLength;
    }
}


// return current root lenght [m]
double Crit3DCrop::computeRootLength(double currentDD, double waterTableDepth)
{
    double newRootLength;

    if (isRootStatic())
    {
        newRootLength = roots.actualRootDepthMax - roots.rootDepthMin;
    }
    else
    {
        if (currentDD <= 0)
        {
            newRootLength = 0.0;
        }
        else
        {
            if (currentDD > roots.degreeDaysRootGrowth)
            {
                newRootLength = roots.actualRootDepthMax - roots.rootDepthMin;
            }
            else
            {
                // in order to avoid numerical divergences when calculating density through cardioid and gamma function
                currentDD = MAXVALUE(currentDD, 1.0);
                newRootLength = root::getRootLengthDD(roots, currentDD, degreeDaysEmergence);
            }
        }
    }

    if (isEqual(waterTableDepth, NODATA))
    {
        return newRootLength;
    }

    // WATERTABLE
    // Nel saturo le radici vanno in asfissia
    // per cui si mantengono a distanza dalla falda nella fase di crescita
    // le radici possono crescere (max 2 cm al giorno) se:
    // la falda è più bassa o si sta abbassando
    // restano invariate se:
    // 1) non sono più in fase di crescita
    // 2) sono già dentro la falda (currentRootDepth > waterTableDepth)
    const double MAX_DAILY_GROWTH = 0.02;             // [m]
    const double MIN_WATERTABLE_DISTANCE = 0.1;       // [m]

    if (! isWaterSurplusResistant()
        && ! isEqual(roots.currentRootLength, NODATA)
        && newRootLength > roots.currentRootLength)
    {
        // la fase di crescita è finita
        if (currentDD > roots.degreeDaysRootGrowth)
            newRootLength = roots.currentRootLength;
        else
            newRootLength = MINVALUE(newRootLength, roots.currentRootLength + MAX_DAILY_GROWTH);

        // maximum root lenght
        double maxRootLenght = waterTableDepth - MIN_WATERTABLE_DISTANCE - roots.rootDepthMin;
        if (newRootLength > maxRootLenght)
        {
            newRootLength = MAXVALUE(roots.currentRootLength, maxRootLenght);
        }
    }

    return newRootLength;
}


/*! \brief updateRootDepth3D
 *  update current root lenght and root depth
 *  function for Criteria3D (update key variables)
 *  \param currentDD:  current degree days sum
 *  \param waterTableDepth      [m]
 *  \param previousRootDepth    [m]
 *  \param totalSoilDepth       [m]
 */
void Crit3DCrop::updateRootDepth3D(double currentDD,  double waterTableDepth, double previousRootDepth, double totalSoilDepth)
{
    // set actualRootDepthMax
    if (isEqual(totalSoilDepth, NODATA) || isEqual(totalSoilDepth, 0))
    {
        roots.actualRootDepthMax = roots.rootDepthMax;
    }
    else
    {
        roots.actualRootDepthMax = std::min(roots.rootDepthMax, totalSoilDepth);
    }

    // set currentRootLength
    if (isEqual(previousRootDepth, NODATA))
    {
        roots.currentRootLength = 0;
    }
    else
    {
        roots.currentRootLength = previousRootDepth - roots.rootDepthMin;
    }

    roots.currentRootLength = computeRootLength(currentDD, waterTableDepth);
    roots.rootDepth = roots.rootDepthMin + roots.currentRootLength;
}


/*! \brief getCoveredSurfaceFraction
 *  \ref Liangxia Zhang, Zhongmin Hu, Jiangwen Fan, Decheng Zhou & Fengpei Tang, 2014
 *  A meta-analysis of the canopy light extinction coefficient in terrestrial ecosystems
 *  "Cropland had the highest value of K (0.62), followed by broadleaf forest (0.59)
 *  shrubland (0.56), grassland (0.50), and needleleaf forest (0.45)"
 *  \return covered surface fraction [-]
 */
double Crit3DCrop::getCoveredSurfaceFraction()
{
    if (idCrop == "" || ! isLiving || LAI < EPSILON) return 0;

    double k = 0.6;      // [-] light extinction coefficient
    return 1 - exp(-k * LAI);
}


double Crit3DCrop::getMaxEvaporation(double ET0)
{
    double evapMax = ET0 * (1.0 - getCoveredSurfaceFraction());
    // TODO check evaporation on wet bare soil
    return evapMax * 0.67;
}


double Crit3DCrop::getMaxTranspiration(double ET0)
{
    if (idCrop == "" || ! isLiving || LAI < EPSILON)
        return 0;

    double coverSurfFraction = getCoveredSurfaceFraction();
    double kcFactor = 1 + (kcMax - 1) * coverSurfFraction;
    return ET0 * coverSurfFraction * kcFactor;
}


/*!
 * \brief getCropWaterDeficit
 * \return sum of water deficit (mm) in the rooting zone
 */
double Crit3DCrop::getCropWaterDeficit(const std::vector<soil::Crit3DLayer> &soilLayers)
{
    //check
    if (! isLiving) return NODATA;
    if (roots.rootDepth <= roots.rootDepthMin) return NODATA;
    if (roots.firstRootLayer == NODATA) return NODATA;

    double waterDeficit = 0.0;
    for (int i = roots.firstRootLayer; i <= roots.lastRootLayer; i++)
    {
        waterDeficit += soilLayers[unsigned(i)].FC - soilLayers[unsigned(i)].waterContent;
    }

    return MAXVALUE(waterDeficit, 0);
}


/*!
 * \brief computeTranspiration
 * \return total transpiration and layerTranspiration vector [mm]
 * or percentage of water stress (if returnWaterStress = true)
 */
double Crit3DCrop::computeTranspiration(double maxTranspiration, const std::vector<soil::Crit3DLayer> &soilLayers, double& waterStress)
{
    // check
    if (idCrop == "" || ! isLiving) return 0;
    if (roots.rootDepth <= roots.rootDepthMin) return 0;
    if (roots.firstRootLayer == NODATA) return 0;
    if (maxTranspiration < EPSILON) return 0;

    double thetaWP;                                 // [m3 m-3] volumetric water content at Wilting Point
    double cropWP;                                  // [mm] wilting point specific for crop
    double waterSurplusThreshold;                        // [mm] water surplus stress threshold
    double waterScarcityThreshold;                  // [mm] water scarcity stress threshold
    double WSS;                                     // [] water surplus stress

    double TRs=0.0;                                 // [mm] actual transpiration with only water scarsity stress
    double TRe=0.0;                                 // [mm] actual transpiration with only water surplus stress
    double totRootDensityWithoutStress = 0.0;       // [-]
    double redistribution = 0.0;                    // [mm]

    // initialize
    unsigned int nrLayers = unsigned(soilLayers.size());
    bool* isLayerStressed = new bool[nrLayers];
    for (unsigned int i = 0; i < nrLayers; i++)
    {
        isLayerStressed[i] = false;
        layerTranspiration[i] = 0;
    }

    // water surplus
    if (isWaterSurplusResistant())
        WSS = 0.0;
    else
        WSS = 0.5;

    for (unsigned int i = unsigned(roots.firstRootLayer); i <= unsigned(roots.lastRootLayer); i++)
    {
        // [mm]
        waterSurplusThreshold = soilLayers[i].SAT - (WSS * (soilLayers[i].SAT - soilLayers[i].FC));

        thetaWP = soil::thetaFromSignPsi(-soil::cmTokPa(psiLeaf), *(soilLayers[i].horizonPtr));
        // [mm]
        cropWP = thetaWP * soilLayers[i].thickness * soilLayers[i].soilFraction * 1000.0;

        // [mm]
        waterScarcityThreshold = soilLayers[i].FC - fRAW * (soilLayers[i].FC - cropWP);

        if ((soilLayers[i].waterContent - waterSurplusThreshold) > EPSILON)
        {
            // WATER SURPLUS
            layerTranspiration[i] = maxTranspiration * roots.rootDensity[i] *
                                    ((soilLayers[i].SAT - soilLayers[i].waterContent)
                                     / (soilLayers[i].SAT - waterSurplusThreshold));

            TRe += layerTranspiration[i];
            TRs += maxTranspiration * roots.rootDensity[i];
            isLayerStressed[i] = true;
        }
        else if (soilLayers[i].waterContent < waterScarcityThreshold)
        {
            // WATER SCARSITY
            if (soilLayers[i].waterContent <= cropWP)
            {
                layerTranspiration[i] = 0;
            }
            else
            {
                layerTranspiration[i] = maxTranspiration * roots.rootDensity[i] *
                                        ((soilLayers[i].waterContent - cropWP) / (waterScarcityThreshold - cropWP));
            }

            TRs += layerTranspiration[i];
            TRe += maxTranspiration * roots.rootDensity[i];
            isLayerStressed[i] = true;
        }
        else
        {
            // normal conditions
            layerTranspiration[i] = maxTranspiration * roots.rootDensity[i];

            TRs += layerTranspiration[i];
            TRe += layerTranspiration[i];

            if ((soilLayers[i].waterContent - layerTranspiration[i]) > waterScarcityThreshold)
            {
                isLayerStressed[i] = false;
                totRootDensityWithoutStress +=  roots.rootDensity[i];
            }
            else
            {
                isLayerStressed[i] = true;
            }
        }
    }

    // WATER STRESS [-]
    double firstWaterStress = 1 - (TRs / maxTranspiration);

    // Hydraulic redistribution
    // the movement of water from moist to dry soil through plant roots
    // TODO add numerical process
    if (firstWaterStress > EPSILON && totRootDensityWithoutStress > EPSILON)
    {
        // redistribution acts on not stressed roots
        redistribution = MINVALUE(firstWaterStress, totRootDensityWithoutStress) * maxTranspiration;

        for (int i = roots.firstRootLayer; i <= roots.lastRootLayer; i++)
        {
            if (! isLayerStressed[i])
            {
                double addLayerTransp = redistribution * (roots.rootDensity[unsigned(i)] / totRootDensityWithoutStress);
                layerTranspiration[unsigned(i)] += addLayerTransp;
                TRs += addLayerTransp;
            }
        }
    }

    waterStress = 1 - (TRs / maxTranspiration);

    double totalTranspiration = 0;
    for (int i = roots.firstRootLayer; i <= roots.lastRootLayer; i++)
    {
        totalTranspiration += layerTranspiration[unsigned(i)];
    }

    delete[] isLayerStressed;
    return totalTranspiration;
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
    else if (cropType == "fallow")
        return FALLOW;
    else if (cropType == "annual_fallow" || cropType == "fallow_annual")
        return FALLOW_ANNUAL;
    else if (cropType == "tree" || cropType == "fruit_tree")
        return TREE;
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
    case FALLOW_ANNUAL:
        return "fallow_annual";
    case TREE:
        return "tree";
    }

    return "No crop type";
}

