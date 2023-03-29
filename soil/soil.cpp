/*!
    CRITERIA3D

    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    You should have received a copy of the GNU General Public License
    along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.

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
    ftomei@arpae.it
*/

#include <math.h>
#include <algorithm>

#include "soil.h"
#include "commonConstants.h"
#include "furtherMathFunctions.h"


namespace soil
{
    Crit3DHorizonDbData::Crit3DHorizonDbData()
    {
        this->horizonNr = NODATA;
        this->upperDepth = NODATA;
        this->lowerDepth = NODATA;
        this->sand = NODATA;
        this->silt = NODATA;
        this->clay = NODATA;
        this->coarseFragments = NODATA;
        this->organicMatter = NODATA;
        this->bulkDensity = NODATA;
        this->thetaSat = NODATA;
        this->kSat = NODATA;
    }

    Crit3DFittingOptions::Crit3DFittingOptions()
    {
        // default
        this->waterRetentionCurve = MODIFIEDVANGENUCHTEN;
        this->useWaterRetentionData = true;
        this->airEntryFixed = true;
        this->mRestriction = true;
    }

    Crit3DLayer::Crit3DLayer()
    {
        this->depth = NODATA;
        this->thickness = NODATA;
        this->soilFraction = NODATA;
        this->waterContent = NODATA;
        this->SAT = NODATA;
        this->FC = NODATA;
        this->WP = NODATA;
        this->HH = NODATA;
        this->critical = NODATA;
        this->maxInfiltration = NODATA;
        this->flux = NODATA;

        this->horizon = nullptr;
    }

    Crit3DTexture::Crit3DTexture()
    {
        this->sand = NODATA;
        this->silt = NODATA;
        this->clay = NODATA;
        this->classUSDA = NODATA;
        this->classNL = NODATA;
        this->classNameUSDA = "UNDEFINED";
    }

    Crit3DTexture::Crit3DTexture (double mySand, double mySilt, double myClay)
    {
        this->sand = mySand;
        this->silt = mySilt;
        this->clay = myClay;
        this->classUSDA = getUSDATextureClass(sand, silt, clay);
        this->classNL = getNLTextureClass(sand, silt, clay);
    }

    Crit3DVanGenuchten::Crit3DVanGenuchten()
    {
        this->alpha = NODATA;
        this->n = NODATA;
        this->m = NODATA;
        this->he = NODATA;
        this->sc = NODATA;
        this->thetaR = NODATA;
        this->thetaS = NODATA;
        this->refThetaS = NODATA;
    }

    Crit3DDriessen::Crit3DDriessen()
    {
        this->k0 = NODATA;
        this->gravConductivity = NODATA;
        this->maxSorptivity = NODATA;
    }

    Crit3DWaterConductivity::Crit3DWaterConductivity()
    {
        this->kSat = NODATA;
        this->l = NODATA;
    }


    Crit3DHorizon::Crit3DHorizon()
    {
        this->upperDepth = NODATA;
        this->lowerDepth = NODATA;

        this->coarseFragments = NODATA;
        this->organicMatter = NODATA;
        this->bulkDensity = NODATA;

        this->fieldCapacity = NODATA;
        this->wiltingPoint = NODATA;
        this->waterContentFC = NODATA;
        this->waterContentWP = NODATA;

        this->PH = NODATA;
        this->CEC = NODATA;
    }

    Crit3DSoil::Crit3DSoil()
    {
        this->cleanSoil();
    }

    void Crit3DSoil::initialize(const std::string& soilCode, int nrHorizons)
    {
        this->cleanSoil();
        this->code = soilCode;
        if (nrHorizons > 0)
        {
            this->nrHorizons = unsigned(nrHorizons);
            this->horizon.resize(this->nrHorizons);
            this->totalDepth = 0;
        }
    }

    void Crit3DSoil::addHorizon(int nHorizon, Crit3DHorizon* newHorizon)
    {
        this->horizon.insert(this->horizon.begin() + nHorizon, *newHorizon);
        this->nrHorizons = nrHorizons + 1;
    }

    void Crit3DSoil::deleteHorizon(int nHorizon)
    {
        this->horizon.erase(this->horizon.begin() + nHorizon);
        this->nrHorizons = nrHorizons - 1;
    }

    int Crit3DSoil::getHorizonIndex(double depth)
    {
       for (unsigned int index = 0; index < nrHorizons; index++)
       {
           if (depth >= horizon[index].upperDepth && depth <= (horizon[index].lowerDepth + EPSILON))
               return int(index);
       }

       return NODATA;
    }

    void Crit3DSoil::cleanSoil()
    {
        for (unsigned int i = 0; i < this->horizon.size(); i++)
        {
            this->horizon[i].dbData.waterRetention.erase(this->horizon[i].dbData.waterRetention.begin(), this->horizon[i].dbData.waterRetention.end());
            this->horizon[i].dbData.waterRetention.clear();
        }
        this->horizon.clear();
        this->nrHorizons = 0;
        this->totalDepth = 0;
        this->id = NODATA;
        this->code = "";
        this->name = "";
    }


    bool Crit3DLayer::setLayer(Crit3DHorizon *horizonPointer)
    {
        if (horizonPointer == nullptr)
            return false;

        horizon = horizonPointer;

        double hygroscopicHumidity = -2000;     // [kPa]
        double waterContentHH = soil::thetaFromSignPsi(hygroscopicHumidity, horizon);

        // [-]
        soilFraction = (1.0 - horizon->coarseFragments);

        // [mm]
        SAT = horizon->vanGenuchten.thetaS * soilFraction * thickness * 1000;
        FC = horizon->waterContentFC * soilFraction * thickness * 1000;
        WP = horizon->waterContentWP * soilFraction * thickness * 1000;
        HH = waterContentHH * soilFraction * thickness * 1000;
        critical = FC;

        return true;
    }


    int getUSDATextureClass(Crit3DTexture texture)
    {
        return getUSDATextureClass(texture.sand, texture.silt, texture.clay);
    }


    // [%]
    int getUSDATextureClass(double sand, double silt, double clay)
    {
        if (int(sand) == int(NODATA) || int(silt) == int(NODATA) || int(clay) == int(NODATA))
            return NODATA;

        if (fabs(double(sand + clay + silt) - 100) > 2.0)
            return NODATA;

        int myClass = NODATA;
        /*! clay */
        if (clay >= 40) myClass = 12;
        /*! silty clay */
        if ((silt >= 40) && (clay >= 40)) myClass = 11;
        /*! sandy clay */
        if ((clay >= 35) && (sand >= 45)) myClass = 10;
        /*! silty loam */
        if (((clay < 27.5) && (silt >= 50) & (silt <= 80)) || ((clay >= 12.5) && (silt >= 80))) myClass = 4;
        /*! silt */
        if ((clay < 12.5) && (silt >= 80)) myClass = 6;
        /*! silty clay loam */
        if ((clay < 40) && (sand < 20) && (clay >= 27.5)) myClass = 8;
        /*! sandy loam  */
        if (((clay < 20) && (sand >= 52.5)) ||
           ((clay < 7.5) && (silt < 50) && (sand >= 42.5) && (sand <= 52.5))) myClass = 3;
        /*! loamy sand */
        if ((sand >= 70) && (clay <= (sand - 70))) myClass = 2;
        /*! sand */
        if ((sand >= 85) && (clay <= (2 * sand -170))) myClass = 1;
        /*! sandy clay loam */
        if ((clay >= 20) && (clay < 35) && (sand >= 45) && (silt < 27.5)) myClass = 7;
        /*! loam */
        if ((clay >= 7.5) && (clay < 27.5) && (sand < 52.5)  && (silt >= 27.5) & (silt < 50)) myClass = 5;
        /*! clay loam */
        if ((clay >= 27.5) && (clay < 40) && (sand >= 20) && (sand < 45)) myClass = 9;

        return myClass;
    }


    int getNLTextureClass(Crit3DTexture texture)
    {
        return getNLTextureClass(texture.sand, texture.silt, texture.clay);
    }

    /*!
     * \brief NL texture (used by Driessen) different from USDA only in clay zone
     * \param sand
     * \param silt
     * \param clay
     * \return result
     */
    int getNLTextureClass(double sand, double silt, double clay)
    {
        if (int(sand) == int(NODATA) || int(silt) == int(NODATA) || int(clay) == int(NODATA))
            return NODATA;

        if (fabs(double(sand + clay + silt) - 100) > 2)
            return NODATA;

        /*! heavy clay */
        if (clay >= 60) return 12;

        if (clay > 40)
        {
            /*! silty clay */
            if (silt > 40)return 11;
            /*! light clay */
            else return 10;
        }
        else return getUSDATextureClass(sand, silt, clay);
    }


    double estimateSpecificDensity(double organicMatter)
    {
        if (int(organicMatter) == int(NODATA))
        {
            organicMatter = MINIMUM_ORGANIC_MATTER;
        }

        /*! Driessen (1986) */
        // return 1 / ((1 - organicMatter) / QUARTZ_DENSITY + organicMatter / 1.43);

        /*! Rühlmann et al. (2006) */
        return 1 / ((1 - organicMatter) / QUARTZ_DENSITY + organicMatter / (1.127 + 0.373*organicMatter));
    }


    // estimate bulk density from total porosity
    double estimateBulkDensity(Crit3DHorizon* horizon, double totalPorosity, bool increaseWithDepth)
    {
        if (int(totalPorosity) == int(NODATA))
            totalPorosity = (horizon->vanGenuchten.refThetaS);

        double specificDensity = estimateSpecificDensity(horizon->organicMatter);
        double refBulkDensity = (1 - totalPorosity) * specificDensity;

        // increase/decrease with depth, reference theta sat at 30cm
        if (increaseWithDepth)
        {
            double depth = (horizon->upperDepth + horizon->lowerDepth) * 0.5;
            double depthCoeff = (depth - 0.30) * 0.05;
            refBulkDensity *= (1.0 + depthCoeff);
        }

        return refBulkDensity;
    }


    double estimateTotalPorosity(Crit3DHorizon* horizon, double bulkDensity)
    {
        if (int(bulkDensity) == int(NODATA)) return NODATA;

        double specificDensity = estimateSpecificDensity(horizon->organicMatter);
        return 1 - (bulkDensity /specificDensity);
    }


    double estimateThetaSat(Crit3DHorizon* horizon, double bulkDensity)
    {
        double totalPorosity = estimateTotalPorosity(horizon, bulkDensity);
        if (int(totalPorosity) == int(NODATA))
            return NODATA;
        else
            return totalPorosity;
    }


    double estimateSaturatedConductivity(Crit3DHorizon* horizon, double bulkDensity)
    {
        if (int(bulkDensity) == int(NODATA)) return NODATA;

        double refTotalPorosity = horizon->vanGenuchten.refThetaS;
        double specificDensity = estimateSpecificDensity(horizon->organicMatter);
        double refBulkDensity = (1 - refTotalPorosity) * specificDensity;

        if (bulkDensity <= refBulkDensity)
            return horizon->waterConductivity.kSat;
        else
        {
            // soil compaction
            double ratio = 1 - (bulkDensity / refBulkDensity);
            return horizon->waterConductivity.kSat * exp(10*ratio);
        }
    }


    int getHorizonIndex(Crit3DSoil* soil, double depth)
    {
       for (unsigned int index = 0; index < soil->nrHorizons; index++)
       {
           if (depth >= soil->horizon[index].upperDepth && depth <= (soil->horizon[index].lowerDepth + EPSILON))
               return(int(index));
       }

       return NODATA;
    }


    int getSoilLayerIndex(std::vector<soil::Crit3DLayer> &soilLayers, double depth)
    {
       for (unsigned int index = 0; index < soilLayers.size(); index++)
       {
           double upperDepth = soilLayers[index].depth - soilLayers[index].thickness/2;
           double lowerDepth = soilLayers[index].depth + soilLayers[index].thickness/2;
           if (depth >= upperDepth && depth <= lowerDepth)
               return signed(index);
       }

       return NODATA;
    }


    /*!
     * \brief Field Capacity water potential as clay function
     * \param horizon
     * \param unit [KPA | METER | CM]
     * \note author: Franco Zinoni
     * \return water potential at field capacity (with sign)
     */
    double getFieldCapacity(Crit3DHorizon* horizon, soil::units unit)
    {
        double fcMin = -10;                 /*!< [kPa] clay < 20% : sandy soils */
        double fcMax = -33;                 /*!< [kPa] clay > 50% : clay soils */

        const double CLAYMIN = 20;
        const double CLAYMAX = 50;

        double fieldCapacity;

        if (double(horizon->texture.clay) <= CLAYMIN)
            fieldCapacity = fcMin;
        else if (double(horizon->texture.clay) >= CLAYMAX)
            fieldCapacity = fcMax;
        else
        {
            double clayFactor = (double(horizon->texture.clay) - CLAYMIN) / (CLAYMAX - CLAYMIN);
            fieldCapacity = (fcMin + (fcMax - fcMin) * clayFactor);
        }

        if (unit == KPA)
            return fieldCapacity;
        else if (unit == METER)
            return kPaToMeters(fieldCapacity);
        else if (unit == CM)
            return kPaToCm(fieldCapacity);
        else
            return fieldCapacity;
    }


    /*!
     * \brief [m] WP = Wilting Point
     * \param unit
     * \return wilting point
     */
    double getWiltingPoint(soil::units unit)
    {           
        if (unit == KPA)
            return -1600;
        else if (unit == METER)
            return kPaToMeters(-1600);
        else if (unit == CM)
            return kPaToCm(-1600);
        else
            return(-1600);
    }


    double kPaToMeters(double value)
    {
        return (value / GRAVITY);
    }

    double metersTokPa(double value)
    {
        return (value * GRAVITY);
    }

    double kPaToCm(double value)
    {
        return kPaToMeters(value) * 100;
    }

    double cmTokPa(double value)
    {
        return metersTokPa(value / 100);
    }


    /*!
     * \brief Compute degree of saturation from volumetric water content
     * \param theta [m^3 m-3] volumetric water content
     * \param horizon pointer to Crit3DHorizon class
     * \return [-] degree of saturation
     */
    double SeFromTheta(double theta, Crit3DHorizon* horizon)
    {
        // check range
        if (theta >= horizon->vanGenuchten.thetaS) return 1;
        if (theta <= horizon->vanGenuchten.thetaR) return 0;

        return (theta - horizon->vanGenuchten.thetaR) / (horizon->vanGenuchten.thetaS - horizon->vanGenuchten.thetaR);
    }


    /*!
     * \brief Compute water potential from volumetric water content
     * \brief using modified Van Genuchten model
     * \param theta: volumetric water content   [m^3 m-3]
     * \param horizon: pointer to Crit3DHorizon class
     * \return water potential                  [kPa]
     */
    double psiFromTheta(double theta, Crit3DHorizon* horizon)

    {
        double Se = SeFromTheta(theta, horizon);
        double temp = pow(1.0 / (Se * horizon->vanGenuchten.sc), 1.0 / horizon->vanGenuchten.m) - 1.0;
        double psi = (1.0 / horizon->vanGenuchten.alpha) * pow(temp, 1.0/ horizon->vanGenuchten.n);
        return psi;
    }


    /*!
     * \brief Compute degree of stauration from signed water potential
     * \brief using modified Van Genuchten model
     * \param signPsi water potential       [kPa]
     * \param horizon
     * \return degree of saturation         [-]
     */
    double degreeOfSaturationFromSignPsi(double signPsi, Crit3DHorizon* horizon)
    {
        if (signPsi >= 0.0) return 1.0;

        double psi = fabs(signPsi);
        if (psi <=  horizon->vanGenuchten.he) return 1.0;

        double degreeOfSaturation = pow(1.0 + pow(horizon->vanGenuchten.alpha * psi, horizon->vanGenuchten.n),
                        - horizon->vanGenuchten.m) / horizon->vanGenuchten.sc;

        return degreeOfSaturation;
    }


    /*!
     * \brief Compute volumetric water content from signed water potential
     * \param signPsi water potential       [kPa]
     * \param horizon
     * \return volumetric water content     [m^3 m-3]
     */
    double thetaFromSignPsi(double signPsi, Crit3DHorizon* horizon)
    {     
        // degree of saturation [-]
        double Se = degreeOfSaturationFromSignPsi(signPsi, horizon);

        double theta = Se * (horizon->vanGenuchten.thetaS - horizon->vanGenuchten.thetaR) + horizon->vanGenuchten.thetaR;
        return theta;
    }


    /*!
     * \brief Compute hydraulic conductivity from degree of saturation
     * \brief using Mualem equation for modified Van Genuchten model
     * \param Se: degree of saturation      [-]
     * \param horizon: pointer to Crit3DHorizon class
     * \return hydraulic conductivity       [cm day^-1]
     * \warning very low values are possible (es: 10^12)
     */
    double waterConductivity(double Se, Crit3DHorizon* horizon)
    {
        if (Se >= 1.) return(horizon->waterConductivity.kSat);

        double myTmp = NODATA;

        double myNumerator = 1. - pow(1. - pow(Se * horizon->vanGenuchten.sc, 1.0 / horizon->vanGenuchten.m), horizon->vanGenuchten.m);
        myTmp = myNumerator / (1. - pow(1. - pow(horizon->vanGenuchten.sc, 1.0 / horizon->vanGenuchten.m), horizon->vanGenuchten.m));

        return (horizon->waterConductivity.kSat * pow(Se, horizon->waterConductivity.l) * pow(myTmp , 2.0));
    }


    /*!
     * \brief Compute water conductivity from signed water potential
     * \param signPsi: water potential       [kPa]
     * \param horizon
     * \return water conductivity           [cm day-1]
     */
    double waterConductivityFromSignPsi(double signPsi, Crit3DHorizon* horizon)
    {
        double theta = soil::thetaFromSignPsi(signPsi, horizon);
        double degreeOfSaturation = SeFromTheta(theta, horizon);
        return waterConductivity(degreeOfSaturation, horizon);
    }


    /*!
     * \brief get water content corresponding to a specific water potential
     * \param psi: water potential  [kPa]
     * \param layer: pointer to Crit3DLayer class
     * \return water content        [mm]
     */
    double getWaterContentFromPsi(double psi, Crit3DLayer* layer)
    {
        double theta = soil::thetaFromSignPsi(-psi, layer->horizon);
        return theta * layer->thickness * layer->soilFraction * 1000;
    }


    /*!
     * \brief get water content corresponding to a specific available water
     * \param availableWater    [-] (0: wilting point, 1: field capacity)
     * \param layer: Crit3DLayer class
     * \return  water content   [mm]
     */
    double getWaterContentFromAW(double availableWater, const Crit3DLayer& layer)
    {
        if (availableWater < 0)
            return layer.WP;

        else if (availableWater > 1)
            return layer.FC;

        else
            return layer.WP + availableWater * (layer.FC - layer.WP);
    }


    /*!
     * \brief return current volumetric water content [m3 m^3]
     */
    double Crit3DLayer::getVolumetricWaterContent()
    {
        // waterContent [mm]
        // thickness [m]
        double theta = waterContent / (thickness * soilFraction * 1000);
        return theta;
    }

    /*!
     * \brief return degree of saturation [-]
     */
    double Crit3DLayer::getDegreeOfSaturation()
    {
        double theta = getVolumetricWaterContent();
        return (theta - horizon->vanGenuchten.thetaR) / (horizon->vanGenuchten.thetaS - horizon->vanGenuchten.thetaR);
    }


    /*!
     * \brief get current water potential
     * \return water potential [kPa]
     */
    double Crit3DLayer::getWaterPotential()
    {
        double theta = getVolumetricWaterContent();
        return psiFromTheta(theta, horizon);
    }


    /*!
     * \brief get current water conductivity
     * \return hydraulic conductivity   [cm day^-1]
     */
    double Crit3DLayer::getWaterConductivity()
    {
        double theta = getVolumetricWaterContent();
        double degreeOfSaturation = SeFromTheta(theta, horizon);
        return waterConductivity(degreeOfSaturation, horizon);
    }


    /*!
      * \brief estimate organic matter as function of depth
      * \param upperDepth: upper depth of soil layer [m]
      * \return organic matter [-]
      */
    double estimateOrganicMatter(double upperDepth)
    {
        // surface 2%
        if (upperDepth == 0.0) return 0.02;
        // first layer 1%
        if (upperDepth > 0 && upperDepth < 0.5) return 0.01;
        // sub-surface 0.5%
        return MINIMUM_ORGANIC_MATTER;
    }


    /*!
     * \brief Set soil properties of one horizon starting from data in soil db
     * \brief it assumes that horizon->dbData and textureClassList are just loaded
     * \return true if soil properties are correct, false otherwise
     */
    bool setHorizon(Crit3DHorizon* horizon, Crit3DTextureClass* textureClassList,
                    Crit3DFittingOptions* fittingOptions, std::string &error)
    {
        error = "";

        // depth [cm]->[m]
        if (horizon->dbData.upperDepth != NODATA && horizon->dbData.lowerDepth != NODATA)
        {
            horizon->upperDepth = horizon->dbData.upperDepth / 100;
            horizon->lowerDepth = horizon->dbData.lowerDepth / 100;
        }
        else
        {
            error += "wrong depth";
            return false;
        }

        // coarse fragments [%] -> 0-1
        if (horizon->dbData.coarseFragments != NODATA
                && horizon->dbData.coarseFragments >= 0
                && horizon->dbData.coarseFragments < 100)
        {
            horizon->coarseFragments = horizon->dbData.coarseFragments / 100;
        }
        else
        {
            // default: no coarse fragment
            horizon->coarseFragments = 0.0;
        }

        // organic matter [%] -> 0-1
        if (horizon->dbData.organicMatter != NODATA
                && horizon->dbData.organicMatter > 0
                && horizon->dbData.organicMatter < 100)
        {
            horizon->organicMatter = horizon->dbData.organicMatter / 100;
        }
        else
        {
            horizon->organicMatter = estimateOrganicMatter(horizon->upperDepth);
        }

        // sand, silt, clay [%]
        horizon->texture.sand = horizon->dbData.sand;
        horizon->texture.silt = horizon->dbData.silt;
        horizon->texture.clay = horizon->dbData.clay;
        if (horizon->texture.sand <= 1 && horizon->texture.silt <= 1 && horizon->texture.clay <= 1)
        {
            horizon->texture.sand *= 100;
            horizon->texture.silt *= 100;
            horizon->texture.clay *= 100;
        }

        // texture
        horizon->texture.classUSDA = soil::getUSDATextureClass(horizon->texture);
        if (horizon->texture.classUSDA == NODATA)
        {
            error = "sand+silt+clay <> 100";
            return false;
        }

        horizon->texture.classNL = soil::getNLTextureClass(horizon->texture);
        horizon->texture.classNameUSDA = textureClassList[horizon->texture.classUSDA].classNameUSDA;

        // assign default parameters from texture class
        horizon->vanGenuchten = textureClassList[horizon->texture.classUSDA].vanGenuchten;
        horizon->waterConductivity = textureClassList[horizon->texture.classUSDA].waterConductivity;
        horizon->Driessen = textureClassList[horizon->texture.classNL].Driessen;

        // theta sat [m3 m-3]
        if (horizon->dbData.thetaSat != NODATA && horizon->dbData.thetaSat > 0 && horizon->dbData.thetaSat < 1)
        {
            horizon->vanGenuchten.thetaS = horizon->dbData.thetaSat;
        }

        // bulk density [g cm-3]
        horizon->bulkDensity = NODATA;
        if (horizon->dbData.bulkDensity != NODATA && horizon->dbData.bulkDensity > 0 && horizon->dbData.bulkDensity < QUARTZ_DENSITY)
        {
            horizon->bulkDensity = horizon->dbData.bulkDensity;  
        }
        else
        {
            horizon->bulkDensity = soil::estimateBulkDensity(horizon, horizon->vanGenuchten.thetaS, true);
        }

        // theta sat from bulk density
        if(horizon->dbData.thetaSat == NODATA)
        {
            horizon->vanGenuchten.thetaS = soil::estimateThetaSat(horizon, horizon->bulkDensity);
        }

        // water retention curve fitting
        if (fittingOptions->useWaterRetentionData && horizon->dbData.waterRetention.size() > 0)
        {
            fittingWaterRetentionCurve(horizon, fittingOptions);

            // bulk density from fitted theta sat
            if (horizon->dbData.bulkDensity == NODATA)
            {
                horizon->bulkDensity = soil::estimateBulkDensity(horizon, horizon->vanGenuchten.thetaS, false);
            }
        }

        // Ksat = saturated water conductivity [cm day-1]
        if (horizon->dbData.kSat != NODATA && horizon->dbData.kSat > 0)
        {
            // check ksat value
            if (horizon->dbData.kSat < (horizon->waterConductivity.kSat / 100))
            {
                horizon->waterConductivity.kSat /= 100;
                error = "Ksat is out of class limits.";
            }
            else if (horizon->dbData.kSat > (horizon->waterConductivity.kSat * 100))
            {
                horizon->waterConductivity.kSat *= 100;
                error = "Ksat is out of class limits.";
            }
            else
            {
                horizon->waterConductivity.kSat = horizon->dbData.kSat;
            }
        }
        else
        {
            horizon->waterConductivity.kSat = soil::estimateSaturatedConductivity(horizon, horizon->bulkDensity);
        }

        // update with coarse fragment (TODO check altre parti del codice!)
        //horizon->vanGenuchten.thetaS *= (1.0 - horizon->coarseFragments);
        //horizon->vanGenuchten.thetaR *= (1.0 - horizon->coarseFragments);

        horizon->CEC = 50.0;
        horizon->PH = 7.7;

        horizon->fieldCapacity = soil::getFieldCapacity(horizon, soil::KPA);
        horizon->wiltingPoint = soil::getWiltingPoint(soil::KPA);
        horizon->waterContentFC = soil::thetaFromSignPsi(horizon->fieldCapacity, horizon);
        horizon->waterContentWP = soil::thetaFromSignPsi(horizon->wiltingPoint, horizon);

        return true;
    }


    /*!
     * \brief fit water retention curve of the horizon
     * \brief using data in water_retention table in soil db
     * \return true if success, false otherwise
     */
    bool fittingWaterRetentionCurve(Crit3DHorizon* horizon, Crit3DFittingOptions* fittingOptions)
    {
        if (! fittingOptions->useWaterRetentionData || horizon->dbData.waterRetention.size() == 0)
        {
            // nothing to do
            return true;
        }

        if (fittingOptions->waterRetentionCurve != MODIFIEDVANGENUCHTEN)
        {
            // TODO other functions
            return false;
        }

        unsigned int nrObsValues = unsigned(horizon->dbData.waterRetention.size());

        // search theta max
        double psiMin = 10000;      // [kpa]
        double thetaMax = 0;        // [m3 m-3]
        for (unsigned int i = 0; i < nrObsValues; i++)
        {
            psiMin = std::min(psiMin, horizon->dbData.waterRetention[i].water_potential);
            thetaMax = std::max(thetaMax, horizon->dbData.waterRetention[i].water_content);
        }
        bool addThetaSat = ((thetaMax < horizon->vanGenuchten.thetaS) && (psiMin > 3));

        // set values
        unsigned int nrValues = nrObsValues;
        unsigned int firstIndex = 0;
        if (addThetaSat)
        {
            nrValues++;
            firstIndex = 1;
        }
        double* x = new double[nrValues];
        double* y = new double[nrValues];

        if (addThetaSat)
        {
            x[0] = 0.0;
            y[0] = horizon->vanGenuchten.thetaS;
        }
        for (unsigned int i = 0; i < nrObsValues; i++)
        {
            x[i + firstIndex] = horizon->dbData.waterRetention[i].water_potential;
            y[i + firstIndex] = horizon->dbData.waterRetention[i].water_content;
        }

        int functionCode;
        unsigned int nrParameters;
        int nrIterations = 200;

        if (fittingOptions->mRestriction)
        {
            functionCode = FUNCTION_CODE_MODIFIED_VAN_GENUCHTEN_RESTRICTED;
            nrParameters = 5;
        }
        else
        {
            functionCode = FUNCTION_CODE_MODIFIED_VAN_GENUCHTEN;
            nrParameters = 6;
        }

        // parameters
        double* param = new double[nrParameters];
        double* pmin = new double[nrParameters];
        double* pmax = new double[nrParameters];
        double* pdelta = new double[nrParameters];

        // water content at saturation [m^3 m^-3]
        param[0] = horizon->vanGenuchten.thetaS;
        pmin[0] = 0;
        pmax[0] = 1;

        // water content residual [m^3 m^-3]
        param[1] = horizon->vanGenuchten.thetaR;
        pmin[1] = 0;
        pmax[1] = horizon->vanGenuchten.thetaR;

        // air entry [kPa]
        param[2] = horizon->vanGenuchten.he;
        if (fittingOptions->airEntryFixed)
        {
            pmin[2] = horizon->vanGenuchten.he;
            pmax[2] = horizon->vanGenuchten.he;
        }
        else
        {
            pmin[2] = 0.01;
            pmax[2] = 10;
        }

        // Van Genuchten alpha parameter [kPa^-1]
        param[3] = horizon->vanGenuchten.alpha;
        pmin[3] = 0.01;
        pmax[3] = 10;

        // Van Genuchten n parameter [-]
        param[4] = horizon->vanGenuchten.n;
        if (fittingOptions->mRestriction)
        {
            pmin[4] = 1;
            pmax[4] = 10;
        }
        else
        {
            pmin[4] = 0.01;
            pmax[4] = 10;

            // Van Genuchten m parameter (restricted: 1-1/n) [-]
            param[5] = horizon->vanGenuchten.m;
            pmin[5] = 0.01;
            pmax[5] = 1;
        }

        for (unsigned int i = 0; i < nrParameters; i++)
        {
            pdelta[i] = (pmax[i]-pmin[i]) * 0.001;
        }

        if ( interpolation::fittingMarquardt(pmin, pmax, param, signed(nrParameters), pdelta,
                                   nrIterations, EPSILON, functionCode, x, y, signed(nrValues)) )
        {
            horizon->vanGenuchten.thetaS = param[0];
            horizon->vanGenuchten.thetaR = param[1];
            horizon->vanGenuchten.he = param[2];
            horizon->vanGenuchten.alpha = param[3];
            horizon->vanGenuchten.n = param[4];
            if (fittingOptions->mRestriction)
            {
                horizon->vanGenuchten.m = 1 - 1 / horizon->vanGenuchten.n;
            }
            else
            {
                horizon->vanGenuchten.m = param[5];
            }
            horizon->vanGenuchten.sc = pow(1 + pow(horizon->vanGenuchten.alpha * horizon->vanGenuchten.he, horizon->vanGenuchten.n), -horizon->vanGenuchten.m);

            return true;
        }
        else
        {
            return false;
        }
    }


    // Compares two Crit3DWaterRetention according to water_potential
    bool sortWaterPotential(soil::Crit3DWaterRetention first, soil::Crit3DWaterRetention second)
    {
        return (first.water_potential < second.water_potential);
    }


    bool Crit3DSoil::setSoilLayers(double layerThicknessMin, double geometricFactor,
                                   std::vector<Crit3DLayer> &soilLayers, std::string &myError)
    {
        soilLayers.clear();

        // layer 0: surface
        soilLayers.resize(1);
        soilLayers[0].depth = 0.0;
        soilLayers[0].thickness = 0.0;

        // layer > 0: soil
        unsigned int i = 1;
        double upperDepth = 0.0;
        double currentThikness = layerThicknessMin;

        while (upperDepth < totalDepth)
        {
            Crit3DLayer newLayer;
            newLayer.thickness = round(currentThikness*100) / 100;
            newLayer.depth = upperDepth + newLayer.thickness * 0.5;

            // last layer: thickness reduced
            if ((upperDepth + newLayer.thickness) > totalDepth)
            {
                newLayer.thickness = totalDepth - upperDepth;
                newLayer.depth = upperDepth + newLayer.thickness/2;
            }

            // get soil horizon
            int horizonIndex = getHorizonIndex(newLayer.depth);
            if (horizonIndex == NODATA)
            {
                myError = "No horizon defined for depth:" + std::to_string(newLayer.depth);
                return false;
            }

            // set layer properties from soil horizon
            Crit3DHorizon* horizonPointer = &(horizon[horizonIndex]);
            if (! newLayer.setLayer(horizonPointer))
            {
                return false;
            }

            soilLayers.push_back(newLayer);

            // update depth
            upperDepth += newLayer.thickness;
            currentThikness *= geometricFactor;
            i++;
        }
        return true;
    }

}

