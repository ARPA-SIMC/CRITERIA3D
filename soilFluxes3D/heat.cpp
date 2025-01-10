/*!
    \name heat.cpp
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


#include <math.h>
#include <stdlib.h>

#include "commonConstants.h"
#include "physics.h"
#include "header/types.h"
#include "header/heat.h"
#include "header/soilPhysics.h"
#include "header/balance.h"
#include "header/water.h"
#include "header/solver.h"
#include "header/soilFluxes3D.h"
#include "header/boundary.h"

static double CourantHeat, fluxCourant;

bool isHeatNode(long i)
{
    return (myStructure.computeHeat &&
            nodeList != nullptr &&
            nodeList[i].extra != nullptr &&
            nodeList[i].extra->Heat != nullptr &&
            ! nodeList[i].isSurface);
}

bool isHeatLinkedNode(TlinkedNode* myLink)
{
    return (myStructure.computeHeat &&
            myLink != nullptr &&
            myLink->linkedExtra != nullptr &&
            myLink->linkedExtra->heatFlux != nullptr);
}

double getH_timeStep(long i, double timeStep, double timeStepWater)
{
    return (nodeList[i].H - nodeList[i].oldH) / timeStepWater * timeStep + nodeList[i].oldH;
}

double computeHeatStorage(double timeStepHeat, double timeStepWater)
{ // [J]
    double myHeatStorage = 0.;
    double myH;
    for (long i = 1; i < myStructure.nrNodes; i++)
    {
        if (timeStepHeat != NODATA && timeStepWater != NODATA)
            myH = getH_timeStep(i, timeStepHeat, timeStepWater);
        else
            myH = nodeList[i].H;

        myHeatStorage += soilFluxes3D::getHeat(i, myH - nodeList[i].z);
    }
    return myHeatStorage;
}

/*!
 * \brief computes sum of heat sink/source (J)
 * \param deltaT
 * \return result
 */
double sumHeatFlow(double deltaT)
{
    double sum = 0.0;
    for (long n = 1; n < myStructure.nrNodes; n++)
    {
        if (nodeList[n].extra->Heat->Qh != 0.)
            sum += nodeList[n].extra->Heat->Qh * deltaT;
    }
    return (sum);
}

void computeHeatBalance(double myTimeStep, double timeStepWater)
{
    balanceCurrentTimeStep.sinkSourceHeat = sumHeatFlow(myTimeStep);

    balanceCurrentTimeStep.storageHeat = computeHeatStorage(myTimeStep, timeStepWater);

    double deltaHeatStorage = balanceCurrentTimeStep.storageHeat - balancePreviousTimeStep.storageHeat;
    balanceCurrentTimeStep.heatMBE = deltaHeatStorage - balanceCurrentTimeStep.sinkSourceHeat;

    double referenceHeat = MAXVALUE(fabs(balanceCurrentTimeStep.sinkSourceHeat), balanceCurrentTimeStep.storageHeat * 1e-6);
    balanceCurrentTimeStep.heatMBR = 1. - balanceCurrentTimeStep.heatMBE / referenceHeat;
}

float readHeatFlux(TlinkedNode* myLink, int fluxType)
{
    if (! isHeatLinkedNode(myLink)) return NODATA;

    if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_TOTAL && fluxType == HEATFLUX_TOTAL)
        return myLink->linkedExtra->heatFlux->fluxes[HEATFLUX_TOTAL];
    else if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_ALL && (fluxType == HEATFLUX_TOTAL ||
            fluxType == HEATFLUX_DIFFUSIVE ||
            fluxType == HEATFLUX_LATENT_ISOTHERMAL ||
            fluxType == HEATFLUX_LATENT_THERMAL ||
            fluxType == HEATFLUX_ADVECTIVE ||
            fluxType == WATERFLUX_LIQUID_ISOTHERMAL ||
            fluxType == WATERFLUX_LIQUID_THERMAL ||
            fluxType == WATERFLUX_VAPOR_ISOTHERMAL ||
            fluxType == WATERFLUX_VAPOR_THERMAL))

        return myLink->linkedExtra->heatFlux->fluxes[fluxType];
    else
        return NODATA;
}

void saveHeatFlux(TlinkedNode* myLink, int fluxType, double myValue)
{
    if (! isHeatLinkedNode(myLink)) return;

    if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_NONE) return;

    if (myLink->linkedExtra->heatFlux->fluxes[HEATFLUX_TOTAL] == NODATA)
        myLink->linkedExtra->heatFlux->fluxes[HEATFLUX_TOTAL] = float(myValue);
    else
        myLink->linkedExtra->heatFlux->fluxes[HEATFLUX_TOTAL] += float(myValue);

    if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_ALL)
        myLink->linkedExtra->heatFlux->fluxes[fluxType] = float(myValue);
}

/*!
 * \brief [m3 m-3] vapor volumetric water equivalent
 * \param [m] h
 * \param [K] temperature
 * \param i
 * \return result
 */
double VaporThetaV(double h, double T, long i)
{
    double theta = theta_from_sign_Psi(h, i);
    double vaporConc = VaporFromPsiTemp(h, T);
    return (vaporConc / WATER_DENSITY * (nodeList[i].Soil->Theta_s - theta));
}

/*!
 * \brief [m2 s-1] binary vapor diffusivity
 * (Do) in Bittelli (2008) or vapor diffusion coefficient in air (Dva) in Monteith (1973)
 * \param myPressure
 * \param myTemperature
 * \return result
 */
double VaporBinaryDiffusivity(double myTemperature)
{	return (VAPOR_DIFFUSIVITY0 * pow(myTemperature / ZEROCELSIUS, 2)); }

/*!
 * \brief [m2 s-1] vapor diffusivity
 * \param i
 * \param myT
 * \return result
 */
double SoilVaporDiffusivity(double ThetaS, double Theta, double myT)
{
	double binaryDiffusivity;	// [m2 s-1]
	double airFilledPorosity;	// [m3 m-3]
    double const beta = 0.66;	// [] Penman 1940
    double const emme = 1;      // [] idem

    binaryDiffusivity = VaporBinaryDiffusivity(myT);
    airFilledPorosity = ThetaS - Theta;

    return (binaryDiffusivity  * beta * pow(airFilledPorosity, emme));
}

/*!
 * \brief [] soil relative humidity
 * \param [m] h
 * \param [K] myT
 * \return result
 */
double SoilRelativeHumidity(double h, double myT)
{	return (exp(MH2O * h * GRAVITY / (R_GAS * myT))); }

/*!
 * \brief [kg s m-3] isothermal vapor conductivity
 * \param i
 * \param h
 * \param myT
 * \return result
 */
double IsothermalVaporConductivity(long i, double h, double myT)
{
    double theta = theta_from_sign_Psi(h, i);
    double Dv = SoilVaporDiffusivity(nodeList[i].Soil->Theta_s, theta, myT);
    double vapor = VaporFromPsiTemp(h, myT);
    return (Dv * vapor * MH2O / (R_GAS * myT));
}

/*!
 * \brief [J m-3 K-1] volumetric heat capacity
 * \param i
 * \param h
 * \param T
 * \return result
 */
double SoilHeatCapacity(long i, double h, double T)
{
    double heatCapacity;
    double theta = theta_from_sign_Psi(h, i);
    double thetaV = VaporThetaV(h, T, i);
    double bulkDensity = estimateBulkDensity(i);
    heatCapacity = bulkDensity / 2.65 * HEAT_CAPACITY_MINERAL +
            theta * HEAT_CAPACITY_WATER;

    if (myStructure.computeHeatVapor)
        heatCapacity += thetaV * HEAT_CAPACITY_AIR;

    return heatCapacity;
}

/*!
 * \brief [] water return flow factor
 * Campbell 1994
 * \param myTheta
 * \param myClayFraction
 * \param myTemperature
 * \return result
 */
double WaterReturnFlowFactor(double myTheta, double myClayFraction, double myTemperature)
{
    double Q0, Q;                                   // [] power
    double xw0 = 0.33 * myClayFraction + 0.078;		// [] cutoff water content
    if (myTheta < (0.01 * xw0))
		return 0.;
	else
    {
        Q0 = 7.25 * myClayFraction + 2.52;
        Q = Q0 * (pow(myTemperature / 303., 2));
    }

    return (1 / (1 + pow(myTheta / xw0, -Q)));
}

/*!
 * \brief compute vapor concentration from matric potential and temperature
 * \param Psi [J kg-1]
 * \param T [K]
 * \return vapor concentration [kg m-3]
 */
double VaporFromPsiTemp(double h, double T)
{
    double mySatVapPressure, mySatVapConcentration, myRelHum;

    mySatVapPressure = saturationVaporPressure(T - ZEROCELSIUS);
    mySatVapConcentration = vaporConcentrationFromPressure(mySatVapPressure, T);
    myRelHum = SoilRelativeHumidity(h, T);

    return mySatVapConcentration * myRelHum;
}

/*!
 * \brief [m2 s-1 K-1] thermal liquid conductivity
 * \param i
 * \param temperature (K)
 * \param h (m)
 * \param Klh (m s-1) isotherma liquid conductivity
 * \return result
 */
double ThermalLiquidConductivity(double temp_celsius, double h, double Klh)
{
    double Gwt = 4.;        // [] gain factor (temperature dependence of soil water retention curve)
    double dGammadT;        // [g s-2 K-1] derivative of surface tension with respect to temperature

    dGammadT = -0.1425 - 0.000576 * temp_celsius;
    return (MAXVALUE(0., Klh * h * Gwt * dGammadT / GAMMA0));
}

/*!
 * \brief [kg m-1 s-1 K-1] thermal vapor conductivity
 * \param i
 * \param temperature (K)
 * \param h (m)
 * \return result
 */
double ThermalVaporConductivity(long i, double temperature, double h)
{
    double myPressure;				// [Pa] total air pressure
	double Dv;						// [m2 s-1] vapor diffusivity
	double svp;						// [Pa] saturation vapor pressure
	double slopesvp;				// [Pa K-1] slope of saturation vapor pressure curve
    double slopesvc;                // [kg m-3 K-1] slope of saturation vapor concentration
    double myVapor;					// [kg m-3] vapor concentration
	double myVaporPressure;			// [Pa] vapor partial pressure
	double hr;						// [] relative humidity
    double tempCelsius;             // [°C] temperature
    double theta;                   // [m3 m-3] volumetric water content
    double eta;                     // [] enhancement factor
    double satDegree;               // [] degree of saturation

    tempCelsius = temperature - ZEROCELSIUS;

    myPressure = pressureFromAltitude(nodeList[i].z);

    theta = theta_from_sign_Psi(h, i);

	// vapor diffusivity
    Dv = SoilVaporDiffusivity(nodeList[i].Soil->Theta_s, theta, temperature);

	// slope of saturation vapor pressure
    svp = saturationVaporPressure(tempCelsius);
    slopesvp = saturationSlope(tempCelsius, svp / 1000);

    // slope of saturation vapor concentration
    slopesvc = slopesvp * MH2O * airMolarDensity(myPressure, temperature) / myPressure;

	// relative humidity
    myVapor = VaporFromPsiTemp(h, temperature);
    myVaporPressure = vaporPressureFromConcentration(myVapor, temperature);
	hr = myVaporPressure / svp;

    // enhancement factor (Cass et al. 1984)
    satDegree = theta / nodeList[i].Soil->Theta_s;
    eta = 9.5 + 3. * satDegree - 8.5 * exp(-pow((1. + 2.6/sqrt(nodeList[i].Soil->clay))*satDegree, 4));

    return (eta * Dv * slopesvc * hr);

}

/*!
 * \brief [W m-1 K-1] air thermal conductivity
 * \param i
 * \param T: temperature [K]
 * \param h: water matric potential [m]
 * \return result
 */
double AirHeatConductivity(long i, double T, double h)
{
    double Kda;						// [W m-1 K-1] thermal conductivity of dry air
    double Ka;						// [W m-1 K-1] thermal conductivity of air
    double myKvt;                   // [kg m-1 s-1 K-1] non isothermal vapor conductivity
    double myLambda;				// [J kg-1] latent heat of vaporization
    double myTCelsiusMean;          // [degC]
    double coeff;                   // [J kg-1]

    // dry air conductivity
    myTCelsiusMean = T - ZEROCELSIUS;
	Kda = 0.024 + 0.0000773 * myTCelsiusMean - 0.000000026 * myTCelsiusMean * myTCelsiusMean;

    Ka = Kda;

    if (myStructure.computeWater)
    {
        myLambda = latentHeatVaporization(T - ZEROCELSIUS);

        coeff= myLambda;

        myKvt = ThermalVaporConductivity(i, T, h);
        Ka += coeff * myKvt;
    }

	return (Ka);
}

/*!
 * \brief [W m-1 K-1] soil thermal conductivity
 * according to Campbell et al. Soil Sci. 158:307-313
 * \param i
 * \param T: temperature [K]
 * \param h: water matric potential [m]
 * \return result
 */
double SoilHeatConductivity(long i, double T, double h)
{
	double ga = 0.088;				// [] deVries shape factor; assume same for all mineral soils
	double gc;						// [] shape factor
	double ea;						// [] air weighting factor
	double es ;						// [] solid weighting factor
	double ew;						// [] water weighting factor
	double Ka;						// [W m-1 K-1] thermal conductivity of air
	double Kw;						// [W m-1 K-1] thermal conductivity of water
	double Kf;						// [W m-1 K-1] thermal conductivity of fluids
	double xa;						// [m3 m-3] volume fraction of air
	double xw;						// [m3 m-3] volume fraction of water
	double xs;						// [m3 m-3] volume fraction of solids
	double myConductivity;			// [W m-1 K-1] total thermal conductivity
	double myTCelsiusMean;
    double fw;						// [] water return flow factor (same in air conductivity)

    myTCelsiusMean = T - ZEROCELSIUS;

	// water conductivity
	Kw = 0.554 + 0.0024 * myTCelsiusMean - 0.00000987 * myTCelsiusMean * myTCelsiusMean;

	// air conductivity
    Ka = AirHeatConductivity(i, T, h);

    xw = theta_from_sign_Psi(h, i);

    fw = WaterReturnFlowFactor(xw, nodeList[i].Soil->clay, myTCelsiusMean + ZEROCELSIUS);
	Kf = Ka + fw * (Kw - Ka);

	gc = 1. - 2. * ga;

    ea = (2. / (1 + (Ka / Kf - 1) * ga) + 1 / (1 + (Ka / Kf - 1) * gc)) / 3.;
	ew = (2. / (1 + (Kw / Kf - 1) * ga) + 1 / (1 + (Kw / Kf - 1) * gc)) / 3.;
    es = (2. / (1 + (KH_mineral / Kf - 1) * ga) + 1 / (1 + (KH_mineral / Kf - 1) * gc)) / 3.;

	xs = 1. - nodeList[i].Soil->Theta_s;
	xa = nodeList[i].Soil->Theta_s - xw;

    myConductivity = (xw * ew * Kw + xa * ea * Ka + xs * es * KH_mineral) / (ew * xw + ea * xa + es * xs);
    return myConductivity;
}

/*!
 * \brief [m3 s-1] Thermal liquid flux
 * \param i
 * \param myLink
 * \return result
 */
double ThermalLiquidFlux(long i, TlinkedNode *myLink, int myProcess, double timeStep, double timeStepWater)
{
    //TODO: inserire time step water per calcolo più preciso

    long j = (*myLink).index;

    // temperatures (K) and water potential (m)
    double tavg, tavgLink, havg, havgLink;
    if (myProcess == PROCESS_WATER && myStructure.computeWater)
    {
        tavg = getTMean(i);
        tavgLink = getTMean(j);
        havg = nodeList[i].H - nodeList[i].z;
        havgLink = nodeList[j].H - nodeList[j].z;
    }
    else if (myProcess == PROCESS_HEAT && myStructure.computeHeat)
    {
        tavg = nodeList[i].extra->Heat->T;
        tavgLink = nodeList[j].extra->Heat->T;
        havg = arithmeticMean(getH_timeStep(i, timeStep, timeStepWater), nodeList[i].oldH) - nodeList[i].z;
        havgLink = arithmeticMean(getH_timeStep(j, timeStep, timeStepWater), nodeList[j].oldH) - nodeList[j].z;
    }
    else
        return NODATA;

    // m2 K-1 s-1
    double Klt = ThermalLiquidConductivity(tavg - ZEROCELSIUS, havg, nodeList[i].k);
    double KltLink = ThermalLiquidConductivity(tavgLink - ZEROCELSIUS, havgLink, nodeList[j].k);
    double meanKlt = computeMean(Klt, KltLink);

    // m s-1
    double myFlowDensity = meanKlt * (tavgLink - tavg) / distance(i, j);

    // m3 s-1
    double myFlow = myFlowDensity * (*myLink).area;

    return (myFlow);
}

/*!
 * \brief [kg s-1] Thermal vapor flux
 * \param i
 * \param myLink
 * \return result
 */
double ThermalVaporFlux(long i, TlinkedNode *myLink, int myProcess, double timeStep, double timeStepWater)
{
    //TODO: inserire time step water per calcolo più preciso

    long j = (*myLink).index;

    // temperatures (K) and water potential (m)
    double tavg, tavgLink, havg, havgLink;
    if (myProcess == PROCESS_WATER && myStructure.computeWater)
    {
        tavg = getTMean(i);
        tavgLink = getTMean(j);
        havg = nodeList[i].H - nodeList[i].z;
        havgLink = nodeList[j].H - nodeList[j].z;
    }
    else
    {
        if (myProcess == PROCESS_HEAT && myStructure.computeHeat)
        {
            tavg = nodeList[i].extra->Heat->T;
            tavgLink = nodeList[j].extra->Heat->T;
            havg = arithmeticMean(getH_timeStep(i, timeStep, timeStepWater), nodeList[i].oldH) - nodeList[i].z;
            havgLink = arithmeticMean(getH_timeStep(j, timeStep, timeStepWater), nodeList[j].oldH) - nodeList[j].z;
        }
        else
            return NODATA;
    }

    // kg m-1 s-1 K-1
    double Kvt = ThermalVaporConductivity(i, tavg, havg);
    double KvtLink = ThermalVaporConductivity(j, tavgLink, havgLink);
    double meanKv = computeMean(Kvt, KvtLink);

    // kg m-2 s-1
    double myFlowDensity = meanKv * (tavgLink - tavg) / distance(i, j);

    // kg s-1
    double myFlow = myFlowDensity * (*myLink).area;

    return (myFlow);
}

/*!
 * \brief isothermal vapor flux
 * \param i
 * \param myLink
 * \return isothermal vapor flux [kg s-1]
 */
double IsothermalVaporFlux(long i, TlinkedNode *myLink, double timeStep, double timeStepWater)
{
    double myKvi;								// [kg s m-3] vapor conductivity
    double psi, psiLink;                        // [J kg-1 = m2 s-2] water matric potential
    double deltaPsi;							// [J kg-1 = m2 s-2] water potential difference
    double myFlux;                              // [kg s-1] latent heat flow
    double Kvi, KviLink;                        // [kg m-3 s-1] isothermal vapor conductivity
    double havg, havglink;                      // [m] average matric potentials

    long j = (*myLink).index;

    havg = arithmeticMean(getH_timeStep(i, timeStep, timeStepWater), nodeList[i].oldH) - nodeList[i].z;
    havglink = arithmeticMean(getH_timeStep(j, timeStep, timeStepWater), nodeList[j].oldH) - nodeList[j].z;

    Kvi = IsothermalVaporConductivity(i, havg, nodeList[i].extra->Heat->T);
    KviLink = IsothermalVaporConductivity(j, havglink, nodeList[j].extra->Heat->T);
    myKvi = computeMean(Kvi, KviLink);

    psi = havg * GRAVITY;
    psiLink = havglink * GRAVITY;

    deltaPsi = (psiLink - psi);

    myFlux = myKvi * deltaPsi / distance(i, j) * myLink->area;

    return (myFlux);
}

/*!
 * \brief isothermal latent heat flux
 * \param i
 * \param myLink
 * \return isothermal latent heat flux [W]
 */
double IsothermalLatentHeatFlux(long i, TlinkedNode *myLink, double timeStep, double timeStepWater)
{
    double lambda, lambdaLink, avgLambda;       // [J kg-1] latent heat of vaporization
    double myLatentFlux;						// [J s-1] latent heat flow

    long j = (*myLink).index;

    lambda = latentHeatVaporization(nodeList[i].extra->Heat->T - ZEROCELSIUS);
    lambdaLink = latentHeatVaporization(nodeList[j].extra->Heat->T - ZEROCELSIUS);
    avgLambda = arithmeticMean(lambda, lambdaLink);

    myLatentFlux = avgLambda * IsothermalVaporFlux(i, myLink, timeStep, timeStepWater);

    return (myLatentFlux);
}

/*!
 * \brief advective isothermal liquid water heat flux
 * \param i
 * \param myLink
 * \return advective liquid water heat flux [W]
 */
double AdvectiveFlux(long i, TlinkedNode *myLink)
{
    double TliqAdv, TvapAdv;
    double liqWaterFlux, vapWaterFlux;
    double advection;

    liqWaterFlux = (*myLink).linkedExtra->heatFlux->waterFlux;

    if (liqWaterFlux < 0.)
        TliqAdv = nodeList[i].extra->Heat->T;
    else
        TliqAdv = nodeList[myLink->index].extra->Heat->T;

    fluxCourant += HEAT_CAPACITY_WATER * liqWaterFlux;
    advection = fluxCourant * TliqAdv;

    vapWaterFlux = (*myLink).linkedExtra->heatFlux->vaporFlux;

    if (vapWaterFlux < 0.)
        TvapAdv = nodeList[i].extra->Heat->T;
    else
        TvapAdv = nodeList[myLink->index].extra->Heat->T;

    double fluxCourantVap = HEAT_CAPACITY_WATER_VAPOR * vapWaterFlux;
    fluxCourant += fluxCourantVap;
    advection += fluxCourantVap * TvapAdv;

    return (advection);
}


double Conduction(long i, TlinkedNode *myLink, double timeStep, double timeStepWater)
{
	double myConductivity, linkConductivity, meanKh;
    double zeta;
    double hAvg, hLinkAvg;
    double myH, myHLink;

    long j = (*myLink).index;
    double myDistance = distance(i, j);

    zeta = myLink->area / myDistance;

    myH = getH_timeStep(i, timeStep, timeStepWater);
    myHLink = getH_timeStep(j, timeStep, timeStepWater);
    hAvg = arithmeticMean(myH, nodeList[i].oldH) - nodeList[i].z;
    hLinkAvg = arithmeticMean(myHLink, nodeList[j].oldH) - nodeList[j].z;

    myConductivity = SoilHeatConductivity(i, nodeList[i].extra->Heat->T, hAvg);
    linkConductivity = SoilHeatConductivity(j, nodeList[j].extra->Heat->T, hLinkAvg);
    meanKh = computeMean(myConductivity, linkConductivity);

    return (zeta * meanKh);
}

bool computeHeatFlux(long i, int myMatrixIndex, TlinkedNode *myLink, double timeStep, double timeStepWater)
{
    if (myLink == nullptr) return false;
    if ((*myLink).index == NOLINK) return false;

    long myLinkIndex = (*myLink).index;
    double myConduction, myAdvectiveFlux, myLatentFlux;
    double nodeDistance;

    if (! isHeatNode(myLinkIndex)) return false;

    myAdvectiveFlux = 0.;
    myLatentFlux = 0.;
    fluxCourant = 0.;

    myConduction = Conduction(i, myLink, timeStep, timeStepWater);
    if (myStructure.computeWater)
    {
        if (myStructure.computeHeatVapor)
        {
            myLatentFlux = IsothermalLatentHeatFlux(i, myLink, timeStep, timeStepWater);
            saveHeatFlux(myLink, HEATFLUX_LATENT_ISOTHERMAL, myLatentFlux);
        }

        if (myStructure.computeHeatAdvection)
        {
            myAdvectiveFlux = AdvectiveFlux(i, myLink);
            saveHeatFlux(myLink, HEATFLUX_ADVECTIVE, myAdvectiveFlux);
        }
    }

    A[i][myMatrixIndex].index = myLinkIndex;
    A[i][myMatrixIndex].val = myConduction;

    invariantFlux[i] += myAdvectiveFlux + myLatentFlux;

    if (fluxCourant != 0)
    {
        nodeDistance = distance(i, myLinkIndex);
        CourantHeat = MAXVALUE(CourantHeat, fabs(fluxCourant) * timeStep / (C[i] * nodeDistance));
    }

    return (true);
}

// should be called only BEFORE heat computation, since A matrix should contain water flux values
void saveNodeWaterFlux(long i, TlinkedNode *link, double timeStepHeat, double timeStepWater)
{
    if (link == nullptr) return;

    double fluxLiquid = 0.;         // m3 s-1
    double fluxVapor = 0.;          // kg s-1
    double isothVapFlux = 0.;
    double isothLiqFlux = 0.;
    double thermLiqFlux = 0.;
    double thermVapFlux = 0.;

    double avgH, avgHLink;
    avgH = getH_timeStep(i, timeStepHeat, timeStepWater);
    avgHLink = getH_timeStep(link->index, timeStepHeat, timeStepWater);

    double matrixValue = getMatrixValue(i, link);
    if (matrixValue != INDEX_ERROR)
        isothLiqFlux = matrixValue * (avgH - avgHLink);

    if (!nodeList[i].isSurface && ! nodeList[link->index].isSurface)
    {
        // compute isothermal vapor flux and subtract from total water flux
        // (because fluxLiquid is computed from A matrix which include isothermal vapor flux component)
        isothVapFlux = IsothermalVaporFlux(i, link, timeStepHeat, timeStepWater);

        // thermal liquid flux
        thermLiqFlux = ThermalLiquidFlux(i, link, PROCESS_HEAT, timeStepHeat, timeStepWater);

        // thermal vapor flux
        thermVapFlux = ThermalVaporFlux(i, link, PROCESS_HEAT, timeStepHeat, timeStepWater);
    }

    fluxLiquid = isothLiqFlux - isothVapFlux / WATER_DENSITY + thermLiqFlux;
    fluxVapor = isothVapFlux + thermVapFlux;

    link->linkedExtra->heatFlux->waterFlux = float(fluxLiquid);
    link->linkedExtra->heatFlux->vaporFlux = float(fluxVapor);

    if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_ALL)
    {
        link->linkedExtra->heatFlux->fluxes[WATERFLUX_LIQUID_ISOTHERMAL] = float(isothLiqFlux);
        link->linkedExtra->heatFlux->fluxes[WATERFLUX_LIQUID_THERMAL] = float(thermLiqFlux);
        link->linkedExtra->heatFlux->fluxes[WATERFLUX_VAPOR_ISOTHERMAL] = float(isothVapFlux);
        link->linkedExtra->heatFlux->fluxes[WATERFLUX_VAPOR_THERMAL] = float(thermVapFlux);
    }

    return;
}

void saveWaterFluxes(double dtHeat, double dtWater)
{
    for (long i = 0; i < myStructure.nrNodes; i++)
        {
            if (&nodeList[i].up != nullptr)
                if (nodeList[i].up.linkedExtra != nullptr)
                    saveNodeWaterFlux(i, &nodeList[i].up, dtHeat, dtWater);

            if (&nodeList[i].down != nullptr)
                if (nodeList[i].down.linkedExtra != nullptr)
                    saveNodeWaterFlux(i, &nodeList[i].down, dtHeat, dtWater);

            for (short j = 0; j < myStructure.nrLateralLinks; j++)
                if (&nodeList[i].lateral[j] != nullptr)
                    if (nodeList[i].lateral[j].linkedExtra != nullptr)
                        saveNodeWaterFlux(i, &nodeList[i].lateral[j], dtHeat, dtWater);

        }
}

void saveNodeHeatFlux(long myIndex, TlinkedNode *myLink, double timeStep, double timeStepWater)
// [W] heat flow between node nodeList[myIndex] and link node myLink
{
   if (! isHeatLinkedNode(myLink)) return;

    long myLinkIndex = (*myLink).index;
    double myDiffHeat, myA;

    int j = 1;
    while ((j < myStructure.maxNrColumns) && (A[myIndex][j].index != NOLINK) && (A[myIndex][j].index != myLinkIndex)) j++;

    if (A[myIndex][j].index == myLinkIndex)
    {
        myA = (A[myIndex][j].val * A[myIndex][0].val);
        myDiffHeat = myA * (nodeList[myIndex].extra->Heat->T - nodeList[myLinkIndex].extra->Heat->T) * myParameters.heatWeightingFactor;
        myDiffHeat += myA * (nodeList[myIndex].extra->Heat->oldT - nodeList[myLinkIndex].extra->Heat->oldT) * (1. - myParameters.heatWeightingFactor);

        // when saving separate fluxes, thermal latent heat has to be subtracted from diffusive,
        // where is incorporated (see AirHeatConductivity)
        if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_ALL)
        {
            if (myStructure.computeHeatVapor)
            {
                double thermalLatentFlux = ThermalVaporFlux(myIndex, myLink, PROCESS_HEAT, timeStep, timeStepWater);
                thermalLatentFlux *= latentHeatVaporization(nodeList[myIndex].extra->Heat->T - ZEROCELSIUS);
                saveHeatFlux(myLink, HEATFLUX_LATENT_THERMAL, thermalLatentFlux);
                saveHeatFlux(myLink, HEATFLUX_DIFFUSIVE, myDiffHeat - thermalLatentFlux);
            }
            else
                saveHeatFlux(myLink, HEATFLUX_DIFFUSIVE, myDiffHeat);

        }
        else
        {
            saveHeatFlux(myLink, HEATFLUX_TOTAL, myDiffHeat);
        }
    }
}

void updateHeatFluxes(double timeStep, double timeStepWater)
{
    if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_NONE) return;

    for (long i = 1; i < myStructure.nrNodes; i++)
    {
        if (nodeList[i].up.index != NOLINK)
            if (nodeList[i].up.linkedExtra->heatFlux != nullptr)
                saveNodeHeatFlux(i, &(nodeList[i].up), timeStep, timeStepWater);

        if (nodeList[i].down.index != NOLINK)
            if (nodeList[i].down.linkedExtra->heatFlux != nullptr)
                saveNodeHeatFlux(i, &(nodeList[i].down), timeStep, timeStepWater);

        for (short j = 0; j < myStructure.nrLateralLinks; j++)
            if (nodeList[i].lateral[j].index != NOLINK)
                if (nodeList[i].lateral[j].linkedExtra->heatFlux != nullptr)
                    saveNodeHeatFlux(i, &(nodeList[i].lateral[j]), timeStep, timeStepWater);
    }
}

void updateBalanceHeat()
{
    balancePreviousTimeStep.storageHeat = balanceCurrentTimeStep.storageHeat;
    balancePreviousTimeStep.sinkSourceHeat = balanceCurrentTimeStep.sinkSourceHeat;
    balanceCurrentPeriod.sinkSourceHeat += balanceCurrentTimeStep.sinkSourceHeat;
}

bool heatBalance(double timeStep, double timeStepWater)
{
    computeHeatBalance(timeStep, timeStepWater);
    return ((fabs(1.-balanceCurrentTimeStep.heatMBR) < myParameters.MBRThreshold));
}

void initializeBalanceHeat()
{
     balanceCurrentTimeStep.sinkSourceHeat = 0.;
     balancePreviousTimeStep.sinkSourceHeat = 0.;
     balanceCurrentPeriod.sinkSourceHeat = 0.;
     balanceWholePeriod.sinkSourceHeat = 0.;

     balanceCurrentTimeStep.heatMBE = 0.;
     balanceCurrentPeriod.heatMBE = 0.;
     balanceWholePeriod.waterMBE = 0.;

     balanceCurrentTimeStep.heatMBR = 1.;
     balanceCurrentPeriod.heatMBR = 1.;
     balanceWholePeriod.heatMBR = 1.;

     balanceWholePeriod.storageHeat = computeHeatStorage(NODATA, NODATA);
     balanceCurrentTimeStep.storageHeat = balanceWholePeriod.storageHeat;
     balancePreviousTimeStep.storageHeat = balanceWholePeriod.storageHeat;
     balanceCurrentPeriod.storageHeat = balanceWholePeriod.storageHeat;
}

void updateBalanceHeatWholePeriod()
{
    /*! update the flows in the balance (balanceWholePeriod) */
    balanceWholePeriod.sinkSourceHeat  += balanceCurrentPeriod.sinkSourceHeat;

    double deltaStoragePeriod = balanceCurrentTimeStep.storageHeat - balanceCurrentPeriod.storageHeat;
    double deltaStorageHistorical = balanceCurrentTimeStep.storageHeat - balanceWholePeriod.storageHeat;

    /*! compute MBE and MBR */
    balanceCurrentPeriod.heatMBE = deltaStoragePeriod - balanceCurrentPeriod.sinkSourceHeat;
    balanceWholePeriod.heatMBE = deltaStorageHistorical - balanceWholePeriod.sinkSourceHeat;
    if ((balanceWholePeriod.storageHeat == 0.) && (balanceWholePeriod.sinkSourceHeat == 0.)) balanceWholePeriod.heatMBR = 1.;
    else if (balanceCurrentTimeStep.storageHeat > fabs(balanceWholePeriod.sinkSourceHeat))
        balanceWholePeriod.heatMBR = balanceCurrentTimeStep.storageHeat / (balanceWholePeriod.storageHeat + balanceWholePeriod.sinkSourceHeat);
    else
        balanceWholePeriod.heatMBR = deltaStorageHistorical / balanceWholePeriod.sinkSourceHeat;

    /*! update storageWater in balanceCurrentPeriod */
    balanceCurrentPeriod.storageHeat = balanceCurrentTimeStep.storageHeat;
}

void restoreHeat()
{
    for (long i = 1; i < myStructure.nrNodes; i++)
        nodeList[i].extra->Heat->T = nodeList[i].extra->Heat->oldT;
}

void initializeHeatFluxes(bool initHeat, bool initWater)
{
    for (long n = 0; n < myStructure.nrNodes; n++)
    {
        initializeNodeHeatFlux(nodeList[n].up.linkedExtra, initHeat, initWater);
        initializeNodeHeatFlux(nodeList[n].down.linkedExtra, initHeat, initWater);
        for (short i = 1; i < myStructure.nrLateralLinks; i++)
           initializeNodeHeatFlux(nodeList[n].lateral[i].linkedExtra, initHeat, initWater);
    }
}

double computeMaximumDeltaT()
{
    double maxDeltaT = 0.;
    for (long i = 1; i < myStructure.nrNodes; i++)
        maxDeltaT = MAXVALUE(maxDeltaT, fabs(nodeList[i].extra->Heat->T - nodeList[i].extra->Heat->oldT));

    return maxDeltaT;
}

bool HeatComputation(double timeStep, double timeStepWater)
{

	long i, j;
    double sum = 0;
    double sumFlow0 = 0;
    double myDeltaTemp0;
    double avgh;
    double heatCapacityVar;
    double dtheta, dthetav;
    double myH;

    initializeHeatFluxes(true, false);
    CourantHeat = 0.;

    for (i = 1; i < myStructure.nrNodes; i++)
    {
        A[i][0].index = i;
        X[i] = nodeList[i].extra->Heat->T;
        nodeList[i].extra->Heat->oldT = nodeList[i].extra->Heat->T;

        myH = getH_timeStep(i, timeStep, timeStepWater);
        avgh = arithmeticMean(nodeList[i].oldH, myH) - nodeList[i].z;
        C[i] = SoilHeatCapacity(i, avgh, nodeList[i].extra->Heat->T) * nodeList[i].volume_area;
    }

    for (i = 1; i < myStructure.nrNodes; i++)
    {
        invariantFlux[i] = 0.;

        myH = getH_timeStep(i, timeStep, timeStepWater);

        // compute heat capacity temporal variation
        // due to changes in water and vapor
        dtheta = theta_from_sign_Psi(myH - nodeList[i].z, i) -
                theta_from_sign_Psi(nodeList[i].oldH - nodeList[i].z, i);

        heatCapacityVar = dtheta * HEAT_CAPACITY_WATER * nodeList[i].extra->Heat->T;

        if (myStructure.computeHeatVapor)
        {
            dthetav = VaporThetaV(myH - nodeList[i].z, nodeList[i].extra->Heat->T, i) -
                    VaporThetaV(nodeList[i].oldH - nodeList[i].z, nodeList[i].extra->Heat->oldT, i);
            heatCapacityVar += dthetav * HEAT_CAPACITY_AIR * nodeList[i].extra->Heat->T;
            heatCapacityVar += dthetav * latentHeatVaporization(nodeList[i].extra->Heat->T - ZEROCELSIUS) * WATER_DENSITY;
        }

        heatCapacityVar *= nodeList[i].volume_area;

        j = 1;
        if (computeHeatFlux(i, j, &(nodeList[i].up), timeStep, timeStepWater)) j++;
        for (short l = 0; l < myStructure.nrLateralLinks; l++)
            if (computeHeatFlux(i, j, &(nodeList[i].lateral[l]), timeStep, timeStepWater)) j++;
        if (computeHeatFlux(i, j, &(nodeList[i].down), timeStep, timeStepWater)) j++;

        // closure
        while (j < myStructure.maxNrColumns)
            A[i][j++].index = NOLINK;

        j = 1;
        sum = 0.;
        sumFlow0 = 0;
        myDeltaTemp0 = 0;

        while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK))
        {
            sum += A[i][j].val * myParameters.heatWeightingFactor;
            myDeltaTemp0 = nodeList[A[i][j].index].extra->Heat->oldT - nodeList[i].extra->Heat->oldT;
            sumFlow0 += A[i][j].val * (1. - myParameters.heatWeightingFactor) * myDeltaTemp0;
            A[i][j++].val *= -(myParameters.heatWeightingFactor);
        }

        /*! sum of diagonal elements */
        avgh = arithmeticMean(nodeList[i].oldH, myH) - nodeList[i].z;
        A[i][0].val = SoilHeatCapacity(i, avgh, nodeList[i].extra->Heat->T) * nodeList[i].volume_area / timeStep + sum;

        /*! b vector (constant terms) */
        b[i] = C[i] * nodeList[i].extra->Heat->oldT / timeStep - heatCapacityVar / timeStep + nodeList[i].extra->Heat->Qh + invariantFlux[i] + sumFlow0;

        // preconditioning
        if (A[i][0].val > 0)
        {
            b[i] /= A[i][0].val;
            j = 1;
            while ((j < myStructure.maxNrColumns) && (A[i][j].index != NOLINK))
                A[i][j++].val /= A[i][0].val;
        }
    }

    // avoiding oscillations (Courant number)
    if (CourantHeat > 1.0)
        if (timeStep > myParameters.delta_t_min)
        {
            halveTimeStep();
            setForcedHalvedTime(true);
            return (false);
        }

    GaussSeidelRelaxation(0, myParameters.ResidualTolerance, PROCESS_HEAT);

    for (i = 1; i < myStructure.nrNodes; i++)
        nodeList[i].extra->Heat->T = X[i];

    // avoiding oscillations (maximum temperature change allowed)
    /*double maxDeltaT = computeMaximumDeltaT();
    double ratioDeltaT = maxDeltaT / myParameters.heatMaximumDeltaT;
    if (maxDeltaT > myParameters.heatMaximumDeltaT)
    {
        while (timeStep / myParameters.current_delta_t < ratioDeltaT && myParameters.current_delta_t > myParameters.delta_t_min)
            {
                halveTimeStep();
                setForcedHalvedTime(true);
            }

        if (myParameters.current_delta_t > myParameters.delta_t_min) return false;
    }*/

    heatBalance(timeStep, timeStepWater);
    updateBalanceHeat();

    updateHeatFluxes(timeStep, timeStepWater);

	// save old temperatures
    for (long n = 1; n < myStructure.nrNodes; n++)
        nodeList[n].extra->Heat->oldT = nodeList[n].extra->Heat->T;

    return (true);
}
