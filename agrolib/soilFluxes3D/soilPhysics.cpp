/*!
    \name soilPhysics.cpp
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
#include "../mathFunctions/physics.h"
#include "header/types.h"
#include "header/soilPhysics.h"
#include "header/solver.h"
#include "header/heat.h"
#include "header/extra.h"

     /*!
     * \brief Computes volumetric water content from current degree of saturation
     * \param myIndex
     * \return result
     */
	double theta_from_Se (unsigned long myIndex)
	{
		return ((myNode[myIndex].Se * (myNode[myIndex].Soil->Theta_s - myNode[myIndex].Soil->Theta_r)) + myNode[myIndex].Soil->Theta_r);
	}

    /*!
     * \brief Computes volumetric water content from degree of saturation
     * \param Se
     * \param myIndex
     * \return result
     */
	double theta_from_Se (double Se, unsigned long myIndex)
	{
		return ((Se * (myNode[myIndex].Soil->Theta_s - myNode[myIndex].Soil->Theta_r)) + myNode[myIndex].Soil->Theta_r);
	}

    /*!
     * \brief Computes volumetric water content from water potential (with sign)
     * \param signPsi
     * \param index
     * \return result
     */
    double theta_from_sign_Psi (double signPsi, unsigned long index)
	{
        if (myNode[index].isSurface) return 1.;

        if (signPsi >= 0.0)
        {
            /*! saturated */
            return myNode[index].Soil->Theta_s;
        }
		else
        {
            double Se = computeSefromPsi(fabs(signPsi),myNode[index].Soil);
            return theta_from_Se(Se, index);
        }
	}


    /*!
     * \brief Computes degree of saturation from volumetric water content
     * \param myIndex
     * \param theta
     * \return result
     */
	double Se_from_theta (unsigned long myIndex, double theta)
	{
        /*! check range */
		if (theta >= myNode[myIndex].Soil->Theta_s) return(1.);
		else if (theta <= myNode[myIndex].Soil->Theta_r) return(0.);
		else return ((theta - myNode[myIndex].Soil->Theta_r) / (myNode[myIndex].Soil->Theta_s - myNode[myIndex].Soil->Theta_r));
	}

    /*!
     * \brief Computes degree of saturation from matric potential (Van Genutchen) Se = [1+(alfa*|h|)^n]^-m
     * \param myPsi
     * \param mySoil
     * \return result Se
     */
    double computeSefromPsi(double myPsi, Tsoil *mySoil)
	{
		double Se = NODATA;

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
			if (myPsi <=  mySoil->VG_he)
				Se = 1.;
			else
				{
                Se = pow(1. + pow(mySoil->VG_alpha * myPsi, mySoil->VG_n), - mySoil->VG_m);
				Se *= (1. / mySoil->VG_Sc);
				}
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
            Se = pow(1. + pow(mySoil->VG_alpha * myPsi, mySoil->VG_n), - mySoil->VG_m);

		return Se;
	}

    /*!
     * \brief Computes current degree of saturation
     * \param myIndex
     * \return result
     */
    double computeSe(unsigned long myIndex)
    {
        /*! saturated */
        if (myNode[myIndex].H >= myNode[myIndex].z) return 1.;

        double psi = fabs(myNode[myIndex].H - myNode[myIndex].z);   /*!< [m] */

        return computeSefromPsi(psi, myNode[myIndex].Soil);
    }


    /*!
     * \brief Computes hydraulic conductivity [m/sec]  (Mualem)
     * K(Se) = Ksat * Se^(L) * {1-[1-Se^(1/m)]^m}^2
     * WARNING: very low values are possible (es: 10^12)
     * \param Se
     * \param mySoil Tsoil pointer
     * \return result
     */
    double computeWaterConductivity(double Se, Tsoil *mySoil)
	{
		if (Se >= 1.) return(mySoil->K_sat );

		double myTmp = NODATA;
        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
		{
			double myNumerator = 1. - pow(1. - pow(Se*mySoil->VG_Sc, 1./mySoil->VG_m), mySoil->VG_m);
			myTmp = myNumerator / (1. - pow(1. - pow(mySoil->VG_Sc, 1./mySoil->VG_m), mySoil->VG_m));
		}
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
			myTmp = 1. - pow(1. - pow(Se, 1./mySoil->VG_m), mySoil->VG_m);

		return (mySoil->K_sat * pow(Se, mySoil->Mualem_L) * pow(myTmp , 2.));
	}


    /*!
     * \brief Computes hydraulic conductivity [m/sec]  (Mualem)
     * K(Se) = Ksat * Se^(L) * {1-[1-Se^(1/m)]^m}^2
     * WARNING: very low values are possible (es: 10^12)
     * \param Ksat
     * \param Se
     * \param VG_Sc
	 * \param VG_n
     * \param VG_m
     * \param Mualem_L
     * \return result
     */
    double compute_K_Mualem(double Ksat, double Se, double VG_Sc, double VG_m, double Mualem_L)
	{
		if (Se >= 1.) return(Ksat);
		double temp= NODATA;

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
        {
            double num = 1. - pow(1. - pow(Se*VG_Sc, 1./VG_m), VG_m);
            temp = num / (1. - pow(1. - pow(VG_Sc, 1./VG_m), VG_m));
        }
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
        {
			temp = 1. - pow(1. - pow(Se, 1./VG_m), VG_m);
        }

		return (Ksat * pow(Se, Mualem_L) * pow(temp , 2.));
	}


    /*!
     * \brief Computes current soil water total (liquid + vapor) conductivity [m sec^-1]
     * \param myIndex
     * \return result
     */
    double computeK(unsigned long myIndex)
    {
        double k = compute_K_Mualem(myNode[myIndex].Soil->K_sat, myNode[myIndex].Se,
                                myNode[myIndex].Soil->VG_Sc, myNode[myIndex].Soil->VG_m,
                                myNode[myIndex].Soil->Mualem_L);

        // vapor isothermal flow
        if (myStructure.computeHeat && myStructure.computeHeatVapor)
        {
            double avgT = getTMean(myIndex);
            double kv = IsothermalVaporConductivity(myIndex, myNode[myIndex].H - myNode[myIndex].z, avgT);
            // from kg s m-3 to m s-1
            kv *= (GRAVITY / WATER_DENSITY);

            k += kv;
        }

        return k;
    }


    /*!
     * \brief Computes Water Potential from degree of saturation
     * \param myIndex
     * \return result
     */
    double psi_from_Se(unsigned long myIndex)
	{
		double m = myNode[myIndex].Soil->VG_m;
		double temp = NODATA;

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
				temp = pow(1./ (myNode[myIndex].Se * myNode[myIndex].Soil->VG_Sc) , 1./ m ) - 1.;
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
				temp = pow(1./ myNode[myIndex].Se, 1./ m ) - 1.;

        return((1./ myNode[myIndex].Soil->VG_alpha) * pow(temp, 1./ myNode[myIndex].Soil->VG_n));
	}

    /*!
     * \brief [m-1] dThetaV/dH
     * \param myIndex
     * \return derivative of vapor volumetric content with respect to H
     */
    double dThetav_dH(unsigned long i, double temperature, double dTheta_dH)
    {
        double h = myNode[i].H - myNode[i].z;
        double hr = SoilRelativeHumidity(h, temperature);
        double satVapPressure = SaturationVaporPressure(temperature - ZEROCELSIUS);
        double satVapConc = VaporConcentrationFromPressure(satVapPressure, temperature);
        double theta = theta_from_sign_Psi(h, i);
        double dThetav_dPsi = (satVapConc * hr / WATER_DENSITY) *
                ((myNode[i].Soil->Theta_s - theta) * MH2O / (R_GAS * temperature) - dTheta_dH / GRAVITY);
        return dThetav_dPsi * GRAVITY;
    }

    /*!
     * \brief [m-1] dTheta/dH  (Van Genutchen)
     * dTheta/dH = dSe/dH (Theta_s-Theta_r)
     * dSe/dH = -sgn(H-z) alfa n m [1+(alfa|(H-z)|)^n]^(-m-1) (alfa|(H-z)|)^n-1
     * \param myIndex
     * \return derivative of water volumetric content with respect to H
     */
	double dTheta_dH(unsigned long myIndex)
	 {
     double alfa = myNode[myIndex].Soil->VG_alpha;
	 double n    = myNode[myIndex].Soil->VG_n;
	 double m    = myNode[myIndex].Soil->VG_m;
	 double dSe_dH;

     double psi = fabs(MINVALUE(myNode[myIndex].H - myNode[myIndex].z, 0.));
     double psiPrevious = fabs(MINVALUE(myNode[myIndex].oldH - myNode[myIndex].z, 0.));

    if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
    {
        if ((psi <= myNode[myIndex].Soil->VG_he) && (psiPrevious <= myNode[myIndex].Soil->VG_he)) return 0.;
    }
    if (myParameters.waterRetentionCurve == VANGENUCHTEN)
    {
        if ((psi == 0.) && (psiPrevious == 0.)) return 0.;
    }

	 if (psi == psiPrevious)
    {
        dSe_dH = alfa * n * m * pow(1. + pow(alfa * psi, n), -(m + 1.)) * pow(alfa * psi, n - 1.);
        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
                dSe_dH *= (1. / myNode[myIndex].Soil->VG_Sc);
    }
	 else
    {
        double theta = computeSefromPsi(psi, myNode[myIndex].Soil);
        double thetaPrevious = computeSefromPsi(psiPrevious, myNode[myIndex].Soil);
        double delta_H = myNode[myIndex].H - myNode[myIndex].oldH;
        dSe_dH = fabs((theta - thetaPrevious) / delta_H);
    }

	 return (dSe_dH * (myNode[myIndex].Soil->Theta_s - myNode[myIndex].Soil->Theta_r));
	 }

    double getThetaMean(long i)
	{
        double myHMean = getHMean(i);

		if (myNode[i].isSurface)
		{
            double mySurfaceWater = MAXVALUE(myHMean - myNode[i].z, 0.);		//[m]
            return (MINVALUE(mySurfaceWater / 0.01, 1.));
		}
		else
		{
            /*! sub-surface */
            return (getTheta(i, myHMean));
		}
	}

    double getTheta(long i, double H)
    {
        double psi = H - myNode[i].z;
        return (theta_from_sign_Psi(psi, i));
    }

    double getTMean(long i)
    {
        if (myStructure.computeHeat && myNode[i].extra->Heat != nullptr)
            return arithmeticMean(myNode[i].extra->Heat->oldT, myNode[i].extra->Heat->T);
        else
            return NODATA;
    }

    double getHMean(long i)
    {
        // is there any efficient way to compute a geometric mean of H?
        return arithmeticMean(myNode[i].oldH, myNode[i].H);
    }

    double getPsiMean(long i)
	{
        double Psi;
        double meanH = getHMean(i);
        Psi = MINVALUE(0., (meanH - myNode[i].z));
        return Psi;
	}

    /*!
     * \brief estimate particle density
     * \param fractionOrganicMatter
     * \return particle density (Mg m-3)
     * [Driessen, 1986]
     */
    double ParticleDensity(double fractionOrganicMatter)
    {
        if (fractionOrganicMatter == NODATA)
            fractionOrganicMatter = 0.02;

        return 1.0 / ((1.0 - fractionOrganicMatter) / QUARTZ_DENSITY + fractionOrganicMatter / 1.43);
    }

    /*!
     * \brief estimate bulk density
     * \param i
     * \return bulk density (Mg m-3)
     */
    double estimateBulkDensity(long i)
    {
        double particleDensity;
        double totalPorosity;

        particleDensity = ParticleDensity(myNode[i].Soil->organicMatter);

        totalPorosity = myNode[i].Soil->Theta_s;

        return (1. - totalPorosity) * particleDensity;
    }
