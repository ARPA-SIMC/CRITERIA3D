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
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include <stdio.h>
#include <math.h>

#include "commonConstants.h"
#include "physics.h"
#include "header/types.h"
#include "header/soilPhysics.h"
#include "header/solver.h"
#include "header/heat.h"
#include "header/extra.h"

     /*!
     * \brief Computes volumetric water content from current degree of saturation
     * \return result
     */
    double theta_from_Se (unsigned long index)
	{
        return ((nodeList[index].Se * (nodeList[index].Soil->Theta_s - nodeList[index].Soil->Theta_r)) + nodeList[index].Soil->Theta_r);
	}

    /*!
     * \brief Computes volumetric water content from degree of saturation
     * \param Se degree of saturation [-]
     * \return result
     */
    double theta_from_Se (double Se, unsigned long index)
	{
        return ((Se * (nodeList[index].Soil->Theta_s - nodeList[index].Soil->Theta_r)) + nodeList[index].Soil->Theta_r);
	}

    /*!
     * \brief Computes volumetric water content from water potential (with sign)
     * \param signPsi   water potential with sign [m]
     * \return volumetric water content [m3 m-3]
     */
    double theta_from_sign_Psi (double signPsi, unsigned long index)
	{
        if (nodeList[index].isSurface) return 1.;

        if (signPsi >= 0.0)
        {
            // saturated
            return nodeList[index].Soil->Theta_s;
        }
		else
        {
            double Se = computeSefromPsi_unsat(fabs(signPsi),nodeList[index].Soil);
            return theta_from_Se(Se, index);
        }
	}


    /*!
     * \brief Computes degree of saturation from volumetric water content
     * \param theta  volumetric water content [m3 m-3]
     * \return result
     */
    double Se_from_theta (unsigned long index, double theta)
	{
        /*! check range */
        if (theta >= nodeList[index].Soil->Theta_s) return(1.);
        else if (theta <= nodeList[index].Soil->Theta_r) return(0.);
        else return ((theta - nodeList[index].Soil->Theta_r) / (nodeList[index].Soil->Theta_s - nodeList[index].Soil->Theta_r));
	}

    /*!
     * \brief Computes degree of saturation from matric potential  Se = [1+(alfa*|h|)^n]^-m
     * \param psi    [m] water potential (absolute value)
     * \param mySoil
     * \return degree of saturation [-]
     */
    double computeSefromPsi_unsat(double psi, Tsoil *mySoil)
	{
		double Se = NODATA;

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
        {
            if (psi <=  mySoil->VG_he)
            {
                // saturated
                Se = 1.;
            }
			else
            {
                Se = pow(1. + pow(mySoil->VG_alpha * psi, mySoil->VG_n), - mySoil->VG_m);
                Se *= (1. / mySoil->VG_Sc);
            }
        }
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
        {
            Se = pow(1. + pow(mySoil->VG_alpha * psi, mySoil->VG_n), - mySoil->VG_m);
        }

		return Se;
	}

    /*!
     * \brief Computes current degree of saturation
     * \return degree of saturation [-]
     */
    double computeSe(unsigned long index)
    {
        if (nodeList[index].H >= nodeList[index].z)
        {
            // saturated
            return 1.;
        }
        else
        {
            // unsaturated
            double psi = fabs(nodeList[index].H - nodeList[index].z);   /*!< [m] */
            return computeSefromPsi_unsat(psi, nodeList[index].Soil);
        }
    }


    /*!
     * \brief Computes hydraulic conductivity, passing a soil structure
     * Mualem equation:
     * K(Se) = Ksat * Se^(L) * {1-[1-Se^(1/m)]^m}^2
     * WARNING: very low values are possible (es: 10^12)
     * \param Se        degree of saturation [-]
     * \param mySoil    Tsoil pointer
     * \return hydraulic conductivity [m/sec]
     */
    double computeWaterConductivity(double Se, Tsoil *mySoil)
	{
		if (Se >= 1.) return(mySoil->K_sat );

        double tmp = NODATA;
        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
		{
			double myNumerator = 1. - pow(1. - pow(Se*mySoil->VG_Sc, 1./mySoil->VG_m), mySoil->VG_m);
            tmp = myNumerator / (1. - pow(1. - pow(mySoil->VG_Sc, 1./mySoil->VG_m), mySoil->VG_m));
		}
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
        {
            tmp = 1. - pow(1. - pow(Se, 1./mySoil->VG_m), mySoil->VG_m);
        }

        return (mySoil->K_sat * pow(Se, mySoil->Mualem_L) * pow(tmp, 2.));
	}


    /*!
     * \brief Computes hydraulic conductivity, passing soil parameters
     * Mualem equation:
     * K(Se) = Ksat * Se^(L) * {1-[1-Se^(1/m)]^m}^2
     * WARNING: very low values are possible (es: 10^12)
     * \param Se        degree of saturation [-]
     * \return hydraulic conductivity [m/sec]
     */
    double compute_K_Mualem(double Ksat, double Se, double VG_Sc, double VG_m, double Mualem_L)
	{
		if (Se >= 1.) return(Ksat);
        double tmp= NODATA;

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
        {
            double num = 1. - pow(1. - pow(Se*VG_Sc, 1./VG_m), VG_m);
            tmp = num / (1. - pow(1. - pow(VG_Sc, 1./VG_m), VG_m));
        }
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
        {
            tmp = 1. - pow(1. - pow(Se, 1./VG_m), VG_m);
        }

        return (Ksat * pow(Se, Mualem_L) * pow(tmp , 2.));
	}


    /*!
     * \brief Computes current soil water total (liquid + vapor) conductivity [m sec^-1]
     * \param index
     * \return result
     */
    double computeK(unsigned long index)
    {
        double k = compute_K_Mualem(nodeList[index].Soil->K_sat, nodeList[index].Se,
                                nodeList[index].Soil->VG_Sc, nodeList[index].Soil->VG_m,
                                nodeList[index].Soil->Mualem_L);

        // vapor isothermal flow
        if (myStructure.computeHeat && myStructure.computeHeatVapor)
        {
            double avgT = getTMean(index);
            double kv = IsothermalVaporConductivity(index, nodeList[index].H - nodeList[index].z, avgT);
            // from kg s m-3 to m s-1
            kv *= (GRAVITY / WATER_DENSITY);

            k += kv;
        }

        return k;
    }


    /*!
     * \brief Computes current water potential from degree of saturation
     * \param index
     * \return water potential [m]
     */
    double psi_from_Se(unsigned long index)
	{
        double m = nodeList[index].Soil->VG_m;
		double temp = NODATA;

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
                temp = pow(1./ (nodeList[index].Se * nodeList[index].Soil->VG_Sc) , 1./ m ) - 1.;
        else if (myParameters.waterRetentionCurve == VANGENUCHTEN)
                temp = pow(1./ nodeList[index].Se, 1./ m ) - 1.;

        return((1./ nodeList[index].Soil->VG_alpha) * pow(temp, 1./ nodeList[index].Soil->VG_n));
	}

    /*!
     * \brief [m-1] dThetaV/dH
     * \param index
     * \return derivative of vapor volumetric content with respect to H
     */
    double dThetav_dH(unsigned long i, double temperature, double dTheta_dH)
    {
        double h = nodeList[i].H - nodeList[i].z;
        double hr = SoilRelativeHumidity(h, temperature);
        double satVapPressure = saturationVaporPressure(temperature - ZEROCELSIUS);
        double satVapConc = vaporConcentrationFromPressure(satVapPressure, temperature);
        double theta = theta_from_sign_Psi(h, i);
        double dThetav_dPsi = (satVapConc * hr / WATER_DENSITY) *
                ((nodeList[i].Soil->Theta_s - theta) * MH2O / (R_GAS * temperature) - dTheta_dH / GRAVITY);
        return dThetav_dPsi * GRAVITY;
    }


    /*!
     * \brief [m-1] dTheta/dH  (Van Genutchen)
     * dTheta/dH = dSe/dH (Theta_s-Theta_r)
     * dSe/dH = -sgn(H-z) alfa n m [1+(alfa|(H-z)|)^n]^(-m-1) (alfa|(H-z)|)^n-1
     * \return derivative of water volumetric content with respect to H
     */
    double dTheta_dH(unsigned long index)
    {
        double alfa = nodeList[index].Soil->VG_alpha;
        double n    = nodeList[index].Soil->VG_n;
        double m    = nodeList[index].Soil->VG_m;

        double currentPsi = fabs(std::min(nodeList[index].H - nodeList[index].z, 0.));
        double previousPsi = fabs(std::min(nodeList[index].oldH - nodeList[index].z, 0.));

        if (myParameters.waterRetentionCurve == VANGENUCHTEN)
        {
            // saturated
            if ((currentPsi == 0.) && (previousPsi == 0.))
                return 0.;
        }

        if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
        {
            // saturated
            if ((currentPsi <= nodeList[index].Soil->VG_he) && (previousPsi <= nodeList[index].Soil->VG_he))
                return 0.;
        }

        double dSe_dH;

        if (currentPsi == previousPsi)
        {
            dSe_dH = alfa * n * m * pow(1. + pow(alfa * currentPsi, n), -(m + 1.)) * pow(alfa * currentPsi, n - 1.);
            if (myParameters.waterRetentionCurve == MODIFIEDVANGENUCHTEN)
            {
                dSe_dH *= (1. / nodeList[index].Soil->VG_Sc);
            }
        }
        else
        {
            double theta = computeSefromPsi_unsat(currentPsi, nodeList[index].Soil);
            double previousTheta = computeSefromPsi_unsat(previousPsi, nodeList[index].Soil);
            dSe_dH = fabs((theta - previousTheta) / (nodeList[index].H - nodeList[index].oldH));
        }

        return dSe_dH * (nodeList[index].Soil->Theta_s - nodeList[index].Soil->Theta_r);
    }


    double getThetaMean(long i)
	{
        double myHMean = getHMean(i);

        if (nodeList[i].isSurface)
		{
            double mySurfaceWater = std::max(myHMean - nodeList[i].z, 0.);		// [m]
            return (std::min(mySurfaceWater / 0.01, 1.));
		}
		else
		{
            /*! sub-surface */
            return (getTheta(i, myHMean));
		}
	}

    double getTheta(long i, double H)
    {
        double psi = H - nodeList[i].z;
        return (theta_from_sign_Psi(psi, i));
    }

    double getTMean(long i)
    {
        if (myStructure.computeHeat && nodeList[i].extra->Heat != nullptr)
            return arithmeticMean(nodeList[i].extra->Heat->oldT, nodeList[i].extra->Heat->T);
        else
            return NODATA;
    }

    double getHMean(long i)
    {
        if ( (nodeList[i].oldH > 0. && nodeList[i].H > 0.)
            || (nodeList[i].oldH < 0. && nodeList[i].H < 0.))
        {
            return logarithmicMean(nodeList[i].oldH, nodeList[i].H);
        }
        else
        {
            return (nodeList[i].oldH + nodeList[i].H) * 0.5;
        }
    }


    double getPsiMean(long i)
	{
        double Psi;
        double meanH = getHMean(i);
        Psi = std::min(0., (meanH - nodeList[i].z));
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

        particleDensity = ParticleDensity(nodeList[i].Soil->organicMatter);

        totalPorosity = nodeList[i].Soil->Theta_s;

        return (1. - totalPorosity) * particleDensity;
    }
