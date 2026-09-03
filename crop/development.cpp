/*!
    \file development.cpp

    \abstract
    Leaf area index development functions

    \authors
    Antonio Volta       avolta@arpae.it
    Fausto Tomei        ftomei@arpae.it
    Gabriele Antolini   gantolini@arpe.it

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


#include "commonConstants.h"
#include "development.h"
#include "crop.h"

#include <algorithm>
#include <math.h>


namespace leafDevelopment
{
    /*!
     * \brief getTheoreticalLAIGrowth   return theoritical (not stressed) Leaf Area Index from degree days
     * \param degreeDays    [DD]
     * \param a             [-] LAI shape form factor a
     * \param b             [DD-1] LAI shape form factor b
     * \param laiMIN        [m2 m-2]
     * \param laiMAX        [m2 m-2]
     * \return LAI          [m2 m-2]
     */
    double getTheoreticalLAIGrowth(double degreeDays, double a, double b,double laiMIN,double laiMAX)
    {
        double lai = laiMIN + (laiMAX-laiMIN)/(1 + exp(a+b*degreeDays));
        return lai;
    }

    /*!
     * \brief getDDfromLAIGrowth
     * \param lai           [m2 m-2] current Leaf Area Index
     * \param a             [-] LAI shape form factor a
     * \param b             [DD-1] LAI shape form factor b
     * \param laiMIN        [m2 m-2]
     * \param laiMAX        [m2 m-2]
     * \return degreeDays
     */
    double getDDfromLAIGrowth(double lai, double a, double b,double laiMIN,double laiMAX)
    {
        double degreeDays = (1/b)*(log((laiMAX - lai)/(lai - laiMIN)) - a);
        return degreeDays;
    }


    // Antonio - new LAI algorithm
    double getNewLai(double fractionTranspirableSoilWater, double previousLai, double a, double b,double laiMIN,double laiMAX,double growthDD,double emergenceDD,double currentDD,double thermalUnits,bool *isSenescence,double* actualLaiMax)
    {
        if (currentDD <= emergenceDD) return laiMIN;

        double newLai;
        if (!(*isSenescence))
        {
            if (previousLai < laiMIN + 0.3) // to evaluate 0.3 - no stress in early growth
            {
                newLai = getTheoreticalLAIGrowth(currentDD - emergenceDD,a,b,laiMIN,laiMAX);
                *actualLaiMax = newLai;
            }
            else
            {
                double incrementalRatio,DD;
                DD = getDDfromLAIGrowth(previousLai,a,b,laiMIN,laiMAX); // DD is ficticious
                incrementalRatio = (getTheoreticalLAIGrowth(DD+5,a,b,laiMIN,laiMAX)-getTheoreticalLAIGrowth(DD-5,a,b,laiMIN,laiMAX))/(10.);
                incrementalRatio *= getLaiStressCoefficient(fractionTranspirableSoilWater);
                if (currentDD < growthDD)
                newLai = std::min(previousLai + thermalUnits*incrementalRatio,laiMAX);
                else
                {
                    *isSenescence = true;
                    newLai = std::min(previousLai + (thermalUnits-currentDD+growthDD)*incrementalRatio,laiMAX);
                }
                *actualLaiMax = newLai;
            }
        }
        else // senescence of LAI to be done
        {
            newLai = *actualLaiMax;
        }
        return newLai;
    }


    double getLaiStressCoefficient(double avgFractionTranspirableSoilWater)
    {
        double stress;
        stress = 1.0/(1.0 + 25.9*exp(-17.3 * avgFractionTranspirableSoilWater)); // from Bindi et al. 1995
        return stress;
    }


    double getLAISenescence(double LaiMin, double LAIStartSenescence, int daysFromStartSenescence)
    {
        double a, b;
        int LENGTH_SENESCENCE = 30;

        if (daysFromStartSenescence > LENGTH_SENESCENCE)
            return LaiMin;

        a = log(std::max(LAIStartSenescence, 0.1));
        b = (log(std::max(LaiMin, 0.01)) - a) / LENGTH_SENESCENCE;

        return exp(a + b * daysFromStartSenescence);
    }


    // degreeDays [°C] after emergence
    double getLAICriteria(const Crit3DCrop* crop, double degreeDays)
    {
        if (degreeDays <= crop->degreeDaysIncrease)
        {
            // LAI increasing curve
            const double exponent = crop->LAIcurve_a + crop->LAIcurve_b * degreeDays;

            return crop->LAImin + (crop->LAImax - crop->LAImin) / (1.0 + exp(exponent));
        }
        else
        {
            // LAI decreasing curve
            constexpr double n4 = 4.0;
            const double c4 = (crop->type == TREE) ? 15.0 : 9.0;

            // real maximum LAI
            const double maxExponent = crop->LAIcurve_a + crop->LAIcurve_b * crop->degreeDaysIncrease;
            const double maxLai = crop->LAImin + (crop->LAImax - crop->LAImin) / (1.0 + exp(maxExponent));

            const double ddSenescence = (degreeDays - crop->degreeDaysIncrease) / std::max(crop->degreeDaysDecrease, 1.0);

            return crop->LAImin + (maxLai - crop->LAImin) / (1.0 + pow(10.0 * ddSenescence / c4, n4));
        }
    }
}

