/*!
    \name Solar Radiation
    \copyright 2011 Gabriele Antolini, Fausto Tomei
    \note  This library uses G_calc_solar_position() by Markus Neteler

    This library is part of CRITERIA3D.
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

    \authors
    Gabriele Antolini gantolini@arpae.it
    Fausto Tomei ftomei@arpae.it
*/

#include "commonConstants.h"
#include "transmissivity.h"
#include "radiationSettings.h"
#include "solarRadiation.h"
#include <math.h>


float computePointTransmissivitySamani(float tmin, float tmax, float samaniCoeff)
{
    if (samaniCoeff != NODATA && tmin != NODATA && tmax != NODATA)
        if (tmin <= tmax)
            return samaniCoeff * sqrt(tmax - tmin);
        else
            return false;
    else
        return NODATA;
}


bool computeTransmissivity(Crit3DRadiationSettings* mySettings, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, int intervalWidth,
                          Crit3DTime myTime, const gis::Crit3DRasterGrid& myDEM)
{
    if (nrMeteoPoints <= 0) return false;

    int hourlyFraction = meteoPoints[0].hourlyFraction;
    int deltaSeconds = 3600 / hourlyFraction;

    int semiInterval = (intervalWidth - 1)/2;
    float semiIntervalSeconds = float(semiInterval * deltaSeconds);
    int myIndex;
    Crit3DTime myTimeIni =  myTime.addSeconds(-semiIntervalSeconds);
    Crit3DTime myTimeFin =  myTime.addSeconds(semiIntervalSeconds);
    Crit3DTime myCurrentTime;
    int myCounter = 0;
    float transmissivity;

    gis::Crit3DPoint myPoint;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        float myRad = meteoPoints[i].getMeteoPointValueH(myTime.date, myTime.getHour(),
                                                         myTime.getMinutes(), globalIrradiance);

        if (myRad != NODATA)
        {
            myIndex = 0;
            float* obsRadVector = new float[unsigned(intervalWidth)];
            myCurrentTime = myTimeIni;
            while (myCurrentTime <= myTimeFin)
            {
                obsRadVector[myIndex] = meteoPoints[i].getMeteoPointValueH(myCurrentTime.date, myCurrentTime.getHour(),
                                                                       myCurrentTime.getMinutes(), globalIrradiance);
                myCurrentTime = myCurrentTime.addSeconds(deltaSeconds);
                myIndex++;
            }

            myPoint.utm.x = meteoPoints[i].point.utm.x;
            myPoint.utm.y = meteoPoints[i].point.utm.y;
            myPoint.z = meteoPoints[i].point.z;

            transmissivity = radiation::computePointTransmissivity(mySettings, myPoint, myTime, obsRadVector,
                                                                   intervalWidth, deltaSeconds, myDEM);

            meteoPoints[i].setMeteoPointValueH(myTime.date, myTime.getHour(), myTime.getMinutes(),
                                               atmTransmissivity, transmissivity);

            if (transmissivity != NODATA) myCounter++;
            delete [] obsRadVector;
        }
    }

    return (myCounter > 0);
}


bool computeTransmissivityFromTRange(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DTime currentTime)
{
    if (nrMeteoPoints <= 0) return false;

    int hourlyFraction = meteoPoints[0].hourlyFraction;
    int deltaSeconds = 3600 / hourlyFraction;

    int counter = 0;
    Crit3DTime timeTmp;
    float temp, tmin, tmax;
    float transmissivity;
    int i;

    for (i = 0; i < nrMeteoPoints; i++)
    {
        tmin = NODATA;
        tmax = NODATA;
        int mySeconds = deltaSeconds;

        timeTmp.date = currentTime.date;
        timeTmp.time = 0;

        while (mySeconds < DAY_SECONDS)
        {
            timeTmp = timeTmp.addSeconds(deltaSeconds);
            temp = meteoPoints[i].getMeteoPointValueH(timeTmp.date, timeTmp.getHour(), timeTmp.getMinutes(), airTemperature);
            if (temp != NODATA)
            {
                if (tmin == NODATA || temp < tmin)
                    tmin = temp;
                if (tmax == NODATA || temp > tmax)
                    tmax = temp;
            }
            mySeconds += deltaSeconds;
        }

        if (tmin != NODATA && tmax != NODATA)
        {
            transmissivity = computePointTransmissivitySamani(tmin, tmax, float(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT));
            if (transmissivity != NODATA)
            {
                // save transmissivity data in memory (all daily values)
                int nrDailyValues = hourlyFraction * 24;
                for (int h = 1; h <= nrDailyValues; h++)
                    meteoPoints[i].setMeteoPointValueH(currentTime.date, h, 0, atmTransmissivity, transmissivity);

                //midnight
                meteoPoints[i].setMeteoPointValueH(currentTime.date.addDays(1),
                                       0, 0, atmTransmissivity, transmissivity);
                counter++;
            }
        }
    }

    return (counter > 0);
}
