/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

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

#ifndef INTERPOLATION_H
#define INTERPOLATION_H

    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif
    #ifndef INTERPOLATIONSETTINGS_H
        #include "interpolationSettings.h"
    #endif
    #ifndef INTERPOLATIONPOINT_H
        #include "interpolationPoint.h"
    #endif
    #ifndef VECTOR_H
        #include <vector>
    #endif

    float getMinHeight(const std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode);
    float getMaxHeight(const std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode);
    float getZmin(const std::vector <Crit3DInterpolationDataPoint> &myPoints);
    float getZmax(const std::vector <Crit3DInterpolationDataPoint> &myPoints);
    float getProxyMinValue(std::vector <Crit3DInterpolationDataPoint> &myPoints, unsigned pos);
    float getProxyMaxValue(std::vector <Crit3DInterpolationDataPoint> &myPoints, unsigned pos);

    bool preInterpolation(std::vector<Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings *mySettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters* myClimate,
                          Crit3DMeteoPoint *myMeteoPoints, int nrMeteoPoints, meteoVariable myVar, Crit3DTime myTime);

    bool krigingEstimateVariogram(float *myDist, float *mySemiVar,int sizeMyVar, int nrMyPoints,float myMaxDistance, double *mySill, double *myNugget, double *myRange, double *mySlope, TkrigingMode *myMode, int nrPointData);
    bool krigLinearPrep(double *mySlope, double *myNugget, int nrPointData);

    void clearInterpolationPoints();

    bool neighbourhoodVariability(meteoVariable myVar, std::vector<Crit3DInterpolationDataPoint> &myInterpolationPoints, Crit3DInterpolationSettings *mySettings, float x, float y, float z, int nMax,
                                  float* devSt, float* avgDeltaZ, float* minDistance);

    float interpolate(std::vector<Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings *mySettings, Crit3DMeteoSettings *meteoSettings, meteoVariable myVar, float myX, float myY, float myZ, std::vector<float> myProxyValues, bool excludeSupplemental);
    void getProxyValuesXY(float x, float y, Crit3DInterpolationSettings* mySettings, std::vector<float> &myValues);

    void detrending(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                    Crit3DProxyCombination myCombination, Crit3DInterpolationSettings *mySettings, Crit3DClimateParameters *myClimate,
                    meteoVariable myVar, Crit3DTime myTime);
    void multipleDetrending(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                 Crit3DProxyCombination myCombination, Crit3DInterpolationSettings* mySettings, meteoVariable myVar);

    bool getUseDetrendingVar(meteoVariable myVar);
    bool isThermal(meteoVariable myVar);
    bool getUseTdVar(meteoVariable myVar);

    float getFirstIntervalHeightValue(std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode);

    bool regressionGeneric(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings,
                           int proxyPos, bool isZeroIntercept);

    bool regressionOrography(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                             Crit3DProxyCombination myCombination, Crit3DInterpolationSettings* mySettings, Crit3DClimateParameters* myClimate,
                             Crit3DTime myTime, meteoVariable myVar, int orogProxyPos);

    void optimalDetrending(meteoVariable myVar, Crit3DMeteoPoint* &myMeteoPoints, int nrMeteoPoints,
                           std::vector <Crit3DInterpolationDataPoint> &outInterpolationPoints,
                           Crit3DInterpolationSettings* mySettings, Crit3DMeteoSettings* meteoSettings, Crit3DClimateParameters* myClimate,
                           const Crit3DTime &myTime);

    bool dynamicSelection(std::vector <Crit3DInterpolationDataPoint> &inputPoints,
                          std::vector <Crit3DInterpolationDataPoint> &selectedPoints,
                          float x, float y, const Crit3DInterpolationSettings& mySettings, bool excludeSupplemental);

    namespace stat_openai
    {
        std::vector<double> multipleLinearRegression(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    }

#endif // INTERPOLATION_H
