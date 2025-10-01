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

    float getMinHeight(const std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode);
    float getMaxHeight(const std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode);
    float getZmin(const std::vector <Crit3DInterpolationDataPoint> &myPoints);
    float getZmax(const std::vector <Crit3DInterpolationDataPoint> &myPoints);
    float getProxyMinValue(const std::vector <Crit3DInterpolationDataPoint> &myPoints, unsigned pos);
    float getProxyMaxValue(const std::vector <Crit3DInterpolationDataPoint> &myPoints, unsigned pos);

    bool preInterpolation(std::vector<Crit3DInterpolationDataPoint> &myPoints,
                          Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings *meteoSettings,
                          Crit3DClimateParameters* climateParameters, Crit3DMeteoPoint *meteoPoints,
                          int nrMeteoPoints, meteoVariable myVar, Crit3DTime myTime, std::string &errorStr);

    void topographicDistanceOptimize(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                                     const std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                                     Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings* meteoSettings);

    double topographicDistanceInternalFunction(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                                               const std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                                               Crit3DInterpolationSettings &interpolationSettings,
                                               Crit3DMeteoSettings* meteoSettings, double khFloat);

    double goldenSectionSearch(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                               const std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                               Crit3DInterpolationSettings &interpolationSettings,
                               Crit3DMeteoSettings* meteoSettings, double a, double b);

    bool krigingEstimateVariogram(float *myDist, float *mySemiVar,int sizeMyVar, int nrMyPoints,float myMaxDistance,
                                  double *mySill, double *myNugget, double *myRange, double *mySlope,
                                  TkrigingMode *myMode, int nrPointData);
    bool krigLinearPrep(double *mySlope, double *myNugget, int nrPointData);

    void clearInterpolationPoints();
    bool checkPrecipitationZero(const std::vector<Crit3DInterpolationDataPoint> &myPoints, float precThreshold, int &nrValidData);

    bool neighbourhoodVariability(meteoVariable myVar, std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                                  const Crit3DInterpolationSettings &interpolationSettings, float x, float y, float z, int maxNrPoints,
                                  float &devSt, float &avgDeltaZ, float &minDistance);

    float interpolate(const std::vector<Crit3DInterpolationDataPoint>& myPoints, Crit3DInterpolationSettings &interpolationSettings,
                      Crit3DMeteoSettings *meteoSettings, meteoVariable variable, float x, float y, float z,
                      const std::vector<double> &proxyValues, bool excludeSupplemental);

    float inverseDistanceWeighted(const std::vector<Crit3DInterpolationDataPoint> &pointList, const std::vector<float>& distances);

    float shepardIdw(const std::vector <Crit3DInterpolationDataPoint>& myPoints, std::vector <float> &distances,
                     Crit3DInterpolationSettings &interpolationSettings, float x, float y);

    float shepardSearchNeighbour(const std::vector <Crit3DInterpolationDataPoint>& inputPoints,
                                 const std::vector <float>& inputDistances,
                                 Crit3DInterpolationSettings &interpolationSettings,
                                 std::vector <Crit3DInterpolationDataPoint>& outputPoints,
                                 std::vector <float>& outputDistances);

    float modifiedShepardIdw(const std::vector <Crit3DInterpolationDataPoint> &myPoints, std::vector<float> &myDistances,
                             Crit3DInterpolationSettings &interpolationSettings, float radius, float x, float y);

    bool getProxyValuesXY(float x, float y, Crit3DInterpolationSettings &interpolationSettings, std::vector<double> &myValues);
    bool getSignificantProxyValuesXY(float x, float y, Crit3DInterpolationSettings &interpolationSettings, std::vector<double> &myValues);

    bool getMultipleDetrendingValues(Crit3DInterpolationSettings &interpolationSettings,
                                     const std::vector<double> &allProxyValues, std::vector<double> &activeProxyValues,
                                     std::vector<std::function<double (double, std::vector<double> &)> > &myFunc,
                                     std::vector<std::vector<double> > &myParameters);

    float retrend(meteoVariable myVar, const std::vector<double>& proxyValues, Crit3DInterpolationSettings &interpolationSettings);

    void detrending(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DProxyCombination inCombination,
                    Crit3DInterpolationSettings &interpolationSettings, Crit3DClimateParameters* climateParameters,
                    meteoVariable myVar, Crit3DTime myTime);

    bool macroAreaDetrending(const Crit3DMacroArea &myArea, meteoVariable myVar,
                             Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings* meteoSettings,
                             Crit3DMeteoPoint *meteoPoints, const std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                             std::vector <Crit3DInterpolationDataPoint> &subsetInterpolationPoints, int elevationPos);

    bool multipleDetrendingMain(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                Crit3DInterpolationSettings &interpolationSettings, meteoVariable myVar, std::string &errorStr);

    bool multipleDetrendingOtherProxiesFitting(int elevationPos, std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                            Crit3DInterpolationSettings &interpolationSettings,
                                            meteoVariable myVar, std::string &errorStr);

    bool multipleDetrendingElevationFitting(int elevationPos, std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                        Crit3DInterpolationSettings &interpolationSettings, meteoVariable myVar,
                                        std::string &errorStr, bool isWeighted);

    void detrendingElevation(int elevationPos, std::vector <Crit3DInterpolationDataPoint> &myPoints,
                             Crit3DInterpolationSettings &interpolationSettings);

    void detrendingOtherProxies(int elevationPos, std::vector<Crit3DInterpolationDataPoint> &myPoints,
                                Crit3DInterpolationSettings &interpolationSettings);
	
    bool glocalDetrendingFitting(const std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                 Crit3DInterpolationSettings &interpolationSettings,  meteoVariable myVar, std::string& errorStr);

    bool getUseDetrendingVar(meteoVariable myVar);
    bool isThermal(meteoVariable myVar);
    bool getUseTdVar(meteoVariable myVar);

    unsigned sortPointsByDistance(unsigned maxNrPoints,
                                  const std::vector<Crit3DInterpolationDataPoint> &pointList, const std::vector<float> &distances,
                                  std::vector<Crit3DInterpolationDataPoint> &outputPointList, std::vector<float> &outputDistances);

    float getFirstIntervalHeightValue(std::vector <Crit3DInterpolationDataPoint> &myPoints, bool useLapseRateCode);

    bool regressionGeneric(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings &interpolationSettings,
                           int proxyPos, bool isZeroIntercept);

    bool regressionOrography(std::vector <Crit3DInterpolationDataPoint> &myPoints,
                             Crit3DProxyCombination myCombination, Crit3DInterpolationSettings &interpolationSettings,
                             Crit3DClimateParameters* climateParameters,
                             Crit3DTime myTime, meteoVariable myVar, int orogProxyPos);

    void optimalDetrending(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                           std::vector<Crit3DInterpolationDataPoint> &outInterpolationPoints,
                           Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings* meteoSettings,
                           Crit3DClimateParameters* climateParameters, const Crit3DTime &myTime);

    std::vector<float> computeDistances(meteoVariable myVar, const std::vector <Crit3DInterpolationDataPoint> &myPoints,
                                        const Crit3DInterpolationSettings &interpolationSettings,
                                        float x, float y, float z, bool excludeSupplemental);

    bool localSelection(const std::vector <Crit3DInterpolationDataPoint> &inputPoints,
                        std::vector <Crit3DInterpolationDataPoint> &selectedPoints,
                        float x, float y, Crit3DInterpolationSettings &interpolationSettings, bool excludeSupplemental);

    bool proxyValidity(std::vector <Crit3DInterpolationDataPoint> &myPoints, int proxyPos,
                       float stdDevThreshold, double &avg, double &stdDev);

    bool proxyValidityWeighted(std::vector <Crit3DInterpolationDataPoint> &myPoints, int proxyPos,
                               float stdDevThreshold);

    bool setMultipleDetrendingHeightTemperatureRange(Crit3DInterpolationSettings &interpolationSettings);

    void calculateFirstGuessCombinations(Crit3DProxy* myProxy);

    bool setFittingParameters_elevation(int elevationPos, Crit3DInterpolationSettings &interpolationSettings,
                                         std::vector<std::function<double(double, std::vector<double>&)>>& myFunc,
                                         std::vector <std::vector<double>> &paramMin, std::vector <std::vector<double>> &paramMax,
                                         std::vector <std::vector<double>> &paramDelta, std::vector <std::vector<double>> &paramFirstGuess,
                                         std::vector<double> &stepSize, int numSteps,
                                         std::string &errorStr);

    bool setFittingParameters_otherProxies(int elevationPos, Crit3DInterpolationSettings &interpolationSettings,
                                        std::vector<std::function<double(double, std::vector<double>&)>>& myFunc,
                                        std::vector <std::vector<double>> &paramMin, std::vector <std::vector<double>> &paramMax,
                                        std::vector <std::vector<double>> &paramDelta, std::vector <std::vector<double>> &paramFirstGuess,
                                        std::string &errorStr);

#endif // INTERPOLATION_H
