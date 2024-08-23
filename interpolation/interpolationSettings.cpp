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

#include <string>

#include "interpolationSettings.h"
#include "basicMath.h"
#include "commonConstants.h"

bool Crit3DInterpolationSettings::getPrecipitationAllZero() const
{
    return precipitationAllZero;
}

void Crit3DInterpolationSettings::setPrecipitationAllZero(bool value)
{
    precipitationAllZero = value;
}

float Crit3DInterpolationSettings::getMinRegressionR2() const
{
    return minRegressionR2;
}

void Crit3DInterpolationSettings::setMinRegressionR2(float value)
{
    minRegressionR2 = value;
}

bool Crit3DInterpolationSettings::getUseLapseRateCode() const
{
    return useLapseRateCode;
}

void Crit3DInterpolationSettings::setUseLapseRateCode(bool value)
{
    useLapseRateCode = value;
}

bool Crit3DInterpolationSettings::getUseBestDetrending() const
{
    return useBestDetrending;
}

void Crit3DInterpolationSettings::setUseBestDetrending(bool value)
{
    useBestDetrending = value;
}

aggregationMethod Crit3DInterpolationSettings::getMeteoGridAggrMethod() const
{
    return meteoGridAggrMethod;
}

void Crit3DInterpolationSettings::setMeteoGridAggrMethod(const aggregationMethod &value)
{
    meteoGridAggrMethod = value;
}

int Crit3DInterpolationSettings::getIndexPointCV() const
{
    return indexPointCV;
}

void Crit3DInterpolationSettings::setIndexPointCV(int value)
{
    indexPointCV = value;
}

gis::Crit3DRasterGrid *Crit3DInterpolationSettings::getCurrentDEM() const
{
    return currentDEM;
}

void Crit3DInterpolationSettings::setCurrentDEM(gis::Crit3DRasterGrid *value)
{
    currentDEM = value;
}

int Crit3DInterpolationSettings::getTopoDist_maxKh() const
{
    return topoDist_maxKh;
}

void Crit3DInterpolationSettings::setTopoDist_maxKh(int value)
{
    topoDist_maxKh = value;
}

int Crit3DInterpolationSettings::getTopoDist_Kh() const
{
    return topoDist_Kh;
}

void Crit3DInterpolationSettings::setTopoDist_Kh(int value)
{
    topoDist_Kh = value;
}

Crit3DProxyCombination Crit3DInterpolationSettings::getOptimalCombination() const
{
    return optimalCombination;
}

void Crit3DInterpolationSettings::setOptimalCombination(const Crit3DProxyCombination &value)
{
    optimalCombination = value;
}

Crit3DProxyCombination Crit3DInterpolationSettings::getSelectedCombination() const
{
    return selectedCombination;
}

void Crit3DInterpolationSettings::setSelectedCombination(const Crit3DProxyCombination &value)
{
    selectedCombination = value;
}

void Crit3DInterpolationSettings::setActiveSelectedCombination(unsigned int index, bool isActive)
{
    selectedCombination.setProxyActive(index, isActive);
    selectedCombination.setProxySignificant(index, false);
}

unsigned Crit3DInterpolationSettings::getIndexHeight() const
{
    return indexHeight;
}

void Crit3DInterpolationSettings::setIndexHeight(unsigned value)
{
    indexHeight = value;
}

void Crit3DInterpolationSettings::setCurrentCombination(Crit3DProxyCombination value)
{
    currentCombination = value;
}

Crit3DProxyCombination Crit3DInterpolationSettings::getCurrentCombination() const
{
    return currentCombination;
}


std::vector<Crit3DProxy> Crit3DInterpolationSettings::getCurrentProxy() const
{
    return currentProxy;
}

void Crit3DInterpolationSettings::setCurrentProxy(const std::vector<Crit3DProxy> &value)
{
    currentProxy = value;
}

bool Crit3DInterpolationSettings::getUseInterpolatedTForRH() const
{
    return useInterpolatedTForRH;
}

void Crit3DInterpolationSettings::setUseInterpolatedTForRH(bool value)
{
    useInterpolatedTForRH = value;
}

bool Crit3DInterpolationSettings::getProxyLoaded() const
{
    return proxyLoaded;
}

void Crit3DInterpolationSettings::setProxyLoaded(bool value)
{
    proxyLoaded = value;
}

const std::vector<float> &Crit3DInterpolationSettings::getKh_series() const
{
    return Kh_series;
}

void Crit3DInterpolationSettings::setKh_series(const std::vector<float> &newKh_series)
{
    Kh_series = newKh_series;
}

const std::vector<float> &Crit3DInterpolationSettings::getKh_error_series() const
{
    return Kh_error_series;
}

void Crit3DInterpolationSettings::addToKhSeries(float kh, float error)
{
    Kh_series.push_back(kh);
    Kh_error_series.push_back(error);
}

void Crit3DInterpolationSettings::initializeKhSeries()
{
    Kh_series.clear();
    Kh_error_series.clear();
}

void Crit3DInterpolationSettings::setKh_error_series(const std::vector<float> &newKh_error_series)
{
    Kh_error_series = newKh_error_series;
}

bool Crit3DInterpolationSettings::getMeteoGridUpscaleFromDem() const
{
    return meteoGridUpscaleFromDem;
}

void Crit3DInterpolationSettings::setMeteoGridUpscaleFromDem(bool newMeteoGridUpscaleFromDem)
{
    meteoGridUpscaleFromDem = newMeteoGridUpscaleFromDem;
}

bool Crit3DInterpolationSettings::getUseMultipleDetrending() const
{
    return useMultipleDetrending;
}

void Crit3DInterpolationSettings::setUseMultipleDetrending(bool newUseMultipleDetrending)
{
    useMultipleDetrending = newUseMultipleDetrending;
}

float Crit3DInterpolationSettings::getPointsBoundingBoxArea() const
{
    return pointsBoundingBoxArea;
}

void Crit3DInterpolationSettings::setPointsBoundingBoxArea(float newPointsBoundingBoxArea)
{
    pointsBoundingBoxArea = newPointsBoundingBoxArea;
}

float Crit3DInterpolationSettings::getLocalRadius() const
{
    return localRadius;
}

void Crit3DInterpolationSettings::setLocalRadius(float newLocalRadius)
{
    localRadius = newLocalRadius;
}

int Crit3DInterpolationSettings::getMinPointsLocalDetrending() const
{
    return minPointsLocalDetrending;
}

void Crit3DInterpolationSettings::setMinPointsLocalDetrending(int newMinPointsLocalDetrending)
{
    minPointsLocalDetrending = newMinPointsLocalDetrending;
}

std::vector<std::vector<double> > Crit3DInterpolationSettings::getFittingParameters() const
{
    return fittingParameters;
}

std::vector<double> Crit3DInterpolationSettings::getProxyFittingParameters(int tempIndex)
{
    if (tempIndex < fittingParameters.size())
        return fittingParameters[tempIndex];
    else {
        fittingParameters.resize(tempIndex + 1);
        return fittingParameters[tempIndex];
    }
}

void Crit3DInterpolationSettings::setFittingParameters(const std::vector<std::vector<double> > &newFittingParameters)
{
    fittingParameters = newFittingParameters;
}

void Crit3DInterpolationSettings::setSingleFittingParameters(std::vector<double> &newFittingParameters, int paramIndex)
{
    if (fittingParameters.size() <= paramIndex)
        fittingParameters.resize(paramIndex+1);
    fittingParameters[paramIndex] = newFittingParameters;
}

void Crit3DInterpolationSettings::addFittingParameters(const std::vector<std::vector<double> > &newFittingParameters)
{
    for (size_t i = 0; i < newFittingParameters.size(); ++i) {
        fittingParameters.push_back(newFittingParameters[i]);
    }
}

std::vector<std::function<double (double, std::vector<double> &)>> Crit3DInterpolationSettings::getFittingFunction() const
{
    return fittingFunction;
}

void Crit3DInterpolationSettings::setFittingFunction(const std::vector<std::function<double (double, std::vector<double> &)> > &newFittingFunction)
{
    fittingFunction = newFittingFunction;
}

void Crit3DInterpolationSettings::setSingleFittingFunction(const std::function<double (double, std::vector<double> &)> &newFittingFunction, unsigned int index)
{
    if (fittingFunction.size() <= index)
        fittingFunction.resize(index + 1);
    fittingFunction[index] = newFittingFunction;

}

TFittingFunction Crit3DInterpolationSettings::getChosenElevationFunction()
{
    int elPos = NODATA;
    for (int i = 0; i < getProxyNr(); i++)
        if (getProxyPragaName(getProxy(i)->getName()) == proxyHeight)
            elPos = i;

    if (elPos != NODATA)
        return getProxy(elPos)->getFittingFunctionName();
    else
        return noFunction;
}

void Crit3DInterpolationSettings::setChosenElevationFunction(TFittingFunction chosenFunction)
{
    const double H0_MIN = -200; //height of inversion point (double piecewise) or first inversion point (triple piecewise)
    const double H0_MAX = 5000;
    const double DELTA_MIN = 300; //height difference between inversion points (for triple piecewise only)
    const double DELTA_MAX = 1000;
    const double SLOPE_MIN = 0.002; //ascending slope
    const double SLOPE_MAX = 0.007;
    const double INVSLOPE_MIN = -0.01; //inversion slope
    const double INVSLOPE_MAX = -0.0015;

    if (getUseMultipleDetrending()) clearFitting();

    int elPos = NODATA;
    for (int i = 0; i < getProxyNr(); i++)
        if (getProxyPragaName(getProxy(i)->getName()) == proxyHeight)
            elPos = i;

    double MIN_T = -20;
    double MAX_T = 40;

    if (!getMinMaxTemperature().empty())
    {
        MIN_T = getMinMaxTemperature()[0];
        MAX_T = getMinMaxTemperature()[1];
    }

    if (elPos != NODATA)
    {
        if (chosenFunction == getProxy(elPos)->getFittingFunctionName() && !getProxy(elPos)->getFittingParametersRange().empty())
        {
            std::vector tempParam = getProxy(elPos)->getFittingParametersRange();

            if (chosenFunction == piecewiseTwo)
            {
                tempParam[1] = MIN_T-2;
                tempParam[5] = MAX_T+2;
            }
            else if (chosenFunction == piecewiseThree)
            {
                tempParam[1] = MIN_T-2;
                tempParam[6] = MAX_T+2;
            }
            else if (chosenFunction == piecewiseThreeFree)
            {
                tempParam[1] = MIN_T-2;
                tempParam[7] = MAX_T+2;
            }

            getProxy(elPos)->setFittingParametersRange(tempParam);
        }
        else if (chosenFunction != getProxy(elPos)->getFittingFunctionName() || getProxy(elPos)->getFittingParametersRange().empty())
        {
            if (chosenFunction == piecewiseTwo)
                getProxy(elPos)->setFittingParametersRange({H0_MIN, MIN_T-2, SLOPE_MIN, INVSLOPE_MIN,
                                                            H0_MAX, MAX_T+2, SLOPE_MAX, INVSLOPE_MAX});
            else if (chosenFunction == piecewiseThree)
                getProxy(elPos)->setFittingParametersRange({H0_MIN, MIN_T-2, DELTA_MIN, SLOPE_MIN, INVSLOPE_MIN,
                                                            H0_MAX, MAX_T+2, DELTA_MAX, SLOPE_MAX, INVSLOPE_MAX});
            else if (chosenFunction == piecewiseThreeFree)
                getProxy(elPos)->setFittingParametersRange({H0_MIN, MIN_T-2, DELTA_MIN, SLOPE_MIN, INVSLOPE_MIN, INVSLOPE_MIN,
                                                            H0_MAX, MAX_T+2, DELTA_MAX, SLOPE_MAX, INVSLOPE_MAX, INVSLOPE_MAX});

            getProxy(elPos)->setFittingFunctionName(chosenFunction);
        }
    }
}

void Crit3DInterpolationSettings::setPointsRange(double min, double max)
{
    pointsRange.clear();
    pointsRange.push_back(min);
    pointsRange.push_back(max);
}

std::vector<double> Crit3DInterpolationSettings::getMinMaxTemperature()
{
    return pointsRange;
}

void Crit3DInterpolationSettings::clearFitting()
{
    fittingFunction.clear();
    fittingParameters.clear();
}

bool Crit3DInterpolationSettings::getProxiesComplete() const
{
    return proxiesComplete;
}

void Crit3DInterpolationSettings::setProxiesComplete(bool newProxiesComplete)
{
    proxiesComplete = newProxiesComplete;
}

Crit3DInterpolationSettings::Crit3DInterpolationSettings()
{
    initialize();
}

void Crit3DInterpolationSettings::initializeProxy()
{
    proxyLoaded = false;

    currentProxy.clear();
    selectedCombination.clear();
    optimalCombination.clear();

    indexHeight = unsigned(NODATA);
}

void Crit3DInterpolationSettings::initialize()
{
    currentDEM = nullptr;
    interpolationMethod = idw;
    useThermalInversion = true;
    useTD = false;
    useLocalDetrending = false;
    topoDist_maxKh = 128;
    useDewPoint = true;
    useInterpolatedTForRH = true;
    useMultipleDetrending = false;
    useBestDetrending = false;
    useLapseRateCode = false;
    minRegressionR2 = float(PEARSONSTANDARDTHRESHOLD);
    meteoGridAggrMethod = aggrAverage;
    meteoGridUpscaleFromDem = true;
    indexHeight = unsigned(NODATA);

    fittingFunction.clear();
    fittingParameters.clear();

    isKrigingReady = false;
    precipitationAllZero = false;
    maxHeightInversion = 1000.;
    indexPointCV = NODATA;
    minPointsLocalDetrending = 20;
    proxiesComplete = true;

    Kh_series.clear();
    Kh_error_series.clear();

    initializeProxy();
}

std::string getKeyStringInterpolationMethod(TInterpolationMethod value)
{
    std::map<std::string, TInterpolationMethod>::const_iterator it;
    std::string key = "";

    for (it = interpolationMethodNames.begin(); it != interpolationMethodNames.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string getKeyStringElevationFunction(TFittingFunction value)
{
    std::map<std::string, TFittingFunction>::const_iterator it;
    std::string key = "";

    for (it = fittingFunctionNames.begin(); it != fittingFunctionNames.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

TInterpolationMethod Crit3DInterpolationSettings::getInterpolationMethod()
{ return interpolationMethod;}

bool Crit3DInterpolationSettings::getUseTD()
{ return (useTD && !useLocalDetrending);}

bool Crit3DInterpolationSettings::getUseLocalDetrending()
{ return useLocalDetrending;}

float Crit3DInterpolationSettings::getMaxHeightInversion()
{ return maxHeightInversion;}

void Crit3DInterpolationSettings::setInterpolationMethod(TInterpolationMethod myValue)
{ interpolationMethod = myValue;}

void Crit3DInterpolationSettings::setUseThermalInversion(bool myValue)
{
    useThermalInversion = myValue;
    selectedCombination.setUseThermalInversion(myValue);
}

void Crit3DInterpolationSettings::setUseTD(bool myValue)
{ useTD = myValue;}

void Crit3DInterpolationSettings::setUseLocalDetrending(bool myValue)
{ useLocalDetrending = myValue;}

void Crit3DInterpolationSettings::setUseDewPoint(bool myValue)
{ useDewPoint = myValue;}

bool Crit3DInterpolationSettings::getUseThermalInversion()
{ return (useThermalInversion);}

bool Crit3DInterpolationSettings::getUseDewPoint()
{ return (useDewPoint);}

size_t Crit3DInterpolationSettings::getProxyNr()
{ return currentProxy.size();}

Crit3DProxy* Crit3DInterpolationSettings::getProxy(unsigned pos)
{ return &(currentProxy[pos]);}

int Crit3DInterpolationSettings::getProxyPosFromName(TProxyVar name)
{
    for (int i = 0; i < getProxyNr(); i++)
    {
        if (getProxyPragaName(getProxyName(i)) == name)
            return i;
    }

    return NODATA;
}

std::string Crit3DProxy::getName() const
{
    return name;
}

void Crit3DProxy::setName(const std::string &value)
{
    name = value;
}

gis::Crit3DRasterGrid *Crit3DProxy::getGrid() const
{
    return grid;
}

void Crit3DProxy::setGrid(gis::Crit3DRasterGrid *value)
{
    grid = value;
}

std::string Crit3DProxy::getGridName() const
{
    return gridName;
}

TProxyVar getProxyPragaName(std::string name_)
{
    if (ProxyVarNames.find(name_) == ProxyVarNames.end())
        return TProxyVar::noProxy;
    else
        return ProxyVarNames.at(name_);
}

void Crit3DProxy::setGridName(const std::string &value)
{
    gridName = value;
}

bool Crit3DProxy::getIsSignificant() const
{
    return isSignificant;
}

void Crit3DProxy::setIsSignificant(bool value)
{
    isSignificant = value;
}

bool Crit3DProxy::getForQualityControl() const
{
    return forQualityControl;
}

void Crit3DProxy::setForQualityControl(bool value)
{
    forQualityControl = value;
}

std::string Crit3DProxy::getProxyTable() const
{
    return proxyTable;
}

void Crit3DProxy::setProxyTable(const std::string &value)
{
    proxyTable = value;
}

std::string Crit3DProxy::getProxyField() const
{
    return proxyField;
}

void Crit3DProxy::setProxyField(const std::string &value)
{
    proxyField = value;
}

float Crit3DProxy::getLapseRateT0() const
{
    return lapseRateT0;
}

void Crit3DProxy::setLapseRateT0(float newLapseRateT0)
{
    lapseRateT0 = newLapseRateT0;
}

float Crit3DProxy::getLapseRateT1() const
{
    return lapseRateT1;
}

void Crit3DProxy::setLapseRateT1(float newLapseRateT1)
{
    lapseRateT1 = newLapseRateT1;
}

float Crit3DProxy::getRegressionIntercept() const
{
    return regressionIntercept;
}

void Crit3DProxy::setRegressionIntercept(float newRegressionIntercept)
{
    regressionIntercept = newRegressionIntercept;
}

float Crit3DProxy::getAvg() const
{
    return avg;
}

void Crit3DProxy::setAvg(float newAvg)
{
    avg = newAvg;
}

float Crit3DProxy::getStdDev() const
{
    return stdDev;
}

void Crit3DProxy::setStdDev(float newStdDev)
{
    stdDev = newStdDev;
}

float Crit3DProxy::getStdDevThreshold() const
{
    return stdDevThreshold;
}

void Crit3DProxy::setStdDevThreshold(float newStdDevThreshold)
{
    stdDevThreshold = newStdDevThreshold;
}

std::vector<double> Crit3DProxy::getFittingParametersRange() const
{
    return fittingParametersRange;
}

void Crit3DProxy::setFittingParametersRange(const std::vector<double> &newFittingParametersRange)
{
    fittingParametersRange.clear();
    fittingParametersRange = newFittingParametersRange;
}

void Crit3DProxy::setFittingFunctionName(TFittingFunction functionName)
{
    fittingFunctionName = functionName;
    return;
}

TFittingFunction Crit3DProxy::getFittingFunctionName()
{
    return fittingFunctionName;
}


Crit3DProxy::Crit3DProxy()
{
    name = "";
    gridName = "";
    grid = new gis::Crit3DRasterGrid();
    isSignificant = false;
    forQualityControl = false;

    regressionR2 = NODATA;
    regressionSlope = NODATA;
    lapseRateH0 = NODATA;
    lapseRateH1 = NODATA;
    lapseRateT0 = NODATA;
    lapseRateT1 = NODATA;
    inversionLapseRate = NODATA;
    inversionIsSignificative = false;
    fittingParametersRange.clear();

    avg = NODATA;
    stdDev = NODATA;
    stdDevThreshold = NODATA;

    proxyTable = "";
    proxyField = "";
}

float Crit3DProxy::getLapseRateH1() const
{
    return lapseRateH1;
}

void Crit3DProxy::setLapseRateH1(float value)
{
    lapseRateH1 = value;
}

float Crit3DProxy::getLapseRateH0() const
{
    return lapseRateH0;
}

void Crit3DProxy::setLapseRateH0(float value)
{
    lapseRateH0 = value;
}

float Crit3DProxy::getInversionLapseRate() const
{
    return inversionLapseRate;
}

void Crit3DProxy::setInversionLapseRate(float value)
{
    inversionLapseRate = value;
}

bool Crit3DProxy::getInversionIsSignificative() const
{
    return inversionIsSignificative;
}

void Crit3DProxy::setInversionIsSignificative(bool value)
{
    inversionIsSignificative = value;
}

void Crit3DProxy::setRegressionR2(float myValue)
{ regressionR2 = myValue;}

float Crit3DProxy::getRegressionR2()
{ return regressionR2;}

void Crit3DProxy::setRegressionSlope(float myValue)
{ regressionSlope = myValue;}

float Crit3DProxy::getRegressionSlope()
{ return regressionSlope;}

double Crit3DProxy::getValue(unsigned int pos, std::vector <double> proxyValues)
{
    if (pos < proxyValues.size())
        return proxyValues[pos];
    else
        return NODATA;
}

void Crit3DProxy::initializeOrography()
{
    setLapseRateH0(0.);
    setLapseRateH1(NODATA);
    setLapseRateT0(NODATA);
    setLapseRateT1(NODATA);
    setInversionLapseRate(NODATA);
    setRegressionSlope(NODATA);
    setRegressionIntercept(NODATA);
    setRegressionR2(NODATA);
    setInversionIsSignificative(false);

    return;
}

void Crit3DInterpolationSettings::addProxy(Crit3DProxy myProxy, bool isActive_)
{
    currentProxy.push_back(myProxy);

    if (getProxyPragaName(myProxy.getName()) == proxyHeight)
        setIndexHeight(int(currentProxy.size())-1);

    selectedCombination.addProxyActive(isActive_);
    selectedCombination.addProxySignificant(false);
    optimalCombination.addProxyActive(isActive_);
    optimalCombination.addProxySignificant(false);
}

std::string Crit3DInterpolationSettings::getProxyName(unsigned pos)
{ return currentProxy[pos].getName();}

double Crit3DInterpolationSettings::getProxyValue(unsigned pos, std::vector <double> proxyValues)
{
    if (pos < currentProxy.size())
        return currentProxy[pos].getValue(pos, proxyValues);
    else
        return NODATA;
}


Crit3DProxyCombination::Crit3DProxyCombination()
{
    clear();
}


void Crit3DProxyCombination::clear()
{
    _isActiveList.clear();
    _isSignificantList.clear();
    _useThermalInversion = false;
}

void Crit3DProxyCombination::resetCombination(unsigned int size)
{
    _isActiveList.resize(size);
    _isSignificantList.resize(size);
    for (unsigned int i = 0; i < size; i++)
    {
        setProxyActive(i, false);
        setProxySignificant(i, false);
    }
    _useThermalInversion = false;
}

unsigned int Crit3DProxyCombination::getActiveProxySize()
{
    unsigned int size = 0;
    for (unsigned int i = 0; i < getProxySize(); i++)
        if (isProxyActive(i)) size++;

    return size;
}

void Crit3DProxyCombination::setAllActiveToFalse()
{
    for (unsigned int i = 0; i < _isActiveList.size(); i++)
        setProxyActive(i, false);

    return;
}

void Crit3DProxyCombination::setAllSignificantToFalse()
{
    for (unsigned int i = 0; i < _isActiveList.size(); i++)
        setProxySignificant(i, false);

    return;
}


bool Crit3DInterpolationSettings::getCombination(int combinationInteger, Crit3DProxyCombination &outCombination)
{
    outCombination = selectedCombination;
    std::string binaryString = decimal_to_binary(unsigned(combinationInteger), int(getProxyNr()+1));

    int indexHeight = getIndexHeight();

    // avoid combinations with inversion (last index) and without orography
    if (combinationInteger % 2 == 1)
    {
        if (indexHeight == NODATA || binaryString[indexHeight] == '0')
            return false;
    }

    for (unsigned int i=0; i < binaryString.length()-1; i++)
    {
        outCombination.setProxyActive(i, (binaryString[i] == '1' && selectedCombination.isProxyActive(i)) );
    }

    outCombination.setUseThermalInversion(binaryString[binaryString.length()-1] == '1' && selectedCombination.getUseThermalInversion());

    return true;
}

