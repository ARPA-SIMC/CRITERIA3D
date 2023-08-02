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

#include "crit3dDate.h"
#include "interpolationSettings.h"
#include "interpolation.h"
#include "basicMath.h"
#include "commonConstants.h"
#include "cmath"


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

float Crit3DInterpolationSettings::getShepardInitialRadius() const
{
    return shepardInitialRadius;
}

void Crit3DInterpolationSettings::setShepardInitialRadius(float value)
{
    shepardInitialRadius = value;
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

void Crit3DInterpolationSettings::setValueSelectedCombination(unsigned int index, bool isActive)
{
    selectedCombination.setValue(index, isActive);
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
    useDynamicLapserate = false;
    topoDist_maxKh = 128;
    useDewPoint = true;
    useInterpolatedTForRH = true;
    useBestDetrending = false;
    useLapseRateCode = false;
    minRegressionR2 = float(PEARSONSTANDARDTHRESHOLD);
    meteoGridAggrMethod = aggrAverage;
    meteoGridUpscaleFromDem = true;
    indexHeight = unsigned(NODATA);

    isKrigingReady = false;
    precipitationAllZero = false;
    maxHeightInversion = 1000.;
    shepardInitialRadius = NODATA;
    indexPointCV = NODATA;

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

TInterpolationMethod Crit3DInterpolationSettings::getInterpolationMethod()
{ return interpolationMethod;}

bool Crit3DInterpolationSettings::getUseTD()
{ return useTD;}

bool Crit3DInterpolationSettings::getUseDynamicLapserate()
{ return useDynamicLapserate;}

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

void Crit3DInterpolationSettings::setUseDynamicLapserate(bool myValue)
{ useDynamicLapserate = myValue;}

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

float Crit3DProxy::getValue(unsigned int pos, std::vector <float> proxyValues)
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

    if (getProxyPragaName(myProxy.getName()) == height)
        setIndexHeight(int(currentProxy.size())-1);

    selectedCombination.addValue(isActive_);
    optimalCombination.addValue(isActive_);
}

std::string Crit3DInterpolationSettings::getProxyName(unsigned pos)
{ return currentProxy[pos].getName();}

float Crit3DInterpolationSettings::getProxyValue(unsigned pos, std::vector <float> proxyValues)
{
    if (pos < currentProxy.size())
        return currentProxy[pos].getValue(pos, proxyValues);
    else
        return NODATA;
}

void Crit3DInterpolationSettings::computeShepardInitialRadius(float area, int nrPoints)
{
    setShepardInitialRadius(sqrt((SHEPARD_AVG_NRPOINTS * area) / (float(PI) * nrPoints)));
}

std::deque<bool> Crit3DProxyCombination::getIsActive() const
{
    return isActive;
}

void Crit3DProxyCombination::setIsActive(const std::deque<bool> &value)
{
    isActive = value;
}

Crit3DProxyCombination::Crit3DProxyCombination()
{
    setUseThermalInversion(false);
}

void Crit3DProxyCombination::clear()
{
    isActive.clear();
}

void Crit3DProxyCombination::addValue(bool isActive_)
{
    isActive.push_back(isActive_);
}

void Crit3DProxyCombination::setValue(unsigned index, bool isActive_)
{
    isActive[index] = isActive_;
}

bool Crit3DProxyCombination::getValue(unsigned index)
{
    return isActive[index];
}

bool Crit3DProxyCombination::getUseThermalInversion() const
{
    return useThermalInversion;
}

void Crit3DProxyCombination::setUseThermalInversion(bool value)
{
    useThermalInversion = value;
}

bool Crit3DInterpolationSettings::getCombination(int combinationInteger, Crit3DProxyCombination &outCombination)
{
    outCombination = selectedCombination;
    std::string binaryString = decimal_to_binary(unsigned(combinationInteger), int(getProxyNr()+1));

    int indexHeight = getIndexHeight();

    // avoid combinations with inversion (last index) and without orography
    if (combinationInteger % 2 == 1)
        if (indexHeight == NODATA || binaryString[indexHeight] == '0')
            return false;

    for (unsigned int i=0; i < binaryString.length()-1; i++)
        outCombination.setValue(i, binaryString[i] == '1' && selectedCombination.getValue(i));

    outCombination.setUseThermalInversion(binaryString[binaryString.length()-1] == '1' && selectedCombination.getUseThermalInversion());

    return true;
}

