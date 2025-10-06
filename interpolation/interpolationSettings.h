#ifndef INTERPOLATIONSETTINGS_H
#define INTERPOLATIONSETTINGS_H

#include <functional>
#ifndef INTERPOLATIONCONSTS_H
        #include "interpolationConstants.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include "statistics.h"


    std::string getKeyStringInterpolationMethod(TInterpolationMethod value);
    std::string getKeyStringElevationFunction(TFittingFunction value);
    TProxyVar getProxyPragaName(std::string name_);

    class Crit3DProxy
    {
    private:
        std::string name;
        std::string gridName;
        gis::Crit3DRasterGrid* grid;
        std::string proxyTable;
        std::string proxyField;
        bool forQualityControl;

        float regressionR2;
        float regressionSlope;
        float regressionIntercept;
        TFittingFunction fittingFunctionName;
        std::vector <double> fittingParametersRange;
        std::vector <int> fittingFirstGuess;
        std::vector <std::vector <double>> firstGuessCombinations;

        float avg;
        float stdDev;
        float stdDevThreshold;

        //orography
        float lapseRateH1;
        float lapseRateH0;
        float lapseRateT0;
        float lapseRateT1;
        float inversionLapseRate;
        bool inversionIsSignificative;

    public:
        Crit3DProxy();

        void initializeOrography();

        std::string getName() const;
        void setName(const std::string &value);
        gis::Crit3DRasterGrid *getGrid() const;
        void setGrid(gis::Crit3DRasterGrid *value);
        std::string getGridName() const;
        void setGridName(const std::string &value);
        void setRegressionR2(float myValue);
        float getRegressionR2();
        void setRegressionSlope(float myValue);
        float getRegressionSlope();
        double getValue(unsigned int pos, std::vector<double> proxyValues) const;
        float getLapseRateH1() const;
        void setLapseRateH1(float value);
        float getLapseRateH0() const;
        void setLapseRateH0(float value);
        float getInversionLapseRate() const;
        void setInversionLapseRate(float value);
        bool getInversionIsSignificative() const;
        void setInversionIsSignificative(bool value);
        bool getForQualityControl() const;
        void setForQualityControl(bool value);
        std::string getProxyTable() const;
        void setProxyTable(const std::string &value);
        std::string getProxyField() const;
        void setProxyField(const std::string &value);
        std::vector<gis::Crit3DRasterGrid *> getGridSeries() const;
        void setGridSeries(const std::vector<gis::Crit3DRasterGrid *> &value);
        float getLapseRateT0() const;
        void setLapseRateT0(float newLapseRateT0);
        float getLapseRateT1() const;
        void setLapseRateT1(float newLapseRateT1);
        float getRegressionIntercept() const;
        void setRegressionIntercept(float newRegressionIntercept);
        float getAvg() const;
        void setAvg(float newAvg);
        float getStdDev() const;
        void setStdDev(float newStdDev);
        float getStdDevThreshold() const;
        void setStdDevThreshold(float newStdDevThreshold);
        std::vector<double> getFittingParametersRange() const;
        void setFittingParametersRange(const std::vector<double> &newFittingParametersRange);
        std::vector<double> getFittingParametersMax() const;
        std::vector<double> getFittingParametersMin() const;
        TFittingFunction getFittingFunctionName();
        void setFittingFunctionName(TFittingFunction functionName);
        std::vector<int> getFittingFirstGuess() const;
        void setFittingFirstGuess(const std::vector<int> &newFittingFirstGuess);
        std::vector <std::vector<double>> getFirstGuessCombinations() const;
        void setFirstGuessCombinations(const std::vector<std::vector<double>> &newFirstGuessCombinations);
    };


    class Crit3DProxyCombination
    {
    private:
        std::vector<bool> _isActiveList;
        std::vector<bool> _isSignificantList;
        bool _useThermalInversion;

    public:
        Crit3DProxyCombination();

        void clear();
        void addProxyActive(bool value) { _isActiveList.push_back(value); }
        void setProxyActive(unsigned index, bool value) { _isActiveList[index] = value; }
        bool isProxyActive(unsigned index) { return _isActiveList[index]; }
        std::vector<bool> getActiveList() { return _isActiveList; }

        void addProxySignificant(bool value) { _isSignificantList.push_back(value); }
        void setProxySignificant(unsigned index, bool value) { _isSignificantList[index] = value; }
        bool isProxySignificant(unsigned index) { return _isSignificantList[index]; }

        void resetCombination(unsigned int size);
        void setAllActiveToFalse();
        void setAllSignificantToFalse();

        unsigned int getActiveProxySize();
        unsigned int getProxySize() const { return unsigned(_isActiveList.size()); }

        bool getUseThermalInversion() const { return _useThermalInversion; }
        void setUseThermalInversion(bool value) { _useThermalInversion = value; }
    };


    class Crit3DMacroArea
    {
    private:
        Crit3DProxyCombination areaCombination;
        std::vector<std::vector<double>> areaParameters;
        std::vector<int> meteoPoints;
        std::vector<float> areaCellsDEM;
        std::vector<float> areaCellsGrid;

    public:
        Crit3DMacroArea();

        void clear();

        int getMeteoPointsNr() const
        { return int(meteoPoints.size()); }

        void setMeteoPoints (std::vector<int> myMeteoPoints) { meteoPoints = myMeteoPoints; }
        std::vector<int> getMeteoPoints() const { return meteoPoints; }

        void setAreaCellsDEM (std::vector<float> myCells) { areaCellsDEM = myCells; }
        const std::vector<float>& getAreaCellsDEM() const { return areaCellsDEM; }
        int getAreaCellsDemSize() const { return int(areaCellsDEM.size()); }

        void setAreaCellsGrid (std::vector<float> myCells) { areaCellsGrid = myCells; }
        std::vector<float> getAreaCellsGrid() const { return areaCellsGrid; }
        int getAreaCellsGridSize() const { return int(areaCellsGrid.size()); }

        void setParameters (std::vector<std::vector<double>> myParameters) { areaParameters = myParameters; }
        std::vector<std::vector<double>> getParameters() const { return areaParameters; }

        void setCombination (Crit3DProxyCombination myCombination) { areaCombination = myCombination; }
        Crit3DProxyCombination getCombination() const { return areaCombination; }
    };


    class Crit3DInterpolationSettings
    {
    private:
        gis::Crit3DRasterGrid* currentDEM; //for TD
		gis::Crit3DRasterGrid* macroAreasMap; //for glocal detrending

        TInterpolationMethod interpolationMethod;

        float minRegressionR2;
        bool useThermalInversion;
        bool useExcludeStationsOutsideDEM;
        bool useTD;
        bool useLocalDetrending;
		bool useGlocalDetrending;
        int maxTdMultiplier;
        bool useLapseRateCode;
        bool useBestDetrending;
        bool useMultipleDetrending;
        bool useDewPoint;
        bool useInterpolatedTForRH;
        bool useDoNotRetrend;
        bool useRetrendOnly;
        int minPointsLocalDetrending;
        bool meteoGridUpscaleFromDem;
        aggregationMethod meteoGridAggrMethod;

        bool isKrigingReady;
        bool precipitationAllZero;
        float maxHeightInversion;
        float pointsBoundingBoxArea;
        float localRadius;
        int indexPointCV;
        int topoDist_maxKh, topoDist_Kh;
        std::vector <float> Kh_series;
        std::vector <float> Kh_error_series;

        bool proxyLoaded;
        std::vector <Crit3DProxy> currentProxy;
        Crit3DProxyCombination selectedCombination;
        Crit3DProxyCombination currentCombination;
        unsigned indexHeight;

        std::vector <std::vector<double>> fittingParameters;
        std::vector<std::function<double(double, std::vector<double>&)>> fittingFunction;
        std::vector<double> pointsRange;

        std::vector<Crit3DMacroArea> macroAreas;
        std::vector <int> macroAreaNumbers;


    public:
        Crit3DInterpolationSettings();

        void initialize();
        void initializeProxy();

        Crit3DProxy* getProxy(unsigned pos) { return &(currentProxy[pos]); }

        std::string getProxyName(unsigned pos) const { return currentProxy[pos].getName(); }

        size_t getProxyNr() const { return currentProxy.size(); }

        bool getUseThermalInversion() const { return (useThermalInversion);}

        bool getUseExcludeStationsOutsideDEM() const { return (useExcludeStationsOutsideDEM); }

        bool getUseTD() const { return (useTD && !useLocalDetrending);}

        TInterpolationMethod getInterpolationMethod() const { return interpolationMethod;}

        bool getUseLocalDetrending() const { return useLocalDetrending;}

        bool getUseGlocalDetrending() const { return useGlocalDetrending;}

        bool getUseDoNotRetrend() const { return useDoNotRetrend; }

        bool getUseRetrendOnly() const { return useRetrendOnly; }

        float getMaxHeightInversion() const { return maxHeightInversion; }

        bool getUseDewPoint() const { return (useDewPoint);}

        bool getPrecipitationAllZero() const { return precipitationAllZero; }

        float getMinRegressionR2() const { return minRegressionR2; }

        bool getUseLapseRateCode() const { return useLapseRateCode; }

        bool getUseBestDetrending() const { return useBestDetrending; }

        gis::Crit3DRasterGrid* getCurrentDEM() const { return currentDEM; }

        std::vector<int> getMacroAreaNumber() const { return macroAreaNumbers; }

        gis::Crit3DRasterGrid* getMacroAreasMap() const { return macroAreasMap; }

        const std::vector<Crit3DMacroArea>& getMacroAreas() const { return macroAreas; }

        const Crit3DMacroArea& getMacroArea(int index) const
        {
            if (index >= macroAreas.size())
            {
                return *(new(Crit3DMacroArea));
            }
            return macroAreas[index];
        }

        int getMacroAreasSize() const { return int(macroAreas.size()); }

        aggregationMethod getMeteoGridAggrMethod() const { return meteoGridAggrMethod; }

        int getMinPointsLocalDetrending() const { return minPointsLocalDetrending; }

        int getIndexPointCV() const { return indexPointCV; }

        bool getProxyLoaded() const { return proxyLoaded; }

        const std::vector<float>& getKh_series() const { return Kh_series; }

        const std::vector<float>& getKh_error_series() const { return Kh_error_series; }

        int getTopoDist_maxKh() const { return topoDist_maxKh; }

        int getTopoDist_Kh() const { return topoDist_Kh; }

        Crit3DProxyCombination getSelectedCombination() const { return selectedCombination; }

        unsigned getIndexHeight() const { return indexHeight; }

        bool getUseInterpolatedTForRH() const { return useInterpolatedTForRH; }

        bool getMeteoGridUpscaleFromDem() const { return meteoGridUpscaleFromDem; }

        std::vector<double> getPointsRange() const { return pointsRange; }

        std::vector<std::vector<double>> getFittingParameters() const { return fittingParameters; }

        bool getUseMultipleDetrending() const { return useMultipleDetrending; }

        float getPointsBoundingBoxArea() const { return pointsBoundingBoxArea; }

        float getLocalRadius() const { return localRadius; }

        void addProxy(Crit3DProxy myProxy, bool isActive_);
        double getProxyValue(unsigned pos, std::vector<double> proxyValues) const;
        bool getCombination(int combinationInteger, Crit3DProxyCombination &outCombination);
        int getProxyPosFromName(TProxyVar name) const;

        void setInterpolationMethod(TInterpolationMethod myValue);
        void setUseThermalInversion(bool myValue);
        void setUseExcludeStationsOutsideDEM(bool myValue);
        void setUseTD(bool myValue);
        void setUseLocalDetrending(bool myValue);
        bool isGlocalReady(bool isGrid);
		void setUseGlocalDetrending(bool myValue);
        void setMacroAreasMap(gis::Crit3DRasterGrid *value);
        void setMacroAreas(std::vector<Crit3DMacroArea> myAreas);
        void pushMacroAreaNumber(int number);

        void clearMacroAreaNumber();

        void setUseDoNotRetrend(bool myValue);
        void setUseRetrendOnly(bool myValue);
        void setUseDewPoint(bool myValue);
        void setPrecipitationAllZero(bool value);
        void setMinRegressionR2(float value);
        void setUseLapseRateCode(bool value);
        void setUseBestDetrending(bool value);
        void setMeteoGridAggrMethod(const aggregationMethod &value);
        void setShepardInitialRadius(float value);
        void setIndexPointCV(int value);
        void setCurrentDEM(gis::Crit3DRasterGrid *value);
        void setTopoDist_maxKh(int value);
        void setTopoDist_Kh(int value);
        Crit3DProxyCombination getOptimalCombination() const;
        void setOptimalCombination(const Crit3DProxyCombination &value);
        void setSelectedCombination(const Crit3DProxyCombination &value);
        void setActiveSelectedCombination(unsigned int index, bool isActive);
        void setIndexHeight(unsigned value);
        Crit3DProxyCombination getCurrentCombination() const;
        void setCurrentCombination(Crit3DProxyCombination value);
        void setSignificantCurrentCombination(unsigned int index, bool isSignificant);
        std::vector<Crit3DProxy> getCurrentProxy() const;
        void setCurrentProxy(const std::vector<Crit3DProxy> &value);
        void setUseInterpolatedTForRH(bool value);
        void setProxyLoaded(bool value);
        void setKh_series(const std::vector<float> &newKh_series);
        void setKh_error_series(const std::vector<float> &newKh_error_series);
        void addToKhSeries(float kh, float error);
        void initializeKhSeries();
        void setMeteoGridUpscaleFromDem(bool newMeteoGridUpscaleFromDem);
        void setUseMultipleDetrending(bool newUseMultipleDetrending);
        void setPointsBoundingBoxArea(float newPointsBoundingBoxArea);
        void setLocalRadius(float newLocalRadius);
        void setMinPointsLocalDetrending(int newMinPointsLocalDetrending);

        std::vector<double> getProxyFittingParameters(int tempIndex);
        void setFittingParameters(const std::vector<std::vector <double>> &newFittingParameters);
        void addFittingParameters(const std::vector<std::vector<double> > &newFittingParameters);
        std::vector<std::function<double (double, std::vector<double> &)> > getFittingFunction() const;
        void setFittingFunction(const std::vector<std::function<double (double, std::vector<double> &)> > &newFittingFunction);
        void addFittingFunction(const std::function<double (double, std::vector<double> &)> &newFittingFunction);
        void clearFitting();
        TFittingFunction getChosenElevationFunction();
        void setChosenElevationFunction(TFittingFunction chosenFunction);
        void setPointsRange(double min, double max);
    };

#endif // INTERPOLATIONSETTINGS_H
