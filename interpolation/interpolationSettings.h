#ifndef INTERPOLATIONSETTINGS_H
#define INTERPOLATIONSETTINGS_H

#include <functional>
#ifndef INTERPOLATIONCONSTS_H
        #include "interpolationConstants.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif


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
        double getValue(unsigned int pos, std::vector<double> proxyValues);
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
        TFittingFunction getFittingFunctionName();
        void setFittingFunctionName(TFittingFunction functionName);
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


    class Crit3DInterpolationSettings
    {
    private:
        gis::Crit3DRasterGrid* currentDEM; //for TD

        TInterpolationMethod interpolationMethod;

        float minRegressionR2;
        bool useThermalInversion;
        bool useTD;
        bool useLocalDetrending;
        int maxTdMultiplier;
        bool useLapseRateCode;
        bool useBestDetrending;
        bool useMultipleDetrending;
        bool useDewPoint;
        bool useInterpolatedTForRH;
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
        bool proxiesComplete;
        std::vector <Crit3DProxy> currentProxy;
        Crit3DProxyCombination selectedCombination;
        Crit3DProxyCombination currentCombination;
        unsigned indexHeight;

        std::vector <std::vector<double>> fittingParameters;
        std::vector<std::function<double(double, std::vector<double>&)>> fittingFunction;
        std::vector<double> pointsRange;


    public:
        Crit3DInterpolationSettings();

        void initialize();
        void initializeProxy();

        Crit3DProxy* getProxy(unsigned pos);
        std::string getProxyName(unsigned pos);
        size_t getProxyNr();
        void addProxy(Crit3DProxy myProxy, bool isActive_);
        double getProxyValue(unsigned pos, std::vector<double> proxyValues);
        bool getCombination(int combinationInteger, Crit3DProxyCombination &outCombination);
        int getProxyPosFromName(TProxyVar name);

        void setInterpolationMethod(TInterpolationMethod myValue);
        TInterpolationMethod getInterpolationMethod();

        void setUseThermalInversion(bool myValue);
        bool getUseThermalInversion();

        void setUseTD(bool myValue);
        bool getUseTD();

        void setUseLocalDetrending(bool myValue);
        bool getUseLocalDetrending();

        void setUseDewPoint(bool myValue);
        bool getUseDewPoint();

        float getMaxHeightInversion();

        bool getPrecipitationAllZero() const;
        void setPrecipitationAllZero(bool value);
        float getMinRegressionR2() const;
        void setMinRegressionR2(float value);
        bool getUseLapseRateCode() const;
        void setUseLapseRateCode(bool value);
        bool getUseBestDetrending() const;
        void setUseBestDetrending(bool value);
        aggregationMethod getMeteoGridAggrMethod() const;
        void setMeteoGridAggrMethod(const aggregationMethod &value);
        float getShepardInitialRadius() const;
        void setShepardInitialRadius(float value);
        int getIndexPointCV() const;
        void setIndexPointCV(int value);
        gis::Crit3DRasterGrid *getCurrentDEM() const;
        void setCurrentDEM(gis::Crit3DRasterGrid *value);
        int getTopoDist_maxKh() const;
        void setTopoDist_maxKh(int value);
        int getTopoDist_Kh() const;
        void setTopoDist_Kh(int value);
        Crit3DProxyCombination getOptimalCombination() const;
        void setOptimalCombination(const Crit3DProxyCombination &value);
        Crit3DProxyCombination getSelectedCombination() const;
        void setSelectedCombination(const Crit3DProxyCombination &value);
        void setActiveSelectedCombination(unsigned int index, bool isActive);
        unsigned getIndexHeight() const;
        void setIndexHeight(unsigned value);
        Crit3DProxyCombination getCurrentCombination() const;
        void setCurrentCombination(Crit3DProxyCombination value);
        void setSignificantCurrentCombination(unsigned int index, bool isSignificant);
        std::vector<Crit3DProxy> getCurrentProxy() const;
        void setCurrentProxy(const std::vector<Crit3DProxy> &value);
        bool getUseInterpolatedTForRH() const;
        void setUseInterpolatedTForRH(bool value);
        bool getProxyLoaded() const;
        void setProxyLoaded(bool value);
        const std::vector<float> &getKh_series() const;
        void setKh_series(const std::vector<float> &newKh_series);
        const std::vector<float> &getKh_error_series() const;
        void setKh_error_series(const std::vector<float> &newKh_error_series);
        void addToKhSeries(float kh, float error);
        void initializeKhSeries();
        bool getMeteoGridUpscaleFromDem() const;
        void setMeteoGridUpscaleFromDem(bool newMeteoGridUpscaleFromDem);
        bool getUseMultipleDetrending() const;
        void setUseMultipleDetrending(bool newUseMultipleDetrending);
        float getPointsBoundingBoxArea() const;
        void setPointsBoundingBoxArea(float newPointsBoundingBoxArea);
        float getLocalRadius() const;
        void setLocalRadius(float newLocalRadius);
        int getMinPointsLocalDetrending() const;
        void setMinPointsLocalDetrending(int newMinPointsLocalDetrending);

        std::vector<std::vector <double>> getFittingParameters() const;
        std::vector<double> getProxyFittingParameters(int tempIndex);
        void setFittingParameters(const std::vector<std::vector <double>> &newFittingParameters);
        void addFittingParameters(const std::vector<std::vector<double> > &newFittingParameters);
        std::vector<std::function<double (double, std::vector<double> &)> > getFittingFunction() const;
        void setFittingFunction(const std::vector<std::function<double (double, std::vector<double> &)> > &newFittingFunction);
        void addFittingFunction(const std::function<double (double, std::vector<double> &)> &newFittingFunction);
        bool getProxiesComplete() const;
        void setProxiesComplete(bool newProxiesComplete);
        void clearFitting();
        TFittingFunction getChosenElevationFunction();
        void setChosenElevationFunction(TFittingFunction chosenFunction);
        void setPointsRange(double min, double max);
        std::vector<double> getPointsRange();
    };

#endif // INTERPOLATIONSETTINGS_H
