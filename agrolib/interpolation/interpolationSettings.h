#ifndef INTERPOLATIONSETTINGS_H
#define INTERPOLATIONSETTINGS_H

    #ifndef INTERPOLATIONCONSTS_H
        #include "interpolationConstants.h"
    #endif
    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef METEOGRID_H
        #include "meteoGrid.h"
    #endif

    #ifndef _DEQUE_
        #include <deque>
    #endif

    std::string getKeyStringInterpolationMethod(TInterpolationMethod value);
    TProxyVar getProxyPragaName(std::string name_);

    class Crit3DProxy
    {
    private:
        std::string name;
        std::string gridName;
        gis::Crit3DRasterGrid* grid;
        std::string proxyTable;
        std::string proxyField;
        bool isSignificant;
        bool forQualityControl;

        float regressionR2;
        float regressionSlope;

        //orography
        float lapseRateH1;
        float lapseRateH0;
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
        bool getIsSignificant() const;
        void setIsSignificant(bool value);
        void setRegressionR2(float myValue);
        float getRegressionR2();
        void setRegressionSlope(float myValue);
        float getRegressionSlope();
        float getValue(unsigned int pos, std::vector <float> proxyValues);
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
    };

    class Crit3DProxyCombination
    {
    private:
        std::deque<bool> isActive;
        bool useThermalInversion;


    public:
        Crit3DProxyCombination();

        void clear();
        bool getUseThermalInversion() const;
        void setUseThermalInversion(bool value);
        void addValue(bool isActive_);
        void setValue(unsigned index, bool isActive_);
        bool getValue(unsigned index);
        std::deque<bool> getIsActive() const;
        void setIsActive(const std::deque<bool> &value);
    };

    class Crit3DInterpolationSettings
    {
    private:
        gis::Crit3DRasterGrid* currentDEM; //for TAD

        TInterpolationMethod interpolationMethod;

        float minRegressionR2;
        bool useThermalInversion;
        bool useTAD;
        bool useLapseRateCode;
        bool useBestDetrending;
        bool useDewPoint;
        bool useInterpolatedTForRH;
        float refHeightWind;
        float surfaceRoughness;

        aggregationMethod meteoGridAggrMethod;

        bool isKrigingReady;
        bool precipitationAllZero;
        float maxHeightInversion;
        float shepardInitialRadius;
        int indexPointCV;
        float topoDist_Kh, topoDist_Kz;

        bool proxyLoaded;
        std::vector <Crit3DProxy> currentProxy;
        Crit3DProxyCombination optimalCombination;
        Crit3DProxyCombination selectedCombination;
        Crit3DProxyCombination *currentCombination;
        unsigned indexHeight;

    public:
        Crit3DInterpolationSettings();

        void initialize();
        void initializeProxy();

        void computeShepardInitialRadius(float area, int nrPoints);

        Crit3DProxy* getProxy(unsigned pos);
        std::string getProxyName(unsigned pos);
        size_t getProxyNr();
        void addProxy(Crit3DProxy myProxy, bool isActive_);
        float getProxyValue(unsigned pos, std::vector <float> proxyValues);
        bool getCombination(int combinationInteger, Crit3DProxyCombination* outCombination);

        void setInterpolationMethod(TInterpolationMethod myValue);
        TInterpolationMethod getInterpolationMethod();

        void setUseThermalInversion(bool myValue);
        bool getUseThermalInversion();

        void setUseTAD(bool myValue);
        bool getUseTAD();

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
        float getTopoDist_Kh() const;
        void setTopoDist_Kh(float value);
        float getTopoDist_Kz() const;
        void setTopoDist_Kz(float value);
        Crit3DProxyCombination getOptimalCombination() const;
        Crit3DProxyCombination* getOptimalCombinationRef();
        void setOptimalCombination(const Crit3DProxyCombination &value);
        Crit3DProxyCombination getSelectedCombination() const;
        Crit3DProxyCombination* getSelectedCombinationRef();
        void setSelectedCombination(const Crit3DProxyCombination &value);
        void setValueSelectedCombination(unsigned int index, bool isActive);
        unsigned getIndexHeight() const;
        void setIndexHeight(unsigned value);
        Crit3DProxyCombination *getCurrentCombination() const;
        void setCurrentCombination(Crit3DProxyCombination *value);
        std::vector<Crit3DProxy> getCurrentProxy() const;
        void setCurrentProxy(const std::vector<Crit3DProxy> &value);
        float getRefHeightWind() const;
        void setRefHeightWind(float value);
        float getSurfaceRoughness() const;
        void setSurfaceRoughness(float value);
        bool getUseInterpolatedTForRH() const;
        void setUseInterpolatedTForRH(bool value);
        bool getProxyLoaded() const;
        void setProxyLoaded(bool value);
    };

#endif // INTERPOLATIONSETTINGS_H
