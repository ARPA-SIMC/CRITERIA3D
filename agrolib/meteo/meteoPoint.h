#ifndef METEOPOINT_H
#define METEOPOINT_H

    #include <string>

    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef QUALITY_H
        #include "quality.h"
    #endif

    struct TObsDataH {
        Crit3DDate date;
        std::vector<float> tAir;
        std::vector<float> prec;
        std::vector<float> rhAir;
        std::vector<float> tDew;
        std::vector<float> irradiance;
        std::vector<float> netIrradiance;
        std::vector<float> et0;
        std::vector<float> windScalInt;
        std::vector<float> windVecX;
        std::vector<float> windVecY;
        std::vector<float> windVecInt;
        std::vector<float> windVecDir;
        std::vector<int> leafW;
        std::vector<float> transmissivity;
        std::vector<float> pressure;
    };

    struct TObsDataD {
        Crit3DDate date;
        float tMin;
        float tMax;
        float tAvg;
        float prec;
        float rhMin;
        float rhMax;
        float rhAvg;
        float globRad;
        float et0_hs;
        float et0_pm;
        float bic;
        float dd_heating;
        float dd_cooling;
        float windVecIntAvg;
        float windVecIntMax;
        float windScalIntAvg;
        float windScalIntMax;
        float windVecDirPrev;
        float windIntMax;
        float leafW;
        float waterTable;       // [m]
    };

    struct TObsDataM {
        int _month;
        int _year;
        float tMin;
        float tMax;
        float tAvg;
        float prec;
        float et0_hs;
        float globRad;
        float bic;
    };

    class Crit3DMeteoPoint {
        public:
            std::string name;
            std::string id;
            std::string dataset;
            std::string state;
            std::string region;
            std::string province;
            std::string municipality;

            gis::Crit3DPoint point;
            double latitude;
            double longitude;
            double area;
            int latInt;
            int lonInt;
            bool isInsideDem;

            bool isUTC;
            bool isForecast;
            int hourlyFraction;

            std::vector<TObsDataH> obsDataH;
            std::vector<TObsDataD> obsDataD;
            std::vector<TObsDataM> obsDataM;

            quality::qualityType quality;
            float currentValue;
            float residual;
            float elaboration;

            float anomaly;
            float anomalyPercentage;
            float climate;

            bool active;
            bool selected;
            bool marked;

            lapseRateCodeType lapseRateCode;

            std::vector <float> proxyValues;
            std::vector<float> glocalWeights;

            std::vector<gis::Crit3DPoint> aggregationPoints;
            long aggregationPointsMaxNr;

            gis::Crit3DRasterGrid* topographicDistanceMap;

            Crit3DMeteoPoint();
            void clear();

            bool initializeObsDataH(int hourlyFraction, int numberOfDays, const Crit3DDate& firstDate);
            void initializeObsDataD(unsigned int numberOfDays, const Crit3DDate& firstDate);
            void initializeObsDataM(unsigned int numberOfMonths, unsigned int month, int year);

            void emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& myDate);
            void emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2);
            bool emptyObsDataH(const Crit3DDate& date1, const Crit3DDate& date2);
            bool emptyObsDataD(const Crit3DDate& date1, const Crit3DDate& date2);
            bool emptyObsDataM(const Crit3DDate& date1, const Crit3DDate& date2);

            bool emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2);

            void cleanAllData();

            bool isDateLoadedH(const Crit3DDate& myDate) const;
            bool isDateTimeLoadedH(const Crit3DTime& myDateTime) const;
            bool isDateIntervalLoadedH(const Crit3DDate& date1, const Crit3DDate& date2) const;
            bool isDateIntervalLoadedH(const Crit3DTime& time1, const Crit3DTime& time2) const;

            float obsDataConsistencyH(meteoVariable myVar, const Crit3DTime& time1, const Crit3DTime& time2) const;

            bool isDateLoadedD(const Crit3DDate& myDate) const;
            bool isDateLoadedM(const Crit3DDate& myDate) const;
            bool isDateIntervalLoadedD(const Crit3DDate& date1, const Crit3DDate& date2) const;
            bool isDateIntervalLoadedM(const Crit3DDate& date1, const Crit3DDate& date2) const;

            bool existDailyData(const Crit3DDate& myDate) const;

            Crit3DDate getFirstDailyData() const;
            Crit3DDate getLastDailyData() const;

            bool existDailyData() const { return ! obsDataD.empty(); }
            bool existHourlyData() const { return ! obsDataH.empty(); }

            float getMeteoPointValue(const Crit3DTime& myTime, meteoVariable myVar, Crit3DMeteoSettings *meteoSettings);

            float getMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar) const;
            bool setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue);
            float getMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings) const;
            float getMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar) const;
            bool setMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, float myValue);
            bool getMeteoPointValueDayH(const Crit3DDate& myDate, TObsDataH *&hourlyValues);
            float getMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar) const;
            bool setMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar, float myValue);

            Crit3DDate getMeteoPointHourlyValuesDate(int index) const;

            float getProxyValue(unsigned pos);
            std::vector<double> getProxyValues();

            void setId(const std::string &myId) { id = myId; }

            void setName(const std::string &myName) { name = myName; }

            void setAltitude(double altitude) { point.z = altitude; }

            void setDataset(const std::string &myDataset) { dataset = myDataset; }

            void setLapseRateCode(const std::string &lapseRateCodeStr);

            bool computeHourlyDerivedVar(const Crit3DTime &dateTime, meteoVariable myVar, bool useNetRad);
            bool computeDailyDerivedVar(const Crit3DDate &date, meteoVariable myVar, Crit3DMeteoSettings &meteoSettings);

            bool computeMonthlyAggregate(const Crit3DDate &firstDate, const Crit3DDate &lastDate, meteoVariable dailyMeteoVar,
                                         Crit3DMeteoSettings *meteoSettings, Crit3DQuality *qualityCheck,
                                         const Crit3DClimateParameters &climateParam);

            void initializeObsDataDFromMp(unsigned int numberOfDays, const Crit3DDate& firstDate,
                                          Crit3DMeteoPoint &refMeteoPoint);
            void initializeObsDataHFromMp(int myHourlyFraction, int numberOfDays, const Crit3DDate& firstDate,
                                          const Crit3DMeteoPoint &meteoPoint);

            float getPercValueVariable(const Crit3DDate &firstDate, const Crit3DDate &lastDate, meteoVariable dailyMeteoVar);

            bool getDailyDataCsv_TPrec(std::string &outStr);
            float getDailyET0_HS(const TObsDataD &dailyData, const Crit3DDate &myDate, Crit3DMeteoSettings* meteoSettings) const;

    private:

    };

    bool isSelectionPointsActive(const std::vector<Crit3DMeteoPoint> &meteoPoints);

#endif // METEOPOINT_H
