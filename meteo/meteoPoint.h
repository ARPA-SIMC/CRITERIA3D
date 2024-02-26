#ifndef METEOPOINT_H
#define METEOPOINT_H

    #ifndef _STRING_
        #include <string>
    #endif
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
        float* tAir;
        float* prec;
        float* rhAir;
        float* tDew;
        float* irradiance;
        float* netIrradiance;
        float* et0;
        float* windScalInt;
        float* windVecX;
        float* windVecY;
        float* windVecInt;
        float* windVecDir;
        int* leafW;
        float* transmissivity;
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

            std::vector<gis::Crit3DPoint> aggregationPoints;
            long aggregationPointsMaxNr;

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
            long nrObsDataDaysH;
            long nrObsDataDaysD;
            long nrObsDataDaysM;

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

            std::vector <float> proxyValues;
            lapseRateCodeType lapseRateCode;
            gis::Crit3DRasterGrid* topographicDistance;

            Crit3DMeteoPoint();
            void clear();

            void initializeObsDataH(int hourlyFraction, int numberOfDays, const Crit3DDate& firstDate);
            void emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& myDate);
            void emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2);
            void emptyObsDataH(const Crit3DDate& date1, const Crit3DDate& date2);
            void emptyObsDataD(const Crit3DDate& date1, const Crit3DDate& date2);
            void emptyObsDataM(const Crit3DDate& date1, const Crit3DDate& date2);

            void cleanObsDataH();
            void cleanObsDataD();
            void cleanObsDataM();

            bool isDateLoadedH(const Crit3DDate& myDate);
            bool isDateTimeLoadedH(const Crit3DTime& myDateTime);
            bool isDateIntervalLoadedH(const Crit3DDate& date1, const Crit3DDate& date2);
            bool isDateIntervalLoadedH(const Crit3DTime& time1, const Crit3DTime& time2);
            float obsDataConsistencyH(meteoVariable myVar, const Crit3DTime& time1, const Crit3DTime& time2);

            void initializeObsDataD(unsigned int numberOfDays, const Crit3DDate& firstDate);
            void emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2);
            bool isDateLoadedD(const Crit3DDate& myDate);
            bool isDateLoadedM(const Crit3DDate& myDate);
            bool isDateIntervalLoadedD(const Crit3DDate& date1, const Crit3DDate& date2);
            bool isDateIntervalLoadedM(const Crit3DDate& date1, const Crit3DDate& date2);

            void initializeObsDataM(unsigned int numberOfMonths, unsigned int month, int year);

            bool existDailyData(const Crit3DDate& myDate);
            Crit3DDate getLastDailyData();

            float getMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar);
            bool setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue);
            float getMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings);
            float getMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar);
            bool setMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, float myValue);
            bool getMeteoPointValueDayH(const Crit3DDate& myDate, TObsDataH *&hourlyValues);
            Crit3DDate getMeteoPointHourlyValuesDate(int index);
            float getMeteoPointValue(const Crit3DTime& myTime, meteoVariable myVar, Crit3DMeteoSettings *meteoSettings);
            float getMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar);
            bool setMeteoPointValueM(const Crit3DDate &myDate, meteoVariable myVar, float myValue);

            float getProxyValue(unsigned pos);
            std::vector<double> getProxyValues();

            void setId(std::string value);
            void setName(std::string name);
            void setDataset(std::string dataset);
            void setLapseRateCode(std::string lapseRateCode);

            bool computeDerivedVariables(Crit3DTime dateTime);
            bool computeMonthlyAggregate(Crit3DDate firstDate, Crit3DDate lastDate, meteoVariable dailyMeteoVar, Crit3DMeteoSettings *meteoSettings, Crit3DQuality *qualityCheck, Crit3DClimateParameters *climateParam);
            TObsDataH *getObsDataH() const;
            void initializeObsDataDFromMp(unsigned int numberOfDays, const Crit3DDate& firstDate, Crit3DMeteoPoint mp);
            void initializeObsDataHFromMp(int myHourlyFraction, int numberOfDays, const Crit3DDate& firstDate, Crit3DMeteoPoint mp);

            bool getDailyDataCsv_TPrec(std::string &outStr);

    private:
            TObsDataH *obsDataH;

    };

    bool isSelectionPointsActive(Crit3DMeteoPoint* meteoPoints,int nrMeteoPoints);

#endif // METEOPOINT_H
