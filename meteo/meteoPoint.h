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
        float* et0;
        float* windVecInt;
        float* windScalInt;
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
        float et0;
        float globRad;
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

            TObsDataD *obsDataD;
            TObsDataM *obsDataM;
            quality::qualityType quality;
            float currentValue;
            float residual;
            float elaboration;

            float anomaly;
            float anomalyPercentage;
            float climate;
            bool active;
            bool selected;

            std::vector <float> proxyValues;
            lapseRateCodeType lapseRateCode;
            gis::Crit3DRasterGrid* topographicDistance;

            Crit3DMeteoPoint();

            void initializeObsDataH(int hourlyFraction, int numberOfDays, const Crit3DDate& firstDate);
            void emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& myDate);
            void emptyVarObsDataH(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2);
            void cleanObsDataH();
            bool isDateLoadedH(const Crit3DDate& myDate);
            bool isDateIntervalLoadedH(const Crit3DDate& date1, const Crit3DDate& date2);
            bool isDateIntervalLoadedH(const Crit3DTime& time1, const Crit3DTime& time2);
            float obsDataConsistencyH(meteoVariable myVar, const Crit3DTime& time1, const Crit3DTime& time2);

            void initializeObsDataD(int numberOfDays, const Crit3DDate& firstDate);
            void emptyVarObsDataD(meteoVariable myVar, const Crit3DDate& date1, const Crit3DDate& date2);
            void cleanObsDataD();
            bool isDateLoadedD(const Crit3DDate& myDate);
            bool isDateIntervalLoadedD(const Crit3DDate& date1, const Crit3DDate& date2);

            void initializeObsDataM(int numberOfMonths, int month, int year);
            void cleanObsDataM();

            float getMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar);
            bool setMeteoPointValueH(const Crit3DDate& myDate, int myHour, int myMinutes, meteoVariable myVar, float myValue);
            float getMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar);
            bool setMeteoPointValueD(const Crit3DDate& myDate, meteoVariable myVar, float myValue);

            bool getMeteoPointValueDayH(const Crit3DDate& myDate, TObsDataH *hourlyValues);
            Crit3DDate getMeteoPointHourlyValuesDate(int index);

            float getMeteoPointValue(const Crit3DTime& myTime, meteoVariable myVar);

            float getProxyValue(unsigned pos);
            std::vector <float> getProxyValues();

        private:
            TObsDataH *obsDataH;

    };


#endif // METEOPOINT_H
