#ifndef WATERTABLE_H
#define WATERTABLE_H

#ifndef WELL_H
    #include "well.h"
#endif
#ifndef METEO_H
    #include "meteo.h"
#endif
#ifndef GIS_H
    #include "gis.h"
#endif
#include <QDate>

#define MAXWELLDISTANCE 5000                           // distanza max: 5 km
#define WATERTABLE_MAXDELTADAYS 90

class WaterTable
{
    public:
        WaterTable(std::vector<float> &inputTMin, std::vector<float> &inputTMax, std::vector<float> &inputPrec, QDate firstMeteoDate, QDate lastMeteoDate,
                   Crit3DMeteoSettings meteoSettings, gis::Crit3DGisSettings gisSettings);
        QString getIdWell() const;
        QDate getFirstDateWell();
        QDate getLastDateWell();
        void initializeWaterTable(Well myWell);
        bool computeWaterTableParameters(Well myWell, int maxNrDays);
        bool computeWTClimate();
        bool computeETP_allSeries();
        bool computeCWBCorrelation(int maxNrDays);
        float computeCWB(QDate myDate, int nrDays);
        bool computeWaterTableIndices();
        float getWaterTableDaily(QDate myDate);
        float getWaterTableClimate(QDate myDate);
        bool computeWaterTableClimate(QDate currentDate, int yearFrom, int yearTo, float* myValue);
        bool getWaterTableInterpolation(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays);
        void computeWaterTableSeries();
        QString getError() const;

        float getAlpha() const;
        float getH0() const;
        int getNrDaysPeriod() const;
        float getR2() const;
        float getRMSE() const;
        float getNASH() const;
        float getEF() const;
        int getNrObsData() const;

        std::vector<QDate> getMyDates() const;
        std::vector<float> getMyHindcastSeries() const;
        std::vector<float> getMyInterpolateSeries() const;
        QMap<QDate, int> getObsDepths();

    private:
        Crit3DMeteoSettings meteoSettings;
        gis::Crit3DGisSettings gisSettings;
        QDate firstDateWell;
        QDate lastDateWell;
        QDate firstMeteoDate;
        QDate lastMeteoDate;
        Well well;
        QString error;

        std::vector<float> inputTMin;
        std::vector<float> inputTMax;
        std::vector<float> inputPrec;
        std::vector<float> etpValues;
        std::vector<float> precValues;

        int nrDaysPeriod;
        float alpha;
        float h0;
        float R2;
        float RMSE;
        float NASH;
        float EF;

        bool isClimateReady;
        float WTClimateMonthly[12];
        float WTClimateDaily[366];
        int nrObsData;

        bool isCWBEquationReady;
        float avgDailyCWB; //[mm]

        // graph
        std::vector<QDate> myDates;
        std::vector<float> myHindcastSeries;
        std::vector<float> myInterpolateSeries;
};

#endif // WATERTABLE_H
