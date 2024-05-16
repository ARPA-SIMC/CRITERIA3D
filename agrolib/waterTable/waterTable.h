#ifndef WATERTABLE_H
#define WATERTABLE_H

#include "well.h"
#include "meteoPoint.h"
#include "meteoGrid.h"
#include <QDate>

#define MAXWELLDISTANCE 5000                           // distanza max: 10 km
#define WATERTABLE_MAXDELTADAYS 90

class WaterTable
{
    public:
        WaterTable(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DMeteoGrid* meteoGrid, bool isMeteoGridLoaded, Crit3DMeteoSettings meteoSettings, gis::Crit3DGisSettings gisSettings);
        QDate getFirstDateWell();
        QDate getLastDateWell();
        void initializeWaterTable(Well myWell);
        bool computeWaterTable(Well myWell, int maxNrDays, int doy1, int doy2);
        bool computeWTClimate();
        bool assignNearestMeteoPoint();
        bool assignWTMeteoData(Crit3DMeteoPoint point);
        bool computeETP_allSeries();
        bool computeCWBCorrelation(int maxNrDays);
        float computeCWB(QDate myDate, int nrDays);
        bool computeWaterTableIndices(int doy1, int doy2);
        float getWaterTableDaily(QDate myDate);
        float getWaterTableClimate(QDate myDate);
        bool computeWaterTableClimate(QDate currentDate, int yearFrom, int yearTo, float* myValue);
        bool getWaterTableHindcast(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays);
        QString getError() const;

    private:
        Crit3DMeteoPoint *meteoPoints;
        int nrMeteoPoints;
        Crit3DMeteoGrid *meteoGrid;
        bool isMeteoGridLoaded;
        Crit3DMeteoSettings meteoSettings;
        gis::Crit3DGisSettings gisSettings;
        QDate firstDateWell;
        QDate lastDateWell;
        QDate firstMeteoDate;
        QDate lastMeteoDate;
        Well well;
        QString error;

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

        bool isMeteoPointLinked;
        bool isCWBEquationReady;
        float avgDailyCWB; //[mm]
        Crit3DMeteoPoint linkedMeteoPoint;
};

#endif // WATERTABLE_H
