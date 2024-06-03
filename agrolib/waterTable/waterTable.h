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
        double computeCWB(QDate myDate, int nrDays);
        bool computeWaterTableIndices();
        float getWaterTableDaily(QDate myDate);
        float getWaterTableClimate(QDate myDate);
        bool computeWaterTableClimate(QDate currentDate, int yearFrom, int yearTo, float* myValue);
        bool getWaterTableInterpolation(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays);
        void computeWaterTableSeries();

        QString getError() const { return error; }

        double getAlpha() const { return alpha; }
        double getH0() const { return h0; }

        float getR2() const { return R2; }
        float getRMSE() const { return RMSE; }
        float getNASH() const { return NASH; }
        float getEF() const { return EF; }

        int getNrDaysPeriod() const { return nrDaysPeriod; }
        int getNrObsData() const { return nrObsData; }

        std::vector<QDate> getMyDates();
        std::vector<float> getMyHindcastSeries();
        std::vector<float> getMyInterpolateSeries();
        QMap<QDate, float> getObsDepths();

        void cleanAllMeteoVector();
        void setInputTMin(const std::vector<float> &newInputTMin);

        void setInputTMax(const std::vector<float> &newInputTMax);

        void setInputPrec(const std::vector<float> &newInputPrec);

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
        double alpha;
        double h0;

        float R2;
        float RMSE;
        float NASH;
        float EF;

        bool isClimateReady;
        float WTClimateMonthly[12];
        float WTClimateDaily[366];
        int nrObsData;

        bool isCWBEquationReady;
        double avgDailyCWB; //[mm]

        // graph
        std::vector<QDate> myDates;
        std::vector<float> myHindcastSeries;
        std::vector<float> myInterpolateSeries;
};

#endif // WATERTABLE_H
