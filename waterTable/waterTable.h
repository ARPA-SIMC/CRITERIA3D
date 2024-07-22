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
        float WTClimateMonthly[12];
        float WTClimateDaily[366];

        QDate firstDate;
        std::vector<float> hindcastSeries;
        std::vector<float> interpolationSeries;

        WaterTable(std::vector<float> &inputTMin, std::vector<float> &inputTMax, std::vector<float> &inputPrec,
                   QDate firstMeteoDate, QDate lastMeteoDate, Crit3DMeteoSettings meteoSettings);

        void initializeWaterTable(Well myWell);
        bool computeWaterTableParameters(Well myWell, int stepDays);
        bool computeWTClimate();
        bool computeETP_allSeries(bool isUpdateAvgCWB);
        bool computeCWBCorrelation(int stepDays);
        double computeCWB(QDate myDate, int nrDays);
        bool computeWaterTableIndices();
        float getWaterTableDaily(QDate myDate);
        float getWaterTableClimate(QDate myDate);
        bool computeWaterTableClimate(QDate currentDate, int yearFrom, int yearTo, float* myValue);
        bool getWaterTableInterpolation(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays);
        void computeWaterTableSeries();

        bool setMeteoData(QDate myDate, float tmin, float tmax, float prec);

        void setInputTMin(const std::vector<float> &newInputTMin);
        void setInputTMax(const std::vector<float> &newInputTMax);
        void setInputPrec(const std::vector<float> &newInputPrec);

        void setFirstMeteoDate(QDate myDate) { firstMeteoDate = myDate; }
        void setLastMeteoDate(QDate myDate) { lastMeteoDate = myDate; }

        QString getError() const { return errorStr; }

        double getAlpha() const { return alpha; }
        double getH0() const { return h0; }

        float getR2() const { return R2; }
        float getRMSE() const { return RMSE; }
        float getEF() const { return EF; }

        int getNrDaysPeriod() const { return nrDaysPeriod; }
        int getNrObsData() const { return nrObsData; }

        QString getIdWell() const { return well.getId(); }
        QDate getFirstDateWell() { return well.getFirstDate(); }
        QDate getLastDateWell() { return well.getLastDate(); }

        Well* getWell() { return &well; }

        void cleanAllMeteoVector();

    private:
        Crit3DMeteoSettings meteoSettings;

        QDate firstMeteoDate;
        QDate lastMeteoDate;
        Well well;
        QString errorStr;

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
        float EF;

        bool isClimateReady;
        int nrObsData;

        bool isCWBEquationReady;
        double avgDailyCWB; //[mm]
};

#endif // WATERTABLE_H
