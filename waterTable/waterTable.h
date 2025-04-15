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

#define MAXWELLDISTANCE 5000                           // maximum distance: 5 km
#define WATERTABLE_MAXDELTADAYS 90


class WaterTable
{
    public:
        float WTClimateMonthly[12];
        float WTClimateDaily[366];

        std::vector<float> hindcastSeries;
        std::vector<float> interpolationSeries;

        WaterTable(std::vector<float> &inputTMin, std::vector<float> &inputTMax, std::vector<float> &inputPrec,
                   QDate firstMeteoDate, QDate lastMeteoDate, Crit3DMeteoSettings meteoSettings);

        void initializeWaterTable(const Well &myWell);
        bool computeWaterTableParameters(const Well &myWell, int stepDays);
        bool computeWTClimate();
        bool computeETP_allSeries(bool isUpdateAvgCWB);
        bool computeCWBCorrelation(int stepDays);
        double computeCWB(const QDate &myDate, int nrDays);
        bool computeWaterTableIndices();
        float getWaterTableDaily(const QDate &myDate);
        float getWaterTableClimate(const QDate &myDate);

        bool computeWaterTableClimate(QDate mycurrentDate, int yearFrom, int yearTo, float* myValue);
        bool getWaterTableInterpolation(QDate myDate, float* myValue, float* myDelta, int* myDeltaDays);

        void computeWaterTableSeries();

        bool setMeteoData(const QDate &date, float tmin, float tmax, float prec);

        void setInputTMin(const std::vector<float> &newInputTMin);
        void setInputTMax(const std::vector<float> &newInputTMax);
        void setInputPrec(const std::vector<float> &newInputPrec);

        void setFirstMeteoDate(const QDate &myDate) { _firstMeteoDate = myDate; }
        void setLastMeteoDate(const QDate &myDate) { _lastMeteoDate = myDate; }

        QString getError() const { return _errorStr; }

        double getAlpha() const { return alpha; }
        double getH0() const { return h0; }

        float getR2() const { return R2; }
        float getRMSE() const { return RMSE; }
        float getEF() const { return EF; }

        int getNrDaysPeriod() const { return nrDaysPeriod; }
        int getNrObsData() const { return nrObsData; }

        QString getIdWell() const { return _well.getId(); }
        QDate getFirstDate() { return std::min(_well.getFirstObsDate(), _firstMeteoDate); }
        QDate getFirstDateWell() { return _well.getFirstObsDate(); }
        QDate getLastDateWell() { return _well.getLastObsDate(); }

        Well* getWell() { return &_well; }

        void cleanAllMeteoVector();

    private:
        Crit3DMeteoSettings _meteoSettings;

        QDate _firstMeteoDate;
        QDate _lastMeteoDate;
        Well _well;
        QString _errorStr;

        std::vector<float> _inputTMin;
        std::vector<float> _inputTMax;
        std::vector<float> _inputPrec;
        std::vector<float> _etpValues;
        std::vector<float> _precValues;

        int nrDaysPeriod;           // [days]
        double alpha;
        double h0;                  // unit of observed watertable data, usually [cm]

        float R2;
        float RMSE;
        float EF;

        bool isClimateReady;
        int nrObsData;

        bool isCWBEquationReady;
        double avgDailyCWB;         // [mm]
};

#endif // WATERTABLE_H
