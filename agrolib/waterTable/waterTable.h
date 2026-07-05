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
        WaterTable();

        WaterTable(const std::vector<float> &inputTMin, const std::vector<float> &inputTMax, const std::vector<float> &inputPrec,
                   const QDate &firstMeteoDate, const QDate &lastMeteoDate, const Crit3DMeteoSettings &meteoSettings);

        void initializeWaterTable();
        void initializeWaterTable(const Well &myWell);
        void cleanAllVectors();

        void setLatLon(double lat, double lon)
        {
            _well.setLatitude(lat);
            _well.setLongitude(lon);
        }

        bool initializeMeteoData(const QDate &firstDate, const QDate &lastDate);
        bool setMeteoData(const QDate &date, float tmin, float tmax, float prec);

        void setParameters(int nrDaysPeriod, double alpha, double h0, double avgDailyCWB);

        bool computeWholeSeriesETP(bool isUpdateAvgCWB);

        bool computeWaterTableParameters(const Well &myWell, int stepDays);
        bool computeWTClimate();
        bool computeCWBCorrelation(int stepDays);
        double computeCWB(const QDate &myDate, int nrDays);
        bool computeWaterTableIndices();

        double getWaterTableDaily(const QDate &myDate);
        float getWaterTableClimate(const QDate &myDate) const;

        bool computeWaterTableClimate(const QDate &currentDate, int yearFrom, int yearTo, float &myValue);
        bool getWaterTableInterpolation(const QDate &myDate, float &myValue, float &myDelta, int &deltaDays);

        void computeWaterTableSeries();

        QString getErrorString() const { return _errorStr; }

        double getAlpha() const { return _alpha; }
        double getH0() const { return _h0; }
        double getAvgDailyCWB() const { return _avgDailyCWB; }
        int getNrDaysPeriod() const { return _nrDaysPeriod; }

        float getR2() const { return _R2; }
        float getRMSE() const { return _RMSE; }
        float getEF() const { return _EF; }

        int getNrObsData() const { return _nrObsData; }
        int getNrInterpolatedData() const { return (int)_interpolationSeries.size(); }

        QString getIdWell() const { return _well.getId(); }
        QDate getFirstDate() const { return std::min(_well.getFirstObsDate(), _firstMeteoDate); }

        float getObsDepth(const QDate &myDate) const;

        float getHindcast(int index) const;
        float getInterpolatedData(int index) const;

    private:
        float _WTClimateMonthly[12];
        float _WTClimateDaily[366];

        std::vector<float> _hindcastSeries;
        std::vector<float> _interpolationSeries;

        Crit3DMeteoSettings _meteoSettings;

        QDate _firstMeteoDate;
        QDate _lastMeteoDate;
        Well _well;
        QString _errorStr;

        std::vector<float> _inputTMin;
        std::vector<float> _inputTMax;
        std::vector<float> _inputPrec;
        std::vector<float> _etpValues;

        int _nrDaysPeriod;           // [days]
        double _alpha;               // [-]
        double _h0;                  // unit of observed watertable data, usually [cm]
        double _avgDailyCWB;         // [mm]
        bool _isCWBEquationReady;

        float _R2;
        float _RMSE;
        float _EF;

        bool _isClimateReady;
        int _nrObsData;
};


#endif // WATERTABLE_H
