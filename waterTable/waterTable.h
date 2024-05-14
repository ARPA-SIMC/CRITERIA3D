#ifndef WATERTABLE_H
#define WATERTABLE_H

#include "well.h"
#include <QDate>

class WaterTable
{
public:
    WaterTable();
    QDate getFirstDate();
    QDate getLastDate();
    void initializeWaterTable(Well myWell);
    bool computeWaterTable(Well myWell, int maxNrDays, int doy1, int doy2);
    bool computeWTClimate();

    QString getError() const;

private:
    QDate firstDate;
    QDate lastDate;
    Well well;
    QString error;

    int nrDaysPeriod;
    float alpha;
    float h0;
    float R2;
    float RMSE;
    float NASH;
    float EF;

    bool isClimateReady;
    std::vector<float> WTClimateMonthly;
    std::vector<float> WTClimateDaily;
    int nrObsData;

    bool isMeteoPointLinked;
    bool isCWBEquationReady;
    float avgDailyCWB; //[mm]
};

#endif // WATERTABLE_H
