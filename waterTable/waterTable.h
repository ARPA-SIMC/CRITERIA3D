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

private:
    QDate firstDate;
    QDate lastDate;
    std::vector<Well> wellList;
    int nrDaysPeriod;
    float alpha;
    float h0;
    float R2;
};

#endif // WATERTABLE_H
